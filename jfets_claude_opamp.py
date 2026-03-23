import numpy as np
from scipy.io import wavfile
from scipy.signal import resample
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import root, least_squares
from concurrent.futures import ProcessPoolExecutor
import os

try:
    from numba import njit
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False
    def njit(*args, **kwargs):
        """No-op decorator when Numba is not installed."""
        def decorator(func):
            return func
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator

if __name__ == "__main__":
    if _HAS_NUMBA:
        print("[ENGINE] Numba JIT available — using compiled transient solver", flush=True)
    else:
        print("[ENGINE] Numba not found — using Python transient solver "
              "(pip install numba for 20-50× speedup)", flush=True)

# --- Design Constants ---
_INPUT_AMPLITUDE = 0.25
_INPUT_VPA_EST   = 0.175
_BLOCKING_FC_HZ  = 36.0
_BLOCKING_T5_MS  = 36.0
_BLOCKING_FC_FLOOR_HZ = 20.0
_BLOCKING_T5_FLOOR_MS = 20.0

# --- OPA1656 Opamp Output Stage Constants ---
_R_BIAS       = 10e6      # DC bias for opamp +IN to V_REF
_RG_OPAMP     = 1000.0    # Global feedback ground resistor (Rg)
_OPAMP_V_MIN  = 0.25      # Output clamp low  (V- + 250mV)
_OPAMP_V_MAX  = 17.75     # Output clamp high (V+ - 250mV) at VDD=18V

# ==========================================================================
#  Numba-JIT Transient Solver Engine  (unchanged from jfets_claude_parallel)
# ==========================================================================

@njit(cache=True)
def _gate_junction_jit(v, v_t, is_0, g_leak):
    if v < 0.0:
        return -(is_0 + g_leak * abs(v))
    v_crit = v_t * np.log(v_t / (1.414 * is_0))
    if v > v_crit:
        i_crit = is_0 * (np.exp(v_crit / v_t) - 1.0)
        g_crit = (is_0 / v_t) * np.exp(v_crit / v_t)
        return i_crit + g_crit * (v - v_crit)
    return is_0 * (np.exp(v / v_t) - 1.0)

@njit(cache=True)
def _jfet_physics_jit(idss, vp, lambda_mod, cgs0, cgd0, v_gs, v_ds):
    v_t = 0.02569
    is_rev = v_ds < 0.0
    v_ds_eff = -v_ds if is_rev else v_ds
    v_gs_eff = (v_gs - v_ds) if is_rev else v_gs
    early = 1.0 + lambda_mod * v_ds_eff
    beta = idss / (vp * vp)
    v_gst = v_gs_eff - vp
    v_slope = 1.5 * v_t
    if v_gs_eff <= vp:
        exp_val = np.exp((v_gs_eff - vp) / v_slope)
        i_chan = 1e-9 * exp_val
        gm = (1e-9 / v_slope) * exp_val
    elif v_ds_eff < v_gst:
        i_chan = beta * (2.0 * v_gst * v_ds_eff - v_ds_eff * v_ds_eff) * early
        gm = 2.0 * beta * v_ds_eff * early
    else:
        ratio = 1.0 - v_gs_eff / vp
        i_chan = idss * ratio * ratio * early
        gm = (2.0 * idss / abs(vp)) * ratio * early
    if is_rev:
        i_chan = -i_chan
    v_gd = v_gs - v_ds
    v_dg = -v_gd
    is_0 = 1.0e-14
    g_leak = 1.33e-13
    alpha_ii = 0.01
    beta_ii = 138.15
    i_gs = _gate_junction_jit(v_gs, v_t, is_0, g_leak)
    i_gd_junction = _gate_junction_jit(v_gd, v_t, is_0, g_leak)
    i_g_ii = 0.0
    if v_dg > 0.1:
        i_g_ii = abs(i_chan) * alpha_ii * np.exp(-beta_ii / v_dg)
    i_gate = i_gs + i_gd_junction + i_g_ii
    i_drain = i_chan - i_gd_junction + i_g_ii
    cgs_dyn = cgs0 / np.sqrt(max(0.01, 1.0 - min(v_gs, 0.55) / 0.6))
    cgd_dyn = cgd0 / np.sqrt(max(0.01, 1.0 - min(v_gd, 0.55) / 0.6))
    return i_drain, gm, i_gate, cgs_dyn, cgd_dyn

@njit(cache=True)
def _kcl_residual_jit(v_guess, v_fixed, dt, dim,
                      r_n1, r_n2, r_g,
                      c_n1, c_n2, c_val, c_esr,
                      l_n1, l_n2, l_val, l_rdc, l_cp,
                      j_nd, j_ng, j_ns, j_params,
                      v_prev_all, i_l_prev, c_states,
                      psu_idx, vdd, r_psu):
    n_total = dim + 4
    v_all = np.empty(n_total)
    v_all[:dim] = v_guess
    v_all[dim] = v_fixed[0]
    v_all[dim + 1] = v_fixed[1]
    v_all[dim + 2] = v_fixed[2]
    v_all[dim + 3] = v_fixed[3]
    res = np.zeros(dim)
    if psu_idx >= 0:
        res[psu_idx] -= (vdd - v_all[psu_idx]) / r_psu
    n_r = r_n1.shape[0]
    for k in range(n_r):
        i1 = r_n1[k]; i2 = r_n2[k]
        i_r = (v_all[i1] - v_all[i2]) * r_g[k]
        if i1 < dim: res[i1] += i_r
        if i2 < dim: res[i2] -= i_r
    n_c = c_n1.shape[0]
    for k in range(n_c):
        i1 = c_n1[k]; i2 = c_n2[k]
        dv = (v_all[i1] - v_all[i2]) - (v_prev_all[i1] - v_prev_all[i2])
        i_c = dv / (c_esr[k] + dt / c_val[k])
        if i1 < dim: res[i1] += i_c
        if i2 < dim: res[i2] -= i_c
    n_l = l_n1.shape[0]
    for k in range(n_l):
        i1 = l_n1[k]; i2 = l_n2[k]
        v_ind = v_all[i1] - v_all[i2]
        denom = 1.0 + l_rdc[k] * dt / l_val[k]
        i_l = (i_l_prev[k] + v_ind * dt / l_val[k]) / denom
        v_ind_prev = v_prev_all[i1] - v_prev_all[i2]
        i_cp = (l_cp[k] / dt) * (v_ind - v_ind_prev)
        i_total = i_l + i_cp
        if i1 < dim: res[i1] += i_total
        if i2 < dim: res[i2] -= i_total
    n_j = j_nd.shape[0]
    for k in range(n_j):
        nd = j_nd[k]; ng = j_ng[k]; ns = j_ns[k]
        v_gs = v_all[ng] - v_all[ns]
        v_ds = v_all[nd] - v_all[ns]
        i_d, _, i_g, _, _ = _jfet_physics_jit(
            j_params[k, 0], j_params[k, 1], j_params[k, 2],
            j_params[k, 3], j_params[k, 4], v_gs, v_ds)
        cgs_p = c_states[k, 0]; cgd_p = c_states[k, 1]
        v_gd = v_gs - v_ds
        v_gs_prev = v_prev_all[ng] - v_prev_all[ns]
        v_ds_prev = v_prev_all[nd] - v_prev_all[ns]
        i_cgs = cgs_p * (v_gs - v_gs_prev) / dt
        i_cgd = cgd_p * (v_gd - (v_gs_prev - v_ds_prev)) / dt
        if nd < dim: res[nd] += (i_d - i_cgd)
        if ns < dim: res[ns] -= (i_d + i_g + i_cgs)
        if ng < dim: res[ng] += (i_g + i_cgs + i_cgd)
    return res * 1e3

@njit(cache=True)
def _newton_solve_jit(v_guess, v_fixed, dt, dim,
                      r_n1, r_n2, r_g, c_n1, c_n2, c_val, c_esr,
                      l_n1, l_n2, l_val, l_rdc, l_cp,
                      j_nd, j_ng, j_ns, j_params,
                      v_prev_all, i_l_prev, c_states,
                      psu_idx, vdd, r_psu, tol=1e-9, max_iter=50):
    args = (v_fixed, dt, dim, r_n1, r_n2, r_g, c_n1, c_n2, c_val, c_esr,
            l_n1, l_n2, l_val, l_rdc, l_cp, j_nd, j_ng, j_ns, j_params,
            v_prev_all, i_l_prev, c_states, psu_idx, vdd, r_psu)
    v = v_guess.copy()
    eps_fd = 1e-8
    for iteration in range(max_iter):
        f0 = _kcl_residual_jit(v, *args)
        norm_f = 0.0
        for ii in range(dim): norm_f += f0[ii] * f0[ii]
        if norm_f < tol * tol * dim: break
        J = np.empty((dim, dim))
        for col in range(dim):
            v_pert = v.copy()
            h = max(eps_fd, abs(v[col]) * eps_fd)
            v_pert[col] += h
            f_pert = _kcl_residual_jit(v_pert, *args)
            for row in range(dim): J[row, col] = (f_pert[row] - f0[row]) / h
        dv = np.linalg.solve(J, -f0)
        alpha = 1.0
        for _ in range(10):
            v_new = v + alpha * dv
            f_new = _kcl_residual_jit(v_new, *args)
            norm_new = 0.0
            for ii in range(dim): norm_new += f_new[ii] * f_new[ii]
            if norm_new < norm_f: break
            alpha *= 0.5
        v = v + alpha * dv
    return v

@njit(cache=True)
def _transient_loop_jit(v_in_array, total_samples, dt, dim,
                        r_n1, r_n2, r_g, c_n1, c_n2, c_val, c_esr,
                        l_n1, l_n2, l_val, l_rdc, l_cp,
                        j_nd, j_ng, j_ns, j_params,
                        v_prev_all_init, i_l_prev_init, c_states_init,
                        psu_idx, vdd, r_psu,
                        input_idx, input_dc,
                        v_fixed_base, monitor_indices, v_guess_init):
    n_mon = monitor_indices.shape[0]
    v_out = np.zeros((n_mon, total_samples))
    v_prev_all = v_prev_all_init.copy()
    i_l_prev = i_l_prev_init.copy()
    c_states = c_states_init.copy()
    v_guess = v_guess_init.copy()
    v_fixed = v_fixed_base.copy()
    n_j = j_nd.shape[0]; n_l = l_n1.shape[0]
    for i in range(total_samples):
        v_inst = v_in_array[i]
        v_fixed[1] = input_dc + v_inst
        v_sol = _newton_solve_jit(v_guess, v_fixed, dt, dim,
            r_n1, r_n2, r_g, c_n1, c_n2, c_val, c_esr,
            l_n1, l_n2, l_val, l_rdc, l_cp,
            j_nd, j_ng, j_ns, j_params,
            v_prev_all, i_l_prev, c_states, psu_idx, vdd, r_psu)
        v_guess = v_sol
        for k in range(dim): v_prev_all[k] = v_sol[k]
        v_prev_all[input_idx] = input_dc + v_inst
        for k in range(n_l):
            v_ind = v_prev_all[l_n1[k]] - v_prev_all[l_n2[k]]
            i_l_prev[k] = (i_l_prev[k] + v_ind * dt / l_val[k]) / (1.0 + l_rdc[k] * dt / l_val[k])
        for k in range(n_j):
            v_gs = v_prev_all[j_ng[k]] - v_prev_all[j_ns[k]]
            v_ds = v_prev_all[j_nd[k]] - v_prev_all[j_ns[k]]
            _, _, _, cgs_n, cgd_n = _jfet_physics_jit(
                j_params[k, 0], j_params[k, 1], j_params[k, 2],
                j_params[k, 3], j_params[k, 4], v_gs, v_ds)
            c_states[k, 0] = cgs_n; c_states[k, 1] = cgd_n
        for k in range(n_mon):
            idx = monitor_indices[k]
            if idx == -1: v_out[k, i] = v_inst
            elif idx < dim: v_out[k, i] = v_sol[idx]
            else: v_out[k, i] = v_prev_all[idx]
    return v_sol, v_prev_all, i_l_prev, c_states, v_out


# --- Core Physical Component Classes (unchanged) ---
class Resistor:
    def __init__(self, name, value, node1, node2):
        self.name, self.value, self.node1, self.node2 = name, max(float(value), 1.0), node1, node2

class Capacitor:
    def __init__(self, name, value, node1, node2, esr=0.01):
        self.name, self.value, self.node1, self.node2, self.esr = name, float(value), node1, node2, esr

class Inductor:
    def __init__(self, name, value, node1, node2, r_dc=1.0, c_p=150e-12):
        self.name, self.value, self.node1, self.node2, self.r_dc, self.c_p = name, float(value), node1, node2, r_dc, c_p

class JFET:
    def __init__(self, name, idss, vp, node_d, node_g, node_s):
        self.name, self.idss, self.vp = name, float(idss), float(vp)
        self.node_d, self.node_g, self.node_s = node_d, node_g, node_s
        self.cgs, self.cgd = 2.0e-12, 2.0e-12
        self.lambda_mod = 0.0073
        self.current_cgs, self.current_cgd = self.cgs, self.cgd


# --- High-Fidelity Physics Engine (unchanged) ---
class Circuit:
    def __init__(self, v_dd_ideal=18.0, r_psu=100.0, v_ctrl_force=0.0):
        self.v_dd_ideal, self.r_psu, self.v_ctrl_force = v_dd_ideal, r_psu, v_ctrl_force
        self.resistors, self.capacitors, self.inductors, self.jfets = [], [], [], []
        self.nodes = ["-", "+", "V_FORCE"]
        self.node_map, self.dc_op = {}, {}
        self.dt, self.t_len = 0.0, 0

    def add(self, comp):
        n_list = []
        if isinstance(comp, JFET): n_list = [comp.node_d, comp.node_g, comp.node_s]
        elif isinstance(comp, (Resistor, Capacitor, Inductor)): n_list = [comp.node1, comp.node2]
        for n in n_list:
            if n not in self.nodes: self.nodes.append(n)
        if isinstance(comp, Resistor): self.resistors.append(comp)
        elif isinstance(comp, Capacitor): self.capacitors.append(comp)
        elif isinstance(comp, Inductor): self.inductors.append(comp)
        elif isinstance(comp, JFET): self.jfets.append(comp)

    def _get_active_nodes(self):
        return [n for n in self.nodes if n not in ["-", "V_FORCE", "V_REF"]]

    def _jfet_physics(self, j, v_gs, v_ds):
        v_t = 0.02569; vp_t = j.vp; idss_t = j.idss
        is_rev = v_ds < 0.0
        v_ds_eff = -v_ds if is_rev else v_ds
        v_gs_eff = v_gs - v_ds if is_rev else v_gs
        early = (1.0 + j.lambda_mod * v_ds_eff)
        beta = idss_t / (vp_t**2); v_gst = v_gs_eff - vp_t
        v_slope = 1.5 * v_t
        if v_gs_eff <= vp_t:
            i_chan = 1e-9 * np.exp((v_gs_eff - vp_t) / v_slope)
            gm = (1e-9 / v_slope) * np.exp((v_gs_eff - vp_t) / v_slope)
        elif v_ds_eff < v_gst:
            i_chan = beta * (2 * v_gst * v_ds_eff - v_ds_eff**2) * early
            gm = 2 * beta * v_ds_eff * early
        else:
            i_chan = idss_t * (1 - v_gs_eff / vp_t)**2 * early
            gm = (2 * idss_t / abs(vp_t)) * (1 - v_gs_eff / vp_t) * early
        i_chan = -i_chan if is_rev else i_chan
        v_gd = v_gs - v_ds; v_dg = -v_gd
        is_0 = 1.0e-14; g_leak = 1.33e-13; alpha_ii = 0.01; beta_ii = 138.15
        def calc_gj(v):
            if v < 0: return -(is_0 + g_leak * abs(v))
            v_crit = v_t * np.log(v_t / (1.414 * is_0))
            if v > v_crit:
                i_crit = is_0 * (np.exp(v_crit / v_t) - 1.0)
                g_crit = (is_0 / v_t) * np.exp(v_crit / v_t)
                return i_crit + g_crit * (v - v_crit)
            return is_0 * (np.exp(v / v_t) - 1.0)
        i_gs = calc_gj(v_gs); i_gd_junction = calc_gj(v_gd)
        i_g_ii = abs(i_chan) * alpha_ii * np.exp(-beta_ii / v_dg) if v_dg > 0.1 else 0.0
        i_gate = i_gs + i_gd_junction + i_g_ii
        i_drain = i_chan - i_gd_junction + i_g_ii
        cgs_dyn = j.cgs / np.sqrt(max(0.01, 1.0 - min(v_gs, 0.55) / 0.6))
        cgd_dyn = j.cgd / np.sqrt(max(0.01, 1.0 - min(v_gd, 0.55) / 0.6))
        return i_drain, gm, i_gate, cgs_dyn, cgd_dyn

    def solve_dc_bias(self, input_node="v_ideal"):
        active_nodes = self._get_active_nodes()
        self.node_map = {name: idx for idx, name in enumerate(active_nodes)}
        dim = len(active_nodes)
        def kcl_equations(v_guess):
            v = {"-": 0.0, "V_FORCE": self.v_ctrl_force, "V_REF": self.v_dd_ideal / 2.0}
            for name, idx in self.node_map.items(): v[name] = v_guess[idx]
            v[input_node] = 0.0
            res = np.zeros(dim)
            if "+" in self.node_map: res[self.node_map["+"]] -= (self.v_dd_ideal - v["+"]) / self.r_psu
            for r in self.resistors:
                i_r = (v[r.node1] - v[r.node2]) / r.value
                if r.node1 in self.node_map: res[self.node_map[r.node1]] += i_r
                if r.node2 in self.node_map: res[self.node_map[r.node2]] -= i_r
            for l in self.inductors:
                i_l_dc = (v[l.node1] - v[l.node2]) / l.r_dc
                if l.node1 in self.node_map: res[self.node_map[l.node1]] += i_l_dc
                if l.node2 in self.node_map: res[self.node_map[l.node2]] -= i_l_dc
            for j in self.jfets:
                v_gs, v_ds = v[j.node_g] - v[j.node_s], v[j.node_d] - v[j.node_s]
                i_d, _, i_g, _, _ = self._jfet_physics(j, v_gs, v_ds)
                if j.node_d in self.node_map: res[self.node_map[j.node_d]] += i_d
                if j.node_s in self.node_map: res[self.node_map[j.node_s]] -= (i_d + i_g)
                if j.node_g in self.node_map: res[self.node_map[j.node_g]] += i_g
            if input_node in self.node_map:
                res[self.node_map[input_node]] = v_guess[self.node_map[input_node]]
            return res * 1e3
        v0 = np.ones(dim) * (self.v_dd_ideal / 2.0)
        if "+" in self.node_map: v0[self.node_map["+"]] = self.v_dd_ideal
        if "D1" in self.node_map: v0[self.node_map["D1"]] = self.v_dd_ideal * 0.75
        if "D2" in self.node_map: v0[self.node_map["D2"]] = self.v_dd_ideal * 0.75
        if "G3" in self.node_map: v0[self.node_map["G3"]] = self.v_dd_ideal / 2.0
        sol = least_squares(kcl_equations, v0, bounds=(-60.0, self.v_dd_ideal + 5.0), method='trf')
        self.dc_op = {"-": 0.0, "V_FORCE": self.v_ctrl_force, "V_REF": self.v_dd_ideal / 2.0}
        for name, idx in self.node_map.items(): self.dc_op[name] = sol.x[idx]
        return self.dc_op

    def solve_ac_thevenin(self, target_cap):
        active_nodes = [n for n in self.nodes if n not in ["+", "-", "v_ideal", "V_FORCE", "V_REF"]]
        node_map = {name: idx for idx, name in enumerate(active_nodes)}
        dim = len(active_nodes)
        Y = np.eye(dim) * 1e-12
        for r in self.resistors:
            g = 1.0 / r.value
            if r.node1 in node_map: Y[node_map[r.node1], node_map[r.node1]] += g
            if r.node2 in node_map: Y[node_map[r.node2], node_map[r.node2]] += g
            if r.node1 in node_map and r.node2 in node_map:
                Y[node_map[r.node1], node_map[r.node2]] -= g; Y[node_map[r.node2], node_map[r.node1]] -= g
        for l in self.inductors:
            g = 1.0 / l.r_dc
            if l.node1 in node_map: Y[node_map[l.node1], node_map[l.node1]] += g
            if l.node2 in node_map: Y[node_map[l.node2], node_map[l.node2]] += g
            if l.node1 in node_map and l.node2 in node_map:
                Y[node_map[l.node1], node_map[l.node2]] -= g; Y[node_map[l.node2], node_map[l.node1]] -= g
        for j in self.jfets:
            v_gs = self.dc_op[j.node_g]-self.dc_op[j.node_s]; v_ds = self.dc_op[j.node_d]-self.dc_op[j.node_s]
            _, gm, _, _, _ = self._jfet_physics(j, v_gs, v_ds)
            if j.node_d in node_map:
                if j.node_g in node_map: Y[node_map[j.node_d], node_map[j.node_g]] += gm
                if j.node_s in node_map: Y[node_map[j.node_d], node_map[j.node_s]] -= gm
            if j.node_s in node_map:
                if j.node_g in node_map: Y[node_map[j.node_s], node_map[j.node_g]] -= gm
                if j.node_s in node_map: Y[node_map[j.node_s], node_map[j.node_s]] += gm
        I_t = np.zeros(dim)
        if target_cap.node1 in node_map: I_t[node_map[target_cap.node1]] = 1.0
        if target_cap.node2 in node_map: I_t[node_map[target_cap.node2]] = -1.0
        V_n = np.linalg.solve(Y, I_t)
        v1 = V_n[node_map[target_cap.node1]] if target_cap.node1 in node_map else 0.0
        v2 = V_n[node_map[target_cap.node2]] if target_cap.node2 in node_map else 0.0
        return abs(v1 - v2)

    def _flatten_for_jit(self, input_node, node_map, dim, v_prev, i_l_prev, c_states):
        fixed_names = ["-", input_node, "V_FORCE", "V_REF"]
        def ni(name):
            return node_map[name] if name in node_map else dim + fixed_names.index(name)
        r_n1 = np.array([ni(r.node1) for r in self.resistors], dtype=np.int64)
        r_n2 = np.array([ni(r.node2) for r in self.resistors], dtype=np.int64)
        r_g = np.array([1.0 / r.value for r in self.resistors])
        c_n1 = np.array([ni(c.node1) for c in self.capacitors], dtype=np.int64)
        c_n2 = np.array([ni(c.node2) for c in self.capacitors], dtype=np.int64)
        c_val = np.array([c.value for c in self.capacitors])
        c_esr = np.array([c.esr for c in self.capacitors])
        l_n1 = np.array([ni(l.node1) for l in self.inductors], dtype=np.int64)
        l_n2 = np.array([ni(l.node2) for l in self.inductors], dtype=np.int64)
        l_val = np.array([l.value for l in self.inductors])
        l_rdc = np.array([l.r_dc for l in self.inductors])
        l_cp = np.array([l.c_p for l in self.inductors])
        j_nd = np.array([ni(j.node_d) for j in self.jfets], dtype=np.int64)
        j_ng = np.array([ni(j.node_g) for j in self.jfets], dtype=np.int64)
        j_ns = np.array([ni(j.node_s) for j in self.jfets], dtype=np.int64)
        j_params = np.array([[j.idss, j.vp, j.lambda_mod, j.cgs, j.cgd] for j in self.jfets])
        n_total = dim + 4
        v_prev_all = np.zeros(n_total)
        for name, idx in node_map.items(): v_prev_all[idx] = v_prev[name]
        v_prev_all[dim] = 0.0; v_prev_all[dim+1] = v_prev.get(input_node, 0.0)
        v_prev_all[dim+2] = self.v_ctrl_force; v_prev_all[dim+3] = self.v_dd_ideal / 2.0
        i_l_arr = np.array([i_l_prev[l.name] for l in self.inductors])
        c_st = np.array([[c_states[j.name][0], c_states[j.name][1]] for j in self.jfets])
        return {"r_n1": r_n1, "r_n2": r_n2, "r_g": r_g, "c_n1": c_n1, "c_n2": c_n2,
                "c_val": c_val, "c_esr": c_esr, "l_n1": l_n1, "l_n2": l_n2, "l_val": l_val,
                "l_rdc": l_rdc, "l_cp": l_cp, "j_nd": j_nd, "j_ng": j_ng, "j_ns": j_ns,
                "j_params": j_params, "psu_idx": node_map.get("+", -1),
                "v_prev_all": v_prev_all, "i_l_prev": i_l_arr, "c_states": c_st, "input_idx": ni(input_node)}

    def solve_transient(self, input_node, monitor_nodes, freqs, amplitude=_INPUT_AMPLITUDE,
                        periods=20.0, samples_per_period=4096, use_saved_state=False):
        self.freqs = freqs; f_base = freqs[0]
        dt = (1.0 / f_base) / samples_per_period; self.dt = dt
        total_samples = int(np.round((periods / f_base) / dt))
        t_start = (self.saved_t_end + dt) if (use_saved_state and hasattr(self, 'saved_t_end')) else 0.0
        t = np.linspace(t_start, t_start + (periods / f_base), total_samples, endpoint=False)
        self.t_len = len(t)
        v_in_array = np.zeros_like(t)
        for f in freqs: v_in_array += (amplitude / len(freqs)) * np.sin(2 * np.pi * f * t)
        active_nodes = [n for n in self.nodes if n not in ["-", input_node, "V_FORCE", "V_REF"]]
        node_map = {name: idx for idx, name in enumerate(active_nodes)}
        dim = len(active_nodes)
        v_out_data = {n: np.zeros_like(t) for n in monitor_nodes}
        if use_saved_state and hasattr(self, 'saved_v_prev'):
            v_prev = dict(self.saved_v_prev); i_l_prev = dict(self.saved_i_l_prev)
            c_states = {j.name: (j.current_cgs, j.current_cgd) for j in self.jfets}
        else:
            v_prev = {n: self.dc_op.get(n, 0.0) for n in self.nodes}
            i_l_prev = {l.name: (self.dc_op.get(l.node1, 0.0) - self.dc_op.get(l.node2, 0.0)) / l.r_dc for l in self.inductors}
            c_states = {j.name: (j.cgs, j.cgd) for j in self.jfets}
        v_guess = np.array([v_prev[n] for n in active_nodes])
        if _HAS_NUMBA:
            flat = self._flatten_for_jit(input_node, node_map, dim, v_prev, i_l_prev, c_states)
            mon_idx = np.array([-1 if n == input_node else node_map.get(n, flat["input_idx"]) for n in monitor_nodes], dtype=np.int64)
            v_fixed_base = np.array([0.0, self.dc_op.get(input_node, 0.0), self.v_ctrl_force, self.v_dd_ideal / 2.0])
            v_sol, vpa_out, il_out, cs_out, v_out_arr = _transient_loop_jit(
                v_in_array, total_samples, dt, dim,
                flat["r_n1"], flat["r_n2"], flat["r_g"], flat["c_n1"], flat["c_n2"], flat["c_val"], flat["c_esr"],
                flat["l_n1"], flat["l_n2"], flat["l_val"], flat["l_rdc"], flat["l_cp"],
                flat["j_nd"], flat["j_ng"], flat["j_ns"], flat["j_params"],
                flat["v_prev_all"], flat["i_l_prev"], flat["c_states"],
                flat["psu_idx"], self.v_dd_ideal, self.r_psu,
                flat["input_idx"], self.dc_op.get(input_node, 0.0),
                v_fixed_base, mon_idx, v_guess)
            for k, n in enumerate(monitor_nodes): v_out_data[n] = v_out_arr[k]
            vp = {}
            for name, idx in node_map.items(): vp[name] = vpa_out[idx]
            vp["-"] = 0.0; vp[input_node] = vpa_out[flat["input_idx"]]
            vp["V_FORCE"] = self.v_ctrl_force; vp["V_REF"] = self.v_dd_ideal / 2.0
            for n in self.nodes:
                if n not in vp: vp[n] = self.dc_op.get(n, 0.0)
            self.saved_v_prev = vp
            self.saved_i_l_prev = {l.name: il_out[k] for k, l in enumerate(self.inductors)}
            for k, j in enumerate(self.jfets): j.current_cgs = cs_out[k, 0]; j.current_cgd = cs_out[k, 1]
            self.saved_t_end = t[-1]
            return t, v_in_array, v_out_data
        # Python fallback
        v_inst_current = [0.0]
        def kcl_t(vg):
            v = {"-": 0.0, "V_FORCE": self.v_ctrl_force, "V_REF": self.v_dd_ideal / 2.0,
                 input_node: self.dc_op.get(input_node, 0.0) + v_inst_current[0]}
            for name, idx in node_map.items(): v[name] = vg[idx]
            res = np.zeros(dim)
            if "+" in node_map: res[node_map["+"]] -= (self.v_dd_ideal - v["+"]) / self.r_psu
            for r in self.resistors:
                i_r = (v[r.node1] - v[r.node2]) / r.value
                if r.node1 in node_map: res[node_map[r.node1]] += i_r
                if r.node2 in node_map: res[node_map[r.node2]] -= i_r
            for l in self.inductors:
                vi = v[l.node1] - v[l.node2]
                il = (i_l_prev[l.name] + vi * dt / l.value) / (1.0 + l.r_dc * dt / l.value)
                icp = (l.c_p / dt) * (vi - (v_prev[l.node1] - v_prev[l.node2]))
                if l.node1 in node_map: res[node_map[l.node1]] += (il + icp)
                if l.node2 in node_map: res[node_map[l.node2]] -= (il + icp)
            for c in self.capacitors:
                ic = ((v[c.node1]-v[c.node2])-(v_prev[c.node1]-v_prev[c.node2])) / (c.esr + dt/c.value)
                if c.node1 in node_map: res[node_map[c.node1]] += ic
                if c.node2 in node_map: res[node_map[c.node2]] -= ic
            for j in self.jfets:
                vgs = v[j.node_g]-v[j.node_s]; vds = v[j.node_d]-v[j.node_s]
                id_, _, ig, _, _ = self._jfet_physics(j, vgs, vds)
                cgsp, cgdp = c_states[j.name]; vgd = vgs - vds
                vgsp = v_prev[j.node_g]-v_prev[j.node_s]; vdsp = v_prev[j.node_d]-v_prev[j.node_s]
                icgs = cgsp * (vgs - vgsp) / dt; icgd = cgdp * (vgd - (vgsp - vdsp)) / dt
                if j.node_d in node_map: res[node_map[j.node_d]] += (id_ - icgd)
                if j.node_s in node_map: res[node_map[j.node_s]] -= (id_ + ig + icgs)
                if j.node_g in node_map: res[node_map[j.node_g]] += (ig + icgs + icgd)
            return res * 1e3
        for i, vi in enumerate(v_in_array):
            v_inst_current[0] = vi
            sol = root(kcl_t, v_guess, method='hybr', tol=1e-9); v_sol = sol.x; v_guess = v_sol
            v_prev[input_node] = self.dc_op.get(input_node, 0.0) + vi
            for name, idx in node_map.items(): v_prev[name] = v_sol[idx]
            for l in self.inductors:
                vi_ = v_prev[l.node1] - v_prev[l.node2]
                i_l_prev[l.name] = (i_l_prev[l.name] + vi_ * dt / l.value) / (1.0 + l.r_dc * dt / l.value)
            for j in self.jfets:
                vgs = v_prev[j.node_g]-v_prev[j.node_s]; vds = v_prev[j.node_d]-v_prev[j.node_s]
                _, _, _, cn1, cn2 = self._jfet_physics(j, vgs, vds)
                c_states[j.name] = (cn1, cn2); j.current_cgs = cn1; j.current_cgd = cn2
            for n in monitor_nodes: v_out_data[n][i] = v_prev[n] if n != input_node else vi
        self.saved_v_prev = dict(v_prev); self.saved_i_l_prev = dict(i_l_prev); self.saved_t_end = t[-1]
        return t, v_in_array, v_out_data


# --- Analyzer (updated for opamp output) ---
class CircuitAnalyzer:
    def __init__(self, circuit, monitor_nodes, input_node="v_ideal", amplitude=_INPUT_AMPLITUDE):
        self.circuit, self.monitor_nodes, self.input_node, self.amplitude = circuit, monitor_nodes, input_node, amplitude
        self.t = self.v_in = self.v_out_data = None
        base_freqs = [100.0, 125.0, 150.0]
        self.freqs = base_freqs + [f*2 for f in base_freqs] + [f*4 for f in base_freqs]
        self.thd_freqs = base_freqs

    def report_dc_bias(self):
        dc = self.circuit.solve_dc_bias(input_node=self.input_node)
        for j in self.circuit.jfets:
            vgs = dc.get(j.node_g, 0)-dc.get(j.node_s, 0); vds = dc.get(j.node_d, 0)-dc.get(j.node_s, 0)
            id_, _, _, _, _ = self.circuit._jfet_physics(j, vgs, vds)
            print(f"[{j.name}] Bias -> V_GS: {vgs:.4f}V | V_DS: {vds:.4f}V | I_D: {id_*1000:.4f}mA")

    def get_max_system_tau(self):
        taus = [c.value * self.circuit.solve_ac_thevenin(c) for c in self.circuit.capacitors if c.node1 != c.node2]
        return max(taus) if taus else 0.001

    def run_transient(self):
        f_base = self.freqs[0]; sys_tau = self.get_max_system_tau()
        total_sec = 10.0 * sys_tau + 20.0 / f_base
        self.t, self.v_in, self.v_out_data = self.circuit.solve_transient(
            input_node=self.input_node, monitor_nodes=self.monitor_nodes,
            freqs=self.freqs, amplitude=self.amplitude,
            periods=int(np.round(total_sec * f_base)), samples_per_period=2048)

    def report_ac_analytics(self):
        print("\n--- Capacitor Thevenin Analytics ---")
        for cap in self.circuit.capacitors:
            if cap.node1 == cap.node2: continue
            R_th = self.circuit.solve_ac_thevenin(cap)
            if R_th < 1e-12: continue
            fc = 1.0 / (2 * np.pi * R_th * cap.value)
            print(f"[{cap.name}] R_th: {R_th/1000:.2f} kΩ | f_c: {fc:.2f} Hz | 5τ: {5*R_th*cap.value*1000:.2f} ms")

    def report_single_tone_thd(self, node="G3", opamp_gain=1.0):
        print(f"\n--- Single-Tone THD Analysis ({node}, gain={opamp_gain:.3f}×) ---")
        v_ref = self.circuit.v_dd_ideal / 2.0
        for f_test in self.thd_freqs:
            self.circuit.solve_dc_bias(input_node=self.input_node)
            self.circuit.solve_transient(input_node=self.input_node, monitor_nodes=[node],
                freqs=[f_test], amplitude=self.amplitude, periods=15.0, samples_per_period=2048)
            _, _, td = self.circuit.solve_transient(input_node=self.input_node, monitor_nodes=[node],
                freqs=[f_test], amplitude=self.amplitude, periods=20.0, samples_per_period=2048, use_saved_state=True)
            vg3 = td[node]
            vout = np.clip(v_ref + opamp_gain * (vg3 - v_ref), _OPAMP_V_MIN, _OPAMP_V_MAX)
            vac = vout - np.mean(vout); N = len(vac)
            Y = np.fft.rfft(vac * np.hanning(N)); xf = np.fft.rfftfreq(N, d=self.circuit.dt)
            mag = (2.0/N * np.abs(Y)) + 1e-12
            idx_f = np.argmin(np.abs(xf - f_test)); mag_f = mag[idx_f]
            w_f = mag_f * _a_weight(f_test); ssq = 0.0
            print(f"\nFundamental: {f_test:.0f} Hz | Level: {mag_f*1000:.2f} mV | A-wt: {w_f*1000:.3f} mV_w")
            for h in range(2, 10):
                hf = f_test * h
                if hf > xf[-1]: break
                mh = mag[np.argmin(np.abs(xf - hf))]; wh = mh * _a_weight(hf); ssq += wh**2
                print(f"  H{h} ({hf:.0f}Hz): {mh*1000:.3f} mV | IHL: {mh/mag_f*100:.2f}% | A-wt: {wh*1000:.4f} mV_w")
            print(f"  THD({f_test:.0f}Hz): {np.sqrt(ssq)/w_f*100:.2f}%" if w_f > 1e-12 else "  THD: N/A")

    def plot_waveforms(self, mode, opamp_gain=1.0):
        fig = plt.figure(figsize=(16, 12))
        plt.suptitle(f"{mode} Mode: OPA1656 Output (G={opamp_gain:.2f}×)", fontweight='bold', fontsize=14)
        gs_ = gridspec.GridSpec(5, 2, figure=fig, width_ratios=[1, 1])
        si = int(np.round((10.0 * self.get_max_system_tau()) / self.circuit.dt))
        ts = self.t[si:]; tp = (ts - ts[0]) * 1000.0
        v_ref = self.circuit.v_dd_ideal / 2.0
        nodes = ["v_ideal", "G1", "D1", "D2", "G3"]
        titles = ["Pickup (v_ideal)", "G1 (pre-Q1)", "D1 (Q1 out)", "D2 (Q2 out)", "Opamp Out"]
        colors = ['gray', 'green', 'blue', 'purple', 'black']
        for i, (nd, ti, co) in enumerate(zip(nodes, titles, colors)):
            ax = fig.add_subplot(gs_[i, 0])
            vs = self.v_out_data[nd][si:]
            if nd == "G3":
                vs = np.clip(v_ref + opamp_gain * (vs - v_ref), _OPAMP_V_MIN, _OPAMP_V_MAX) - v_ref
            elif nd in ("D1", "D2"):
                vs = vs - np.mean(vs)
            ax.plot(tp, vs, color=co, linewidth=1.5); ax.set_title(ti, fontsize=11)
            ax.grid(True, alpha=0.3); ax.set_xlim(0, 40)
            if i == 4: ax.set_xlabel("Time (ms)")
        ax_fft = fig.add_subplot(gs_[:, 1])
        vg3 = self.v_out_data["G3"][si:]
        vop = np.clip(v_ref + opamp_gain * (vg3 - v_ref), _OPAMP_V_MIN, _OPAMP_V_MAX)
        vac = vop - np.mean(vop); N = len(vac)
        Y = np.fft.rfft(vac * np.hanning(N)); xf = np.fft.rfftfreq(N, d=self.circuit.dt)
        mag = (2.0/N * np.abs(Y)) + 1e-12; wm = mag * _a_weight(xf)
        ax_fft.fill_between(xf[1:], wm[1:], 1e-9, color='gray', alpha=0.15)
        cc = ['red','blue','green','orange','purple','cyan','magenta','brown','olive']
        for fv, c in zip(self.freqs, cc):
            hf = [fv*h for h in range(1,10) if fv*h < xf[-1]]
            hv = [wm[np.argmin(np.abs(xf-f))] for f in hf]
            mk, sl, _ = ax_fft.stem(hf, hv, linefmt=c, basefmt=' ', markerfmt='o')
            plt.setp(mk, color=c, markersize=4, alpha=0.8); plt.setp(sl, color=c, alpha=0.6)
        ax_fft.set_title("A-Weighted Spectrum"); ax_fft.set_xscale('log'); ax_fft.set_yscale('log')
        ax_fft.set_xlim(50, 10000); ax_fft.set_ylim(1e-6, wm[1:].max()*2); ax_fft.grid(True, which="both", alpha=0.2)
        plt.tight_layout(); plt.savefig(f'mode_{mode}_analysis.png'); plt.close()

    def export_audio(self, mode, opamp_gain=1.0, target_duration_sec=4.0, target_sr=44100):
        si = int(np.round((10.0 * self.get_max_system_tau()) / self.circuit.dt))
        v_ref = self.circuit.v_dd_ideal / 2.0
        vg3 = self.v_out_data["G3"][si:]
        vop = np.clip(v_ref + opamp_gain * (vg3 - v_ref), _OPAMP_V_MIN, _OPAMP_V_MAX)
        tsc = int(np.round(0.05 * target_sr)); c44 = resample(vop, tsc); c44 -= np.mean(c44)
        N44 = int(np.round((1.0/25.0)*target_sr))
        zc = np.where((c44[:-1]<=0)&(c44[1:]>0))[0]; sc = zc[zc>N44]
        zs = sc[0] if len(sc)>0 else N44; dp = c44[zs:zs+N44].copy()
        es = int(np.round(target_duration_sec*target_sr))
        sw = np.tile(dp, int(np.ceil(es/len(dp))))[:es]
        mx = np.max(np.abs(sw)); nw = (sw/mx) if mx>0 else sw
        wavfile.write(f'mode_{mode}_audio.wav', target_sr, np.int16(nw*32767))
        print(f"--- Audio exported: mode_{mode}_audio.wav ---")


def process_unified_circuit(sim, comp_list):
    for comp in comp_list:
        ct = comp.get("type"); nm = comp.get("name")
        if ct == "Resistor": sim.add(Resistor(nm, comp["val"], comp["n1"], comp["n2"]))
        elif ct == "Capacitor": sim.add(Capacitor(nm, comp["val"], comp["n1"], comp["n2"]))
        elif ct == "Inductor": sim.add(Inductor(nm, comp["val"], comp["n1"], comp["n2"], r_dc=comp.get("r_dc",1.0), c_p=comp.get("c_p",150e-12)))
        elif ct == "JFET": sim.add(JFET(nm, comp["idss"], comp["vp"], comp["nd"], comp["ng"], comp["ns"]))

def _a_weight(f):
    if np.isscalar(f):
        if f < 1.0: return 1e-6
        f2 = f**2
        return (12194**2 * f**4) / ((f2 + 20.6**2) * np.sqrt((f2 + 107.7**2) * (f2 + 737.9**2)) * (f2 + 12194**2))
    f = np.asarray(f, dtype=float); f2 = f**2
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(f > 1.0, (12194**2 * f**4) / ((f2+20.6**2)*np.sqrt((f2+107.7**2)*(f2+737.9**2))*(f2+12194**2)), 1e-6)

def get_vpa_metric(v_out, dt, freqs):
    f_base = freqs[0]; ws = 2.0/f_base
    si = -int(np.round(ws/dt)) if len(v_out) > int(np.round(ws/dt)) else 0
    vs = v_out[si:]; vac = vs - np.mean(vs); N = len(vac)
    Y = np.fft.rfft(vac * np.hanning(N)); xf = np.fft.rfftfreq(N, d=dt)
    mag = (2.0/N * np.abs(Y)) + 1e-12
    s = 0.0
    for f in freqs:
        for h in range(1, 10):
            idx = np.argmin(np.abs(xf - (f*h))); s += (mag[idx] * _a_weight(f*h))**2
    return np.sqrt(s)

def calc_self_bias(alpha, idss, vp_abs):
    Id = alpha * idss; Vs = vp_abs * (1.0 - np.sqrt(alpha)); Rs = Vs / Id
    return Vs, Rs, Id

# --- E24 tables ---
_E24_BASES = [1.0,1.1,1.2,1.3,1.5,1.6,1.8,2.0,2.2,2.4,2.7,3.0,3.3,3.6,3.9,4.3,4.7,5.1,5.6,6.2,6.8,7.5,8.2,9.1]
_E24_CAPS = np.array([b*m for m in [1e-12,10e-12,100e-12,1e-9,10e-9,100e-9] for b in _E24_BASES])
_MONITOR_NODES = ["v_ideal", "G1", "D1", "G2", "D2", "G3"]
_CAL_FREQ = [1000.0]
_TARGET_VPA = 0.75


def _eval_circuit_from_config(config, c3_dummy=0.0, r7_dummy=0.0, full_run=False, high_res=False):
    core_list = [
        {"type": "Inductor",   "name": "L_PICKUP",  "val": 4.5,               "n1": "v_ideal", "n2": "IN", "r_dc": 8000.0, "c_p": 150e-12},
        {"type": "Capacitor",  "name": "C1",         "val": config["c1"],      "n1": "IN",      "n2": "G1"},
        {"type": "Resistor",   "name": "R1",         "val": config["r_g_calc"],"n1": "G1",      "n2": "-"},
        {"type": "JFET",       "name": "Q1",         "idss": config["idss_t"], "vp": config["vp_t"], "nd": "D1", "ng": "G1", "ns": "S1"},
        {"type": "Resistor",   "name": "R3",         "val": config["rd1"],     "n1": "+",       "n2": "D1"},
        {"type": "Resistor",   "name": "R2",         "val": config["rs1"],     "n1": "S1",      "n2": "-"},
        {"type": "Capacitor",  "name": "C2",         "val": config["c2"],      "n1": "D1",      "n2": "G2"},
        {"type": "Resistor",   "name": "R4",         "val": config["r_g_calc"],"n1": "G2",      "n2": "-"},
        {"type": "JFET",       "name": "Q2",         "idss": config["idss_t"], "vp": config["vp_t"], "nd": "D2", "ng": "G2", "ns": "S2"},
        {"type": "Resistor",   "name": "R6",         "val": config["rd2"],     "n1": "+",       "n2": "D2"},
        {"type": "Resistor",   "name": "R5",         "val": config["rs2"],     "n1": "S2",      "n2": "-"},
        {"type": "Capacitor",  "name": "C_REF",      "val": config.get("c_ref", 47e-6),   "n1": "V_REF",   "n2": "-"},
        {"type": "Capacitor",  "name": "C3",         "val": config["c3"],      "n1": "D2",      "n2": "G3"},
        {"type": "Resistor",   "name": "R_BIAS",     "val": config.get("r_bias", _R_BIAS), "n1": "G3",      "n2": "V_REF"},
    ]
    sim = Circuit(v_dd_ideal=18.0, r_psu=10.0)
    process_unified_circuit(sim, core_list)
    cal_freq = config.get("cal_freq", _CAL_FREQ)
    monitor = config.get("monitor_nodes", _MONITOR_NODES)
    analyzer = CircuitAnalyzer(circuit=sim, monitor_nodes=monitor, input_node="v_ideal", amplitude=_INPUT_AMPLITUDE)
    if not full_run:
        spp = 2048 if high_res else 512
        analyzer.circuit.solve_dc_bias(input_node="v_ideal")
        analyzer.circuit.solve_transient(input_node="v_ideal", monitor_nodes=["G3"], freqs=cal_freq, periods=80.0, samples_per_period=spp)
        analyzer.t, analyzer.v_in, analyzer.v_out_data = analyzer.circuit.solve_transient(
            input_node="v_ideal", monitor_nodes=["v_ideal", "G3"], freqs=cal_freq, periods=20.0, samples_per_period=spp, use_saved_state=True)
        return analyzer
    return analyzer


def _resolve_global_resistors(idss=0.0055, vp_abs=2.0, vdd=18.0,
                               target_fc_hz=_BLOCKING_FC_HZ, target_recovery_ms=_BLOCKING_T5_MS):
    tau_floor = 1.0 / (2.0 * np.pi * target_fc_hz)
    tau_ceiling = (target_recovery_ms / 1000.0) / 5.0
    e24_r = np.array(sorted(set(b*m for m in [1,10,100,1e3,1e4,1e5,1e6,1e7] for b in _E24_BASES)))
    r_vol = 500e3; r_bias = _R_BIAS; rg_opamp = _RG_OPAMP
    # R_g
    igss_max = 1e-9; r_g_max = 0.01 * vp_abs / igss_max; r_pickup_dcr = 8000.0
    r_g_calc = None
    for rg in np.sort(e24_r[(e24_r >= 100e3) & (e24_r <= r_g_max)])[::-1]:
        rth1 = float(rg) + r_pickup_dcr; rth2 = float(rg)
        c1lo, c1hi = tau_floor/rth1, tau_ceiling/rth1; c2lo, c2hi = tau_floor/rth2, tau_ceiling/rth2
        if np.any((_E24_CAPS >= c1lo-1e-15) & (_E24_CAPS <= c1hi+1e-15)) and np.any((_E24_CAPS >= c2lo-1e-15) & (_E24_CAPS <= c2hi+1e-15)):
            r_g_calc = float(rg); break
    if r_g_calc is None: r_g_calc = float(e24_r[(e24_r >= 100e3) & (e24_r <= r_g_max)][-1])
    print(f"[GLOBAL] R_g: {r_g_calc/1e6:.2f} MΩ | R_vol: {r_vol/1e3:.0f} kΩ", flush=True)
    # Rs / alpha
    a1, a2 = 0.30, 0.25
    vs1, rs1, id1 = calc_self_bias(a1, idss, vp_abs)
    vs2, rs2, id2 = calc_self_bias(a2, idss, vp_abs)
    gm1 = 2*idss*np.sqrt(a1)/vp_abs; gm2 = 2*idss*np.sqrt(a2)/vp_abs
    re1 = 1/gm1; re2 = 1/gm2
    # km sweep (no Q3 constraint)
    best_km = None; best_s = -1.0
    for km1 in np.arange(vp_abs+0.5, vdd-vp_abs-1.0, 0.25):
        rd1t = (vdd-vp_abs-km1)/id1
        if rd1t <= 0: continue
        g1 = rd1t/(rs1+re1)
        for km2 in np.arange(vp_abs+0.5, vdd-vp_abs-1.0, 0.25):
            rd2t = (vdd-vp_abs-km2)/id2
            if rd2t <= 0: continue
            g2 = rd2t/(rs2+re2)
            if (_INPUT_AMPLITUDE*g1*g2) < (_INPUT_VPA_EST*3.0/0.9): continue
            vd2q = vp_abs+km2; d2pk = _INPUT_AMPLITUDE*g1*g2; hr = min(vd2q, vdd-vd2q)
            if d2pk > hr*0.99: continue
            sc = hr - d2pk
            if sc > best_s: best_s = sc; best_km = (km1, km2)
    if best_km is None: best_km = (vdd/2-vp_abs, vdd/2-vp_abs)
    km_table = {"Clean": best_km, "OD1": (best_km[0], 0.5), "OD2": (0.5, 0.5)}
    per_mode_rd = {}
    for mode, (km1, km2) in km_table.items():
        per_mode_rd[mode] = ((vdd-vp_abs-km1)/id1, (vdd-vp_abs-km2)/id2)
    print(f"[Clean km] km1={best_km[0]:.2f} km2={best_km[1]:.2f}", flush=True)
    for m, (rd1, rd2) in per_mode_rd.items(): print(f"[{m}] Rd1={rd1:.1f}Ω Rd2={rd2:.1f}Ω")
    print(f"[GLOBAL] OPA1656: R_bias={r_bias/1e6:.0f}MΩ Rg={rg_opamp:.0f}Ω swing={_OPAMP_V_MIN:.2f}–{_OPAMP_V_MAX:.2f}V")
    c_ref = 47e-6
    return {"rs1":rs1,"rs2":rs2,"id1":id1,"id2":id2,"vs1":vs1,"vs2":vs2,"a1":a1,"a2":a2,
            "r_g_calc":r_g_calc,"r_vol":r_vol,"r_bias":r_bias,"rg_opamp":rg_opamp,"c_ref":c_ref,
            "idss_t":idss,"vp_t":-vp_abs,"tau_floor":tau_floor,"tau_ceiling":tau_ceiling,"per_mode_rd":per_mode_rd}


def _probe_mode_rth(args):
    mode, resistors = args
    rs1, rs2, r_g = resistors["rs1"], resistors["rs2"], resistors["r_g_calc"]
    idss_t, vp_t = resistors["idss_t"], resistors["vp_t"]
    rd1, rd2 = resistors["per_mode_rd"][mode]
    r_bias = resistors.get("r_bias", _R_BIAS); c_ph = 100e-9
    core = [
        {"type":"Inductor","name":"L_PICKUP","val":4.5,"n1":"v_ideal","n2":"IN","r_dc":8000.0,"c_p":150e-12},
        {"type":"Capacitor","name":"C1","val":c_ph,"n1":"IN","n2":"G1"},
        {"type":"Resistor","name":"R1","val":r_g,"n1":"G1","n2":"-"},
        {"type":"JFET","name":"Q1","idss":idss_t,"vp":vp_t,"nd":"D1","ng":"G1","ns":"S1"},
        {"type":"Resistor","name":"R3","val":rd1,"n1":"+","n2":"D1"},
        {"type":"Resistor","name":"R2","val":rs1,"n1":"S1","n2":"-"},
        {"type":"Capacitor","name":"C2","val":c_ph,"n1":"D1","n2":"G2"},
        {"type":"Resistor","name":"R4","val":r_g,"n1":"G2","n2":"-"},
        {"type":"JFET","name":"Q2","idss":idss_t,"vp":vp_t,"nd":"D2","ng":"G2","ns":"S2"},
        {"type":"Resistor","name":"R6","val":rd2,"n1":"+","n2":"D2"},
        {"type":"Resistor","name":"R5","val":rs2,"n1":"S2","n2":"-"},
        {"type":"Capacitor","name":"C_REF","val":resistors.get("c_ref",47e-6),"n1":"V_REF","n2":"-"},
        {"type":"Capacitor","name":"C3","val":c_ph,"n1":"D2","n2":"G3"},
        {"type":"Resistor","name":"R_BIAS","val":r_bias,"n1":"G3","n2":"V_REF"},
    ]
    sim = Circuit(v_dd_ideal=18.0, r_psu=10.0); process_unified_circuit(sim, core)
    sim.solve_dc_bias(input_node="v_ideal")
    rth_c1 = sim.solve_ac_thevenin(Capacitor("P1",1e-9,"IN","G1"))
    rth_c2 = sim.solve_ac_thevenin(Capacitor("P2",1e-9,"D1","G2"))
    rth_c3 = sim.solve_ac_thevenin(Capacitor("P3",1e-9,"D2","G3"))
    print(f"[{mode}] R_th → C1:{rth_c1/1000:.1f}k C2:{rth_c2/1000:.1f}k C3:{rth_c3/1000:.1f}k", flush=True)
    return {"mode":mode, "rth_c1":rth_c1, "rth_c2":rth_c2, "rth_c3":rth_c3}


def _reconcile_global_caps(mode_rth_list, modes, tau_floor, tau_ceiling):
    rth_by = {r["mode"]: r for r in mode_rth_list}; caps = {}
    for cn, rk in [("C1","rth_c1"), ("C2","rth_c2")]:
        rths = [rth_by[m][rk] for m in modes]
        clo = max(tau_floor/r for r in rths); chi = min(tau_ceiling/r for r in rths)
        cmid = (clo+chi)/2.0
        if chi >= clo:
            iw = _E24_CAPS[(_E24_CAPS >= clo-chi*1e-9) & (_E24_CAPS <= chi+chi*1e-9)]
            best = float(iw[np.argmin(np.abs(iw - cmid))]) if len(iw) > 0 else float(_E24_CAPS[np.argmin(np.abs(_E24_CAPS - cmid))])
        else:
            best = float(_E24_CAPS[np.argmin(np.abs(_E24_CAPS - clo))])
        caps[cn] = {"value": best}
        print(f"[GLOBAL] {cn}: {best*1e9:.3f} nF | window: [{clo*1e9:.3f}, {chi*1e9:.3f}] nF", flush=True)
    return caps


def _find_e24_rf(gain_required, rg=_RG_OPAMP):
    rf_ideal = rg * (gain_required - 1.0)
    if rf_ideal <= 0: return 0.0, 1.0
    e24_r = sorted(set(b*m for m in [1,10,100,1e3,1e4,1e5,1e6] for b in _E24_BASES))
    e24_arr = np.array(e24_r)
    rf_e24 = float(e24_arr[np.argmin(np.abs(e24_arr - rf_ideal))])
    return rf_e24, 1.0 + rf_e24 / rg


def write_component_tsv(all_mode_components, global_resistors=None, filename="component_bom.tsv"):
    modes = ["Clean", "OD1", "OD2"]
    def fR(v):
        if v is None: return "DNP"
        if v >= 1e6: return f"{v/1e6:.3f} MΩ"
        if v >= 1e3: return f"{v/1e3:.3f} kΩ"
        return f"{v:.1f} Ω"
    def fC(v):
        if v is None: return "DNP"
        if v >= 1e-6: return f"{v*1e6:.3f} µF"
        if v >= 1e-9: return f"{v*1e9:.3f} nF"
        return f"{v*1e12:.3f} pF"
    def parR(rb, rt):
        if rb <= rt+1: return None
        return (rb*rt)/(rb-rt)
    rd1b = max(all_mode_components[m]["rd1"] for m in modes)
    rd2b = max(all_mode_components[m]["rd2"] for m in modes)
    gr = global_resistors or {}
    with open(filename, "w") as f:
        f.write("Section\tRef Des\tDescription\tValue\tClean\tOD1\tOD2\n")
        f.write("---\t---\t---\t---\t---\t---\t---\n")
        f.write(f"GLOBAL\tR1/R4\tGate Bias\t{fR(gr.get('r_g_calc',10e6))}\t—\t—\t—\n")
        f.write(f"GLOBAL\tR_VOL\tVolume Pot\t{fR(gr.get('r_vol',500e3))}\t—\t—\t—\n")
        f.write(f"GLOBAL\tC_REF\tV_REF Bypass\t{fC(gr.get('c_ref',47e-6))}\t—\t—\t—\n")
        f.write(f"GLOBAL\tL_PICKUP\tPickup\t4.5H (8kΩ DCR)\t—\t—\t—\n")
        f.write(f"GLOBAL\tC1\tInput Coupling\t{fC(all_mode_components['Clean']['c1'])}\t—\t—\t—\n")
        f.write(f"GLOBAL\tC2\tInterstage Coupling\t{fC(all_mode_components['Clean']['c2'])}\t—\t—\t—\n\n")
        f.write(f"MODE_R\tR3\tQ1 Drain (base={fR(rd1b)})\t{fR(rd1b)}")
        for m in modes: f.write(f"\t{fR(all_mode_components[m]['rd1'])} / par={fR(parR(rd1b, all_mode_components[m]['rd1']))}")
        f.write("\n")
        f.write(f"MODE_R\tR6\tQ2 Drain (base={fR(rd2b)})\t{fR(rd2b)}")
        for m in modes: f.write(f"\t{fR(all_mode_components[m]['rd2'])} / par={fR(parR(rd2b, all_mode_components[m]['rd2']))}")
        f.write("\n\n")
        f.write(f"OPAMP\tU1\tOPA1656 Dual Audio Op Amp\tSOIC-8, VDD=18V\t—\t—\t—\n")
        f.write(f"OPAMP\tR_BIAS\tOpamp +IN bias to V_REF\t{fR(gr.get('r_bias',_R_BIAS))}\t—\t—\t—\n")
        f.write(f"OPAMP\tRg\tFeedback ground R (global)\t{fR(gr.get('rg_opamp',_RG_OPAMP))}\t—\t—\t—\n")
        f.write("OPAMP\tC3\tD2→Opamp coupling (per-mode)\tper-mode")
        for m in modes: f.write(f"\t{fC(all_mode_components[m]['c3'])}")
        f.write("\n")
        f.write("OPAMP\tRf\tFeedback R (per-mode)\tper-mode")
        for m in modes: f.write(f"\t{fR(all_mode_components[m]['rf_opamp'])} (G={all_mode_components[m]['opamp_gain']:.3f}×)")
        f.write("\n")
    print(f"\n>>> Component BOM written to: {filename}")


def execute_mode_analytics(mode, config, opamp_gain, rf_opamp):
    v_ref = 18.0 / 2.0
    print(f"\n{'='*60}\n[{mode}] C3={config['c3']*1e9:.2f}nF | Rf={rf_opamp:.0f}Ω | Gain={opamp_gain:.3f}×\n{'='*60}")
    analyzer = _eval_circuit_from_config(config, full_run=True)
    analyzer.report_dc_bias(); analyzer.report_ac_analytics(); analyzer.run_transient()
    si = int(np.round((10.0 * analyzer.get_max_system_tau()) / analyzer.circuit.dt))
    vg3 = analyzer.v_out_data["G3"][si:]
    vop = np.clip(v_ref + opamp_gain * (vg3 - v_ref), _OPAMP_V_MIN, _OPAMP_V_MAX)
    vpa_g3 = get_vpa_metric(analyzer.v_out_data["G3"], analyzer.circuit.dt, analyzer.freqs)
    vpa_out = get_vpa_metric(vop, analyzer.circuit.dt, analyzer.freqs)
    print(f"\n[{mode}] V_pa@G3: {vpa_g3:.4f} V_w | V_pa@out: {vpa_out:.4f} V_w | Vpp: {np.ptp(vop):.3f} V")
    analyzer.report_single_tone_thd(node="G3", opamp_gain=opamp_gain)
    analyzer.plot_waveforms(mode=mode, opamp_gain=opamp_gain)
    analyzer.export_audio(mode=mode, opamp_gain=opamp_gain)


# ==================================================================
#  MAIN — Single-pass pipeline (no SCF, no Phase B/C combo search)
# ==================================================================
if __name__ == "__main__":
    modes = ["Clean", "OD1", "OD2"]
    n_cpus = os.cpu_count() or 4
    tau_floor   = 1.0 / (2.0 * np.pi * _BLOCKING_FC_HZ)
    tau_ceiling = (_BLOCKING_T5_MS / 1000.0) / 5.0

    print("=" * 72)
    print(f"  JFET PREAMP + OPA1656 OUTPUT — {n_cpus} CPUs")
    print(f"  Blocking: tau ∈ [{tau_floor*1000:.2f}, {tau_ceiling*1000:.2f}] ms")
    print("=" * 72, flush=True)

    # Phase A: Global resistors
    resistors = _resolve_global_resistors()

    # Phase A2: Thevenin probes (single pass — no SCF needed)
    print("\n--- Thevenin Probes ---", flush=True)
    with ProcessPoolExecutor(max_workers=min(n_cpus, 3)) as pool:
        mode_rth_list = list(pool.map(_probe_mode_rth, [(m, resistors) for m in modes]))
    rth_by_mode = {r["mode"]: r for r in mode_rth_list}

    # Phase A3: Global caps (C1, C2)
    global_caps = _reconcile_global_caps(mode_rth_list, modes, tau_floor, tau_ceiling)
    c1 = global_caps["C1"]["value"]; c2 = global_caps["C2"]["value"]

    # Per-mode C3 (biased to floor)
    c3_by_mode = {}
    for m in modes:
        rth_c3 = rth_by_mode[m]["rth_c3"]
        clo = tau_floor / rth_c3; chi = tau_ceiling / rth_c3
        iw = _E24_CAPS[(_E24_CAPS >= clo-chi*1e-9) & (_E24_CAPS <= chi+chi*1e-9)]
        c3_by_mode[m] = float(iw[0]) if len(iw) > 0 else float(_E24_CAPS[np.argmin(np.abs(_E24_CAPS - clo))])
        print(f"[{m}] C3: {c3_by_mode[m]*1e9:.3f} nF | R_th: {rth_c3/1000:.1f}k")

    # Phase B: One transient per mode → analytical gain → E24 Rf
    print(f"\n{'='*72}\n  PHASE B — VPA + Opamp Gain (3 evals total)\n{'='*72}", flush=True)
    mode_configs = {}; mode_gain = {}; mode_rf = {}
    for m in modes:
        rd1, rd2 = resistors["per_mode_rd"][m]
        cfg = {"mode":m, "c1":c1, "c2":c2, "c3":c3_by_mode[m],
               "rd1":rd1, "rs1":resistors["rs1"], "rd2":rd2, "rs2":resistors["rs2"],
               "r_g_calc":resistors["r_g_calc"], "r_bias":resistors["r_bias"],
               "c_ref":resistors["c_ref"], "idss_t":resistors["idss_t"], "vp_t":resistors["vp_t"],
               "tau_floor":tau_floor, "tau_ceiling":tau_ceiling,
               "cal_freq":_CAL_FREQ, "monitor_nodes":_MONITOR_NODES, "target_vpa":_TARGET_VPA}
        mode_configs[m] = cfg
        ana = _eval_circuit_from_config(cfg)
        vpa_g3 = get_vpa_metric(ana.v_out_data["G3"], ana.circuit.dt, _CAL_FREQ)
        gain_req = _TARGET_VPA / vpa_g3 if vpa_g3 > 1e-6 else 1.0
        rf, g_actual = _find_e24_rf(gain_req)
        mode_gain[m] = g_actual; mode_rf[m] = rf
        print(f"[{m}] V_pa@G3={vpa_g3:.4f} | gain_req={gain_req:.3f}× → Rf={rf:.0f}Ω → G={g_actual:.3f}× → V_pa_out={vpa_g3*g_actual:.4f}")

    spread = max(mode_gain[m] * get_vpa_metric(_eval_circuit_from_config(mode_configs[m]).v_out_data["G3"],
                _eval_circuit_from_config(mode_configs[m]).circuit.dt, _CAL_FREQ) for m in modes) - \
             min(mode_gain[m] * get_vpa_metric(_eval_circuit_from_config(mode_configs[m]).v_out_data["G3"],
                _eval_circuit_from_config(mode_configs[m]).circuit.dt, _CAL_FREQ) for m in modes)
    print(f"\n  Cross-mode VPA spread: {spread*1000:.2f} mV_w (E24 Rf rounding only)")

    # Build BOM dict
    all_mode_components = {}
    for m in modes:
        rd1, rd2 = resistors["per_mode_rd"][m]
        all_mode_components[m] = {"rd1":rd1, "rs1":resistors["rs1"], "rd2":rd2, "rs2":resistors["rs2"],
                                   "c1":c1, "c2":c2, "c3":c3_by_mode[m],
                                   "rf_opamp":mode_rf[m], "opamp_gain":mode_gain[m]}

    # Phase D: Full analytics
    for m in modes:
        execute_mode_analytics(m, mode_configs[m], mode_gain[m], mode_rf[m])

    write_component_tsv(all_mode_components, global_resistors=resistors, filename="component_bom.tsv")
