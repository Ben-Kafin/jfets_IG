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

# Print engine status only in main process (not in ProcessPoolExecutor workers)
if __name__ == "__main__":
    if _HAS_NUMBA:
        print("[ENGINE] Numba JIT available — using compiled transient solver", flush=True)
    else:
        print("[ENGINE] Numba not found — using Python transient solver "
              "(pip install numba for 20-50× speedup)", flush=True)

# --- Design Constants ---
_INPUT_AMPLITUDE = 0.25  # peak pickup voltage (V) — hot humbucker worst case
_INPUT_VPA_EST   = 0.175 # A-weighted raw pickup level (V_w) — estimated from multi-tone sim
_BLOCKING_FC_HZ  = 36.0  # primary blocking bound: fc ≤ 36 Hz (6²)
_BLOCKING_T5_MS  = 36.0  # primary blocking bound: 5τ ≤ 42 ms (6 × 7 = pickup combos)
_BLOCKING_FC_FLOOR_HZ = 20.0   # secondary floor: fc ≥ 10 Hz
_BLOCKING_T5_FLOOR_MS = 20.0   # secondary floor: 5τ ≥ 20 ms

# ==========================================================================
#  Numba-JIT Transient Solver Engine
# ==========================================================================
#  Eliminates all Python overhead from the inner transient loop:
#    - No dict lookups (topology pre-flattened to integer-indexed arrays)
#    - No Python object attribute access (JFET params in flat float arrays)
#    - No scipy.optimize.root overhead (hand-rolled Newton-Raphson)
#    - Full timestep loop compiled to machine code
#
#  Unified node indexing: v_all[0:dim] = active unknowns,
#    v_all[dim+0]=GND, v_all[dim+1]=input, v_all[dim+2]=V_FORCE, v_all[dim+3]=V_REF
#  Components store integer indices into this unified array.
# ==========================================================================

@njit(cache=True)
def _gate_junction_jit(v, v_t, is_0, g_leak):
    """Gate-source or gate-drain junction current (inlined for JIT)."""
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
    """
    Exact replica of Circuit._jfet_physics, Numba-compiled.
    Returns (i_drain, gm, i_gate, cgs_dynamic, cgd_dynamic).
    """
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
    """
    KCL residual for Newton solver — pure array operations, no Python objects.
    v_guess: (dim,) active node voltages
    v_fixed: (4,) = [GND, input, V_FORCE, V_REF]
    """
    n_total = dim + 4
    v_all = np.empty(n_total)
    v_all[:dim] = v_guess
    v_all[dim] = v_fixed[0]      # GND
    v_all[dim + 1] = v_fixed[1]  # input
    v_all[dim + 2] = v_fixed[2]  # V_FORCE
    v_all[dim + 3] = v_fixed[3]  # V_REF

    res = np.zeros(dim)

    # PSU: current into + node
    if psu_idx >= 0:
        res[psu_idx] -= (vdd - v_all[psu_idx]) / r_psu

    # Resistors
    n_r = r_n1.shape[0]
    for k in range(n_r):
        i1 = r_n1[k]
        i2 = r_n2[k]
        i_r = (v_all[i1] - v_all[i2]) * r_g[k]
        if i1 < dim:
            res[i1] += i_r
        if i2 < dim:
            res[i2] -= i_r

    # Capacitors
    n_c = c_n1.shape[0]
    for k in range(n_c):
        i1 = c_n1[k]
        i2 = c_n2[k]
        dv = (v_all[i1] - v_all[i2]) - (v_prev_all[i1] - v_prev_all[i2])
        i_c = dv / (c_esr[k] + dt / c_val[k])
        if i1 < dim:
            res[i1] += i_c
        if i2 < dim:
            res[i2] -= i_c

    # Inductors
    n_l = l_n1.shape[0]
    for k in range(n_l):
        i1 = l_n1[k]
        i2 = l_n2[k]
        v_ind = v_all[i1] - v_all[i2]
        denom = 1.0 + l_rdc[k] * dt / l_val[k]
        i_l = (i_l_prev[k] + v_ind * dt / l_val[k]) / denom
        v_ind_prev = v_prev_all[i1] - v_prev_all[i2]
        i_cp = (l_cp[k] / dt) * (v_ind - v_ind_prev)
        i_total = i_l + i_cp
        if i1 < dim:
            res[i1] += i_total
        if i2 < dim:
            res[i2] -= i_total

    # JFETs
    n_j = j_nd.shape[0]
    for k in range(n_j):
        nd = j_nd[k]
        ng = j_ng[k]
        ns = j_ns[k]
        v_gs = v_all[ng] - v_all[ns]
        v_ds = v_all[nd] - v_all[ns]
        i_d, _, i_g, _, _ = _jfet_physics_jit(
            j_params[k, 0], j_params[k, 1], j_params[k, 2],
            j_params[k, 3], j_params[k, 4], v_gs, v_ds)

        cgs_p = c_states[k, 0]
        cgd_p = c_states[k, 1]
        v_gd = v_gs - v_ds
        v_gs_prev = v_prev_all[ng] - v_prev_all[ns]
        v_ds_prev = v_prev_all[nd] - v_prev_all[ns]
        i_cgs = cgs_p * (v_gs - v_gs_prev) / dt
        i_cgd = cgd_p * (v_gd - (v_gs_prev - v_ds_prev)) / dt

        if nd < dim:
            res[nd] += (i_d - i_cgd)
        if ns < dim:
            res[ns] -= (i_d + i_g + i_cgs)
        if ng < dim:
            res[ng] += (i_g + i_cgs + i_cgd)

    return res * 1e3


@njit(cache=True)
def _newton_solve_jit(v_guess, v_fixed, dt, dim,
                      r_n1, r_n2, r_g,
                      c_n1, c_n2, c_val, c_esr,
                      l_n1, l_n2, l_val, l_rdc, l_cp,
                      j_nd, j_ng, j_ns, j_params,
                      v_prev_all, i_l_prev, c_states,
                      psu_idx, vdd, r_psu,
                      tol=1e-9, max_iter=50):
    """
    Hand-rolled Newton-Raphson with finite-difference Jacobian.
    For dim=12, Jacobian computation = 12 residual evals — trivial in JIT.
    """
    args = (v_fixed, dt, dim,
            r_n1, r_n2, r_g,
            c_n1, c_n2, c_val, c_esr,
            l_n1, l_n2, l_val, l_rdc, l_cp,
            j_nd, j_ng, j_ns, j_params,
            v_prev_all, i_l_prev, c_states,
            psu_idx, vdd, r_psu)

    v = v_guess.copy()
    eps_fd = 1e-8

    for iteration in range(max_iter):
        f0 = _kcl_residual_jit(v, *args)
        norm_f = 0.0
        for ii in range(dim):
            norm_f += f0[ii] * f0[ii]
        if norm_f < tol * tol * dim:
            break

        # Finite-difference Jacobian
        J = np.empty((dim, dim))
        for col in range(dim):
            v_pert = v.copy()
            h = max(eps_fd, abs(v[col]) * eps_fd)
            v_pert[col] += h
            f_pert = _kcl_residual_jit(v_pert, *args)
            for row in range(dim):
                J[row, col] = (f_pert[row] - f0[row]) / h

        # Solve J @ dv = -f0
        dv = np.linalg.solve(J, -f0)

        # Damped line search
        alpha = 1.0
        for _ in range(10):
            v_new = v + alpha * dv
            f_new = _kcl_residual_jit(v_new, *args)
            norm_new = 0.0
            for ii in range(dim):
                norm_new += f_new[ii] * f_new[ii]
            if norm_new < norm_f:
                break
            alpha *= 0.5
        v = v + alpha * dv

    return v


@njit(cache=True)
def _transient_loop_jit(v_in_array, total_samples, dt, dim,
                        r_n1, r_n2, r_g,
                        c_n1, c_n2, c_val, c_esr,
                        l_n1, l_n2, l_val, l_rdc, l_cp,
                        j_nd, j_ng, j_ns, j_params,
                        v_prev_all_init, i_l_prev_init, c_states_init,
                        psu_idx, vdd, r_psu,
                        input_idx, input_dc,
                        v_fixed_base, monitor_indices,
                        v_guess_init):
    """
    Full transient time-stepping loop — compiled, zero Python overhead.

    v_fixed_base: (4,) = [GND, input_dc, V_FORCE, V_REF] (input updated per step)
    monitor_indices: (n_mon,) int — unified indices of nodes to record
                     -1 = record input signal directly
    """
    n_mon = monitor_indices.shape[0]
    v_out = np.zeros((n_mon, total_samples))
    v_prev_all = v_prev_all_init.copy()
    i_l_prev = i_l_prev_init.copy()
    c_states = c_states_init.copy()
    v_guess = v_guess_init.copy()
    v_fixed = v_fixed_base.copy()
    n_j = j_nd.shape[0]
    n_l = l_n1.shape[0]

    for i in range(total_samples):
        # Update input node voltage for current step
        v_inst = v_in_array[i]
        v_fixed[1] = input_dc + v_inst

        # Newton solve
        v_sol = _newton_solve_jit(
            v_guess, v_fixed, dt, dim,
            r_n1, r_n2, r_g,
            c_n1, c_n2, c_val, c_esr,
            l_n1, l_n2, l_val, l_rdc, l_cp,
            j_nd, j_ng, j_ns, j_params,
            v_prev_all, i_l_prev, c_states,
            psu_idx, vdd, r_psu)
        v_guess = v_sol

        # Update v_prev_all: active nodes from solution
        for k in range(dim):
            v_prev_all[k] = v_sol[k]
        # Fixed nodes: input already set, others don't change
        v_prev_all[input_idx] = input_dc + v_inst

        # Update inductor currents
        for k in range(n_l):
            v_ind = v_prev_all[l_n1[k]] - v_prev_all[l_n2[k]]
            i_l_prev[k] = (i_l_prev[k] + v_ind * dt / l_val[k]) / (1.0 + l_rdc[k] * dt / l_val[k])

        # Update JFET dynamic capacitances
        for k in range(n_j):
            v_gs = v_prev_all[j_ng[k]] - v_prev_all[j_ns[k]]
            v_ds = v_prev_all[j_nd[k]] - v_prev_all[j_ns[k]]
            _, _, _, cgs_n, cgd_n = _jfet_physics_jit(
                j_params[k, 0], j_params[k, 1], j_params[k, 2],
                j_params[k, 3], j_params[k, 4], v_gs, v_ds)
            c_states[k, 0] = cgs_n
            c_states[k, 1] = cgd_n

        # Record monitored nodes
        for k in range(n_mon):
            idx = monitor_indices[k]
            if idx == -1:
                v_out[k, i] = v_inst
            elif idx < dim:
                v_out[k, i] = v_sol[idx]
            else:
                v_out[k, i] = v_prev_all[idx]

    return v_sol, v_prev_all, i_l_prev, c_states, v_out


# --- Core Physical Component Classes ---
class Resistor:
    def __init__(self, name, value, node1, node2):
        self.name = name
        self.value = max(float(value), 1.0)
        self.node1 = node1
        self.node2 = node2

class Capacitor:
    def __init__(self, name, value, node1, node2, esr=0.01):
        self.name = name
        self.value = float(value)
        self.node1 = node1
        self.node2 = node2
        self.esr = esr

class Inductor:
    def __init__(self, name, value, node1, node2, r_dc=1.0, c_p=150e-12):
        self.name = name
        self.value = float(value)
        self.node1 = node1
        self.node2 = node2
        self.r_dc = r_dc
        self.c_p = c_p

class JFET:
    def __init__(self, name, idss, vp, node_d, node_g, node_s):
        self.name = name
        self.idss = float(idss)
        self.vp = float(vp)
        self.node_d = node_d
        self.node_g = node_g
        self.node_s = node_s
        self.cgs, self.cgd = 2.0e-12, 2.0e-12
        self.lambda_mod = 0.0073
        self.current_cgs = self.cgs
        self.current_cgd = self.cgd

# --- High-Fidelity Physics Engine ---
class Circuit:
    def __init__(self, v_dd_ideal=18.0, r_psu=100.0, v_ctrl_force=0.0):
        self.v_dd_ideal, self.r_psu = v_dd_ideal, r_psu
        self.v_ctrl_force = v_ctrl_force
        self.resistors, self.capacitors, self.inductors, self.jfets = [], [], [], []
        self.nodes = ["-", "+", "V_FORCE"]
        self.node_map, self.dc_op = {}, {}
        self.dt = 0.0
        self.t_len = 0

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
        v_t = 0.02569
        vp_t = j.vp
        idss_t = j.idss
        is_rev = v_ds < 0.0
        v_ds_eff = -v_ds if is_rev else v_ds
        v_gs_eff = v_gs - v_ds if is_rev else v_gs
        early = (1.0 + j.lambda_mod * v_ds_eff)
        beta = idss_t / (vp_t**2)
        v_gst = v_gs_eff - vp_t
        
        v_slope = 1.5 * v_t # Dynamic Subthreshold Ideality (LSK489 Match)
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
        
        v_gd = v_gs - v_ds
        v_dg = -v_gd
        is_0 = 1.0e-14
        g_leak = 1.33e-13
        alpha_ii = 0.01
        beta_ii = 138.15
        
        def calc_gate_junction(v):
            if v < 0:
                return -(is_0 + g_leak * abs(v))
            v_crit = v_t * np.log(v_t / (1.414 * is_0))
            if v > v_crit:
                i_crit = is_0 * (np.exp(v_crit / v_t) - 1.0)
                g_crit = (is_0 / v_t) * np.exp(v_crit / v_t)
                return i_crit + g_crit * (v - v_crit)
            return is_0 * (np.exp(v / v_t) - 1.0)

        i_gs = calc_gate_junction(v_gs)
        i_gd_junction = calc_gate_junction(v_gd)
        
        i_g_ii = 0.0
        if v_dg > 0.1:
            i_g_ii = abs(i_chan) * alpha_ii * np.exp(-beta_ii / v_dg)
            
        i_gate = i_gs + i_gd_junction + i_g_ii
        i_drain = i_chan - i_gd_junction + i_g_ii
        
        # Capacitance values are returned for the current node state; updates happen post-solve
        cgs_dynamic = j.cgs / np.sqrt(max(0.01, 1.0 - min(v_gs, 0.55) / 0.6))
        cgd_dynamic = j.cgd / np.sqrt(max(0.01, 1.0 - min(v_gd, 0.55) / 0.6))
        
        return i_drain, gm, i_gate, cgs_dynamic, cgd_dynamic

    def solve_dc_bias(self, input_node="v_ideal"):
        active_nodes = self._get_active_nodes()
        self.node_map = {name: idx for idx, name in enumerate(active_nodes)}
        dim = len(active_nodes)
        def kcl_equations(v_guess):
            v = {"-": 0.0, "V_FORCE": self.v_ctrl_force, "V_REF": self.v_dd_ideal / 2.0}
            for name, idx in self.node_map.items(): v[name] = v_guess[idx]
            v[input_node] = 0.0
            residuals = np.zeros(dim)
            if "+" in self.node_map:
                residuals[self.node_map["+"]] -= (self.v_dd_ideal - v["+"]) / self.r_psu
            for r in self.resistors:
                i_r = (v[r.node1] - v[r.node2]) / r.value
                if r.node1 in self.node_map: residuals[self.node_map[r.node1]] += i_r
                if r.node2 in self.node_map: residuals[self.node_map[r.node2]] -= i_r
            for l in self.inductors:
                i_l_dc = (v[l.node1] - v[l.node2]) / l.r_dc
                if l.node1 in self.node_map: residuals[self.node_map[l.node1]] += i_l_dc
                if l.node2 in self.node_map: residuals[self.node_map[l.node2]] -= i_l_dc
            for j in self.jfets:
                v_gs, v_ds = v[j.node_g] - v[j.node_s], v[j.node_d] - v[j.node_s]
                # Sync: Unpack all 5 values; gm and capacitances are unused in DC solve
                i_d, _, i_g, _, _ = self._jfet_physics(j, v_gs, v_ds)
                if j.node_d in self.node_map: residuals[self.node_map[j.node_d]] += i_d
                if j.node_s in self.node_map: residuals[self.node_map[j.node_s]] -= (i_d + i_g)
                if j.node_g in self.node_map: residuals[self.node_map[j.node_g]] += i_g
            if input_node in self.node_map:
                residuals[self.node_map[input_node]] = v_guess[self.node_map[input_node]] - 0.0
            return residuals * 1e3
            
        v_guess_init = np.ones(dim) * (self.v_dd_ideal / 2.0)
        if "+" in self.node_map: v_guess_init[self.node_map["+"]] = self.v_dd_ideal
        if "D1" in self.node_map: v_guess_init[self.node_map["D1"]] = self.v_dd_ideal * 0.75
        if "D2" in self.node_map: v_guess_init[self.node_map["D2"]] = self.v_dd_ideal * 0.75
        
        if "G3" in self.node_map: v_guess_init[self.node_map["G3"]] = 0.0
        if "S3" in self.node_map: v_guess_init[self.node_map["S3"]] = 1.0
        
        # Bounds anchored by Datasheet BV_GSS (-60V) 
        sol = least_squares(kcl_equations, v_guess_init, bounds=(-60.0, self.v_dd_ideal + 5.0), method='trf')
        v_sol = sol.x
        
        self.dc_op = {"-": 0.0, "V_FORCE": self.v_ctrl_force, "V_REF": self.v_dd_ideal / 2.0}
        for name, idx in self.node_map.items(): self.dc_op[name] = v_sol[idx]
        return self.dc_op

    def solve_ac_thevenin(self, target_cap):
        active_nodes = [n for n in self.nodes if n not in ["+", "-", "v_ideal", "V_FORCE", "V_REF"]]
        node_map = {name: idx for idx, name in enumerate(active_nodes)}
        dim = len(active_nodes)
        Y = np.zeros((dim, dim))
        Y += np.eye(dim) * 1e-12 
        for r in self.resistors:
            g = 1.0 / r.value
            if r.node1 in node_map: Y[node_map[r.node1], node_map[r.node1]] += g
            if r.node2 in node_map: Y[node_map[r.node2], node_map[r.node2]] += g
            if r.node1 in node_map and r.node2 in node_map:
                Y[node_map[r.node1], node_map[r.node2]] -= g
                Y[node_map[r.node2], node_map[r.node1]] -= g
        for l in self.inductors:
            g = 1.0 / l.r_dc
            if l.node1 in node_map: Y[node_map[l.node1], node_map[l.node1]] += g
            if l.node2 in node_map: Y[node_map[l.node2], node_map[l.node2]] += g
            if l.node1 in node_map and l.node2 in node_map:
                Y[node_map[l.node1], node_map[l.node2]] -= g
                Y[node_map[l.node2], node_map[l.node1]] -= g
        for j in self.jfets:
            v_gs, v_ds = (self.dc_op[j.node_g]-self.dc_op[j.node_s]), (self.dc_op[j.node_d]-self.dc_op[j.node_s])
            # Sync: Extract gm; currents and dynamic capacitances are unused in linear AC solve
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
        """
        Pack circuit topology into flat NumPy arrays for the JIT engine.
        Unified node indexing: [0:dim]=active, dim=GND, dim+1=input,
        dim+2=V_FORCE, dim+3=V_REF.
        """
        fixed_names = ["-", input_node, "V_FORCE", "V_REF"]

        def node_idx(name):
            if name in node_map:
                return node_map[name]
            return dim + fixed_names.index(name)

        # Resistors: (n1_idx, n2_idx, conductance)
        r_n1 = np.array([node_idx(r.node1) for r in self.resistors], dtype=np.int64)
        r_n2 = np.array([node_idx(r.node2) for r in self.resistors], dtype=np.int64)
        r_g = np.array([1.0 / r.value for r in self.resistors])

        # Capacitors: (n1_idx, n2_idx, value, esr)
        c_n1 = np.array([node_idx(c.node1) for c in self.capacitors], dtype=np.int64)
        c_n2 = np.array([node_idx(c.node2) for c in self.capacitors], dtype=np.int64)
        c_val_arr = np.array([c.value for c in self.capacitors])
        c_esr_arr = np.array([c.esr for c in self.capacitors])

        # Inductors: (n1_idx, n2_idx, value, r_dc, c_p)
        l_n1 = np.array([node_idx(l.node1) for l in self.inductors], dtype=np.int64)
        l_n2 = np.array([node_idx(l.node2) for l in self.inductors], dtype=np.int64)
        l_val_arr = np.array([l.value for l in self.inductors])
        l_rdc = np.array([l.r_dc for l in self.inductors])
        l_cp_arr = np.array([l.c_p for l in self.inductors])

        # JFETs: (nd, ng, ns) + (idss, vp, lambda_mod, cgs0, cgd0)
        j_nd = np.array([node_idx(j.node_d) for j in self.jfets], dtype=np.int64)
        j_ng = np.array([node_idx(j.node_g) for j in self.jfets], dtype=np.int64)
        j_ns = np.array([node_idx(j.node_s) for j in self.jfets], dtype=np.int64)
        j_params = np.array([[j.idss, j.vp, j.lambda_mod, j.cgs, j.cgd]
                             for j in self.jfets])

        # PSU: + node index (-1 if not active)
        psu_idx = node_map.get("+", -1)

        # v_prev_all: unified voltage array from previous step
        n_total = dim + 4
        active_nodes = [n for n, _ in sorted(node_map.items(), key=lambda x: x[1])]
        v_prev_all = np.zeros(n_total)
        for name, idx in node_map.items():
            v_prev_all[idx] = v_prev[name]
        v_prev_all[dim] = 0.0                    # GND
        v_prev_all[dim + 1] = v_prev.get(input_node, 0.0)  # input
        v_prev_all[dim + 2] = self.v_ctrl_force   # V_FORCE
        v_prev_all[dim + 3] = self.v_dd_ideal / 2.0  # V_REF

        # Inductor currents: flat array in same order as self.inductors
        i_l_prev_arr = np.array([i_l_prev[l.name] for l in self.inductors])

        # JFET cap states: (n_jfets, 2) = (cgs, cgd)
        c_states_arr = np.array([[c_states[j.name][0], c_states[j.name][1]]
                                 for j in self.jfets])

        # Input node unified index
        input_idx = node_idx(input_node)

        return {
            "r_n1": r_n1, "r_n2": r_n2, "r_g": r_g,
            "c_n1": c_n1, "c_n2": c_n2, "c_val": c_val_arr, "c_esr": c_esr_arr,
            "l_n1": l_n1, "l_n2": l_n2, "l_val": l_val_arr, "l_rdc": l_rdc, "l_cp": l_cp_arr,
            "j_nd": j_nd, "j_ng": j_ng, "j_ns": j_ns, "j_params": j_params,
            "psu_idx": psu_idx,
            "v_prev_all": v_prev_all, "i_l_prev": i_l_prev_arr,
            "c_states": c_states_arr, "input_idx": input_idx,
            "active_nodes": active_nodes,
        }

    def solve_transient(self, input_node, monitor_nodes, freqs, amplitude=_INPUT_AMPLITUDE, periods=20.0, samples_per_period=4096, use_saved_state=False):
        self.freqs = freqs
        f_base = self.freqs[0]
        dt = (1.0 / f_base) / samples_per_period
        self.dt = dt
        total_samples = int(np.round((periods / f_base) / dt))
        t_start = 0.0
        if use_saved_state and hasattr(self, 'saved_t_end'):
            t_start = self.saved_t_end + dt
        t = np.linspace(t_start, t_start + (periods / f_base), total_samples, endpoint=False)
        self.t_len = len(t)
        v_in_array = np.zeros_like(t)
        for f in self.freqs: v_in_array += (amplitude / len(self.freqs)) * np.sin(2 * np.pi * f * t)
        
        active_nodes = [n for n in self.nodes if n not in ["-", input_node, "V_FORCE", "V_REF"]]
        node_map = {name: idx for idx, name in enumerate(active_nodes)}
        dim = len(active_nodes)
        v_out_data = {n: np.zeros_like(t) for n in monitor_nodes}
        
        if use_saved_state and hasattr(self, 'saved_v_prev'):
            v_prev = {k: v for k, v in self.saved_v_prev.items()}
            i_l_prev = {k: v for k, v in self.saved_i_l_prev.items()}
            c_states = {j.name: (j.current_cgs, j.current_cgd) for j in self.jfets}
        else:
            v_prev = {n: self.dc_op.get(n, 0.0) for n in self.nodes}
            i_l_prev = {l.name: (self.dc_op.get(l.node1, 0.0) - self.dc_op.get(l.node2, 0.0)) / l.r_dc for l in self.inductors}
            c_states = {j.name: (j.cgs, j.cgd) for j in self.jfets}
        
        v_guess = np.array([v_prev[n] for n in active_nodes])

        # ============================================================
        #  JIT FAST PATH — Numba-compiled transient loop
        # ============================================================
        if _HAS_NUMBA:
            flat = self._flatten_for_jit(input_node, node_map, dim,
                                         v_prev, i_l_prev, c_states)

            # Build monitor indices: -1 = record input signal
            mon_idx = np.array([
                -1 if n == input_node else node_map.get(n, flat["input_idx"])
                for n in monitor_nodes
            ], dtype=np.int64)

            v_fixed_base = np.array([
                0.0,                                   # GND
                self.dc_op.get(input_node, 0.0),       # input DC (updated per step)
                self.v_ctrl_force,                      # V_FORCE
                self.v_dd_ideal / 2.0,                  # V_REF
            ])

            v_sol, v_prev_all_out, i_l_out, c_states_out, v_out_arr = \
                _transient_loop_jit(
                    v_in_array, total_samples, dt, dim,
                    flat["r_n1"], flat["r_n2"], flat["r_g"],
                    flat["c_n1"], flat["c_n2"], flat["c_val"], flat["c_esr"],
                    flat["l_n1"], flat["l_n2"], flat["l_val"], flat["l_rdc"], flat["l_cp"],
                    flat["j_nd"], flat["j_ng"], flat["j_ns"], flat["j_params"],
                    flat["v_prev_all"], flat["i_l_prev"], flat["c_states"],
                    flat["psu_idx"], self.v_dd_ideal, self.r_psu,
                    flat["input_idx"], self.dc_op.get(input_node, 0.0),
                    v_fixed_base, mon_idx, v_guess)

            # Unpack outputs into dict format for compatibility
            for k, n in enumerate(monitor_nodes):
                v_out_data[n] = v_out_arr[k]

            # Save state for use_saved_state continuation
            v_prev_out = {}
            for name, idx in node_map.items():
                v_prev_out[name] = v_prev_all_out[idx]
            v_prev_out["-"] = 0.0
            v_prev_out[input_node] = v_prev_all_out[flat["input_idx"]]
            v_prev_out["V_FORCE"] = self.v_ctrl_force
            v_prev_out["V_REF"] = self.v_dd_ideal / 2.0
            for n in self.nodes:
                if n not in v_prev_out:
                    v_prev_out[n] = self.dc_op.get(n, 0.0)
            self.saved_v_prev = v_prev_out

            i_l_prev_out = {}
            for k, l in enumerate(self.inductors):
                i_l_prev_out[l.name] = i_l_out[k]
            self.saved_i_l_prev = i_l_prev_out

            # Update JFET dynamic cap state for inheritance
            for k, j in enumerate(self.jfets):
                j.current_cgs = c_states_out[k, 0]
                j.current_cgd = c_states_out[k, 1]

            self.saved_t_end = t[-1]
            return t, v_in_array, v_out_data

        # ============================================================
        #  PYTHON FALLBACK — original solver (no Numba)
        # ============================================================
        v_inst_current = [0.0]
        
        def kcl_transient(v_guess_t):
            v = {"-": 0.0, "V_FORCE": self.v_ctrl_force, "V_REF": self.v_dd_ideal / 2.0, input_node: self.dc_op.get(input_node, 0.0) + v_inst_current[0]}
            for name, idx in node_map.items(): v[name] = v_guess_t[idx]
            residuals = np.zeros(dim)
            if "+" in node_map:
                residuals[node_map["+"]] -= (self.v_dd_ideal - v["+"]) / self.r_psu
            for r in self.resistors:
                i_r = (v[r.node1] - v[r.node2]) / r.value
                if r.node1 in node_map: residuals[node_map[r.node1]] += i_r
                if r.node2 in node_map: residuals[node_map[r.node2]] -= i_r
            for l in self.inductors:
                v_ind = v[l.node1] - v[l.node2]
                i_l_series = (i_l_prev[l.name] + v_ind * dt / l.value) / (1.0 + l.r_dc * dt / l.value)
                i_c_p = (l.c_p / dt) * (v_ind - (v_prev[l.node1] - v_prev[l.node2]))
                if l.node1 in node_map: residuals[node_map[l.node1]] += (i_l_series + i_c_p)
                if l.node2 in node_map: residuals[node_map[l.node2]] -= (i_l_series + i_c_p)
            for c in self.capacitors:
                i_c = ((v[c.node1]-v[c.node2])-(v_prev[c.node1]-v_prev[c.node2])) / (c.esr + dt/c.value)
                if c.node1 in node_map: residuals[node_map[c.node1]] += i_c
                if c.node2 in node_map: residuals[node_map[c.node2]] -= i_c
            for j in self.jfets:
                v_gs, v_ds = v[j.node_g]-v[j.node_s], v[j.node_d]-v[j.node_s]
                i_d, _, i_g, _, _ = self._jfet_physics(j, v_gs, v_ds)
                cgs_prev, cgd_prev = c_states[j.name]
                v_gd = v_gs - v_ds
                v_gs_prev, v_ds_prev = v_prev[j.node_g]-v_prev[j.node_s], v_prev[j.node_d]-v_prev[j.node_s]
                i_cgs = cgs_prev * (v_gs - v_gs_prev) / dt
                i_cgd = cgd_prev * (v_gd - (v_gs_prev - v_ds_prev)) / dt
                if j.node_d in node_map: residuals[node_map[j.node_d]] += (i_d - i_cgd)
                if j.node_s in node_map: residuals[node_map[j.node_s]] -= (i_d + i_g + i_cgs)
                if j.node_g in node_map: residuals[node_map[j.node_g]] += (i_g + i_cgs + i_cgd)
            return residuals * 1e3

        for i, v_inst in enumerate(v_in_array):
            v_inst_current[0] = v_inst
            sol = root(kcl_transient, v_guess, method='hybr', tol=1e-9)
            v_sol = sol.x
            v_guess = v_sol 
            v_prev[input_node] = self.dc_op.get(input_node, 0.0) + v_inst
            for name, idx in node_map.items(): v_prev[name] = v_sol[idx]
            for l in self.inductors: 
                v_ind = v_prev[l.node1] - v_prev[l.node2]
                i_l_prev[l.name] = (i_l_prev[l.name] + v_ind * dt / l.value) / (1.0 + l.r_dc * dt / l.value)
            for j in self.jfets:
                v_gs, v_ds = v_prev[j.node_g]-v_prev[j.node_s], v_prev[j.node_d]-v_prev[j.node_s]
                _, _, _, cgs_n, cgd_n = self._jfet_physics(j, v_gs, v_ds)
                c_states[j.name] = (cgs_n, cgd_n)
                j.current_cgs, j.current_cgd = cgs_n, cgd_n
            for n in monitor_nodes: v_out_data[n][i] = v_prev[n] if n != input_node else v_inst
            
        self.saved_v_prev = {k: v for k, v in v_prev.items()}
        self.saved_i_l_prev = {k: v for k, v in i_l_prev.items()}
        self.saved_t_end = t[-1]
        return t, v_in_array, v_out_data

# --- Analyzer, Plotting, and Export ---
class CircuitAnalyzer:
    def __init__(self, circuit, monitor_nodes, input_node="v_ideal", amplitude=_INPUT_AMPLITUDE):
        self.circuit = circuit
        self.monitor_nodes = monitor_nodes
        self.input_node = input_node
        self.amplitude = amplitude
        self.t = None
        self.v_in = None
        self.v_out_data = None
        base_freqs = [100.0, 125.0, 150.0]
        oct1_freqs = [f * 2.0 for f in base_freqs]
        oct2_freqs = [f * 4.0 for f in base_freqs]
        self.freqs = base_freqs + oct1_freqs + oct2_freqs
        self.thd_freqs = base_freqs  # single-tone THD only on base frequencies

    def report_dc_bias(self):
        dc_results = self.circuit.solve_dc_bias(input_node=self.input_node)
        for j in self.circuit.jfets:
            v_gs = dc_results.get(j.node_g, 0.0) - dc_results.get(j.node_s, 0.0)
            v_ds = dc_results.get(j.node_d, 0.0) - dc_results.get(j.node_s, 0.0)
            # Sync: Unpack all 5 values to match updated _jfet_physics signature
            i_d, _, _, _, _ = self.circuit._jfet_physics(j, v_gs, v_ds)
            print(f"[{j.name}] Bias -> V_GS: {v_gs:.4f}V | V_DS: {v_ds:.4f}V | I_D: {i_d*1000:.4f}mA")

    def get_max_system_tau(self):
        # Physics Probe: Detect the actual slowest RC path in the solved hardware
        taus = [c.value * self.circuit.solve_ac_thevenin(c) for c in self.circuit.capacitors if c.node1 != c.node2]
        return max(taus) if taus else 0.001

    def run_transient(self):
        f_base = self.freqs[0]
        sys_tau = self.get_max_system_tau()
        # Physics-Driven Timing: 10-Tau Settlement + 20-Period Measurement Window
        t_settle = 10.0 * sys_tau
        t_meas = 20.0 / f_base
        total_sec = t_settle + t_meas
        
        integer_periods = int(np.round(total_sec * f_base))
        self.t, self.v_in, self.v_out_data = self.circuit.solve_transient(
            input_node=self.input_node, monitor_nodes=self.monitor_nodes, 
            freqs=self.freqs, amplitude=self.amplitude,
            periods=integer_periods, samples_per_period=2048
        )

    def report_ac_analytics(self):
        print("\n--- Capacitor Thevenin Analytics ---")
        for cap in self.circuit.capacitors:
            if cap.node1 == cap.node2: continue 
            
            R_th = self.circuit.solve_ac_thevenin(cap)
            if R_th < 1.0e-12: continue 
            
            fc = 1.0 / (2 * np.pi * R_th * cap.value)
            t_rec = 5 * R_th * cap.value
            print(f"[{cap.name}] R_th: {R_th/1000:.2f} kOhms | f_c: {fc:.2f} Hz | 5-Tau Recovery: {t_rec*1000:.2f} ms")

    def report_single_tone_thd(self, node="OUT"):
        """
        True THD measurement via single-tone transient sweeps.

        Runs a separate short transient for each base frequency (100, 125, 150 Hz)
        with ONLY that frequency as input.  No cross-contamination from other
        fundamentals or intermodulation products.  This is how real THD analyzers
        work — one tone at a time.
        """
        print(f"\n--- Single-Tone THD Analysis ({node}) ---")

        for f_test in self.thd_freqs:
            # Run a short single-tone transient
            self.circuit.solve_dc_bias(input_node=self.input_node)
            # 15 periods settlement (circuit already biased from main transient)
            self.circuit.solve_transient(
                input_node=self.input_node, monitor_nodes=[node],
                freqs=[f_test], amplitude=self.amplitude,
                periods=15.0, samples_per_period=2048)
            _, _, thd_data = self.circuit.solve_transient(
                input_node=self.input_node, monitor_nodes=[node],
                freqs=[f_test], amplitude=self.amplitude,
                periods=20.0, samples_per_period=2048, use_saved_state=True)

            v_out = thd_data[node]
            v_ac = v_out - np.mean(v_out)
            N = len(v_ac)
            Y = np.fft.rfft(v_ac * np.hanning(N))
            xf = np.fft.rfftfreq(N, d=self.circuit.dt)
            mag = (2.0 / N * np.abs(Y)) + 1e-12

            idx_f = np.argmin(np.abs(xf - f_test))
            mag_f = mag[idx_f]
            w_mag_f = mag_f * _a_weight(f_test)

            sum_sq_harm = 0.0
            print(f"\nFundamental: {f_test:.0f} Hz | "
                  f"Level: {mag_f*1000:.2f} mV | "
                  f"A-weighted: {w_mag_f*1000:.3f} mV_w")

            for h in range(2, 10):
                hf = f_test * h
                if hf > xf[-1]:
                    break
                idx_h = np.argmin(np.abs(xf - hf))
                mag_h = mag[idx_h]
                w_mag_h = mag_h * _a_weight(hf)
                sum_sq_harm += w_mag_h**2
                ihl = (mag_h / mag_f * 100.0) if mag_f > 1e-9 else 0.0
                print(f"  H{h} ({hf:.0f} Hz): {mag_h*1000:.3f} mV | "
                      f"IHL: {ihl:.2f}% | "
                      f"A-wt: {w_mag_h*1000:.4f} mV_w")

            thd = np.sqrt(sum_sq_harm) / w_mag_f * 100.0 if w_mag_f > 1e-12 else 0.0
            print(f"  THD({f_test:.0f}Hz): {thd:.2f}%")

    def plot_waveforms(self, mode):
        fig = plt.figure(figsize=(16, 12))
        plt.suptitle(f"{mode} Mode: Signal Chain and Harmonic Analysis (Steady State Extract)", fontweight='bold', fontsize=14)
        gs = gridspec.GridSpec(5, 2, figure=fig, width_ratios=[1, 1])
        
        # Automated Settlement: wait for 10x the actual system Tau
        start_idx = int(np.round((10.0 * self.get_max_system_tau()) / self.circuit.dt))
        t_sliced = self.t[start_idx:]
        t_plot_ms = (t_sliced - 0.10) * 1000.0
        
        nodes_to_plot = ["v_ideal", "G1", "D1", "D2", "OUT"]
        titles = [
            "Pre-Pickup Source (v_ideal)",
            "Pre-J1 Gate Input (G1)",
            "J1 Drain Output (D1)",
            "J2 Drain Output (D2)",
            "Final Master Output (OUT)"
        ]
        colors = ['gray', 'green', 'blue', 'purple', 'black']
        
        for i, (node, title, color) in enumerate(zip(nodes_to_plot, titles, colors)):
            ax = fig.add_subplot(gs[i, 0])
            v_sliced = self.v_out_data[node][start_idx:]
            
            if node in ["D1", "D2"]:
                v_sliced = v_sliced - np.mean(v_sliced)
                ax.set_ylabel("AC Amplitude (V)", fontsize=9)
            else:
                ax.set_ylabel("Amplitude (V)", fontsize=9)
                
            ax.plot(t_plot_ms, v_sliced, color=color, linewidth=1.5)
            ax.set_title(title, fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 40)
            if i == 4:
                ax.set_xlabel("Time (ms)", fontsize=10)
            else:
                ax.set_xticklabels([])
                
        ax_fft = fig.add_subplot(gs[:, 1])
        v_out_ac = self.v_out_data["OUT"][start_idx:] - np.mean(self.v_out_data["OUT"][start_idx:])
        N = len(v_out_ac)
        Y = np.fft.rfft(v_out_ac * np.hanning(N))
        xf = np.fft.rfftfreq(N, d=self.circuit.dt)
        mag = (2.0/N * np.abs(Y)) + 1e-12 

        weighted_mag = mag * _a_weight(xf)  # absolute A-weighted voltage (V_w)

        # Full spectrum as gray fill (log-log, positive upward)
        ax_fft.fill_between(xf[1:], weighted_mag[1:], 1e-9, color='gray', alpha=0.15)
        ax_fft.plot(xf[1:], weighted_mag[1:], color='gray', alpha=0.3, linewidth=0.5)

        # Stem plot: each input frequency and its harmonics going UPWARD
        color_cycle = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'brown', 'olive']
        for f_val, c in zip(self.freqs, color_cycle):
            h_freqs_all = np.arange(1, 10) * f_val
            h_freqs_valid = [hf for hf in h_freqs_all if hf < xf[-1]]
            h_vals = [weighted_mag[np.argmin(np.abs(xf - hf))] for hf in h_freqs_valid]
            marker, stemlines, baseline = ax_fft.stem(
                h_freqs_valid, h_vals,
                linefmt=c, basefmt=' ', markerfmt='o')
            plt.setp(marker, color=c, markersize=4, alpha=0.8)
            plt.setp(stemlines, color=c, alpha=0.6, linewidth=1.5)
            
        ax_fft.set_title("A-Weighted Harmonic Spectrum", fontsize=12)
        ax_fft.set_xlabel("Frequency (Hz) [Log Scale]", fontsize=10)
        ax_fft.set_ylabel("A-Weighted Level (V_w)", fontsize=10)
        ax_fft.set_xscale('log')
        ax_fft.set_yscale('log')
        ax_fft.set_xlim(50, 10000)
        ax_fft.set_ylim(1e-6, weighted_mag[1:].max() * 2.0)
        ax_fft.grid(True, which="both", alpha=0.2)
        
        plt.tight_layout()
        plt.savefig(f'mode_{mode}_analysis.png')
        plt.close()

    def export_audio(self, mode, target_duration_sec=4.0, target_sr=44100):
        print(f"--- Exporting Audio: {mode} Mode ---")
        # Automated Settlement: wait for 10x the actual system Tau
        start_idx = int(np.round((10.0 * self.get_max_system_tau()) / self.circuit.dt))
        v_chunk = self.v_out_data["OUT"][start_idx:]
        
        target_samples_chunk = int(np.round(0.05 * target_sr))
        chunk_44k = resample(v_chunk, target_samples_chunk)
        chunk_44k -= np.mean(chunk_44k)
        
        N_44k = int(np.round((1.0 / 25.0) * target_sr))
        zero_crossings = np.where((chunk_44k[:-1] <= 0) & (chunk_44k[1:] > 0))[0]
        safe_crossings = zero_crossings[zero_crossings > N_44k]
        
        z_start = safe_crossings[0] if len(safe_crossings) > 0 else N_44k
        z_end = z_start + N_44k
        deterministic_period = chunk_44k[z_start:z_end].copy()
        
        exact_samples = int(np.round(target_duration_sec * target_sr))
        tiles_needed = int(np.ceil(exact_samples / len(deterministic_period)))
        sliced_wave = np.tile(deterministic_period, tiles_needed)[:exact_samples]
        
        max_val = np.max(np.abs(sliced_wave))
        normalized_wave = (sliced_wave / max_val) if max_val > 0 else sliced_wave
        wavfile.write(f'mode_{mode}_audio.wav', target_sr, np.int16(normalized_wave * 32767))

def process_unified_circuit(sim, comp_list):
    for comp in comp_list:
        ctype = comp.get("type")
        name = comp.get("name")
        if ctype == "Resistor": sim.add(Resistor(name, comp["val"], comp["n1"], comp["n2"]))
        elif ctype == "Capacitor": sim.add(Capacitor(name, comp["val"], comp["n1"], comp["n2"]))
        elif ctype == "Inductor": sim.add(Inductor(name, comp["val"], comp["n1"], comp["n2"], r_dc=comp.get("r_dc", 1.0), c_p=comp.get("c_p", 150e-12)))
        elif ctype == "JFET": sim.add(JFET(name, comp["idss"], comp["vp"], comp["nd"], comp["ng"], comp["ns"]))

def _a_weight(f):
    """IEC 61672 A-weighting transfer function (scalar or array)."""
    if np.isscalar(f):
        if f < 1.0: return 1e-6
        f2 = f**2
        return (12194**2 * f**4) / ((f2 + 20.6**2) * np.sqrt((f2 + 107.7**2) * (f2 + 737.9**2)) * (f2 + 12194**2))
    f = np.asarray(f, dtype=float)
    f2 = f**2
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.where(f > 1.0,
            (12194**2 * f**4) / ((f2 + 20.6**2) * np.sqrt((f2 + 107.7**2) * (f2 + 737.9**2)) * (f2 + 12194**2)),
            1e-6)
    return result


def get_vpa_metric(v_out, dt, freqs):
    # Automatic Windowing: Capture exactly 2 base periods
    f_base = freqs[0]
    window_sec = 2.0 / f_base
    start_idx = -int(np.round(window_sec / dt)) if len(v_out) > int(np.round(window_sec/dt)) else 0
    v_steady = v_out[start_idx:]
    v_ac = v_steady - np.mean(v_steady)
    N = len(v_ac)
    
    Y = np.fft.rfft(v_ac * np.hanning(N))
    xf = np.fft.rfftfreq(N, d=dt)
    mag = (2.0/N * np.abs(Y)) + 1e-12

    vpa_sq_sum = 0.0
    for f in freqs:
        for h in range(1, 10):
            idx = np.argmin(np.abs(xf - (f * h)))
            vpa_sq_sum += (mag[idx] * _a_weight(f * h))**2
    return np.sqrt(vpa_sq_sum)

# --- Physics-Derived Bias Point Solver ---
def calc_self_bias(alpha, idss, vp_abs):
    """
    Derive self-bias voltages and resistors from a dimensionless operating
    fraction alpha = Id/IDSS, using the Shockley square-law equation.

    Args:
        alpha   : Target Id/IDSS fraction (0 < alpha < 1). Expresses how far
                  below IDSS the device is biased. Derived from LSK489 datasheet
                  plots (gfs vs Id, noise vs f, output characteristics).
        idss    : Device IDSS in Amps (datasheet parameter).
        vp_abs  : |Vp|, the absolute pinch-off voltage in Volts (datasheet).

    Returns:
        (Vs, Rs, Id) — source voltage, source resistor, quiescent drain current.

    Note: Vs = |Vp| * (1 - sqrt(alpha)) falls out of Shockley directly.
          For self-biased N-channel with gate at AC ground, Vgs = -Vs.
    """
    Id = alpha * idss
    Vs = vp_abs * (1.0 - np.sqrt(alpha))
    Rs = Vs / Id
    return Vs, Rs, Id

# --- Parallelized Search Architecture ---
#
# Picklable module-level functions:
#   _eval_circuit_from_config  — builds circuit from a serializable config dict
#   _compute_r7_mid            — deterministic R7 from tau midpoint
#   _eval_cap_worker           — evaluates one (config, cap, R7) + blocking validation
#   _resolve_global_resistors  — R_g from IGSS, Rs from Shockley, per-mode Rd from km
#   _probe_mode_rth            — full-circuit Thevenin probes per mode
#   _reconcile_global_caps     — cross-mode E24 cap sizing (both bounds enforced)
#
# Blocking Enforcement:
#   ALL coupling caps (C1, C2, C3, C4) must satisfy:
#     tau_floor ≤ R_th × C ≤ tau_ceiling
#   R_th is measured in the full circuit (with L_PICKUP, all JFETs, R7).
#   Caps are sized in _reconcile_global_caps (E24-only for coupling caps).
#   Every candidate is re-validated in _eval_cap_worker.
#
# Combination-Cap Architecture:
#   C3_shunt(mode) = C_base + C_delta(mode)
#   The optimizer directly enumerates all (C_base, C_delta) E24 pairs.
#   No raw E24 sweep — combinations ARE the search space.
#
# Parallelism:
#   Phase A:  3 modes run _probe_mode_rth in parallel
#   Phase B1: Sparse E24 scan per mode (VPA gate filter)
#   Phase B2: Fine combo evaluation in viable range only
#   Phase C:  Cross-mode trio matching + C_base decomposition

# Shared E24 cap table (module-level, computed once)
_E24_BASES = [1.0, 1.1, 1.2, 1.3, 1.5, 1.6, 1.8, 2.0, 2.2, 2.4, 2.7, 3.0,
              3.3, 3.6, 3.9, 4.3, 4.7, 5.1, 5.6, 6.2, 6.8, 7.5, 8.2, 9.1]
_E24_CAPS = np.array([b * m for m in [1e-12, 10e-12, 100e-12, 1e-9, 10e-9, 100e-9]
                      for b in _E24_BASES])
# Indices for 1 nF → 100 nF search range
_IDX_1NF   = int(np.argmin(np.abs(_E24_CAPS - 1e-9)))
_IDX_100NF = int(np.argmin(np.abs(_E24_CAPS - 100e-9)))
_CAP_SEARCH_RANGE = _E24_CAPS[_IDX_1NF : _IDX_100NF + 1]

_MONITOR_NODES = ["v_ideal", "G1", "D1", "G2", "D2", "S3", "OUT"]
_CAL_FREQ      = [1000.0]
_TARGET_VPA    = 0.75

# Delta caps for combination search: E24 values ≥ 1 nF (sub-nF deltas are
# negligible for totals in the 30–70 nF range where this circuit operates)
_E24_DELTA_CAPS = _E24_CAPS[_E24_CAPS >= 1e-9]

def _cap_key(cap_value):
    """Round cap to 0.1 pF resolution for float-safe dict keying."""
    return round(float(cap_value) * 1e13)


def _eval_circuit_from_config(config, c3_shunt_test, r_tot_test,
                               full_run=False, high_res=False):
    """
    Build and evaluate a circuit from a serializable config dict.
    Module-level replacement for the old closure-based eval_circuit.

    Returns a CircuitAnalyzer with transient data populated.
    """
    c3_shunt_test = max(1e-12, c3_shunt_test)
    r_tot_test    = max(1.0, r_tot_test)
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
        {"type": "Capacitor",  "name": "C3_shunt",   "val": c3_shunt_test,    "n1": "G3",      "n2": "V_REF"},
        {"type": "Resistor",   "name": "R7",         "val": r_tot_test,       "n1": "G3",      "n2": "V_REF"},
        {"type": "JFET",       "name": "Q3",         "idss": config.get("idss_q3", 0.0055), "vp": config.get("vp_q3", -2.0), "nd": "+", "ng": "G3", "ns": "S3"},
        {"type": "JFET",       "name": "Q4",         "idss": config.get("idss_q4", 0.0055), "vp": config.get("vp_q4", -2.0), "nd": "S3", "ng": "-", "ns": "S_CS"},
        {"type": "Resistor",   "name": "Rs_CS",      "val": config.get("rs_cs", 100.0),  "n1": "S_CS",    "n2": "-"},
        {"type": "Capacitor",  "name": "C4",         "val": config["c4"],      "n1": "S3",      "n2": "OUT"},
        {"type": "Resistor",   "name": "VOL_POT",    "val": config.get("r_vol", 500.0e3),  "n1": "OUT",     "n2": "-"},
    ]
    sim = Circuit(v_dd_ideal=18.0, r_psu=10.0)
    process_unified_circuit(sim, core_list)
    cal_freq = config.get("cal_freq", _CAL_FREQ)
    monitor  = config.get("monitor_nodes", _MONITOR_NODES)
    analyzer = CircuitAnalyzer(circuit=sim, monitor_nodes=monitor,
                               input_node="v_ideal", amplitude=_INPUT_AMPLITUDE)
    if not full_run:
        spp = 2048 if high_res else 512
        analyzer.circuit.solve_dc_bias(input_node="v_ideal")
        # 80-Period synchronized settlement
        analyzer.circuit.solve_transient(
            input_node="v_ideal", monitor_nodes=["OUT"],
            freqs=cal_freq, periods=80.0, samples_per_period=spp)
        # 20-Period precision measurement with Powell Hybrid Solver
        analyzer.t, analyzer.v_in, analyzer.v_out_data = \
            analyzer.circuit.solve_transient(
                input_node="v_ideal", monitor_nodes=["v_ideal", "OUT"],
                freqs=cal_freq, periods=20.0, samples_per_period=spp,
                use_saved_state=True)
        return analyzer
    return analyzer


def _compute_r7_mid(cap, c3, r_no_r7, tau_floor, tau_ceiling, rth_g3_offset=0.0):
    """
    Deterministic R7 from the midpoint of the tau window.

    The Thevenin R at D2↔G3 = R7 + rth_g3_offset (Q2 drain-side impedance).
    The blocking tau at G3 = (R7 + rth_g3_offset) × (C3 + C3_shunt).
    So: R7 = tau_mid / c_total - rth_g3_offset.
    """
    c_tot   = c3 + cap
    tau_mid = (tau_floor + tau_ceiling) / 2.0
    r_th_target = tau_mid / c_tot
    r7 = r_th_target - rth_g3_offset
    if r7 <= 0:
        return 1.0   # R7 can't be negative; offset alone exceeds target
    if r7 >= r_no_r7:
        return 1e9
    return r7


def _eval_cap_worker(args):
    """
    Flat worker for ProcessPoolExecutor.
    Evaluates one (config, cap, R7) triple at 512 SPP (fast mode).
    Also validates blocking bounds on all coupling caps (C1, C2, C3, C4)
    in the actual circuit with this specific R7.

    Returns a result dict with VPA, error, and blocking_ok flag.
    """
    config, cap_value, r7_value, mode = args
    cal_freq   = config.get("cal_freq", _CAL_FREQ)
    target_vpa = config.get("target_vpa", _TARGET_VPA)
    tau_floor  = config["tau_floor"]
    tau_ceiling = config["tau_ceiling"]

    analyzer = _eval_circuit_from_config(config, cap_value, r7_value,
                                          full_run=False, high_res=False)
    vpa = get_vpa_metric(analyzer.v_out_data["OUT"], analyzer.circuit.dt, cal_freq)

    # --- Blocking validation on all coupling caps ---
    blocking_ok = True
    blocking_detail = {}

    # C1, C2, C4: standalone both-bound check
    for cname, n1, n2 in [("C1", "IN", "G1"), ("C2", "D1", "G2"),
                           ("C4", "S3", "OUT")]:
        rth = analyzer.circuit.solve_ac_thevenin(Capacitor(f"BV_{cname}", 1e-9, n1, n2))
        cap_obj = next((c for c in analyzer.circuit.capacitors if c.name == cname), None)
        if cap_obj is None:
            continue
        tau = rth * cap_obj.value
        fc  = 1.0 / (2.0 * np.pi * tau) if tau > 0 else float('inf')
        t5  = 5.0 * tau
        ok  = (tau_floor <= tau <= tau_ceiling)
        blocking_detail[cname] = {"rth": rth, "tau": tau, "fc": fc, "t5": t5, "ok": ok}
        if not ok:
            blocking_ok = False

    # G3 node: total = C3 + C3_shunt, both-bound check on combined tau
    rth_g3 = analyzer.circuit.solve_ac_thevenin(Capacitor("BV_G3", 1e-9, "D2", "G3"))
    c3_obj    = next((c for c in analyzer.circuit.capacitors if c.name == "C3"), None)
    shunt_obj = next((c for c in analyzer.circuit.capacitors if c.name == "C3_shunt"), None)
    if c3_obj and shunt_obj:
        c_total_g3 = c3_obj.value + shunt_obj.value
        tau_g3 = rth_g3 * c_total_g3
        fc_g3  = 1.0 / (2.0 * np.pi * tau_g3) if tau_g3 > 0 else float('inf')
        t5_g3  = 5.0 * tau_g3
        ok_g3  = (tau_floor <= tau_g3 <= tau_ceiling)
        blocking_detail["G3_total"] = {
            "rth": rth_g3, "tau": tau_g3, "fc": fc_g3, "t5": t5_g3, "ok": ok_g3,
            "c3": c3_obj.value, "c3_shunt": shunt_obj.value, "c_total": c_total_g3,
        }
        if not ok_g3:
            blocking_ok = False

    return {
        "cap": cap_value, "vpa": vpa, "r7": r7_value, "mode": mode,
        "vpa_error": abs(vpa - target_vpa),
        "blocking_ok": blocking_ok,
        "blocking_detail": blocking_detail,
    }


def _resolve_global_resistors(idss=0.0055, vp_abs=2.0, vdd=18.0,
                               target_fc_hz=_BLOCKING_FC_HZ,
                               target_recovery_ms=_BLOCKING_T5_MS):
    """
    Compute ALL global resistors from LSK489 datasheet + blocking constraints.

    Solved (same all modes):
        R_g   : Gate bias — largest E24 where C1/C2 have E24 caps in blocking window,
                subject to IGSS offset ceiling (20 MΩ).
        Rs1/Rs2 : Source resistors from Shockley alpha.
        Rs_CS : Q4 JFET current source — headroom-centered, self-bias solver.

    Fixed (same all modes):
        R_vol : 500 kΩ volume pot (hardware constraint — not optimized).

    Per-mode:
        Rd1/Rd2 : Drain resistors from knee margin.
    """
    tau_floor   = 1.0 / (2.0 * np.pi * target_fc_hz)
    tau_ceiling = (target_recovery_ms / 1000.0) / 5.0

    # E24 resistor series
    e24_r_bases = [1.0, 1.1, 1.2, 1.3, 1.5, 1.6, 1.8, 2.0, 2.2, 2.4,
                   2.7, 3.0, 3.3, 3.6, 3.9, 4.3, 4.7, 5.1, 5.6, 6.2,
                   6.8, 7.5, 8.2, 9.1]
    e24_resistors = np.array(sorted(set(
        b * m for m in [1, 10, 100, 1e3, 1e4, 1e5, 1e6, 1e7]
        for b in e24_r_bases
    )))

    # === R_vol: fixed hardware constraint ===
    r_vol = 500e3   # 500 kΩ volume pot — not optimizable

    # === R_g: IGSS ceiling + blocking window compatibility ===
    #
    # IGSS ≤ 1 nA → R_g_max = 1% |Vp| / IGSS = 20 MΩ.
    # R_g dominates R_th at G1 (R_th ≈ R_g + R_pickup_DCR) and G2 (R_th ≈ R_g).
    # Iterate E24 resistors from R_g_max downward until both C1 and C2
    # have an E24 cap fitting [tau_floor/R_th, tau_ceiling/R_th].
    igss_max = 1e-9
    v_offset_max = 0.01 * vp_abs
    r_g_max = v_offset_max / igss_max
    r_pickup_dcr = 8000.0

    r_g_calc = None
    r_g_candidates = e24_resistors[(e24_resistors >= 100e3) & (e24_resistors <= r_g_max)]
    r_g_candidates = np.sort(r_g_candidates)[::-1]  # largest first
    for rg in r_g_candidates:
        rth_c1_est = float(rg) + r_pickup_dcr
        rth_c2_est = float(rg)
        c1_min, c1_max = tau_floor / rth_c1_est, tau_ceiling / rth_c1_est
        c2_min, c2_max = tau_floor / rth_c2_est, tau_ceiling / rth_c2_est
        eps = 1e-15
        has_c1 = bool(np.any((_E24_CAPS >= c1_min - eps) & (_E24_CAPS <= c1_max + eps)))
        has_c2 = bool(np.any((_E24_CAPS >= c2_min - eps) & (_E24_CAPS <= c2_max + eps)))
        if has_c1 and has_c2:
            r_g_calc = float(rg)
            break
    if r_g_calc is None:
        r_g_calc = float(r_g_candidates[-1])  # smallest candidate as fallback
        print("  WARNING: No R_g found with E24 C1/C2 in blocking window. "
              f"Using {r_g_calc/1e3:.1f} kΩ", flush=True)

    v_offset = igss_max * r_g_calc
    print(f"[GLOBAL] R_g: {r_g_calc/1e6:.2f} MΩ (IGSS≤{igss_max*1e9:.0f}nA → "
          f"V_offset={v_offset*1e6:.1f}µV | blocking-compatible)", flush=True)
    print(f"[GLOBAL] R_vol: {r_vol/1e3:.0f} kΩ (fixed hardware)", flush=True)

    # === Rs1/Rs2 from Shockley alpha (Clean mode — most linear) ===
    a1_global = 0.30   # Q1: Id~1.65mA, gfs~4mS
    a2_global = 0.25   # Q2: Id~1.375mA

    vs1, rs1, id1 = calc_self_bias(a1_global, idss, vp_abs)
    vs2, rs2, id2 = calc_self_bias(a2_global, idss, vp_abs)

    # === Per-mode Rd from knee margin ===
    #
    # Clipping architecture (Horowitz §3.2.3 informed):
    #   Clean: Q1 clean (large km1), Q2 clean (large km2) — pure linear.
    #          km values DERIVED from three simultaneous constraints:
    #            1. C3 blocking: R_th_C3(Clean) must create a non-empty tau
    #               intersection with R_th_C3(OD modes).
    #            2. Q2 headroom: cascade gain × input peak < D2 swing.
    #            3. Q3 headroom: output Vpp < 2.6V (source follower limit).
    #   OD1:   Q1 clean (same km1 as Clean), Q2 clips (km2=0.5) — single-stage
    #          distortion. Only Rd2 switches from Clean → OD1 (one pole).
    #   OD2:   Q1 clips (km1=0.5), Q2 clips (km2=0.5) — dual-stage distortion.

    # --- OD modes: fixed by design intent (at clipping knee) ---
    km_od = {
        "OD2": (0.5, 0.5),   # both at soft-clip knee
    }

    # --- Clean mode: derive km from constraints ---
    # gm estimates for gain calculation (Shockley: gm = 2*IDSS*sqrt(alpha)/|Vp|)
    gm1 = 2.0 * idss * np.sqrt(a1_global) / vp_abs
    gm2 = 2.0 * idss * np.sqrt(a2_global) / vp_abs
    re1 = 1.0 / gm1   # intrinsic source impedance Q1
    re2 = 1.0 / gm2   # intrinsic source impedance Q2

    # Input peak: multi-tone constructive interference at G1
    # 9 tones at amplitude/9 each, worst-case all align ≈ amplitude
    input_peak = _INPUT_AMPLITUDE

    # Q3 headroom: output peak must be < half of Q3's total swing
    q3_half_swing = (vp_abs + 0.6) / 2.0  # ±1.3V for |Vp|=2.0

    # Sweep km1, km2 for Clean: find maximum headroom within constraints.
    # Scan from high km (most headroom) downward. Both km must be > |Vp|
    # (otherwise Vd < 2*|Vp| and the JFET can't saturate properly).
    best_clean_km = None
    best_clean_score = -1.0

    for km1_test in np.arange(vp_abs + 0.5, vdd - vp_abs - 1.0, 0.25):
        rd1_test = (vdd - vp_abs - km1_test) / id1
        if rd1_test <= 0:
            continue
        gain1 = rd1_test / (rs1 + re1)

        for km2_test in np.arange(vp_abs + 0.5, vdd - vp_abs - 1.0, 0.25):
            rd2_test = (vdd - vp_abs - km2_test) / id2
            if rd2_test <= 0:
                continue
            gain2 = rd2_test / (rs2 + re2)

            # Constraint 0: Minimum cascade gain — D2 peak must be large
            # enough that the output is at least louder than the raw input.
            # Input VPA ≈ 0.175 V_w (estimated from A-weighted multi-tone at
            # _INPUT_AMPLITUDE). Use this as the absolute floor.
            cascade_gain = gain1 * gain2
            min_d2_peak = _INPUT_VPA_EST * 3.0 / 0.9
            if (input_peak * cascade_gain) < min_d2_peak:
                continue

            # Constraint 1: Q2 headroom — D2 peak within rails
            vd2_q = vp_abs + km2_test  # quiescent drain voltage
            d1_peak = input_peak * gain1
            d2_peak = d1_peak * gain2
            headroom_d2 = min(vd2_q, vdd - vd2_q)
            if d2_peak > headroom_d2 * 0.99:  # 1% cap tolerance margin
                continue

            # Constraint 2: Q3 headroom — VPA target × crest factor < swing
            # Crest factor for uncompressed multi-tone: empirically ~3× from
            # simulation (verified in Phase C with actual waveform data).
            # This is a pre-filter; the real Q3 check runs after Phase C.
            q3_est_peak = _TARGET_VPA * 3.0
            if q3_est_peak > (2.0 * q3_half_swing) * 0.99:  # 1% cap tolerance margin
                continue

            # Score: maximize combined D2 + Q3 headroom margin
            margin_d2 = headroom_d2 - d2_peak
            margin_q3 = (2.0 * q3_half_swing) - q3_est_peak
            score = margin_d2 + margin_q3

            if score > best_clean_score:
                best_clean_score = score
                best_clean_km = (km1_test, km2_test)

    if best_clean_km is None:
        # Fallback: conservative values
        best_clean_km = (vdd / 2.0 - vp_abs, vdd / 2.0 - vp_abs)
        print("  WARNING: Clean km solver found no valid pair. Using symmetric mid-rail.",
              flush=True)

    km_table = {
        "Clean": best_clean_km,
        "OD1":   (best_clean_km[0], 0.5),  # Q1 same as Clean, Q2 at clipping knee
        "OD2":   km_od["OD2"],
    }

    print(f"[Clean km solver] km1={best_clean_km[0]:.2f} km2={best_clean_km[1]:.2f} | "
          f"cascade gain: {(((vdd-vp_abs-best_clean_km[0])/id1)/(rs1+re1)) * (((vdd-vp_abs-best_clean_km[1])/id2)/(rs2+re2)):.1f}× | "
          f"D2 headroom: {min(vp_abs+best_clean_km[1], vdd-vp_abs-best_clean_km[1]):.2f}V")
    print(f"[OD1] km1={best_clean_km[0]:.2f} (shared with Clean) km2=0.50")

    per_mode_rd = {}
    for mode, (km1, km2) in km_table.items():
        vd1 = vp_abs + km1
        vd2 = vp_abs + km2
        rd1 = (vdd - vd1) / id1
        rd2 = (vdd - vd2) / id2
        per_mode_rd[mode] = (rd1, rd2)

    print(f"[GLOBAL] Q1: alpha={a1_global:.2f} | Rs={rs1:.1f}Ω | "
          f"Id={id1*1000:.3f}mA | Vs={vs1:.3f}V")
    print(f"[GLOBAL] Q2: alpha={a2_global:.2f} | Rs={rs2:.1f}Ω | "
          f"Id={id2*1000:.3f}mA | Vs={vs2:.3f}V")
    for mode, (rd1, rd2) in per_mode_rd.items():
        km1, km2 = km_table[mode]
        print(f"[{mode}] Rd1={rd1:.1f}Ω (km={km1}) | Rd2={rd2:.1f}Ω (km={km2})")

    # === Q3 Source Follower + Q4 JFET Current Source ===
    #
    # Q3 = single LSK489 half: IDSS_Q3 = IDSS, |Vp| same.
    # Q4 = other half of same LSK489 package (matched pair).
    # Drain → VDD, Gate → G3 (DC ≈ V_REF = VDD/2 through R7), Source → S3.
    #
    # Q4 JFET current source: Drain → S3, Gate → GND, Source → Rs_CS → GND.
    # Self-bias: Vgs_Q4 = -Id × Rs_CS.  Constant current eliminates
    # load-side asymmetry. Higher Q3 alpha linearizes the follower's
    # own square-law transfer (gm variation shrinks as fraction of gm_Q).
    #
    # Strategy: maximize Q3 alpha (minimize Rs_CS) while ensuring the
    # worst-case signal peak fits within Q3's swing limits.
    #
    # Swing limits:
    #   Negative: Q3 gate forward-biases at Vgs = +0.6V → Vs_min = V_REF − 0.6V
    #   Positive: Q3 cutoff at Vs = V_REF + |Vp_Q3|
    #   Q4 saturation: Vds_Q4 ≥ |Vp_Q4| (V(S3) - V(S_CS) > |Vp|)
    v_ref   = vdd / 2.0
    idss_q3 = idss              # single LSK489 half
    vp_q3   = vp_abs
    idss_q4 = idss              # other half of same package (matched)
    vp_q4   = vp_abs

    vs_min_swing = v_ref - 0.6
    vs_max_swing = v_ref + vp_q3

    # Signal headroom requirement: VPA_target × crest_factor / 2 + margin
    # Crest factor 3× for multi-tone, 1% tolerance margin
    signal_half_peak = _TARGET_VPA * 3.0 / 2.0
    headroom_margin = signal_half_peak * 0.01  # 1% tolerance
    min_headroom = signal_half_peak + headroom_margin

    # Step 1: Find highest alpha where BOTH headrooms exceed min_headroom
    # and Q4 stays in saturation.
    # Sweep from high alpha (most linear) downward.
    best_alpha_q3 = None
    for a_test in np.linspace(0.99, 0.05, 1000):
        vs_test = v_ref + vp_q3 * (1.0 - np.sqrt(a_test))
        hr_neg = vs_test - vs_min_swing
        hr_pos = vs_max_swing - vs_test
        # Q4 Vds check: V(S3) - V(S_CS) > |Vp_Q4|
        # V(S_CS) = Id × Rs_CS.  At max alpha (Rs_CS→0), V(S_CS)→0, so
        # Vds_Q4 ≈ Vs_Q3.  Conservative: check vs_test > vp_q4.
        if hr_neg >= min_headroom and hr_pos >= min_headroom and vs_test > vp_q4:
            best_alpha_q3 = a_test
            break

    if best_alpha_q3 is None:
        # Fallback: centered headroom (original approach)
        vs_target = (vs_min_swing + vs_max_swing) / 2.0
        best_vs_err = float('inf')
        best_alpha_q3 = 0.5
        for a_test in np.linspace(0.01, 0.99, 500):
            vs_test = v_ref + vp_q3 * (1.0 - np.sqrt(a_test))
            err = abs(vs_test - vs_target)
            if err < best_vs_err:
                best_vs_err = err
                best_alpha_q3 = a_test
        print("  WARNING: No alpha satisfies headroom constraint. "
              "Falling back to centered headroom.", flush=True)

    id_target = best_alpha_q3 * idss_q3

    # Step 2: Rs_CS from Q4 self-bias to deliver target current
    # Id = IDSS_Q4 × (1 − Id·Rs_CS / |Vp_Q4|)²
    # → Rs_CS = |Vp_Q4| × (1 − √(Id/IDSS_Q4)) / Id
    alpha_q4 = id_target / idss_q4
    if alpha_q4 >= 1.0:
        # Target exceeds Q4 IDSS — use Vgs=0 (Rs_CS=0), Id=IDSS_Q4
        rs_cs_solved = 0.0
        print(f"  WARNING: Q4 current source target ({id_target*1000:.2f}mA) "
              f"exceeds IDSS_Q4 ({idss_q4*1000:.1f}mA). Using Rs_CS=0.", flush=True)
    else:
        rs_cs_solved = vp_q4 * (1.0 - np.sqrt(alpha_q4)) / id_target

    # Round to nearest E24
    valid_rs = e24_resistors[e24_resistors > 0]
    if rs_cs_solved > 0:
        rs_cs = float(valid_rs[np.argmin(np.abs(valid_rs - rs_cs_solved))])
    else:
        rs_cs = 0.0   # jumper wire

    # Step 3: Exact Id with rounded Rs_CS (quadratic solve on Q4)
    #   Id = IDSS_Q4 × (1 − Id·Rs_CS / |Vp_Q4|)²
    #   Let u = Id·Rs_CS (voltage across Rs_CS).
    #   u/Rs_CS = IDSS_Q4 × (1 − u/|Vp_Q4|)²
    #   A·u² − (2A·P + 1)·u + A·P² = 0  where A = IDSS_Q4·Rs_CS/P², P = |Vp_Q4|
    if rs_cs > 0:
        P4 = vp_q4
        A4 = idss_q4 * rs_cs / (P4**2)
        qa4 = A4
        qb4 = -(2.0 * A4 * P4 + 1.0)
        qc4 = A4 * P4**2
        disc4 = qb4**2 - 4.0 * qa4 * qc4
        if disc4 >= 0:
            u1 = (-qb4 + np.sqrt(disc4)) / (2.0 * qa4)
            u2 = (-qb4 - np.sqrt(disc4)) / (2.0 * qa4)
            # Physical: 0 ≤ u < |Vp_Q4| (JFET in conduction, Vgs > Vp)
            phys4 = [u for u in [u1, u2] if 0 <= u < P4]
            if phys4:
                u_best = min(phys4, key=lambda u: abs(u / rs_cs - id_target))
                id_q4_actual = u_best / rs_cs
            else:
                id_q4_actual = id_target
        else:
            id_q4_actual = id_target
    else:
        id_q4_actual = idss_q4  # Rs_CS=0 → Vgs=0 → Id=IDSS

    # Q3 operating point with actual Q4 current
    id_q3_actual = id_q4_actual  # series: Q3 drain current = Q4 drain current
    vs_q3_actual = v_ref + vp_q3 * (1.0 - np.sqrt(id_q3_actual / idss_q3))
    vgs_q3_actual = v_ref - vs_q3_actual
    headroom_neg = vs_q3_actual - vs_min_swing
    headroom_pos = vs_max_swing - vs_q3_actual

    # Q4 headroom: needs Vds_Q4 ≥ |Vp_Q4| for saturation
    vs_q4_actual = id_q4_actual * rs_cs if rs_cs > 0 else 0.0
    vds_q4_quiescent = vs_q3_actual - vs_q4_actual
    q4_sat_ok = vds_q4_quiescent > vp_q4

    print(f"[GLOBAL] Q3 (single half): IDSS={idss_q3*1000:.1f}mA | |Vp|={vp_q3:.1f}V")
    print(f"[GLOBAL] Q4 current source (other half, matched): IDSS={idss_q4*1000:.1f}mA | "
          f"|Vp|={vp_q4:.1f}V")
    print(f"[GLOBAL] Rs_CS: {rs_cs:.1f}Ω (E24) | Id={id_q4_actual*1000:.2f}mA | "
          f"Vs_Q4={vs_q4_actual:.3f}V | Vds_Q4={vds_q4_quiescent:.2f}V "
          f"{'✓' if q4_sat_ok else '✗ NOT IN SATURATION'}")
    print(f"[GLOBAL] Q3: alpha={id_q3_actual/idss_q3:.3f} | "
          f"Vs={vs_q3_actual:.3f}V | Vgs={vgs_q3_actual:.3f}V | "
          f"Vds={vdd - vs_q3_actual:.3f}V")
    print(f"[GLOBAL] Q3 swing: [{vs_min_swing:.2f}V, {vs_max_swing:.2f}V] | "
          f"headroom: +{headroom_pos:.2f}V / -{headroom_neg:.2f}V "
          f"(min required: {min_headroom:.2f}V)")

    # === C_REF: V_REF bypass capacitor (not in signal path, no blocking) ===
    c_ref = 47e-6   # 47 µF electrolytic — stabilizes V_REF (VDD/2) node
    print(f"[GLOBAL] C_REF: {c_ref*1e6:.0f} µF (V_REF bypass)")

    return {
        "rs1": rs1, "rs2": rs2, "id1": id1, "id2": id2,
        "vs1": vs1, "vs2": vs2,
        "a1": a1_global, "a2": a2_global,
        "r_g_calc": r_g_calc,
        "r_vol": r_vol,
        "rs_cs": rs_cs,
        "c_ref": c_ref,
        "idss_t": idss, "vp_t": -vp_abs,
        "idss_q3": idss_q3, "vp_q3": -vp_q3,
        "idss_q4": idss_q4, "vp_q4": -vp_q4,
        "vs_q3": vs_q3_actual,
        "headroom_pos": headroom_pos, "headroom_neg": headroom_neg,
        "tau_floor": tau_floor, "tau_ceiling": tau_ceiling,
        "per_mode_rd": per_mode_rd,
    }


def _probe_mode_rth(args):
    """
    Build the full circuit with global Rs and per-mode Rd, then probe
    R_th at all four coupling cap positions and R_no_R7.

    Designed for ProcessPoolExecutor dispatch.
    Args tuple: (mode, resistors) or (mode, resistors, r7_override)
    Returns dict: {mode, rth_c1..c4, r_no_r7}
    """
    if len(args) == 3:
        mode, resistors, r7_override = args
    else:
        mode, resistors = args
        r7_override = None

    rs1      = resistors["rs1"]
    rs2      = resistors["rs2"]
    r_g_calc = resistors["r_g_calc"]
    idss_t   = resistors["idss_t"]
    vp_t     = resistors["vp_t"]
    rd1, rd2 = resistors["per_mode_rd"][mode]

    # R7: use override from SCF feedback, or estimate from tau midpoint
    if r7_override is not None:
        r7_est = r7_override
    else:
        tau_floor   = 1.0 / (2.0 * np.pi * _BLOCKING_FC_HZ)
        tau_ceiling = (_BLOCKING_T5_MS / 1000.0) / 5.0
        tau_mid = (tau_floor + tau_ceiling) / 2.0
        r7_est  = max(1000.0, tau_mid / 100e-9)
    c_ph    = 100e-9  # placeholder cap

    core = [
        {"type": "Inductor",  "name": "L_PICKUP",  "val": 4.5,       "n1": "v_ideal", "n2": "IN", "r_dc": 8000.0, "c_p": 150e-12},
        {"type": "Capacitor", "name": "C1",         "val": c_ph,      "n1": "IN",      "n2": "G1"},
        {"type": "Resistor",  "name": "R1",         "val": r_g_calc,  "n1": "G1",      "n2": "-"},
        {"type": "JFET",      "name": "Q1",         "idss": idss_t, "vp": vp_t, "nd": "D1", "ng": "G1", "ns": "S1"},
        {"type": "Resistor",  "name": "R3",         "val": rd1,      "n1": "+",       "n2": "D1"},
        {"type": "Resistor",  "name": "R2",         "val": rs1,      "n1": "S1",      "n2": "-"},
        {"type": "Capacitor", "name": "C2",         "val": c_ph,      "n1": "D1",      "n2": "G2"},
        {"type": "Resistor",  "name": "R4",         "val": r_g_calc,  "n1": "G2",      "n2": "-"},
        {"type": "JFET",      "name": "Q2",         "idss": idss_t, "vp": vp_t, "nd": "D2", "ng": "G2", "ns": "S2"},
        {"type": "Resistor",  "name": "R6",         "val": rd2,      "n1": "+",       "n2": "D2"},
        {"type": "Resistor",  "name": "R5",         "val": rs2,      "n1": "S2",      "n2": "-"},
        {"type": "Capacitor", "name": "C_REF",      "val": resistors.get("c_ref", 47e-6),    "n1": "V_REF",   "n2": "-"},
        {"type": "Capacitor", "name": "C3",         "val": c_ph,      "n1": "D2",      "n2": "G3"},
        {"type": "Capacitor", "name": "C3_shunt",   "val": c_ph,      "n1": "G3",      "n2": "V_REF"},
        {"type": "Resistor",  "name": "R7",         "val": r7_est,    "n1": "G3",      "n2": "V_REF"},
        {"type": "JFET",      "name": "Q3",         "idss": resistors.get("idss_q3", 0.0055), "vp": resistors.get("vp_q3", -2.0), "nd": "+", "ng": "G3", "ns": "S3"},
        {"type": "JFET",      "name": "Q4",         "idss": resistors.get("idss_q4", 0.0055), "vp": resistors.get("vp_q4", -2.0), "nd": "S3", "ng": "-", "ns": "S_CS"},
        {"type": "Resistor",  "name": "Rs_CS",      "val": resistors.get("rs_cs", 100.0),  "n1": "S_CS",    "n2": "-"},
        {"type": "Capacitor", "name": "C4",         "val": c_ph,      "n1": "S3",      "n2": "OUT"},
        {"type": "Resistor",  "name": "VOL_POT",    "val": resistors.get("r_vol", 500.0e3),   "n1": "OUT",     "n2": "-"},
    ]
    sim = Circuit(v_dd_ideal=18.0, r_psu=10.0)
    process_unified_circuit(sim, core)
    sim.solve_dc_bias(input_node="v_ideal")

    rth_c1 = sim.solve_ac_thevenin(Capacitor("P_C1", 1e-9, "IN",  "G1"))
    rth_c2 = sim.solve_ac_thevenin(Capacitor("P_C2", 1e-9, "D1",  "G2"))
    rth_c3 = sim.solve_ac_thevenin(Capacitor("P_C3", 1e-9, "D2",  "G3"))
    rth_c4 = sim.solve_ac_thevenin(Capacitor("P_C4", 1e-9, "S3",  "OUT"))

    # R_no_R7: same circuit but with R7 → ∞
    core_open = [dict(c) for c in core]
    for c in core_open:
        if c.get("name") == "R7":
            c["val"] = 1e12
    sim_open = Circuit(v_dd_ideal=18.0, r_psu=10.0)
    process_unified_circuit(sim_open, core_open)
    sim_open.solve_dc_bias(input_node="v_ideal")
    r_no_r7 = sim_open.solve_ac_thevenin(Capacitor("T1", 1e-9, "D2", "G3"))

    print(f"[{mode}] R_th → C1:{rth_c1/1000:.1f}k C2:{rth_c2/1000:.1f}k "
          f"C3:{rth_c3/1000:.1f}k C4:{rth_c4/1000:.1f}k | "
          f"R_no_R7:{r_no_r7/1000:.1f}k", flush=True)

    # R_th at D2↔G3 = R_D2_side + R7.  Extract R_D2_side so _compute_r7_mid
    # can set R7 correctly:  R7 = tau_target / c_total - rth_g3_offset
    rth_g3_offset = rth_c3 - r7_est   # Q2 drain-side impedance contribution
    print(f"[{mode}] G3 offset: R_th_C3({rth_c3/1000:.1f}k) - R7_probe({r7_est/1000:.1f}k) "
          f"= {rth_g3_offset/1000:.1f}k (Rd2-side)", flush=True)

    return {
        "mode": mode,
        "rth_c1": rth_c1, "rth_c2": rth_c2, "rth_c3": rth_c3, "rth_c4": rth_c4,
        "r_no_r7": r_no_r7,
        "rth_g3_offset": rth_g3_offset,
    }


def _reconcile_global_caps(mode_rth_list, modes, tau_floor, tau_ceiling,
                           sec_fc_floor=_BLOCKING_FC_FLOOR_HZ,
                           sec_t5_floor=_BLOCKING_T5_FLOOR_MS / 1000.0):
    """
    For each coupling cap, find the single GLOBAL E24 value that satisfies
    the blocking window across ALL modes' R_th values.

    Dual-threshold targeting:
      Primary bounds (absolute, never violated):
        tau_floor ≤ R_th × C ≤ tau_ceiling
      Secondary floors (always active, bias selection within primary window):
        fc  ≥ sec_fc_floor  (default _BLOCKING_FC_FLOOR_HZ)
        5τ  ≥ sec_t5_floor  (default _BLOCKING_T5_FLOOR_MS)

      Within the primary window, candidates are scored by their minimum
      margin above both secondary floors.  This pushes selection away from
      whichever secondary floor has the least headroom.

    C1, C2 — standard both-bound intersection, single E24.
    C4 — both-bound intersection, combo (base+delta) allowed.
    C3 — per-mode (not handled here; solved independently per mode).
    """
    rth_by_mode = {r["mode"]: r for r in mode_rth_list}
    caps = {}

    # Secondary floor scoring: within the primary window, bias selection
    # toward candidates with maximum margin above both secondary floors.
    def _sec_score(c_val, rth_values):
        """Minimum margin above both secondary floors across all modes.
        Higher = better. Negative means a floor is violated."""
        min_margin = float('inf')
        for rth in rth_values:
            tau = rth * c_val
            fc = 1.0 / (2.0 * np.pi * tau) if tau > 0 else float('inf')
            t5 = 5.0 * tau
            min_margin = min(min_margin, fc - sec_fc_floor, t5 - sec_t5_floor)
        return min_margin

    def _pick_best(candidates, rth_values, c_mid):
        """From candidates in primary window, pick the one with best
        secondary margin. Ties broken by proximity to window center."""
        if len(candidates) == 0:
            return None
        scored = [(float(c), _sec_score(float(c), rth_values), abs(float(c) - c_mid))
                  for c in candidates]
        scored.sort(key=lambda x: (-x[1], x[2]))
        return scored[0][0]

    # --- C1, C2: both-bound intersection, single E24 only ---
    for cname, rth_key in [("C1","rth_c1"), ("C2","rth_c2")]:
        rth_values = [rth_by_mode[m][rth_key] for m in modes]
        c_min = max(tau_floor   / rth for rth in rth_values)
        c_max = min(tau_ceiling / rth for rth in rth_values)
        c_mid = (c_min + c_max) / 2.0
        rth_strs = ", ".join(f"{m}:{rth_by_mode[m][rth_key]/1000:.1f}k" for m in modes)

        if c_max >= c_min:
            eps = c_max * 1e-9
            in_window = _E24_CAPS[(_E24_CAPS >= c_min - eps) & (_E24_CAPS <= c_max + eps)]
            if len(in_window) > 0:
                best = _pick_best(in_window, rth_values, c_mid)
                info = {"value": best, "is_combo": False, "base": best, "delta": None}
            else:
                closest = float(_E24_CAPS[np.argmin(np.abs(_E24_CAPS - c_mid))])
                info = {"value": closest, "is_combo": False, "base": closest, "delta": None}
                print(f"  WARNING [GLOBAL/{cname}]: No single E24 in window "
                      f"[{c_min*1e9:.3f}, {c_max*1e9:.3f}] nF. "
                      f"Using closest: {closest*1e9:.2f} nF", flush=True)
        else:
            closest = float(_E24_CAPS[np.argmin(np.abs(_E24_CAPS - c_min))])
            info = {"value": closest, "is_combo": False, "base": closest, "delta": None}
            print(f"  WARNING [GLOBAL/{cname}]: Empty intersection! "
                  f"c_min={c_min*1e9:.3f} > c_max={c_max*1e9:.3f} nF. "
                  f"R_th: {rth_strs}. Using {closest*1e9:.2f} nF", flush=True)

        print(f"[GLOBAL] {cname}: {info['value']*1e9:.3f} nF [E24] | "
              f"window: [{c_min*1e9:.3f}, {c_max*1e9:.3f}] nF | R_th: {rth_strs}",
              flush=True)
        caps[cname] = info

    # --- C4: both-bound intersection, combo allowed (global, same all modes) ---
    rth_c4_values = [rth_by_mode[m]["rth_c4"] for m in modes]
    c4_min = max(tau_floor   / rth for rth in rth_c4_values)
    c4_max = min(tau_ceiling / rth for rth in rth_c4_values)
    c4_mid = (c4_min + c4_max) / 2.0
    rth_strs = ", ".join(f"{m}:{rth_by_mode[m]['rth_c4']/1000:.1f}k" for m in modes)

    if c4_max >= c4_min:
        eps = c4_max * 1e-9
        in_window = _E24_CAPS[(_E24_CAPS >= c4_min - eps) & (_E24_CAPS <= c4_max + eps)]
        if len(in_window) > 0:
            best = _pick_best(in_window, rth_c4_values, c4_mid)
            info = {"value": best, "is_combo": False, "base": best, "delta": None}
        else:
            # Try base+delta combo — score by secondary targeting
            combos_in_window = []
            for base in _E24_CAPS[_E24_CAPS <= c4_max]:
                for delta in _E24_DELTA_CAPS:
                    total = float(base) + float(delta)
                    if c4_min <= total <= c4_max:
                        combos_in_window.append((float(base), float(delta), total))
            if combos_in_window:
                totals = [t for _, _, t in combos_in_window]
                best_total = _pick_best(totals, rth_c4_values, c4_mid)
                best_combo = next((b, d, t) for b, d, t in combos_in_window
                                  if abs(t - best_total) < 1e-15)
                base, delta, total = best_combo
                info = {"value": total, "is_combo": True, "base": base, "delta": delta}
            else:
                closest = float(_E24_CAPS[np.argmin(np.abs(_E24_CAPS - c4_mid))])
                info = {"value": closest, "is_combo": False, "base": closest, "delta": None}
                print("  WARNING [GLOBAL/C4]: No E24 or combo in window "
                      f"[{c4_min*1e9:.3f}, {c4_max*1e9:.3f}] nF. "
                      f"Using closest: {closest*1e9:.2f} nF", flush=True)
    else:
        closest = float(_E24_CAPS[np.argmin(np.abs(_E24_CAPS - c4_min))])
        info = {"value": closest, "is_combo": False, "base": closest, "delta": None}
        print("  WARNING [GLOBAL/C4]: Empty intersection! "
              f"c_min={c4_min*1e9:.3f} > c_max={c4_max*1e9:.3f} nF. "
              f"R_th: {rth_strs}. Using {closest*1e9:.2f} nF", flush=True)

    tag = "COMBO" if info["is_combo"] else "E24"
    print(f"[GLOBAL] C4: {info['value']*1e9:.3f} nF [{tag}] | "
          f"window: [{c4_min*1e9:.3f}, {c4_max*1e9:.3f}] nF | R_th: {rth_strs}",
          flush=True)
    caps["C4"] = info

    return caps



# --- BOM / Switch-Network Component Table Writer ---
def write_component_tsv(all_mode_components, global_resistors=None, filename="component_bom.tsv"):
    """
    Write a tab-separated component table for the switch network.

    General switch-network design rules (no DC pop on mode change):
      Resistors  : hardwired base = MAX value across all modes.
                   Switched parallel R brings the effective R down to mode target.
                   R_parallel = (R_base * R_target) / (R_base - R_target)
                   DNP if mode target equals base (no parallel needed).

      Capacitors : hardwired base = MIN value across all modes.
                   Switched parallel C adds up to mode target.
                   C_parallel = C_target - C_base
                   DNP if mode target equals base (no parallel needed).

    G3 network (per-mode switched unit):
      Each mode has its own C3 + C3_shunt + R7, switched as one unit.
      No hardwired base — the entire G3 network is per-mode.
    """
    modes = ["Clean", "OD1", "OD2"]

    def fmt_r(ohms):
        if ohms is None: return "DNP"
        if ohms >= 1e6:  return f"{ohms/1e6:.3f} MΩ"
        if ohms >= 1e3:  return f"{ohms/1e3:.3f} kΩ"
        return f"{ohms:.1f} Ω"

    def fmt_c(farads):
        if farads is None: return "DNP"
        if farads >= 1e-3: return f"{farads*1e3:.3f} mF"
        if farads >= 1e-6: return f"{farads*1e6:.3f} µF"
        if farads >= 1e-9: return f"{farads*1e9:.3f} nF"
        return f"{farads*1e12:.3f} pF"

    def parallel_r(r_base, r_target):
        """Parallel R needed to bring r_base down to r_target. None = DNP."""
        if r_base <= r_target + 1.0:
            return None
        return (r_base * r_target) / (r_base - r_target)

    def parallel_c(c_base, c_target):
        """Parallel C needed to bring c_base up to c_target. None = DNP."""
        delta = c_target - c_base
        if delta < 1e-15:
            return None
        return delta

    # ================================================================
    # STANDARD COMPONENT BASE VALUES
    # ================================================================
    rd1_base = max(all_mode_components[m]["rd1"] for m in modes)
    rs1_base = max(all_mode_components[m]["rs1"] for m in modes)
    rd2_base = max(all_mode_components[m]["rd2"] for m in modes)
    rs2_base = max(all_mode_components[m]["rs2"] for m in modes)
    c1_base  = min(all_mode_components[m]["c1"]  for m in modes)
    c2_base  = min(all_mode_components[m]["c2"]  for m in modes)
    c4_base  = min(all_mode_components[m]["c4"]  for m in modes)

    # ================================================================
    # BUILD ROW DATA
    # ================================================================
    # row format:
    #   (section_label, ref_des, description, hardwired_value, mode_dict)
    # mode_dict keys = mode name, values = (total_str, switched_str)

    GLOBAL  = "GLOBAL"
    MODE_R  = "MODE_R"
    MODE_C  = "MODE_C"
    G3_NET  = "G3_NET"   # per-mode G3 network (C3 + C3_shunt + R7)

    r_g_val  = global_resistors["r_g_calc"] if global_resistors else 10e6
    rs_cs_val = global_resistors["rs_cs"] if global_resistors else 100.0
    r_vol_val = global_resistors["r_vol"] if global_resistors else 500e3

    rows = [
        # ---- GLOBAL FIXED COMPONENTS ----
        (GLOBAL, "R1 / R4", "Q1 / Q2 Gate Bias Resistors (LSK489 IGSS-derived, blocking-compatible)",
            fmt_r(r_g_val), {}),
        (GLOBAL, "Q4 + Rs_CS", "Q4 JFET Current Source (gate→GND, source→Rs_CS→GND) — symmetric Q3 output",
            f"Q4: LSK489 single half  |  Rs_CS: {fmt_r(rs_cs_val)}", {}),
        (GLOBAL, "R_VOL",   "Volume Potentiometer (fixed hardware)",
            fmt_r(r_vol_val), {}),
        (GLOBAL, "C_REF",   "V_REF Bypass Capacitor",
            fmt_c(global_resistors["c_ref"] if global_resistors else 47e-6), {}),
        (GLOBAL, "L_PICKUP","Pickup Inductance",
            "4.500 H  (DCR: 8.000 kΩ  Cp: 150 pF)", {}),

        # ---- Q1 DRAIN / SOURCE RESISTORS ----
        (MODE_R, "R3", "Q1 Drain Resistor — hardwired base",
            fmt_r(rd1_base),
            {m: (fmt_r(all_mode_components[m]["rd1"]),
                 fmt_r(parallel_r(rd1_base, all_mode_components[m]["rd1"])))
             for m in modes}),

        (MODE_R, "R2", "Q1 Source Resistor — hardwired base",
            fmt_r(rs1_base),
            {m: (fmt_r(all_mode_components[m]["rs1"]),
                 fmt_r(parallel_r(rs1_base, all_mode_components[m]["rs1"])))
             for m in modes}),

        # ---- Q2 DRAIN / SOURCE RESISTORS ----
        (MODE_R, "R6", "Q2 Drain Resistor — hardwired base",
            fmt_r(rd2_base),
            {m: (fmt_r(all_mode_components[m]["rd2"]),
                 fmt_r(parallel_r(rd2_base, all_mode_components[m]["rd2"])))
             for m in modes}),

        (MODE_R, "R5", "Q2 Source Resistor — hardwired base",
            fmt_r(rs2_base),
            {m: (fmt_r(all_mode_components[m]["rs2"]),
                 fmt_r(parallel_r(rs2_base, all_mode_components[m]["rs2"])))
             for m in modes}),

        # ---- COUPLING CAPS (C1, C2, C4 — global) ----
        (MODE_C, "C1", "Input Coupling Capacitor — hardwired base",
            fmt_c(c1_base),
            {m: (fmt_c(all_mode_components[m]["c1"]),
                 fmt_c(parallel_c(c1_base, all_mode_components[m]["c1"])))
             for m in modes}),

        (MODE_C, "C2", "Interstage Coupling Capacitor — hardwired base",
            fmt_c(c2_base),
            {m: (fmt_c(all_mode_components[m]["c2"]),
                 fmt_c(parallel_c(c2_base, all_mode_components[m]["c2"])))
             for m in modes}),

        (MODE_C, "C4", "Output Coupling Capacitor — hardwired base",
            fmt_c(c4_base),
            {m: (fmt_c(all_mode_components[m]["c4"]),
                 fmt_c(parallel_c(c4_base, all_mode_components[m]["c4"])))
             for m in modes}),

        # ---- G3 NETWORK (per-mode: C3 + C3_shunt + R7 switched as one unit) ----
        (G3_NET, "C3 + C3_shunt + R7",
            "G3 Network — per-mode switched unit (C3 couples D2→G3, C3_shunt+R7 bias G3→V_REF)",
            "NO HARDWIRED BASE",
            {m: (
                f"C3={fmt_c(all_mode_components[m]['c3'])}  |  "
                f"C3_shunt={fmt_c(all_mode_components[m]['c3_shunt'])}  |  "
                f"R7={fmt_r(all_mode_components[m]['r7'])}",
                "[UNIT — switch all three together]"
             ) for m in modes}),
    ]

    # ================================================================
    # WRITE TSV
    # ================================================================
    with open(filename, "w", encoding="utf-8") as f:

        # header
        header = [
            "Section", "Ref Des", "Description", "Hardwired Base Value",
            "Clean — Mode Total",  "Clean — Switched Element(s)",
            "OD1 — Mode Total",    "OD1 — Switched Element(s)",
            "OD2 — Mode Total",    "OD2 — Switched Element(s)",
        ]
        f.write("\t".join(header) + "\n")
        f.write("\t".join(["---"] * len(header)) + "\n")

        prev_section = None
        for (section, ref, desc, base_val, mode_dict) in rows:
            if prev_section is not None and section != prev_section:
                f.write("\n")
            prev_section = section

            cells = [section, ref, desc, base_val]

            if section == GLOBAL:
                for m in modes:
                    cells += ["(global — same all modes)", "N/A"]
            elif section == G3_NET:
                # per-mode G3 network
                for m in modes:
                    total_str, note_str = mode_dict[m]
                    cells += [total_str, note_str]
            else:
                # standard MODE_R / MODE_C rows
                for m in modes:
                    total, parallel = mode_dict[m]
                    cells += [total, parallel]

            f.write("\t".join(cells) + "\n")

        # footer notes
        f.write("\n")
        notes = [
            ["NOTE", "DNP = Do Not Populate — this mode uses the hardwired base; switch stays open."],
            ["NOTE", "Resistors: hardwired base = MAX(all modes). Parallel switched R lowers effective R."],
            ["NOTE", "Capacitors: hardwired base = MIN(all modes). Parallel switched C raises effective C."],
            ["NOTE", "G3 network: C3 + C3_shunt + R7 are per-mode — switch all three as one unit."],
            ["NOTE", "No hardwired base at G3. Each mode has its own complete G3 network."],
        ]
        for n in notes:
            f.write("\t".join(n) + "\n")

    print(f"\n>>> Component BOM written to: {filename}")


# ---- execute_mode_analytics uses config-based eval ----
def execute_mode_analytics(mode, c3_shunt_target, rtot_target, config):
    analyzer_test = _eval_circuit_from_config(config, c3_shunt_target, rtot_target,
                                               full_run=False)
    c3_val = [c.value for c in analyzer_test.circuit.capacitors if c.name == "C3"][0]
    c_total_ac = c3_val + c3_shunt_target

    print(f"\n[{mode}] --- Circuit Parameters ---")
    print(f"Hardware C3_shunt: {c3_shunt_target*1e9:.2f} nF (Standard E24)")
    print(f"Hardware R7 Trimmer: {rtot_target/1000:.2f} kOhms")
    print(f"RC Time Constant (C3 Output Node): {rtot_target * c_total_ac:.4f} s")

    analyzer = _eval_circuit_from_config(config, c3_shunt_target, rtot_target,
                                          full_run=True)
    analyzer.report_dc_bias()
    analyzer.report_ac_analytics()
    analyzer.run_transient()

    # Sync: Use the same physics-driven index for gain/Vpp calculations
    start_idx = int(np.round((10.0 * analyzer.get_max_system_tau()) / analyzer.circuit.dt))
    v_ideal_ss = analyzer.v_out_data["v_ideal"][start_idx:]
    out_ss     = analyzer.v_out_data["OUT"][start_idx:]
    vpp_ref    = np.max(v_ideal_ss) - np.min(v_ideal_ss)
    vpp_out    = np.max(out_ss)     - np.min(out_ss)

    vpa_final = get_vpa_metric(analyzer.v_out_data["OUT"], analyzer.circuit.dt, analyzer.freqs)
    print(f"\n[{mode} MODE] Cascade Acoustic Summary (STEADY-STATE):")
    print(f"  Vpp In: {vpp_ref:.4f} V")
    print(f"  Vpp Post-C4: {vpp_out:.4f} V")
    print(f"  A-Weighted V_pa: {vpa_final:.4f} V_w")

    analyzer.report_single_tone_thd(node="OUT")
    analyzer.plot_waveforms(mode=mode)
    analyzer.export_audio(mode=mode)


if __name__ == "__main__":
    modes = ["Clean", "OD1", "OD2"]
    n_cpus = os.cpu_count() or 4
    tau_floor   = 1.0 / (2.0 * np.pi * _BLOCKING_FC_HZ)
    tau_ceiling = (_BLOCKING_T5_MS / 1000.0) / 5.0

    # ==================================================================
    #  PHASE A — Global component resolution
    # ==================================================================
    print("=" * 72)
    print(f"  PARALLEL COMBINATION-CAP SEARCH — {n_cpus} CPUs detected")
    print(f"  Blocking window: tau ∈ [{tau_floor*1000:.2f}, {tau_ceiling*1000:.2f}] ms")
    print("  Global: C1, C2, C4, Rs1, Rs2  |  Per-mode: C3, Rd1, Rd2, R7, C3_shunt")
    print("=" * 72, flush=True)

    # Step 1: Resolve global source resistors + per-mode drain resistors
    print("\n--- Phase A Step 1: Global Resistor Resolution ---", flush=True)
    resistors = _resolve_global_resistors()

    # ==================================================================
    #  Phase A2-init: Initial Thevenin probes (tau-midpoint R7 guess)
    #  Used ONLY to select per-mode C3 — frozen for all SCF iterations.
    # ==================================================================
    print("\n--- Phase A Step 2-init: Initial Thevenin Probes (C3 selection) ---",
          flush=True)
    init_probe_args = [(m, resistors) for m in modes]
    with ProcessPoolExecutor(max_workers=min(n_cpus, len(modes))) as pool:
        init_rth_list = list(pool.map(_probe_mode_rth, init_probe_args))
    init_rth_by_mode = {r["mode"]: r for r in init_rth_list}

    # --- Per-mode C3 selection (FROZEN — not re-evaluated in SCF loop) ---
    # C3 is part of the switched G3 network (C3 + C3_shunt + R7).
    # Each mode picks the smallest E24 C3 satisfying its own blocking window
    # at R_th_C3, maximizing C3_shunt budget.
    c3_by_mode = {}
    for m in modes:
        rth_c3 = init_rth_by_mode[m]["rth_c3"]
        c3_min = tau_floor / rth_c3
        c3_max = tau_ceiling / rth_c3
        eps = c3_max * 1e-9
        in_window = _E24_CAPS[(_E24_CAPS >= c3_min - eps) & (_E24_CAPS <= c3_max + eps)]
        if len(in_window) > 0:
            c3_val = float(in_window[0])  # smallest → max C3_shunt budget
        else:
            c3_val = float(_E24_CAPS[np.argmin(np.abs(_E24_CAPS - c3_min))])
            print(f"  WARNING [{m}/C3]: No E24 in window "
                  f"[{c3_min*1e9:.2f}, {c3_max*1e9:.2f}] nF. "
                  f"Using closest: {c3_val*1e9:.2f} nF", flush=True)
        c3_by_mode[m] = c3_val
        print(f"[{m}] C3: {c3_val*1e9:.3f} nF [E24] (FROZEN) | "
              f"window: [{c3_min*1e9:.2f}, {c3_max*1e9:.2f}] nF | "
              f"R_th_C3: {rth_c3/1000:.1f}k", flush=True)

    # ==================================================================
    #  SCF LOOP — Iterate Phases A2–C until R7 values converge (< 1%)
    #  C3 is frozen above; only R7 + C3_shunt are iterated.
    # ==================================================================
    _SCF_MAX_ITER = 5
    _SCF_THRESHOLD = 0.01  # 1% convergence
    prev_r7 = {m: None for m in modes}

    for scf_iter in range(1, _SCF_MAX_ITER + 1):
        print(f"\n{'='*72}")
        print(f"  SCF ITERATION {scf_iter} / {_SCF_MAX_ITER}")
        print(f"{'='*72}", flush=True)

        # --- Phase A2: Probe R_th with current R7 estimates ---
        print("\n--- Phase A Step 2: Per-Mode Thevenin Probes ---", flush=True)
        probe_args = [(m, resistors, prev_r7[m]) for m in modes]
        with ProcessPoolExecutor(max_workers=min(n_cpus, len(modes))) as pool:
            mode_rth_list = list(pool.map(_probe_mode_rth, probe_args))
        rth_by_mode = {r["mode"]: r for r in mode_rth_list}

        # --- Phase A3: Reconcile global cap values (C1, C2, C4 only) ---
        print("\n--- Phase A Step 3: Global Cap Reconciliation ---", flush=True)
        global_caps = _reconcile_global_caps(mode_rth_list, modes,
                                              tau_floor, tau_ceiling)

        # --- Phase A4: Config assembly (C3 frozen from init probe) ---
        c1 = global_caps["C1"]["value"]
        c2 = global_caps["C2"]["value"]
        c4 = global_caps["C4"]["value"]

        mode_configs = {}
        for m in modes:
            rd1, rd2 = resistors["per_mode_rd"][m]
            rth = rth_by_mode[m]
            mode_configs[m] = {
                "mode":           m,
                "c1": c1, "c2": c2, "c3": c3_by_mode[m], "c4": c4,
                "rd1": rd1, "rs1": resistors["rs1"],
                "rd2": rd2, "rs2": resistors["rs2"],
                "r_g_calc":       resistors["r_g_calc"],
                "r_vol":          resistors["r_vol"],
                "rs_cs":          resistors["rs_cs"],
                "c_ref":          resistors["c_ref"],
                "idss_t":         resistors["idss_t"],
                "vp_t":           resistors["vp_t"],
                "idss_q3":        resistors["idss_q3"],
                "vp_q3":          resistors["vp_q3"],
                "idss_q4":        resistors["idss_q4"],
                "vp_q4":          resistors["vp_q4"],
                "r_no_r7":        rth["r_no_r7"],
                "rth_g3_offset":  rth["rth_g3_offset"],
                "tau_floor":      tau_floor,
                "tau_ceiling":    tau_ceiling,
                "cal_freq":       _CAL_FREQ,
                "monitor_nodes":  _MONITOR_NODES,
                "target_vpa":     _TARGET_VPA,
            }
            print(f"[{m}] Config ready | Rd1={rd1:.1f}Ω | Rd2={rd2:.1f}Ω | "
                  f"C3={c3_by_mode[m]*1e9:.2f}nF (frozen) | "
                  f"r_no_r7={rth['r_no_r7']/1000:.1f}kΩ", flush=True)

        # --- Phase B: Per-mode VPA scan + fine evaluation ---
        total_min = 1e-9
        total_max = 200e-9
        # VPA gate: blocking is absolute priority. VPA must be louder than
        # raw input, preferably near target. Lower bound = input VPA estimate.
        vpa_gate_lo = _INPUT_VPA_EST
        vpa_gate_hi = _TARGET_VPA * 1.25

        # Step 1: Sparse E24 scan per mode
        e24_totals = [float(c) for c in _E24_CAPS if total_min <= c <= total_max]
        scan_items = []
        for m in modes:
            cfg = mode_configs[m]
            for ct in e24_totals:
                r7 = _compute_r7_mid(ct, cfg["c3"], cfg["r_no_r7"],
                                      cfg["tau_floor"], cfg["tau_ceiling"],
                                      cfg.get("rth_g3_offset", 0.0))
                scan_items.append((cfg, ct, r7, m))

        print("\n--- Phase B Step 1: Sparse VPA Scan ---")
        print(f"  {len(e24_totals)} E24 totals × {len(modes)} modes = "
              f"{len(scan_items)} scan evals", flush=True)

        with ProcessPoolExecutor(max_workers=min(n_cpus, len(scan_items))) as pool:
            scan_results = list(pool.map(_eval_cap_worker, scan_items))

        # Per-mode: find viable total range
        viable_range = {}
        for m in modes:
            mode_vpas = [(scan_items[i][1], scan_results[i]["vpa"])
                         for i in range(len(scan_items))
                         if scan_items[i][3] == m
                         and scan_results[i].get("blocking_ok", True)
                         and vpa_gate_lo <= scan_results[i]["vpa"] <= vpa_gate_hi]
            if mode_vpas:
                totals_ok = [t for t, v in mode_vpas]
                viable_range[m] = (min(totals_ok), max(totals_ok))
                print(f"  [{m}] Viable: {min(totals_ok)*1e9:.1f} – "
                      f"{max(totals_ok)*1e9:.1f} nF "
                      f"({len(totals_ok)} passed)", flush=True)
            else:
                viable_range[m] = (total_min, total_max)
                print(f"  [{m}] WARNING: No E24 total in VPA gate — using full range",
                      flush=True)

        # Per-mode viable ranges with 20% margin each
        for m in modes:
            lo, hi = viable_range[m]
            margin = (hi - lo) * 0.20
            viable_range[m] = (max(total_min, lo - margin), min(total_max, hi + margin))

        # Step 2: Fine enumeration — only totals within ANY mode's viable range,
        # evaluated only for the modes whose range contains that total.
        work_items = []
        seen_keys = set()
        for c_base in _CAP_SEARCH_RANGE:
            for delta in _E24_DELTA_CAPS:
                total = float(c_base) + float(delta)
                for m in modes:
                    lo, hi = viable_range[m]
                    if lo <= total <= hi:
                        tk = _cap_key(total)
                        if (m, tk) not in seen_keys:
                            seen_keys.add((m, tk))
                            cfg = mode_configs[m]
                            r7 = _compute_r7_mid(total, cfg["c3"], cfg["r_no_r7"],
                                                  cfg["tau_floor"], cfg["tau_ceiling"],
                                                  cfg.get("rth_g3_offset", 0.0))
                            work_items.append((cfg, total, r7, m))

        n_items = len(work_items)
        n_workers = min(n_cpus, n_items) if n_items > 0 else 1
        n_unique = len({_cap_key(wi[1]) for wi in work_items})
        range_strs = ", ".join(f"{m}:[{viable_range[m][0]*1e9:.1f},{viable_range[m][1]*1e9:.1f}]"
                               for m in modes)
        print("\n--- Phase B Step 2: Fine Evaluation ---")
        print(f"  Per-mode ranges: {range_strs}")
        print(f"  {n_unique} unique totals × relevant modes = "
              f"{n_items} evals across {n_workers} workers", flush=True)

        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            all_results = list(pool.map(_eval_cap_worker, work_items))

        # Build per-mode lookup: cap_key → result (blocking-valid, in VPA gate)
        mode_results = {m: {} for m in modes}
        n_block = 0
        for r in all_results:
            if not r.get("blocking_ok", True):
                n_block += 1
                continue
            if r["vpa"] < vpa_gate_lo or r["vpa"] > vpa_gate_hi:
                continue
            mode_results[r["mode"]][_cap_key(r["cap"])] = r

        print(f"  Fine eval: {len(all_results)} evals | {n_block} blocked | "
              + ", ".join(f"{m}:{len(mode_results[m])}" for m in modes),
              flush=True)

        # ==================================================================
        #  PHASE C — Cross-mode trio matching (per-mode G3 network)
        # ==================================================================
        #  Each mode has its own C3 + C3_shunt + R7 switched as one unit.
        #  Find the best-matching trio — one C3_shunt per mode, all three
        #  VPAs within the gate and closest to each other.

        print("\n--- Phase C: Cross-Mode Trio Matching ---", flush=True)

        best_trio_score = (float('inf'), float('inf'))
        best_trio = None

        for tk_c in mode_results["Clean"]:
            rc = mode_results["Clean"][tk_c]
            for tk_1 in mode_results["OD1"]:
                r1 = mode_results["OD1"][tk_1]
                for tk_2 in mode_results["OD2"]:
                    r2 = mode_results["OD2"][tk_2]
                    vpas = [rc["vpa"], r1["vpa"], r2["vpa"]]
                    spread = max(vpas) - min(vpas)
                    sum_err = sum(abs(v - _TARGET_VPA) for v in vpas)
                    score = (spread, sum_err)
                    if score < best_trio_score:
                        best_trio_score = score
                        best_trio = {"Clean": rc, "OD1": r1, "OD2": r2}

        if best_trio is None:
            raise RuntimeError("No valid cross-mode trio found. "
                               "Check blocking bounds and VPA target.")

        print(f"  Best trio spread: {best_trio_score[0]*1000:.2f} mV_w | "
              f"sum |err|: {best_trio_score[1]:.4f} V_w")

        # Report
        print("\n" + "=" * 72)
        print("  CROSS-MODE SELECTION — RESULTS (per-mode G3 network)")
        print("=" * 72)
        print(f"  Cross-mode spread: {best_trio_score[0]*1000:.2f} mV_w | "
              f"Sum |err|: {best_trio_score[1]:.4f} V_w")
        print("-" * 72)

        c3_shunt_targets = {}
        rtot_targets = {}
        all_mode_components = {}
        for m in modes:
            r = best_trio[m]
            cfg = mode_configs[m]
            c3_v = cfg["c3"]
            c_total_g3 = c3_v + r["cap"]
            rth_g3 = r["r7"] + cfg["rth_g3_offset"]
            tau_g3 = rth_g3 * c_total_g3
            f_g3 = 1.0 / (2.0 * np.pi * tau_g3) if tau_g3 > 0 else float('inf')
            t5_g3 = 5.0 * tau_g3

            c3_shunt_targets[m] = r["cap"]
            rtot_targets[m] = r["r7"]
            all_mode_components[m] = {
                "rd1": cfg["rd1"], "rs1": resistors["rs1"],
                "rd2": cfg["rd2"], "rs2": resistors["rs2"],
                "c1": c1, "c2": c2, "c3": c3_v, "c4": c4,
                "c3_shunt": r["cap"], "r7": r["r7"],
            }
            print(f"  [{m:>5s}] C3: {c3_v*1e9:.2f} nF | "
                  f"C3_shunt: {r['cap']*1e9:.2f} nF | "
                  f"R7: {r['r7']/1000:7.2f} kΩ | V_pa: {r['vpa']:.4f} V_w | "
                  f"Fc: {f_g3:.1f} Hz | 5τ: {t5_g3*1000:.1f} ms")
        print("=" * 72)

        # Blocking detail
        for m in modes:
            bd = best_trio[m].get("blocking_detail", {})
            if bd:
                print(f"  [{m}] Blocking: " + " | ".join(
                    f"{cn}: fc={d['fc']:.1f}Hz 5τ={d['t5']*1000:.1f}ms "
                    f"{'✓' if d['ok'] else '✗'}"
                    for cn, d in bd.items()))

        # Q3 headroom validation
        print(f"\n  Q3 headroom validation (Q4 current source, Rs_CS={resistors['rs_cs']:.1f}Ω):")
        print(f"  Q3 DC: Vs={resistors['vs_q3']:.3f}V | "
              f"headroom +{resistors['headroom_pos']:.2f}V / "
              f"-{resistors['headroom_neg']:.2f}V")
        for m in modes:
            vpa = best_trio[m]["vpa"]
            # Crest factor ~3× for uncompressed multi-tone (empirical).
            # Phase D full transient provides the definitive check.
            vpp_est = vpa * 3.0
            margin_pos = resistors["headroom_pos"] - vpp_est / 2.0
            margin_neg = resistors["headroom_neg"] - vpp_est / 2.0
            ok = margin_pos > 0 and margin_neg > 0
            print(f"  [{m}] V_pa={vpa:.4f} → est Vpp≈{vpp_est:.2f}V | "
                  f"margin: +{margin_pos:.2f}V / -{margin_neg:.2f}V "
                  f"{'✓' if ok else '✗ CLIP RISK'}")
        print()

        # ==================================================================

        # --- SCF Convergence Check ---
        current_r7 = {m: rtot_targets[m] for m in modes}
        if all(prev_r7[m] is not None for m in modes):
            max_change = max(
                abs(current_r7[m] - prev_r7[m]) / prev_r7[m]
                for m in modes)
            print(f"  SCF R7 max change: {max_change*100:.3f}%", flush=True)
            if max_change < _SCF_THRESHOLD:
                print(f"  SCF CONVERGED in {scf_iter} iteration(s)", flush=True)
                break
        else:
            print("  SCF: first iteration (no prior R7)", flush=True)
        prev_r7 = current_r7
    else:
        print(f"  WARNING: SCF did NOT converge in {_SCF_MAX_ITER} iterations",
              flush=True)

    #  PHASE D — Full analytics (sequential)
    # ==================================================================
    for m in modes:
        execute_mode_analytics(m, c3_shunt_targets[m], rtot_targets[m],
                               mode_configs[m])

    write_component_tsv(all_mode_components, global_resistors=resistors,
                        filename="component_bom.tsv")