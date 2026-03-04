# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 14:30:16 2026

@author: bkafin
"""

import numpy as np
from scipy.optimize import root
import matplotlib.pyplot as plt
import schemdraw
import schemdraw.elements as elm

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
    def __init__(self, name, idss, vp, node_d, node_g, node_s, ambient_temp=25.0):
        self.name = name
        self.idss = float(idss)
        self.vp = float(vp)
        self.node_d = node_d
        self.node_g = node_g
        self.node_s = node_s
        self.ambient_temp = ambient_temp
        self.cgs, self.cgd = 2.0e-12, 2.0e-12
        self.theta_ja, self.c_th = 416.67, 0.2
        self.lambda_mod = 0.0073
        self.t_j = ambient_temp 
        self.current_cgs = self.cgs
        self.current_cgd = self.cgd

class Circuit:
    def __init__(self, v_dd_ideal=18.0, r_psu=100.0):
        self.v_dd_ideal, self.r_psu = v_dd_ideal, r_psu
        self.resistors, self.capacitors, self.inductors, self.jfets = [], [], [], []
        self.nodes = ["-", "+"]
        self.node_map, self.dc_op = {} , {}

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
        return [n for n in self.nodes if n not in ["-"]]

    def _jfet_physics(self, j, v_gs, v_ds, t_j_state):
        t_j_safe = np.clip(t_j_state, -55.0, 150.0)
        vp_t = j.vp - 0.0022 * (t_j_safe - 25.0)
        idss_t = j.idss * ((t_j_safe + 273.15) / 298.15)**-1.5
        is_rev = v_ds < 0.0
        v_ds_eff = -v_ds if is_rev else v_ds
        v_gs_eff = v_gs - v_ds if is_rev else v_gs
        early = (1.0 + j.lambda_mod * v_ds_eff)
        beta = idss_t / (vp_t**2)
        v_gst = v_gs_eff - vp_t
        if v_gs_eff <= vp_t:
            i_chan = 1e-9 * np.exp((v_gs_eff - vp_t) / 0.1)
            gm = (1e-9 / 0.1) * np.exp((v_gs_eff - vp_t) / 0.1)
        elif v_ds_eff < v_gst:
            i_chan = beta * (2 * v_gst * v_ds_eff - v_ds_eff**2) * early
            gm = 2 * beta * v_ds_eff * early
        else:
            i_chan = idss_t * (1 - v_gs_eff / vp_t)**2 * early
            gm = (2 * idss_t / abs(vp_t)) * (1 - v_gs_eff / vp_t) * early
        i_chan = -i_chan if is_rev else i_chan
        v_gd = v_gs - v_ds
        def _diode(v):
            if v < 0.6: return 1e-12 * (np.exp(v / 0.026) - 1.0)
            i_0 = 1e-12 * (np.exp(0.6 / 0.026) - 1.0)
            g_0 = (1e-12 / 0.026) * np.exp(0.6 / 0.026)
            return i_0 + g_0 * (v - 0.6)
        i_gs = _diode(v_gs)
        i_gd = _diode(v_gd)
        i_gate = i_gs + i_gd
        i_drain = i_chan - i_gd
        j.current_cgs = j.cgs / np.sqrt(max(0.01, 1.0 - min(v_gs, 0.55) / 0.6))
        j.current_cgd = j.cgd / np.sqrt(max(0.01, 1.0 - min(v_gd, 0.55) / 0.6))
        return i_drain, gm, i_gate, t_j_state

    def solve_dc_bias(self, input_node="v_ideal"):
        active_nodes = self._get_active_nodes()
        self.node_map = {name: idx for idx, name in enumerate(active_nodes)}
        dim = len(active_nodes)
        def kcl_equations(v_guess):
            v = {"-": 0.0}
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
                t_j_eq = j.ambient_temp
                for _ in range(3):
                    id_tmp, _, _, _ = self._jfet_physics(j, v_gs, v_ds, t_j_eq)
                    t_j_eq = j.ambient_temp + (v_ds * id_tmp * j.theta_ja)
                i_d, _, i_g, _ = self._jfet_physics(j, v_gs, v_ds, t_j_eq)
                if j.node_d in self.node_map: residuals[self.node_map[j.node_d]] += i_d
                if j.node_s in self.node_map: residuals[self.node_map[j.node_s]] -= (i_d + i_g)
                if j.node_g in self.node_map: residuals[self.node_map[j.node_g]] += i_g
            if input_node in self.node_map:
                residuals[self.node_map[input_node]] = v_guess[self.node_map[input_node]] - 0.0
            return residuals * 1e9
        v_sol = root(kcl_equations, np.zeros(dim), method='lm').x
        self.dc_op = {"-": 0.0}
        for name, idx in self.node_map.items(): self.dc_op[name] = v_sol[idx]
        return self.dc_op

    def solve_ac_thevenin(self, target_cap):
        active_nodes = [n for n in self.nodes if n not in ["+", "-", "v_ideal"]]
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
            _, gm, _, _ = self._jfet_physics(j, v_gs, v_ds, j.ambient_temp)
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

    def solve_transient(self, input_node, monitor_nodes, amplitude=0.5, periods=20.0, samples_per_period=4096):
        self.freqs = [82.41, 110.00, 146.83, 196.00, 246.94, 329.63]
        dt = (1.0 / 82.41) / samples_per_period
        self.dt = dt
        t = np.arange(0, periods / 82.41, dt)
        self.t_len = len(t)
        v_in_array = np.zeros_like(t)
        for f in self.freqs: v_in_array += (amplitude / len(self.freqs)) * np.sin(2 * np.pi * f * t)
        active_nodes = [n for n in self.nodes if n not in ["-", input_node]]
        node_map = {name: idx for idx, name in enumerate(active_nodes)}
        dim = len(active_nodes)
        v_out_data = {n: np.zeros_like(t) for n in monitor_nodes}
        v_prev = {n: self.dc_op.get(n, 0.0) for n in self.nodes}
        i_l_prev = {l.name: (self.dc_op.get(l.node1, 0.0) - self.dc_op.get(l.node2, 0.0)) / l.r_dc for l in self.inductors}
        v_guess = np.array([v_prev[n] for n in active_nodes])
        
        k_B = 1.380649e-23
        q_e = 1.602176e-19
        
        for i, v_inst in enumerate(v_in_array):
            n_v = {r.name: np.random.normal(0, np.sqrt(4 * k_B * 298.15 * r.value / dt)) for r in self.resistors}
            n_i = {j.name: np.random.normal(0, np.sqrt(2 * q_e * max(abs(j.idss), 1e-9) / dt)) for j in self.jfets}

            def kcl_transient(v_guess_t):
                v = {"-": 0.0, input_node: self.dc_op.get(input_node, 0.0) + v_inst}
                for name, idx in node_map.items(): v[name] = v_guess_t[idx]
                residuals = np.zeros(dim)
                if "+" in node_map:
                    residuals[node_map["+"]] -= (self.v_dd_ideal - v["+"]) / self.r_psu
                for r in self.resistors:
                    i_r = (v[r.node1] - v[r.node2] + n_v[r.name]) / r.value
                    if r.node1 in node_map: residuals[node_map[r.node1]] += i_r
                    if r.node2 in node_map: residuals[node_map[r.node2]] -= i_r
                for l in self.inductors:
                    v_ind = v[l.node1] - v[l.node2]
                    i_l_series = (i_l_prev[l.name] + v_ind * dt / l.value) / (1.0 + l.r_dc * dt / l.value)
                    i_c_p = (l.c_p / dt) * (v_ind - (v_prev[l.node1] - v_prev[l.node2]))
                    i_l = i_l_series + i_c_p
                    if l.node1 in node_map: residuals[node_map[l.node1]] += i_l
                    if l.node2 in node_map: residuals[node_map[l.node2]] -= i_l
                for c in self.capacitors:
                    i_c = ((v[c.node1]-v[c.node2])-(v_prev[c.node1]-v_prev[c.node2])) / (c.esr + dt/c.value)
                    if c.node1 in node_map: residuals[node_map[c.node1]] += i_c
                    if c.node2 in node_map: residuals[node_map[c.node2]] -= i_c
                for j in self.jfets:
                    v_gs, v_ds = v[j.node_g]-v[j.node_s], v[j.node_d]-v[j.node_s]
                    i_d, _, i_g, _ = self._jfet_physics(j, v_gs, v_ds, j.t_j)
                    v_gd = v_gs - v_ds
                    v_gs_prev, v_ds_prev = v_prev[j.node_g]-v_prev[j.node_s], v_prev[j.node_d]-v_prev[j.node_s]
                    
                    i_cgs = j.current_cgs * (v_gs - v_gs_prev) / dt
                    i_cgd = j.current_cgd * (v_gd - (v_gs_prev - v_ds_prev)) / dt
                    
                    i_g_tot = i_g + i_cgs + i_cgd
                    i_d_tot = i_d - i_cgd + n_i[j.name]
                    i_s_tot = i_d + i_g + i_cgs + n_i[j.name]
                    
                    if j.node_d in node_map: residuals[node_map[j.node_d]] += i_d_tot
                    if j.node_s in node_map: residuals[node_map[j.node_s]] -= i_s_tot
                    if j.node_g in node_map: residuals[node_map[j.node_g]] += i_g_tot
                return residuals * 1e9
            
            v_sol = root(kcl_transient, v_guess, method='lm').x
            v_guess = v_sol 
            v_prev[input_node] = self.dc_op.get(input_node, 0.0) + v_inst
            for name, idx in node_map.items(): v_prev[name] = v_sol[idx]
            
            for j in self.jfets:
                v_gs, v_ds = v_prev[j.node_g]-v_prev[j.node_s], v_prev[j.node_d]-v_prev[j.node_s]
                p_d = v_ds * (j.idss * (1 - v_gs/j.vp)**2 if v_gs > j.vp else 0)
                j.t_j += ((p_d/j.c_th) - (j.t_j-j.ambient_temp)/(j.theta_ja*j.c_th)) * dt
            for l in self.inductors: 
                v_ind = v_prev[l.node1] - v_prev[l.node2]
                i_l_prev[l.name] = (i_l_prev[l.name] + v_ind * dt / l.value) / (1.0 + l.r_dc * dt / l.value)
            for n in monitor_nodes: v_out_data[n][i] = v_prev[n] if n != input_node else v_inst
        return t, v_in_array, v_out_data

    def calculate_thd(self, v_out):
        v_out = v_out - np.mean(v_out)
        fft_o = np.abs(np.fft.rfft(v_out * np.hanning(len(v_out))))
        fft_f = np.fft.rfftfreq(self.t_len, d=self.dt)
        fundamental_mask = np.zeros_like(fft_o, dtype=bool)
        harmonic_energies = {2: 0.0, 3: 0.0, 4: 0.0}
        for f in self.freqs:
            idx = np.argmin(np.abs(fft_f - f))
            fundamental_mask[max(0, idx-5):min(len(fft_o), idx+6)] = True
            for h in harmonic_energies.keys():
                h_idx = np.argmin(np.abs(fft_f - f * h))
                harmonic_energies[h] += np.sum(fft_o[max(0, h_idx-3):min(len(fft_o), h_idx+4)]**2)
        f_e = np.sum(fft_o[fundamental_mask]**2)
        noise_mask = ~(fundamental_mask)
        noise_mask[0:5] = False
        n_e = np.sum(fft_o[noise_mask]**2)
        snr = 10 * np.log10(f_e / n_e) if n_e > 0 else 100.0
        thdn = np.sqrt(max(0, np.sum(fft_o[1:]**2) - f_e) / f_e) if f_e > 0 else 0.0
        ihl = {h: (np.sqrt(e / f_e) * 100) if f_e > 0 else 0.0 for h, e in harmonic_energies.items()}
        return thdn, np.max(v_out)-np.min(v_out), ihl, snr


def generate_schematic(mode):
    with schemdraw.Drawing(file=f"schematic_{mode}.png", show=False) as d:
        d.config(fontsize=10)
        
        if mode == "Clean":
            rs2_label = '3.717kΩ'
        elif mode == "OD1":
            rs2_label = '1kΩ'
        elif mode == "OD2":
            rs2_label = '500Ω'
            
        d += elm.Dot().label('IN')
        d += elm.Capacitor().right().label('C_IN\n22nF')
        node_g1 = d.here
        d += elm.Resistor().down().label('R_G_IN\n1MΩ')
        d += elm.Ground()
        
        j1 = d.add(elm.JFetN().flip().anchor('gate').label('J1\nLSK489').at(node_g1))
        
        d += elm.Resistor().down().label('R_S1\n10kΩ').at(j1.source)
        d += elm.Ground()
        
        d += elm.Line().up().length(1).at(j1.drain)
        if mode == "Clean":
            d += elm.Resistor().up().label('R_LINK_J1\n47Ω')
        elif mode == "OD1":
            d += elm.Resistor().up().label('R_STARVE_J1\n12kΩ')
        elif mode == "OD2":
            d += elm.Resistor().up().label('R_STARVE_J1\n40kΩ')
        d += elm.Vdd().label('+18V')

        d += elm.Line().right().length(1.5).at(j1.source)
        d += elm.Capacitor().right().label('C_MID\n22nF')
        node_g2 = d.here
        
        d += elm.Resistor().down().label('R_G2\n1MΩ')
        d += elm.Ground()
        
        d += elm.Resistor().up().label('R_G1\n6.88MΩ').at(node_g2)
        d += elm.Vdd().label('+18V')

        j2 = d.add(elm.JFetN().flip().anchor('gate').label('J2\nLSK489').at(node_g2))
        
        d += elm.Resistor().down().label(f'R_S2\n{rs2_label}').at(j2.source)
        d += elm.Ground()
        
        d += elm.Line().up().length(0.5).at(j2.drain)
        node_d2 = d.here
        d += elm.Resistor().up().label('R_D2\n6.8kΩ')
        node_vdd_local = d.here
        
        if mode == "Clean" or mode == "OD1":
            d += elm.Resistor().up().label('R_LINK\n47Ω').at(node_vdd_local)
            d += elm.Vdd().label('+18V')
        elif mode == "OD2":
            d += elm.Resistor().up().label('R_SAG\n10kΩ').at(node_vdd_local)
            d += elm.Vdd().label('+18V')
            
            d += elm.Line().right().length(1.5).at(node_vdd_local)
            d += elm.Capacitor().down().label('C_SAG\n4.7μF')
            d += elm.Ground()

        d += elm.Capacitor().right().label('C_OUT\n22nF').at(node_d2)
        node_out = d.here
        d += elm.Resistor().down().label('R_LOAD\n100kΩ')
        d += elm.Ground()
        
        d += elm.Dot().label('OUT').at(node_out)

# --- Component Placement Reconstruction ---
def build_and_run_simulation(mode="Clean"):
    print(f"\n--- {mode} Mode Analysis ---")
    sim = Circuit(v_dd_ideal=18.0, r_psu=100.0)
    
    # --- STAGE 1: J1 Source Follower (Input Buffer) ---
    sim.add(JFET("J1", idss=0.0055, vp=-2.5, node_d="D1", node_g="G1", node_s="S1"))
    sim.add(Resistor("R_G_IN", 1.0e6, "G1", "-"))
    sim.add(Capacitor("C_IN", 22e-9, "IN", "G1"))
    sim.add(Resistor("R_S1", 10000.0, "S1", "-"))
    
    # Interstage Coupling High-Pass Filter
    sim.add(Capacitor("C_MID", 22e-9, "S1", "G2"))
    
    # --- STAGE 2: J2 Common-Source Amplifier (Gain = 1.60 V/V) ---
    sim.add(JFET("J2", idss=0.0055, vp=-2.5, node_d="D2", node_g="G2", node_s="S2"))
    sim.add(Resistor("R_G1", 6.88e6, "+", "G2"))
    sim.add(Resistor("R_G2", 1.0e6, "G2", "-"))
    sim.add(Resistor("R_D2", 6800.0, "VDD_Local", "D2"))
    
    if mode == "Clean":
        sim.add(Resistor("R_S2", 3717.0, "S2", "-"))
    elif mode == "OD1":
        sim.add(Resistor("R_S2", 1000.0, "S2", "-"))
    elif mode == "OD2":
        sim.add(Resistor("R_S2", 500.0, "S2", "-"))
    
    # --- TOPOLOGICAL MODE SWITCHING LOGIC ---
    if mode == "Clean":
        sim.add(Resistor("R_LINK_J1", 47.0, "+", "D1"))
        sim.add(Resistor("R_LINK", 47.0, "+", "VDD_Local"))
    elif mode == "OD1":
        sim.add(Resistor("R_STARVE_J1", 12000.0, "+", "D1"))
        sim.add(Resistor("R_LINK", 47.0, "+", "VDD_Local"))
    elif mode == "OD2":
        sim.add(Resistor("R_STARVE_J1", 40000.0, "+", "D1"))
        sim.add(Resistor("R_SAG", 10000.0, "+", "VDD_Local"))
        sim.add(Capacitor("C_SAG", 4.7e-6, "VDD_Local", "-"))

    # --- OUTPUT COUPLING ---
    sim.add(Capacitor("C_OUT", 22e-9, "D2", "OUT"))
    sim.add(Resistor("R_LOAD", 100000.0, "OUT", "-"))

    # Execute DC operating point derivation
    dc_results = sim.solve_dc_bias(input_node="IN")
    
    for j in sim.jfets:
        v_gs = dc_results.get(j.node_g, 0.0) - dc_results.get(j.node_s, 0.0)
        v_ds = dc_results.get(j.node_d, 0.0) - dc_results.get(j.node_s, 0.0)
        i_d, _, _, t_j = sim._jfet_physics(j, v_gs, v_ds, j.ambient_temp)
        print(f"[{j.name}] V_GS: {v_gs:.4f}V | V_DS: {v_ds:.4f}V | I_D: {i_d*1000:.4f}mA | T_j: {t_j:.2f}°C")

    # Execute Time-Domain Transient Analysis
    monitor_nodes = ["IN", "G1", "S1", "G2", "D2", "OUT"]
    if mode == "OD2":
        monitor_nodes.append("VDD_Local")
        
    t, v_in, v_out_data = sim.solve_transient(input_node="IN", monitor_nodes=monitor_nodes, amplitude=0.5)
    
    ref_vpp = np.max(v_out_data["G1"]) - np.min(v_out_data["G1"])
    out_vpp = np.max(v_out_data["OUT"]) - np.min(v_out_data["OUT"])
    print(f"Total Circuit Path Gain (Vpp_out/Vpp_ref): {out_vpp/ref_vpp:.4f} V/V")
    
    base_ac = v_out_data["IN"] - np.mean(v_out_data["IN"])
    for n in monitor_nodes:
        thdn, vpp, ihl, snr = sim.calculate_thd(v_out_data[n])
        target_ac = v_out_data[n] - np.mean(v_out_data[n])
        phase_str = "In-Phase (0°)" if np.sum(base_ac * target_ac) >= 0 else "Inverted (180°)"
        print(f"[{n}] Vpp: {vpp:.4f}V | THD+N: {thdn*100:.4f}% | SNR: {snr:.2f}dB | Phase: {phase_str}")
        if n == "OUT": 
            print(f"IHL: 2nd: {ihl[2]:.4f}% | 3rd: {ihl[3]:.4f}% | 4th: {ihl[4]:.4f}%")
            
    for cap in sim.capacitors:
        if cap.name == "C_SAG": continue
        R_th = sim.solve_ac_thevenin(cap)
        print(f"[{cap.name}] R_th: {R_th/1000:.2f} kOhms | f_c: {1.0/(2*np.pi*R_th*cap.value):.2f} Hz | Total Recovery: {(5*R_th*cap.value)*1000:.2f} ms")

    titles = {
        "IN": "Base Input Reference / Pre-Coupling (IN)",
        "G1": "J1 Gate / Divider Network (G1)",
        "S1": "J1 Source / Non-Inverting Follower Output (S1)",
        "G2": "J2 Gate / Post-Coupling Input (G2)",
        "D2": "J2 Drain / Pre-Volume Output (D2)",
        "OUT": "Final Master Output (OUT)",
        "VDD_Local": "Passive RC Supply Sag (VDD_Local)"
    }
    
    plt.figure(figsize=(10, 2 * len(monitor_nodes)))
    plt.suptitle(f"{mode} Mode (Full Physical Model)")
    for i, n in enumerate(monitor_nodes):
        ax = plt.subplot(len(monitor_nodes), 1, i+1)
        ax.plot(t*1000, v_out_data[n], label=n)
        ax.set_title(titles.get(n, f"Signal at Node: {n}"))
        ax.grid(True)
    plt.tight_layout()
    plt.savefig(f'mode_{mode}.png')
    plt.close()

if __name__ == "__main__":
    for op_mode in ["Clean", "OD1", "OD2"]:
        generate_schematic(mode=op_mode)
        build_and_run_simulation(mode=op_mode)