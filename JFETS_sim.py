import numpy as np
from scipy.optimize import root
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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

    def solve_transient(self, input_node, monitor_nodes, freqs, amplitude=0.5, periods=20.0, samples_per_period=4096):
        self.freqs = freqs
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


class CircuitAnalyzer:
    def __init__(self, circuit, monitor_nodes, input_node="v_ideal", amplitude=2.0):
        self.circuit = circuit
        self.monitor_nodes = monitor_nodes
        self.input_node = input_node
        self.amplitude = amplitude
        self.t = None
        self.v_in = None
        self.v_out_data = None
        self.freqs = [82.41, 110.00, 146.83, 196.00, 246.94, 329.63]

    def report_dc_bias(self):
        dc_results = self.circuit.solve_dc_bias(input_node=self.input_node)
        for j in self.circuit.jfets:
            v_gs = dc_results.get(j.node_g, 0.0) - dc_results.get(j.node_s, 0.0)
            v_ds = dc_results.get(j.node_d, 0.0) - dc_results.get(j.node_s, 0.0)
            i_d, _, _, t_j = self.circuit._jfet_physics(j, v_gs, v_ds, j.ambient_temp)
            print(f"[{j.name}] Bias -> V_GS: {v_gs:.4f}V | V_DS: {v_ds:.4f}V | I_D: {i_d*1000:.4f}mA | T_j: {t_j:.2f}°C")

    def run_transient(self):
        self.t, self.v_in, self.v_out_data = self.circuit.solve_transient(
            input_node=self.input_node, 
            monitor_nodes=self.monitor_nodes, 
            freqs=self.freqs,
            amplitude=self.amplitude,
            samples_per_period=4096
        )
        ref_vpp = np.max(self.v_out_data[self.input_node]) - np.min(self.v_out_data[self.input_node])
        out_vpp = np.max(self.v_out_data["OUT"]) - np.min(self.v_out_data["OUT"])
        print(f"Total Circuit Path Gain (Vpp_out/Vpp_ref): {out_vpp/ref_vpp:.4f} V/V")

        area_target = np.mean(np.abs(self.v_out_data[self.input_node]))
        area_j1 = np.mean(np.abs(self.v_out_data["D1"] - np.mean(self.v_out_data["D1"])))
        area_out = np.mean(np.abs(self.v_out_data["OUT"]))
        print(f"Mean Absolute Area (Target): {area_target:.4f} V*s")
        print(f"Mean Absolute Area (Post-J1): {area_j1:.4f} V*s")
        print(f"Mean Absolute Area (Post-J2 OUT): {area_out:.4f} V*s")

    def report_multi_frequency_thd(self, node="OUT"):
        print(f"\n--- Origin-Sorted Multi-Frequency THD Tracker ({node}) ---")
        v_out = self.v_out_data[node]
        v_out_ac = v_out - np.mean(v_out)
        N = len(v_out_ac)
        
        Y = np.fft.rfft(v_out_ac * np.hanning(N))
        xf = np.fft.rfftfreq(N, d=self.circuit.dt)
        mag = 2.0/N * np.abs(Y)
        
        for f in self.freqs:
            idx_f = np.argmin(np.abs(xf - f))
            mag_f = mag[idx_f]
            print(f"Fundamental Source: {f:>6.2f} Hz | Mag: {mag_f:.4f}V")
            for h in [2, 3, 4]:
                target_f = f * h
                idx_h = np.argmin(np.abs(xf - target_f))
                mag_h = mag[idx_h]
                ihl = (mag_h / mag_f * 100.0) if mag_f > 1e-6 else 0.0
                print(f"  -> {h}x Harmonic ({target_f:>6.2f} Hz): Mag: {mag_h:.6f}V | IHL: {ihl:.4f}%")

    def report_ac_analytics(self):
        print("\n--- Capacitor Thevenin Analytics ---")
        for cap in self.circuit.capacitors:
            R_th = self.circuit.solve_ac_thevenin(cap)
            fc = 1.0 / (2 * np.pi * R_th * cap.value)
            t_rec = 5 * R_th * cap.value
            print(f"[{cap.name}] R_th: {R_th/1000:.2f} kOhms | f_c: {fc:.2f} Hz | 5-Tau Recovery: {t_rec*1000:.2f} ms")

    def plot_waveforms(self, mode):
        fig = plt.figure(figsize=(16, 10))
        plt.suptitle(f"{mode} Mode: Signal Chain and Harmonic Analysis", fontweight='bold', fontsize=14)
        gs = gridspec.GridSpec(4, 2, figure=fig, width_ratios=[1, 1])
        
        nodes_to_plot = ["v_ideal", "G1", "D1", "OUT"]
        titles = [
            "Pre-Pickup Source (v_ideal)",
            "Pre-J1 Gate Input (G1)",
            "J1 Drain Output (D1)",
            "Final Master Output (OUT)"
        ]
        colors = ['gray', 'green', 'blue', 'black']
        t_ms = self.t * 1000
        
        for i, (node, title, color) in enumerate(zip(nodes_to_plot, titles, colors)):
            ax = fig.add_subplot(gs[i, 0])
            ax.plot(t_ms, self.v_out_data[node], color=color, linewidth=1.5)
            ax.set_title(title, fontsize=11)
            ax.set_ylabel("Amplitude (V)", fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 40)
            if i == 3:
                ax.set_xlabel("Time (ms)", fontsize=10)
            else:
                ax.set_xticklabels([])
                
        ax_fft = fig.add_subplot(gs[:, 1])
        v_out_ac = self.v_out_data["OUT"] - np.mean(self.v_out_data["OUT"])
        N = len(v_out_ac)
        Y = np.fft.rfft(v_out_ac * np.hanning(N))
        xf = np.fft.rfftfreq(N, d=self.circuit.dt)
        mag = 2.0/N * np.abs(Y)
        
        ax_fft.plot(xf, mag, color='gray', alpha=0.2, label="IMD Floor")
        
        color_cycle = ['red', 'blue', 'green', 'orange', 'purple', 'cyan']
        for f_val, c in zip(self.freqs, color_cycle):
            harmonics = np.arange(1, 10)
            h_freqs = harmonics * f_val
            h_mags = [mag[np.argmin(np.abs(xf - hf))] for hf in h_freqs]
            marker, stemlines, _ = ax_fft.stem(h_freqs, h_mags, linefmt=c, basefmt=' ')
            plt.setp(marker, color=c, markersize=4, alpha=0.8)
            plt.setp(stemlines, color=c, alpha=0.6, linewidth=1.5)
            
        ax_fft.set_title("Origin-Sorted Harmonic Spectrum (OUT)", fontsize=12)
        ax_fft.set_xlabel("Frequency (Hz)", fontsize=10)
        ax_fft.set_ylabel("Magnitude (V)", fontsize=10)
        ax_fft.set_xlim(0, 3500)
        ax_fft.set_ylim(0, max(np.max(mag) * 1.1, 0.5))
        ax_fft.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'mode_{mode}_analysis.png')
        plt.close()


def build_core_topology(sim, mode):
    sim.add(Inductor("L_PICKUP", 4.5, "v_ideal", "IN", r_dc=8000.0, c_p=150e-12))
    sim.add(Capacitor("C_IN", 22e-9, "IN", "G1"))
    sim.add(Resistor("R_G1", 1.0e6, "G1", "-"))
    sim.add(JFET("J1", idss=0.0055, vp=-2.5, node_d="D1", node_g="G1", node_s="S1"))
    
    if mode == "Clean":
        sim.add(Resistor("R_S1", 2200.0, "S1", "-"))  
        sim.add(Resistor("R_D1", 10000.0, "+", "D1"))
    elif mode == "OD1":
        sim.add(Resistor("R_S1", 2200.0, "S1", "-"))   
        sim.add(Capacitor("C_S1", 22e-6, "S1", "-"))
        sim.add(Resistor("R_D1", 10000.0, "+", "D1"))
    elif mode == "OD2":
        sim.add(Resistor("R_S1", 470.0, "S1", "-"))   
        sim.add(Capacitor("C_S1", 22e-6, "S1", "-"))
        sim.add(Resistor("R_D1", 10000.0, "+", "D1"))
        
    sim.add(Capacitor("C_MID", 22e-9, "D1", "G2"))
    sim.add(Resistor("R_G2", 1.0e6, "G2", "-"))
    sim.add(JFET("J2", idss=0.0055, vp=-2.5, node_d="D2", node_g="G2", node_s="S2"))
    sim.add(Resistor("R_D2", 10000.0, "+", "D2"))
    
    sim.add(Resistor("R_S2", 2200.0, "S2", "-"))


def generate_schematic(mode, r_top_val, r_bot_val):
    with schemdraw.Drawing(file=f"schematic_{mode}.png", show=False) as d:
        d.config(fontsize=10)
        
        d += elm.Dot().label('v_ideal')
        d += elm.Inductor().right().label('L_PICKUP\n4.5H')
        node_in = d.here
        d += elm.Dot().label('IN').at(node_in)
        
        d += elm.Capacitor().right().label('C_IN\n22nF')
        node_g1 = d.here
        d += elm.Resistor().down().label('R_G1\n1MΩ')
        d += elm.Ground()
        
        j1 = d.add(elm.JFetN().flip().anchor('gate').label('J1\nLSK489').at(node_g1))
        
        if mode == "Clean":
            rs1_val = "2.2kΩ"
        elif mode == "OD1":
            rs1_val = "2.2kΩ"
        elif mode == "OD2":
            rs1_val = "470Ω"
            
        d += elm.Resistor().down().label(f'R_S1\n{rs1_val}').at(j1.source)
        d += elm.Ground()
        
        if mode in ["OD1", "OD2"]:
            d += elm.Line().right().length(1.5).at(j1.source)
            d += elm.Capacitor().down().label('C_S1\n22μF')
            d += elm.Ground()
        
        d += elm.Line().up().length(0.5).at(j1.drain)
        node_d1 = d.here
        d += elm.Dot().label('D1').at(node_d1)
        d += elm.Resistor().up().label('R_D1\n10kΩ')
        d += elm.Vdd().label('+18V')
        
        d += elm.Line().right().length(1.5).at(node_d1)
        d += elm.Capacitor().right().label('C_MID\n22nF')
        node_g2 = d.here
        
        d += elm.Resistor().down().label('R_G2\n1MΩ')
        d += elm.Ground()
        
        j2 = d.add(elm.JFetN().flip().anchor('gate').label('J2\nLSK489').at(node_g2))
        
        d += elm.Resistor().down().label('R_S2\n2.2kΩ').at(j2.source)
        d += elm.Ground()
        
        d += elm.Line().up().length(0.5).at(j2.drain)
        node_d2 = d.here
        d += elm.Resistor().up().label('R_D2\n10kΩ')
        d += elm.Vdd().label('+18V')
        
        d += elm.Capacitor().right().label('C_OUT\n22nF').at(node_d2)
        node_div_mid = d.here
        d += elm.Resistor().right().label(f'R_VOL_TOP\n{r_top_val/1000:.2f}kΩ').at(node_div_mid)
        node_out = d.here
        d += elm.Resistor().down().label(f'R_VOL_BOT\n{r_bot_val/1000:.2f}kΩ').at(node_out)
        d += elm.Ground()
        
        d += elm.Dot().label('OUT').at(node_out)


if __name__ == "__main__":
    monitor_nodes = ["v_ideal", "G1", "D1", "OUT"]
    
    for mode in ["Clean", "OD1", "OD2"]:
        print("\n========================================")
        print(f"--- Constructing Physical Model: {mode} Mode ---")
        print("========================================")
        
        # --- PRE-FLIGHT CALIBRATION ---
        sim_pre = Circuit(v_dd_ideal=18.0, r_psu=100.0)
        build_core_topology(sim_pre, mode)
        sim_pre.add(Capacitor("C_OUT_PRE", 22e-9, "D2", "OUT_PRE"))
        sim_pre.add(Resistor("R_LOAD_PRE", 100000.0, "OUT_PRE", "-"))
        
        sim_pre.solve_dc_bias(input_node="v_ideal")
        _, _, v_out_pre = sim_pre.solve_transient(
            input_node="v_ideal", 
            monitor_nodes=["OUT_PRE"], 
            freqs=[82.41, 110.00, 146.83, 196.00, 246.94, 329.63], 
            amplitude=0.25, 
            samples_per_period=1024
        )
        
        raw_vpp = np.max(v_out_pre["OUT_PRE"]) - np.min(v_out_pre["OUT_PRE"])
        attenuation_factor = 1.0 / raw_vpp if raw_vpp > 0 else 1.0
        r_bot_val = min(100000.0, max(1.0, 100000.0 * attenuation_factor))
        r_top_val = 100000.0 - r_bot_val
        if r_top_val < 1.0:
            r_top_val = 1.0
            r_bot_val = 100000.0
            
        print(f"Pre-Flight Raw Vpp: {raw_vpp:.4f}V | Attenuation Target: {attenuation_factor:.4f}")
        print(f"Calculated Divider -> R_TOP: {r_top_val/1000:.2f}kΩ | R_BOT: {r_bot_val/1000:.2f}kΩ")

        # --- FINAL HIGH-RESOLUTION SIMULATION ---
        sim = Circuit(v_dd_ideal=18.0, r_psu=100.0)
        build_core_topology(sim, mode)
        sim.add(Capacitor("C_OUT", 22e-9, "D2", "DIVIDER_MID"))
        sim.add(Resistor("R_VOL_TOP", r_top_val, "DIVIDER_MID", "OUT"))
        sim.add(Resistor("R_VOL_BOT", r_bot_val, "OUT", "-"))

        analyzer = CircuitAnalyzer(circuit=sim, monitor_nodes=monitor_nodes, input_node="v_ideal", amplitude=0.25)
        
        analyzer.report_dc_bias()
        analyzer.run_transient()
        analyzer.report_multi_frequency_thd(node="OUT")
        analyzer.report_ac_analytics()
        analyzer.plot_waveforms(mode=mode)
        
        generate_schematic(mode=mode, r_top_val=r_top_val, r_bot_val=r_bot_val)