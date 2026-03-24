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
# These are physical/hardware constants, not user-tunable parameters.
# User inputs (blocking thresholds, volume ratio, mode descriptors) are
# defined in the if __name__ block — nowhere else.
_INPUT_AMPLITUDE = 0.25  # peak pickup voltage (V) — hot humbucker worst case
# Input VPA is COMPUTED (not estimated) by _compute_input_vpa() from
# _INPUT_AMPLITUDE, _CAL_FREQ, and _a_weight.  No hardcoded constant.

# OPA1656 rail limits (from datasheet: output within 250mV of rails with 2kΩ load)
# These are VALIDATION BOUNDS — the opamp is a linear buffer and must never clip.
# If the signal at BUF exceeds these limits, the upstream design is wrong.
_OPAMP_V_MIN = 0.25    # V- + 250mV (GND-referenced)
_OPAMP_V_MAX = 17.75   # V+ - 250mV (18V supply)

# ==========================================================================
#  LSK489 Datasheet Physics — Reconstructed Equations
# ==========================================================================
#  Every curve in the LSK489A/B datasheet (pp 4–7) is a plot of the
#  Shockley JFET model with three parameters: IDSS, VP, λ.
#  This class reconstructs ALL datasheet curves as callable functions
#  so that operating points are derived, never hand-read from graphs.
# ==========================================================================

class LSK489_Model:
    """
    Complete N-channel JFET model reconstructed from LSK489 datasheet.

    All curves are derived from (IDSS, VP, λ).  No values are read from
    plots — the plots themselves are outputs of these equations.

    Datasheet cross-reference for each method:
        drain_current          → Output Characteristics, p4 (large plot)
        transconductance       → Drain Current & Transconductance vs VGS(off), p4 (top-left)
                                 Common-Source Forward Transconductance vs ID, p7 (bottom-right)
        output_conductance     → Output Conductance vs Drain Current, p7 (bottom-left)
        rds_on                 → On-Resistance and Output Conductance vs VGS(off), p7 (top-left)
        sat_triode_boundary    → knee location in Output Characteristics, p4
        self_bias_point        → Transfer Characteristics, p5 (bottom-left)
                                 Operating Characteristics, p5 (top-left, triode detail)
        noise_density          → Equivalent Input Noise Voltage vs Frequency, p4 (bottom-right)
        capacitance_cgs/cgd    → Common-Source Input/Reverse Feedback Capacitance, p5/p7
        gate_current           → Operating Gate Current, p4 (top-right)
        voltage_gain_cs        → Circuit Voltage Gain vs Drain Current, p6 (bottom-left)
    """

    def __init__(self, idss, vp_abs, lambda_mod=0.01):
        """
        Args:
            idss:       Drain-source saturation current (A).
                        LSK489A: 2.5–8.5 mA, typ 5.5 mA.
            vp_abs:     |VP|, absolute pinch-off voltage (V).
                        LSK489A: 1.5–3.5 V.
            lambda_mod: Channel-length modulation parameter (1/V).
                        Not directly on datasheet; extracted from Output
                        Characteristics slope in saturation.  Typ ~0.01.
        """
        self.idss = idss
        self.vp = -vp_abs          # Internal convention: VP < 0 for N-channel
        self.vp_abs = vp_abs
        self.lambda_mod = lambda_mod
        self.beta = idss / (vp_abs ** 2)   # Transconductance parameter

        # Gate junction model (same parameters used in _gate_junction_jit)
        self.vt = 0.02569          # kT/q at 25°C (V)
        self.is_0 = 1.0e-14        # Reverse saturation current (A)
        self.phi = 0.6             # Built-in potential (V)

        # Capacitance zero-bias values (from datasheet: Ciss=4pF, Crss=2pF)
        # Ciss = Cgs + Cgd → Cgs0 ≈ Ciss - Crss = 2 pF
        self.cgs0 = 2.0e-12
        self.cgd0 = 2.0e-12        # Cgd0 ≈ Crss = 2 pF

        # Noise parameters (derived from datasheet p4 bottom-right)
        # en_flat = 1.8 nV/√Hz (white floor at ≥1 kHz)
        # en(10Hz) = 3.5 nV/√Hz → K_1f = (3.5² - 1.8²) × 10 nV²
        self.en_flat = 1.8e-9
        self._k_1f = (3.5e-9**2 - self.en_flat**2) * 10.0

        # Gate forward conduction threshold (diode turn-on ~0.6V)
        self.vgs_gate_threshold = 0.6

    # --- Output Characteristics (datasheet p4, large plot) ---

    def drain_current(self, vgs, vds):
        """
        Shockley drain current for both saturation and triode regions.

        Saturation (VDS ≥ VGS − VP):
            ID = IDSS × (1 − VGS/VP)² × (1 + λ·VDS)

        Triode (VDS < VGS − VP):
            ID = β × (2·(VGS−VP)·VDS − VDS²) × (1 + λ·VDS)
            where β = IDSS / VP²

        Cutoff (VGS ≤ VP):
            ID ≈ 0
        """
        if vgs <= self.vp:
            return 0.0
        early = 1.0 + self.lambda_mod * abs(vds)
        vgst = vgs - self.vp  # Always positive above cutoff
        if vds >= vgst:
            ratio = 1.0 - vgs / self.vp
            return self.idss * ratio**2 * early
        else:
            return self.beta * (2.0 * vgst * vds - vds**2) * early

    # --- Forward Transconductance (datasheet p4 top-left, p7 bottom-right) ---

    def transconductance(self, vgs, vds):
        """
        gm = ∂ID/∂VGS.

        Saturation: gm = (2·IDSS/|VP|) × (1 − VGS/VP) × (1 + λ·VDS)
        Triode:     gm = 2·β·VDS × (1 + λ·VDS)
        """
        if vgs <= self.vp:
            return 0.0
        early = 1.0 + self.lambda_mod * abs(vds)
        vgst = vgs - self.vp
        if vds >= vgst:
            return (2.0 * self.idss / self.vp_abs) * (1.0 - vgs / self.vp) * early
        else:
            return 2.0 * self.beta * vds * early

    # --- Output Conductance (datasheet p7 bottom-left) ---

    def output_conductance(self, vgs, vds):
        """
        gds = ∂ID/∂VDS.

        Saturation: gds = λ × IDSS × (1 − VGS/VP)²
        Triode:     gds = β × (2·(VGS−VP) − 2·VDS) × (1 + λ·VDS) + λ×ID
        """
        if vgs <= self.vp:
            return 0.0
        vgst = vgs - self.vp
        early = 1.0 + self.lambda_mod * abs(vds)
        if vds >= vgst:
            ratio = 1.0 - vgs / self.vp
            return self.lambda_mod * self.idss * ratio**2
        else:
            id_triode = self.beta * (2.0 * vgst * vds - vds**2) * early
            return self.beta * (2.0 * vgst - 2.0 * vds) * early + self.lambda_mod * id_triode

    # --- On-Resistance (datasheet p7 top-left) ---

    def rds_on(self, vgs):
        """
        Drain-source on-resistance at VDS ≈ 0.

        rds(on) = 1 / (2·β·(VGS − VP)) = |VP|² / (2·IDSS·(VGS − VP))
        """
        if vgs <= self.vp:
            return float('inf')
        return 1.0 / (2.0 * self.beta * (vgs - self.vp))

    # --- Saturation-Triode Boundary ---

    def sat_triode_boundary(self, vgs):
        """
        VDS at the saturation-triode boundary (the "knee" in Output
        Characteristics where curves flatten from triode into saturation).

        VDS_boundary = VGS − VP = VGS + |VP|

        For self-biased stage at fraction alpha = ID/IDSS:
            VGS = VP × (1 − √α) = −|VP| × (1 − √α)
            VDS_boundary = −|VP|×(1−√α) + |VP| = |VP|×√α

        Key result: the drain voltage VD at the boundary is ALWAYS |VP|,
        regardless of alpha.  This is because:
            VD = VS + VDS_boundary
               = |VP|×(1−√α) + |VP|×√α
               = |VP|
        """
        return max(0.0, vgs - self.vp)

    def boundary_at_alpha(self, alpha):
        """VDS at saturation-triode boundary for a self-biased stage."""
        return self.vp_abs * np.sqrt(alpha)

    def drain_voltage_at_boundary(self):
        """
        VD at the saturation-triode boundary — always |VP| for any alpha.
        This is the key design anchor for 'soft' clipping mode.
        """
        return self.vp_abs

    # --- Self-Bias Operating Point (datasheet p5 Transfer Characteristics) ---

    def self_bias_point(self, alpha):
        """
        Complete DC operating point for self-biased common-source stage.

        From Shockley:  ID = IDSS × (1 − VGS/VP)²
        Self-bias:      VGS = −ID × RS
        Solving:        VGS = VP × (1 − √α)
                        VS = −VGS = |VP| × (1 − √α)
                        RS = VS / ID
        """
        if alpha <= 0 or alpha > 1.0:
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        id_q = alpha * self.idss
        vgs = self.vp * (1.0 - np.sqrt(alpha))     # Negative for N-ch
        vs = -vgs                                    # Positive, above GND
        rs = vs / id_q if id_q > 0 else float('inf')
        # gm in saturation (VDS >> boundary for accuracy)
        gm = self.transconductance(vgs, 10.0)
        vds_boundary = self.sat_triode_boundary(vgs)
        return {
            'alpha': alpha, 'id': id_q, 'vgs': vgs, 'vs': vs, 'rs': rs,
            'gm': gm, 'vds_boundary': vds_boundary,
            'vd_at_boundary': vs + vds_boundary,     # = |VP| always
        }

    # --- Noise (datasheet p4 bottom-right) ---

    def noise_density(self, freq_hz):
        """
        Input-referred voltage noise density en(f) in V/√Hz.
        en = √(en_flat² + K/f)

        Derived from two datasheet points:
            en(1 kHz) = 1.8 nV/√Hz → en_flat
            en(10 Hz) = 3.5 nV/√Hz → K = (3.5² − 1.8²) × 10 nV²
        """
        return np.sqrt(self.en_flat**2 + self._k_1f / max(freq_hz, 0.1))

    # --- Capacitances (datasheet p5 top-right, p7 top-right) ---

    def capacitance_cgs(self, vgs):
        """Cgs(VGS) = Cgs0 / √(1 − VGS/φ), junction capacitance model."""
        return self.cgs0 / np.sqrt(max(0.01, 1.0 - min(vgs, 0.55) / self.phi))

    def capacitance_cgd(self, vgd):
        """Cgd(VGD) = Cgd0 / √(1 − VGD/φ), junction capacitance model."""
        return self.cgd0 / np.sqrt(max(0.01, 1.0 - min(vgd, 0.55) / self.phi))

    # --- Gate Current (datasheet p4 top-right) ---

    def gate_current(self, vgs):
        """
        Gate junction current.  Forward-biases at VGS > ~0.6V.
        Exponential turn-on per PN junction diode equation.
        Abs max: IG(F) = 10 mA (datasheet).
        """
        if vgs < 0:
            return -(self.is_0 + 1.33e-13 * abs(vgs))  # Reverse leakage
        return self.is_0 * (np.exp(min(vgs, 1.0) / self.vt) - 1.0)

    # --- Common-Source Voltage Gain (datasheet p6 bottom-left) ---

    def voltage_gain_cs(self, alpha, rd):
        """
        Unbypassed common-source gain: Av = gm × RD / (1 + gm × RS).
        Equivalent to: Av = RD / (RS + 1/gm)
        """
        bp = self.self_bias_point(alpha)
        re = 1.0 / bp['gm']        # Intrinsic emitter/source impedance
        return rd / (bp['rs'] + re)

    # --- Mode-Specific RD Computation ---

    def compute_rd(self, alpha, descriptor, vdd):
        """
        Compute drain resistor from operating region descriptor.

        Args:
            alpha:      ID/IDSS fraction (self-bias operating point)
            descriptor: 'linear' or 'soft'
            vdd:        Supply voltage

        Returns:
            (rd, bias_point_dict)

        Physics:
            'linear': VD placed at midpoint between |VP| and VDD for
                      maximum symmetric swing headroom.  RD follows from
                      RD = (VDD − VD) / ID.

            'soft':   VD placed at |VP| — the saturation-triode boundary.
                      Signal peaks push into triode knee of Output
                      Characteristics (p4).  RD = (VDD − |VP|) / ID.
        """
        bp = self.self_bias_point(alpha)
        id_q = bp['id']
        vd_boundary = bp['vd_at_boundary']   # = |VP|

        if descriptor == 'soft':
            # Drain at sat-triode boundary: peaks enter triode
            rd = (vdd - vd_boundary) / id_q
            return rd, bp

        elif descriptor == 'linear':
            # This is a placeholder — the actual linear RD is optimized
            # by the headroom sweep in _resolve_circuit_parameters.
            # Here we return the maximum-headroom starting point:
            # VD centered between boundary and VDD
            vd_target = (vdd + vd_boundary) / 2.0
            rd = (vdd - vd_target) / id_q
            return rd, bp

        elif descriptor == 'hard':
            raise ValueError(
                "CONFLICT: 'hard' (gate clipping) requires RS to approach 0Ω "
                "so VGS quiescent approaches 0V. This makes RS per-mode "
                "switched (currently global). Use compute_hard_bias() instead."
            )
        else:
            raise ValueError(
                f"Unknown descriptor '{descriptor}'. "
                f"Valid options: 'linear', 'soft', 'hard'"
            )

    def compute_hard_bias(self, vdd, target_vgs_quiescent=-0.1):
        """
        Compute RS and alpha for gate clipping mode.

        Gate conduction requires VGS > +0.6V on signal peaks.
        Quiescent VGS must be near 0V (slightly negative).

        From Shockley: VGS = VP × (1 − √α)
        For VGS ≈ target: α = (1 − target/VP)²
        """
        alpha = (1.0 - target_vgs_quiescent / self.vp) ** 2
        alpha = np.clip(alpha, 0.01, 0.99)
        bp = self.self_bias_point(alpha)
        # Gate conduction onset: need signal_peak > 0.6 − VGS_q
        gate_onset_signal = self.vgs_gate_threshold - bp['vgs']
        return alpha, bp, gate_onset_signal


# ==========================================================================
#  Input Validation — Comprehensive Error Checking
# ==========================================================================

_VALID_Q_DESCRIPTORS = {'linear', 'soft', 'hard'}

class ConfigError(Exception):
    """Raised when user inputs contain conflicts or invalid values."""
    pass


def validate_inputs(modes, volume_ratio, blocking, bloom_light, bloom_heavy,
                    vdd=18.0, jfet_model=None):
    """
    Comprehensive validation of all user inputs.

    Checks:
      qual-qual:   descriptor conflicts between stages/modes
      qual-quant:  descriptors vs. blocking/bloom thresholds
      quant-quant: threshold window validity

    Args:
        modes:        dict of {mode_name: {"Q1": str, "Q2": str, "bloom": False|"light"|"heavy"}}
        volume_ratio: VPA_out / VPA_in target
        blocking:     dict with fc_hz, t5_ms, fc_floor_hz, t5_floor_ms
        bloom_light:  dict with fc_hz, t5_ms, fc_floor_hz, t5_floor_ms
        bloom_heavy:  dict with fc_hz, t5_ms, fc_floor_hz, t5_floor_ms
        vdd:          Supply voltage
        jfet_model:   LSK489_Model instance

    Raises:
        ConfigError with specific conflict description.
    """
    errors = []

    # ---- Bloom field validation ----
    _VALID_BLOOM = {False, "light", "heavy"}
    for mode_name, mode_def in modes.items():
        bv = mode_def.get('bloom', False)
        if bv not in _VALID_BLOOM:
            errors.append(
                f"[{mode_name}] Invalid bloom value '{bv}'. "
                f"Must be one of: {_VALID_BLOOM}"
            )

    # ---- Quantitative vs. Quantitative: Blocking window ----
    tau_ceil = blocking['t5_ms'] / 1000.0 / 5.0
    tau_floor = 1.0 / (2.0 * np.pi * blocking['fc_hz'])
    if tau_floor > tau_ceil:
        errors.append(
            f"BLOCKING: tau_floor ({tau_floor*1000:.2f} ms, from fc={blocking['fc_hz']:.1f} Hz) "
            f"> tau_ceiling ({tau_ceil*1000:.2f} ms, from 5τ={blocking['t5_ms']:.1f} ms). "
            f"No valid blocking tau window exists. "
            f"Either increase BLOCKING_FC_HZ or increase BLOCKING_T5_MS."
        )

    tau_floor_sec = blocking['t5_floor_ms'] / 1000.0 / 5.0   # tau ≥ this
    tau_ceil_sec  = 1.0 / (2.0 * np.pi * blocking['fc_floor_hz'])  # tau ≤ this
    if tau_floor_sec > tau_ceil_sec:
        errors.append(
            f"BLOCKING SECONDARY: t5_floor ({blocking['t5_floor_ms']:.1f} ms → "
            f"tau≥{tau_floor_sec*1000:.2f}ms) conflicts with "
            f"fc_floor ({blocking['fc_floor_hz']:.1f} Hz → "
            f"tau≤{tau_ceil_sec*1000:.2f}ms). Secondary window is inverted."
        )

    # ---- Quantitative vs. Quantitative: Bloom windows ----
    bloom_levels_used = set(m.get('bloom', False) for m in modes.values()) - {False}
    bloom_dicts = {"light": bloom_light, "heavy": bloom_heavy}
    bloom_tau_floors = {}

    for level in bloom_levels_used:
        bd = bloom_dicts[level]
        label = f"BLOOM_{level.upper()}"
        bt_ceil = bd['t5_ms'] / 1000.0 / 5.0
        bt_floor = 1.0 / (2.0 * np.pi * bd['fc_hz'])
        bloom_tau_floors[level] = bt_floor

        if bt_floor > bt_ceil:
            errors.append(
                f"{label}: tau_floor ({bt_floor*1000:.2f} ms, from fc={bd['fc_hz']:.1f} Hz) "
                f"> tau_ceiling ({bt_ceil*1000:.2f} ms, from 5τ={bd['t5_ms']:.1f} ms). "
                f"No valid {level} bloom tau window exists."
            )
        # Each bloom level should be slower than blocking
        if bt_floor < tau_floor:
            errors.append(
                f"{label}: bloom tau_floor ({bt_floor*1000:.2f} ms) is FASTER than "
                f"blocking tau_floor ({tau_floor*1000:.2f} ms). Bloom should be slower "
                f"than blocking for audible compression swell."
            )

    # Light must be faster than heavy (if both are used)
    if "light" in bloom_tau_floors and "heavy" in bloom_tau_floors:
        if bloom_tau_floors["light"] >= bloom_tau_floors["heavy"]:
            errors.append(
                f"BLOOM ORDERING: light tau_floor ({bloom_tau_floors['light']*1000:.2f} ms) "
                f"≥ heavy tau_floor ({bloom_tau_floors['heavy']*1000:.2f} ms). "
                f"Light bloom must be faster than heavy bloom."
            )

    # ---- Qualitative: Descriptor validity ----
    for mode_name, mode_def in modes.items():
        for stage in ['Q1', 'Q2']:
            desc = mode_def.get(stage)
            if desc is None:
                errors.append(
                    f"[{mode_name}] Missing descriptor for {stage}. "
                    f"Must be one of: {_VALID_Q_DESCRIPTORS}"
                )
            elif desc not in _VALID_Q_DESCRIPTORS:
                errors.append(
                    f"[{mode_name}] Invalid descriptor '{desc}' for {stage}. "
                    f"Must be one of: {_VALID_Q_DESCRIPTORS}"
                )

    # ---- Qualitative vs. Qualitative: Q3 prohibition ----
    for mode_name, mode_def in modes.items():
        if 'Q3' in mode_def:
            errors.append(
                f"[{mode_name}] Q3 descriptor specified ('{mode_def['Q3']}'). "
                f"Q3 is a source follower — VGS swing is only (1−Av)×signal ≈ 5%. "
                f"Neither gate conduction (+0.6V) nor triode operation is reachable "
                f"within supply headroom. Q3 is always linear. "
                f"Remove Q3 from the mode table."
            )

    # ---- Qualitative vs. Qualitative: 'hard' requires per-mode RS ----
    hard_stages = {}
    linear_soft_stages = {}
    for mode_name, mode_def in modes.items():
        for stage in ['Q1', 'Q2']:
            desc = mode_def.get(stage, 'linear')
            if desc == 'hard':
                hard_stages.setdefault(stage, []).append(mode_name)
            else:
                linear_soft_stages.setdefault(stage, []).append(mode_name)

    for stage, hard_modes in hard_stages.items():
        if stage in linear_soft_stages:
            ls_modes = linear_soft_stages[stage]
            errors.append(
                f"{stage}:hard in {hard_modes} requires RS ≈ 0Ω (VGS quiescent near 0V). "
                f"But {stage}:linear/soft in {ls_modes} requires RS = "
                f"{jfet_model.self_bias_point(0.3)['rs']:.0f}Ω "
                f"(VGS = {jfet_model.self_bias_point(0.3)['vgs']:.2f}V). "
                f"RS_{stage[1]} must become per-mode switched (currently global). "
                f"This adds RS_{stage[1]} to the rotary switch component set."
            )

    # ---- Qualitative vs. Quantitative: volume ratio feasibility ----
    if volume_ratio <= 0:
        errors.append(
            f"VOLUME_RATIO must be > 0, got {volume_ratio}. "
            f"This is the ratio VPA_out/VPA_in."
        )
    if jfet_model is not None and volume_ratio > 50:
        # Rough cascade gain upper bound
        bp = jfet_model.self_bias_point(0.3)
        gm = bp['gm']
        rd_max = (vdd - jfet_model.vp_abs) / bp['id']
        gain_max = (rd_max / (bp['rs'] + 1.0/gm)) ** 2  # Two stages
        errors.append(
            f"VOLUME_RATIO={volume_ratio} may be unreachable. "
            f"Maximum theoretical two-stage cascade gain ≈ {gain_max:.0f}×. "
            f"The shunt network can only attenuate, not amplify."
        ) if volume_ratio > gain_max else None

    # ---- Qualitative vs. Quantitative: 'hard' + volume_ratio ----
    for mode_name, mode_def in modes.items():
        for stage in ['Q1', 'Q2']:
            if mode_def.get(stage) == 'hard' and jfet_model is not None:
                # Gate clipping compresses output — check if gain is sufficient
                _, bp, onset = jfet_model.compute_hard_bias(vdd)
                if onset > _INPUT_AMPLITUDE:
                    errors.append(
                        f"[{mode_name}] {stage}:hard — gate conduction onset at "
                        f"{onset*1000:.0f} mV signal, but input peak is only "
                        f"{_INPUT_AMPLITUDE*1000:.0f} mV. Gate junction will never "
                        f"conduct. Increase input amplitude or use 'soft' instead."
                    )

    # ---- Qualitative vs. Quantitative: bloom without clipping ----
    for mode_name, mode_def in modes.items():
        if mode_def.get('bloom', False):
            q1_desc = mode_def.get('Q1', 'linear')
            q2_desc = mode_def.get('Q2', 'linear')
            bloom_level = mode_def['bloom']
            if q1_desc == 'linear' and q2_desc == 'linear':
                errors.append(
                    f"[{mode_name}] bloom='{bloom_level}', but Q1:linear and Q2:linear. "
                    f"Bloom requires upstream clipping to shift the DC operating "
                    f"point — without clipping, the coupling cap never charges "
                    f"asymmetrically, so there's no bias recovery to create the "
                    f"compression swell. Set at least one of Q1/Q2 to 'soft' or 'hard'."
                )

    # Filter out None entries (from conditional appends)
    errors = [e for e in errors if e is not None]

    if errors:
        header = f"\n{'='*72}\n  INPUT VALIDATION FAILED — {len(errors)} error(s)\n{'='*72}\n"
        body = "\n\n".join(f"  ERROR {i+1}: {e}" for i, e in enumerate(errors))
        raise ConfigError(header + body + "\n")

    return True


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
        # Bug 1 fix: forced input node (v_ideal) is not in the active node
        # state vector — update its v_prev_all slot explicitly so that the
        # inductor c_p stamp (v_ind_prev) sees the correct previous-step value.
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

    def solve_dc_bias(self, input_node="v_ideal", seed_voltages=None):
        """
        Solve DC operating point via least-squares KCL.

        Args:
            input_node:    Name of the input stimulus node (forced to 0V DC).
            seed_voltages: Optional dict {node_name: voltage} of physics-derived
                           initial guesses.  When provided, these replace the
                           default VDD/2 guess for specific nodes, guaranteeing
                           the solver approaches the SATURATION root (not the
                           spurious triode root that exists at VDS ≈ 0.89V).

                           Typical seeds from _resolve_circuit_parameters:
                             S1 = Id1 × Rs1 (source voltage Q1)
                             D1 = VDD − Id1 × Rd1 (drain voltage Q1)
                             S2 = Id2 × Rs2 (source voltage Q2)
                             D2 = VDD − Id2 × Rd2 (drain voltage Q2)
                             G3 = VDD/2 (V_REF through R7)
                             S3 = Q3 source from Shockley
                             S_CS = Id_Q4 × Rs_CS
        """
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
                i_d, _, i_g, _, _ = self._jfet_physics(j, v_gs, v_ds)
                if j.node_d in self.node_map: residuals[self.node_map[j.node_d]] += i_d
                if j.node_s in self.node_map: residuals[self.node_map[j.node_s]] -= (i_d + i_g)
                if j.node_g in self.node_map: residuals[self.node_map[j.node_g]] += i_g
            if input_node in self.node_map:
                residuals[self.node_map[input_node]] = v_guess[self.node_map[input_node]] - 0.0
            return residuals * 1e3

        # --- Initial guess: physics-derived seeds or fallback ---
        v_guess_init = np.ones(dim) * (self.v_dd_ideal / 2.0)
        if "+" in self.node_map:
            v_guess_init[self.node_map["+"]] = self.v_dd_ideal

        if seed_voltages is not None:
            # Use physics-derived voltages from _resolve_circuit_parameters.
            # These guarantee the solver approaches the saturation root.
            for node_name, voltage in seed_voltages.items():
                if node_name in self.node_map:
                    v_guess_init[self.node_map[node_name]] = voltage
        else:
            # Legacy fallback — crude guesses (prone to triode root trap)
            if "D1" in self.node_map:
                v_guess_init[self.node_map["D1"]] = self.v_dd_ideal * 0.75
            if "D2" in self.node_map:
                v_guess_init[self.node_map["D2"]] = self.v_dd_ideal * 0.75
            if "G3" in self.node_map:
                v_guess_init[self.node_map["G3"]] = 0.0
            if "S3" in self.node_map:
                v_guess_init[self.node_map["S3"]] = 1.0

        sol = least_squares(kcl_equations, v_guess_init,
                            bounds=(-60.0, self.v_dd_ideal + 5.0), method='trf')
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

    def solve_transient(self, input_node, monitor_nodes, freqs, amplitude=_INPUT_AMPLITUDE, periods=20.0, samples_per_period=4096, use_saved_state=False, v_in_override=None):
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
        if v_in_override is not None:
            # Use pre-built input waveform (e.g., envelope-shaped pick attack)
            if len(v_in_override) >= total_samples:
                v_in_array = v_in_override[:total_samples]
            else:
                # Pad with last value if override is slightly too short
                v_in_array = np.zeros(total_samples)
                v_in_array[:len(v_in_override)] = v_in_override
                v_in_array[len(v_in_override):] = v_in_override[-1]
        else:
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
        # Bug 1 fix: dedicated scalar for the forced input node's previous-step
        # voltage.  v_ideal is excluded from active nodes and the state vector,
        # so we track it explicitly rather than relying on v_prev[input_node].
        v_prev_input = [self.dc_op.get(input_node, 0.0)]
        
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
                # Use dedicated scalar for forced input node (Bug 1 fix)
                vpn1 = v_prev_input[0] if l.node1 == input_node else v_prev[l.node1]
                vpn2 = v_prev_input[0] if l.node2 == input_node else v_prev[l.node2]
                i_c_p = (l.c_p / dt) * (v_ind - (vpn1 - vpn2))
                if l.node1 in node_map: residuals[node_map[l.node1]] += (i_l_series + i_c_p)
                if l.node2 in node_map: residuals[node_map[l.node2]] -= (i_l_series + i_c_p)
            for c in self.capacitors:
                # For capacitors: use dedicated scalar if a terminal is the forced input
                cpn1 = v_prev_input[0] if c.node1 == input_node else v_prev[c.node1]
                cpn2 = v_prev_input[0] if c.node2 == input_node else v_prev[c.node2]
                i_c = ((v[c.node1]-v[c.node2])-(cpn1 - cpn2)) / (c.esr + dt/c.value)
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
            # Update v_prev for active nodes and the forced input scalar
            v_prev[input_node] = self.dc_op.get(input_node, 0.0) + v_inst
            v_prev_input[0] = v_prev[input_node]  # Bug 1 fix: sync scalar
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
        seeds = getattr(self, '_seed_voltages', None)
        dc_results = self.circuit.solve_dc_bias(input_node=self.input_node,
                                                 seed_voltages=seeds)
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
            seeds = getattr(self, '_seed_voltages', None)
            self.circuit.solve_dc_bias(input_node=self.input_node,
                                        seed_voltages=seeds)
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

    def _render_plot(self, mode, start_idx, window_ms, title_suffix,
                     filename, subtract_dc=True):
        """Core plot renderer — called once per output image."""
        sys_tau = self.get_max_system_tau()

        fig = plt.figure(figsize=(16, 16))
        plt.suptitle(f"{mode} Mode: Signal Chain and Harmonic Analysis ({title_suffix})", fontweight='bold', fontsize=14)
        gs = gridspec.GridSpec(7, 2, figure=fig, width_ratios=[1, 1])
        
        t_sliced = self.t[start_idx:]
        t_plot_ms = (t_sliced - t_sliced[0]) * 1000.0
        
        nodes_to_plot = ["v_ideal", "G1", "D1", "D2", "S3", "BUF", "OUT"]
        titles = [
            "Pre-Pickup Source (v_ideal)",
            "Pre-Q1 Gate Input (G1)",
            "Q1 Drain Output (D1)",
            "Q2 Drain Output (D2)",
            "Q3/Q4 Follower Output (S3)",
            "OPA1656 Parallel Buffer Output (BUF)",
            "Final Output (OUT) — post 10Ω||10Ω isolation"
        ]
        colors = ['gray', 'green', 'blue', 'purple', 'teal', 'orange', 'black']
        
        for i, (node, title, color) in enumerate(zip(nodes_to_plot, titles, colors)):
            ax = fig.add_subplot(gs[i, 0])
            if node not in self.v_out_data:
                ax.text(0.5, 0.5, f"Node '{node}' not in transient data",
                        ha='center', va='center', transform=ax.transAxes)
                ax.set_title(title, fontsize=11)
                continue

            v_sliced = self.v_out_data[node][start_idx:].copy()

            if node in ["D1", "D2", "S3", "BUF"] and subtract_dc:
                v_sliced = v_sliced - np.mean(v_sliced)
                ax.set_ylabel("AC Amplitude (V)", fontsize=9)
            elif node in ["D1", "D2", "S3", "BUF"] and not subtract_dc:
                ax.set_ylabel("Voltage (V)", fontsize=9)
            else:
                ax.set_ylabel("Amplitude (V)", fontsize=9)
                
            ax.plot(t_plot_ms, v_sliced, color=color, linewidth=1.5)
            ax.set_title(title, fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, window_ms)
            if i == 6:
                ax.set_xlabel("Time (ms)", fontsize=10)
            else:
                ax.set_xticklabels([])
                
        ax_fft = fig.add_subplot(gs[:, 1])

        # FFT always on steady-state data (skip 10τ) regardless of plot window
        fft_start_idx = int(np.round((10.0 * sys_tau) / self.circuit.dt))
        v_out_raw = self.v_out_data["OUT"][fft_start_idx:]
        v_out_ac = v_out_raw - np.mean(v_out_raw)

        N = len(v_out_ac)
        Y = np.fft.rfft(v_out_ac * np.hanning(N))
        xf = np.fft.rfftfreq(N, d=self.circuit.dt)
        mag = (2.0/N * np.abs(Y)) + 1e-12 

        weighted_mag = mag * _a_weight(xf)

        ax_fft.fill_between(xf[1:], weighted_mag[1:], 1e-9, color='gray', alpha=0.15)
        ax_fft.plot(xf[1:], weighted_mag[1:], color='gray', alpha=0.3, linewidth=0.5)

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
            
        ax_fft.set_title("A-Weighted Harmonic Spectrum (OUT)", fontsize=12)
        ax_fft.set_xlabel("Frequency (Hz) [Log Scale]", fontsize=10)
        ax_fft.set_ylabel("A-Weighted Level (V_w)", fontsize=10)
        ax_fft.set_xscale('log')
        ax_fft.set_yscale('log')
        ax_fft.set_xlim(50, 10000)
        ax_fft.set_ylim(1e-6, weighted_mag[1:].max() * 2.0)
        ax_fft.grid(True, which="both", alpha=0.2)
        
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    def plot_waveforms(self, mode, mode_def=None):
        """
        Plot signal chain waveforms and harmonic spectrum.

        Non-bloom modes: one plot — steady-state extract (~40ms)
        Bloom modes:     two plots — bloom recovery transient + steady-state
        """
        bloom_active = mode_def.get('bloom', False) if mode_def else False
        sys_tau = self.get_max_system_tau()
        ss_start_idx = int(np.round((10.0 * sys_tau) / self.circuit.dt))

        # Plot 1 (all modes): steady-state
        self._render_plot(mode, ss_start_idx, window_ms=40.0,
                          title_suffix="Steady State Extract",
                          filename=f'mode_{mode}_analysis.png',
                          subtract_dc=True)

        # Plot 2 (bloom modes only): pick attack → bloom recovery
        if bloom_active:
            print(f"  [{mode}] Running bloom attack simulation...")
            self.run_bloom_attack(attack_mult=2.5, attack_cycles=5)

            # Build bloom recovery plot from attack sim data
            t_ms = (self.bloom_t - self.bloom_t[0]) * 1000.0
            window_ms = t_ms[-1]

            fig = plt.figure(figsize=(16, 18))
            plt.suptitle(f"{mode} Mode: Bloom({bloom_active}) — Pick Attack → Recovery",
                         fontweight='bold', fontsize=14)
            gs = gridspec.GridSpec(8, 1, figure=fig)

            # Row 0: Input envelope showing the attack spike
            ax_env = fig.add_subplot(gs[0])
            ax_env.plot(t_ms, self.bloom_v_in, color='gray', linewidth=1.0, alpha=0.6)
            ax_env.fill_between(t_ms, self.bloom_v_in, 0, color='gray', alpha=0.1)
            ax_env.axvline(self.bloom_attack_end_ms, color='red', ls='--', alpha=0.5, label='Attack ends')
            ax_env.axvline(self.bloom_decay_end_ms, color='orange', ls='--', alpha=0.5, label='Decay ends')
            ax_env.set_title("Input Signal (pick attack envelope)", fontsize=11)
            ax_env.set_ylabel("V", fontsize=9)
            ax_env.legend(fontsize=8, loc='upper right')
            ax_env.grid(True, alpha=0.3)
            ax_env.set_xlim(0, window_ms)
            ax_env.set_xticklabels([])

            # Rows 1–7: signal chain nodes (DC-coupled to show bias shift)
            nodes_to_plot = ["G1", "D1", "D2", "G3", "S3", "BUF", "OUT"]
            titles = [
                "Pre-Q1 Gate Input (G1)",
                "Q1 Drain Output (D1)",
                "Q2 Drain Output (D2)",
                "Q3 Gate / G3 Network (G3) — bloom node",
                "Q3/Q4 Follower Output (S3)",
                "OPA1656 Buffer Output (BUF)",
                "Final Output (OUT)"
            ]
            colors = ['green', 'blue', 'purple', 'red', 'teal', 'orange', 'black']

            for i, (node, title, color) in enumerate(zip(nodes_to_plot, titles, colors)):
                ax = fig.add_subplot(gs[i + 1])
                if node not in self.bloom_data:
                    ax.text(0.5, 0.5, f"Node '{node}' not in bloom data",
                            ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(title, fontsize=11)
                    continue

                v = self.bloom_data[node][:len(t_ms)]
                ax.plot(t_ms, v, color=color, linewidth=1.0)
                ax.axvline(self.bloom_attack_end_ms, color='red', ls='--', alpha=0.3)
                ax.axvline(self.bloom_decay_end_ms, color='orange', ls='--', alpha=0.3)
                ax.set_title(title, fontsize=11)
                ax.set_ylabel("Voltage (V)", fontsize=9)
                ax.grid(True, alpha=0.3)
                ax.set_xlim(0, window_ms)
                if i == 6:
                    ax.set_xlabel("Time (ms)", fontsize=10)
                else:
                    ax.set_xticklabels([])

            plt.tight_layout()
            plt.savefig(f'mode_{mode}_bloom_recovery.png')
            plt.close()
            print(f"  [{mode}] Bloom plot saved: mode_{mode}_bloom_recovery.png")

            # Export bloom-specific audio: steady → attack → recovery → steady
            self.export_bloom_audio(mode=mode)

    def export_audio(self, mode, target_duration_sec=4.0, target_sr=44100):
        """
        Export audio using zero-crossing-aligned period extraction with
        fold-and-crossfade seamless looping.

        Strategy:
          1. Compute the GCD-based true period (deterministic repeat length).
          2. Find rising zero crossings in the steady-state output.
          3. Extract one period plus ~15% overlap extension past the second ZC.
          4. Resample to 44.1 kHz.
          5. Fold the overlap tail back onto the head with a raised-cosine
             crossfade — the tile's last sample flows seamlessly into its
             first sample, eliminating slope discontinuity at the seam.
          6. Tile to target duration.

        The result loops with zero clicks because the crossfade zone ensures
        continuity in both amplitude and derivative at every tile boundary.
        """
        from math import gcd
        from functools import reduce

        print(f"--- Exporting Audio: {mode} Mode ---")
        start_idx = int(np.round((10.0 * self.get_max_system_tau()) / self.circuit.dt))
        # Guard: ensure at least 2 periods of data remain after settling skip
        min_samples = int(np.round(2.0 / self.freqs[0] / self.circuit.dt))
        if start_idx > len(self.v_out_data["OUT"]) - min_samples:
            start_idx = max(0, len(self.v_out_data["OUT"]) - min_samples)
        v_ss = self.v_out_data["OUT"][start_idx:].copy()
        v_ss -= np.mean(v_ss)

        # True repeat period from input frequencies (deterministic)
        freq_ints = [max(1, int(round(f))) for f in self.freqs]
        f_repeat = reduce(gcd, freq_ints)
        true_period_sec = 1.0 / f_repeat
        period_samples_sim = int(np.round(true_period_sec / self.circuit.dt))

        print(f"  Input freqs GCD = {f_repeat} Hz → true period = {true_period_sec*1000:.2f} ms")

        one_period_ext_sim = self._extract_zc_period(v_ss, period_samples_sim)

        # The extract includes a ~15% overlap extension past the second ZC.
        # Resample the whole extended segment to 44.1 kHz, then fold-and-crossfade.
        actual_period_sec = len(one_period_ext_sim) * self.circuit.dt
        ext_samples_44k = max(1, int(np.round(actual_period_sec * target_sr)))
        ext_44k = resample(one_period_ext_sim, ext_samples_44k)
        ext_44k -= np.mean(ext_44k)  # Remove any DC from resampling

        # Determine the period length at 44.1 kHz (before extension)
        # The extension fraction matches what _extract_zc_period used
        period_frac = period_samples_sim / len(one_period_ext_sim)
        period_samples_44k = max(1, int(np.round(ext_samples_44k * period_frac)))

        # Fold-and-crossfade: creates a tile that loops with zero discontinuity
        one_period_44k = self._make_seamless_loop(ext_44k, period_samples_44k)

        # Tile to target duration
        exact_samples = int(np.round(target_duration_sec * target_sr))
        tiles_needed = int(np.ceil(exact_samples / len(one_period_44k)))
        full_wave = np.tile(one_period_44k, tiles_needed)[:exact_samples]

        max_val = np.max(np.abs(full_wave))
        normalized = (full_wave / max_val) if max_val > 0 else full_wave
        wavfile.write(f'mode_{mode}_audio.wav', target_sr, np.int16(normalized * 32767))
        actual_period_ms = period_samples_44k / target_sr * 1000
        print(f"  Exported: {target_duration_sec:.1f}s @ {target_sr}Hz | "
              f"period: {period_samples_44k} samples ({actual_period_ms:.2f}ms) | "
              f"crossfade: {ext_samples_44k - period_samples_44k} samples")

    @staticmethod
    def _extract_zc_period(v_ss, period_samples_target, extend_frac=0.15):
        """
        Extract one period from steady-state data, aligned to rising zero crossings,
        PLUS an overlap extension for seamless-loop crossfading.

        Returns an array of length ≈ period + extend, where:
          - [:period]  = one true period (ZC-aligned)
          - [period:]  = overlap extension (continuation past the second ZC)

        The extension fraction (default 15% of the period) provides enough
        material for fold-and-crossfade looping.

        Falls back to the last period_samples_target samples (no extension)
        if fewer than 2 rising ZCs are found.
        """
        extend_samples = max(16, int(period_samples_target * extend_frac))

        # Find all rising zero crossings: v[i] <= 0 and v[i+1] > 0
        signs = np.sign(v_ss)
        # Treat exact zeros as negative so the crossing is detected at the transition
        signs[signs == 0] = -1
        rising_zc = np.where((signs[:-1] < 0) & (signs[1:] > 0))[0]

        if len(rising_zc) < 2:
            # Fallback: no ZC alignment possible
            return v_ss[-period_samples_target:]

        # Search backward from the last rising ZC for a pair ~1 period apart.
        # Allow ±5% tolerance on the period length.
        # Also require enough headroom past the second ZC for the extension.
        tol = 0.05
        lo = int(period_samples_target * (1.0 - tol))
        hi = int(period_samples_target * (1.0 + tol))

        best_pair = None
        best_err = float('inf')
        for j in range(len(rising_zc) - 1, 0, -1):
            zc_end = rising_zc[j]
            # Need extend_samples of headroom past zc_end
            if zc_end + extend_samples >= len(v_ss):
                continue
            for k in range(j - 1, -1, -1):
                span = zc_end - rising_zc[k]
                if span > hi:
                    break  # k decreasing → span only grows; stop searching
                if lo <= span <= hi:
                    err = abs(span - period_samples_target)
                    if err < best_err:
                        best_err = err
                        best_pair = (rising_zc[k], zc_end)
            if best_pair is not None:
                break  # Take the first good match from the end (most settled)

        if best_pair is None:
            # Wider search: take any two rising ZCs closest to target period
            for j in range(len(rising_zc) - 1, 0, -1):
                zc_end = rising_zc[j]
                if zc_end + extend_samples >= len(v_ss):
                    continue
                for k in range(j - 1, -1, -1):
                    span = rising_zc[j] - rising_zc[k]
                    err = abs(span - period_samples_target)
                    if err < best_err:
                        best_err = err
                        best_pair = (rising_zc[k], rising_zc[j])
                    if span > period_samples_target * 2:
                        break
                if best_pair is not None and best_err < period_samples_target * 0.1:
                    break

        if best_pair is None:
            return v_ss[-period_samples_target:]

        zc_start, zc_end = best_pair
        # Return period + overlap extension
        end_with_ext = min(zc_end + extend_samples, len(v_ss))
        return v_ss[zc_start:end_with_ext]

    @staticmethod
    def _make_seamless_loop(extended, period_len):
        """
        Fold-and-crossfade: turn an extended period into a perfectly seamless loop tile.

        Given:
          extended[:period_len]   — one full period
          extended[period_len:]   — overlap continuation past the end ZC

        The overlap is the waveform's natural continuation.  When the tile repeats,
        sample [period_len-1] is followed by sample [0] — a slope discontinuity.

        Fix: fold the tail (overlap) back onto the head with a raised-cosine
        crossfade.  After folding, sample [0] IS the natural continuation, and
        it smoothly blends back into the original waveform over the fade zone.

        Returns a tile of exactly `period_len` samples that loops seamlessly.
        """
        fade_len = len(extended) - period_len
        if fade_len <= 0:
            # No extension available — return as-is (degenerate case)
            return extended[:period_len].copy()

        tile = extended[:period_len].copy()
        tail = extended[period_len:]  # natural continuation past one period

        # Raised-cosine crossfade (smoother than linear at the edges)
        t = np.linspace(0, np.pi, fade_len)
        fade_out = 0.5 * (1.0 + np.cos(t))   # 1 → 0
        fade_in  = 1.0 - fade_out              # 0 → 1

        # Fold tail onto head:
        #   tile[0] becomes tail[0] (the natural continuation) — seamless at the seam
        #   tile[fade_len-1] stays ≈ original — seamless into the body
        tile[:fade_len] = tile[:fade_len] * fade_in + tail * fade_out

        return tile

    def run_bloom_attack(self, attack_mult=2.5, attack_cycles=5):
        """
        Simulate a hard pick attack followed by normal sustain to show bloom.

        Phase 1 (settle):  normal amplitude for 10τ — re-establish steady state
        Phase 2 (plotted): attack burst → exponential decay → normal sustain
                           through 12τ recovery — this is what gets plotted

        The bloom effect is the slow compression swell visible in the
        sustain phase as the coupling caps recover from the attack.
        """
        f_base = self.freqs[0]
        sys_tau = self.get_max_system_tau()
        spp = 2048
        dt = (1.0 / f_base) / spp

        # Phase 1: settle at normal amplitude (uses existing DC bias)
        settle_periods = int(np.round(10.0 * sys_tau * f_base))
        self.circuit.solve_transient(
            input_node=self.input_node, monitor_nodes=self.monitor_nodes,
            freqs=self.freqs, amplitude=self.amplitude,
            periods=max(settle_periods, 1), samples_per_period=spp)

        # Phase 2: attack + decay + sustain
        # Compute periods and sample count exactly as solve_transient will
        t_attack  = attack_cycles / f_base
        t_decay   = 2.0 / f_base
        t_sustain = 12.0 * sys_tau
        t_phase2  = t_attack + t_decay + t_sustain
        phase2_periods = int(np.round(t_phase2 * f_base))
        phase2_periods = max(phase2_periods, 1)
        phase2_samples = int(np.round(phase2_periods * spp))

        # Build time array matching solve_transient's convention
        t_local = np.arange(phase2_samples) * dt

        # Build base multi-tone signal
        v_base = np.zeros(phase2_samples)
        for f in self.freqs:
            v_base += (self.amplitude / len(self.freqs)) * np.sin(2 * np.pi * f * t_local)

        # Build amplitude envelope
        envelope = np.ones(phase2_samples)
        idx_attack_end = int(np.round(t_attack / dt))
        idx_decay_end  = int(np.round((t_attack + t_decay) / dt))

        # Attack phase: hard pick
        envelope[:idx_attack_end] = attack_mult
        # Decay phase: exponential ramp from attack_mult → 1.0
        n_decay = idx_decay_end - idx_attack_end
        if n_decay > 0:
            envelope[idx_attack_end:idx_decay_end] = 1.0 + (attack_mult - 1.0) * np.exp(
                -3.0 * np.linspace(0, 1, n_decay))
        # Sustain phase: already 1.0

        v_in_shaped = v_base * envelope

        # Run phase 2 with shaped input
        t_bloom, _, v_bloom_data = self.circuit.solve_transient(
            input_node=self.input_node, monitor_nodes=self.monitor_nodes,
            freqs=self.freqs, amplitude=self.amplitude,
            periods=phase2_periods, samples_per_period=spp,
            use_saved_state=True,
            v_in_override=v_in_shaped)

        # Store for plotting
        self.bloom_t = t_bloom
        self.bloom_v_in = v_in_shaped[:len(t_bloom)]
        self.bloom_data = v_bloom_data
        self.bloom_attack_end_ms = t_attack * 1000.0
        self.bloom_decay_end_ms = (t_attack + t_decay) * 1000.0

        print(f"  Bloom attack sim: {attack_mult:.1f}× for {attack_cycles} cycles | "
              f"recovery window: {t_sustain*1000:.0f}ms")

    def export_bloom_audio(self, mode, target_sr=44100):
        """
        Export bloom-specific audio: steady → attack → recovery → steady.

        Structure:
          ~1 second  : pre-attack steady-state (tiled true period)
          variable   : full bloom transient (attack → decay → recovery)
          ~1 second  : post-recovery steady-state (tiled true period)

        Total duration varies per mode based on recovery time.
        Seamless at both splice points — fold-and-crossfade looped tiles.
        """
        from math import gcd
        from functools import reduce

        if not hasattr(self, 'bloom_data') or "OUT" not in self.bloom_data:
            print(f"  [{mode}] No bloom sim data — skipping bloom audio export")
            return

        print(f"--- Exporting Bloom Audio: {mode} Mode ---")

        # True period from known input frequencies (same as export_audio)
        freq_ints = [max(1, int(round(f))) for f in self.freqs]
        f_repeat = reduce(gcd, freq_ints)
        true_period_sec = 1.0 / f_repeat

        dt_sim = self.circuit.dt
        period_samples_sim = int(np.round(true_period_sec / dt_sim))

        # --- Pre-attack steady-state period (from main transient) ---
        sys_tau = self.get_max_system_tau()
        ss_start = int(np.round((10.0 * sys_tau) / dt_sim))
        v_ss = self.v_out_data["OUT"][ss_start:].copy()
        v_ss -= np.mean(v_ss)

        one_pre_ext_sim = self._extract_zc_period(v_ss, period_samples_sim)
        actual_pre_sec = len(one_pre_ext_sim) * dt_sim
        pre_ext_44k_len = max(1, int(np.round(actual_pre_sec * target_sr)))
        pre_ext_44k = resample(one_pre_ext_sim, pre_ext_44k_len)
        pre_ext_44k -= np.mean(pre_ext_44k)

        # Seamless loop tile (fold-and-crossfade)
        pre_period_frac = period_samples_sim / len(one_pre_ext_sim)
        pre_period_44k = max(1, int(np.round(pre_ext_44k_len * pre_period_frac)))
        one_pre_44k = self._make_seamless_loop(pre_ext_44k, pre_period_44k)

        # Tile for ~1 second
        pre_tiles = int(np.ceil(target_sr / len(one_pre_44k)))
        pre_section = np.tile(one_pre_44k, pre_tiles)[:target_sr]

        # --- Bloom transient (full attack → decay → recovery) ---
        v_bloom_out = self.bloom_data["OUT"].copy()
        v_bloom_out -= np.mean(v_bloom_out[-period_samples_sim:])  # DC-remove using post-recovery mean
        bloom_dt = (self.bloom_t[-1] - self.bloom_t[0]) / (len(self.bloom_t) - 1)
        bloom_samples_44k = max(1, int(np.round(len(v_bloom_out) * bloom_dt * target_sr)))
        bloom_section = resample(v_bloom_out, bloom_samples_44k)

        # --- Post-recovery steady-state period (from end of bloom sim) ---
        post_period_samples = int(np.round(true_period_sec / bloom_dt))
        v_bloom_tail = v_bloom_out.copy()
        v_bloom_tail -= np.mean(v_bloom_tail)

        one_post_ext_sim = self._extract_zc_period(v_bloom_tail, post_period_samples)
        actual_post_sec = len(one_post_ext_sim) * bloom_dt
        post_ext_44k_len = max(1, int(np.round(actual_post_sec * target_sr)))
        post_ext_44k = resample(one_post_ext_sim, post_ext_44k_len)
        post_ext_44k -= np.mean(post_ext_44k)

        # Seamless loop tile (fold-and-crossfade)
        post_period_frac = post_period_samples / len(one_post_ext_sim)
        post_period_44k = max(1, int(np.round(post_ext_44k_len * post_period_frac)))
        one_post_44k = self._make_seamless_loop(post_ext_44k, post_period_44k)

        # Tile for ~1 second
        post_tiles = int(np.ceil(target_sr / len(one_post_44k)))
        post_section = np.tile(one_post_44k, post_tiles)[:target_sr]

        # --- Assemble: pre + bloom transient + post ---
        full_wave = np.concatenate([pre_section, bloom_section, post_section])

        max_val = np.max(np.abs(full_wave))
        normalized = (full_wave / max_val) if max_val > 0 else full_wave
        filename = f'mode_{mode}_bloom_audio.wav'
        wavfile.write(filename, target_sr, np.int16(normalized * 32767))

        total_sec = len(full_wave) / target_sr
        print(f"  Exported: {filename} | {total_sec:.2f}s @ {target_sr}Hz")
        print(f"  Structure: ~1.0s steady → {bloom_samples_44k/target_sr:.2f}s transient → ~1.0s steady")

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


def _compute_input_vpa(amplitude, freqs):
    """
    Analytical VPA of a multi-tone input signal (pure sines, no distortion).

    Each tone has amplitude = amplitude / n_tones, matching solve_transient's
    amplitude-splitting convention.  For pure sines, harmonics H2+ are zero,
    so VPA = RSS of A-weighted fundamentals only.

    This replaces the old hardcoded _INPUT_VPA_EST = 0.175 with a value
    derived from the actual signal parameters and A-weighting curve.
    """
    n = len(freqs)
    amp_per_tone = amplitude / n
    vpa_sq = sum((amp_per_tone * _a_weight(f)) ** 2 for f in freqs)
    return np.sqrt(vpa_sq)


# --- Parallelized Search Architecture ---
#
# Picklable module-level functions:
#   _eval_circuit_from_config  — builds circuit from a serializable config dict
#   _compute_r7_mid            — deterministic R7 from tau midpoint
#   _eval_cap_worker           — evaluates one (config, cap, R7) + blocking validation
#   _resolve_global_resistors  — Rs from Shockley (via LSK489_Model), per-mode Rd from descriptors
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

_MONITOR_NODES = ["v_ideal", "IN", "G1", "D1", "G2", "D2", "G3", "S3", "BUF", "OUT"]
_CAL_FREQ      = [1000.0]

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
        # Tone pot + tone cap: series RC shunt from IN to GND.
        # Together with L_PICKUP, C_P, and R1 they form the pickup resonant system.
        {"type": "Capacitor",  "name": "C_TONE",    "val": 22e-9,                                "n1": "IN",      "n2": "TONE"},
        {"type": "Resistor",   "name": "R_TONE",    "val": config.get("r_tone", 500e3),          "n1": "TONE",    "n2": "-"},
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
        # OPA1656 parallel unity buffer: C4 → BUF (opamp input/output) → R_ISO → OUT
        # BUF node represents the opamp's output (unity gain = input).
        # R_ISO_EFF = R_ISO_A || R_ISO_B = 10Ω || 10Ω = 5Ω (two channels in parallel).
        # The opamp is a LINEAR buffer — it must never clip.  If the signal
        # at BUF approaches the rails, that is a design error upstream.
        {"type": "Capacitor",  "name": "C4",         "val": config["c4"],      "n1": "S3",      "n2": "BUF"},
        {"type": "Resistor",   "name": "R_ISO_EFF",  "val": config.get("r_iso_eff", 5.0),  "n1": "BUF",     "n2": "OUT"},
        {"type": "Resistor",   "name": "VOL_POT",    "val": config.get("r_vol", 500.0e3),  "n1": "OUT",     "n2": "-"},
    ]
    sim = Circuit(v_dd_ideal=18.0, r_psu=10.0)
    process_unified_circuit(sim, core_list)
    cal_freq = config.get("cal_freq", _CAL_FREQ)
    monitor  = config.get("monitor_nodes", _MONITOR_NODES)
    analyzer = CircuitAnalyzer(circuit=sim, monitor_nodes=monitor,
                               input_node="v_ideal", amplitude=_INPUT_AMPLITUDE)

    # --- Compute physics-derived DC bias seeds from config ---
    # These are the design-intent voltages already computed by
    # _resolve_circuit_parameters.  Passing them to solve_dc_bias
    # guarantees the solver finds the saturation root (not the
    # spurious triode root at VDS ≈ 0.89V).
    seed_voltages = None
    if "seed_voltages" in config:
        seed_voltages = config["seed_voltages"]
    elif all(k in config for k in ["vs1", "vs2", "id1", "id2"]):
        vdd = 18.0
        v_ref = vdd / 2.0
        seed_voltages = {
            "S1":   config["vs1"],
            "D1":   vdd - config["id1"] * config["rd1"],
            "S2":   config["vs2"],
            "D2":   vdd - config["id2"] * config["rd2"],
            "G1":   0.0,    # Gate at AC ground through R_g
            "G2":   0.0,    # Gate at AC ground through R_g
            "G3":   v_ref,  # V_REF through R7
            "BUF":  0.0,    # AC-coupled through C4, DC = 0
            "OUT":  0.0,    # AC-coupled through C4+R_ISO, DC = 0
        }
        # Q3/Q4 seeds if available
        if "vs_q3" in config:
            seed_voltages["S3"] = config["vs_q3"]
        if "rs_cs" in config and "id_q4" in config:
            seed_voltages["S_CS"] = config["id_q4"] * config["rs_cs"]
        elif "rs_cs" in config and "vs_q3" in config:
            # Approximate: Q4 current ≈ Q3 current (series)
            pass  # S_CS will use VDD/2 fallback

    if not full_run:
        spp = 2048 if high_res else 512
        analyzer.circuit.solve_dc_bias(input_node="v_ideal",
                                        seed_voltages=seed_voltages)
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

    # full_run: store seeds on analyzer so report_dc_bias and run_transient use them
    analyzer._seed_voltages = seed_voltages
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
    target_vpa = config["target_vpa"]  # Required — set from VOLUME_RATIO in main
    tau_floor  = config["tau_floor"]       # Per-mode (bloom or blocking) — for G3
    tau_ceiling = config["tau_ceiling"]    # Per-mode (bloom or blocking) — for G3
    # Tightened blocking bounds (secondary floors applied) — for global caps C1/C2/C4
    tau_blk_floor   = config.get("tau_blocking_floor", tau_floor)
    tau_blk_ceiling = config.get("tau_blocking_ceiling", tau_ceiling)

    analyzer = _eval_circuit_from_config(config, cap_value, r7_value,
                                          full_run=False, high_res=False)
    vpa = get_vpa_metric(analyzer.v_out_data["OUT"], analyzer.circuit.dt, cal_freq)

    # --- Blocking validation on all coupling caps ---
    blocking_ok = True
    blocking_detail = {}

    # C1, C2, C4: global caps — use tightened blocking bounds
    for cname, n1, n2 in [("C1", "IN", "G1"), ("C2", "D1", "G2"),
                           ("C4", "S3", "BUF")]:
        rth = analyzer.circuit.solve_ac_thevenin(Capacitor(f"BV_{cname}", 1e-9, n1, n2))
        cap_obj = next((c for c in analyzer.circuit.capacitors if c.name == cname), None)
        if cap_obj is None:
            continue
        tau = rth * cap_obj.value
        fc  = 1.0 / (2.0 * np.pi * tau) if tau > 0 else float('inf')
        t5  = 5.0 * tau
        ok  = (tau_blk_floor <= tau <= tau_blk_ceiling)
        blocking_detail[cname] = {"rth": rth, "tau": tau, "fc": fc, "t5": t5, "ok": ok}
        if not ok:
            blocking_ok = False

    # G3 node: total = C3 + C3_shunt — use per-mode bounds (bloom or blocking)
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


def _resolve_global_resistors(idss, vp_abs, vdd,
                               target_fc_hz, target_recovery_ms,
                               mode_table, volume_ratio,
                               jfet_model=None):
    """
    Compute ALL global resistors from LSK489 datasheet + blocking constraints.

    Now accepts a mode_table with per-mode descriptors:
        mode_table = {"Clean": {"Q1": "linear", "Q2": "linear", "bloom": False}, ...}

    Drain resistors are derived from the descriptor:
        'linear': headroom-optimized via sweep (VDS stays well above sat-triode boundary)
        'soft':   VD placed at |VP| — the sat-triode boundary from Shockley equation
                  (datasheet Output Characteristics p4, knee location)

    volume_ratio: VPA_out / VPA_in — used for Q3 headroom pre-check.
                  Actual VPA target is computed from measured input VPA later.

    Solved (same all modes):
        R_g   : Gate bias — largest E24 where C1/C2 have E24 caps in blocking window.
        Rs1/Rs2 : Source resistors from Shockley alpha.
        Rs_CS : Q4 JFET current source — headroom-centered, self-bias solver.

    Per-mode:
        Rd1/Rd2 : Drain resistors from operating region descriptor.
    """
    if jfet_model is None:
        jfet_model = LSK489_Model(idss, vp_abs)
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

    # === Tone pot + tone cap: pickup resonant network ===
    # Series RC from IN to GND.  Together with L_PICKUP, C_P, and R_g
    # they form the pickup's resonant system (LC peak frequency + Q).
    # r_tone is user-adjustable; model at fully open (500 kΩ) as baseline.
    r_tone  = 500e3   # Tone pot (fully open baseline) — user adjustable
    c_tone  = 22e-9   # 22 nF tone cap (fixed hardware)

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

    # === Rs1/Rs2 from Shockley alpha ===
    #
    # Global alpha sets the quiescent operating point for linear/soft modes.
    # If 'hard' appears for a stage in ANY mode, that stage's RS becomes
    # per-mode switched using a parallel-resistor architecture:
    #   RS_base (global, always connected) = normal linear RS
    #   RS_hard_shunt (switched, only for hard mode) = parallel value
    #   Effective = RS_base || RS_hard_shunt ≈ RS_hard_target
    #
    # This ensures no open-circuit moment during rotary switching — the
    # base resistor is always in the circuit, and the shunt only changes
    # the effective value when connected.
    a1_global = 0.30   # Q1: Id~1.65mA, gfs~4mS
    a2_global = 0.25   # Q2: Id~1.375mA

    bp1 = jfet_model.self_bias_point(a1_global)
    vs1, rs1, id1 = bp1['vs'], bp1['rs'], bp1['id']
    bp2 = jfet_model.self_bias_point(a2_global)
    vs2, rs2, id2 = bp2['vs'], bp2['rs'], bp2['id']

    # Check if hard mode is used for either stage
    q1_has_hard = any(m.get("Q1") == "hard" for m in mode_table.values())
    q2_has_hard = any(m.get("Q2") == "hard" for m in mode_table.values())
    rs_is_per_mode = {}  # {stage: True} if RS is per-mode

    per_mode_rs = {}  # {mode_name: (rs1_effective, rs2_effective)}
    rs_hard_shunts = {}  # {stage: rs_shunt_value} for BOM

    for stage, has_hard, rs_global, alpha_global in [
        ("Q1", q1_has_hard, rs1, a1_global),
        ("Q2", q2_has_hard, rs2, a2_global),
    ]:
        if has_hard:
            rs_is_per_mode[stage] = True
            # Hard mode: VGS quiescent near 0V → alpha near 1.0
            _, bp_hard, gate_onset = jfet_model.compute_hard_bias(vdd)
            rs_hard_target = bp_hard['rs']

            # Round hard RS target to nearest E24
            valid_rs = e24_resistors[e24_resistors > 0]
            rs_hard_e24 = float(valid_rs[np.argmin(np.abs(valid_rs - rs_hard_target))])

            # Parallel shunt: RS_base || RS_shunt = RS_hard_e24
            # RS_shunt = 1 / (1/RS_hard_e24 - 1/RS_base)
            if rs_hard_e24 < rs_global:
                rs_shunt = 1.0 / (1.0 / rs_hard_e24 - 1.0 / rs_global)
                rs_shunt_e24 = float(valid_rs[np.argmin(np.abs(valid_rs - rs_shunt))])
                # Actual effective with E24 shunt
                rs_hard_actual = (rs_global * rs_shunt_e24) / (rs_global + rs_shunt_e24)
                rs_hard_shunts[stage] = rs_shunt_e24
                print(f"[GLOBAL] {stage}:hard detected — RS becomes per-mode switched:")
                print(f"  RS_base (always connected): {rs_global:.1f}Ω")
                print(f"  RS_hard_shunt (switched in for hard): {rs_shunt_e24:.1f}Ω (E24)")
                print(f"  Effective {stage} RS in hard mode: "
                      f"{rs_global:.1f}Ω || {rs_shunt_e24:.1f}Ω = {rs_hard_actual:.1f}Ω")
                print(f"  Gate conduction onset: signal peak > {gate_onset*1000:.0f}mV")
            else:
                # Hard RS is larger than global — shouldn't happen normally
                rs_hard_actual = rs_hard_e24
                rs_hard_shunts[stage] = None
                print(f"  WARNING: {stage}:hard RS ({rs_hard_e24:.1f}Ω) ≥ "
                      f"global RS ({rs_global:.1f}Ω). No parallel shunt needed — "
                      f"but this means VGS is already near 0V in all modes.",
                      flush=True)
        else:
            rs_is_per_mode[stage] = False

    # === Per-mode Rd from operating region descriptors ===
    #
    # The saturation-triode boundary (the "knee" in the Output Characteristics
    # plot, datasheet p4) is at VDS = VGS − VP.  For self-biased stages:
    #   VDS_boundary = |VP| × √α
    #   VD at boundary = VS + VDS_boundary = |VP|  (always, for any alpha)
    #
    # 'linear': VD placed to maximize headroom above |VP|, subject to:
    #   1. Cascade gain × input peak > minimum useful D2 signal
    #   2. D2 peak fits within drain rails
    #   3. Q3 follower can pass the output without clipping
    #
    # 'soft': VD placed at |VP| — the boundary itself.  Signal peaks
    #   push into the triode region of the Output Characteristics.
    #   RD = (VDD − |VP|) / ID.  Deterministic, no sweep.

    gm1 = jfet_model.transconductance(
        jfet_model.self_bias_point(a1_global)['vgs'], 10.0)
    gm2 = jfet_model.transconductance(
        jfet_model.self_bias_point(a2_global)['vgs'], 10.0)
    re1 = 1.0 / gm1
    re2 = 1.0 / gm2

    vd_boundary = jfet_model.drain_voltage_at_boundary()  # = |VP|, always
    input_peak = _INPUT_AMPLITUDE

    # Computed VPA target for pre-check (from volume ratio × analytical input VPA)
    target_vpa_est = volume_ratio * _compute_input_vpa(_INPUT_AMPLITUDE, _CAL_FREQ)

    # --- Per-mode: compute Rd for each descriptor ---
    per_mode_rd = {}
    _linear_rd_cache = {}  # Cache linear RD per stage across modes

    for mode_name, mode_def in mode_table.items():
        rds = {}
        for stage, (alpha, id_q, rs_q, re_q) in [
            ("Q1", (a1_global, id1, rs1, re1)),
            ("Q2", (a2_global, id2, rs2, re2)),
        ]:
            descriptor = mode_def[stage]

            if descriptor == 'soft':
                # Drain at saturation-triode boundary: VD = |VP|
                # RD = (VDD − |VP|) / ID
                # This is the Shockley-derived equivalent of the old km=0.5
                # (km=0 would be exactly at boundary; old code used km=0.5
                # which is 0.5V above — but the true boundary IS |VP|)
                rd = (vdd - vd_boundary) / id_q
                rds[stage] = rd
                print(f"  [{mode_name}] {stage}:soft → VD={vd_boundary:.2f}V "
                      f"(sat-triode boundary from Shockley) | "
                      f"Rd={rd:.1f}Ω", flush=True)

            elif descriptor == 'linear':
                # Minimum-gain linear: sweep VD from near VDD (low Rd, low gain)
                # downward, stopping at the first VD where cascade gain exceeds
                # volume_ratio with margin.  Lower gain = less square-law THD.
                # The shunt network attenuates any excess above the volume target.
                if stage in _linear_rd_cache:
                    # Reuse: linear RD for a stage is the same across modes
                    rds[stage] = _linear_rd_cache[stage]
                    continue

                # Minimum cascade gain: volume_ratio × 1.5 margin so the shunt
                # has room to work.  Split between stages as sqrt.
                min_cascade = volume_ratio * 1.5
                if stage == "Q1":
                    # Q1 solved first — assume Q2 will have similar gain
                    min_stage_gain = np.sqrt(min_cascade)
                else:
                    # Q2 solved second — use actual Q1 Rd
                    rd1_lin = _linear_rd_cache.get("Q1")
                    if rd1_lin is not None:
                        gain1_actual = rd1_lin / (rs1 + re1)
                        min_stage_gain = min_cascade / gain1_actual
                    else:
                        min_stage_gain = np.sqrt(min_cascade)

                best_rd = None

                # Sweep from near VDD (lowest gain) downward toward boundary
                for vd_test in np.arange(vdd - 1.0,
                                          vd_boundary + 0.5, -0.25):
                    rd_test = (vdd - vd_test) / id_q
                    if rd_test <= 0:
                        continue
                    gain = rd_test / (rs_q + re_q)
                    headroom = min(vd_test - vd_boundary,
                                   vdd - vd_test)
                    if headroom <= 0:
                        continue
                    # For Q2: verify D2 signal doesn't clip
                    if stage == "Q2":
                        rd1_lin = _linear_rd_cache.get("Q1")
                        if rd1_lin is not None:
                            gain1 = rd1_lin / (rs1 + re1)
                            d2_peak = input_peak * gain1 * gain
                            if d2_peak > headroom * 0.95:
                                continue
                    # Accept the first VD where gain is sufficient
                    if gain >= min_stage_gain:
                        best_rd = rd_test
                        break

                if best_rd is None:
                    # Fallback: VD at midpoint between boundary and VDD
                    vd_fallback = (vdd + vd_boundary) / 2.0
                    best_rd = (vdd - vd_fallback) / id_q
                    print(f"  WARNING [{mode_name}/{stage}]: Linear RD sweep "
                          f"found no valid solution. Using mid-rail fallback: "
                          f"VD={vd_fallback:.2f}V, Rd={best_rd:.1f}Ω",
                          flush=True)
                else:
                    vd_q = vdd - best_rd * id_q
                    gain_actual = best_rd / (rs_q + re_q)
                    margin_above_boundary = vd_q - vd_boundary
                    print(f"  [{mode_name}] {stage}:linear → "
                          f"VD={vd_q:.2f}V ({margin_above_boundary:.2f}V "
                          f"above boundary) | Rd={best_rd:.1f}Ω | "
                          f"gain={gain_actual:.2f}× (min needed: "
                          f"{min_stage_gain:.2f}×)", flush=True)

                rds[stage] = best_rd
                _linear_rd_cache[stage] = best_rd

            elif descriptor == 'hard':
                # Gate clipping: RS is near 0, VGS near 0V.
                # RS switching is handled above via parallel shunt.
                # RD still sets the drain voltage — place at same location
                # as 'soft' (boundary) since we want the gate to clip,
                # not the drain.  The drain just needs to be in saturation.
                if not rs_is_per_mode.get(stage, False):
                    raise ConfigError(
                        f"[{mode_name}] {stage}:hard but RS per-mode switching "
                        f"was not set up. This is an internal error."
                    )
                # Use the hard-mode alpha for this stage's RD computation
                _, bp_hard, _ = jfet_model.compute_hard_bias(vdd)
                id_hard = bp_hard['id']
                # VD at midpoint for headroom (gate does the clipping, not drain)
                vd_target = (vdd + vd_boundary) / 2.0
                rd = (vdd - vd_target) / id_hard
                rds[stage] = rd
                print(f"  [{mode_name}] {stage}:hard → VD={vd_target:.2f}V "
                      f"(gate clips, drain in saturation) | "
                      f"Rd={rd:.1f}Ω | Id_hard={id_hard*1000:.2f}mA",
                      flush=True)

        per_mode_rd[mode_name] = (rds["Q1"], rds["Q2"])

    print(f"\n[GLOBAL] Q1: alpha={a1_global:.2f} | Rs={rs1:.1f}Ω | "
          f"Id={id1*1000:.3f}mA | Vs={vs1:.3f}V")
    print(f"[GLOBAL] Q2: alpha={a2_global:.2f} | Rs={rs2:.1f}Ω | "
          f"Id={id2*1000:.3f}mA | Vs={vs2:.3f}V")
    for mode_name, (rd1, rd2) in per_mode_rd.items():
        q1d = mode_table[mode_name]["Q1"]
        q2d = mode_table[mode_name]["Q2"]
        vd1 = vdd - rd1 * id1
        vd2 = vdd - rd2 * id2
        gain1 = rd1 / (rs1 + re1)
        gain2 = rd2 / (rs2 + re2)
        print(f"[{mode_name}] Q1:{q1d} Rd1={rd1:.1f}Ω VD1={vd1:.2f}V gain={gain1:.1f}× | "
              f"Q2:{q2d} Rd2={rd2:.1f}Ω VD2={vd2:.2f}V gain={gain2:.1f}× | "
              f"cascade={gain1*gain2:.1f}×")

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
    signal_half_peak = target_vpa_est * 3.0 / 2.0
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

    # === Compute per-mode RS values ===
    for mode_name, mode_def in mode_table.items():
        rs1_eff = rs1  # default: global
        rs2_eff = rs2
        for stage, rs_global, rs_idx in [("Q1", rs1, "rs1"), ("Q2", rs2, "rs2")]:
            if mode_def.get(stage) == 'hard' and stage in rs_hard_shunts:
                shunt = rs_hard_shunts[stage]
                if shunt is not None:
                    eff = (rs_global * shunt) / (rs_global + shunt)
                else:
                    # No shunt needed (shouldn't happen in normal flow)
                    _, bp_hard, _ = jfet_model.compute_hard_bias(vdd)
                    eff = bp_hard['rs']
                if rs_idx == "rs1":
                    rs1_eff = eff
                else:
                    rs2_eff = eff
        per_mode_rs[mode_name] = (rs1_eff, rs2_eff)

    # === OPA1656 Parallel Unity Buffer ===
    #
    # Both channels of one OPA1656 (U3) wired as parallel unity-gain buffers:
    #   +IN_A and +IN_B tied together → receive signal from C4
    #   −IN_A tied to OUT_A, −IN_B tied to OUT_B (unity-gain feedback)
    #   OUT_A through R_ISO_A to common output node
    #   OUT_B through R_ISO_B to common output node
    #
    # Isolation resistors prevent the two channels from fighting due to
    # tiny offset differences (typ ±0.5 mV per OPA1656 datasheet p6).
    # Effective output impedance = R_ISO_A || R_ISO_B.
    #
    # Benefits (physical, not modeled in transient sim):
    #   - Voltage noise: 2.9 nV/√Hz → 2.9/√2 ≈ 2.05 nV/√Hz
    #   - Output current: 100 mA per channel → 200 mA total peak
    #   - Offset averaging: each channel's Vos averages out partially
    #
    # In transient sim: modeled as R_ISO_EFF in series with output.
    # The opamp is a LINEAR unity buffer — it must never clip.
    # Rail limits are a design validation check, not a signal modification.
    # If signal at BUF exceeds [0.25V, 17.75V], the upstream shunt network
    # or Q3 headroom is wrong.

    r_iso_per_channel = 10.0  # Ω — prevents channel-to-channel fighting
    r_iso_eff = r_iso_per_channel / 2.0  # Two channels in parallel: 10Ω || 10Ω = 5Ω

    # OPA1656 datasheet parameters (p6-7)
    opamp_en        = 2.9e-9   # nV/√Hz at 10 kHz (single channel)
    opamp_en_parallel = opamp_en / np.sqrt(2)  # √2 improvement from parallel
    opamp_isc       = 100e-3   # 100 mA short-circuit current per channel

    print("[GLOBAL] OPA1656 (U3): both channels in parallel unity-gain buffer")
    print(f"  R_ISO per channel: {r_iso_per_channel:.0f}Ω | "
          f"effective: {r_iso_per_channel:.0f}Ω || {r_iso_per_channel:.0f}Ω = "
          f"{r_iso_eff:.1f}Ω")
    print(f"  Noise: {opamp_en*1e9:.1f} nV/√Hz → "
          f"{opamp_en_parallel*1e9:.2f} nV/√Hz (parallel)")
    print(f"  Rail limits: [{_OPAMP_V_MIN:.2f}V, {_OPAMP_V_MAX:.2f}V] "
          f"(validation — signal must stay within)")
    print(f"  Output current: 2× {opamp_isc*1000:.0f}mA = "
          f"{2*opamp_isc*1000:.0f}mA peak")

    # Tone network summary
    f_tone_pole = 1.0 / (2.0 * np.pi * r_tone * c_tone)
    print(f"\n[GLOBAL] Tone network: R_TONE={r_tone/1e3:.0f}kΩ (fully open) | "
          f"C_TONE={c_tone*1e9:.0f}nF | "
          f"RC pole={f_tone_pole:.1f}Hz")

    return {
        "rs1": rs1, "rs2": rs2, "id1": id1, "id2": id2,
        "vs1": vs1, "vs2": vs2,
        "a1": a1_global, "a2": a2_global,
        "r_g_calc": r_g_calc,
        "r_vol": r_vol,
        "r_tone": r_tone, "c_tone": c_tone,
        "rs_cs": rs_cs,
        "c_ref": c_ref,
        "idss_t": idss, "vp_t": -vp_abs,
        "idss_q3": idss_q3, "vp_q3": -vp_q3,
        "idss_q4": idss_q4, "vp_q4": -vp_q4,
        "vs_q3": vs_q3_actual,
        "id_q4": id_q4_actual,
        "vs_q4": vs_q4_actual,
        "headroom_pos": headroom_pos, "headroom_neg": headroom_neg,
        "tau_floor": tau_floor, "tau_ceiling": tau_ceiling,
        "per_mode_rd": per_mode_rd,
        "per_mode_rs": per_mode_rs,
        "rs_is_per_mode": rs_is_per_mode,
        "rs_hard_shunts": rs_hard_shunts,
        "mode_table": mode_table,
        "volume_ratio": volume_ratio,
        "jfet_model": jfet_model,
        # OPA1656 parallel unity buffer
        "r_iso_per_channel": r_iso_per_channel,
        "r_iso_eff": r_iso_eff,
        "opamp_en_parallel": opamp_en_parallel,
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

    # Fix #9: Use per-mode RS when hard mode makes RS per-mode switched
    if "per_mode_rs" in resistors and mode in resistors["per_mode_rs"]:
        rs1, rs2 = resistors["per_mode_rs"][mode]

    r_g_calc = resistors["r_g_calc"]
    idss_t   = resistors["idss_t"]
    vp_t     = resistors["vp_t"]
    rd1, rd2 = resistors["per_mode_rd"][mode]

    # R7: use override from SCF feedback, or estimate from tau midpoint
    if r7_override is not None:
        r7_est = r7_override
    else:
        tf = resistors["tau_floor"]
        tc = resistors["tau_ceiling"]
        tau_mid = (tf + tc) / 2.0
        r7_est  = max(1000.0, tau_mid / 100e-9)
    c_ph    = 100e-9  # placeholder cap

    core = [
        {"type": "Inductor",  "name": "L_PICKUP",  "val": 4.5,       "n1": "v_ideal", "n2": "IN", "r_dc": 8000.0, "c_p": 150e-12},
        # Tone pot + tone cap: series RC shunt from IN to GND
        {"type": "Capacitor", "name": "C_TONE",    "val": 22e-9,                               "n1": "IN",      "n2": "TONE"},
        {"type": "Resistor",  "name": "R_TONE",    "val": resistors.get("r_tone", 500e3),      "n1": "TONE",    "n2": "-"},
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
        {"type": "Capacitor", "name": "C4",         "val": c_ph,      "n1": "S3",      "n2": "BUF"},
        {"type": "Resistor",  "name": "R_ISO_EFF",  "val": resistors.get("r_iso_eff", 5.0),   "n1": "BUF",     "n2": "OUT"},
        {"type": "Resistor",  "name": "VOL_POT",    "val": resistors.get("r_vol", 500.0e3),   "n1": "OUT",     "n2": "-"},
    ]
    sim = Circuit(v_dd_ideal=18.0, r_psu=10.0)
    process_unified_circuit(sim, core)
    sim.solve_dc_bias(input_node="v_ideal")

    rth_c1 = sim.solve_ac_thevenin(Capacitor("P_C1", 1e-9, "IN",  "G1"))
    rth_c2 = sim.solve_ac_thevenin(Capacitor("P_C2", 1e-9, "D1",  "G2"))
    rth_c3 = sim.solve_ac_thevenin(Capacitor("P_C3", 1e-9, "D2",  "G3"))
    rth_c4 = sim.solve_ac_thevenin(Capacitor("P_C4", 1e-9, "S3",  "BUF"))

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
                           sec_fc_floor, sec_t5_floor):
    """
    For each coupling cap, find the single GLOBAL E24 value that satisfies
    the blocking window across ALL modes' R_th values.

    Dual-threshold enforcement:
      Primary bounds:
        tau_floor ≤ R_th × C ≤ tau_ceiling
      Secondary floors (hard — tighten the primary window):
        fc  ≥ sec_fc_floor  (Hz)  → tau ≤ 1/(2π×fc_floor)
        5τ  ≥ sec_t5_floor  (s)   → tau ≥ t5_floor/5

      The effective window is the intersection of primary and secondary:
        tau_min = max(tau_floor, sec_t5_floor / 5)
        tau_max = min(tau_ceiling, 1 / (2π × sec_fc_floor))

      Within this tightened window, candidates are scored by their minimum
      margin above both secondary floors.  This avoids favoring one bound
      over the other when multiple E24 values fit.

    C1, C2 — standard both-bound intersection, single E24.
    C4 — both-bound intersection, combo (base+delta) allowed.
    C3 — per-mode (not handled here; solved independently per mode).
    """
    rth_by_mode = {r["mode"]: r for r in mode_rth_list}
    caps = {}

    # Tighten primary window with secondary hard floors
    tau_sec_min = sec_t5_floor / 5.0              # from 5τ ≥ t5_floor
    tau_sec_max = 1.0 / (2.0 * np.pi * sec_fc_floor)  # from fc ≥ fc_floor
    tau_eff_floor   = max(tau_floor, tau_sec_min)
    tau_eff_ceiling = min(tau_ceiling, tau_sec_max)

    if tau_eff_floor > tau_eff_ceiling:
        print(f"  WARNING: Secondary floors tighten window to empty! "
              f"tau∈[{tau_eff_floor*1000:.2f}, {tau_eff_ceiling*1000:.2f}] ms. "
              f"Falling back to primary window.", flush=True)
        tau_eff_floor = tau_floor
        tau_eff_ceiling = tau_ceiling
    else:
        print(f"  Effective tau window: [{tau_eff_floor*1000:.2f}, "
              f"{tau_eff_ceiling*1000:.2f}] ms "
              f"(primary [{tau_floor*1000:.2f}, {tau_ceiling*1000:.2f}] "
              f"tightened by secondary fc≥{sec_fc_floor:.1f}Hz, "
              f"5τ≥{sec_t5_floor*1000:.1f}ms)", flush=True)

    # Secondary margin scoring: within the tightened window, bias selection
    # toward candidates with maximum margin above both secondary floors.
    # Since secondary is now hard-enforced, this is purely a tiebreaker.
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
        c_min = max(tau_eff_floor   / rth for rth in rth_values)
        c_max = min(tau_eff_ceiling / rth for rth in rth_values)
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
    c4_min = max(tau_eff_floor   / rth for rth in rth_c4_values)
    c4_max = min(tau_eff_ceiling / rth for rth in rth_c4_values)
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
        (GLOBAL, "U1 (LSK489A #1)", "Q1 + Q2 — Monolithic Dual N-Channel JFET (gain/distortion stages)",
            "LSK489A SOIC-8  |  Ch.A = Q1, Ch.B = Q2", {}),
        (GLOBAL, "U2 (LSK489A #2)", "Q3 + Q4 — Monolithic Dual N-Channel JFET (follower + current sink)",
            "LSK489A SOIC-8  |  Ch.A = Q3 (follower), Ch.B = Q4 (current sink)", {}),
        (GLOBAL, "U3 (OPA1656)", "OPA1656 Dual Op-Amp — Both channels in parallel as unity-gain buffer",
            "OPA1656 SOIC-8  |  Ch.A+B parallel, 10Ω isolation per channel", {}),
        (GLOBAL, "R_ISO_A / R_ISO_B", "OPA1656 Output Isolation Resistors (prevent channel fighting)",
            "2× 10.0 Ω  (OUT_A→node, OUT_B→node)", {}),
        (GLOBAL, "R1 / R4", "Q1 / Q2 Gate Bias Resistors (LSK489 IGSS-derived, blocking-compatible)",
            fmt_r(r_g_val), {}),
        (GLOBAL, "Q4 + Rs_CS", "Q4 JFET Current Source (gate→GND, source→Rs_CS→GND) — symmetric Q3 output",
            f"Q4: LSK489A Ch.B (U2)  |  Rs_CS: {fmt_r(rs_cs_val)}", {}),
        (GLOBAL, "R_VOL",   "Volume Potentiometer (fixed hardware)",
            fmt_r(r_vol_val), {}),
        (GLOBAL, "C_REF",   "V_REF Bypass Capacitor",
            fmt_c(global_resistors["c_ref"] if global_resistors else 47e-6), {}),
        (GLOBAL, "L_PICKUP","Pickup Inductance",
            "4.500 H  (DCR: 8.000 kΩ  Cp: 150 pF)", {}),
        (GLOBAL, "C_TONE", "Tone Cap (series with R_TONE, shunt IN→GND)",
            fmt_c(global_resistors["c_tone"] if global_resistors else 22e-9), {}),
        (GLOBAL, "R_TONE", "Tone Pot (series with C_TONE, shunt IN→GND) — user adjustable",
            fmt_r(global_resistors["r_tone"] if global_resistors else 500e3), {}),

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
def execute_mode_analytics(mode, c3_shunt_target, rtot_target, config, mode_def=None):
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

    # OPA1656 rail validation — the buffer is LINEAR, it must never clip.
    # Check the BUF node (opamp output) against datasheet rail limits.
    if "BUF" in analyzer.v_out_data:
        v_buf = analyzer.v_out_data["BUF"][start_idx:]
        v_buf_dc = np.mean(v_buf)
        v_buf_min = np.min(v_buf)
        v_buf_max = np.max(v_buf)
        # BUF is AC-coupled via C4, so DC ≈ 0.  The opamp's absolute output
        # is V_REF + v_buf_ac.  Check against absolute rail limits.
        v_ref = analyzer.circuit.v_dd_ideal / 2.0
        v_abs_min = v_ref + (v_buf_min - v_buf_dc)
        v_abs_max = v_ref + (v_buf_max - v_buf_dc)
        margin_lo = v_abs_min - _OPAMP_V_MIN
        margin_hi = _OPAMP_V_MAX - v_abs_max
        if margin_lo < 0 or margin_hi < 0:
            print(f"  ✗ OPA1656 RAIL VIOLATION: BUF signal [{v_abs_min:.2f}V, "
                  f"{v_abs_max:.2f}V] exceeds rails [{_OPAMP_V_MIN:.2f}V, "
                  f"{_OPAMP_V_MAX:.2f}V]. Fix upstream: reduce shunt "
                  f"attenuation or increase Q3 headroom.")
        else:
            print(f"  ✓ OPA1656 linear: BUF [{v_abs_min:.2f}V, {v_abs_max:.2f}V] "
                  f"within rails | margin: +{margin_hi:.2f}V / -{margin_lo:.2f}V")

    analyzer.report_single_tone_thd(node="OUT")
    analyzer.plot_waveforms(mode=mode, mode_def=mode_def)
    analyzer.export_audio(mode=mode)



def run_preamp_design(idss, vp_abs, vdd, modes, volume_ratio,
                       blocking, bloom_light, bloom_heavy):
    """
    Full preamp design pipeline: validation, component resolution,
    VPA optimization, and analytics.

    Args:
        idss:          JFET IDSS (A) — LSK489A typical 5.5 mA
        vp_abs:        |VP| (V) — LSK489A typical 2.0 V
        vdd:           Supply voltage (V)
        modes:         dict of {mode_name: {"Q1": str, "Q2": str, "bloom": False|"light"|"heavy"}}
        volume_ratio:  Target VPA_out / VPA_in
        blocking:      dict with fc_hz, t5_ms, fc_floor_hz, t5_floor_ms
        bloom_light:   dict with fc_hz, t5_ms, fc_floor_hz, t5_floor_ms
        bloom_heavy:   dict with fc_hz, t5_ms, fc_floor_hz, t5_floor_ms
    """
    # ==================================================================
    #  VALIDATION — Check all inputs before computation
    # ==================================================================

    jfet_model = LSK489_Model(idss, vp_abs)

    validate_inputs(modes, volume_ratio, blocking, bloom_light, bloom_heavy,
                    vdd=vdd, jfet_model=jfet_model)

    # --- Derive target VPA from volume ratio ---
    _INPUT_VPA = _compute_input_vpa(_INPUT_AMPLITUDE, _CAL_FREQ)
    _TARGET_VPA = volume_ratio * _INPUT_VPA

    mode_names = list(modes.keys())
    n_cpus = os.cpu_count() or 4
    tau_floor   = 1.0 / (2.0 * np.pi * blocking['fc_hz'])
    tau_ceiling = blocking['t5_ms'] / 1000.0 / 5.0

    # Effective blocking tau: primary window tightened by secondary floors
    tau_eff_floor   = max(tau_floor, (blocking['t5_floor_ms'] / 1000.0) / 5.0)
    tau_eff_ceiling = min(tau_ceiling, 1.0 / (2.0 * np.pi * blocking['fc_floor_hz']))

    # Bloom tau windows (for modes with bloom="light" or "heavy")
    bloom_tau = {
        "light": {
            "floor":   1.0 / (2.0 * np.pi * bloom_light['fc_hz']),
            "ceiling": bloom_light['t5_ms'] / 1000.0 / 5.0,
        },
        "heavy": {
            "floor":   1.0 / (2.0 * np.pi * bloom_heavy['fc_hz']),
            "ceiling": bloom_heavy['t5_ms'] / 1000.0 / 5.0,
        },
    }

    # ==================================================================
    #  PHASE A — Global component resolution
    # ==================================================================
    print("=" * 72)
    print("  JFET PREAMP — Descriptor-Driven Mode Architecture")
    print(f"  {n_cpus} CPUs | LSK489A: idss={idss*1000:.1f}mA |VP|={vp_abs:.1f}V")
    print(f"  Volume ratio: {volume_ratio:.1f}× → target VPA ≈ {_TARGET_VPA:.3f} V_w")
    for mn, md in modes.items():
        bloom_str = f" +bloom({md['bloom']})" if md.get('bloom') else ""
        print(f"  {mn}: Q1:{md['Q1']} Q2:{md['Q2']}{bloom_str}")
    print(f"  Blocking: τ∈[{tau_floor*1000:.2f}, {tau_ceiling*1000:.2f}] ms")
    bloom_levels_used = set(m.get('bloom', False) for m in modes.values()) - {False}
    for lvl in sorted(bloom_levels_used):
        bt = bloom_tau[lvl]
        print(f"  Bloom({lvl}): τ∈[{bt['floor']*1000:.1f}, {bt['ceiling']*1000:.1f}] ms")
    print(f"  OPA1656 buffer: unity gain, rails [{_OPAMP_V_MIN:.2f}, {_OPAMP_V_MAX:.2f}]V")
    print("=" * 72, flush=True)

    # Step 1: Resolve global source resistors + per-mode drain resistors
    print("\n--- Phase A Step 1: Global Resistor Resolution ---", flush=True)
    resistors = _resolve_global_resistors(
        idss=idss, vp_abs=vp_abs, vdd=vdd,
        target_fc_hz=blocking['fc_hz'], target_recovery_ms=blocking['t5_ms'],
        mode_table=modes, volume_ratio=volume_ratio, jfet_model=jfet_model,
    )

    # ==================================================================
    #  Phase A2-init: Initial Thevenin probes (C3 selection)
    # ==================================================================
    print("\n--- Phase A Step 2-init: Initial Thevenin Probes (C3 selection) ---",
          flush=True)
    init_probe_args = [(m, resistors) for m in mode_names]
    with ProcessPoolExecutor(max_workers=min(n_cpus, len(mode_names))) as pool:
        init_rth_list = list(pool.map(_probe_mode_rth, init_probe_args))
    init_rth_by_mode = {r["mode"]: r for r in init_rth_list}

    # Per-mode C3 selection (FROZEN)
    c3_by_mode = {}
    for m in mode_names:
        # Use bloom tau window for bloom modes' C3 selection
        mode_def = modes[m]
        bloom_level = mode_def.get('bloom', False)
        if bloom_level:
            tf = bloom_tau[bloom_level]["floor"]
            tc = bloom_tau[bloom_level]["ceiling"]
        else:
            tf, tc = tau_floor, tau_ceiling

        rth_c3 = init_rth_by_mode[m]["rth_c3"]
        c3_min = tf / rth_c3
        c3_max = tc / rth_c3
        eps = c3_max * 1e-9
        in_window = _E24_CAPS[(_E24_CAPS >= c3_min - eps) & (_E24_CAPS <= c3_max + eps)]
        if len(in_window) > 0:
            c3_val = float(in_window[0])
        else:
            c3_val = float(_E24_CAPS[np.argmin(np.abs(_E24_CAPS - c3_min))])
            tau_label = f"bloom({bloom_level})" if bloom_level else "blocking"
            print(f"  WARNING [{m}/C3]: No E24 in {tau_label} window "
                  f"[{c3_min*1e9:.2f}, {c3_max*1e9:.2f}] nF. "
                  f"Using closest: {c3_val*1e9:.2f} nF", flush=True)
        c3_by_mode[m] = c3_val
        print(f"[{m}] C3: {c3_val*1e9:.3f} nF [E24] (FROZEN) | "
              f"window: [{c3_min*1e9:.2f}, {c3_max*1e9:.2f}] nF | "
              f"R_th_C3: {rth_c3/1000:.1f}k", flush=True)

    # ==================================================================
    #  SCF LOOP
    # ==================================================================
    _SCF_MAX_ITER = 5
    _SCF_THRESHOLD = 0.01
    prev_r7 = {m: None for m in mode_names}

    for scf_iter in range(1, _SCF_MAX_ITER + 1):
        print(f"\n{'='*72}")
        print(f"  SCF ITERATION {scf_iter} / {_SCF_MAX_ITER}")
        print(f"{'='*72}", flush=True)

        # Phase A2: Thevenin probes
        print("\n--- Phase A Step 2: Per-Mode Thevenin Probes ---", flush=True)
        probe_args = [(m, resistors, prev_r7[m]) for m in mode_names]
        with ProcessPoolExecutor(max_workers=min(n_cpus, len(mode_names))) as pool:
            mode_rth_list = list(pool.map(_probe_mode_rth, probe_args))
        rth_by_mode = {r["mode"]: r for r in mode_rth_list}

        # Phase A3: Global cap reconciliation (C1, C2, C4)
        print("\n--- Phase A Step 3: Global Cap Reconciliation ---", flush=True)
        global_caps = _reconcile_global_caps(mode_rth_list, mode_names,
                                              tau_floor, tau_ceiling,
                                              sec_fc_floor=blocking['fc_floor_hz'],
                                              sec_t5_floor=blocking['t5_floor_ms'] / 1000.0)

        # Phase A4: Config assembly
        c1 = global_caps["C1"]["value"]
        c2 = global_caps["C2"]["value"]
        c4 = global_caps["C4"]["value"]

        mode_configs = {}
        for m in mode_names:
            rd1, rd2 = resistors["per_mode_rd"][m]
            rth = rth_by_mode[m]
            mode_def = modes[m]

            # Per-mode RS (uses parallel shunt when hard is active)
            if m in resistors.get("per_mode_rs", {}):
                rs1_eff, rs2_eff = resistors["per_mode_rs"][m]
            else:
                rs1_eff = resistors["rs1"]
                rs2_eff = resistors["rs2"]

            # Per-mode tau window (bloom level or blocking)
            bloom_level = mode_def.get('bloom', False)
            if bloom_level:
                m_tau_floor = bloom_tau[bloom_level]["floor"]
                m_tau_ceiling = bloom_tau[bloom_level]["ceiling"]
            else:
                m_tau_floor = tau_floor
                m_tau_ceiling = tau_ceiling

            mode_configs[m] = {
                "mode":           m,
                "c1": c1, "c2": c2, "c3": c3_by_mode[m], "c4": c4,
                "rd1": rd1, "rs1": rs1_eff,
                "rd2": rd2, "rs2": rs2_eff,
                "r_g_calc":       resistors["r_g_calc"],
                "r_vol":          resistors["r_vol"],
                "r_tone":         resistors["r_tone"],
                "rs_cs":          resistors["rs_cs"],
                "r_iso_eff":      resistors["r_iso_eff"],
                "c_ref":          resistors["c_ref"],
                "idss_t":         resistors["idss_t"],
                "vp_t":           resistors["vp_t"],
                "idss_q3":        resistors["idss_q3"],
                "vp_q3":          resistors["vp_q3"],
                "idss_q4":        resistors["idss_q4"],
                "vp_q4":          resistors["vp_q4"],
                "r_no_r7":        rth["r_no_r7"],
                "rth_g3_offset":  rth["rth_g3_offset"],
                "tau_floor":      m_tau_floor,
                "tau_ceiling":    m_tau_ceiling,
                "tau_blocking_floor":   tau_eff_floor,
                "tau_blocking_ceiling": tau_eff_ceiling,
                "cal_freq":       _CAL_FREQ,
                "monitor_nodes":  _MONITOR_NODES,
                "target_vpa":     _TARGET_VPA,
                # Seed voltage fields for DC bias fix
                "vs1": resistors["vs1"], "vs2": resistors["vs2"],
                "id1": resistors["id1"], "id2": resistors["id2"],
                "vs_q3": resistors["vs_q3"],
                "id_q4": resistors.get("id_q4", 0.0),
            }
            q1d = mode_def["Q1"]
            q2d = mode_def["Q2"]
            bloom_str = f" +bloom({mode_def['bloom']})" if mode_def.get('bloom') else ""
            print(f"[{m}] Q1:{q1d} Rd1={rd1:.1f}Ω Rs1={rs1_eff:.1f}Ω | "
                  f"Q2:{q2d} Rd2={rd2:.1f}Ω Rs2={rs2_eff:.1f}Ω | "
                  f"C3={c3_by_mode[m]*1e9:.2f}nF{bloom_str}", flush=True)

        # --- Phase B: Per-mode VPA scan + fine evaluation ---
        total_min = 1e-9
        total_max = 200e-9
        vpa_gate_lo = _INPUT_VPA
        vpa_gate_hi = _TARGET_VPA * 1.25

        # Step 1: Sparse E24 scan per mode
        e24_totals = [float(c) for c in _E24_CAPS if total_min <= c <= total_max]
        scan_items = []
        for m in mode_names:
            cfg = mode_configs[m]
            for ct in e24_totals:
                r7 = _compute_r7_mid(ct, cfg["c3"], cfg["r_no_r7"],
                                      cfg["tau_floor"], cfg["tau_ceiling"],
                                      cfg.get("rth_g3_offset", 0.0))
                scan_items.append((cfg, ct, r7, m))

        print("\n--- Phase B Step 1: Sparse VPA Scan ---")
        print(f"  {len(e24_totals)} E24 totals × {len(mode_names)} modes = "
              f"{len(scan_items)} scan evals", flush=True)

        with ProcessPoolExecutor(max_workers=min(n_cpus, len(scan_items))) as pool:
            scan_results = list(pool.map(_eval_cap_worker, scan_items))

        # Per-mode viable range
        viable_range = {}
        for m in mode_names:
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

        for m in mode_names:
            lo, hi = viable_range[m]
            margin = (hi - lo) * 0.20
            viable_range[m] = (max(total_min, lo - margin), min(total_max, hi + margin))

        # Step 2: Fine enumeration
        work_items = []
        seen_keys = set()
        for c_base in _CAP_SEARCH_RANGE:
            for delta in _E24_DELTA_CAPS:
                total = float(c_base) + float(delta)
                for m in mode_names:
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
                               for m in mode_names)
        print("\n--- Phase B Step 2: Fine Evaluation ---")
        print(f"  Per-mode ranges: {range_strs}")
        print(f"  {n_unique} unique totals × relevant modes = "
              f"{n_items} evals across {n_workers} workers", flush=True)

        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            all_results = list(pool.map(_eval_cap_worker, work_items))

        mode_results = {m: {} for m in mode_names}
        n_block = 0
        for r in all_results:
            if not r.get("blocking_ok", True):
                n_block += 1
                continue
            if r["vpa"] < vpa_gate_lo or r["vpa"] > vpa_gate_hi:
                continue
            mode_results[r["mode"]][_cap_key(r["cap"])] = r

        print(f"  Fine eval: {len(all_results)} evals | {n_block} blocked | "
              + ", ".join(f"{m}:{len(mode_results[m])}" for m in mode_names),
              flush=True)

        # ==================================================================
        #  PHASE C — Cross-mode trio matching
        # ==================================================================
        print("\n--- Phase C: Cross-Mode Trio Matching ---", flush=True)

        best_trio_score = (float('inf'), float('inf'))
        best_trio = None

        # Build per-mode result lists for iteration
        mode_result_lists = {m: list(mode_results[m].values()) for m in mode_names}

        # For 3 modes, iterate all combinations
        from itertools import product as iterproduct
        for combo in iterproduct(*[mode_result_lists[m] for m in mode_names]):
            vpas = [r["vpa"] for r in combo]
            spread = max(vpas) - min(vpas)
            sum_err = sum(abs(v - _TARGET_VPA) for v in vpas)
            score = (spread, sum_err)
            if score < best_trio_score:
                best_trio_score = score
                best_trio = {m: r for m, r in zip(mode_names, combo)}

        if best_trio is None:
            raise RuntimeError("No valid cross-mode trio found. "
                               "Check blocking bounds and VPA target.")

        print(f"  Best trio spread: {best_trio_score[0]*1000:.2f} mV_w | "
              f"sum |err|: {best_trio_score[1]:.4f} V_w")

        print("\n" + "=" * 72)
        print("  CROSS-MODE SELECTION — RESULTS (per-mode G3 network)")
        print("=" * 72)
        print(f"  Target VPA: {_TARGET_VPA:.4f} V_w (volume ratio {volume_ratio:.1f}×)")
        print("-" * 72)

        c3_shunt_targets = {}
        rtot_targets = {}
        all_mode_components = {}
        for m in mode_names:
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
                "rd1": cfg["rd1"], "rs1": cfg["rs1"],
                "rd2": cfg["rd2"], "rs2": cfg["rs2"],
                "c1": c1, "c2": c2, "c3": c3_v, "c4": c4,
                "c3_shunt": r["cap"], "r7": r["r7"],
            }
            bloom_str = f" [BLOOM({modes[m]['bloom'].upper()})]" if modes[m].get('bloom') else ""
            print(f"  [{m:>5s}] Q1:{modes[m]['Q1']} Q2:{modes[m]['Q2']}{bloom_str} | "
                  f"C3_shunt: {r['cap']*1e9:.2f} nF | "
                  f"R7: {r['r7']/1000:7.2f} kΩ | V_pa: {r['vpa']:.4f} V_w | "
                  f"Fc: {f_g3:.1f} Hz | 5τ: {t5_g3*1000:.1f} ms")
        print("=" * 72)

        # Blocking/bloom detail
        for m in mode_names:
            bd = best_trio[m].get("blocking_detail", {})
            tau_label = f"Bloom({modes[m]['bloom']})" if modes[m].get('bloom') else "Blocking"
            if bd:
                print(f"  [{m}] {tau_label}: " + " | ".join(
                    f"{cn}: fc={d['fc']:.1f}Hz 5τ={d['t5']*1000:.1f}ms "
                    f"{'✓' if d['ok'] else '✗'}"
                    for cn, d in bd.items()))

        # Q3 headroom validation
        print(f"\n  Q3 headroom (Q4 current source, Rs_CS={resistors['rs_cs']:.1f}Ω):")
        print(f"  Q3 DC: Vs={resistors['vs_q3']:.3f}V | "
              f"headroom +{resistors['headroom_pos']:.2f}V / "
              f"-{resistors['headroom_neg']:.2f}V")
        for m in mode_names:
            vpa = best_trio[m]["vpa"]
            vpp_est = vpa * 3.0
            margin_pos = resistors["headroom_pos"] - vpp_est / 2.0
            margin_neg = resistors["headroom_neg"] - vpp_est / 2.0
            ok = margin_pos > 0 and margin_neg > 0
            print(f"  [{m}] V_pa={vpa:.4f} → est Vpp≈{vpp_est:.2f}V | "
                  f"margin: +{margin_pos:.2f}V / -{margin_neg:.2f}V "
                  f"{'✓' if ok else '✗ CLIP RISK'}")

        print(f"\n  OPA1656 parallel unity buffer (linear — must not clip): "
              f"rails [{_OPAMP_V_MIN:.2f}, {_OPAMP_V_MAX:.2f}]V")
        print()

        # --- SCF Convergence Check ---
        current_r7 = {m: rtot_targets[m] for m in mode_names}
        if all(prev_r7[m] is not None for m in mode_names):
            max_change = max(
                abs(current_r7[m] - prev_r7[m]) / prev_r7[m]
                for m in mode_names)
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

    # ==================================================================
    #  PHASE D — Full analytics with volume ratio reporting
    # ==================================================================
    for m in mode_names:
        execute_mode_analytics(m, c3_shunt_targets[m], rtot_targets[m],
                               mode_configs[m], mode_def=modes[m])

    # Final volume ratio report
    print("\n" + "=" * 72)
    print("  VOLUME RATIO SUMMARY")
    print("=" * 72)
    print(f"  Target: {volume_ratio:.1f}× (VPA_out / VPA_in)")
    print(f"  Input VPA (computed): {_INPUT_VPA:.4f} V_w "
          f"({_INPUT_AMPLITUDE:.3f}V × A_w(1kHz))")
    print(f"  Target VPA: {_TARGET_VPA:.4f} V_w")
    for m in mode_names:
        vpa = best_trio[m]["vpa"]
        actual_ratio = vpa / _INPUT_VPA
        print(f"  [{m}] VPA={vpa:.4f} V_w → ratio={actual_ratio:.2f}× "
              f"({'✓' if abs(actual_ratio - volume_ratio) / volume_ratio < 0.1 else '⚠ >10% off'})")
    print("=" * 72)

    write_component_tsv(all_mode_components, global_resistors=resistors,
                        filename="component_bom.tsv")

if __name__ == "__main__":

    # --- JFET Device Parameters (LSK489A datasheet) ---
    IDSS   = 0.0055     # 5.5 mA typical (A grade: 2.5–8.5 mA)
    VP_ABS = 2.0        # |VP| = 2.0 V typical (range: 1.5–3.5 V)
    VDD    = 18.0       # Single supply voltage

    # --- Blocking Thresholds (fast recovery — pickup switching) ---
    blocking = {
        'fc_hz':       33.3,   # Primary ceiling: fc ≤ 36 Hz
        't5_ms':       33.3,   # Primary ceiling: 5τ ≤ 36 ms
        'fc_floor_hz': 16.0,   # Secondary floor: fc ≥ 20 Hz
        't5_floor_ms': 16.0,   # Secondary floor: 5τ ≥ 20 ms
    }

    # --- Bloom Thresholds (slow recovery — compression swell) ---
    # Only applied to G3 shunt network in modes with bloom="light" or "heavy".
    bloom_light = {
        'fc_hz':       8.0,    # Corner frequency
        't5_ms':       120.0,  # Recovery time
        'fc_floor_hz': 4.0,    # Minimum corner
        't5_floor_ms': 60.0,   # Minimum recovery
    }
    bloom_heavy = {
        'fc_hz':       2.0,    # Very low corner — sustain holds
        't5_ms':       500.0,  # Half-second bloom tail
        'fc_floor_hz': 0.5,    # Minimum bloom corner
        't5_floor_ms': 200.0,  # Minimum bloom recovery
    }

    # --- Distortion Mode Definitions ---
    # Q1 and Q2 descriptors:
    #   'linear' — saturation region (Transfer Characteristics, datasheet p5)
    #              RD maximizes headroom above sat-triode boundary
    #   'soft'   — drain at sat-triode boundary (Output Characteristics knee, p4)
    #              signal peaks enter triode region for smooth compression
    #   'hard'   — gate junction forward-biases (Gate Current plot, p4)
    #              input clamped by PN diode; RS becomes per-mode switched
    #
    # 'bloom': False, "light", or "heavy"
    #   Controls G3 shunt network recovery time. Only meaningful when
    #   upstream stages clip. Thresholds defined above.
    MODES = {
        "Clean": {"Q1": "linear", "Q2": "linear", "bloom": False},
        "OD1":   {"Q1": "linear", "Q2": "soft",   "bloom": False},
        "OD2":   {"Q1": "soft",   "Q2": "soft",   "bloom": False},
    }

    # --- Volume Target ---
    # Ratio of output VPA to input VPA (A-weighted).
    # All modes are matched to this same ratio via the G3 shunt network.
    VOLUME_RATIO = 3

    # --- Run ---
    run_preamp_design(IDSS, VP_ABS, VDD, MODES, VOLUME_RATIO,
                       blocking, bloom_light, bloom_heavy)
