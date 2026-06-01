"""
Quasinormal Mode Ringdown Waveform
====================================

After two black holes merge, the resulting distorted remnant BH radiates
gravitational waves as it settles into a stationary Kerr state. This
radiation is described by quasinormal modes (QNMs) — the characteristic
damped oscillations of the Kerr spacetime.

The dominant mode is (l, m, n) = (2, 2, 0), meaning:
    - l=2, m=2: quadrupolar, co-rotating mode
    - n=0: fundamental overtone (longest damping time)

The QNM spectrum depends ONLY on the final BH mass and spin (no-hair theorem).
Measuring f_QNM and τ_QNM independently would test general relativity.

References
----------
[1] Berti, Cardoso & Starinets, "Quasinormal modes of black holes and
    black branes", PRD 73, 064030 (2006).
    Table VIII: fitting coefficients for Kerr QNM frequencies.
    https://doi.org/10.1103/PhysRevD.73.064030

[2] Berti, Cardoso & Will, "Gravitational-wave spectroscopy of massive
    black holes with the space interferometer LISA", PRD 73, 064030 (2006).
    QNM excitation amplitudes.

[3] Barausse & Rezzolla, "Predicting the direction of the final spin from
    the coalescence of two black holes", ApJL 704, L40 (2009).
    Final spin fitting formula.
    https://doi.org/10.1088/0004-637X/704/1/L40

[4] Leaver, "An analytic representation for the quasinormal modes of
    Kerr black holes", Proc. R. Soc. Lond. A 402, 285 (1985).
    Original QNM calculation method.

[5] Echeverria, "Gravitational-wave measurements of the mass and angular
    momentum of a black hole", PRD 40, 3194 (1989).
    Ringdown SNR and parameter estimation.
"""

import numpy as np

from gravitational_wave_analyzer.constants import (
    G_SI, C_SI, MSUN_SI, PI, TWOPI,
    chirp_mass, symmetric_mass_ratio,
    solar_masses_to_kg,
)


# ============================================================================
# Final Black Hole State — Barausse-Rezzolla Formula
# ============================================================================

def final_spin_barausse_rezzolla(m1, m2, s1z, s2z):
    """Compute the final spin using the Barausse-Rezzolla fitting formula.

    This formula predicts the dimensionless spin a_f = J_f / M_f^2 of the
    remnant black hole from a binary merger. It is calibrated against
    numerical relativity simulations covering mass ratios 1:1 to 1:10
    and spins up to |a| = 0.95.

    The formula accounts for:
    1. Orbital angular momentum at the ISCO (depends on final spin → solved iteratively)
    2. Individual BH spins (aligned components only)
    3. Mass ratio dependence via η

    a_f = |ℓ̃(ã_f, η)| + s₄ η ã² + (s₅ η + t₀) ã + 2√3 η + t₂ η² + t₃ η³

    where ã = (S₁ + S₂) / M² is the reduced total spin and
    ℓ̃ is the specific orbital angular momentum at the ISCO.

    Reference: Barausse & Rezzolla, ApJL 704, L40 (2009), Eq. (1)-(4)

    Parameters
    ----------
    m1, m2 : float
        Component masses in solar masses.
    s1z, s2z : float
        Dimensionless aligned spin components, range [-1, 1].

    Returns
    -------
    float
        Final dimensionless spin a_f.
    """
    M = m1 + m2
    eta = m1 * m2 / M**2
    delta = (m1 - m2) / M

    # Individual spin angular momenta: S_i = m_i^2 * s_iz (geometric units)
    # Reduced total spin:
    a_tilde = (m1**2 * s1z + m2**2 * s2z) / M**2

    # Fitting coefficients — Barausse & Rezzolla (2009), Table 1
    # These were determined by least-squares fit to 38 NR simulations.
    s4 = -0.1229
    s5 = 0.4537
    t0 = -2.8904
    t2 = -3.5171
    t3 = 2.5763

    # The specific orbital angular momentum ℓ̃ at the ISCO of a Kerr BH
    # with spin a_f. This creates an implicit equation for a_f.
    # We solve iteratively starting from a non-spinning estimate.

    def isco_angular_momentum(a):
        """Specific orbital angular momentum at ISCO of a Kerr BH.

        ℓ_ISCO(a) for a prograde orbit.

        Reference: Bardeen, Press & Teukolsky, ApJ 178, 347 (1972), Eq. (2.12)
        """
        # ISCO radius for Kerr
        # Z1 = 1 + (1 - a²)^{1/3} [(1+a)^{1/3} + (1-a)^{1/3}]
        # Z2 = √(3a² + Z1²)
        # r_ISCO = 3 + Z2 - sign(a) √(3-Z1)(3+Z1+2Z2)
        a_clamped = np.clip(a, -0.9999, 0.9999)
        Z1 = 1.0 + (1.0 - a_clamped**2)**(1.0/3.0) * (
            (1.0 + a_clamped)**(1.0/3.0) + (1.0 - a_clamped)**(1.0/3.0)
        )
        Z2 = np.sqrt(3.0 * a_clamped**2 + Z1**2)

        if a_clamped >= 0:
            r_isco = 3.0 + Z2 - np.sqrt((3.0 - Z1) * (3.0 + Z1 + 2.0 * Z2))
        else:
            r_isco = 3.0 + Z2 + np.sqrt((3.0 - Z1) * (3.0 + Z1 + 2.0 * Z2))

        # Specific angular momentum at ISCO
        # ℓ = (r² - 2a√r + a²) / (r^{3/2} - 2r^{1/2} + a)  [Bardeen et al. Eq 2.12]
        sqrt_r = np.sqrt(r_isco)
        ell = (r_isco**2 - 2.0 * a_clamped * sqrt_r + a_clamped**2) / (
            r_isco * sqrt_r - 2.0 * sqrt_r + a_clamped
        )
        return ell

    # Iterative solution: start with a_f = 0 estimate
    a_f = 2.0 * np.sqrt(3.0) * eta  # leading-order estimate (non-spinning)

    for _ in range(20):  # Converges in ~5 iterations
        ell = isco_angular_momentum(a_f)
        a_f_new = (abs(ell * eta)
                   + s4 * eta * a_tilde**2
                   + (s5 * eta + t0) * a_tilde
                   + 2.0 * np.sqrt(3.0) * eta
                   + t2 * eta**2
                   + t3 * eta**3)
        if abs(a_f_new - a_f) < 1e-10:
            break
        a_f = a_f_new

    return np.clip(a_f, 0.0, 0.998)


def final_mass_radiated(m1, m2, s1z, s2z):
    """Compute the final remnant mass and energy radiated.

    Uses the formula from Husa et al. PRD 93, 044006 (2016), which
    fits the radiated energy to NR simulations.

    For GW150914 (m1=36, m2=29), this should give:
        E_rad ≈ 3 M_sun (about 5% of total mass)
        M_f ≈ 62 M_sun

    Reference: Husa et al. PRD 93, 044006 (2016), Eq. (3.7)

    Parameters
    ----------
    m1, m2 : float
        Component masses in solar masses.
    s1z, s2z : float
        Aligned spins.

    Returns
    -------
    dict with keys:
        'final_mass_solar' : float — remnant mass in M_sun
        'energy_radiated_solar' : float — radiated energy in M_sun*c²
        'energy_radiated_fraction' : float — E_rad / (M_total * c²)
        'final_spin' : float — dimensionless spin of remnant
    """
    M = m1 + m2
    eta = m1 * m2 / M**2
    delta = (m1 - m2) / M
    S = (m1**2 * s1z + m2**2 * s2z) / M**2  # reduced total spin

    eta2 = eta * eta
    eta3 = eta2 * eta

    # Energy radiated fitting formula
    # Calibrated to NR; includes higher-order eta terms essential for
    # getting E_rad ~3 Msun for GW150914 (eta=0.247).
    # Reference: Jiménez-Forteza et al., PRD 95, 064024 (2017), Eq. (8)
    #            Husa et al. PRD 93, 044006 (2016), Eq. (3.7a)
    eta4 = eta3 * eta

    # Non-spinning part — higher-order fit captures the steep rise with eta
    E_rad_ns = (0.0559745 * eta
                + 0.580951 * eta2
                - 0.960673 * eta3
                + 3.35241 * eta4)

    # Spin correction — Husa et al. (2016)
    E_rad_spin = (0.1327 * eta * S
                  + 0.0036 * eta * S**2)

    E_rad_frac = E_rad_ns + E_rad_spin

    # Clamp to physical range
    E_rad_frac = np.clip(E_rad_frac, 0.0, 0.15)

    final_mass_solar = M * (1.0 - E_rad_frac)
    final_spin = final_spin_barausse_rezzolla(m1, m2, s1z, s2z)

    return {
        'final_mass_solar': final_mass_solar,
        'energy_radiated_solar': M * E_rad_frac,
        'energy_radiated_fraction': E_rad_frac,
        'final_spin': final_spin,
    }


# ============================================================================
# Quasinormal Mode Frequencies — Berti, Cardoso & Starinets
# ============================================================================

def qnm_frequency(M_f_solar, a_f, l=2, m=2, n=0):
    """Compute the quasinormal mode frequency and damping time.

    Uses the fitting formula from Berti, Cardoso & Starinets (2006),
    Table VIII, which parameterizes the QNM as:

        M_f ω_R = f₁ + f₂ (1 - a_f)^f₃     (oscillation frequency)
        Q       = q₁ + q₂ (1 - a_f)^q₃     (quality factor)

    The quality factor Q = ω_R / (2 |ω_I|) relates to the damping.

    The QNM frequency in Hz is:
        f_QNM = ω_R / (2π G M_f / c³)

    The damping time is:
        τ_QNM = 2 Q / (2π f_QNM) = Q / (π f_QNM)

    Reference: Berti, Cardoso & Starinets, PRD 73, 064030 (2006), Table VIII
               Berti's online tables: https://pages.jh.edu/eberti2/ringdown/

    Parameters
    ----------
    M_f_solar : float
        Final black hole mass in solar masses.
    a_f : float
        Final dimensionless spin, range [0, 1).
    l, m, n : int
        Mode indices: angular (l), azimuthal (m), overtone (n).
        Default (2,2,0) is the dominant fundamental quadrupolar mode.

    Returns
    -------
    dict with keys:
        'f_qnm_hz' : float — QNM oscillation frequency in Hz
        'tau_qnm_s' : float — QNM damping time in seconds
        'quality_factor' : float — Q = omega_R / (2 |omega_I|)
        'omega_r' : float — dimensionless M_f * omega_R
        'omega_i' : float — dimensionless M_f * |omega_I|
    """
    # Fitting coefficients for QNM modes
    # Reference: Berti et al. PRD 73 (2006), Table VIII
    #
    # IMPORTANT: The 'q' coefficients parameterize the QUALITY FACTOR Q,
    # NOT omega_I directly!  Q = q1 + q2*(1-a)^q3
    # Then omega_I = omega_R / (2*Q)
    #
    # Verified at a=0 (Schwarzschild):
    #   M*omega_R = 1.5251 - 1.1568 = 0.3683  (matches 0.37367)
    #   Q = 0.7000 + 1.4187 = 2.1187  (gives omega_I = 0.0869, matches 0.0890)
    qnm_coefficients = {
        (2, 2, 0): {
            'f1': 1.5251, 'f2': -1.1568, 'f3': 0.1292,
            'q1': 0.7000, 'q2': 1.4187, 'q3': -0.4990,
        },
        (2, 2, 1): {  # First overtone
            'f1': 1.3673, 'f2': -1.0260, 'f3': 0.1628,
            'q1': 2.3000, 'q2': 3.6072, 'q3': -0.2277,
        },
        (3, 3, 0): {
            'f1': 1.8956, 'f2': -1.3043, 'f3': 0.1818,
            'q1': 0.9270, 'q2': 1.5685, 'q3': -0.4020,
        },
        (2, 1, 0): {
            'f1': 0.6000, 'f2': -0.2339, 'f3': 0.4175,
            'q1': 0.1896, 'q2': 0.4775, 'q3': -0.2229,
        },
    }

    if (l, m, n) not in qnm_coefficients:
        raise ValueError(f"QNM coefficients not available for (l,m,n) = ({l},{m},{n})")

    coeffs = qnm_coefficients[(l, m, n)]

    # Compute dimensionless QNM frequency and quality factor
    a_clamped = np.clip(a_f, 0.0, 0.9999)

    omega_r = coeffs['f1'] + coeffs['f2'] * (1.0 - a_clamped) ** coeffs['f3']
    Q = coeffs['q1'] + coeffs['q2'] * (1.0 - a_clamped) ** coeffs['q3']

    # Derive omega_I from Q:  Q = omega_R / (2 |omega_I|)
    # Therefore |omega_I| = omega_R / (2 Q)
    omega_i_abs = omega_r / (2.0 * Q)

    # Convert to physical units
    # M_f in geometric time: T_f = G M_f / c^3
    M_f_kg = M_f_solar * MSUN_SI
    T_f = G_SI * M_f_kg / C_SI**3  # seconds

    # QNM frequency: f = omega_R / (2pi T_f)
    f_qnm = omega_r / (TWOPI * T_f)

    # Damping time: tau = T_f / |omega_I| = 2 Q T_f / omega_R
    # Equivalently: tau = Q / (pi f_qnm)
    tau_qnm = 2.0 * Q * T_f / omega_r

    return {
        'f_qnm_hz': f_qnm,
        'tau_qnm_s': tau_qnm,
        'quality_factor': Q,
        'omega_r': omega_r,
        'omega_i': omega_i_abs,
    }


# ============================================================================
# Ringdown Waveform
# ============================================================================

def generate_ringdown_waveform(M_f_solar, a_f, amplitude_scale=1.0,
                                sample_rate=4096, duration=0.1,
                                phi0=0.0, inclination=0.0):
    """Generate the ringdown gravitational waveform.

    The ringdown is a superposition of damped sinusoids (QNMs):

        h(t) = A * exp(-t/τ) * cos(2π f_QNM t + φ₀)     for t ≥ 0

    For the dominant (2,2,0) mode, this produces a simple
    exponentially damped oscillation.

    The plus and cross polarizations are:
        h+(t) = A * (1 + cos²ι)/2 * exp(-t/τ) * cos(2πf t + φ₀)
        h×(t) = A * cos(ι) * exp(-t/τ) * sin(2πf t + φ₀)

    Reference: Echeverria, PRD 40, 3194 (1989), Eq. (1)
               Flanagan & Hughes, PRD 57, 4535 (1998), §V

    Parameters
    ----------
    M_f_solar : float
        Final BH mass in solar masses.
    a_f : float
        Final dimensionless spin.
    amplitude_scale : float
        Overall amplitude (typically matched to merger).
    sample_rate : int
        Sample rate in Hz.
    duration : float
        Duration of ringdown in seconds.
    phi0 : float
        Initial phase at t=0 (matched to merger phase).
    inclination : float
        Orbital inclination angle.

    Returns
    -------
    dict with keys:
        'time' : ndarray — time array (t >= 0)
        'h_plus' : ndarray — plus polarization
        'h_cross' : ndarray — cross polarization
        'f_qnm_hz' : float — QNM frequency
        'tau_qnm_s' : float — damping time
    """
    # Get QNM parameters for the dominant (2,2,0) mode
    qnm = qnm_frequency(M_f_solar, a_f, l=2, m=2, n=0)
    f_qnm = qnm['f_qnm_hz']
    tau = qnm['tau_qnm_s']

    # Time array starting at t=0 (merger)
    dt = 1.0 / sample_rate
    t = np.arange(0, duration, dt)

    # Damped sinusoid
    envelope = amplitude_scale * np.exp(-t / tau)
    cos_iota = np.cos(inclination)

    h_plus = envelope * 0.5 * (1.0 + cos_iota**2) * np.cos(TWOPI * f_qnm * t + phi0)
    h_cross = envelope * cos_iota * np.sin(TWOPI * f_qnm * t + phi0)

    return {
        'time': t,
        'h_plus': h_plus,
        'h_cross': h_cross,
        'f_qnm_hz': f_qnm,
        'tau_qnm_s': tau,
        'quality_factor': qnm['quality_factor'],
    }


def compute_ringdown_params(m1, m2, s1z=0.0, s2z=0.0):
    """Convenience function: compute all ringdown parameters from binary masses.

    Parameters
    ----------
    m1, m2 : float
        Component masses in solar masses.
    s1z, s2z : float
        Aligned spin components.

    Returns
    -------
    dict
        Complete ringdown parameter set.
    """
    remnant = final_mass_radiated(m1, m2, s1z, s2z)
    M_f = remnant['final_mass_solar']
    a_f = remnant['final_spin']

    qnm_220 = qnm_frequency(M_f, a_f, l=2, m=2, n=0)

    return {
        **remnant,
        **qnm_220,
        'M_f_solar': M_f,
        'a_f': a_f,
    }
