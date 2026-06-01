"""
Post-Newtonian Inspiral Waveform Generator
============================================

Generates the gravitational wave strain h(t) from the inspiral phase of
a compact binary coalescence using the 3.5 post-Newtonian (PN) approximation.

The inspiral is modeled as an adiabatic sequence of quasi-circular orbits:
    1. Compute the orbital binding energy E(v) to 3PN order
    2. Compute the gravitational wave energy flux F(v) to 3.5PN order
    3. Use the energy balance equation dE/dt = -F to evolve the orbital frequency
    4. Integrate the phase Φ(t) = ∫ ω(t') dt' to build the time-domain strain

The PN expansion parameter is v = (π M f)^(1/3), the orbital velocity in
geometric units (c=1). Here M is the total mass and f is the GW frequency.

References
----------
[1] Blanchet, "Gravitational Radiation from Post-Newtonian Sources and
    Inspiralling Compact Binaries", Living Reviews in Relativity 17, 2 (2014).
    Primary reference for all PN coefficients.
    https://doi.org/10.12942/lrr-2014-2

[2] Cutler & Flanagan, "Gravitational waves from merging compact binaries:
    How accurately can one extract the binary's parameters from the inspiral
    waveform?", PRD 49, 2658 (1994).
    Chirp mass definition, signal model.

[3] Arun et al., "Parameter estimation of inspiralling compact binaries
    using 3.5 post-Newtonian gravitational wave phasing", PRD 71, 084008 (2005).
    3.5PN time-domain phasing coefficients.

[4] Kidder, "Coalescing binary systems of compact objects to
    (post)^{5/2}-Newtonian order. V. Spin effects",
    PRD 52, 821 (1995). Spin-orbit coupling terms.

[5] Maggiore, "Gravitational Waves: Theory and Experiments", Vol. 1,
    Oxford University Press (2007). Textbook reference for antenna patterns.

[6] Poisson & Will, "Gravitational waves from inspiralling compact binaries:
    Parameter estimation using second-post-Newtonian waveforms",
    PRD 52, 848 (1995). Higher-order amplitude corrections.
"""

import numpy as np
from scipy.integrate import solve_ivp

from gravitational_wave_analyzer.constants import (
    G_SI, C_SI, MSUN_SI, MPC_SI, PI, TWOPI, PI_SQ,
    EULER_GAMMA, TSUN_SI,
    chirp_mass, symmetric_mass_ratio, effective_spin, isco_frequency,
    solar_masses_to_kg, mpc_to_meters,
)


# ============================================================================
# 3.5 Post-Newtonian Energy and Flux
# ============================================================================

def orbital_energy_3pn(v, eta):
    """Compute the orbital binding energy E(v) to 3PN order.

    E(v) = -(1/2) * η * v^2 * [1 + E_2 v^2 + E_3 v^3 + E_4 v^4 + E_5 v^5 + E_6 v^6]

    where v = (π M f)^{1/3} is the PN expansion parameter.

    Reference: Blanchet, LRR 17, 2 (2014), Eq. (234).
    The energy is in geometric units (G = c = 1, mass M = 1).

    Parameters
    ----------
    v : float or array
        PN velocity parameter v = (π M f)^(1/3).
    eta : float
        Symmetric mass ratio η = m1*m2 / (m1+m2)^2.

    Returns
    -------
    float or array
        Binding energy E(v) in geometric units, divided by (η * M * c^2).
        Multiply by η * M * c^2 to get SI energy.
    """
    # 1PN coefficient — Blanchet LRR 2014, Eq. (234)
    E2 = -(3.0 / 4.0) - (1.0 / 12.0) * eta

    # 2PN coefficient
    E4 = -(27.0 / 8.0) + (19.0 / 8.0) * eta - (1.0 / 24.0) * eta**2

    # 3PN coefficient — includes the crucial λ = -1987/3080 ambiguity parameter
    # resolved by dimensional regularization
    E6 = (-(675.0 / 64.0)
          + (34445.0 / 576.0 - 205.0 / 96.0 * PI_SQ) * eta
          - (155.0 / 96.0) * eta**2
          - (35.0 / 5184.0) * eta**3)

    v2 = v * v
    v4 = v2 * v2
    v6 = v4 * v2

    return -0.5 * eta * v2 * (1.0 + E2 * v2 + E4 * v4 + E6 * v6)


def gw_luminosity_3p5pn(v, eta):
    """Compute the gravitational wave energy flux F(v) to 3.5PN order.

    F(v) = (32/5) * η^2 * v^10 * [1 + F_2 v^2 + F_3 v^3 + ... + F_7 v^7]

    This is the total power radiated in gravitational waves, computed from
    the multipole moments of the source. The 3.5PN accuracy includes:
    - Instantaneous (local in time) contributions to 3PN
    - Hereditary tail contributions at 1.5PN, 2.5PN, 3PN, 3.5PN
    - Tail-of-tail contributions at 3PN

    Reference: Blanchet, LRR 17, 2 (2014), Eq. (314).

    Parameters
    ----------
    v : float or array
        PN velocity parameter.
    eta : float
        Symmetric mass ratio.

    Returns
    -------
    float or array
        Energy flux in geometric units, divided by (c^5 / G).
    """
    # --- PN flux coefficients ---
    # Each coefficient F_n corresponds to v^n relative to the Newtonian flux.

    # 1PN (v^2) — Blanchet LRR Eq. (314)
    F2 = -(1247.0 / 336.0) - (35.0 / 12.0) * eta

    # 1.5PN (v^3) — tail contribution (hereditary)
    # The tail arises from backscattering of GWs off the background curvature.
    F3 = 4.0 * PI

    # 2PN (v^4)
    F4 = -(44711.0 / 9072.0) + (9271.0 / 504.0) * eta + (65.0 / 18.0) * eta**2

    # 2.5PN (v^5) — tail at next order
    F5 = -(8191.0 / 672.0) * PI - (583.0 / 24.0) * eta * PI

    # 3PN (v^6) — includes tail-of-tail and memory contributions
    # The log(v) term arises from tail-of-tail interactions.
    # We handle the log term separately in the calculation below.
    F6_const = (6643739519.0 / 69854400.0
                + 16.0 / 3.0 * PI_SQ
                - 1712.0 / 105.0 * EULER_GAMMA
                - 856.0 / 105.0 * np.log(16.0)  # log term evaluated at v=1 scale
                + (-134543.0 / 7776.0 + 41.0 / 48.0 * PI_SQ) * eta
                - 94403.0 / 3024.0 * eta**2
                - 775.0 / 324.0 * eta**3)
    F6_log_coeff = -1712.0 / 105.0  # coefficient of log(v)

    # 3.5PN (v^7)
    F7 = (-(16285.0 / 504.0) + (214745.0 / 1728.0) * eta
          + (193385.0 / 3024.0) * eta**2) * PI

    v2 = v * v
    v3 = v2 * v
    v4 = v2 * v2
    v5 = v4 * v
    v6 = v4 * v2
    v7 = v6 * v

    # Avoid log(0) — v should always be positive
    log_v = np.log(v) if v > 0 else 0.0

    flux = (32.0 / 5.0) * eta**2 * v**10 * (
        1.0
        + F2 * v2
        + F3 * v3
        + F4 * v4
        + F5 * v5
        + (F6_const + F6_log_coeff * log_v) * v6
        + F7 * v7
    )

    return flux


def dv_dt_3p5pn(v, eta):
    """Compute dv/dt from the energy balance equation.

    The energy balance: dE/dt = -F(v) implies
        dv/dt = -F(v) / (dE/dv)

    where dE/dv is computed analytically from the 3PN energy.

    Parameters
    ----------
    v : float
        PN velocity parameter.
    eta : float
        Symmetric mass ratio.

    Returns
    -------
    float
        Time derivative of v in geometric units (1/M).
    """
    # Compute dE/dv analytically from E(v)
    # E(v) = -(1/2) η v^2 [1 + E2 v^2 + E4 v^4 + E6 v^6]
    # dE/dv = -η v [1 + 2*E2 v^2 + 3*E4 v^4 + 4*E6 v^6] (approximate)
    # More precisely, differentiate term by term:

    E2 = -(3.0 / 4.0) - (1.0 / 12.0) * eta
    E4 = -(27.0 / 8.0) + (19.0 / 8.0) * eta - (1.0 / 24.0) * eta**2
    E6 = (-(675.0 / 64.0)
          + (34445.0 / 576.0 - 205.0 / 96.0 * PI_SQ) * eta
          - (155.0 / 96.0) * eta**2
          - (35.0 / 5184.0) * eta**3)

    v2 = v * v
    v4 = v2 * v2
    v6 = v4 * v2

    # dE/dv = -η * v * (1 + 2*E2*v^2 + 3*E4*v^4 + 4*E6*v^6)
    # Factor of 2 in first term: d/dv [-(1/2) η v^2] = -η v
    # d/dv [-(1/2) η v^2 * E2 v^2] = -(1/2) η E2 * 4v^3 = -2 η E2 v^3
    # etc.
    dEdv = -eta * v * (1.0 + 2.0 * E2 * v2 + 3.0 * E4 * v4 + 4.0 * E6 * v6)

    F = gw_luminosity_3p5pn(v, eta)

    if abs(dEdv) < 1e-100:
        return 0.0

    return -F / dEdv


# ============================================================================
# Spin Corrections to the Phase Evolution
# ============================================================================

def spin_phase_correction(v, eta, chi_eff, chi_s, chi_a, delta_m):
    """Compute spin corrections to the GW phasing.

    Spin effects enter the phase primarily through spin-orbit coupling
    (at 1.5PN) and spin-spin coupling (at 2PN).

    χ_s = (s1z + s2z) / 2 — symmetric spin combination
    χ_a = (s1z - s2z) / 2 — antisymmetric spin combination
    δ = (m1 - m2) / (m1 + m2) — mass asymmetry parameter

    Reference: Kidder, PRD 52, 821 (1995), Eqs. (2.4)-(2.9)
               Arun et al., PRD 71, 084008 (2005), Appendix A

    Parameters
    ----------
    v : float
        PN velocity parameter.
    eta : float
        Symmetric mass ratio.
    chi_eff : float
        Effective spin parameter.
    chi_s : float
        Symmetric spin: (s1z + s2z) / 2.
    chi_a : float
        Antisymmetric spin: (s1z - s2z) / 2.
    delta_m : float
        Mass asymmetry parameter: (m1 - m2) / (m1 + m2).

    Returns
    -------
    float
        Spin contribution to the accumulated phase (radians).
    """
    v2 = v * v
    v3 = v2 * v
    v4 = v2 * v2
    v5 = v4 * v

    # 1.5PN spin-orbit — Kidder PRD 52 (1995), Eq. (2.4a)
    # Leading spin effect in the phase. Enters through the energy flux.
    beta_so = (113.0 / 12.0) * chi_eff  # simplified for aligned spins

    # 2PN spin-spin — Kidder PRD 52 (1995), Eq. (2.4b)
    # Self-interaction of each BH's spin with the orbital angular momentum.
    sigma_ss = -(247.0 / 48.0) * eta * chi_s**2

    # Spin corrections to dΦ/dt (accumulated over orbital evolution)
    # These multiply into the TaylorT4 phasing formula.
    # Returned as a multiplicative correction factor to the flux.
    phase_corr = beta_so * v3 + sigma_ss * v4

    return phase_corr


# ============================================================================
# Antenna Pattern Functions
# ============================================================================

def antenna_pattern(ra, dec, psi, detector='H1'):
    """Compute the antenna response functions F+ and Fx for a GW detector.

    A laser interferometer has a quadrupolar antenna pattern determined
    by the angle between the GW propagation direction and the detector arms.

    For LIGO Hanford (H1):
        Latitude:  46.455° N
        Longitude: 119.408° W
        Arm orientation: 126° from North (x-arm) and 36° from North (y-arm)

    For LIGO Livingston (L1):
        Latitude:  30.563° N
        Longitude: 90.774° W
        Arm orientation: 108° from North (x-arm) and 18° from North (y-arm)

    The antenna pattern functions are:
        F+(θ, φ, ψ) = (1/2)(1 + cos²θ) cos(2φ) cos(2ψ) - cosθ sin(2φ) sin(2ψ)
        Fx(θ, φ, ψ) = (1/2)(1 + cos²θ) cos(2φ) sin(2ψ) + cosθ sin(2φ) cos(2ψ)

    where θ is the polar angle, φ is the azimuthal angle in the detector frame,
    and ψ is the GW polarization angle.

    Reference: Maggiore, "Gravitational Waves" Vol.1 (2007), Eqs. (7.27)-(7.31)
               Jaranowski, Królak & Schutz, PRD 58, 063001 (1998)

    Parameters
    ----------
    ra : float
        Right ascension of GW source (radians).
    dec : float
        Declination of GW source (radians).
    psi : float
        Polarization angle (radians).
    detector : str
        Detector identifier: 'H1' (Hanford) or 'L1' (Livingston).

    Returns
    -------
    tuple (F_plus, F_cross)
        Antenna response functions, dimensionless, range [-1, 1].
    """
    # For simplicity, we use the "long-wavelength limit" antenna pattern
    # which is valid when the GW wavelength >> detector arm length (always true
    # for LIGO's 4km arms and GW frequencies < 10 kHz).

    # Transform (ra, dec) to detector-frame angles (θ, φ).
    # For a simplified treatment (ignoring Earth rotation and sidereal time),
    # we use θ = π/2 - dec, φ = ra as an approximation.
    # A full treatment would include Greenwich Mean Sidereal Time (GMST).
    theta = PI / 2.0 - dec
    phi = ra

    cos_theta = np.cos(theta)
    cos_2phi = np.cos(2.0 * phi)
    sin_2phi = np.sin(2.0 * phi)
    cos_2psi = np.cos(2.0 * psi)
    sin_2psi = np.sin(2.0 * psi)

    # Antenna pattern functions — Maggiore Eq. (7.31)
    F_plus = (0.5 * (1.0 + cos_theta**2) * cos_2phi * cos_2psi
              - cos_theta * sin_2phi * sin_2psi)

    F_cross = (0.5 * (1.0 + cos_theta**2) * cos_2phi * sin_2psi
               + cos_theta * sin_2phi * cos_2psi)

    return F_plus, F_cross


# ============================================================================
# Time-Domain Inspiral Waveform Generator (TaylorT4)
# ============================================================================

def generate_inspiral_waveform(m1_solar, m2_solar, s1z=0.0, s2z=0.0,
                                distance_mpc=410.0, inclination=0.0,
                                f_lower=20.0, sample_rate=4096,
                                ra=0.0, dec=0.0, psi=0.0):
    """Generate the time-domain inspiral waveform using TaylorT4 approximant.

    TaylorT4 numerically integrates the coupled ODEs for orbital phase and
    frequency, using the 3.5PN-accurate energy balance:

        dv/dt = -F(v) / (dE/dv)       — frequency evolution
        dΦ/dt = v^3 / M               — phase evolution

    The integration starts at f_lower (20 Hz) and terminates at the ISCO
    frequency or when the PN approximation breaks down (v ≈ 0.4).

    This is the TaylorT4 approximant as defined in:
        Buonanno et al., PRD 80, 084043 (2009), §II.B

    Parameters
    ----------
    m1_solar : float
        Mass of primary BH in solar masses.
    m2_solar : float
        Mass of secondary BH in solar masses.
    s1z : float
        Dimensionless spin of primary along L (-1 to 1).
    s2z : float
        Dimensionless spin of secondary along L (-1 to 1).
    distance_mpc : float
        Luminosity distance in Megaparsecs.
    inclination : float
        Orbital inclination angle (radians). 0 = face-on, π/2 = edge-on.
    f_lower : float
        Starting GW frequency in Hz.
    sample_rate : int
        Output sample rate in Hz.
    ra : float
        Right ascension (radians).
    dec : float
        Declination (radians).
    psi : float
        Polarization angle (radians).

    Returns
    -------
    dict with keys:
        'time' : ndarray — time array in seconds (merger at t=0)
        'h_plus' : ndarray — plus polarization strain
        'h_cross' : ndarray — cross polarization strain
        'frequency' : ndarray — instantaneous GW frequency in Hz
        'phase' : ndarray — accumulated GW phase in radians
        'h_detector' : ndarray — detector-frame strain h(t) = F+ h+ + Fx hx
        'params' : dict — computed physical parameters
    """
    # --- Convert to SI units ---
    m1_kg = solar_masses_to_kg(m1_solar)
    m2_kg = solar_masses_to_kg(m2_solar)
    M_total_kg = m1_kg + m2_kg
    M_total_solar = m1_solar + m2_solar
    d_L = mpc_to_meters(distance_mpc)

    # --- Compute binary parameters ---
    eta = symmetric_mass_ratio(m1_solar, m2_solar)
    Mc = chirp_mass(m1_solar, m2_solar)
    Mc_kg = solar_masses_to_kg(Mc)
    chi_eff_val = effective_spin(m1_solar, m2_solar, s1z, s2z)

    # Symmetric and antisymmetric spin combinations
    chi_s = 0.5 * (s1z + s2z)
    chi_a = 0.5 * (s1z - s2z)
    delta_m = (m1_solar - m2_solar) / M_total_solar

    # ISCO frequency (Schwarzschild)
    f_isco = isco_frequency(M_total_solar)

    # Total mass in geometric time units: M_geo = G * M_total / c^3
    M_geo = G_SI * M_total_kg / C_SI**3  # seconds

    # Starting velocity from f_lower
    # v = (π M f)^(1/3) in geometric units
    v_start = (PI * M_geo * f_lower) ** (1.0 / 3.0)

    # ISCO velocity — v_ISCO = (π M f_ISCO)^(1/3) = 1/√6 ≈ 0.408
    v_isco = (PI * M_geo * f_isco) ** (1.0 / 3.0)

    # Maximum velocity: stop at ISCO or when PN breaks down
    v_max = min(v_isco, 0.45)

    # --- ODE system for TaylorT4 ---
    # State vector: y = [Φ (GW phase), v (PN velocity)]
    # dΦ/dt = 2π f = 2 v^3 / M_geo      (GW phase = 2 × orbital phase)
    # dv/dt = -F(v) / (dE/dv)

    def ode_rhs(t, y):
        """Right-hand side of the TaylorT4 ODE system.

        The factor of 2 in dΦ/dt is because the GW frequency is twice
        the orbital frequency for the dominant (l=2, m=2) mode.
        """
        phase_val, v_val = y

        if v_val <= 0 or v_val >= v_max:
            return [0.0, 0.0]

        # GW phase evolution: dΦ/dt = 2 * v^3 / M_geo
        dphi_dt = 2.0 * v_val**3 / M_geo

        # Velocity evolution from energy balance
        dvdt = dv_dt_3p5pn(v_val, eta)

        # Add spin corrections to the velocity evolution
        # Spin modifies the flux, which modifies dv/dt
        # Leading spin-orbit at 1.5PN: extra term in the flux
        # Reference: Kidder PRD 52 (1995), Eq. (2.4a)
        beta_so = (113.0 / 12.0 - 19.0 / 3.0 * eta) * chi_eff_val
        spin_flux_correction = 1.0 + beta_so * v_val**3

        # The spin correction modifies dv/dt multiplicatively at leading order
        dvdt *= spin_flux_correction

        # Convert from geometric time (units of M) to SI seconds
        # In geometric units, dt_geo = dt_SI / M_geo
        # So dv/dt_SI = dv/dt_geo / M_geo
        dvdt /= M_geo

        return [dphi_dt, dvdt]

    # --- Termination event: stop at v_max ---
    def v_reached_max(t, y):
        return y[1] - v_max
    v_reached_max.terminal = True
    v_reached_max.direction = 1

    # --- Estimate integration time ---
    # Leading order: t_merger ≈ (5/256) M_geo / (η v^8)
    # Reference: Maggiore Eq. (4.21)
    t_merge_estimate = (5.0 / 256.0) * M_geo / (eta * v_start**8)

    # Add 20% margin to ensure we reach ISCO
    t_max = 1.2 * t_merge_estimate

    # --- Initial conditions ---
    y0 = [0.0, v_start]  # [Φ(0) = 0, v(0) = v_start]

    # --- Integrate with adaptive RK45 ---
    dt = 1.0 / sample_rate
    t_eval = np.arange(0, t_max, dt)

    sol = solve_ivp(
        ode_rhs, [0, t_max], y0,
        method='RK45',
        t_eval=t_eval,
        rtol=1e-10,    # High accuracy for phase coherence
        atol=1e-12,
        events=[v_reached_max],
        max_step=dt,   # Ensure we don't skip samples
        dense_output=True,
    )

    if sol.status == -1:
        raise RuntimeError(f"ODE integration failed: {sol.message}")

    t_raw = sol.t
    phi_raw = sol.y[0]
    v_raw = sol.y[1]

    # --- Compute GW frequency from v ---
    # f_GW = v^3 / (π M_geo) — this is the GW frequency (2 × orbital)
    f_raw = v_raw**3 / (PI * M_geo)

    # --- Compute strain amplitude ---
    # The leading-order (Newtonian) strain amplitude in the time domain:
    #
    #   h₀(t) = 4 (G Mc / c²)^(5/4) (π f / c)^(2/3) / d_L
    #
    # This is the "restricted" PN waveform: Newtonian amplitude but
    # PN-accurate phase. This is standard practice because phase accuracy
    # matters far more than amplitude accuracy for matched filtering.
    #
    # Reference: Cutler & Flanagan PRD 49 (1994), Eq. (2.3)

    Mc_meters = G_SI * Mc_kg / C_SI**2  # chirp mass in meters (geometric length)

    # Amplitude: A(f) = (4/d_L) * (G Mc / c^2)^(5/4) * (π f / c)^(2/3)
    # But it's cleaner to compute via v:
    # A = (4 η M_geo c) / d_L * v^2
    # since h ~ (M/r) v^2 at leading order
    amplitude = 4.0 * eta * C_SI * M_geo * v_raw**2 / d_L

    # --- Compute both polarizations ---
    # h+(t) = A(t) * (1 + cos²ι)/2 * cos(Φ(t))
    # hx(t) = A(t) * cos(ι) * sin(Φ(t))
    #
    # ι is the inclination angle between the orbital angular momentum
    # and the line of sight. Face-on (ι=0) gives circular polarization.
    # Edge-on (ι=π/2) gives linear polarization (only h+).
    #
    # Reference: Maggiore Vol.1 (2007), Eq. (4.16)

    cos_iota = np.cos(inclination)
    cos_iota_sq = cos_iota**2

    h_plus = amplitude * 0.5 * (1.0 + cos_iota_sq) * np.cos(phi_raw)
    h_cross = amplitude * cos_iota * np.sin(phi_raw)

    # --- Detector response ---
    F_plus, F_cross = antenna_pattern(ra, dec, psi)
    h_detector = F_plus * h_plus + F_cross * h_cross

    # --- Shift time so that the end of inspiral is at t = 0 ---
    t_shifted = t_raw - t_raw[-1]

    # --- Package results ---
    params = {
        'chirp_mass_solar': Mc,
        'chirp_mass_kg': Mc_kg,
        'symmetric_mass_ratio': eta,
        'total_mass_solar': M_total_solar,
        'total_mass_kg': M_total_kg,
        'effective_spin': chi_eff_val,
        'f_isco_hz': f_isco,
        'f_start_hz': f_lower,
        'f_end_hz': float(f_raw[-1]),
        'duration_seconds': float(t_raw[-1] - t_raw[0]),
        'v_start': float(v_start),
        'v_end': float(v_raw[-1]),
        'distance_mpc': distance_mpc,
        'distance_meters': d_L,
        'inclination': inclination,
        'F_plus': F_plus,
        'F_cross': F_cross,
        'num_cycles': float(phi_raw[-1] / TWOPI),
        'peak_strain': float(np.max(np.abs(h_detector))),
    }

    return {
        'time': t_shifted,
        'h_plus': h_plus,
        'h_cross': h_cross,
        'h_detector': h_detector,
        'frequency': f_raw,
        'phase': phi_raw,
        'velocity': v_raw,
        'amplitude': amplitude,
        'params': params,
    }
