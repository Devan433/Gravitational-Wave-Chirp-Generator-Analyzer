"""
TaylorF2 Stationary-Phase Approximation Waveform
==================================================

Implements the TaylorF2 frequency-domain waveform model for compact
binary inspiral. This is the simplest and fastest waveform model used
in gravitational wave data analysis.

TaylorF2 uses the **stationary phase approximation (SPA)** to express
the Fourier transform of the inspiral waveform analytically:

    h̃(f) = A f^{-7/6} exp(i Ψ(f))

where the phase Ψ(f) is expanded in powers of v = (π M f)^{1/3} to
3.5 post-Newtonian order. The amplitude uses the leading Newtonian
(restricted PN) approximation.

Key properties:
    - **Inspiral only**: valid up to ~ISCO frequency, NO merger/ringdown
    - **Extremely fast**: pure analytic formula, no ODE integration
    - **Gold standard for template banks**: most LIGO template banks use
      TaylorF2 because of its speed and accuracy during inspiral
    - **Diverges from IMRPhenomD near merger**: this is expected and
      demonstrates why full IMR models were developed

References
----------
[1] Cutler & Flanagan, "Gravitational waves from merging compact binaries:
    How accurately can one extract the binary's parameters from the inspiral
    waveform?", PRD 49, 2658 (1994).
    — Defines the SPA waveform and chirp mass.

[2] Arun et al., "Parameter estimation of inspiralling compact binaries
    using 3.5 post-Newtonian gravitational wave phasing",
    PRD 71, 084008 (2005).
    — Complete 3.5PN SPA phase coefficients.

[3] Buonanno, Iyer, Ochsner, Pan & Sathyaprakash, "Comparison of
    post-Newtonian templates for compact binary inspiral signals in
    gravitational-wave detectors", PRD 80, 084043 (2009).
    — Systematic comparison of PN approximants including TaylorF2.

[4] Poisson & Will, PRD 52, 848 (1995).
    — 2PN phasing coefficients.
"""

import numpy as np
from scipy.fft import irfft

from gravitational_wave_analyzer.constants import (
    G_SI, C_SI, MSUN_SI, MPC_SI, PI, TWOPI, PI_SQ,
    EULER_GAMMA,
    chirp_mass, symmetric_mass_ratio, effective_spin,
    solar_masses_to_kg, mpc_to_meters, isco_frequency,
)


def taylorf2_phase_coefficients(eta, chi_eff=0.0):
    """Compute the TaylorF2 SPA phase coefficients to 3.5PN order.

    The SPA phase is:
        Ψ(f) = 2π f t_c - φ_c + Σ_k ψ_k v^{k-5}

    where v = (π M f)^{1/3} and k runs from 0 to 7 (0PN to 3.5PN).

    Reference: Arun et al. PRD 71, 084008 (2005), Eq. (3.18)
               Buonanno et al. PRD 80, 084043 (2009), Eq. (A1)

    Parameters
    ----------
    eta : float
        Symmetric mass ratio.
    chi_eff : float
        Effective spin parameter.

    Returns
    -------
    list of float
        Phase coefficients [ψ_0, ψ_1, ψ_2, ψ_3, ψ_4, ψ_5, ψ_5l, ψ_6, ψ_6l, ψ_7]
    """
    eta2 = eta * eta
    eta3 = eta2 * eta

    # 0PN (Newtonian) — the dominant chirp phase
    # This term alone gives the "chirp" character of the waveform.
    psi_0 = 3.0 / (128.0 * eta)

    # 0.5PN — zero by time-reversal symmetry
    psi_1 = 0.0

    # 1PN
    psi_2 = 3.0 / (128.0 * eta) * (
        20.0 / 9.0 * (743.0 / 336.0 + 11.0 / 4.0 * eta)
    )

    # 1.5PN — includes spin-orbit coupling
    # The 4π term is from gravitational-wave tails (hereditary effect)
    psi_3 = 3.0 / (128.0 * eta) * (
        -16.0 * PI
        + (113.0 / 3.0 - 76.0 / 3.0 * eta) * chi_eff
    )

    # 2PN
    psi_4 = 3.0 / (128.0 * eta) * (
        10.0 * (3058673.0 / 1016064.0 + 5429.0 / 1008.0 * eta
                + 617.0 / 144.0 * eta2)
    )

    # 2.5PN — tail contribution
    psi_5 = 3.0 / (128.0 * eta) * (
        PI * (38645.0 / 756.0 - 65.0 / 9.0 * eta)
    )

    # 2.5PN logarithmic term
    psi_5l = 3.0 / (128.0 * eta) * (
        PI * (38645.0 / 756.0 - 65.0 / 9.0 * eta) * 3.0
    )

    # 3PN — includes tail-of-tail
    psi_6 = 3.0 / (128.0 * eta) * (
        11583231236531.0 / 4694215680.0
        - 640.0 / 3.0 * PI_SQ
        - 6848.0 / 21.0 * EULER_GAMMA
        - 6848.0 / 63.0 * np.log(4.0)
        + (-15737765635.0 / 3048192.0 + 2255.0 / 12.0 * PI_SQ) * eta
        + 76055.0 / 1728.0 * eta2
        - 127825.0 / 1296.0 * eta3
    )

    # 3PN logarithmic term
    psi_6l = 3.0 / (128.0 * eta) * (
        -6848.0 / 63.0
    )

    # 3.5PN
    psi_7 = 3.0 / (128.0 * eta) * (
        PI * (77096675.0 / 254016.0 + 378515.0 / 1512.0 * eta
              - 74045.0 / 756.0 * eta2)
    )

    return [psi_0, psi_1, psi_2, psi_3, psi_4, psi_5, psi_5l, psi_6, psi_6l, psi_7]


def generate_taylorf2_waveform(m1_solar, m2_solar, s1z=0.0, s2z=0.0,
                                distance_mpc=410.0, inclination=0.0,
                                f_lower=20.0, sample_rate=4096,
                                ra=0.0, dec=0.0, psi=0.0):
    """Generate the TaylorF2 inspiral waveform.

    Computes the frequency-domain waveform analytically using the
    stationary phase approximation, then IFFTs to time domain.

    The waveform is:
        h̃(f) = A f^{-7/6} exp(i Ψ(f))

    where A encodes the distance, masses, and inclination, and
    Ψ(f) is the PN phase expansion.

    IMPORTANT: This model is inspiral-only. It is valid up to
    approximately the ISCO frequency and does NOT include merger
    or ringdown. Comparing with IMRPhenomD demonstrates exactly
    where and why PN breaks down.

    Parameters
    ----------
    m1_solar, m2_solar : float
        Component masses in solar masses.
    s1z, s2z : float
        Aligned spin components.
    distance_mpc : float
        Luminosity distance in Mpc.
    inclination : float
        Orbital inclination in radians.
    f_lower : float
        Starting frequency in Hz.
    sample_rate : int
        Output sample rate in Hz.
    ra, dec, psi : float
        Sky position and polarization.

    Returns
    -------
    dict — Waveform data compatible with the full_waveform interface:
        'time', 'h_plus', 'h_cross', 'h_detector', 'frequency',
        'phase', 'amplitude', 'params'
    """
    # --- Physical parameters ---
    m1_kg = solar_masses_to_kg(m1_solar)
    m2_kg = solar_masses_to_kg(m2_solar)
    M_total_kg = m1_kg + m2_kg
    M_total_solar = m1_solar + m2_solar
    d_L = mpc_to_meters(distance_mpc)

    eta = symmetric_mass_ratio(m1_solar, m2_solar)
    Mc = chirp_mass(m1_solar, m2_solar)
    Mc_kg = solar_masses_to_kg(Mc)
    chi_eff_val = effective_spin(m1_solar, m2_solar, s1z, s2z)

    # Total mass in geometric time: M_geo = G M / c^3
    M_geo = G_SI * M_total_kg / C_SI**3

    # ISCO frequency — TaylorF2 terminates here
    f_isco = isco_frequency(M_total_solar)
    f_max = min(f_isco, sample_rate / 2.0)

    # --- Frequency-domain setup ---
    # Duration estimate (leading order) — used to set FFT length
    # t_inspiral ≈ (5/256) * M_chirp^{-5/3} * (π f_lower)^{-8/3} / η
    Mc_geo = G_SI * Mc_kg / C_SI**3
    t_inspiral = (5.0 / 256.0) * (PI * f_lower * M_geo)**(-8.0/3.0) * M_geo / eta
    t_inspiral = min(t_inspiral, 60.0)  # cap at 60 seconds for very low masses

    # FFT length: ensure adequate frequency resolution
    N = int(max(2.0 * t_inspiral, 4.0) * sample_rate)
    N = 2 ** int(np.ceil(np.log2(N)))  # next power of 2

    dt = 1.0 / sample_rate
    df = 1.0 / (N * dt)
    N_pos = N // 2 + 1  # positive frequency bins for irfft

    # Frequency array (positive frequencies only for irfft)
    freqs = np.arange(N_pos) * df

    # --- SPA Phase Ψ(f) ---
    coeffs = taylorf2_phase_coefficients(eta, chi_eff_val)
    psi_0, psi_1, psi_2, psi_3, psi_4, psi_5, psi_5l, psi_6, psi_6l, psi_7 = coeffs

    # Build phase for f >= f_lower and f <= f_max
    mask = (freqs >= f_lower) & (freqs <= f_max)
    f_active = freqs[mask]

    if len(f_active) == 0:
        raise ValueError(f"No frequencies in band [{f_lower}, {f_max}] Hz. "
                         f"Check mass range (ISCO={f_isco:.1f} Hz).")

    # v = (π M f)^{1/3} — the PN expansion parameter
    v = (PI * M_geo * f_active) ** (1.0 / 3.0)
    v2 = v * v
    v3 = v2 * v
    v4 = v2 * v2
    v5 = v4 * v
    v6 = v4 * v2
    v7 = v6 * v
    log_v = np.log(v)

    # Assemble the SPA phase
    # Ψ(f) = (3/(128η)) v^{-5} [1 + ψ_2 v^2 + ψ_3 v^3 + ...]
    # We compute each term in the v expansion
    phase = (psi_0 * v**(-5)
             + psi_1 * v**(-4)
             + psi_2 * v**(-3)
             + psi_3 * v**(-2)
             + psi_4 * v**(-1)
             + psi_5 * np.log(v)   # 2.5PN log term
             + psi_5 * 1.0         # 2.5PN constant part
             + psi_6 * v
             + psi_6l * v * log_v
             + psi_7 * v2)

    # Apply the conventional SPA prefactor
    phase = TWOPI * f_active * 0.0 - PI / 4.0 + phase  # t_c=0, φ_c=0

    # --- SPA Amplitude ---
    # A(f) = √(5/24) π^{-2/3} (G Mc / c^3)^{5/6} / d_L × f^{-7/6}
    # Reference: Cutler & Flanagan PRD 49 (1994), Eq. (3.4)
    amp_prefactor = (np.sqrt(5.0 / 24.0) / (PI ** (2.0/3.0))
                     * (G_SI * Mc_kg / C_SI**3) ** (5.0/6.0)
                     * C_SI / d_L)

    amplitude_fd = amp_prefactor * f_active ** (-7.0/6.0)

    # Inclination-dependent polarizations in frequency domain
    cos_iota = np.cos(inclination)
    Fp = 0.5 * (1.0 + cos_iota**2)
    Fc = cos_iota

    # --- Build complex frequency-domain signal ---
    h_plus_fd = np.zeros(N_pos, dtype=complex)
    h_cross_fd = np.zeros(N_pos, dtype=complex)

    h_plus_fd[mask] = amplitude_fd * Fp * np.exp(1j * phase)
    h_cross_fd[mask] = amplitude_fd * Fc * np.exp(1j * (phase + PI / 2.0))

    # --- IFFT to time domain ---
    h_plus_td = irfft(h_plus_fd, n=N) * df * N  # normalize
    h_cross_td = irfft(h_cross_fd, n=N) * df * N

    # --- Trim to the physical signal region ---
    # Find the region where the signal has significant amplitude
    amplitude_td = np.sqrt(h_plus_td**2 + h_cross_td**2)
    max_amp = np.max(amplitude_td)

    if max_amp > 0:
        # Find where signal exceeds 0.1% of peak
        threshold = 0.001 * max_amp
        signal_mask = amplitude_td > threshold
        if np.any(signal_mask):
            indices = np.where(signal_mask)[0]
            start_idx = max(0, indices[0] - 100)
            end_idx = min(N, indices[-1] + 100)
        else:
            start_idx = 0
            end_idx = N
    else:
        start_idx = 0
        end_idx = N

    # Trim
    h_plus_td = h_plus_td[start_idx:end_idx]
    h_cross_td = h_cross_td[start_idx:end_idx]
    amplitude_td = amplitude_td[start_idx:end_idx]
    n_samples = len(h_plus_td)

    # Time array — place the end (ISCO) at t=0
    time_arr = np.arange(n_samples) * dt
    time_arr = time_arr - time_arr[-1]  # end at t=0

    # Instantaneous frequency from analytic signal
    analytic = h_plus_td + 1j * h_cross_td
    inst_phase = np.unwrap(np.angle(analytic))
    inst_freq = np.gradient(inst_phase, dt) / TWOPI
    inst_freq = np.clip(inst_freq, 0, sample_rate / 2)

    # Detector response
    from gravitational_wave_analyzer.physics.waveform import antenna_pattern
    F_plus, F_cross = antenna_pattern(ra, dec, psi)
    h_detector = F_plus * h_plus_td + F_cross * h_cross_td

    # --- Package results ---
    params = {
        'chirp_mass_solar': Mc,
        'symmetric_mass_ratio': eta,
        'total_mass_solar': M_total_solar,
        'effective_spin': chi_eff_val,
        'f_isco_hz': f_isco,
        'f_start_hz': f_lower,
        'f_end_hz': float(f_max),
        'duration_seconds': float(n_samples * dt),
        'distance_mpc': distance_mpc,
        'peak_strain': float(np.max(np.abs(h_detector))),
        'model': 'TaylorF2',
        'pn_order': '3.5PN',
        'note': 'Inspiral only — no merger or ringdown',
    }

    return {
        'time': time_arr,
        'h_plus': h_plus_td,
        'h_cross': h_cross_td,
        'h_detector': h_detector,
        'frequency': inst_freq,
        'phase': inst_phase,
        'amplitude': amplitude_td,
        'params': params,
    }
