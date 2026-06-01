"""
Matched Filtering — Gravitational Wave Detection Pipeline
==========================================================

Implements the matched filtering algorithm exactly as LIGO uses it
to detect gravitational wave signals buried in detector noise.

Matched filtering is the OPTIMAL linear filter for detecting a known
signal in stationary Gaussian noise. It maximizes the signal-to-noise
ratio (SNR), and the maximum achievable SNR is a fundamental theorem:

    ρ_opt² = 4 ∫₀^∞ |h̃(f)|² / S_n(f) df

This is the Neyman-Pearson optimal detection statistic.

The detection pipeline:
1. Generate template waveform h(t)
2. Generate or load detector noise n(t) with PSD S_n(f)
3. Inject signal: d(t) = n(t) + α·h(t) where α sets the SNR
4. Compute SNR(t) = IFFT[ d̃(f) h̃*(f) / S_n(f) ] / σ_h
5. Peak of |SNR(t)| gives the detection time and measured SNR

References
----------
[1] Allen et al., "FINDCHIRP: An algorithm for detection of gravitational
    waves from inspiraling compact binaries", PRD 85, 122006 (2012).
    The definitive reference for LIGO's matched filtering pipeline.
    https://doi.org/10.1103/PhysRevD.85.122006

[2] Wainstein & Zubakov, "Extraction of Signals from Noise" (1962).
    Original derivation of the matched filter theorem.

[3] Sathyaprakash & Dhurandhar, "Choice of filters for the detection of
    gravitational waves from coalescing binaries", PRD 44, 3819 (1991).

[4] Cutler & Flanagan, PRD 49, 2658 (1994).
    Noise-weighted inner product and optimal SNR.
"""

import numpy as np
from scipy.fft import fft, ifft, fftfreq

from gravitational_wave_analyzer.constants import PI, TWOPI
from gravitational_wave_analyzer.data.ligo_sensitivity import (
    aLIGO_psd, generate_colored_noise,
)


def compute_optimal_snr(h_td, sample_rate, f_lower=20.0):
    """Compute the optimal (matched filter) SNR for a given waveform.

    The optimal SNR is the maximum possible SNR achievable when the
    template exactly matches the signal and the noise is Gaussian:

        ρ²_opt = 4 ∫_{f_lower}^{f_Nyquist} |h̃(f)|² / S_n(f) df

    This represents the "loudness" of the signal relative to the
    detector noise. For GW150914 at 410 Mpc, ρ_opt ≈ 24.

    Reference: Cutler & Flanagan PRD 49 (1994), Eq. (2.5)

    Parameters
    ----------
    h_td : ndarray
        Time-domain template waveform.
    sample_rate : float
        Sample rate in Hz.
    f_lower : float
        Lower frequency cutoff in Hz.

    Returns
    -------
    float
        Optimal SNR (dimensionless).
    """
    N = len(h_td)
    dt = 1.0 / sample_rate
    df = 1.0 / (N * dt)

    # FFT of the template
    h_fd = fft(h_td) * dt  # Normalize: h̃(f) = Σ h(t) e^{2πift} dt

    # Frequency array
    freqs = fftfreq(N, dt)

    # Only use positive frequencies
    pos_mask = (freqs >= f_lower) & (freqs <= sample_rate / 2)

    # PSD at these frequencies
    psd = aLIGO_psd(np.abs(freqs[pos_mask]))

    # Optimal SNR²: 4 ∫ |h̃(f)|² / S_n(f) df
    # Using discrete sum: 4 Σ |h̃(f_k)|² / S_n(f_k) × df
    integrand = np.abs(h_fd[pos_mask])**2 / psd
    snr_sq = 4.0 * np.sum(integrand) * df

    return np.sqrt(max(snr_sq, 0.0))


def matched_filter(data, template, sample_rate, f_lower=20.0,
                    psd_data=None):
    """Perform matched filtering: compute the SNR time series.

    The matched filter output is:

        z(t) = 4 IFFT[ d̃(f) h̃*(f) / S_n(f) ]

    The complex SNR is:

        ρ(t) = z(t) / σ_h

    where σ_h² = 4 ∫ |h̃(f)|² / S_n(f) df is the template normalization.

    The detection statistic is |ρ(t)|.

    Reference: Allen et al. PRD 85, 122006 (2012), Eqs. (3)-(7)

    Parameters
    ----------
    data : ndarray
        Time-domain data d(t) = n(t) + h(t) (noise + possible signal).
    template : ndarray
        Time-domain template waveform h(t) (from our model).
    sample_rate : float
        Sample rate in Hz.
    f_lower : float
        Lower frequency cutoff in Hz.
    psd_data : ndarray, optional
        Pre-computed PSD. If None, uses aLIGO design curve.

    Returns
    -------
    dict with keys:
        'snr_timeseries' : ndarray — |ρ(t)| SNR time series
        'snr_complex' : ndarray — complex ρ(t)
        'peak_snr' : float — maximum |ρ|
        'peak_time_index' : int — index of maximum
        'peak_time' : float — time of maximum in seconds
        'sigma' : float — template normalization factor
    """
    N = len(data)
    dt = 1.0 / sample_rate
    df = 1.0 / (N * dt)

    # --- Ensure template and data have same length ---
    template_padded = np.zeros(N)
    n_template = min(len(template), N)
    template_padded[:n_template] = template[:n_template]

    # --- FFT of data and template ---
    data_fd = fft(data) * dt
    template_fd = fft(template_padded) * dt

    # --- Frequency array ---
    freqs = fftfreq(N, dt)

    # --- PSD ---
    if psd_data is not None:
        psd = psd_data
    else:
        psd = aLIGO_psd(np.abs(freqs))

    # --- Apply frequency cutoffs ---
    # Zero out frequencies below f_lower and above Nyquist
    # This is equivalent to high-pass filtering the data
    freq_mask = (np.abs(freqs) >= f_lower) & (np.abs(freqs) <= sample_rate / 2)

    # Prevent division by zero in PSD
    psd_safe = np.maximum(psd, 1e-100)

    # --- Matched filter integral ---
    # z̃(f) = d̃(f) h̃*(f) / S_n(f)
    # z(t) = IFFT[ z̃(f) ] × 4 × df × N   (normalization for discrete IFFT)
    integrand = np.zeros(N, dtype=complex)
    integrand[freq_mask] = (data_fd[freq_mask]
                            * np.conj(template_fd[freq_mask])
                            / psd_safe[freq_mask])

    # IFFT gives the time-domain correlation
    z_td = ifft(integrand) / dt  # undo the dt normalization

    # Scale by 4 (one-sided PSD convention)
    z_td *= 4.0

    # --- Template normalization (σ_h) ---
    # σ² = 4 ∫ |h̃(f)|² / S_n(f) df
    sigma_sq = 4.0 * np.sum(
        np.abs(template_fd[freq_mask])**2 / psd_safe[freq_mask]
    ) * df
    sigma = np.sqrt(max(sigma_sq, 1e-100))

    # --- SNR time series ---
    snr_complex = z_td / sigma
    snr_abs = np.abs(snr_complex)

    # --- Find peak ---
    peak_idx = np.argmax(snr_abs)
    peak_snr = float(snr_abs[peak_idx])
    peak_time = float(peak_idx * dt)

    return {
        'snr_timeseries': snr_abs,
        'snr_complex': snr_complex,
        'peak_snr': peak_snr,
        'peak_time_index': peak_idx,
        'peak_time': peak_time,
        'sigma': sigma,
    }


def inject_signal(noise, signal, target_snr, sample_rate, f_lower=20.0):
    """Inject a gravitational wave signal into noise at a specified SNR.

    The injection amplitude is scaled so that the optimal matched filter
    SNR equals target_snr:

        d(t) = n(t) + α × h(t)

    where α = target_snr / ρ_opt(h), and ρ_opt(h) is the optimal SNR
    of the unnormalized template.

    Reference: Allen et al. PRD 85 (2012), §IV.A

    Parameters
    ----------
    noise : ndarray
        Time-domain noise n(t).
    signal : ndarray
        Time-domain signal h(t) (unnormalized).
    target_snr : float
        Desired matched filter SNR.
    sample_rate : float
        Sample rate in Hz.
    f_lower : float
        Lower frequency cutoff in Hz.

    Returns
    -------
    dict with keys:
        'data' : ndarray — d(t) = n(t) + α h(t)
        'signal_scaled' : ndarray — α h(t)
        'scale_factor' : float — α
        'optimal_snr_raw' : float — ρ_opt of unscaled signal
    """
    # Compute optimal SNR of the raw (unscaled) signal
    # Need to match lengths
    N = len(noise)
    signal_padded = np.zeros(N)
    n_sig = min(len(signal), N)

    # Place signal in the middle of the noise
    # This ensures the signal is away from the edges
    offset = N // 2 - n_sig // 2
    offset = max(0, offset)
    signal_padded[offset:offset + n_sig] = signal[:n_sig]

    rho_opt_raw = compute_optimal_snr(signal_padded, sample_rate, f_lower)

    if rho_opt_raw < 1e-30:
        raise ValueError("Signal optimal SNR is effectively zero. "
                         "Check waveform amplitude and distance.")

    # Scale factor to achieve target SNR
    alpha = target_snr / rho_opt_raw

    # Inject
    signal_scaled = alpha * signal_padded
    data = noise + signal_scaled

    return {
        'data': data,
        'signal_scaled': signal_scaled,
        'scale_factor': alpha,
        'optimal_snr_raw': rho_opt_raw,
        'injection_offset': offset,
    }


def run_detection_pipeline(waveform_result, target_snr=25.0,
                            noise_duration=16.0, sample_rate=4096,
                            f_lower=20.0, seed=42):
    """Run the complete matched filtering detection pipeline.

    This orchestrates the full LIGO-style detection:
    1. Generate colored Gaussian noise
    2. Inject the GW signal at the target SNR
    3. Run matched filter
    4. Report detection statistics

    Parameters
    ----------
    waveform_result : dict
        Output from generate_full_waveform().
    target_snr : float
        Target injection SNR.
    noise_duration : float
        Duration of noise segment in seconds.
    sample_rate : int
        Sample rate in Hz.
    f_lower : float
        Lower frequency cutoff.
    seed : int
        Random seed for noise generation.

    Returns
    -------
    dict with all detection pipeline results.
    """
    # --- Generate colored noise ---
    noise_result = generate_colored_noise(
        noise_duration, sample_rate, seed=seed
    )
    noise = noise_result['noise']

    # --- Extract signal ---
    signal = waveform_result['h_detector']

    # --- Inject signal ---
    injection = inject_signal(
        noise, signal, target_snr, sample_rate, f_lower
    )

    # --- Run matched filter ---
    # Use the signal as its own template (perfect match)
    # In practice, LIGO uses a bank of templates spanning the parameter space
    template = injection['signal_scaled']  # use the injected signal as template

    mf_result = matched_filter(
        injection['data'], template, sample_rate, f_lower
    )

    # --- Compute optimal SNR of the injected signal ---
    optimal_snr = compute_optimal_snr(
        injection['signal_scaled'], sample_rate, f_lower
    )

    # --- Time array ---
    t_noise = noise_result['time']

    return {
        'time': t_noise,
        'data': injection['data'],
        'noise': noise,
        'signal_injected': injection['signal_scaled'],
        'snr_timeseries': mf_result['snr_timeseries'],
        'snr_complex': mf_result['snr_complex'],
        'peak_snr': mf_result['peak_snr'],
        'peak_time': mf_result['peak_time'],
        'optimal_snr': optimal_snr,
        'target_snr': target_snr,
        'scale_factor': injection['scale_factor'],
        'injection_offset': injection['injection_offset'],
        'psd': noise_result['psd'],
        'psd_freqs': noise_result['freqs'],
    }
