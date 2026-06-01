"""
Q-Transform Spectrogram for Gravitational Wave Visualization
==============================================================

Implements the Constant-Q Transform (CQT) used by LIGO to produce the
iconic time-frequency spectrograms seen in their detection papers.

Why Q-transform instead of standard STFT?
------------------------------------------
A standard spectrogram uses a fixed window size, giving uniform time and
frequency resolution. But GW chirps sweep from ~20 Hz to ~300 Hz:
- At 20 Hz, we need long windows (good frequency resolution) to resolve
  the slowly evolving inspiral
- At 300 Hz, we need short windows (good time resolution) to track the
  rapid merger

The Q-transform solves this with logarithmic frequency tiling:
    Q = f / Δf = constant

This means each frequency bin has the same number of cycles under the
window, naturally providing better time resolution at high frequencies
and better frequency resolution at low frequencies.

LIGO's Omega pipeline and GWpy both use this approach.

References
----------
[1] Brown, "Calculation of a constant Q spectral transform",
    J. Acoust. Soc. Am. 89, 425 (1991).
    Original constant-Q algorithm.

[2] Chatterji et al., "Multiresolution techniques for the detection of
    gravitational-wave bursts", CQG 21, S1809 (2004).
    Application of Q-transform to GW data (the Omega pipeline).

[3] Chatterji, "The search for gravitational wave bursts in data from
    the second LIGO science run", PhD thesis, MIT (2005).
    Detailed description of the Q-transform tiling and energy normalization.
"""

import numpy as np
from scipy.signal import spectrogram as scipy_spectrogram
from scipy.fft import fft, ifft, fftfreq

from gravitational_wave_analyzer.constants import PI, TWOPI


def q_transform(time_series, sample_rate, q_range=(4, 64),
                f_range=(20, 2048), n_freq_bins=200,
                tres=0.002, whiten=True):
    """Compute the Q-transform spectrogram of a time series.

    The Q-transform tiles the time-frequency plane with basis functions
    (called "tiles") that have constant Q = f/Δf. For each tile:

    1. Window the data in the frequency domain with a bisquare
       window centered at the tile frequency
    2. IFFT to get the time series for that tile
    3. Compute the energy |z(t)|²

    The bisquare (Tukey) window in the frequency domain is:

        w(f') = [1 - (f'/σ_f)²]²   for |f'| < σ_f
        w(f') = 0                    otherwise

    where f' = f - f_tile and σ_f = f_tile / (2Q) is the bandwidth.

    Parameters
    ----------
    time_series : ndarray
        Input strain data h(t).
    sample_rate : float
        Sample rate in Hz.
    q_range : tuple (Q_min, Q_max)
        Range of Q values to search. The algorithm selects the Q
        that maximizes the signal energy in each tile.
    f_range : tuple (f_min, f_max)
        Frequency range in Hz for the spectrogram.
    n_freq_bins : int
        Number of logarithmically spaced frequency bins.
    tres : float
        Desired time resolution in seconds.
    whiten : bool
        If True, normalize each frequency row by its RMS to flatten
        the noise floor (whitening).

    Returns
    -------
    dict with keys:
        'times' : ndarray — time array (seconds)
        'frequencies' : ndarray — frequency array (Hz), log-spaced
        'energy' : 2D ndarray — normalized energy, shape (n_freq, n_time)
        'q_used' : float — Q value used for the transform
    """
    N = len(time_series)
    dt = 1.0 / sample_rate
    T = N * dt

    # --- Select Q value ---
    # Use the geometric mean of the Q range as a starting point.
    # A more sophisticated implementation would optimize Q per tile.
    Q = np.sqrt(q_range[0] * q_range[1])

    # --- Frequency bins (logarithmic spacing) ---
    f_min, f_max = f_range
    f_min = max(f_min, sample_rate / N)  # Can't resolve below df = 1/T
    f_max = min(f_max, sample_rate / 2)  # Nyquist
    frequencies = np.geomspace(f_min, f_max, n_freq_bins)

    # --- Time bins ---
    n_time = int(T / tres)
    times = np.linspace(0, T, n_time, endpoint=False)

    # --- FFT of input data ---
    data_fft = fft(time_series)
    freqs_fft = fftfreq(N, dt)

    # --- Compute Q-transform energy ---
    energy = np.zeros((n_freq_bins, n_time))

    for i, f0 in enumerate(frequencies):
        # Bandwidth for this tile: Δf = f0 / Q
        sigma_f = f0 / (2.0 * Q)

        # Number of frequency bins within the window
        # The bisquare window has compact support: |f - f0| < sigma_f
        df = 1.0 / T  # frequency resolution of the FFT

        # Indices of FFT bins within the window
        f_low = f0 - sigma_f
        f_high = f0 + sigma_f
        idx_low = max(int(f_low / df), 1)
        idx_high = min(int(f_high / df) + 1, N // 2)

        if idx_high <= idx_low:
            continue

        # Extract the relevant portion of the FFT
        f_slice = freqs_fft[idx_low:idx_high]
        data_slice = data_fft[idx_low:idx_high]

        # --- Bisquare window ---
        # w(f) = [1 - ((f - f0) / σ_f)²]²    for |f - f0| < σ_f
        # This is a smooth, well-localized window in frequency space.
        # Reference: Chatterji (2005), §2.5.1
        u = (f_slice - f0) / sigma_f
        window = np.zeros_like(u)
        mask_w = np.abs(u) < 1.0
        window[mask_w] = (1.0 - u[mask_w]**2)**2

        # Normalize the window energy
        norm = np.sqrt(np.sum(window**2))
        if norm > 0:
            window /= norm

        # Apply window to data
        windowed = data_slice * window

        # --- Time-domain signal for this tile ---
        # Pad windowed data into a full-length array and IFFT
        tile_fft = np.zeros(N, dtype=complex)
        tile_fft[idx_low:idx_high] = windowed
        tile_td = ifft(tile_fft) * N  # Scale to preserve amplitude

        # --- Energy: |z(t)|² ---
        tile_energy = np.abs(tile_td)**2

        # --- Downsample to output time resolution ---
        # Interpolate tile_energy onto the output time grid
        t_full = np.arange(N) * dt
        energy[i, :] = np.interp(times, t_full, tile_energy)

    # --- Whitening (normalize each frequency row) ---
    if whiten:
        for i in range(n_freq_bins):
            row_rms = np.sqrt(np.mean(energy[i, :]**2))
            if row_rms > 0:
                energy[i, :] /= row_rms

    # --- Normalize overall energy to [0, 1] for plotting ---
    max_energy = np.max(energy)
    if max_energy > 0:
        energy /= max_energy

    return {
        'times': times,
        'frequencies': frequencies,
        'energy': energy,
        'q_used': Q,
    }


def standard_spectrogram(time_series, sample_rate, nperseg=1024,
                          noverlap=None, f_range=(20, 2048)):
    """Compute a standard STFT spectrogram using scipy.

    Uses settings similar to LIGO's visualization pipeline.

    Parameters
    ----------
    time_series : ndarray
        Input strain data.
    sample_rate : float
        Sample rate in Hz.
    nperseg : int
        FFT segment length. 1024 at 4096 Hz gives ~4 Hz resolution.
    noverlap : int
        Overlap between segments. Default 7/8 of nperseg.
    f_range : tuple
        Frequency range to return.

    Returns
    -------
    dict with keys:
        'times', 'frequencies', 'power' — spectrogram data
    """
    if noverlap is None:
        noverlap = int(nperseg * 7 / 8)

    f, t, Sxx = scipy_spectrogram(
        time_series,
        fs=sample_rate,
        nperseg=nperseg,
        noverlap=noverlap,
        window='tukey',
        scaling='density',
        mode='psd',
    )

    # Clip to requested frequency range
    f_mask = (f >= f_range[0]) & (f <= f_range[1])

    return {
        'times': t,
        'frequencies': f[f_mask],
        'power': Sxx[f_mask, :],
    }


def theoretical_frequency_track(m1_solar, m2_solar, t_merger, times,
                                  f_lower=20.0):
    """Compute the theoretical GW frequency evolution for overlay on spectrogram.

    Uses the leading-order (Newtonian) chirp formula:

    f(t) = (1/π) × (5/(256(t_c - t)))^{3/8} × (G M_c / c³)^{-5/8}

    where t_c is the coalescence time and M_c is the chirp mass.

    This formula is exact at leading (Newtonian) order and provides
    a good visual guide for the inspiral frequency evolution.

    Reference: Maggiore "Gravitational Waves" Vol.1 (2007), Eq. (4.21)

    Parameters
    ----------
    m1_solar, m2_solar : float
        Component masses in solar masses.
    t_merger : float
        Time of merger in seconds (same time coordinate as `times`).
    times : ndarray
        Time array for which to compute f(t).
    f_lower : float
        Starting frequency (Hz). Track capped below this.

    Returns
    -------
    ndarray
        Theoretical GW frequency at each time, in Hz.
        Returns NaN for times after merger.
    """
    from gravitational_wave_analyzer.constants import (
        G_SI, C_SI, MSUN_SI, chirp_mass as compute_chirp_mass,
    )

    Mc_solar = compute_chirp_mass(m1_solar, m2_solar)
    Mc_kg = Mc_solar * MSUN_SI

    # Chirp mass in geometric time units: τ_c = G Mc / c³
    Mc_geo = G_SI * Mc_kg / C_SI**3  # seconds

    tau = t_merger - times  # time to coalescence

    f_track = np.full_like(times, np.nan, dtype=float)

    # Only compute for times before merger (τ > 0)
    valid = tau > 0

    # Leading-order frequency evolution:
    # f(τ) = (1/π) (5/(256τ))^{3/8} Mc_geo^{-5/8}
    # Reference: Maggiore Eq. (4.21)
    f_track[valid] = (
        (1.0 / PI)
        * (5.0 / (256.0 * tau[valid])) ** (3.0 / 8.0)
        * Mc_geo ** (-5.0 / 8.0)
    )

    # Cap below f_lower
    f_track[f_track < f_lower] = np.nan

    return f_track
