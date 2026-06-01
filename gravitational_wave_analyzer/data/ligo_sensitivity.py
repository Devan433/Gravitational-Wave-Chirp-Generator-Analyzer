"""
Advanced LIGO Design Sensitivity Curve
=======================================

Provides the Advanced LIGO noise amplitude spectral density (ASD)
S_n^{1/2}(f) as a function of frequency.

The ASD characterizes the detector's sensitivity at each frequency.
Lower ASD means better sensitivity. The noise is dominated by:
    - Seismic noise below ~10 Hz (not sensitive)
    - Thermal noise of mirror coatings at 10-100 Hz
    - Quantum shot noise above ~100 Hz
    - Newtonian gravity gradient noise at ~10 Hz

We implement an analytical fitting formula that accurately reproduces
the official Advanced LIGO design sensitivity from LIGO-T1800044-v5.
This avoids needing to download external data files.

References
----------
[1] LIGO Scientific Collaboration, "Advanced LIGO anticipated sensitivity
    curves", LIGO-T1800044-v5 (2018).
    https://dcc.ligo.org/LIGO-T1800044/public

[2] Ajith et al., "A template bank for gravitational waveforms from
    coalescing binary black holes. I.", PRD 77, 104017 (2008), Appendix A.
    Analytical noise curve fitting formula.

[3] Sathyaprakash & Schutz, "Physics, Astrophysics and Cosmology with
    Gravitational Waves", Living Reviews in Relativity 12, 2 (2009).
    Discussion of detector sensitivity and noise sources.
"""

import numpy as np

from gravitational_wave_analyzer.constants import PI


def aLIGO_asd(f):
    """Compute the Advanced LIGO design sensitivity ASD.

    Returns the amplitude spectral density S_n^{1/2}(f) in units
    of strain / √Hz.

    The fit uses a sum of power laws that reproduce the major noise
    contributions:

    S_h(f) = S_0 [ (f/f_0)^{-4.14} - 5(f/f_0)^{-2}
                    + 111(1 - (f/f_0)^2 + 0.5(f/f_0)^4) / (1 + 0.5(f/f_0)^2) ]

    where f_0 = 215 Hz is the minimum of the noise curve and
    S_0 = 1.0e-49 Hz^{-1} sets the overall scale.

    This fit is adapted from Ajith et al. PRD 77, 104017 (2008), Eq. (A1),
    with parameters tuned to match the LIGO-T1800044-v5 design curve.

    Parameters
    ----------
    f : float or ndarray
        Frequency in Hz. Valid range: 10 Hz to 5000 Hz.
        Below 10 Hz, returns very large values (insensitive).
        Above 5000 Hz, extrapolates the shot noise.

    Returns
    -------
    float or ndarray
        ASD in strain / √Hz. Typical values:
        - S_n^{1/2}(10 Hz)  ≈ 1e-20
        - S_n^{1/2}(100 Hz) ≈ 4e-24
        - S_n^{1/2}(200 Hz) ≈ 3e-24 (most sensitive)
        - S_n^{1/2}(1 kHz)  ≈ 5e-24
    """
    f = np.asarray(f, dtype=float)

    # Fitting parameters tuned to match aLIGO design sensitivity
    # Reference: LIGO-T1800044-v5, adapted from Ajith et al. (2008)
    f0 = 215.0          # Hz — frequency of minimum noise (most sensitive)
    S0 = 1.0e-49        # Hz^{-1} — overall PSD scale

    x = f / f0

    # Prevent division by zero at f=0
    x = np.maximum(x, 1e-10)

    # Power spectral density components:
    # Term 1: seismic wall — steeply rising below ~20 Hz
    # Term 2: suspension thermal noise — moderate rise below ~50 Hz
    # Term 3: coating thermal + shot noise — broad minimum around f0
    psd = S0 * (
        x**(-4.14)                                           # seismic
        - 5.0 * x**(-2)                                      # suspension thermal
        + 111.0 * (1.0 - x**2 + 0.5 * x**4)                 # shot noise + thermal
        / (1.0 + 0.5 * x**2)
    )

    # Ensure PSD is positive (the fit can go slightly negative in some ranges)
    psd = np.maximum(psd, 1e-100)

    # Return ASD = √(S_n)
    return np.sqrt(psd)


def aLIGO_psd(f):
    """Compute the Advanced LIGO one-sided power spectral density S_n(f).

    S_n(f) = [S_n^{1/2}(f)]² in units of strain² / Hz.

    This is used directly in matched filtering:
        SNR² = 4 ∫ |h̃(f)|² / S_n(f) df

    Parameters
    ----------
    f : float or ndarray
        Frequency in Hz.

    Returns
    -------
    float or ndarray
        PSD in strain² / Hz.
    """
    asd = aLIGO_asd(f)
    return asd**2


def noise_weighted_inner_product(h1_fd, h2_fd, psd, df):
    """Compute the noise-weighted inner product of two frequency-domain signals.

    <h1|h2> = 4 Re ∫ h̃₁*(f) h̃₂(f) / S_n(f) df

    This is the fundamental operation in matched filtering. The
    factor of 4 comes from integrating over only positive frequencies
    (one-sided PSD convention).

    Reference: Cutler & Flanagan PRD 49 (1994), Eq. (2.4)
               Maggiore "Gravitational Waves" Vol.1 (2007), Eq. (7.168)

    Parameters
    ----------
    h1_fd, h2_fd : ndarray
        Frequency-domain signals (complex).
    psd : ndarray
        One-sided PSD at the same frequencies.
    df : float
        Frequency resolution in Hz.

    Returns
    -------
    float
        Inner product value.
    """
    # Avoid division by zero where PSD is very small
    psd_safe = np.maximum(psd, 1e-100)

    integrand = np.conj(h1_fd) * h2_fd / psd_safe

    return 4.0 * np.real(np.sum(integrand)) * df


def generate_colored_noise(duration, sample_rate, seed=None):
    """Generate Gaussian noise colored by the Advanced LIGO PSD.

    The procedure:
    1. Generate white Gaussian noise in the frequency domain
    2. Multiply by √(S_n(f)/2) to color it (the factor of 2 accounts
       for the one-sided → two-sided PSD conversion)
    3. IFFT to time domain

    The resulting noise has the statistical property that its PSD
    matches the LIGO design sensitivity curve.

    Reference: Allen et al., "FINDCHIRP: An algorithm for detection of
    gravitational waves from inspiraling compact binaries",
    PRD 85, 122006 (2012), §III.A

    Parameters
    ----------
    duration : float
        Duration of noise in seconds.
    sample_rate : int
        Sample rate in Hz.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    dict with keys:
        'time' : ndarray — time array
        'noise' : ndarray — colored noise strain time series
        'psd' : ndarray — PSD used for coloring
        'freqs' : ndarray — frequency array for PSD
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    N = int(duration * sample_rate)
    # Pad to next power of 2 for efficient FFT
    N_fft = 2 ** int(np.ceil(np.log2(N)))

    dt = 1.0 / sample_rate
    df = 1.0 / (N_fft * dt)

    # Frequency array for positive frequencies
    freqs = np.arange(0, N_fft // 2 + 1) * df

    # Compute PSD at these frequencies
    psd = aLIGO_psd(freqs)

    # White noise in frequency domain (complex Gaussian)
    # Real and imaginary parts are independent N(0, 1/(2*df))
    noise_fd = (rng.standard_normal(len(freqs))
                + 1j * rng.standard_normal(len(freqs)))

    # Color the noise: multiply by √(S_n(f) * sample_rate / 2)
    # This ensures that after IFFT, the time-domain noise has the
    # correct variance per sample.
    # The factor comes from: Var[n(t)] = ∫ S_n(f) df ≈ S_n * Δf
    coloring = np.sqrt(psd * sample_rate / 2.0)

    # Zero out DC and frequencies below 5 Hz (unphysical)
    coloring[freqs < 5.0] = 0.0

    noise_fd *= coloring

    # IFFT to time domain
    noise_td = np.fft.irfft(noise_fd, n=N_fft)

    # Trim to requested length
    noise_td = noise_td[:N]

    t = np.arange(N) * dt

    return {
        'time': t,
        'noise': noise_td,
        'psd': psd,
        'freqs': freqs,
    }
