"""
Parameter Estimation via Grid-Search Matched Filtering
========================================================

Implements the **inverse problem** of gravitational wave astronomy:
given an observed waveform, estimate the source parameters (masses, spins).

Real LIGO parameter estimation uses Bayesian inference with stochastic
samplers (LALInference, Bilby). This module implements a simpler but
conceptually equivalent approach: exhaustive grid search over the
parameter space, computing the match (normalized overlap) for each
template.

The **match** between two waveforms is defined as:

    M(h₁, h₂) = max_{t_c, φ_c} ⟨h₁|h₂⟩ / √(⟨h₁|h₁⟩ ⟨h₂|h₂⟩)

This is the noise-weighted inner product, maximized over time and phase
shifts, and normalized to [0, 1]. A match of 1.0 means perfect agreement.

The grid search proceeds in two stages:
    1. Coarse grid (large step) — rapidly narrows the parameter space
    2. Fine grid (small step) — refines around the best coarse match

References
----------
[1] Owen & Sathyaprakash, "Matched filtering of gravitational waves
    from inspiraling compact binaries: Computational cost and template
    placement", PRD 60, 022002 (1999).
    — Template bank placement and match computation.

[2] Allen et al., "FINDCHIRP: An algorithm for detection of gravitational
    waves from inspiraling compact binaries", PRD 85, 122006 (2012).
    — Matched filtering pipeline used at LIGO.

[3] Veitch et al., "Parameter estimation on gravitational waves from
    neutron-star binaries with spinning components",
    PRD 91, 042003 (2015).
    — Bayesian PE methods (the "full" approach we're simplifying here).
"""

import numpy as np
from scipy.fft import fft, ifft, fftfreq

from gravitational_wave_analyzer.constants import PI, TWOPI
from gravitational_wave_analyzer.data.ligo_sensitivity import aLIGO_psd


def compute_match(h1, h2, sample_rate, f_lower=20.0):
    """Compute the match (fitting factor) between two waveforms.

    The match is the noise-weighted overlap maximized over time and
    phase shifts, normalized to [0, 1]:

        M = max_{t_c, φ_c} |⟨h₁|h₂⟩| / √(⟨h₁|h₁⟩ ⟨h₂|h₂⟩)

    Maximization over time shift is done efficiently via IFFT.
    Maximization over phase is done by taking the absolute value
    of the complex overlap.

    Parameters
    ----------
    h1, h2 : ndarray
        Time-domain waveforms (real arrays).
    sample_rate : float
        Sample rate in Hz.
    f_lower : float
        Lower frequency cutoff.

    Returns
    -------
    float
        Match value in [0, 1]. Values > 0.97 are considered
        "effectual" for detection purposes.
    """
    # Ensure same length
    N = max(len(h1), len(h2))
    # Pad to next power of 2
    N = 2 ** int(np.ceil(np.log2(N)))

    h1_padded = np.zeros(N)
    h2_padded = np.zeros(N)
    h1_padded[:len(h1)] = h1
    h2_padded[:len(h2)] = h2

    dt = 1.0 / sample_rate
    df = 1.0 / (N * dt)

    # FFT
    h1_fd = fft(h1_padded) * dt
    h2_fd = fft(h2_padded) * dt

    # Frequency array and PSD
    freqs = fftfreq(N, dt)
    psd = aLIGO_psd(np.abs(freqs))
    psd_safe = np.maximum(psd, 1e-100)

    # Frequency mask
    freq_mask = (np.abs(freqs) >= f_lower) & (np.abs(freqs) <= sample_rate / 2)

    # Compute ⟨h₁|h₁⟩ (template norm)
    sigma1_sq = 4.0 * np.sum(np.abs(h1_fd[freq_mask])**2 / psd_safe[freq_mask]) * df
    sigma2_sq = 4.0 * np.sum(np.abs(h2_fd[freq_mask])**2 / psd_safe[freq_mask]) * df

    if sigma1_sq < 1e-100 or sigma2_sq < 1e-100:
        return 0.0

    sigma1 = np.sqrt(sigma1_sq)
    sigma2 = np.sqrt(sigma2_sq)

    # Compute the cross-correlation in frequency domain
    # z̃(f) = h̃₁*(f) h̃₂(f) / S_n(f)
    integrand = np.zeros(N, dtype=complex)
    integrand[freq_mask] = (np.conj(h1_fd[freq_mask])
                            * h2_fd[freq_mask]
                            / psd_safe[freq_mask])

    # IFFT gives the time-domain cross-correlation
    z_td = ifft(integrand) / dt * 4.0

    # Maximize over time shift by taking max of |z(t)|
    # Maximize over phase shift by taking absolute value (complex z)
    overlap_max = np.max(np.abs(z_td))

    # Normalize
    match = overlap_max / (sigma1 * sigma2)

    # Clamp to [0, 1] (numerical errors can push slightly above 1)
    return float(np.clip(match, 0.0, 1.0))


def grid_search_pe(observed_h, sample_rate, f_lower=20.0,
                    m1_range=(10, 80), m2_range=(10, 80),
                    distance_mpc=410.0, coarse_step=3.0,
                    fine_step=0.5, fine_radius=6.0):
    """Run a two-stage grid search to estimate binary parameters.

    Stage 1 (Coarse): Evaluate match on a coarse grid of (m1, m2).
    Stage 2 (Fine): Refine around the best coarse match.

    This is a simplified version of what LIGO's template bank does,
    restricted to the mass parameters (spins fixed to zero for speed).

    Parameters
    ----------
    observed_h : ndarray
        The "observed" time-domain waveform to match against.
    sample_rate : float
        Sample rate in Hz.
    f_lower : float
        Lower frequency cutoff.
    m1_range : tuple
        (min, max) range for primary mass in solar masses.
    m2_range : tuple
        (min, max) range for secondary mass in solar masses.
    distance_mpc : float
        Distance for template generation.
    coarse_step : float
        Mass step for coarse grid (solar masses).
    fine_step : float
        Mass step for fine grid (solar masses).
    fine_radius : float
        Radius around best coarse match for fine grid.

    Returns
    -------
    dict with keys:
        'best_m1' : float — best-fit primary mass
        'best_m2' : float — best-fit secondary mass
        'best_match' : float — match value at best fit
        'coarse_grid' : dict — m1_values, m2_values, match_matrix
        'fine_grid' : dict — m1_values, m2_values, match_matrix
        'all_candidates' : list — top matches sorted by match value
        'n_templates' : int — total templates evaluated
    """
    from gravitational_wave_analyzer.physics.full_waveform import (
        generate_full_waveform,
    )

    n_templates = 0
    all_results = []

    # --- Stage 1: Coarse Grid ---
    m1_coarse = np.arange(m1_range[0], m1_range[1] + 0.1, coarse_step)
    m2_coarse = np.arange(m2_range[0], m2_range[1] + 0.1, coarse_step)

    coarse_match = np.zeros((len(m1_coarse), len(m2_coarse)))

    for i, m1 in enumerate(m1_coarse):
        for j, m2 in enumerate(m2_coarse):
            if m2 > m1:
                coarse_match[i, j] = 0.0
                continue

            try:
                template_result = generate_full_waveform(
                    m1_solar=float(m1), m2_solar=float(m2),
                    distance_mpc=distance_mpc,
                    f_lower=f_lower, sample_rate=sample_rate,
                    method='imrphenomd',
                )
                template_h = template_result['h_detector']

                match_val = compute_match(
                    observed_h, template_h, sample_rate, f_lower
                )
                coarse_match[i, j] = match_val
                all_results.append({
                    'm1': float(m1), 'm2': float(m2),
                    'match': match_val, 'stage': 'coarse'
                })
                n_templates += 1
            except Exception:
                coarse_match[i, j] = 0.0

    # Find best coarse match
    best_idx = np.unravel_index(np.argmax(coarse_match), coarse_match.shape)
    best_m1_coarse = float(m1_coarse[best_idx[0]])
    best_m2_coarse = float(m2_coarse[best_idx[1]])

    # --- Stage 2: Fine Grid around best coarse match ---
    m1_fine_min = max(m1_range[0], best_m1_coarse - fine_radius)
    m1_fine_max = min(m1_range[1], best_m1_coarse + fine_radius)
    m2_fine_min = max(m2_range[0], best_m2_coarse - fine_radius)
    m2_fine_max = min(m2_range[1], best_m2_coarse + fine_radius)

    m1_fine = np.arange(m1_fine_min, m1_fine_max + 0.1, fine_step)
    m2_fine = np.arange(m2_fine_min, m2_fine_max + 0.1, fine_step)

    fine_match = np.zeros((len(m1_fine), len(m2_fine)))

    for i, m1 in enumerate(m1_fine):
        for j, m2 in enumerate(m2_fine):
            if m2 > m1:
                fine_match[i, j] = 0.0
                continue

            try:
                template_result = generate_full_waveform(
                    m1_solar=float(m1), m2_solar=float(m2),
                    distance_mpc=distance_mpc,
                    f_lower=f_lower, sample_rate=sample_rate,
                    method='imrphenomd',
                )
                template_h = template_result['h_detector']

                match_val = compute_match(
                    observed_h, template_h, sample_rate, f_lower
                )
                fine_match[i, j] = match_val
                all_results.append({
                    'm1': float(m1), 'm2': float(m2),
                    'match': match_val, 'stage': 'fine'
                })
                n_templates += 1
            except Exception:
                fine_match[i, j] = 0.0

    # Find overall best
    fine_best_idx = np.unravel_index(np.argmax(fine_match), fine_match.shape)
    best_m1 = float(m1_fine[fine_best_idx[0]])
    best_m2 = float(m2_fine[fine_best_idx[1]])
    best_match = float(fine_match[fine_best_idx[0], fine_best_idx[1]])

    # Sort all results by match
    all_results.sort(key=lambda x: x['match'], reverse=True)

    return {
        'best_m1': best_m1,
        'best_m2': best_m2,
        'best_match': best_match,
        'coarse_grid': {
            'm1_values': m1_coarse.tolist(),
            'm2_values': m2_coarse.tolist(),
            'match_matrix': coarse_match.tolist(),
        },
        'fine_grid': {
            'm1_values': m1_fine.tolist(),
            'm2_values': m2_fine.tolist(),
            'match_matrix': fine_match.tolist(),
        },
        'all_candidates': all_results[:10],  # top 10
        'n_templates': n_templates,
    }
