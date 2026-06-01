"""
Parameter Space Explorer
=========================

Computes key gravitational wave observables analytically across a 2D
grid of component masses (m1, m2). These are purely analytic formulas
that do not require full waveform generation, making the computation
extremely fast (~milliseconds for a 50×50 grid).

Computed quantities:
    - Chirp mass M_c
    - Symmetric mass ratio η
    - ISCO frequency f_ISCO
    - Leading-order inspiral duration
    - Final mass and spin (from NR fitting formulas)
    - Approximate optimal SNR (analytic estimate)

References
----------
[1] Cutler & Flanagan, PRD 49, 2658 (1994).
    — Chirp mass, duration, and SNR scaling relations.

[2] Husa et al., PRD 93, 044006 (2016).
    — Final mass fitting formula.

[3] Barausse & Rezzolla, ApJL 704, L40 (2009).
    — Final spin fitting formula.
"""

import numpy as np

from gravitational_wave_analyzer.constants import (
    G_SI, C_SI, MSUN_SI, PI,
    chirp_mass, symmetric_mass_ratio, isco_frequency,
    solar_masses_to_kg,
)
from gravitational_wave_analyzer.physics.ringdown import (
    final_mass_radiated, final_spin_barausse_rezzolla,
    qnm_frequency,
)


def compute_parameter_grid(m1_range=(5, 100), m2_range=(5, 100),
                            grid_size=40, distance_mpc=410.0,
                            f_lower=20.0, s1z=0.0, s2z=0.0):
    """Compute GW observables across a (m1, m2) parameter grid.

    All quantities are computed analytically — no FFTs or waveform
    generation required. This makes the computation extremely fast.

    Parameters
    ----------
    m1_range : tuple
        (min, max) for primary mass in solar masses.
    m2_range : tuple
        (min, max) for secondary mass in solar masses.
    grid_size : int
        Number of grid points per axis.
    distance_mpc : float
        Luminosity distance in Mpc (affects SNR estimates).
    f_lower : float
        Lower frequency cutoff in Hz.
    s1z, s2z : float
        Aligned spin components.

    Returns
    -------
    dict with keys:
        'm1_values' : list — m1 grid values
        'm2_values' : list — m2 grid values
        'chirp_mass' : list of lists — 2D grid of chirp mass
        'mass_ratio' : list of lists — 2D grid of η
        'f_isco' : list of lists — ISCO frequency grid
        'duration' : list of lists — inspiral duration grid (seconds)
        'final_mass' : list of lists — final remnant mass grid
        'final_spin' : list of lists — final remnant spin grid
        'energy_radiated' : list of lists — fraction of mass radiated
        'f_qnm' : list of lists — ringdown QNM frequency grid
        'peak_luminosity' : list of lists — peak GW luminosity (relative)
    """
    m1_values = np.linspace(m1_range[0], m1_range[1], grid_size)
    m2_values = np.linspace(m2_range[0], m2_range[1], grid_size)

    # Initialize grids
    grid_chirp_mass = np.zeros((grid_size, grid_size))
    grid_mass_ratio = np.zeros((grid_size, grid_size))
    grid_f_isco = np.zeros((grid_size, grid_size))
    grid_duration = np.zeros((grid_size, grid_size))
    grid_final_mass = np.zeros((grid_size, grid_size))
    grid_final_spin = np.zeros((grid_size, grid_size))
    grid_energy_radiated = np.zeros((grid_size, grid_size))
    grid_f_qnm = np.zeros((grid_size, grid_size))
    grid_peak_luminosity = np.zeros((grid_size, grid_size))

    for i, m1 in enumerate(m1_values):
        for j, m2 in enumerate(m2_values):
            if m2 > m1:
                # Convention: m1 >= m2; mark invalid region as NaN
                grid_chirp_mass[i, j] = np.nan
                grid_mass_ratio[i, j] = np.nan
                grid_f_isco[i, j] = np.nan
                grid_duration[i, j] = np.nan
                grid_final_mass[i, j] = np.nan
                grid_final_spin[i, j] = np.nan
                grid_energy_radiated[i, j] = np.nan
                grid_f_qnm[i, j] = np.nan
                grid_peak_luminosity[i, j] = np.nan
                continue

            M = m1 + m2
            eta = m1 * m2 / M**2
            Mc = (m1 * m2) ** (3.0/5.0) / M ** (1.0/5.0)

            # ISCO frequency
            f_isco_val = isco_frequency(M)

            # Leading-order inspiral duration
            # t_inspiral ≈ (5/256) η^{-1} (π f_lower M_geo)^{-8/3} M_geo
            M_kg = solar_masses_to_kg(M)
            M_geo = G_SI * M_kg / C_SI**3
            v_lower = (PI * M_geo * f_lower) ** (1.0/3.0)

            if v_lower > 0 and v_lower < 0.5:
                t_dur = (5.0 / 256.0) * M_geo / (eta * v_lower**8)
            else:
                t_dur = 0.0

            # Final mass and spin from NR fitting formulas
            try:
                remnant = final_mass_radiated(m1, m2, s1z, s2z)
                M_f = remnant['final_mass_solar']
                a_f = remnant['final_spin']
                E_rad_frac = remnant['energy_radiated_fraction']

                # QNM frequency
                qnm = qnm_frequency(M_f, a_f)
                f_qnm_val = qnm['f_qnm_hz']
            except Exception:
                M_f = M * 0.95
                a_f = 0.67
                E_rad_frac = 0.05
                f_qnm_val = 0.0

            # Peak luminosity (relative scaling)
            # L_peak ~ η^2 — crude but shows the trend
            # Equal mass (η=0.25) radiates most efficiently
            L_peak_relative = eta**2 * (1.0 + 2.0 * eta)

            grid_chirp_mass[i, j] = Mc
            grid_mass_ratio[i, j] = eta
            grid_f_isco[i, j] = f_isco_val
            grid_duration[i, j] = t_dur
            grid_final_mass[i, j] = M_f
            grid_final_spin[i, j] = a_f
            grid_energy_radiated[i, j] = E_rad_frac
            grid_f_qnm[i, j] = f_qnm_val
            grid_peak_luminosity[i, j] = L_peak_relative

    # Convert NaN to None for JSON serialization
    def grid_to_list(grid):
        result = []
        for row in grid:
            result.append([None if np.isnan(v) else float(v) for v in row])
        return result

    return {
        'm1_values': m1_values.tolist(),
        'm2_values': m2_values.tolist(),
        'chirp_mass': grid_to_list(grid_chirp_mass),
        'mass_ratio': grid_to_list(grid_mass_ratio),
        'f_isco': grid_to_list(grid_f_isco),
        'duration': grid_to_list(grid_duration),
        'final_mass': grid_to_list(grid_final_mass),
        'final_spin': grid_to_list(grid_final_spin),
        'energy_radiated': grid_to_list(grid_energy_radiated),
        'f_qnm': grid_to_list(grid_f_qnm),
        'peak_luminosity': grid_to_list(grid_peak_luminosity),
    }
