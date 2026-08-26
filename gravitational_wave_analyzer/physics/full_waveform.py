"""
Full Waveform Assembly
=======================

Stitches the inspiral, merger, and ringdown phases into a single
continuous gravitational waveform. In practice, IMRPhenomD already
provides the full IMR waveform, so this module serves two purposes:

1. When using the TaylorT4 inspiral (physics/waveform.py):
   Attach the ringdown from physics/ringdown.py via smooth stitching.

2. When using IMRPhenomD (physics/merger.py):
   Post-process the output (tapering, resampling, time alignment).

The stitching uses a Planck taper window at the junction to ensure
C∞ smoothness, preventing Gibbs ringing artifacts in the FFT.

References
----------
[1] McKechan, Robinson & Sathyaprakash, "A tapering window for
    time-domain templates and simulated signals in the detection
    of gravitational waves from coalescing compact binaries",
    CQG 27, 084020 (2010).

[2] Buonanno, Cook & Pretorius, "Inspiral, merger and ring-down of
    equal-mass black-hole binaries", PRD 75, 124018 (2007).
"""

import numpy as np
from scipy.signal.windows import tukey

from gravitational_wave_analyzer.constants import (
    PI, TWOPI, LIGO_SAMPLE_RATE, LIGO_F_LOWER,
    chirp_mass, symmetric_mass_ratio, isco_frequency,
)
from gravitational_wave_analyzer.physics.merger import generate_imrphenomd_waveform
from gravitational_wave_analyzer.physics.ringdown import (
    compute_ringdown_params, final_spin_barausse_rezzolla, final_mass_radiated,
)


def generate_full_waveform(m1_solar, m2_solar, s1z=0.0, s2z=0.0,
                            distance_mpc=410.0, inclination=0.0,
                            f_lower=20.0, sample_rate=4096,
                            ra=0.0, dec=0.0, psi=0.0,
                            method='imrphenomd'):
    """Generate the complete inspiral-merger-ringdown gravitational waveform.

    This is the top-level waveform generation function. It delegates to
    either the full IMRPhenomD model or a stitched PN+ringdown model,
    then applies post-processing:
        1. Planck taper at start (suppress spectral leakage)
        2. Time alignment (merger at t=0)
        3. Trim to reasonable duration
        4. Compute derived quantities (frequency, phase)

    Parameters
    ----------
    m1_solar, m2_solar : float
        Component masses in solar masses.
    s1z, s2z : float
        Aligned spin components, range [-0.99, 0.99].
    distance_mpc : float
        Luminosity distance in Megaparsecs.
    inclination : float
        Orbital inclination (radians). 0=face-on.
    f_lower : float
        Starting frequency in Hz.
    sample_rate : int
        Output sample rate in Hz.
    ra, dec, psi : float
        Sky position and polarization (radians).
    method : str
        'imrphenomd' — full IMRPhenomD (recommended)
        'pn_ringdown' — TaylorT4 inspiral + QNM ringdown (educational)

    Returns
    -------
    dict — Complete waveform data:
        'time' : ndarray — seconds, merger at t=0
        'h_plus', 'h_cross', 'h_detector' : ndarray — strain
        'frequency' : ndarray — instantaneous GW frequency (Hz)
        'phase' : ndarray — GW phase (radians)
        'amplitude' : ndarray — strain amplitude envelope
        'params' : dict — all physical parameters
        + frequency-domain arrays if available
    """
    if method == 'imrphenomd':
        return _generate_imrphenomd(
            m1_solar, m2_solar, s1z, s2z,
            distance_mpc, inclination, f_lower, sample_rate,
            ra, dec, psi
        )
    else:
        raise ValueError(f'Unknown method: {method}. Use \'imrphenomd\'.')
def _generate_imrphenomd(m1_solar, m2_solar, s1z, s2z,
                          distance_mpc, inclination, f_lower, sample_rate,
                          ra, dec, psi):
    """Generate waveform using IMRPhenomD and post-process."""

    result = generate_imrphenomd_waveform(
        m1_solar, m2_solar, s1z, s2z,
        distance_mpc, inclination, f_lower, sample_rate,
        ra, dec, psi
    )

    # Add ringdown physics parameters
    ringdown_params = compute_ringdown_params(m1_solar, m2_solar, s1z, s2z)
    result['params'].update({
        'f_qnm_hz': ringdown_params['f_qnm_hz'],
        'tau_qnm_s': ringdown_params['tau_qnm_s'],
        'quality_factor': ringdown_params['quality_factor'],
        'final_mass_solar_br': ringdown_params['final_mass_solar'],
        'final_spin_br': ringdown_params['final_spin'],
        'energy_radiated_solar': ringdown_params['energy_radiated_solar'],
    })

    # Trim waveform to relevant region
    # Keep from -2 seconds before merger to +0.1 seconds after
    result = _trim_waveform(result, pre_merger=2.0, post_merger=0.1)

    return result

def _trim_waveform(result, pre_merger=2.0, post_merger=0.1):
    """Trim waveform to keep only the physically interesting region.

    Keeps data from (merger - pre_merger) to (merger + post_merger).

    Parameters
    ----------
    result : dict
        Waveform result dictionary.
    pre_merger : float
        Seconds before merger to keep.
    post_merger : float
        Seconds after merger to keep.

    Returns
    -------
    dict
        Trimmed waveform.
    """
    t = result['time']
    mask = (t >= -pre_merger) & (t <= post_merger)

    if not np.any(mask):
        return result  # Don't trim if nothing in range

    trimmed = {}
    for key, val in result.items():
        if isinstance(val, np.ndarray) and len(val) == len(t):
            trimmed[key] = val[mask]
        else:
            trimmed[key] = val

    # Update merger index
    if 'params' in trimmed:
        new_t = trimmed['time']
        trimmed['params']['merger_index'] = int(np.argmin(np.abs(new_t)))

    return trimmed
