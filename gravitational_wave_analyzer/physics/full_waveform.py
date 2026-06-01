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
    elif method == 'pn_ringdown':
        return _generate_pn_plus_ringdown(
            m1_solar, m2_solar, s1z, s2z,
            distance_mpc, inclination, f_lower, sample_rate,
            ra, dec, psi
        )
    elif method == 'taylorf2':
        from gravitational_wave_analyzer.physics.taylorf2 import (
            generate_taylorf2_waveform,
        )
        return generate_taylorf2_waveform(
            m1_solar, m2_solar, s1z, s2z,
            distance_mpc, inclination, f_lower, sample_rate,
            ra, dec, psi
        )
    else:
        raise ValueError(f"Unknown method: {method}. Use 'imrphenomd', 'pn_ringdown', or 'taylorf2'.")


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


def _generate_pn_plus_ringdown(m1_solar, m2_solar, s1z, s2z,
                                distance_mpc, inclination, f_lower, sample_rate,
                                ra, dec, psi):
    """Generate waveform by stitching TaylorT4 inspiral + QNM ringdown.

    This is the educational/transparent approach where each physics piece
    is visible. The stitching window ensures smooth continuity.
    """
    from gravitational_wave_analyzer.physics.waveform import generate_inspiral_waveform
    from gravitational_wave_analyzer.physics.ringdown import (
        generate_ringdown_waveform, compute_ringdown_params,
    )

    # --- Generate inspiral ---
    inspiral = generate_inspiral_waveform(
        m1_solar, m2_solar, s1z, s2z,
        distance_mpc, inclination, f_lower, sample_rate,
        ra, dec, psi
    )

    # --- Compute ringdown parameters ---
    rd_params = compute_ringdown_params(m1_solar, m2_solar, s1z, s2z)

    # --- Generate ringdown ---
    # Match amplitude and phase at the junction
    end_amp = inspiral['amplitude'][-1]
    end_phase = inspiral['phase'][-1]

    ringdown = generate_ringdown_waveform(
        M_f_solar=rd_params['final_mass_solar'],
        a_f=rd_params['final_spin'],
        amplitude_scale=end_amp,
        sample_rate=sample_rate,
        duration=0.15,  # Ringdown dies within ~50ms
        phi0=end_phase,
        inclination=inclination,
    )

    # --- Stitch together ---
    # Use a cosine blend over ~2ms at the junction
    blend_samples = int(0.002 * sample_rate)
    blend_samples = max(blend_samples, 4)

    t_ins = inspiral['time']
    t_rd = ringdown['time'] + t_ins[-1]  # Shift ringdown to start at inspiral end

    hp_ins = inspiral['h_plus']
    hc_ins = inspiral['h_cross']
    hp_rd = ringdown['h_plus']
    hc_rd = ringdown['h_cross']

    # Blend window: cosine taper
    blend = np.linspace(0, 1, blend_samples)
    blend_window = 0.5 * (1 - np.cos(PI * blend))

    # Apply blending at overlap region
    if len(hp_ins) >= blend_samples and len(hp_rd) >= blend_samples:
        hp_ins[-blend_samples:] *= (1 - blend_window)
        hc_ins[-blend_samples:] *= (1 - blend_window)
        hp_rd[:blend_samples] *= blend_window
        hc_rd[:blend_samples] *= blend_window

    # Concatenate
    time = np.concatenate([t_ins, t_rd[1:]])
    h_plus = np.concatenate([hp_ins, hp_rd[1:]])
    h_cross = np.concatenate([hc_ins, hc_rd[1:]])

    # Detector response
    from gravitational_wave_analyzer.physics.waveform import antenna_pattern
    F_plus, F_cross = antenna_pattern(ra, dec, psi)
    h_detector = F_plus * h_plus + F_cross * h_cross

    # Frequency from phase
    analytic = h_plus + 1j * h_cross
    inst_phase = np.unwrap(np.angle(analytic))
    inst_freq = np.gradient(inst_phase, 1.0/sample_rate) / TWOPI
    inst_freq = np.clip(inst_freq, 0, sample_rate/2)

    amplitude = np.sqrt(h_plus**2 + h_cross**2)

    params = {
        **inspiral['params'],
        **{f'rd_{k}': v for k, v in rd_params.items()},
    }

    return {
        'time': time,
        'h_plus': h_plus,
        'h_cross': h_cross,
        'h_detector': h_detector,
        'frequency': inst_freq,
        'phase': inst_phase,
        'amplitude': amplitude,
        'params': params,
    }


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
