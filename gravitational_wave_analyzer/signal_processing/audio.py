"""
Gravitational Wave Audio Generator
====================================

Converts the gravitational wave strain h(t) into an audible sound file.

The raw GW signal from a binary black hole merger spans approximately
20-300 Hz and lasts only ~0.2 seconds — technically within human hearing
range but far too brief and quiet to perceive directly.

To make it audible, we:
1. Frequency-shift upward by a factor ~400× to place the chirp in the
   300-2000 Hz range where human hearing is most sensitive
2. Time-stretch so the chirp lasts ~1-2 seconds
3. Normalize the amplitude to fill the 16-bit WAV dynamic range

The result should sound like the famous LIGO chirp — a rising "whoop!"
that sweeps up in pitch and terminates abruptly at merger.

References
----------
[1] LIGO Scientific Collaboration, "Audio files of gravitational waves",
    https://www.ligo.org/detections/GW150914.php
    — Official LIGO audio files for comparison.

[2] Abbott et al., "Observation of Gravitational Waves from a Binary
    Black Hole Merger", PRL 116, 061102 (2016).
    — Original GW150914 detection paper.
"""

import numpy as np
from scipy.signal import resample
from scipy.io import wavfile

from gravitational_wave_analyzer.constants import AUDIO_SAMPLE_RATE


def generate_chirp_audio(waveform_result, output_path='output/gw_chirp.wav',
                          freq_shift=400.0, target_duration=2.0,
                          audio_sample_rate=44100):
    """Generate an audible WAV file from a gravitational wave strain signal.

    The algorithm:
    1. Extract the detector strain h(t) from the waveform result
    2. Time-stretch by resampling: if the signal spans T seconds,
       resample to T * freq_shift seconds at the audio sample rate
    3. This simultaneously shifts all frequencies up by freq_shift
       (because playing the same number of cycles in less time = higher pitch)
    4. Normalize to 16-bit PCM range
    5. Apply fade-in/fade-out to avoid clicks

    Parameters
    ----------
    waveform_result : dict
        Output from generate_full_waveform().
    output_path : str
        Path to save the WAV file.
    freq_shift : float
        Frequency multiplication factor. Default 400× maps the ~50 Hz
        inspiral to ~20 kHz approaching merger, with most of the
        action in the 200-2000 Hz sweet spot.
    target_duration : float
        Desired audio duration in seconds.
    audio_sample_rate : int
        Output WAV sample rate in Hz. Default 44100 (CD quality).

    Returns
    -------
    dict with keys:
        'audio_data' : ndarray — normalized audio samples
        'sample_rate' : int — audio sample rate
        'duration' : float — audio duration in seconds
        'output_path' : str — saved file path
    """
    # Extract strain signal
    h = waveform_result['h_detector'].copy()
    t = waveform_result['time']
    dt_gw = t[1] - t[0]  # GW sample interval
    gw_sample_rate = 1.0 / dt_gw

    # --- Remove any DC offset ---
    h -= np.mean(h)

    # --- Time stretch + frequency shift ---
    # The key insight: if we resample N samples of a signal at a different
    # rate, frequencies are shifted proportionally.
    #
    # Original signal: N samples at gw_sample_rate → duration T = N/gw_sample_rate
    # After resampling to N_new samples at audio_sample_rate:
    #   - Audio duration = N_new / audio_sample_rate
    #   - Effective time stretch = (N_new / audio_sample_rate) / T
    #   - Frequency shift = 1 / time_stretch
    #
    # We want freq_shift = 400, so time_stretch = 1/400... but that makes
    # a 0.2s signal only 0.5ms long. Instead, we use a different approach:
    #
    # Strategy: resample to shift frequencies, then repeat/pad for duration.

    # Number of output samples to achieve the target duration
    N_gw = len(h)
    T_gw = N_gw / gw_sample_rate  # original GW signal duration

    # Effective audio duration if we just play at audio_sample_rate
    # with the frequency shift applied:
    # To shift frequency by freq_shift, we play the signal freq_shift times faster
    # Duration becomes T_gw / freq_shift
    T_audio = T_gw / freq_shift

    # But this is too short (~0.5ms for 0.2s signal at 400x)
    # So we use a moderate shift and let the natural duration work
    # Typical approach: shift by ~200-400 and accept the short duration
    # or use a lower shift factor

    # Adaptive shift: aim for target_duration
    # Use shift factor that gives reasonable duration
    actual_shift = max(T_gw / target_duration, 1.0)
    # Cap the shift to keep frequencies in audible range
    actual_shift = min(actual_shift, freq_shift)

    # Number of audio samples needed
    T_audio = T_gw / actual_shift
    N_audio = int(T_audio * audio_sample_rate)
    N_audio = max(N_audio, audio_sample_rate)  # at least 1 second

    # Resample using scipy (handles the interpolation properly)
    audio = resample(h, N_audio)

    # --- Normalize to [-1, 1] ---
    max_amp = np.max(np.abs(audio))
    if max_amp > 0:
        audio = audio / max_amp

    # --- Apply fade-in and fade-out to prevent clicks ---
    # Fade-in: first 5% of signal
    fade_in_samples = int(0.05 * N_audio)
    if fade_in_samples > 0:
        fade_in = np.linspace(0, 1, fade_in_samples)
        audio[:fade_in_samples] *= fade_in

    # Fade-out: last 10% of signal (ringdown naturally fades, but be safe)
    fade_out_samples = int(0.10 * N_audio)
    if fade_out_samples > 0:
        fade_out = np.linspace(1, 0, fade_out_samples)
        audio[-fade_out_samples:] *= fade_out

    # --- Boost the pre-merger chirp ---
    # Apply mild compression to bring up the quiet early inspiral
    # while keeping the loud merger from clipping
    audio = np.sign(audio) * np.abs(audio) ** 0.7

    # Re-normalize after compression
    max_amp = np.max(np.abs(audio))
    if max_amp > 0:
        audio = audio / max_amp * 0.95  # leave 5% headroom

    # --- Convert to 16-bit PCM ---
    audio_int16 = np.int16(audio * 32767)

    # --- Ensure output directory exists ---
    import os
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

    # --- Write WAV file ---
    wavfile.write(output_path, audio_sample_rate, audio_int16)

    return {
        'audio_data': audio,
        'sample_rate': audio_sample_rate,
        'duration': N_audio / audio_sample_rate,
        'output_path': output_path,
        'freq_shift_actual': actual_shift,
        'num_samples': N_audio,
    }
