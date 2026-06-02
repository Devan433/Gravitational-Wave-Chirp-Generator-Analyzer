"""
Gravitational Wave Chirp Generator & Analyzer — Main Entry Point
=================================================================

This script orchestrates the complete gravitational wave analysis pipeline:

    1. Parse user parameters (or use GW150914 defaults)
    2. Generate the full IMR waveform using IMRPhenomD
    3. Compute ringdown parameters (final mass, spin, QNM frequencies)
    4. Generate the audible chirp WAV file
    5. Compute the Q-transform spectrogram
    6. Run the matched filtering detection pipeline
    7. Create the 7-panel publication dashboard
    8. Print a physics summary including GW150914 validation

Usage
-----
    # Default: GW150914 parameters
    python main.py

    # Custom masses and distance
    python main.py --m1 50 --m2 30 --distance 800

    # With spins
    python main.py --m1 36 --m2 29 --s1z 0.3 --s2z -0.1 --snr 30

    # Full specification (GW150914 sky location)
    python main.py --m1 36 --m2 29 --s1z 0 --s2z 0 --distance 410 \\
                   --inclination 0.3 --ra 1.95 --dec -1.27 --snr 25

References
----------
[1] Abbott et al., "Observation of Gravitational Waves from a Binary
    Black Hole Merger", PRL 116, 061102 (2016).
    GW150914: m1=36, m2=29, d=410 Mpc, a_f=0.67, M_f=62 Msun
"""

import argparse
import os
import sys
import time
import numpy as np


def parse_arguments():
    """Parse command-line arguments with GW150914 defaults."""
    parser = argparse.ArgumentParser(
        description='Gravitational Wave Chirp Generator & Analyzer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                          # GW150914 defaults
  python main.py --m1 50 --m2 30          # Heavier binary
  python main.py --m1 36 --m2 29 --snr 50 # Strong injection
        """
    )

    # --- Binary parameters ---
    parser.add_argument('--m1', type=float, default=36.0,
                        help='Mass of primary BH in solar masses (default: 36, range: 5-100)')
    parser.add_argument('--m2', type=float, default=29.0,
                        help='Mass of secondary BH in solar masses (default: 29, range: 5-100)')
    parser.add_argument('--s1z', type=float, default=0.0,
                        help='Spin of primary along L (default: 0, range: -0.99 to 0.99)')
    parser.add_argument('--s2z', type=float, default=0.0,
                        help='Spin of secondary along L (default: 0, range: -0.99 to 0.99)')
    parser.add_argument('--distance', type=float, default=410.0,
                        help='Luminosity distance in Mpc (default: 410)')
    parser.add_argument('--inclination', type=float, default=0.0,
                        help='Orbital inclination in radians (default: 0 = face-on)')
    parser.add_argument('--ra', type=float, default=1.95,
                        help='Right ascension in radians (default: 1.95)')
    parser.add_argument('--dec', type=float, default=-1.27,
                        help='Declination in radians (default: -1.27)')
    parser.add_argument('--psi', type=float, default=0.0,
                        help='Polarization angle in radians (default: 0)')

    # --- Pipeline parameters ---
    parser.add_argument('--snr', type=float, default=25.0,
                        help='Target injection SNR for matched filtering (default: 25)')
    parser.add_argument('--sample-rate', type=int, default=4096,
                        help='Sample rate in Hz (default: 4096, LIGO standard)')
    parser.add_argument('--f-lower', type=float, default=20.0,
                        help='Lower frequency cutoff in Hz (default: 20)')
    parser.add_argument('--output-dir', type=str, default='output',
                        help='Output directory (default: output/)')
    parser.add_argument('--method', type=str, default='imrphenomd',
                        choices=['imrphenomd', 'pn_ringdown'],
                        help='Waveform method (default: imrphenomd)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for noise generation (default: 42)')
    parser.add_argument('--skip-audio', action='store_true',
                        help='Skip audio generation')
    parser.add_argument('--skip-spectrogram', action='store_true',
                        help='Skip Q-transform (can be slow)')
    parser.add_argument('--skip-matched-filter', action='store_true',
                        help='Skip matched filtering pipeline')

    return parser.parse_args()


def print_header():
    """Print a beautiful header."""
    print("\n" + "=" * 72)
    print("  GRAVITATIONAL WAVE CHIRP GENERATOR & ANALYZER")
    print("  Pure Python Implementation - IMRPhenomD Waveform Model")
    print("  Validated against GW150914 (LIGO/Virgo)")
    print("=" * 72)


def print_physics_summary(params, detection_result=None, audio_result=None):
    """Print a comprehensive physics summary to the console."""
    print("\n" + "-" * 72)
    print("  PHYSICS SUMMARY")
    print("-" * 72)

    print(f"\n  Binary System:")
    print(f"    Total mass:            M  = {params.get('total_mass_solar', 0):.2f} Msun")
    print(f"    Chirp mass:            Mc = {params.get('chirp_mass_solar', 0):.2f} Msun")
    print(f"    Symmetric mass ratio:  eta = {params.get('symmetric_mass_ratio', 0):.4f}")
    print(f"    Effective spin:        chi = {params.get('effective_spin', 0):.3f}")
    print(f"    Distance:              dL  = {params.get('distance_mpc', 0):.0f} Mpc")

    print(f"\n  Inspiral:")
    print(f"    Start frequency:       f_low  = {params.get('f_start_hz', params.get('f_isco_hz', 20)):.1f} Hz")
    print(f"    ISCO frequency:        f_ISCO = {params.get('f_isco_hz', 0):.1f} Hz")

    print(f"\n  Ringdown:")
    print(f"    Ring frequency:        f_ring = {params.get('f_ring_hz', 0):.1f} Hz")
    print(f"    Damping time:          t_QNM  = {params.get('tau_qnm_s', 0)*1000:.2f} ms")
    print(f"    Quality factor:        Q      = {params.get('quality_factor', 0):.1f}")

    print(f"\n  Final State:")
    print(f"    Final mass:            Mf = {params.get('final_mass_solar', params.get('final_mass_solar_br', 0)):.1f} Msun")
    print(f"    Final spin:            af = {params.get('final_spin', params.get('final_spin_br', 0)):.3f}")
    e_rad = params.get('energy_radiated_solar', 0)
    print(f"    Energy radiated:       dE = {e_rad:.2f} Msun*c^2")
    if e_rad > 0:
        # Convert to watts for perspective
        E_joules = e_rad * 1.989e30 * (3e8)**2
        print(f"                              = {E_joules:.2e} Joules")
        print(f"                              (more than all stars in the")
        print(f"                               observable universe emit in 1 second)")

    print(f"\n  Signal:")
    print(f"    Peak strain:           |h| = {params.get('peak_strain', 0):.2e}")
    print(f"    Duration (20 Hz):      T   = {params.get('duration_seconds', 0):.3f} s")

    if detection_result is not None:
        print(f"\n  Detection:")
        print(f"    Optimal SNR:           rho_opt  = {detection_result.get('optimal_snr', 0):.1f}")
        print(f"    Recovered SNR:         rho_peak = {detection_result.get('peak_snr', 0):.1f}")
        print(f"    Target SNR:            rho_tgt  = {detection_result.get('target_snr', 0):.1f}")

    if audio_result is not None:
        print(f"\n  Audio:")
        print(f"    Duration:              T_audio = {audio_result.get('duration', 0):.2f} s")
        print(f"    Frequency shift:       x{audio_result.get('freq_shift_actual', 0):.0f}")
        print(f"    Output:                {audio_result.get('output_path', 'N/A')}")

    print()


def validate_against_gw150914(params):
    """Compare our results to GW150914 published values."""
    print("\n" + "-" * 72)
    print("  GW150914 VALIDATION")
    print("  Reference: Abbott et al., PRL 116, 061102 (2016)")
    print("-" * 72)

    # Published values with uncertainties
    # Source: Abbott et al. (2016), Table I
    published = {
        'chirp_mass_solar':  (28.3, 30.0, 32.3),   # (low, central, high)
        'final_mass_solar':  (58.0, 62.0, 66.0),
        'final_spin':        (0.60, 0.67, 0.72),
        'energy_radiated_solar': (2.5, 3.0, 3.5),   # approximate
        'f_isco_hz':         (None, 67.6, None),     # Schwarzschild, M=65 Msun
    }

    our = {
        'chirp_mass_solar': params.get('chirp_mass_solar', 0),
        'final_mass_solar': params.get('final_mass_solar',
                                       params.get('final_mass_solar_br', 0)),
        'final_spin': params.get('final_spin',
                                 params.get('final_spin_br', 0)),
        'energy_radiated_solar': params.get('energy_radiated_solar', 0),
        'f_isco_hz': params.get('f_isco_hz', 0),
    }

    print(f"\n  {'Parameter':<25} {'Our Value':>12} {'Published':>12} {'Status':>10}")
    print(f"  {'-' * 25} {'-' * 12} {'-' * 12} {'-' * 10}")

    for key, (low, central, high) in published.items():
        our_val = our.get(key, 0)
        pub_str = f"{central:.2f}"

        if low is not None and high is not None:
            if low <= our_val <= high:
                status = "  PASS"
            elif abs(our_val - central) / central < 0.15:
                status = "~ CLOSE"
            else:
                status = "x CHECK"
        else:
            if abs(our_val - central) / central < 0.10:
                status = "  PASS"
            else:
                status = "~ CHECK"

        label = key.replace('_', ' ').title()
        print(f"  {label:<25} {our_val:>12.2f} {pub_str:>12} {status:>10}")

    print()


def main():
    """Main entry point — orchestrates the complete analysis pipeline."""
    args = parse_arguments()
    print_header()

    # --- Validate inputs ---
    assert 5 <= args.m1 <= 200, "m1 must be between 5 and 200 solar masses"
    assert 5 <= args.m2 <= 200, "m2 must be between 5 and 200 solar masses"
    assert -0.99 <= args.s1z <= 0.99, "s1z must be between -0.99 and 0.99"
    assert -0.99 <= args.s2z <= 0.99, "s2z must be between -0.99 and 0.99"
    assert args.distance > 0, "Distance must be positive"

    # Ensure m1 >= m2
    if args.m2 > args.m1:
        args.m1, args.m2 = args.m2, args.m1
        args.s1z, args.s2z = args.s2z, args.s1z
        print("  [Note: Swapped m1 and m2 so that m1 >= m2]")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n  Parameters:")
    print(f"    m1 = {args.m1:.1f} Msun,  m2 = {args.m2:.1f} Msun")
    print(f"    s1z = {args.s1z:.2f},    s2z = {args.s2z:.2f}")
    print(f"    distance = {args.distance:.0f} Mpc")
    print(f"    inclination = {args.inclination:.2f} rad")
    print(f"    method = {args.method}")

    # Step 1: Generate full IMR waveform
    print("\n  [1/6] Generating waveform...", end=" ", flush=True)
    t0 = time.time()

    from gravitational_wave_analyzer.physics.full_waveform import generate_full_waveform

    waveform = generate_full_waveform(
        m1_solar=args.m1,
        m2_solar=args.m2,
        s1z=args.s1z,
        s2z=args.s2z,
        distance_mpc=args.distance,
        inclination=args.inclination,
        f_lower=args.f_lower,
        sample_rate=args.sample_rate,
        ra=args.ra,
        dec=args.dec,
        psi=args.psi,
        method=args.method,
    )

    print(f"done ({time.time() - t0:.2f}s)")
    print(f"        Waveform length: {len(waveform['time'])} samples "
          f"({waveform['params'].get('duration_seconds', 0):.3f}s)")

    # Step 2: Generate audio chirp
    audio_result = None
    if not args.skip_audio:
        print("  [2/6] Generating audio chirp...", end=" ", flush=True)
        t0 = time.time()

        from gravitational_wave_analyzer.signal_processing.audio import generate_chirp_audio

        audio_path = os.path.join(args.output_dir, 'gw_chirp.wav')
        audio_result = generate_chirp_audio(
            waveform, output_path=audio_path,
        )
        print(f"done ({time.time() - t0:.2f}s)")
        print(f"        Saved: {audio_path} ({audio_result['duration']:.2f}s)")
    else:
        print("  [2/6] Audio generation skipped.")

    # Step 3: Q-transform spectrogram
    spectrogram_result = None
    freq_track = None
    if not args.skip_spectrogram:
        print("  [3/6] Computing Q-transform spectrogram...", end=" ", flush=True)
        t0 = time.time()

        from gravitational_wave_analyzer.signal_processing.spectrogram import (
            q_transform, theoretical_frequency_track,
        )

        spectrogram_result = q_transform(
            waveform['h_detector'],
            args.sample_rate,
            q_range=(4, 64),
            f_range=(20, 1024),
            n_freq_bins=150,
            tres=0.001,
        )

        # Compute theoretical frequency track
        merger_time = waveform['time'][waveform['params'].get('merger_index',
                                                               len(waveform['time']) // 2)]
        n_track = len(spectrogram_result['times'])
        t_track = np.linspace(waveform['time'][0], waveform['time'][-1], n_track)
        freq_track = theoretical_frequency_track(
            args.m1, args.m2, 0.0, t_track, args.f_lower
        )

        print(f"done ({time.time() - t0:.2f}s)")
    else:
        print("  [3/6] Spectrogram computation skipped.")

    # Step 4: Matched filtering
    detection_result = None
    if not args.skip_matched_filter:
        print("  [4/6] Running matched filter pipeline...", end=" ", flush=True)
        t0 = time.time()

        from gravitational_wave_analyzer.signal_processing.matched_filter import (
            run_detection_pipeline,
        )

        detection_result = run_detection_pipeline(
            waveform,
            target_snr=args.snr,
            noise_duration=16.0,
            sample_rate=args.sample_rate,
            f_lower=args.f_lower,
            seed=args.seed,
        )

        print(f"done ({time.time() - t0:.2f}s)")
        print(f"        Peak SNR: {detection_result['peak_snr']:.1f} "
              f"(target: {args.snr:.1f})")
    else:
        print("  [4/6] Matched filtering skipped.")

    # Step 5: Generate dashboard
    print("  [5/6] Creating dashboard...", end=" ", flush=True)
    t0 = time.time()

    from gravitational_wave_analyzer.visualization.dashboard import create_dashboard

    dashboard_path = os.path.join(args.output_dir, 'dashboard.png')
    create_dashboard(
        waveform,
        detection_result=detection_result,
        spectrogram_result=spectrogram_result,
        freq_track=freq_track,
        output_path=dashboard_path,
        dpi=300,
    )

    print(f"done ({time.time() - t0:.2f}s)")

    # Step 6: Print results
    print("  [6/6] Computing results...")

    print_physics_summary(waveform['params'], detection_result, audio_result)

    # --- GW150914 validation (only if using default masses) ---
    if abs(args.m1 - 36.0) < 1 and abs(args.m2 - 29.0) < 1:
        validate_against_gw150914(waveform['params'])

    # --- Final output summary ---
    print("-" * 72)
    print("  OUTPUT FILES")
    print("-" * 72)
    print(f"    Dashboard:     {dashboard_path}")
    if audio_result:
        print(f"    Audio chirp:   {audio_result['output_path']}")
    print(f"    Output dir:    {os.path.abspath(args.output_dir)}")
    print()
    print("=" * 72)
    print("  Analysis complete. All outputs saved.")
    print("=" * 72 + "\n")


if __name__ == '__main__':
    main()
