"""
7-Panel Publication-Quality Dashboard
=======================================

Generates a comprehensive matplotlib figure displaying all aspects
of the gravitational wave analysis in a single dashboard:

    Panel 1: Time-domain waveform h(t) — h+ and h× polarizations
    Panel 2: Frequency-domain amplitude spectral density vs LIGO noise
    Panel 3: Q-transform spectrogram (the iconic LIGO plot)
    Panel 4: Matched filter SNR time series
    Panel 5: Frequency evolution f(t)
    Panel 6: Phase evolution Φ(t)
    Panel 7: Parameter summary table

The layout uses a dark theme inspired by LIGO's publication style.

References
----------
[1] Abbott et al., PRL 116, 061102 (2016), Figures 1-3.
    Style reference for GW150914 visualization.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for file output
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LogNorm
import matplotlib.ticker as mticker

from gravitational_wave_analyzer.constants import PI, TWOPI


def create_dashboard(waveform_result, detection_result=None,
                      spectrogram_result=None, freq_track=None,
                      output_path='output/dashboard.png',
                      dpi=300):
    """Create the complete 7-panel analysis dashboard.

    Parameters
    ----------
    waveform_result : dict
        Output from generate_full_waveform().
    detection_result : dict, optional
        Output from run_detection_pipeline().
    spectrogram_result : dict, optional
        Output from q_transform().
    freq_track : ndarray, optional
        Theoretical frequency track for overlay.
    output_path : str
        Path to save the dashboard image.
    dpi : int
        Resolution of output image.

    Returns
    -------
    str
        Path to saved dashboard image.
    """
    import os
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

    # --- Dark theme setup ---
    plt.style.use('dark_background')

    # Custom color palette
    colors = {
        'h_plus': '#00d4ff',      # cyan-blue for h+
        'h_cross': '#ff6b35',     # orange for h×
        'frequency': '#00ff88',   # green for frequency
        'phase': '#ff44cc',       # magenta for phase
        'snr': '#ffdd00',         # gold for SNR
        'noise': '#666666',       # gray for noise
        'ligo_curve': '#ff4444',  # red for LIGO ASD
        'signal_asd': '#00d4ff',  # cyan for signal ASD
        'grid': '#333333',        # subtle grid
        'text': '#ffffff',        # white text
        'accent': '#00d4ff',      # accent color
        'bg': '#0a0a0a',          # near-black background
        'panel_bg': '#111111',    # panel background
    }

    # --- Figure layout ---
    fig = plt.figure(figsize=(24, 18), facecolor=colors['bg'])
    fig.patch.set_facecolor(colors['bg'])

    # GridSpec: 4 rows × 3 columns
    # Row 0: waveform (2 cols) + ASD (1 col)
    # Row 1: spectrogram (2 cols) + frequency evolution (1 col)
    # Row 2: SNR (2 cols) + phase evolution (1 col)
    # Row 3: parameter summary (full width)
    gs = GridSpec(4, 3, figure=fig,
                  height_ratios=[1, 1, 1, 0.6],
                  hspace=0.35, wspace=0.3,
                  left=0.06, right=0.96, top=0.94, bottom=0.04)

    params = waveform_result['params']

    # ===================================================================
    # Panel 1: Time-domain waveform h(t)
    # ===================================================================
    ax1 = fig.add_subplot(gs[0, 0:2])
    ax1.set_facecolor(colors['panel_bg'])

    t = waveform_result['time']
    hp = waveform_result['h_plus']
    hc = waveform_result['h_cross']

    # Focus on the last second (most interesting part)
    t_start = max(t[0], -0.5)
    mask_t = (t >= t_start) & (t <= 0.05)

    if np.any(mask_t):
        t_plot = t[mask_t] * 1000  # convert to milliseconds
        hp_plot = hp[mask_t]
        hc_plot = hc[mask_t]
    else:
        t_plot = t * 1000
        hp_plot = hp
        hc_plot = hc

    ax1.plot(t_plot, hp_plot, color=colors['h_plus'], alpha=0.9,
             linewidth=0.8, label=r'$h_+$ (plus)')
    ax1.plot(t_plot, hc_plot, color=colors['h_cross'], alpha=0.7,
             linewidth=0.8, label=r'$h_\times$ (cross)')

    # Mark merger
    ax1.axvline(x=0, color='white', linestyle='--', alpha=0.3, linewidth=0.5)
    ax1.text(0, ax1.get_ylim()[1] * 0.85, ' merger', color='white',
             fontsize=8, alpha=0.5)

    ax1.set_xlabel('Time [ms]', fontsize=11, color=colors['text'])
    ax1.set_ylabel('Strain h(t)', fontsize=11, color=colors['text'])
    ax1.set_title('Gravitational Wave Strain — Time Domain',
                  fontsize=13, color=colors['accent'], fontweight='bold')
    ax1.legend(loc='upper left', fontsize=9, framealpha=0.3)
    ax1.grid(True, alpha=0.15, color=colors['grid'])
    ax1.ticklabel_format(axis='y', style='scientific', scilimits=(-21, -21))

    # ===================================================================
    # Panel 2: Frequency-domain ASD
    # ===================================================================
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.set_facecolor(colors['panel_bg'])

    # LIGO noise curve
    from gravitational_wave_analyzer.data.ligo_sensitivity import aLIGO_asd
    f_noise = np.geomspace(10, 3000, 1000)
    asd_noise = aLIGO_asd(f_noise)

    ax2.loglog(f_noise, asd_noise, color=colors['ligo_curve'],
               linewidth=1.5, alpha=0.8, label='aLIGO design')

    # Signal ASD
    if 'f_array' in waveform_result and 'amplitude_fd' in waveform_result:
        f_sig = waveform_result['f_array']
        a_sig = waveform_result['amplitude_fd']
        # Convert amplitude to ASD-like units for comparison
        # ASD ~ |h̃(f)| × √f
        asd_sig = np.abs(a_sig) * np.sqrt(f_sig)
        # Scale for visibility
        scale = np.median(asd_noise[f_noise > 30]) / np.median(asd_sig[asd_sig > 0] + 1e-100)
        ax2.loglog(f_sig, asd_sig * scale * 0.5, color=colors['signal_asd'],
                   linewidth=1.0, alpha=0.8, label='GW signal')

    # Mark key frequencies
    if 'f_isco_hz' in params:
        ax2.axvline(x=params['f_isco_hz'], color=colors['frequency'],
                    linestyle=':', alpha=0.4, linewidth=0.8)
        ax2.text(params['f_isco_hz'] * 1.1, ax2.get_ylim()[0] if ax2.get_ylim()[0] > 0 else 1e-24,
                 f"$f_{{ISCO}}$={params['f_isco_hz']:.0f}Hz",
                 color=colors['frequency'], fontsize=7, alpha=0.6)

    ax2.set_xlabel('Frequency [Hz]', fontsize=11, color=colors['text'])
    ax2.set_ylabel(r'ASD [strain/$\sqrt{\rm Hz}$]', fontsize=11, color=colors['text'])
    ax2.set_title('Amplitude Spectral Density', fontsize=13,
                  color=colors['accent'], fontweight='bold')
    ax2.legend(loc='upper right', fontsize=8, framealpha=0.3)
    ax2.set_xlim(10, 3000)
    ax2.set_ylim(1e-25, 1e-19)
    ax2.grid(True, which='both', alpha=0.1, color=colors['grid'])

    # ===================================================================
    # Panel 3: Q-transform spectrogram
    # ===================================================================
    ax3 = fig.add_subplot(gs[1, 0:2])
    ax3.set_facecolor(colors['panel_bg'])

    if spectrogram_result is not None:
        energy = spectrogram_result['energy']
        s_times = spectrogram_result['times']
        s_freqs = spectrogram_result['frequencies']

        # Shift time so merger is at t=0
        merger_idx = params.get('merger_index', len(t) // 2)
        t_offset = t[min(merger_idx, len(t) - 1)]
        s_times_shifted = s_times - (s_times[-1] / 2)  # center roughly

        # Plot with log frequency axis
        energy_plot = np.maximum(energy, 1e-10)

        pcm = ax3.pcolormesh(
            s_times_shifted * 1000,  # ms
            s_freqs,
            energy_plot,
            cmap='inferno',
            shading='gouraud',
            rasterized=True,
        )

        ax3.set_yscale('log')
        ax3.set_ylim(20, 1024)
        ax3.set_ylabel('Frequency [Hz]', fontsize=11, color=colors['text'])
        ax3.set_xlabel('Time [ms]', fontsize=11, color=colors['text'])

        # Overlay theoretical frequency track
        if freq_track is not None:
            track_times = np.linspace(s_times_shifted[0], s_times_shifted[-1], len(freq_track))
            valid_track = ~np.isnan(freq_track) & (freq_track > 20) & (freq_track < 1024)
            if np.any(valid_track):
                ax3.plot(track_times[valid_track] * 1000,
                         freq_track[valid_track],
                         color='white', linewidth=1.0, alpha=0.6,
                         linestyle='--', label='Theory')
                ax3.legend(loc='upper left', fontsize=8, framealpha=0.3)

        # Colorbar
        cbar = plt.colorbar(pcm, ax=ax3, pad=0.01, aspect=30)
        cbar.set_label('Normalized Energy', fontsize=9, color=colors['text'])
    else:
        ax3.text(0.5, 0.5, 'Q-transform not computed',
                 transform=ax3.transAxes, ha='center', va='center',
                 color=colors['text'], fontsize=12)

    ax3.set_title('Q-Transform Spectrogram',
                  fontsize=13, color=colors['accent'], fontweight='bold')

    # ===================================================================
    # Panel 4: SNR time series
    # ===================================================================
    ax4 = fig.add_subplot(gs[2, 0:2])
    ax4.set_facecolor(colors['panel_bg'])

    if detection_result is not None:
        snr_ts = detection_result['snr_timeseries']
        t_snr = detection_result['time']

        # Focus on the region around the peak
        peak_idx = detection_result.get('peak_time_index',
                                         np.argmax(snr_ts))
        dt_snr = t_snr[1] - t_snr[0]

        # Window: ±1 second around peak
        window_samples = int(1.0 / dt_snr)
        i_start = max(0, peak_idx - window_samples)
        i_end = min(len(snr_ts), peak_idx + window_samples)

        t_plot_snr = (t_snr[i_start:i_end] - t_snr[peak_idx]) * 1000  # ms
        snr_plot = snr_ts[i_start:i_end]

        ax4.plot(t_plot_snr, snr_plot, color=colors['snr'],
                 linewidth=0.8, alpha=0.9)

        # Detection threshold
        ax4.axhline(y=8.0, color='red', linestyle='--', alpha=0.5,
                    linewidth=1.0, label='Detection threshold (ρ=8)')

        # Mark peak
        peak_snr = detection_result['peak_snr']
        ax4.plot(0, peak_snr, 'v', color='white', markersize=8)
        ax4.text(5, peak_snr, f' ρ_peak = {peak_snr:.1f}',
                 color='white', fontsize=10, fontweight='bold')

        ax4.legend(loc='upper left', fontsize=9, framealpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'Matched filter not run',
                 transform=ax4.transAxes, ha='center', va='center',
                 color=colors['text'], fontsize=12)

    ax4.set_xlabel('Time relative to peak [ms]', fontsize=11, color=colors['text'])
    ax4.set_ylabel('SNR ρ(t)', fontsize=11, color=colors['text'])
    ax4.set_title('Matched Filter SNR Time Series',
                  fontsize=13, color=colors['accent'], fontweight='bold')
    ax4.grid(True, alpha=0.15, color=colors['grid'])

    # ===================================================================
    # Panel 5: Frequency evolution
    # ===================================================================
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.set_facecolor(colors['panel_bg'])

    freq = waveform_result['frequency']
    if np.any(mask_t):
        freq_plot = freq[mask_t]
        # Smooth the frequency evolution (can be noisy from numerical derivatives)
        from scipy.ndimage import uniform_filter1d
        freq_smooth = uniform_filter1d(freq_plot, size=min(50, len(freq_plot) // 10 + 1))
        freq_smooth = np.clip(freq_smooth, 0, 2048)

        ax5.plot(t_plot, freq_smooth, color=colors['frequency'],
                 linewidth=1.2, alpha=0.9)

        # Mark ISCO
        if 'f_isco_hz' in params:
            ax5.axhline(y=params['f_isco_hz'], color='white',
                        linestyle=':', alpha=0.3, linewidth=0.8)
            ax5.text(t_plot[0] + 5, params['f_isco_hz'] + 5,
                     f"$f_{{ISCO}}$ = {params['f_isco_hz']:.1f} Hz",
                     color='white', fontsize=8, alpha=0.5)

    ax5.set_xlabel('Time [ms]', fontsize=11, color=colors['text'])
    ax5.set_ylabel('Frequency [Hz]', fontsize=11, color=colors['text'])
    ax5.set_title('Frequency Evolution',
                  fontsize=13, color=colors['accent'], fontweight='bold')
    ax5.grid(True, alpha=0.15, color=colors['grid'])
    ax5.set_ylim(0, min(500, params.get('f_ring_hz', 300) * 2))

    # ===================================================================
    # Panel 6: Phase evolution
    # ===================================================================
    ax6 = fig.add_subplot(gs[2, 2])
    ax6.set_facecolor(colors['panel_bg'])

    phase = waveform_result['phase']
    if np.any(mask_t):
        phase_plot = phase[mask_t]
        # Normalize phase: subtract linear trend for visibility
        phase_detrended = phase_plot - np.linspace(phase_plot[0], phase_plot[-1], len(phase_plot))

        ax6.plot(t_plot, phase_plot / (2 * PI), color=colors['phase'],
                 linewidth=1.0, alpha=0.9)

    ax6.set_xlabel('Time [ms]', fontsize=11, color=colors['text'])
    ax6.set_ylabel('Phase [cycles]', fontsize=11, color=colors['text'])
    ax6.set_title('Phase Evolution',
                  fontsize=13, color=colors['accent'], fontweight='bold')
    ax6.grid(True, alpha=0.15, color=colors['grid'])

    # ===================================================================
    # Panel 7: Parameter summary
    # ===================================================================
    ax7 = fig.add_subplot(gs[3, :])
    ax7.set_facecolor(colors['panel_bg'])
    ax7.axis('off')

    # Build parameter table
    param_lines = [
        ('Component Masses', f"m₁ = {params.get('total_mass_solar', 0) - params.get('symmetric_mass_ratio', 0):.1f} M☉,  m₂ = ... M☉"),
        ('Chirp Mass', f"Mc = {params.get('chirp_mass_solar', 0):.2f} M☉"),
        ('Mass Ratio η', f"η = {params.get('symmetric_mass_ratio', 0):.4f}"),
        ('Effective Spin', f"χ_eff = {params.get('effective_spin', 0):.3f}"),
        ('ISCO Frequency', f"f_ISCO = {params.get('f_isco_hz', 0):.1f} Hz"),
        ('Ringdown Freq', f"f_ring = {params.get('f_ring_hz', 0):.1f} Hz"),
        ('Damping Time', f"τ_QNM = {params.get('tau_qnm_s', 0)*1000:.2f} ms"),
        ('Final Mass', f"M_f = {params.get('final_mass_solar', 0):.1f} M☉"),
        ('Final Spin', f"a_f = {params.get('final_spin', 0):.3f}"),
        ('Energy Radiated', f"ΔE = {params.get('energy_radiated_solar', 0):.2f} M☉c²"),
        ('Distance', f"d_L = {params.get('distance_mpc', 0):.0f} Mpc"),
        ('Peak Strain', f"|h|_max = {params.get('peak_strain', 0):.2e}"),
    ]

    if detection_result is not None:
        param_lines.append(('Optimal SNR', f"ρ_opt = {detection_result.get('optimal_snr', 0):.1f}"))
        param_lines.append(('Recovered SNR', f"ρ_peak = {detection_result.get('peak_snr', 0):.1f}"))

    # Render as multi-column text
    n_cols = 4
    n_rows = (len(param_lines) + n_cols - 1) // n_cols

    for i, (label, value) in enumerate(param_lines):
        col = i // n_rows
        row = i % n_rows
        x = 0.02 + col * 0.25
        y = 0.85 - row * 0.22

        ax7.text(x, y, f"{label}:", transform=ax7.transAxes,
                 fontsize=10, color='#888888', fontweight='normal',
                 fontfamily='monospace')
        ax7.text(x + 0.13, y, value, transform=ax7.transAxes,
                 fontsize=10, color=colors['accent'], fontweight='bold',
                 fontfamily='monospace')

    ax7.set_title('Physical Parameters',
                  fontsize=13, color=colors['accent'], fontweight='bold',
                  loc='left', pad=10)

    # ===================================================================
    # Main title
    # ===================================================================
    fig.suptitle(
        'Gravitational Wave Chirp Analysis Dashboard',
        fontsize=20, color='white', fontweight='bold', y=0.98
    )

    # Subtitle with event info
    subtitle = (f"Binary Black Hole Merger  •  "
                f"M_total = {params.get('total_mass_solar', 0):.1f} M☉  •  "
                f"d = {params.get('distance_mpc', 0):.0f} Mpc  •  "
                f"IMRPhenomD Waveform")
    fig.text(0.5, 0.955, subtitle, ha='center', fontsize=12,
             color='#888888', style='italic')

    # --- Save ---
    fig.savefig(output_path, dpi=dpi, facecolor=colors['bg'],
                edgecolor='none', bbox_inches='tight')
    plt.close(fig)

    print(f"Dashboard saved to: {output_path}")
    return output_path
