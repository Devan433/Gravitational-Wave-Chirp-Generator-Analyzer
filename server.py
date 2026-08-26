# FastAPI Backend for Gravitational Wave Analyzer
#
# Wraps the existing pure-Python physics pipeline and exposes it as a
# JSON API for the interactive web frontend.
#
# Endpoints:
#   GET  /                serves the React frontend (index.html)
#   POST /analyze         runs the full analysis pipeline, returns JSON
#
# Usage:
#   uvicorn server:app --reload --port 8000

import sys
import os
import base64
import time
import numpy as np

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, List

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

app = FastAPI(title="Gravitational Wave Analyzer API")

# CORS — allow the frontend to call the API from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend static files
app.mount("/static", StaticFiles(directory="frontend"), name="static")


@app.get("/")
async def root():
    """Serve the main frontend page."""
    return FileResponse("frontend/index.html", media_type="text/html; charset=utf-8")


class AnalyzeRequest(BaseModel):
    m1: float = 36.0
    m2: float = 29.0
    s1z: float = 0.0
    s2z: float = 0.0
    distance: float = 410.0
    inclination: float = 0.0
    snr: float = 25.0
    method: str = "imrphenomd"


def downsample_array(arr, max_points=2000):
    """Downsample an array to max_points for efficient JSON transfer."""
    if len(arr) <= max_points:
        return arr.tolist()
    step = len(arr) // max_points
    return arr[::step].tolist()


@app.post("/analyze")
async def analyze(req: AnalyzeRequest):
    """Run the complete gravitational wave analysis pipeline.

    Calls the same physics code as main.py, packages all results
    as JSON for the frontend to render with Plotly.
    """
    t_start = time.time()

    try:
        # --- Step 1: Generate waveform ---
        from gravitational_wave_analyzer.physics.full_waveform import (
            generate_full_waveform,
        )

        waveform = generate_full_waveform(
            m1_solar=req.m1,
            m2_solar=req.m2,
            s1z=req.s1z,
            s2z=req.s2z,
            distance_mpc=req.distance,
            inclination=req.inclination,
            f_lower=20.0,
            sample_rate=4096,
            method=req.method,
        )

        # --- Step 3: Q-transform spectrogram ---
        from gravitational_wave_analyzer.signal_processing.spectrogram import (
            q_transform,
            theoretical_frequency_track,
        )

        spec = q_transform(
            waveform["h_detector"],
            4096,
            q_range=(4, 64),
            f_range=(20, 1024),
            n_freq_bins=100,
            tres=0.002,
        )

        # Theoretical chirp track
        merger_idx = waveform["params"].get(
            "merger_index", len(waveform["time"]) // 2
        )
        t_track = spec["times"] + waveform["time"][0]
        freq_track = theoretical_frequency_track(
            req.m1, req.m2, 0.0, t_track, 20.0
        )

        # --- Step 4: Matched filtering ---
        from gravitational_wave_analyzer.signal_processing.matched_filter import (
            run_detection_pipeline,
        )

        det = run_detection_pipeline(
            waveform, target_snr=req.snr, noise_duration=16.0,
            sample_rate=4096, f_lower=20.0, seed=42,
        )

        # --- Step 5: Compute ringdown params ---
        from gravitational_wave_analyzer.physics.ringdown import (
            compute_ringdown_params,
        )

        rd = compute_ringdown_params(req.m1, req.m2, req.s1z, req.s2z)

        elapsed = time.time() - t_start

        # --- Package params ---
        params = waveform["params"]
        # Use waveform params for final state (IMRPhenomD formula)
        # Use ringdown params only for QNM frequencies
        e_rad_solar = float(params.get("total_mass_solar", 0)) * float(params.get("energy_radiated_fraction", 0))

        physics = {
            "total_mass": float(params.get("total_mass_solar", 0)),
            "chirp_mass": float(params.get("chirp_mass_solar", 0)),
            "eta": float(params.get("symmetric_mass_ratio", 0)),
            "chi_eff": float(params.get("effective_spin", 0)),
            "f_start_hz": float(params.get("f_start_hz", 20)),
            "f_isco_hz": float(params.get("f_isco_hz", 0)),
            "f_ring_hz": float(params.get("f_ring_hz", 0)),
            "f_damp_hz": float(params.get("f_damp_hz", 0)),
            "final_mass": float(params.get("final_mass_solar", 0)),
            "final_spin": float(params.get("final_spin", 0)),
            "energy_radiated": float(rd.get("energy_radiated_solar", 0)),
            "f_qnm_hz": float(rd.get("f_qnm_hz", 0)),
            "tau_qnm_s": float(rd.get("tau_qnm_s", 0)),
            "quality_factor": float(rd.get("quality_factor", 0)),
            "peak_strain": float(params.get("peak_strain", 0)),
            "duration": float(params.get("duration_seconds", 0)),
            "distance_mpc": req.distance,
            "optimal_snr": float(det.get("optimal_snr", 0)),
            "peak_snr": float(det.get("peak_snr", 0)),
            "target_snr": req.snr,
        }

        # --- Build response ---
        # Downsample large arrays for fast JSON transfer
        max_wf_pts = 3000
        max_snr_pts = 2000

        wf_time = downsample_array(waveform["time"] * 1000, max_wf_pts)  # ms
        wf_hplus = downsample_array(waveform["h_plus"], max_wf_pts)
        wf_hcross = downsample_array(waveform["h_cross"], max_wf_pts)
        wf_freq = downsample_array(waveform["frequency"], max_wf_pts)

        # SNR timeseries — center around peak
        snr_ts = det["snr_timeseries"]
        snr_t = det["time"]
        peak_t = det["peak_time"]
        snr_time_ms = downsample_array((snr_t - peak_t) * 1000, max_snr_pts)
        snr_vals = downsample_array(snr_ts, max_snr_pts)

        # Spectrogram — convert to lists, offset time to match waveform
        spec_times = (spec["times"] + waveform["time"][0]) * 1000  # ms
        spec_freqs = spec["frequencies"]
        spec_energy = spec["energy"]

        # Downsample spectrogram time axis if needed
        max_spec_t = 500
        if len(spec_times) > max_spec_t:
            step = len(spec_times) // max_spec_t
            spec_times = spec_times[::step]
            spec_energy = spec_energy[:, ::step]

        # Frequency track for overlay
        ft = freq_track.copy()
        ft_times = (t_track * 1000)
        if len(ft_times) > max_spec_t:
            step = len(ft_times) // max_spec_t
            ft_times = ft_times[::step]
            ft = ft[::step]

        response = {
            "status": "success",
            "elapsed": round(elapsed, 2),
            "physics": physics,
            "waveform": {
                "time": wf_time,
                "h_plus": wf_hplus,
                "h_cross": wf_hcross,
                "frequency": wf_freq,
            },
            "spectrogram": {
                "times": spec_times.tolist(),
                "frequencies": spec_freqs.tolist(),
                "energy": spec_energy.tolist(),
            },
            "freq_track": {
                "times": ft_times.tolist() if isinstance(ft_times, np.ndarray) else ft_times,
                "freqs": [float(x) if not np.isnan(x) else None for x in ft],
            },
            "snr": {
                "time": snr_time_ms,
                "values": snr_vals,
                "peak_snr": det["peak_snr"],
                "threshold": 8.0,
            },
        }

        return response

    except Exception as e:
        import traceback
        return {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc(),
        }

