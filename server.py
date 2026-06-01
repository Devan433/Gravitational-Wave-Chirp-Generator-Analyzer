# FastAPI Backend for Gravitational Wave Analyzer
# =================================================
#
# Wraps the existing pure-Python physics pipeline and exposes it as a
# JSON API for the interactive web frontend.
#
# Endpoints:
#   GET  /                serves the React frontend (index.html)
#   POST /analyze         runs the full analysis pipeline, returns JSON
#   POST /compare         compare multiple waveform models side-by-side
#   POST /estimate        parameter estimation via grid-search
#   POST /network         multi-detector network analysis
#   POST /parameter_space parameter space explorer heatmaps
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
    return FileResponse("frontend/index.html")


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

        # --- Step 2: Generate audio ---
        from gravitational_wave_analyzer.signal_processing.audio import (
            generate_chirp_audio,
        )

        os.makedirs("output", exist_ok=True)
        wav_path = os.path.join("output", "gw_chirp.wav")
        audio_result = generate_chirp_audio(waveform, output_path=wav_path)

        # Read WAV as base64 for browser playback
        wav_b64 = ""
        if os.path.exists(wav_path):
            with open(wav_path, "rb") as f:
                wav_b64 = base64.b64encode(f.read()).decode("ascii")

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
            "audio_base64": wav_b64,
            "audio_duration": float(audio_result.get("duration", 0)),
        }

        return response

    except Exception as e:
        import traceback
        return {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


# ============================================================================
# Model Comparison Endpoint
# ============================================================================

class CompareRequest(BaseModel):
    m1: float = 36.0
    m2: float = 29.0
    s1z: float = 0.0
    s2z: float = 0.0
    distance: float = 410.0
    inclination: float = 0.0
    models: List[str] = ["imrphenomd", "taylorf2"]


@app.post("/compare")
async def compare(req: CompareRequest):
    """Generate waveforms with multiple models for side-by-side comparison."""
    t_start = time.time()
    try:
        from gravitational_wave_analyzer.physics.full_waveform import (
            generate_full_waveform,
        )

        max_wf_pts = 3000
        results = {}

        for model_name in req.models:
            try:
                wf = generate_full_waveform(
                    m1_solar=req.m1, m2_solar=req.m2,
                    s1z=req.s1z, s2z=req.s2z,
                    distance_mpc=req.distance,
                    inclination=req.inclination,
                    f_lower=20.0, sample_rate=4096,
                    method=model_name,
                )

                results[model_name] = {
                    "time": downsample_array(wf["time"] * 1000, max_wf_pts),
                    "h_plus": downsample_array(wf["h_plus"], max_wf_pts),
                    "frequency": downsample_array(wf["frequency"], max_wf_pts),
                    "params": {
                        k: float(v) if isinstance(v, (int, float, np.floating, np.integer)) else str(v)
                        for k, v in wf.get("params", {}).items()
                    },
                    "status": "ok",
                }
            except Exception as e:
                results[model_name] = {
                    "status": "error",
                    "error": str(e),
                }

        return {
            "status": "success",
            "elapsed": round(time.time() - t_start, 2),
            "models": results,
        }

    except Exception as e:
        import traceback
        return {"status": "error", "error": str(e), "traceback": traceback.format_exc()}


# ============================================================================
# Parameter Estimation Endpoint
# ============================================================================

class EstimateRequest(BaseModel):
    m1_true: float = 36.0
    m2_true: float = 29.0
    s1z_true: float = 0.0
    s2z_true: float = 0.0
    distance: float = 410.0
    m1_min: float = 15.0
    m1_max: float = 60.0
    m2_min: float = 15.0
    m2_max: float = 60.0
    coarse_step: float = 3.0
    fine_step: float = 0.5


@app.post("/estimate")
async def estimate(req: EstimateRequest):
    """Run parameter estimation: generate observed waveform, then grid-search."""
    t_start = time.time()
    try:
        from gravitational_wave_analyzer.physics.full_waveform import (
            generate_full_waveform,
        )
        from gravitational_wave_analyzer.signal_processing.parameter_estimation import (
            grid_search_pe,
        )

        # Generate the "true" observed waveform
        observed = generate_full_waveform(
            m1_solar=req.m1_true, m2_solar=req.m2_true,
            s1z=req.s1z_true, s2z=req.s2z_true,
            distance_mpc=req.distance,
            f_lower=20.0, sample_rate=4096,
            method='imrphenomd',
        )

        # Run grid search PE
        pe_result = grid_search_pe(
            observed_h=observed['h_detector'],
            sample_rate=4096,
            f_lower=20.0,
            m1_range=(req.m1_min, req.m1_max),
            m2_range=(req.m2_min, req.m2_max),
            distance_mpc=req.distance,
            coarse_step=req.coarse_step,
            fine_step=req.fine_step,
        )

        # Generate best-fit waveform for overlay
        best_wf = generate_full_waveform(
            m1_solar=pe_result['best_m1'],
            m2_solar=pe_result['best_m2'],
            distance_mpc=req.distance,
            f_lower=20.0, sample_rate=4096,
            method='imrphenomd',
        )

        max_wf_pts = 2000

        return {
            "status": "success",
            "elapsed": round(time.time() - t_start, 2),
            "true_params": {
                "m1": req.m1_true, "m2": req.m2_true,
                "s1z": req.s1z_true, "s2z": req.s2z_true,
            },
            "estimated_params": {
                "m1": pe_result['best_m1'],
                "m2": pe_result['best_m2'],
                "match": pe_result['best_match'],
            },
            "coarse_grid": pe_result['coarse_grid'],
            "fine_grid": pe_result['fine_grid'],
            "top_candidates": pe_result['all_candidates'],
            "n_templates": pe_result['n_templates'],
            "observed_waveform": {
                "time": downsample_array(observed['time'] * 1000, max_wf_pts),
                "h_plus": downsample_array(observed['h_plus'], max_wf_pts),
            },
            "bestfit_waveform": {
                "time": downsample_array(best_wf['time'] * 1000, max_wf_pts),
                "h_plus": downsample_array(best_wf['h_plus'], max_wf_pts),
            },
        }

    except Exception as e:
        import traceback
        return {"status": "error", "error": str(e), "traceback": traceback.format_exc()}


# ============================================================================
# Multi-Detector Network Endpoint
# ============================================================================

class NetworkRequest(BaseModel):
    m1: float = 36.0
    m2: float = 29.0
    s1z: float = 0.0
    s2z: float = 0.0
    distance: float = 410.0
    inclination: float = 0.0
    ra: float = 1.95
    dec: float = -1.27
    psi: float = 0.82
    detectors: List[str] = ["H1", "L1", "V1", "K1"]


# Detector locations (lat, lon in radians, arm azimuth in radians)
DETECTOR_INFO = {
    'H1': {
        'name': 'LIGO Hanford',
        'lat': np.radians(46.455),
        'lon': np.radians(-119.408),
        'arm_azimuth': np.radians(126.0),
        'arm_length_km': 4.0,
    },
    'L1': {
        'name': 'LIGO Livingston',
        'lat': np.radians(30.563),
        'lon': np.radians(-90.774),
        'arm_azimuth': np.radians(108.0),
        'arm_length_km': 4.0,
    },
    'V1': {
        'name': 'Virgo',
        'lat': np.radians(43.631),
        'lon': np.radians(10.505),
        'arm_azimuth': np.radians(71.5),
        'arm_length_km': 3.0,
    },
    'K1': {
        'name': 'KAGRA',
        'lat': np.radians(36.410),
        'lon': np.radians(137.306),
        'arm_azimuth': np.radians(90.0),
        'arm_length_km': 3.0,
    },
}

EARTH_RADIUS_M = 6.371e6  # meters


def detector_position_ecef(lat, lon):
    """Get Earth-centered Earth-fixed position of a detector."""
    x = EARTH_RADIUS_M * np.cos(lat) * np.cos(lon)
    y = EARTH_RADIUS_M * np.cos(lat) * np.sin(lon)
    z = EARTH_RADIUS_M * np.sin(lat)
    return np.array([x, y, z])


def gw_source_direction(ra, dec):
    """Convert (RA, Dec) to a unit vector (ECEF, simplified)."""
    x = np.cos(dec) * np.cos(ra)
    y = np.cos(dec) * np.sin(ra)
    z = np.sin(dec)
    return np.array([x, y, z])


def time_delay_between_detectors(ra, dec, det1_key, det2_key):
    """Compute the light travel time delay between two detectors."""
    d1 = DETECTOR_INFO[det1_key]
    d2 = DETECTOR_INFO[det2_key]
    pos1 = detector_position_ecef(d1['lat'], d1['lon'])
    pos2 = detector_position_ecef(d2['lat'], d2['lon'])
    n_hat = gw_source_direction(ra, dec)
    # Time delay = (pos2 - pos1) · n_hat / c
    c = 299792458.0  # m/s
    delay = np.dot(pos2 - pos1, n_hat) / c
    return float(delay)


def antenna_pattern_for_detector(ra, dec, psi, det_key):
    """Compute antenna pattern for a given detector."""
    det = DETECTOR_INFO[det_key]
    # Simplified antenna pattern using detector coordinates
    theta = np.pi / 2.0 - dec
    phi = ra - det['lon']  # approximate hour angle

    cos_theta = np.cos(theta)
    cos_2phi = np.cos(2.0 * phi)
    sin_2phi = np.sin(2.0 * phi)
    cos_2psi = np.cos(2.0 * psi)
    sin_2psi = np.sin(2.0 * psi)

    F_plus = (0.5 * (1.0 + cos_theta**2) * cos_2phi * cos_2psi
              - cos_theta * sin_2phi * sin_2psi)
    F_cross = (0.5 * (1.0 + cos_theta**2) * cos_2phi * sin_2psi
               + cos_theta * sin_2phi * cos_2psi)

    return float(F_plus), float(F_cross)


@app.post("/network")
async def network(req: NetworkRequest):
    """Compute multi-detector network response."""
    t_start = time.time()
    try:
        from gravitational_wave_analyzer.physics.full_waveform import (
            generate_full_waveform,
        )
        from gravitational_wave_analyzer.signal_processing.matched_filter import (
            compute_optimal_snr,
        )

        # Generate base waveform (face-on, no antenna pattern)
        wf = generate_full_waveform(
            m1_solar=req.m1, m2_solar=req.m2,
            s1z=req.s1z, s2z=req.s2z,
            distance_mpc=req.distance,
            inclination=req.inclination,
            f_lower=20.0, sample_rate=4096,
        )

        max_wf_pts = 2000
        detector_results = {}
        snr_sum_sq = 0.0
        ref_det = req.detectors[0] if req.detectors else 'H1'

        for det_key in req.detectors:
            if det_key not in DETECTOR_INFO:
                continue

            det_info = DETECTOR_INFO[det_key]
            F_plus, F_cross = antenna_pattern_for_detector(
                req.ra, req.dec, req.psi, det_key
            )

            # Project strain onto this detector
            h_det = F_plus * wf['h_plus'] + F_cross * wf['h_cross']

            # Compute optimal SNR for this detector
            opt_snr = compute_optimal_snr(h_det, 4096, 20.0)
            snr_sum_sq += opt_snr**2

            # Time delay relative to first detector
            if det_key == ref_det:
                delay_ms = 0.0
            else:
                delay_ms = time_delay_between_detectors(
                    req.ra, req.dec, ref_det, det_key
                ) * 1000  # convert to ms

            detector_results[det_key] = {
                "name": det_info['name'],
                "F_plus": round(F_plus, 4),
                "F_cross": round(F_cross, 4),
                "response": round(np.sqrt(F_plus**2 + F_cross**2), 4),
                "optimal_snr": round(opt_snr, 1),
                "arrival_delay_ms": round(delay_ms, 3),
                "arm_length_km": det_info['arm_length_km'],
                "strain": {
                    "time": downsample_array(wf['time'] * 1000, max_wf_pts),
                    "h": downsample_array(h_det, max_wf_pts),
                },
            }

        network_snr = np.sqrt(snr_sum_sq)

        return {
            "status": "success",
            "elapsed": round(time.time() - t_start, 2),
            "detectors": detector_results,
            "network_snr": round(float(network_snr), 1),
            "sky_position": {
                "ra_rad": req.ra,
                "dec_rad": req.dec,
                "ra_deg": round(np.degrees(req.ra), 2),
                "dec_deg": round(np.degrees(req.dec), 2),
            },
        }

    except Exception as e:
        import traceback
        return {"status": "error", "error": str(e), "traceback": traceback.format_exc()}


# ============================================================================
# Parameter Space Explorer Endpoint
# ============================================================================

class ParameterSpaceRequest(BaseModel):
    m1_min: float = 10.0
    m1_max: float = 80.0
    m2_min: float = 10.0
    m2_max: float = 80.0
    grid_size: int = 35
    distance: float = 410.0
    s1z: float = 0.0
    s2z: float = 0.0
    quantity: str = "chirp_mass"


@app.post("/parameter_space")
async def parameter_space(req: ParameterSpaceRequest):
    """Compute parameter space heatmaps."""
    t_start = time.time()
    try:
        from gravitational_wave_analyzer.physics.parameter_space import (
            compute_parameter_grid,
        )

        grid = compute_parameter_grid(
            m1_range=(req.m1_min, req.m1_max),
            m2_range=(req.m2_min, req.m2_max),
            grid_size=req.grid_size,
            distance_mpc=req.distance,
            f_lower=20.0,
            s1z=req.s1z, s2z=req.s2z,
        )

        return {
            "status": "success",
            "elapsed": round(time.time() - t_start, 3),
            "grid": grid,
        }

    except Exception as e:
        import traceback
        return {"status": "error", "error": str(e), "traceback": traceback.format_exc()}
