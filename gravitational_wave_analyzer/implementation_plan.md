# Gravitational Wave Analyzer — Feature Expansion Plan

Implement 4 major features in order: Multiple Waveform Models → Parameter Estimation → Detector Comparison → Parameter Space Explorer.

## Proposed Changes

### Feature 1: Multiple Waveform Models (Comparison View)

The backend already supports two waveform generation methods in [full_waveform.py](file:///C:/Users/loq/Desktop/gravity/gravitational_wave_analyzer/physics/full_waveform.py#L88-L101): `imrphenomd` and `pn_ringdown`. We need to expose both and add a TaylorF2 (frequency-domain analytic) model, then let the frontend overlay them.

#### [MODIFY] [server.py](file:///C:/Users/loq/Desktop/gravity/server.py)
- Add `method` field to `AnalyzeRequest` (default: `"imrphenomd"`)
- Add new `POST /compare` endpoint that generates waveforms for **all available models** with the same parameters and returns them in a single response
- Each model result includes: `time`, `h_plus`, `frequency`, `params`, and a `model_name` label

#### [NEW] [taylorf2.py](file:///C:/Users/loq/Desktop/gravity/gravitational_wave_analyzer/physics/taylorf2.py)
- Implement the **TaylorF2** stationary-phase approximation waveform
- This is the simplest frequency-domain PN model: analytic closed-form phase Ψ(f) to 3.5PN
- Formula: h̃(f) = A f^{-7/6} exp(iΨ(f)), where Ψ(f) has known PN coefficients
- Convert to time domain via IFFT for overlay comparison
- Reference: Cutler & Flanagan PRD 49 (1994); Buonanno et al. PRD 80 (2009)

#### [MODIFY] [full_waveform.py](file:///C:/Users/loq/Desktop/gravity/gravitational_wave_analyzer/physics/full_waveform.py)
- Add `method='taylorf2'` branch that calls the new TaylorF2 generator
- Return the same dict structure (time, h_plus, h_cross, h_detector, frequency, phase, params)

#### [MODIFY] [index.html](file:///C:/Users/loq/Desktop/gravity/frontend/index.html)
- Add a "Model Comparison" toggle/checkbox group in the sidebar (IMRPhenomD ✓, TaylorT4, TaylorF2)
- When comparison mode is active, the waveform and frequency plots overlay all selected models with different colors and a legend
- Add a "Model Differences" summary card showing where PN models diverge from IMRPhenomD (phase difference, frequency at divergence)

---

### Feature 2: Parameter Estimation (Inverse Problem)

Build a grid-search parameter estimator: given an "observed" waveform, find the best-fit masses and spins by maximizing the match (normalized inner product) over a grid of templates.

#### [NEW] [parameter_estimation.py](file:///C:/Users/loq/Desktop/gravity/gravitational_wave_analyzer/signal_processing/parameter_estimation.py)
- `compute_match(h1, h2, sample_rate, f_lower)` — compute the normalized noise-weighted overlap between two waveforms (maximized over time and phase shifts), returns a value in [0, 1]
- `grid_search_pe(observed_waveform, m1_range, m2_range, spin_range, sample_rate, f_lower)` — generate templates on a coarse grid, compute match for each, return top matches sorted by match value
- Use a **two-stage** search:
  1. **Coarse grid** (step=2 M☉ for masses, step=0.2 for spins): ~200 templates, takes ~2-3 seconds
  2. **Fine grid** around the best coarse match (step=0.5 M☉, step=0.05 for spins): ~100 more templates
- Return: best-fit parameters, match value, grid of match values for visualization (heatmap)

#### [MODIFY] [server.py](file:///C:/Users/loq/Desktop/gravity/server.py)
- Add `POST /estimate` endpoint:
  - Input: `m1_true`, `m2_true`, `s1z_true`, `s2z_true` (the "hidden" true parameters), `noise_snr`, and search ranges
  - Generates the "observed" signal, injects into noise
  - Runs grid search PE
  - Returns: true params, estimated params, match matrix (for heatmap), top-N candidates
- Add `POST /estimate_status` for progress tracking (optional, may use streaming instead)

#### [MODIFY] [index.html](file:///C:/Users/loq/Desktop/gravity/frontend/index.html)
- Add a new **"Parameter Estimation"** tab/mode in the sidebar
- User sets "true" parameters (hidden from the estimator) and search ranges
- Click "Run PE" → backend searches → frontend displays:
  1. A heatmap of match values over (m1, m2) grid with the peak marked
  2. The estimated vs true parameters comparison table
  3. The best-fit template overlaid on the "observed" waveform

---

### Feature 3: Detector Comparison (Multi-Detector Network)

Add Virgo and KAGRA detector models with their own PSDs and antenna patterns, then show how the same signal appears differently at each detector.

#### [MODIFY] [ligo_sensitivity.py](file:///C:/Users/loq/Desktop/gravity/gravitational_wave_analyzer/data/ligo_sensitivity.py)
- Add `virgo_psd(f)` — Advanced Virgo design sensitivity curve (similar analytic fit, different parameters: f0=150 Hz, less sensitive at low frequencies)
- Add `kagra_psd(f)` — KAGRA design sensitivity curve (optimized for lower frequencies)
- Add `detector_psd(f, detector='H1')` dispatcher function

#### [MODIFY] [waveform.py](file:///C:/Users/loq/Desktop/gravity/gravitational_wave_analyzer/physics/waveform.py)
- Extend `antenna_pattern()` to support `detector='V1'` (Virgo) and `detector='K1'` (KAGRA)
- Add detector coordinates:
  - **Virgo (V1):** Lat 43.631° N, Long 10.505° E, arm azimuth 71.5°
  - **KAGRA (K1):** Lat 36.410° N, Long 137.306° E, arm azimuth 90°
- Add `time_delay(ra, dec, det1, det2)` — compute the light travel time between two detectors for a given sky position (used for triangulation)

#### [MODIFY] [server.py](file:///C:/Users/loq/Desktop/gravity/server.py)
- Add `POST /network` endpoint:
  - Input: binary parameters + sky position (ra, dec, psi)
  - Output: for each detector (H1, L1, V1, K1): antenna response (F+, F×), optimal SNR, projected strain h(t), arrival time offset
  - Also return: network SNR = √(Σ ρᵢ²)

#### [MODIFY] [index.html](file:///C:/Users/loq/Desktop/gravity/frontend/index.html)
- Add "Detector Network" section with:
  - Checkboxes for H1, L1, V1, K1
  - Sky position inputs (RA, Dec in degrees)
  - Results table: detector name, F+, F×, optimal SNR, arrival time delay
  - Overlaid strain plots showing how the signal looks at each detector
  - Network SNR display

---

### Feature 4: Parameter Space Explorer

Visualize how key observables (duration, peak frequency, chirp mass, SNR) change across the (m1, m2) parameter space as interactive heatmaps.

#### [NEW] [parameter_space.py](file:///C:/Users/loq/Desktop/gravity/gravitational_wave_analyzer/physics/parameter_space.py)
- `compute_parameter_grid(m1_range, m2_range, distance_mpc, quantities)` — for each (m1, m2) point:
  - Compute analytically (no full waveform generation needed for most):
    - Chirp mass: `(m1*m2)^(3/5) / (m1+m2)^(1/5)`
    - Symmetric mass ratio: `m1*m2 / (m1+m2)^2`
    - ISCO frequency: `c^3 / (6^(3/2) π G M_total)`
    - Leading-order inspiral duration: `(5/256) * (π f_lower)^(-8/3) * M_chirp^(-5/3)` (in geometric units)
    - Peak strain (leading order)
    - Final mass and final spin (from fitting formulas)
  - Return as 2D arrays for heatmap plotting

#### [MODIFY] [server.py](file:///C:/Users/loq/Desktop/gravity/server.py)
- Add `POST /parameter_space` endpoint:
  - Input: `m1_min`, `m1_max`, `m2_min`, `m2_max`, `grid_size` (default 30), `distance_mpc`
  - Output: 2D grids for each quantity + axis labels
  - This is very fast (pure analytic, no FFTs) — should return in <100ms

#### [MODIFY] [index.html](file:///C:/Users/loq/Desktop/gravity/frontend/index.html)
- Add "Parameter Space" tab/panel with:
  - Mass range sliders (m1: 5–100, m2: 5–100)
  - Dropdown to select which quantity to visualize
  - Plotly heatmap with (m1, m2) axes and color = selected quantity
  - Crosshair or marker showing current selected parameters
  - Clicking on the heatmap sets the m1/m2 sliders and can trigger a full analysis

---

## User Review Required

> [!IMPORTANT]
> **UI Layout Decision:** The current frontend is a single-page app with a left sidebar + 2×2 chart grid. Adding 4 new features will require a navigation system. I propose adding a **tabbed interface** at the top of the main content area:
> - **Tab 1: Analysis** (current default — waveform, spectrogram, SNR, frequency plots)
> - **Tab 2: Model Comparison** (Feature 1 — overlay different waveform models)
> - **Tab 3: Parameter Estimation** (Feature 2 — grid search PE with heatmap)
> - **Tab 4: Detector Network** (Feature 3 — multi-detector comparison)
> - **Tab 5: Parameter Space** (Feature 4 — parameter space heatmaps)
>
> The left sidebar parameters panel stays shared across all tabs. Is this layout acceptable?

> [!WARNING]
> **TaylorF2 Limitations:** The TaylorF2 model is only valid for the inspiral phase (up to ~ISCO frequency). It will visibly diverge from IMRPhenomD near merger. This is actually a **feature**, not a bug — it demonstrates exactly why full IMR models were needed. The comparison plot will show where the PN approximation breaks down.

> [!WARNING]
> **Parameter Estimation Runtime:** The grid-search PE with ~300 templates will take approximately 3-5 seconds on a typical laptop. This is acceptable for a demo but I'll add a progress indicator. The alternative (MCMC/nested sampling) would be much slower and harder to visualize.

## Verification Plan

### Automated Tests
- Run `python -m pytest` after implementation (if tests exist)
- Run `python -c "from gravitational_wave_analyzer.physics.taylorf2 import generate_taylorf2_waveform; print('TaylorF2 OK')"` to verify import
- Start the server and test each new endpoint with `curl`:
  ```bash
  curl -X POST http://localhost:8000/compare -H "Content-Type: application/json" -d '{"m1":36,"m2":29}'
  curl -X POST http://localhost:8000/estimate -H "Content-Type: application/json" -d '{"m1_true":36,"m2_true":29}'
  curl -X POST http://localhost:8000/network -H "Content-Type: application/json" -d '{"m1":36,"m2":29,"ra":1.95,"dec":-1.27}'
  curl -X POST http://localhost:8000/parameter_space -H "Content-Type: application/json" -d '{"m1_min":10,"m1_max":80,"m2_min":10,"m2_max":80}'
  ```

### Manual Verification
- Open the web UI and test each tab visually
- Verify TaylorF2 vs IMRPhenomD diverge near merger (expected behavior)
- Verify PE grid search recovers the true parameters within 1–2 grid steps
- Verify Virgo/KAGRA SNRs are lower than LIGO (expected — less sensitive detectors)
- Verify parameter space heatmaps show physically correct trends (higher mass → lower ISCO frequency, shorter duration)
