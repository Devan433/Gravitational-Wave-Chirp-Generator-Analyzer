# Gravitational Wave Observatory

> **Live Demo → [chirp-ecfp.onrender.com](https://chirp-ecfp.onrender.com)**

A high-fidelity, pure-Python gravitational wave analysis suite that simulates, detects, and visualizes signals from binary black hole mergers — the same physics used by LIGO. Validated against the real GW150914 detection (Abbott et al., PRL 116, 061102, 2016).

---

## What It Does

Generate gravitational waveforms from binary black hole mergers using three physics models, detect injected signals in simulated LIGO noise, recover source parameters, and explore the full detector network response — all from a browser with no installation required.

**Three waveform models:**
- **IMRPhenomD** — full Inspiral–Merger–Ringdown, calibrated against numerical relativity (primary model)
- **TaylorF2** — analytic Stationary Phase Approximation to 3.5 post-Newtonian order (inspiral only)
- **TaylorT4 + Ringdown** — time-domain ODE integration stitched to quasi-normal mode ringdown

**Signal processing pipeline:**
- Matched filtering with noise-weighted SNR time series (optimal Neyman-Pearson statistic)
- Constant-Q transform spectrogram (same algorithm as LIGO's Omega pipeline)
- Two-stage grid-search parameter estimation to recover masses from an observed waveform
- Multi-detector network analysis with antenna patterns and time-delay triangulation (H1, L1, V1, K1)
- Audio sonification — hear the gravitational wave chirp as a WAV file

**Validated against real data:**
- Default parameters reproduce GW150914: chirp mass 28.3 M☉, final mass ~62 M☉, final spin ~0.67, f_QNM ~250 Hz
- Real LIGO strain data downloadable from GWOSC for direct comparison

---

## Live Demo

**[chirp-ecfp.onrender.com](https://chirp-ecfp.onrender.com)**

> First load may take ~30 seconds to spin up (free tier). After that it's fast.

Try this in order:
1. Click **Load GW150914 Preset** — loads the first ever detected gravitational wave parameters
2. Click **Generate Waveform** — runs the full IMRPhenomD pipeline
3. Click **Play** on the audio player — hear the actual chirp sound
4. Switch to **Model Comparison** tab — see why IMR models replaced PN approximants
5. Switch to **Detector Network** tab — see how H1, L1, Virgo, and KAGRA each respond

---

## Quick Start (Local)

```bash
git clone https://github.com/Devan433/Gravitational-Wave-Chirp-Generator-Analyzer.git
cd Gravitational-Wave-Chirp-Generator-Analyzer
pip install -r requirements.txt

# Run the web interface
uvicorn server:app --reload --port 8000
# Open http://localhost:8000
```

**Command line:**
```bash
# GW150914 parameters (default)
python main.py

# Heavier binary, farther away
python main.py --m1 50 --m2 30 --distance 800

# Full custom configuration
python main.py --m1 36 --m2 29 --s1z 0.3 --s2z -0.1 --distance 410 \
               --ra 1.95 --dec -1.27 --snr 25
```

Output plots, spectrograms, and audio files are saved to `output/`.

---

## Project Structure

```
├── server.py                          # FastAPI backend (REST API + static serving)
├── main.py                            # CLI entry point
├── gravitational_wave_analyzer/
│   ├── physics/
│   │   ├── merger.py                  # IMRPhenomD waveform model (1121 lines)
│   │   ├── taylorf2.py                # TaylorF2 SPA model
│   │   ├── waveform.py                # TaylorT4 time-domain model
│   │   ├── ringdown.py                # Quasi-normal mode frequencies
│   │   ├── full_waveform.py           # Unified interface for all three models
│   │   └── parameter_space.py         # Analytic parameter grid (no FFTs)
│   ├── signal_processing/
│   │   ├── matched_filter.py          # Neyman-Pearson optimal detection
│   │   ├── spectrogram.py             # Constant-Q transform
│   │   ├── audio.py                   # Chirp sonification → WAV
│   │   └── parameter_estimation.py    # Grid-search mass recovery
│   ├── data/
│   │   ├── ligo_sensitivity.py        # Advanced LIGO analytic PSD model
│   │   └── gwosc_connector.py         # Real LIGO data downloader (GWOSC)
│   └── visualization/
│       └── dashboard.py               # 7-panel matplotlib publication figure
├── frontend/                          # Vanilla JS + Plotly.js SPA
├── validate_real_detection.py         # GW150914 real-data reproduction pipeline
└── test_comprehensive.py              # 13-section test suite, ~120 assertions
```

---

## GW150914 Validation

| Parameter | Published (Abbott et al. 2016) | This Project |
|-----------|-------------------------------|--------------|
| Chirp mass | 28.3 ± 0.1 M☉ | 28.3 M☉ ✓ |
| Final mass | 62.3 ± 3.1 M☉ | ~62 M☉ ✓ |
| Final spin | 0.67 ± 0.05 | ~0.67 ✓ |
| Energy radiated | ~3.0 M☉c² | ~3 M☉ ✓ |
| ISCO frequency | ~67.6 Hz | 67.6 Hz ✓ |
| Ringdown frequency | ~250 Hz | ~250 Hz ✓ |
| Peak strain (410 Mpc) | ~10⁻²¹ | ~10⁻²¹ ✓ |

---

## Tech Stack

**Backend:** Python 3.11+, FastAPI, NumPy, SciPy, Matplotlib
**Frontend:** Vanilla JavaScript, Plotly.js
**Physics models:** IMRPhenomD (Husa et al. PRD 93, 2016), TaylorF2 (Arun et al. PRD 71, 2005), TaylorT4
**Deployment:** Render (live demo), Docker-compatible

Zero dependency on LALSuite or PyCBC — entirely standalone.

---

## References

- Abbott et al., *Observation of Gravitational Waves from a Binary Black Hole Merger*, PRL 116, 061102 (2016)
- Husa et al., *Frequency-domain gravitational waves from nonprecessing black-hole binaries*, PRD 93, 044006 (2016)
- Khan et al., *Frequency-domain gravitational waves from nonprecessing black-hole binaries II*, PRD 93, 044007 (2016)
- Arun et al., *The 3.5PN gravitational wave polarisations*, PRD 71, 084008 (2005)
- Allen et al., *FINDCHIRP: An algorithm for detection of gravitational waves*, PRD 85, 122006 (2012)
- Berti, Cardoso & Will, *Gravitational-wave spectroscopy of massive black holes*, PRD 73, 064030 (2006)
