# Gravitational Wave Chirp Generator & Analyzer 🌌

![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688)
![Status](https://img.shields.io/badge/status-active-success)

A comprehensive, pure-Python pipeline designed to simulate, analyze, and visualize gravitational waves from Binary Black Hole (BBH) mergers. Built with a robust CLI engine and a fully interactive FastAPI + React web dashboard. 

The physics engine is validated against the **GW150914** publication values.

## ✨ Features

- **Waveform Simulation**: Generates full Inspiral-Merger-Ringdown (IMR) waveforms using the `IMRPhenomD` model.
- **Audio Generation**: Converts simulated strains into audible "chirp" `.wav` files.
- **Time-Frequency Analysis**: Computes Q-transform spectrograms overlayed with theoretical frequency tracks.
- **Matched Filtering Pipeline**: Injects signals into simulated detector noise and recovers them to compute optimal and recovered Signal-to-Noise Ratios (SNR).
- **Physics Estimations**: Calculates key parameters including chirp mass, final state mass/spin, radiated energy, and ringdown Quasinormal Mode (QNM) frequencies.
- **Multi-Detector Networks**: Simulates network responses across global detectors (LIGO Hanford/Livingston, Virgo, KAGRA) considering antenna patterns and time delays.
- **Interactive Dashboard**: A FastAPI-powered backend serving a sleek React frontend to drive simulations and explore parameter spaces visually.

## 🚀 Quick Start

### Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/Devan433/Gravitational-Wave-Chirp-Generator-Analyzer.git
cd Gravitational-Wave-Chirp-Generator-Analyzer
pip install -r requirements.txt
```

### Running the Web Dashboard

Start the FastAPI server:

```bash
uvicorn server:app --reload --port 8000
```
Then navigate to `http://localhost:8000` in your browser.

### Command Line Interface

You can run the full analysis pipeline directly from the terminal. 

```bash
# Run with GW150914 defaults
python main.py

# Simulate a heavier binary system
python main.py --m1 50 --m2 30 --distance 800

# Full specification with spins and sky location
python main.py --m1 36 --m2 29 --s1z 0.3 --s2z -0.1 --distance 410 \
               --ra 1.95 --dec -1.27 --snr 25
```
Results, including the 7-panel dashboard plot and `.wav` audio chirp, are saved in the `output/` directory.

## 📂 Project Structure

- `main.py`: The core CLI orchestrator.
- `server.py`: FastAPI application serving the REST API and static frontend.
- `gravitational_wave_analyzer/`: The core physics engine.
  - `physics/`: Waveform generation, parameter space exploration, ringdown physics.
  - `signal_processing/`: Audio generation, spectrograms, parameter estimation, and matched filtering.
  - `visualization/`: Dashboard plotting logic.
- `frontend/`: The HTML/JS/CSS source for the interactive web UI.
- `output/`: Generated outputs (audio, plots).

## 🔬 Validation

The underlying physics pipeline is continuously validated against the published results for GW150914 (Abbott et al., PRL 116, 061102). Running `main.py` with default parameters will print a verification table comparing generated values (chirp mass, final mass, energy radiated) against LIGO/Virgo's published constraints.
