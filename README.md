# Gravitational Wave Chirp Generator & Analyzer

A Python project for simulating and analyzing gravitational waves from binary black hole mergers. It generates inspiral–merger–ringdown waveforms, produces audible chirps and spectrograms, performs matched filtering, and models detector responses for LIGO, Virgo, and KAGRA.

The project includes both a command-line interface and a FastAPI + React web interface.

## Features
* Generate gravitational waveforms using the IMRPhenomD model
* Create audible chirp (.wav) files from simulated strain data
* Generate spectrograms and frequency evolution plots
* Inject signals into detector noise and recover them using matched filtering
* Estimate source parameters such as chirp mass, final mass, final spin, and radiated energy
* Simulate responses from multiple detectors (LIGO Hanford, LIGO Livingston, Virgo, and KAGRA)
* Explore parameter variations through a web dashboard built with FastAPI and React
* Compare simulated GW150914 results with published values

## Quick Start

### Installation
```bash
git clone https://github.com/Devan433/Gravitational-Wave-Chirp-Generator-Analyzer.git
cd Gravitational-Wave-Chirp-Generator-Analyzer
pip install -r requirements.txt
```

### Run the Web Interface
```bash
uvicorn server:app --reload --port 8000
```
Open:
http://localhost:8000

### Run from the Command Line
```bash
# GW150914 parameters
python main.py

# Heavier binary
python main.py --m1 50 --m2 30 --distance 800

# Custom configuration
python main.py --m1 36 --m2 29 --s1z 0.3 --s2z -0.1 --distance 410 \
               --ra 1.95 --dec -1.27 --snr 25
```

Generated plots, spectrograms, and audio files are saved in the `output/` directory.

## Project Structure
```text
main.py                  Command-line entry point
server.py                FastAPI backend
physics/                 Waveform and merger calculations
signal_processing/       Filtering, spectrograms, audio generation
visualization/           Plot generation
frontend/                React frontend
output/                  Generated results
```

## Validation

Default parameters correspond approximately to the GW150914 binary black hole merger. Simulated quantities such as chirp mass, remnant mass, and radiated energy can be compared with values reported by the LIGO Scientific Collaboration and Virgo Collaboration publications.
