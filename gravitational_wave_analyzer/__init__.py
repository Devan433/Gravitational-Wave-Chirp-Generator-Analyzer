"""
Gravitational Wave Chirp Generator & Analyzer
==============================================

A production-grade gravitational wave analysis pipeline implementing:
- 3.5 Post-Newtonian inspiral waveforms
- IMRPhenomD phenomenological merger model
- Quasinormal mode ringdown
- Matched filtering detection (identical to LIGO pipeline)
- Q-transform spectrograms
- Audible chirp audio generation

All physics implemented from first principles using numpy/scipy.
Validated against GW150914 published results.

Author: Gravitational Wave Analysis Project
License: MIT
"""

__version__ = "1.0.0"
