# Recent Updates

The following changes were made since the last push:

## Backend Fixes
- Forced Uvicorn web server to serve the HTML file with `charset=utf-8` to fix browser encoding issues that were rendering buttons incorrectly (such as "A-q" or "Â-¶" appearing).
- Completely wiped and clean-installed `numpy`, `scipy`, `astropy`, and all other Python dependencies by running a forced re-installation against `requirements.txt`. This resolved hidden SyntaxErrors and IndentationErrors inside the site-packages that were breaking the gravitational wave physics calculation endpoints.

## Frontend Cleanup
- Removed the incorrectly decoded Unicode play button icon (`▶`) from the "Generate Waveform" button. It now cleanly displays "Generate Waveform".
- Cleaned up residual frontend testing scripts and intermediate CSS files that were no longer needed.

## Testing & Validation
- Added `test_live_server.py` and `test_comprehensive.py` to independently verify physics modules and the live server API without spinning up the UI.
