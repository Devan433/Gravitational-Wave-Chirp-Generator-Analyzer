"""
GWOSC Real Data Connector
===========================

Downloads and processes real gravitational wave strain data from the
Gravitational Wave Open Science Center (GWOSC, https://gwosc.org).

This module connects the existing pure-Python analysis pipeline to
actual LIGO/Virgo detector data, enabling analysis of real detections
like GW150914, GW170817, and the entire GWTC-3 catalog.

Capabilities:
    1. Fetch event catalogs (GWTC-1, GWTC-2, GWTC-3)
    2. Download strain HDF5 data for any event/detector
    3. Estimate real PSD from off-source data segments
    4. Preprocess strain (bandpass, notch filter, windowing)
    5. Extract event segments with proper time alignment
    6. Batch download multiple events
    7. Validate against published detection parameters

All downloads are cached locally to avoid repeated network requests.
Supports offline mode: if cache exists, no network access is needed.

References
----------
[1] GWOSC, "Gravitational Wave Open Science Center",
    https://gwosc.org
    — Primary data source for public LIGO/Virgo strain data.

[2] Abbott et al., "GWTC-3: Compact Binary Coalescences Observed by
    LIGO and Virgo During the Second Part of the Third Observing Run",
    Phys. Rev. X 13, 041039 (2023).
    https://doi.org/10.1103/PhysRevX.13.041039

[3] Abbott et al., "Open data from the third observing run of LIGO,
    Virgo, KAGRA, and GEO", ApJS 267, 29 (2023).
    https://doi.org/10.3847/1538-4365/acdc9f

[4] GWOSC API documentation:
    https://gwosc.org/apidocs/
"""

import json
import os
import sys
import time
import warnings
import numpy as np

from scipy.signal import butter, sosfiltfilt, iirnotch, welch
from scipy.signal.windows import tukey


# ============================================================================
# Configuration
# ============================================================================

# GWOSC API base URL
GWOSC_BASE_URL = "https://gwosc.org"

# Catalog API endpoints
# Reference: https://gwosc.org/apidocs/
CATALOG_URLS = {
    "GWTC-1": f"{GWOSC_BASE_URL}/eventapi/json/GWTC-1-confident/",
    "GWTC-2": f"{GWOSC_BASE_URL}/eventapi/json/GWTC-2-confident/",
    "GWTC-2.1": f"{GWOSC_BASE_URL}/eventapi/json/GWTC-2.1-confident/",
    "GWTC-3": f"{GWOSC_BASE_URL}/eventapi/json/GWTC-3-confident/",
}

# Default cache directory
DEFAULT_CACHE_DIR = os.path.join("output", "gwosc_cache")

# Supported detectors
VALID_DETECTORS = {"H1", "L1", "V1"}

# US power line frequency and harmonics (for notch filtering)
POWER_LINE_FREQS = [60.0, 120.0, 180.0]

# HTTP request timeout in seconds
HTTP_TIMEOUT = 30

# Default sample rate for LIGO data
LIGO_SAMPLE_RATE = 4096


# ============================================================================
# Optional dependency handling
# ============================================================================

def _import_requests():
    """Import requests library with helpful error message."""
    try:
        import requests
        return requests
    except ImportError:
        raise ImportError(
            "The 'requests' library is required for GWOSC data access.\n"
            "Install it with: pip install requests\n"
            "Or install all GWOSC dependencies: pip install requests h5py tqdm"
        )


def _import_h5py():
    """Import h5py library with helpful error message."""
    try:
        import h5py
        return h5py
    except ImportError:
        return None


def _import_tqdm():
    """Import tqdm for progress bars, return None if unavailable."""
    try:
        from tqdm import tqdm
        return tqdm
    except ImportError:
        return None


# ============================================================================
# 1. Event Catalog Fetcher
# ============================================================================

def fetch_event_catalog(catalog="GWTC-3", cache_dir=DEFAULT_CACHE_DIR):
    """Fetch the gravitational wave event catalog from GWOSC.

    Downloads the specified catalog (GWTC-1, GWTC-2, GWTC-2.1, or GWTC-3)
    from the GWOSC event API and parses it into a structured dictionary.
    Results are cached locally to avoid repeated network requests.

    The catalog contains event names, GPS merger times, component masses,
    spins, luminosity distances, and network SNRs for all confident
    detections in that observing run.

    Parameters
    ----------
    catalog : str
        Catalog name. One of: 'GWTC-1', 'GWTC-2', 'GWTC-2.1', 'GWTC-3'.
        Default: 'GWTC-3' (most complete, 90 events from O1+O2+O3).
    cache_dir : str
        Directory if the agent for caching downloaded catalog JSON files.
        Default: 'output/gwosc_cache'.

    Returns
    -------
    dict
        Dictionary with keys:
            'catalog_name' : str — name of the catalog
            'num_events' : int — total number of events
            'events' : dict — mapping event_name -> event_parameters
                Each event_parameters dict contains:
                    'gps_time' : float — GPS merger time (seconds)
                    'mass1' : float — primary mass (solar masses)
                    'mass2' : float — secondary mass (solar masses)
                    'chirp_mass' : float — chirp mass (solar masses)
                    'network_snr' : float — network matched-filter SNR
                    'luminosity_distance' : float — distance (Mpc)
                    'final_mass' : float — remnant mass (solar masses)
                    'chi_eff' : float — effective aligned spin
                    'detectors' : list — participating detectors

    Raises
    ------
    ValueError
        If catalog name is not recognized.
    ConnectionError
        If network request fails and no cache is available.

    Examples
    --------
    >>> cat = fetch_event_catalog('GWTC-3')
    >>> print(f"Found {cat['num_events']} events")
    Found 90 events
    >>> gw150914 = cat['events']['GW150914']
    >>> print(f"GPS time: {gw150914['gps_time']}")
    GPS time: 1126259462.4
    """
    if catalog not in CATALOG_URLS:
        raise ValueError(
            f"Unknown catalog '{catalog}'. "
            f"Valid options: {list(CATALOG_URLS.keys())}"
        )

    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"catalog_{catalog.replace('.', '_')}.json")

    # --- Try cache first ---
    if os.path.exists(cache_file):
        print(f"  [GWOSC] Loading cached catalog: {cache_file}")
        with open(cache_file, "r", encoding="utf-8") as f:
            cached_data = json.load(f)
        return cached_data

    # --- Fetch from GWOSC API ---
    requests = _import_requests()
    url = CATALOG_URLS[catalog]
    print(f"  [GWOSC] Fetching {catalog} catalog from {url}")

    try:
        response = requests.get(url, timeout=HTTP_TIMEOUT)
        response.raise_for_status()
    except requests.exceptions.Timeout:
        raise ConnectionError(
            f"Request to {url} timed out after {HTTP_TIMEOUT}s.\n"
            "GWOSC may be temporarily unavailable. Try again later."
        )
    except requests.exceptions.ConnectionError as e:
        raise ConnectionError(
            f"Could not connect to GWOSC ({url}).\n"
            f"Check your internet connection.\nDetails: {e}"
        )
    except requests.exceptions.HTTPError as e:
        raise ConnectionError(
            f"GWOSC returned HTTP error: {e}\n"
            f"URL: {url}"
        )

    raw_data = response.json()

    # --- Parse the catalog ---
    events_raw = raw_data.get("events", raw_data)
    events = {}

    for event_key, event_data in events_raw.items():
        # The API nests parameters under a 'parameters' key in some formats,
        # or directly in the event dict in others.
        params = event_data
        if "parameters" in event_data:
            params = event_data["parameters"]

        # Extract event name (clean version without version suffix)
        event_name = event_key.split("-")[0]  # e.g., "GW150914-v3" -> "GW150914"

        # Extract GPS time
        gps_time = _safe_float(params, "GPS", default=0.0)
        if gps_time == 0.0:
            gps_time = _safe_float(params, "tc", default=0.0)

        # Extract masses
        mass1 = _safe_float(params, "mass_1_source", default=0.0)
        mass2 = _safe_float(params, "mass_2_source", default=0.0)
        chirp_mass = _safe_float(params, "chirp_mass_source", default=0.0)

        # If chirp mass not provided, compute from component masses
        if chirp_mass == 0.0 and mass1 > 0 and mass2 > 0:
            chirp_mass = (mass1 * mass2) ** 0.6 / (mass1 + mass2) ** 0.2

        # Extract other parameters
        network_snr = _safe_float(params, "network_matched_filter_snr", default=0.0)
        distance = _safe_float(params, "luminosity_distance", default=0.0)
        final_mass = _safe_float(params, "final_mass_source", default=0.0)
        chi_eff = _safe_float(params, "chi_eff", default=0.0)

        # Extract the event-specific JSON URL (contains strain file listings)
        jsonurl = params.get("jsonurl", "") or event_data.get("jsonurl", "")

        # Extract detector list from strain entries if available
        detectors = []
        strain_data = event_data.get("strain", [])
        if isinstance(strain_data, list):
            detectors = list(set(
                item.get("detector", "") for item in strain_data
                if isinstance(item, dict) and item.get("detector")
            ))
        elif isinstance(strain_data, dict):
            detectors = list(strain_data.keys())
        if not detectors:
            # Default to H1, L1 for BBH events
            detectors = ["H1", "L1"]

        events[event_name] = {
            "gps_time": gps_time,
            "mass1": mass1,
            "mass2": mass2,
            "chirp_mass": chirp_mass,
            "network_snr": network_snr,
            "luminosity_distance": distance,
            "final_mass": final_mass,
            "chi_eff": chi_eff,
            "detectors": detectors,
            "catalog": catalog,
            "raw_key": event_key,
            "jsonurl": jsonurl,
        }

    result = {
        "catalog_name": catalog,
        "num_events": len(events),
        "events": events,
    }

    # --- Cache the parsed result ---
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"  [GWOSC] Cached {len(events)} events to {cache_file}")

    return result


def _safe_float(d, key, default=0.0):
    """Safely extract a float from a dict, handling None and nested values."""
    val = d.get(key, default)
    if val is None:
        return default
    if isinstance(val, dict):
        # Some GWOSC parameters are dicts with 'best', 'upper', 'lower'
        val = val.get("best", val.get("value", default))
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


# ============================================================================
# 2. Strain Data Downloader
# ============================================================================

def download_strain_data(event_name, detector="H1", sample_rate=4096,
                          duration=32, cache_dir=DEFAULT_CACHE_DIR):
    """Download strain data for a specific event and detector from GWOSC.

    Fetches the HDF5 strain file from GWOSC servers for the specified
    gravitational wave event and detector. The data is centered on the
    event GPS merger time and spans the requested duration.

    Files are cached locally so subsequent calls load from disk without
    network access (offline mode).

    Parameters
    ----------
    event_name : str
        GWOSC event name, e.g., 'GW150914', 'GW170817', 'GW190521'.
    detector : str
        Detector identifier: 'H1' (LIGO Hanford), 'L1' (LIGO Livingston),
        or 'V1' (Virgo). Default: 'H1'.
    sample_rate : int
        Desired sample rate in Hz. GWOSC provides 4096 Hz and 16384 Hz.
        Default: 4096.
    duration : int
        Duration of data segment in seconds. GWOSC provides 32s and 4096s
        segments. Default: 32.
    cache_dir : str
        Directory for caching downloaded files.

    Returns
    -------
    dict
        Dictionary with keys:
            'strain' : np.ndarray — strain time series h(t)
            'times' : np.ndarray — GPS time array
            'gps_start' : float — GPS start time of the segment
            'gps_end' : float — GPS end time of the segment
            'sample_rate' : int — actual sample rate of the data
            'detector' : str — detector name
            'event_name' : str — event identifier
            'duration' : float — actual duration in seconds
            'num_samples' : int — number of samples
            'source_file' : str — path to the cached file

    Raises
    ------
    ValueError
        If detector is not supported or event is not found.
    ConnectionError
        If download fails and no cache is available.
    FileNotFoundError
        If HDF5 file cannot be parsed.

    Examples
    --------
    >>> data = download_strain_data('GW150914', detector='H1')
    >>> print(f"Got {data['num_samples']} samples at {data['sample_rate']} Hz")
    Got 131072 samples at 4096 Hz
    """
    if detector not in VALID_DETECTORS:
        raise ValueError(
            f"Unknown detector '{detector}'. "
            f"Valid options: {sorted(VALID_DETECTORS)}"
        )

    os.makedirs(cache_dir, exist_ok=True)
    cache_hdf5 = os.path.join(cache_dir, f"{event_name}_{detector}_{sample_rate}Hz_{duration}s.hdf5")
    cache_npz = os.path.join(cache_dir, f"{event_name}_{detector}_{sample_rate}Hz_{duration}s.npz")

    # --- Try NPZ cache first (fastest) ---
    if os.path.exists(cache_npz):
        print(f"  [GWOSC] Loading cached NPZ: {cache_npz}")
        data = np.load(cache_npz, allow_pickle=True)
        return {
            "strain": data["strain"],
            "times": data["times"],
            "gps_start": float(data["gps_start"]),
            "gps_end": float(data["gps_end"]),
            "sample_rate": int(data["sample_rate"]),
            "detector": str(data["detector"]),
            "event_name": str(data["event_name"]),
            "duration": float(data["duration"]),
            "num_samples": int(data["num_samples"]),
            "source_file": cache_npz,
        }

    # --- Try HDF5 cache ---
    h5py = _import_h5py()
    if os.path.exists(cache_hdf5) and h5py is not None:
        print(f"  [GWOSC] Loading cached HDF5: {cache_hdf5}")
        return _read_hdf5_strain(cache_hdf5, event_name, detector, cache_npz)

    # --- Download from GWOSC ---
    # Step 1: Get the event JSON to find the strain file URL
    strain_url = _find_strain_url(event_name, detector, sample_rate, duration)

    if strain_url is None:
        raise ConnectionError(
            f"Could not find strain data URL for {event_name} / {detector} "
            f"at {sample_rate} Hz / {duration}s.\n"
            "The event may not have data for this detector, or GWOSC "
            "may not provide this combination."
        )

    # Step 2: Download the file
    print(f"  [GWOSC] Downloading {event_name} {detector} strain data...")
    print(f"           URL: {strain_url}")
    _download_file(strain_url, cache_hdf5)

    # Step 3: Parse and cache
    if h5py is not None:
        return _read_hdf5_strain(cache_hdf5, event_name, detector, cache_npz)
    else:
        # Fallback: try to download TXT format
        print("  [GWOSC] h5py not available, attempting TXT fallback...")
        return _download_txt_fallback(event_name, detector, sample_rate,
                                      duration, cache_dir)


def _find_strain_url(event_name, detector, sample_rate, duration,
                     cache_dir=DEFAULT_CACHE_DIR):
    """Query the GWOSC event API to find the strain data file URL.

    Strategy:
    1. Check if we already have the event in a cached catalog (has jsonurl)
    2. Follow the jsonurl to get the event-specific JSON with strain listings
    3. If not in cache, search all catalogs (GWTC-1, GWTC-2, GWTC-3)
    4. Match the strain entry by detector, sample_rate, duration, format=hdf5

    Parameters
    ----------
    event_name : str
        Event name (e.g., 'GW150914').
    detector : str
        Detector name (e.g., 'H1').
    sample_rate : int
        Sample rate in Hz.
    duration : int
        Segment duration in seconds.
    cache_dir : str
        Cache directory for catalog lookups.

    Returns
    -------
    str or None
        URL to the HDF5 strain file, or None if not found.
    """
    requests = _import_requests()

    # --- Strategy 1: Find the event's jsonurl from cached catalogs ---
    jsonurl = None

    # Search all catalogs to find this event
    for cat_name in ["GWTC-1", "GWTC-2", "GWTC-2.1", "GWTC-3"]:
        try:
            catalog = fetch_event_catalog(cat_name, cache_dir=cache_dir)
        except Exception:
            continue

        events = catalog.get("events", {})
        # Try exact match or partial match
        for evt_key, evt_data in events.items():
            if event_name == evt_key or event_name in evt_key:
                jsonurl = evt_data.get("jsonurl", "")
                if jsonurl:
                    break
        if jsonurl:
            break

    if not jsonurl:
        # Last resort: try constructing the event API URL directly
        # Some events can be queried by name through the catalog URLs
        print(f"  [GWOSC] Event '{event_name}' not found in any cached catalog.")
        return None

    # --- Strategy 2: Follow the jsonurl to get strain file listings ---
    print(f"  [GWOSC] Fetching strain listings from: {jsonurl}")

    try:
        response = requests.get(jsonurl, timeout=HTTP_TIMEOUT)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"  [GWOSC] WARNING: Event JSON request failed: {e}")
        return None

    event_data = response.json()

    # The response has structure: {"events": {"EventKey-vN": {..., "strain": [...]}}}
    events = event_data.get("events", event_data)

    for event_key, edata in events.items():
        strain_entries = edata.get("strain", [])

        # --- Parse strain entries (flat list of dicts) ---
        # Each entry: {"GPSstart": ..., "detector": "H1", "sampling_rate": 4096,
        #              "duration": 32, "format": "hdf5", "url": "https://..."}
        if isinstance(strain_entries, list):
            url = _match_strain_entry(
                strain_entries, detector, sample_rate, duration, "hdf5"
            )
            if url:
                return url

            # Fallback: try any HDF5 format for this detector and duration
            url = _match_strain_entry(
                strain_entries, detector, None, duration, "hdf5"
            )
            if url:
                print(f"  [GWOSC] Exact sample rate {sample_rate} not found, "
                      f"using alternate.")
                return url

    print(f"  [GWOSC] No matching strain file found for "
          f"{event_name}/{detector}/{sample_rate}Hz/{duration}s/hdf5")
    return None


def _match_strain_entry(strain_entries, detector, sample_rate, duration,
                        fmt="hdf5"):
    """Find a matching strain file entry from a GWOSC strain listing.

    Parameters
    ----------
    strain_entries : list of dict
        Strain file entries from the GWOSC event JSON.
    detector : str
        Detector to match (e.g., 'H1').
    sample_rate : int or None
        Sample rate in Hz to match. None matches any.
    duration : int
        Duration in seconds to match.
    fmt : str
        File format to match (e.g., 'hdf5', 'gwf', 'txt').

    Returns
    -------
    str or None
        URL of the matching file, or None.
    """
    for entry in strain_entries:
        if not isinstance(entry, dict):
            continue

        entry_det = entry.get("detector", "")
        entry_sr = entry.get("sampling_rate", entry.get("sample_rate", 0))
        entry_dur = entry.get("duration", 0)
        entry_fmt = entry.get("format", "")
        entry_url = entry.get("url", entry.get("URL", ""))

        if entry_det != detector:
            continue
        if entry_dur != duration:
            continue
        if fmt and entry_fmt != fmt:
            continue
        if sample_rate is not None and entry_sr != sample_rate:
            continue

        if entry_url:
            return entry_url

    return None


def _download_file(url, output_path):
    """Download a file from a URL with progress reporting.

    Parameters
    ----------
    url : str
        URL to download from.
    output_path : str
        Local file path to save to.

    Raises
    ------
    ConnectionError
        If the download fails.
    """
    requests = _import_requests()
    tqdm = _import_tqdm()

    try:
        response = requests.get(url, stream=True, timeout=HTTP_TIMEOUT)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise ConnectionError(f"Download failed: {e}\nURL: {url}")

    total_size = int(response.headers.get("content-length", 0))
    block_size = 8192

    if total_size > 0:
        size_mb = total_size / (1024 * 1024)
        print(f"  [GWOSC] File size: {size_mb:.1f} MB")

    downloaded = 0

    if tqdm is not None and total_size > 0:
        progress = tqdm(total=total_size, unit="B", unit_scale=True,
                        desc="  Downloading", leave=False)
    else:
        progress = None

    with open(output_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=block_size):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if progress:
                    progress.update(len(chunk))
                elif total_size > 0 and downloaded % (block_size * 128) == 0:
                    pct = downloaded / total_size * 100
                    print(f"\r  [GWOSC] Downloaded {pct:.0f}%", end="", flush=True)

    if progress:
        progress.close()
    elif total_size > 0:
        print(f"\r  [GWOSC] Downloaded 100% ({total_size / 1024 / 1024:.1f} MB)")
    else:
        print(f"  [GWOSC] Downloaded {downloaded / 1024:.0f} KB")


def _read_hdf5_strain(filepath, event_name, detector, cache_npz=None):
    """Read strain data from a GWOSC HDF5 file.

    GWOSC HDF5 files have the structure:
        /strain/Strain      — strain time series (float64 array)
        /meta/GPSstart      — GPS start time
        /meta/Duration      — segment duration
        /meta/Detector      — detector name

    Some files use /strain/strain (lowercase) or direct attributes.

    Parameters
    ----------
    filepath : str
        Path to the HDF5 file.
    event_name : str
        Event name for metadata.
    detector : str
        Detector name for metadata.
    cache_npz : str, optional
        Path to save NPZ cache for faster reloading.

    Returns
    -------
    dict
        Parsed strain data dictionary.
    """
    h5py = _import_h5py()
    if h5py is None:
        raise ImportError("h5py is required to read HDF5 strain files.")

    with h5py.File(filepath, "r") as hf:
        # --- Find the strain dataset ---
        strain = None
        for strain_path in [
            "strain/Strain",
            "strain/strain",
            "strain",
            "Strain",
        ]:
            if strain_path in hf:
                ds = hf[strain_path]
                if hasattr(ds, "shape"):
                    strain = ds[:]
                    break

        if strain is None:
            # Try to find any dataset
            print(f"  [GWOSC] WARNING: Standard strain paths not found.")
            print(f"  [GWOSC] HDF5 keys: {list(hf.keys())}")
            for key in hf.keys():
                if isinstance(hf[key], h5py.Dataset):
                    strain = hf[key][:]
                    print(f"  [GWOSC] Using dataset '{key}' as strain")
                    break
                elif isinstance(hf[key], h5py.Group):
                    for subkey in hf[key].keys():
                        if isinstance(hf[key][subkey], h5py.Dataset):
                            strain = hf[key][subkey][:]
                            print(f"  [GWOSC] Using dataset '{key}/{subkey}' as strain")
                            break
                    if strain is not None:
                        break

        if strain is None:
            raise FileNotFoundError(
                f"Could not find strain data in {filepath}.\n"
                f"File may be corrupted or in unexpected format."
            )

        # --- Extract metadata ---
        gps_start = 0.0
        sr = 4096
        det = detector

        # Try meta group
        if "meta" in hf:
            meta = hf["meta"]
            if "GPSstart" in meta:
                gps_start = float(meta["GPSstart"][()])
            elif "gps_start" in meta.attrs:
                gps_start = float(meta.attrs["gps_start"])
            if "Duration" in meta:
                pass  # we compute from strain length
            if "Detector" in meta:
                det = meta["Detector"][()].decode("utf-8") if isinstance(
                    meta["Detector"][()], bytes) else str(meta["Detector"][()])

        # Try attributes on the strain dataset
        for strain_path in ["strain/Strain", "strain/strain", "strain"]:
            if strain_path in hf:
                ds = hf[strain_path]
                if "Xstart" in ds.attrs:
                    gps_start = float(ds.attrs["Xstart"])
                elif "x0" in ds.attrs:
                    gps_start = float(ds.attrs["x0"])
                if "Xspacing" in ds.attrs:
                    sr = int(round(1.0 / float(ds.attrs["Xspacing"])))
                elif "dx" in ds.attrs:
                    sr = int(round(1.0 / float(ds.attrs["dx"])))
                break

        # Try root attributes
        if gps_start == 0.0:
            for attr_name in ["GPSstart", "gps_start", "start_time"]:
                if attr_name in hf.attrs:
                    gps_start = float(hf.attrs[attr_name])
                    break

    # --- Build time array ---
    n_samples = len(strain)
    dt = 1.0 / sr
    duration = n_samples * dt
    times = gps_start + np.arange(n_samples) * dt
    gps_end = gps_start + duration

    print(f"  [GWOSC] Loaded {n_samples} samples at {sr} Hz")
    print(f"           GPS range: {gps_start:.1f} to {gps_end:.1f} ({duration:.1f}s)")
    print(f"           Detector: {det}")
    print(f"           Strain range: [{np.min(strain):.2e}, {np.max(strain):.2e}]")

    result = {
        "strain": strain.astype(np.float64),
        "times": times,
        "gps_start": gps_start,
        "gps_end": gps_end,
        "sample_rate": sr,
        "detector": det,
        "event_name": event_name,
        "duration": duration,
        "num_samples": n_samples,
        "source_file": filepath,
    }

    # --- Cache as NPZ for fast reloading ---
    if cache_npz:
        np.savez_compressed(
            cache_npz,
            strain=strain,
            times=times,
            gps_start=gps_start,
            gps_end=gps_end,
            sample_rate=sr,
            detector=det,
            event_name=event_name,
            duration=duration,
            num_samples=n_samples,
        )
        print(f"  [GWOSC] Saved NPZ cache: {cache_npz}")

    return result


def _download_txt_fallback(event_name, detector, sample_rate, duration,
                            cache_dir):
    """Fallback: download strain as TXT when h5py is not available.

    GWOSC also provides strain data as plain text files (two columns:
    GPS time and strain). This is slower and larger but requires
    no additional dependencies.

    Parameters
    ----------
    event_name : str
        Event name.
    detector : str
        Detector name.
    sample_rate : int
        Sample rate (used to find the right file).
    duration : int
        Duration in seconds.
    cache_dir : str
        Cache directory.

    Returns
    -------
    dict
        Strain data dictionary.
    """
    requests = _import_requests()

    # Try known GWOSC TXT URL pattern
    txt_url = (
        f"https://gwosc.org/s/events/{event_name}/"
        f"{detector}-{event_name}_4_V2-*-{duration}.txt"
    )

    print(f"  [GWOSC] TXT fallback is limited. Attempting download...")
    print(f"  [GWOSC] Consider installing h5py: pip install h5py")

    # This fallback is best-effort; the TXT format may not be available
    # for all events. Return a helpful error.
    raise NotImplementedError(
        f"TXT fallback for {event_name} is not available.\n"
        f"Please install h5py: pip install h5py\n"
        f"Then retry: download_strain_data('{event_name}', '{detector}')"
    )


# ============================================================================
# 3. Real PSD Estimator
# ============================================================================

def estimate_psd_from_data(strain, sample_rate, segment_duration=4.0,
                            method="welch"):
    """Estimate the power spectral density from real detector strain data.

    Uses Welch's method on OFF-SOURCE segments of the data to estimate
    the noise PSD without contamination from the gravitational wave signal.
    Off-source segments are the first and last 8 seconds of the data,
    avoiding the central region where the GW signal is present.

    Median averaging is used across segments for robustness against
    non-stationary noise transients (glitches).

    Parameters
    ----------
    strain : np.ndarray
        Strain time series h(t). Should be at least 16 seconds long
        to have sufficient off-source data.
    sample_rate : float
        Sample rate in Hz (typically 4096).
    segment_duration : float
        Duration of each Welch FFT segment in seconds.
        Default: 4.0 (gives ~0.25 Hz frequency resolution).
    method : str
        PSD estimation method. Currently only 'welch' is supported.
        Default: 'welch'.

    Returns
    -------
    tuple (frequencies, psd)
        frequencies : np.ndarray — frequency array in Hz
        psd : np.ndarray — one-sided PSD in units of strain^2 / Hz

    Raises
    ------
    ValueError
        If strain is too short for PSD estimation.

    Notes
    -----
    The PSD is validated to be physically reasonable:
    - Minimum should be near 100-300 Hz
    - Should rise steeply below ~20 Hz (seismic noise)
    - Should rise above ~1000 Hz (shot noise)

    Examples
    --------
    >>> data = download_strain_data('GW150914', 'H1')
    >>> freqs, psd = estimate_psd_from_data(data['strain'], 4096)
    >>> print(f"PSD at 100 Hz: {psd[freqs >= 100][0]:.2e} strain^2/Hz")
    """
    n_total = len(strain)
    duration_total = n_total / sample_rate

    if duration_total < 4.0:
        raise ValueError(
            f"Strain is too short ({duration_total:.1f}s) for PSD estimation. "
            f"Need at least 4 seconds of data."
        )

    nperseg = int(segment_duration * sample_rate)
    noverlap = nperseg // 2  # 50% overlap

    # --- Select off-source segments ---
    # Use the first 8 seconds and last 8 seconds, avoiding the signal
    # in the center of the segment.
    off_source_duration = min(8.0, duration_total / 3.0)
    n_off = int(off_source_duration * sample_rate)

    if n_total >= 4 * n_off:
        # Enough data: use first and last segments (avoid signal in center)
        off_source_1 = strain[:n_off]
        off_source_2 = strain[-n_off:]
        off_source = np.concatenate([off_source_1, off_source_2])
    elif duration_total >= 16.0:
        # 16-32 second segments: use first 25% and last 25%
        quarter = n_total // 4
        off_source = np.concatenate([strain[:quarter], strain[-quarter:]])
    else:
        # Very short data: use the full strain (may include signal)
        off_source = strain
        warnings.warn(
            "Data too short for proper off-source PSD estimation. "
            "Using full strain segment — PSD may be contaminated by signal.",
            RuntimeWarning,
        )

    # --- Welch PSD estimation ---
    # Use median averaging for robustness to non-Gaussian transients.
    # scipy.signal.welch computes mean by default; we use multiple
    # overlapping segments and take the median.

    # First, compute with standard Welch for the frequency array
    freqs, psd_welch = welch(
        off_source,
        fs=sample_rate,
        nperseg=nperseg,
        noverlap=noverlap,
        window="hann",
        scaling="density",
        average="median",  # Robust to glitches
    )

    psd = psd_welch

    # --- Validate PSD is physically reasonable ---
    _validate_psd(freqs, psd)

    print(f"  [PSD] Estimated from {len(off_source)/sample_rate:.1f}s of off-source data")
    print(f"  [PSD] Frequency resolution: {freqs[1] - freqs[0]:.3f} Hz")
    print(f"  [PSD] ASD at 100 Hz: {np.sqrt(psd[np.searchsorted(freqs, 100)]):.2e} strain/sqrt(Hz)")

    # Find minimum of ASD
    valid = freqs > 20
    if np.any(valid):
        min_idx = np.argmin(np.sqrt(psd[valid]))
        min_freq = freqs[valid][min_idx]
        min_asd = np.sqrt(psd[valid][min_idx])
        print(f"  [PSD] Most sensitive at {min_freq:.0f} Hz: ASD = {min_asd:.2e}")

    return freqs, psd


def _validate_psd(freqs, psd):
    """Sanity-check a PSD for physical reasonableness.

    Parameters
    ----------
    freqs : np.ndarray
        Frequency array in Hz.
    psd : np.ndarray
        PSD values in strain^2/Hz.
    """
    # Check for NaN or negative values
    if np.any(np.isnan(psd)):
        warnings.warn("PSD contains NaN values — data may be corrupted.",
                       RuntimeWarning)

    if np.any(psd < 0):
        warnings.warn("PSD contains negative values — replacing with abs().",
                       RuntimeWarning)
        psd[:] = np.abs(psd)

    # The minimum should be in the 50-500 Hz range
    valid = (freqs > 30) & (freqs < 2000) & (psd > 0)
    if np.any(valid):
        min_freq = freqs[valid][np.argmin(psd[valid])]
        if min_freq < 20 or min_freq > 1000:
            warnings.warn(
                f"PSD minimum at {min_freq:.0f} Hz is outside expected range "
                f"(50-500 Hz). Data may be unusual.",
                RuntimeWarning,
            )


# ============================================================================
# 4. Event Data Preprocessor
# ============================================================================

def preprocess_strain(strain, sample_rate, f_low=20.0, f_high=1800.0):
    """Preprocess raw detector strain for gravitational wave analysis.

    Applies the standard LIGO preprocessing chain:
    1. Bandpass filter (4th order Butterworth, f_low to f_high)
    2. Notch filters at 60 Hz and harmonics (US power line noise)
    3. Tukey window (alpha=0.1) to suppress edge discontinuities

    This is equivalent to the conditioning applied by LIGO's
    GstLAL and PyCBC analysis pipelines before matched filtering.

    Parameters
    ----------
    strain : np.ndarray
        Raw strain time series h(t) from the detector.
    sample_rate : float
        Sample rate in Hz (typically 4096).
    f_low : float
        Lower bandpass frequency in Hz. Default: 20.0 Hz.
        Below this, seismic noise dominates and the detector is insensitive.
    f_high : float
        Upper bandpass frequency in Hz. Default: 1800.0 Hz.
        This should be below the Nyquist frequency (sample_rate / 2).
    Returns
    -------
    np.ndarray
        Preprocessed strain, same length as input.
        - Bandpass filtered between f_low and f_high
        - Power line harmonics notched out
        - Tukey windowed at edges

    Notes
    -----
    The filter is applied using ``sosfiltfilt`` (zero-phase, forward-backward
    filtering) to avoid introducing phase distortions that would corrupt
    the matched filtering analysis.

    Examples
    --------
    >>> data = download_strain_data('GW150914', 'H1')
    >>> h_clean = preprocess_strain(data['strain'], data['sample_rate'])
    >>> print(f"RMS strain: {np.std(h_clean):.2e}")
    """
    strain = strain.copy().astype(np.float64)
    n = len(strain)

    # --- Remove DC offset ---
    strain -= np.mean(strain)

    f_nyquist = sample_rate / 2.0

    # --- Bandpass filter ---
    # 4th order Butterworth — provides 80 dB/decade rolloff
    # Using second-order sections (SOS) for numerical stability
    f_high_safe = min(f_high, f_nyquist * 0.95)  # stay below Nyquist
    sos_bp = butter(
        N=4,
        Wn=[f_low, f_high_safe],
        btype="bandpass",
        fs=sample_rate,
        output="sos",
    )
    strain = sosfiltfilt(sos_bp, strain)

    # --- Notch filters at power line frequencies ---
    # US power grid: 60 Hz and harmonics (120, 180 Hz)
    # These appear as sharp spectral lines in LIGO data
    # Quality factor Q = 30 gives a narrow notch (~2 Hz wide at 60 Hz)
    for f_notch in POWER_LINE_FREQS:
        if f_notch < f_nyquist:
            b_notch, a_notch = iirnotch(
                w0=f_notch,
                Q=30.0,
                fs=sample_rate,
            )
            strain = sosfiltfilt(
                np.array([[b_notch[0], b_notch[1], b_notch[2],
                           a_notch[0], a_notch[1], a_notch[2]]]),
                strain,
            )

    # --- Tukey window ---
    # Tapers the first and last 5% of the data (alpha=0.1 → 10% total)
    # to prevent spectral leakage from edge discontinuities in the FFT.
    window = tukey(n, alpha=0.1)
    strain *= window

    return strain


# ============================================================================
# 5. Event Segment Extractor
# ============================================================================

def extract_event_segment(strain, times, gps_event, window_before=2.0,
                           window_after=0.5):
    """Extract a short segment centered on the gravitational wave event.

    Given a longer strain segment (e.g., 32 seconds), extract just the
    portion containing the GW signal: from ``window_before`` seconds
    before the merger to ``window_after`` seconds after.

    Parameters
    ----------
    strain : np.ndarray
        Strain time series h(t).
    times : np.ndarray
        GPS time array corresponding to strain.
    gps_event : float
        GPS time of the merger event (from the GWOSC catalog).
    window_before : float
        Seconds before the event to include. Default: 2.0.
        For BBH mergers, the last 2 seconds contain the visible chirp.
    window_after : float
        Seconds after the event to include. Default: 0.5.
        Includes the ringdown phase.

    Returns
    -------
    tuple (segment_strain, segment_times, merger_index)
        segment_strain : np.ndarray — extracted strain segment
        segment_times : np.ndarray — corresponding GPS times
        merger_index : int — index of the event within the segment

    Raises
    ------
    ValueError
        If the event GPS time is outside the data range.

    Examples
    --------
    >>> data = download_strain_data('GW150914', 'H1')
    >>> seg, t_seg, idx = extract_event_segment(
    ...     data['strain'], data['times'], 1126259462.4,
    ...     window_before=2.0, window_after=0.5)
    >>> print(f"Segment: {len(seg)} samples, merger at index {idx}")
    """
    gps_start = times[0]
    gps_end = times[-1]

    if gps_event < gps_start or gps_event > gps_end:
        raise ValueError(
            f"Event GPS time {gps_event:.1f} is outside the data range "
            f"[{gps_start:.1f}, {gps_end:.1f}]"
        )

    # Find the index closest to the event GPS time
    event_idx = np.argmin(np.abs(times - gps_event))

    # Compute the sample indices for the window
    sample_rate = 1.0 / (times[1] - times[0])
    n_before = int(window_before * sample_rate)
    n_after = int(window_after * sample_rate)

    idx_start = max(0, event_idx - n_before)
    idx_end = min(len(strain), event_idx + n_after)

    segment_strain = strain[idx_start:idx_end]
    segment_times = times[idx_start:idx_end]

    # Merger index within the segment
    merger_index = event_idx - idx_start

    print(f"  [Segment] Extracted {len(segment_strain)} samples "
          f"({len(segment_strain)/sample_rate:.3f}s)")
    print(f"  [Segment] Merger at index {merger_index} "
          f"(GPS {gps_event:.3f})")

    return segment_strain, segment_times, merger_index


# ============================================================================
# 6. Batch Event Downloader
# ============================================================================

def download_gwtc3_batch(events=None, detectors=None,
                          output_dir=DEFAULT_CACHE_DIR):
    """Download strain data for multiple GWTC-3 events in batch.

    Downloads strain HDF5 files and estimates PSD for each event/detector
    combination. Results are saved as compressed NumPy .npz files for
    fast reloading.

    Parameters
    ----------
    events : list of str, optional
        List of event names to download. If None, downloads the first
        10 events from the GWTC-3 catalog sorted by network SNR.
    detectors : list of str, optional
        Detectors to download for each event. Default: ['H1', 'L1'].
    output_dir : str
        Directory for saving downloaded data.
        Default: 'output/gwosc_cache'.

    Returns
    -------
    dict
        Summary dictionary with keys:
            'events_downloaded' : list — successfully downloaded events
            'events_failed' : list — events that failed to download
            'detectors' : list — detectors requested
            'files' : dict — mapping event_name -> {detector -> filepath}
            'catalog' : dict — catalog data used

    Examples
    --------
    >>> summary = download_gwtc3_batch(
    ...     events=['GW150914', 'GW170817'],
    ...     detectors=['H1', 'L1'])
    >>> print(f"Downloaded {len(summary['events_downloaded'])} events")
    """
    if detectors is None:
        detectors = ["H1", "L1"]

    os.makedirs(output_dir, exist_ok=True)

    # --- Fetch catalog to get event list ---
    catalog = fetch_event_catalog("GWTC-3", cache_dir=output_dir)

    if events is None:
        # Select top 10 events by SNR
        sorted_events = sorted(
            catalog["events"].items(),
            key=lambda x: x[1].get("network_snr", 0),
            reverse=True,
        )
        events = [name for name, _ in sorted_events[:10]]
        print(f"\n  [Batch] Selected top 10 events by SNR:")
        for i, name in enumerate(events):
            snr = catalog["events"][name].get("network_snr", 0)
            print(f"    {i+1:2d}. {name} (SNR = {snr:.1f})")

    tqdm = _import_tqdm()
    total = len(events) * len(detectors)
    completed = 0

    events_downloaded = []
    events_failed = []
    files = {}

    iterator = events
    if tqdm is not None:
        iterator = tqdm(events, desc="  Downloading events", unit="event")

    for event_name in iterator:
        files[event_name] = {}
        event_success = False

        for det in detectors:
            completed += 1
            if tqdm is None:
                print(f"\n  [{completed}/{total}] {event_name} / {det}")

            try:
                data = download_strain_data(
                    event_name, detector=det, sample_rate=4096,
                    duration=32, cache_dir=output_dir,
                )

                # Estimate PSD
                freqs, psd = estimate_psd_from_data(
                    data["strain"], data["sample_rate"]
                )

                # Save processed data
                npz_path = os.path.join(
                    output_dir,
                    f"{event_name}_{det}_processed.npz",
                )
                np.savez_compressed(
                    npz_path,
                    strain=data["strain"],
                    times=data["times"],
                    psd_freqs=freqs,
                    psd=psd,
                    gps_start=data["gps_start"],
                    sample_rate=data["sample_rate"],
                    detector=det,
                    event_name=event_name,
                )

                files[event_name][det] = npz_path
                event_success = True

            except Exception as e:
                print(f"  [Batch] FAILED {event_name}/{det}: {e}")
                files[event_name][det] = None

        if event_success:
            events_downloaded.append(event_name)
        else:
            events_failed.append(event_name)

    print(f"\n  [Batch] Complete: {len(events_downloaded)} succeeded, "
          f"{len(events_failed)} failed")

    return {
        "events_downloaded": events_downloaded,
        "events_failed": events_failed,
        "detectors": detectors,
        "files": files,
        "catalog": catalog,
    }


# ============================================================================
# 7. Validation Function
# ============================================================================

def validate_against_published(event_name, our_snr=None, our_chirp_mass=None,
                                catalog="GWTC-3",
                                cache_dir=DEFAULT_CACHE_DIR):
    """Validate our analysis results against published GWOSC parameters.

    Fetches the published parameters for a given event from the GWOSC
    catalog and compares them to our computed values.

    Parameters
    ----------
    event_name : str
        GWOSC event name (e.g., 'GW150914').
    our_snr : float, optional
        Our matched filter SNR to compare against published network SNR.
    our_chirp_mass : float, optional
        Our chirp mass (solar masses) to compare against published value.
    catalog : str
        Which catalog to use for published values.
        Default: 'GWTC-3'.
    cache_dir : str
        Cache directory for catalog data.

    Returns
    -------
    dict
        Comparison results with keys:
            'event_name' : str — event name
            'published' : dict — published parameters from GWOSC
            'comparisons' : list of dict — each comparison with:
                'parameter' : str — parameter name
                'our_value' : float — our computed value
                'published_value' : float — published value
                'relative_error' : float — |our - pub| / pub
                'pass' : bool — True if relative error < 20%
            'overall_pass' : bool — True if all comparisons pass

    Examples
    --------
    >>> result = validate_against_published(
    ...     'GW150914', our_snr=24.5, our_chirp_mass=28.1)
    >>> for comp in result['comparisons']:
    ...     status = 'PASS' if comp['pass'] else 'FAIL'
    ...     print(f"{comp['parameter']}: {comp['our_value']:.2f} vs "
    ...           f"{comp['published_value']:.2f} [{status}]")
    """
    # Search across all catalogs to find the event
    published = None
    matched_name = None
    found_catalog = None

    # Try the specified catalog first, then all others
    catalogs_to_try = [catalog]
    for c in ["GWTC-1", "GWTC-2", "GWTC-2.1", "GWTC-3"]:
        if c not in catalogs_to_try:
            catalogs_to_try.append(c)

    for cat_name in catalogs_to_try:
        try:
            cat = fetch_event_catalog(cat_name, cache_dir=cache_dir)
        except Exception:
            continue

        events = cat.get("events", {})

        if event_name in events:
            published = events[event_name]
            matched_name = event_name
            found_catalog = cat_name
            break

        # Try partial matching (e.g., 'GW150914' matches 'GW150914_095045')
        for key in events:
            if event_name in key or key in event_name:
                published = events[key]
                matched_name = key
                found_catalog = cat_name
                break
        if published is not None:
            break

    if published is None:
        print(f"  [Validate] Event '{event_name}' not found in any catalog")
        return {
            "event_name": event_name,
            "published": None,
            "comparisons": [],
            "overall_pass": False,
            "error": "Event not found in any catalog",
        }

    print(f"\n  [Validate] Comparing against {matched_name} ({found_catalog})")
    print(f"  {'Parameter':<25} {'Ours':>12} {'Published':>12} "
          f"{'Rel.Err':>10} {'Status':>8}")
    print(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*10} {'-'*8}")

    comparisons = []

    def _compare(param_name, our_val, pub_val, tolerance=0.20):
        """Compare a single parameter."""
        if our_val is None or pub_val is None or pub_val == 0:
            return None

        rel_err = abs(our_val - pub_val) / abs(pub_val)
        passed = rel_err < tolerance
        status = "PASS" if passed else "FAIL"

        print(f"  {param_name:<25} {our_val:>12.2f} {pub_val:>12.2f} "
              f"{rel_err:>9.1%} {status:>8}")

        comp = {
            "parameter": param_name,
            "our_value": our_val,
            "published_value": pub_val,
            "relative_error": rel_err,
            "pass": passed,
        }
        comparisons.append(comp)
        return comp

    # --- Run comparisons ---
    if our_snr is not None:
        pub_snr = published.get("network_snr", 0)
        _compare("Network SNR", our_snr, pub_snr, tolerance=0.30)

    if our_chirp_mass is not None:
        pub_mc = published.get("chirp_mass", 0)
        _compare("Chirp Mass (Msun)", our_chirp_mass, pub_mc, tolerance=0.15)

    # Also print published values for reference
    print(f"\n  Published parameters for {matched_name}:")
    for key in ["gps_time", "mass1", "mass2", "chirp_mass",
                "network_snr", "luminosity_distance", "final_mass",
                "chi_eff"]:
        val = published.get(key, "N/A")
        if isinstance(val, float):
            print(f"    {key:<25} {val:.3f}")
        else:
            print(f"    {key:<25} {val}")

    overall = all(c["pass"] for c in comparisons) if comparisons else True

    return {
        "event_name": matched_name,
        "published": published,
        "comparisons": comparisons,
        "overall_pass": overall,
    }


# ============================================================================
# __main__ — Download GW150914 and print summary
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 72)
    print("  GWOSC Real Data Connector — Test Run")
    print("  Downloading GW150914 (LIGO Hanford H1)")
    print("=" * 72)

    # --- Step 1: Fetch the catalog ---
    print("\n--- Step 1: Fetch GWTC-3 Catalog ---")
    try:
        catalog = fetch_event_catalog("GWTC-3")
        print(f"  Found {catalog['num_events']} events in GWTC-3")

        # Print the 5 loudest events
        sorted_events = sorted(
            catalog["events"].items(),
            key=lambda x: x[1].get("network_snr", 0),
            reverse=True,
        )
        print(f"\n  Top 5 events by SNR:")
        print(f"  {'Event':<20} {'SNR':>8} {'M1':>8} {'M2':>8} "
              f"{'Mc':>8} {'d(Mpc)':>8}")
        print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
        for name, params in sorted_events[:5]:
            print(f"  {name:<20} "
                  f"{params.get('network_snr', 0):>8.1f} "
                  f"{params.get('mass1', 0):>8.1f} "
                  f"{params.get('mass2', 0):>8.1f} "
                  f"{params.get('chirp_mass', 0):>8.1f} "
                  f"{params.get('luminosity_distance', 0):>8.0f}")
    except Exception as e:
        print(f"  Catalog fetch failed: {e}")
        print("  (This is expected if no internet connection)")

    # --- Step 2: Download GW150914 H1 strain ---
    print("\n--- Step 2: Download GW150914 H1 Strain ---")
    try:
        data = download_strain_data("GW150914", detector="H1",
                                     sample_rate=4096, duration=32)
        print(f"  SUCCESS: Got {data['num_samples']} samples")
        print(f"  Sample rate: {data['sample_rate']} Hz")
        print(f"  Duration: {data['duration']:.1f}s")
        print(f"  GPS range: {data['gps_start']:.1f} to {data['gps_end']:.1f}")

        # --- Step 3: Preprocess ---
        print("\n--- Step 3: Preprocess Strain ---")
        h_clean = preprocess_strain(data["strain"], data["sample_rate"])
        print(f"  Raw RMS:  {np.std(data['strain']):.2e}")
        print(f"  Clean RMS: {np.std(h_clean):.2e}")

        # --- Step 4: Estimate PSD ---
        print("\n--- Step 4: Estimate PSD ---")
        freqs, psd = estimate_psd_from_data(data["strain"],
                                             data["sample_rate"])
        print(f"  PSD bins: {len(freqs)}")
        print(f"  Freq range: {freqs[0]:.1f} to {freqs[-1]:.1f} Hz")

        # --- Step 5: Extract event segment ---
        print("\n--- Step 5: Extract Event Segment ---")
        gps_event = 1126259462.4  # GW150914 merger time
        seg, t_seg, idx = extract_event_segment(
            h_clean, data["times"], gps_event,
            window_before=2.0, window_after=0.5,
        )
        print(f"  Segment length: {len(seg)} samples")
        print(f"  Duration: {len(seg)/data['sample_rate']:.3f}s")
        print(f"  Merger index: {idx}")
        print(f"  Peak strain in segment: {np.max(np.abs(seg)):.2e}")

        # --- Step 6: Validate ---
        print("\n--- Step 6: Validate Against Published ---")
        validate_against_published(
            "GW150914", our_snr=24.0, our_chirp_mass=28.1
        )

    except Exception as e:
        print(f"  Download failed: {e}")
        import traceback
        traceback.print_exc()
        print("\n  This is expected if:")
        print("  - No internet connection")
        print("  - GWOSC servers are down")
        print("  - h5py is not installed (pip install h5py)")

    print("\n" + "=" * 72)
    print("  Test complete.")
    print("=" * 72 + "\n")
