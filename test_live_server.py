"""Test the live server endpoints."""
import urllib.request
import json
import sys

BASE = 'http://localhost:8000'
passed = 0
failed = 0

def test(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  [PASS] {name}")
    else:
        failed += 1
        print(f"  [FAIL] {name}  --  {detail}")

def post_json(path, data):
    payload = json.dumps(data).encode()
    req = urllib.request.Request(f'{BASE}{path}', data=payload,
                                headers={'Content-Type': 'application/json'})
    resp = urllib.request.urlopen(req, timeout=60)
    return json.loads(resp.read().decode('utf-8'))

print("\n=== 1. FRONTEND SERVING ===")
try:
    resp = urllib.request.urlopen(f'{BASE}/')
    html = resp.read().decode('utf-8')
    test("GET / returns HTML", len(html) > 1000, f"len={len(html)}")
    test("HTML has doctype/html", '<!DOCTYPE' in html or '<html' in html)
    test("HTML links app.js", 'app.js' in html)
    test("HTML links styles.css", 'styles.css' in html)
except Exception as e:
    test("GET / reachable", False, str(e))

# Static files
for path in ['/static/styles.css', '/static/app.js']:
    try:
        resp = urllib.request.urlopen(f'{BASE}{path}')
        data = resp.read()
        test(f"GET {path} serves ({len(data)} bytes)", len(data) > 100)
    except Exception as e:
        test(f"GET {path}", False, str(e))

print("\n=== 2. POST /analyze ===")
try:
    r = post_json('/analyze', {'m1': 36, 'm2': 29, 'distance': 410})
    test("/analyze returns dict", isinstance(r, dict))
    for key in ['physics', 'waveform', 'spectrogram', 'snr']:
        test(f"/analyze has '{key}'", key in r)
    # Audio is at top level as audio_base64
    test("/analyze has 'audio_base64'", 'audio_base64' in r)
    test("audio_base64 not empty", len(r.get('audio_base64', '')) > 100)
    # Check physics values (key is 'chirp_mass' not 'chirp_mass_solar')
    p = r.get('physics', {})
    mc = p.get('chirp_mass', p.get('chirp_mass_solar', 0))
    test("chirp_mass ~ 28.3", abs(mc - 28.3) < 1.5, f"Mc={mc:.2f}")
    # Waveform data
    wf = r.get('waveform', {})
    test("waveform has time array", 'time' in wf and len(wf['time']) > 100)
    test("waveform has h_plus array", 'h_plus' in wf and len(wf['h_plus']) > 100)
    # SNR
    snr = r.get('snr', {})
    peak_snr = snr.get('peak_snr', p.get('peak_snr', 0))
    test("peak_snr > 0", peak_snr > 0, f"peak_snr={peak_snr}")
except Exception as e:
    test("/analyze endpoint works", False, str(e))

print("\n=== 3. POST /compare ===")
try:
    r = post_json('/compare', {'m1': 36, 'm2': 29})
    test("/compare returns dict", isinstance(r, dict))
    test("/compare has 'models'", 'models' in r)
    models = r.get('models', {})
    test("/compare has 'imrphenomd'", 'imrphenomd' in models)
    test("/compare has 'taylorf2'", 'taylorf2' in models)
except Exception as e:
    test("/compare endpoint works", False, str(e))

print("\n=== 4. POST /estimate ===")
try:
    r = post_json('/estimate', {'m1': 36, 'm2': 29})
    test("/estimate returns dict", isinstance(r, dict))
    test("/estimate has estimated_params", 'estimated_params' in r)
    ep = r.get('estimated_params', {})
    # Key is 'm1' not 'best_m1'
    m1 = ep.get('m1', ep.get('best_m1', 0))
    test("recovered m1 reasonable", m1 is not None and abs(m1 - 36) < 15, f"m1={m1}")
except Exception as e:
    test("/estimate endpoint works", False, str(e))

print("\n=== 5. POST /network ===")
try:
    r = post_json('/network', {'m1': 36, 'm2': 29})
    test("/network returns dict", isinstance(r, dict))
    test("/network has 'detectors'", 'detectors' in r)
    test("/network has 'network_snr'", 'network_snr' in r)
    test("/network SNR > 0", r.get('network_snr', 0) > 0)
    dets = r.get('detectors', {})
    for d in ['H1', 'L1', 'V1', 'K1']:
        test(f"/network has {d}", d in dets)
except Exception as e:
    test("/network endpoint works", False, str(e))

print("\n=== 6. POST /parameter_space ===")
try:
    r = post_json('/parameter_space', {'m1': 36, 'm2': 29})
    test("/parameter_space returns dict", isinstance(r, dict))
    test("/parameter_space has 'grid'", 'grid' in r)
except Exception as e:
    test("/parameter_space endpoint works", False, str(e))

print(f"\n{'='*60}")
print(f"  RESULTS: {passed} passed, {failed} failed")
print(f"{'='*60}\n")

sys.exit(1 if failed > 0 else 0)
