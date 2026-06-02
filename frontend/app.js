/* eslint-disable */
/* jshint esversion: 8 */
/* global document, window, $, Chart, fetch, requestAnimationFrame, console, AudioContext */

const ACC    = '#7ba7c8';
const ACC2   = 'rgba(123,167,200,0.55)';
const ACC_F  = 'rgba(123,167,200,0.08)';
const MER    = '#b5623c';
const MER2   = 'rgba(181,98,60,0.55)';
const MER_F  = 'rgba(181,98,60,0.08)';
const RING   = '#7b6da0';
const RING_F = 'rgba(123,109,160,0.08)';
const LIVE   = '#5a9a78';
const LIVE_F = 'rgba(90,154,120,0.08)';
const SIG    = '#d8dde6';
const SIG2   = 'rgba(216,221,230,0.35)';
const ALERT  = '#8a4040';
const GRID   = 'rgba(160,180,220,0.06)';
const TICK   = 'rgba(160,180,220,0.3)';
const MONO   = "'Space Mono',monospace";

const MODEL_COLORS = { imrphenomd: ACC, taylorf2: MER, pn_ringdown: RING };
const MODEL_LABELS = { imrphenomd: 'IMRPhenomD', taylorf2: 'TaylorF2 (SPA)', pn_ringdown: 'TaylorT4+Ringdown' };

const S = {
  m1: 36, m2: 29,
  s1z: 0, s2z: 0,
  distance: 410, inclination: 0, snr: 25,
  result: null, audioEl: null, playing: false,
};

const docId = id => document.getElementById(id);
const charts = new Map();

function setHtmlSafe(elId, val, unit, unitClass="pc-unit") {
  const el = docId(elId);
  if (!el) return;
  el.textContent = val;
  if (unit) {
    const span = document.createElement('span');
    span.className = unitClass;
    span.textContent = unit;
    el.appendChild(span);
  }
}


function destroyChart(id) {
  if (charts.has(id)) { charts.get(id).destroy(); charts.delete(id); }
}

function makeChart(canvasId, type, data, options) {
  destroyChart(canvasId);
  const cv = docId(canvasId);
  if (!cv) return null;
  charts.set(canvasId, new Chart(cv, { type, data, options }));
  return charts.get(canvasId);
}

function chartOpts(xlbl, ylbl, extra) {
  const o = {
    responsive: true, maintainAspectRatio: false,
    animation: { duration: 350 },
    plugins: { legend: { display: false }, tooltip: {
      backgroundColor: 'rgba(8,12,20,0.92)', borderColor: 'rgba(160,180,220,0.15)',
      borderWidth: 1, titleFont: { family: MONO, size: 8 }, bodyFont: { family: MONO, size: 9 },
      titleColor: TICK, bodyColor: SIG, padding: 8, cornerRadius: 4,
    }},
    scales: {
      x: { grid: { color: GRID, lineWidth: 0.5 }, border: { color: 'rgba(160,180,220,0.07)' },
           ticks: { color: TICK, font: { family: MONO, size: 7 }, maxTicksLimit: 7 },
           title: { display: !!xlbl, text: xlbl, color: TICK, font: { size: 8, family: MONO } } },
      y: { grid: { color: GRID, lineWidth: 0.5 }, border: { color: 'rgba(160,180,220,0.07)' },
           ticks: { color: TICK, font: { family: MONO, size: 7 }, maxTicksLimit: 5 },
           title: { display: !!ylbl, text: ylbl, color: TICK, font: { size: 8, family: MONO } } },
    },
  };
  if (extra) {
    if (extra.scales) {
      if (extra.scales.x) Object.assign(o.scales.x, extra.scales.x);
      if (extra.scales.y) Object.assign(o.scales.y, extra.scales.y);
    }
    if (extra.plugins) Object.assign(o.plugins, extra.plugins);
  }
  return o;
}

function tick() {
  const n = new Date(), p = v => String(v).padStart(2, '0');
  docId('clock').textContent =
    `${n.getUTCFullYear()}-${p(n.getUTCMonth()+1)}-${p(n.getUTCDate())} ` +
    `${p(n.getUTCHours())}:${p(n.getUTCMinutes())}:${p(n.getUTCSeconds())} UTC`;
}
tick(); setInterval(tick, 1000);

function switchTab(id) {
  document.querySelectorAll('.tp').forEach(t => t.classList.remove('on'));
  document.querySelectorAll('.tb').forEach(b => b.classList.remove('on'));
  docId('tp-' + id).classList.add('on');
  document.querySelector(`[data-tab="${id}"]`).classList.add('on');
  requestAnimationFrame(() => {
    Object.values(charts).forEach(c => { try { c.resize(); } catch(e) {} });
  });
}
document.querySelectorAll('.tb').forEach(btn => {
  btn.addEventListener('click', () => switchTab(btn.dataset.tab));
});

(function() {
  const cv = docId('cosmos'), cx = cv.getContext('2d');
  let W, H, stars = [];
  function resize() {
    W = cv.width = innerWidth; H = cv.height = innerHeight;
    stars = [];
    for (let i = 0; i < 600; i++) {
      const mw = Math.random() < 0.4;
      stars.push({
        x: (mw ? 0.3 + Math.random() * 0.4 : Math.random()) * W,
        y: (mw ? 0.1 + Math.random() * 0.8 : Math.random()) * H,
        r: mw ? Math.random() * 0.6 + 0.1 : Math.random() * 0.8 + 0.1,
        a: Math.random() * 0.7 + 0.15,
        tw: Math.random() * Math.PI * 2,
        spd: (Math.random() * 0.4 + 0.1) * (Math.random() < 0.5 ? 1 : -1),
        hue: Math.random() < 0.08 ? [180, 210, 260, 300][Math.floor(Math.random() * 4)] : null,
      });
    }
  }
  resize(); window.addEventListener('resize', resize);
  function draw() {
    cx.clearRect(0, 0, W, H);
    const t = Date.now() * 0.001;
    stars.forEach(s => {
      const al = s.a * (0.5 + 0.5 * Math.sin(t * s.spd + s.tw));
      cx.beginPath(); cx.arc(s.x, s.y, s.r, 0, Math.PI * 2);
      cx.fillStyle = s.hue !== null
        ? `hsla(${s.hue},40%,85%,${al * 0.6})`
        : `rgba(210,220,235,${al})`;
      cx.fill();
    });
    requestAnimationFrame(draw);
  }
  draw();
})();

(function() {
  const cv = docId('bh-ambient'), cx = cv.getContext('2d');
  const W = 520, H = 520; cv.width = W; cv.height = H;
  let phase = 0;
  function draw() {
    cx.clearRect(0, 0, W, H);
    const CX = W * 0.65, CY = H * 0.35;
    for (let r = 120; r > 0; r -= 2) {
      const al = (1 - r / 120) * 0.012;
      cx.beginPath();
      cx.ellipse(CX, CY, r * 2.2, r * 0.55, -0.25, 0, Math.PI * 2);
      const g = cx.createRadialGradient(CX, CY, 0, CX, CY, r * 2.2);
      g.addColorStop(0, 'rgba(181,98,60,0)');
      g.addColorStop(0.5, `rgba(181,98,60,${al})`);
      g.addColorStop(1, 'rgba(100,50,20,0)');
      cx.fillStyle = g; cx.fill();
    }
    for (let i = 0; i < 6; i++) {
      const r = 55 + i * 55 + ((phase * 80) % 55);
      const al = Math.max(0, (1 - r / 380) * 0.15);
      cx.beginPath(); cx.arc(CX, CY, r, 0, Math.PI * 2);
      cx.strokeStyle = `rgba(123,167,200,${al})`; cx.lineWidth = 0.8; cx.stroke();
    }
    cx.beginPath(); cx.arc(CX, CY, 46, 0, Math.PI * 2);
    cx.strokeStyle = 'rgba(123,167,200,0.08)'; cx.lineWidth = 1; cx.stroke();
    const r0 = 30;
    cx.beginPath(); cx.arc(CX, CY, r0, 0, Math.PI * 2);
    cx.fillStyle = '#000'; cx.fill();
    cx.beginPath(); cx.arc(CX, CY, r0, 0, Math.PI * 2);
    cx.strokeStyle = `rgba(181,98,60,${0.3 + 0.1 * Math.sin(phase * 1.5)})`;
    cx.lineWidth = 1.5; cx.stroke();
    phase += 0.008;
    requestAnimationFrame(draw);
  }
  draw();
})();

const orbitState = { phi: 0 };
(function() {
  const cv = docId('orbit-canvas'), cx = cv.getContext('2d');
  const W = 240, H = 148, CX = 120, CY = 74;

  function bh(x, y, r, col, gcolBase) {
    const g = cx.createRadialGradient(x, y, 0, x, y, r * 3);
    g.addColorStop(0, gcolBase.replace(',1)', ',0.18)'));
    g.addColorStop(1, gcolBase.replace(',1)', ',0)'));
    cx.beginPath(); cx.arc(x, y, r * 3, 0, Math.PI * 2);
    cx.fillStyle = g; cx.fill();
    cx.beginPath(); cx.arc(x, y, r, 0, Math.PI * 2);
    cx.fillStyle = '#000'; cx.fill();
    cx.beginPath(); cx.arc(x, y, r, 0, Math.PI * 2);
    cx.strokeStyle = col; cx.lineWidth = 1; cx.stroke();
  }

  function draw() {
    cx.clearRect(0, 0, W, H);
    const m1 = S.m1, m2 = S.m2, M = m1 + m2;
    const a = 80;
    cx.save(); cx.globalAlpha = 0.09;
    for (let r = 30; r < 140; r += 24) {
      cx.beginPath(); cx.arc(CX, CY, r, 0, Math.PI * 2);
      cx.strokeStyle = '#7ba7c8'; cx.lineWidth = 0.6; cx.stroke();
    }
    cx.restore();
    cx.beginPath(); cx.ellipse(CX, CY, a, a * 0.52, 0, 0, Math.PI * 2);
    cx.strokeStyle = 'rgba(160,180,220,0.1)'; cx.lineWidth = 1;
    cx.setLineDash([3, 5]); cx.stroke(); cx.setLineDash([]);
    const r1 = a * m2 / M, r2 = a * m1 / M;
    const phi = orbitState.phi;
    const x1 = CX + r1 * Math.cos(phi), y1 = CY + r1 * Math.sin(phi) * 0.52;
    const x2 = CX - r2 * Math.cos(phi), y2 = CY - r2 * Math.sin(phi) * 0.52;
    const prog = Math.abs(Math.sin(phi * 0.3));
    for (let i = 1; i <= 4; i++) {
      cx.beginPath(); cx.arc(CX, CY, 42 + i * 26, 0, Math.PI * 2);
      cx.strokeStyle = `rgba(181,98,60,${0.04 * prog / i})`; cx.lineWidth = 0.8; cx.stroke();
    }
    const s1 = Math.max(7.5, 6 + m1 / 6.5), s2 = Math.max(6, 4.5 + m2 / 6.5);
    bh(x1, y1, s1, 'rgba(123,167,200,0.7)', 'rgba(123,167,200,1)');
    bh(x2, y2, s2, 'rgba(181,98,60,0.6)', 'rgba(181,98,60,1)');
    orbitState.phi += 0.022;
    requestAnimationFrame(draw);
  }
  draw();
})();

function computeBasicPhysics(m1, m2, s1z, s2z) {
  const M = m1 + m2;
  const eta = (m1 * m2) / (M * M);
  const Mc = M * Math.pow(eta, 0.6);
  const Msec = M * 4.926e-6; // solar mass in seconds
  const fisco = 1 / (Math.pow(6, 1.5) * Math.PI * Msec);
  const chi_eff = (m1 * s1z + m2 * s2z) / M;
  return { M, eta, Mc, fisco, chi_eff };
}

function updateDerivedPhysics() {
  const p = computeBasicPhysics(S.m1, S.m2, S.s1z, S.s2z);
  setHtmlSafe('p-mtot', p.M.toFixed(1), 'M☉', 'pc-unit');
  setHtmlSafe('p-mc', p.Mc.toFixed(2), 'M☉', 'pc-unit');
  docId('p-eta').textContent = p.eta.toFixed(4);
  docId('p-chi').textContent = p.chi_eff.toFixed(3);
  setHtmlSafe('p-fisco', p.fisco.toFixed(1), 'Hz', 'pc-unit');
  // Ratio bar
  docId('ratio-m1').textContent = `m₁ ${S.m1} M☉`;
  docId('ratio-m2').textContent = `m₂ ${S.m2} M☉`;
  docId('ratio-eta').textContent = `η = ${p.eta.toFixed(4)}`;
  docId('ratio-fill').style.width = `${(S.m1 / p.M) * 100}%`;
  // Confidence bars
  docId('conf-mtot').style.width = `${Math.min(100, p.M / 160 * 100)}%`;
  docId('conf-mc').style.width = `${Math.min(100, p.Mc / 65 * 100)}%`;
}

function initSliders() {
  docId('sl-m1').addEventListener('input', function() {
    S.m1 = parseFloat(this.value);
    if (S.m1 < S.m2) {
      S.m2 = S.m1;
      docId('sl-m2').value = S.m2;
      docId('v-m2').textContent = S.m2.toFixed(1) + ' M☉';
    }
    docId('v-m1').textContent = S.m1.toFixed(1) + ' M☉';
    updateDerivedPhysics();
  });
  docId('sl-m2').addEventListener('input', function() {
    S.m2 = parseFloat(this.value);
    if (S.m2 > S.m1) {
      S.m1 = S.m2;
      docId('sl-m1').value = S.m1;
      docId('v-m1').textContent = S.m1.toFixed(1) + ' M☉';
    }
    docId('v-m2').textContent = S.m2.toFixed(1) + ' M☉';
    updateDerivedPhysics();
  });
  docId('sl-s1z').addEventListener('input', function() {
    S.s1z = parseFloat(this.value);
    docId('v-s1z').textContent = S.s1z.toFixed(2);
    updateDerivedPhysics();
  });
  docId('sl-s2z').addEventListener('input', function() {
    S.s2z = parseFloat(this.value);
    docId('v-s2z').textContent = S.s2z.toFixed(2);
    updateDerivedPhysics();
  });
  docId('sl-dist').addEventListener('input', function() {
    S.distance = parseFloat(this.value);
    docId('v-dist').textContent = S.distance.toFixed(0) + ' Mpc';
  });
  docId('sl-inc').addEventListener('input', function() {
    S.inclination = parseFloat(this.value);
    docId('v-inc').textContent = S.inclination.toFixed(2) + ' rad';
  });
}

function ironRgb(v) {
  if (v < 0.4) {
    const t = v / 0.4;
    return [Math.floor(20 * t), Math.floor(12 * t), Math.floor(30 + 20 * t)];
  }
  if (v < 0.7) {
    const t = (v - 0.4) / 0.3;
    return [Math.floor(90 * t + 8), Math.floor(20 * t + 5), Math.floor(50 - 20 * t)];
  }
  const t = (v - 0.7) / 0.3;
  return [Math.floor(181 + 40 * t), Math.floor(98 + 100 * t), Math.floor(60 - 40 * t)];
}

function viridisRgb(v) {
  if (v < 0.25) { const t = v / 0.25; return [Math.floor(68 * t), Math.floor(1 + 50 * t), Math.floor(84 + 84 * t)]; }
  if (v < 0.5) { const t = (v - 0.25) / 0.25; return [Math.floor(68 - 35 * t), Math.floor(51 + 80 * t), Math.floor(168 - 20 * t)]; }
  if (v < 0.75) { const t = (v - 0.5) / 0.25; return [Math.floor(33 + 90 * t), Math.floor(131 + 50 * t), Math.floor(148 - 70 * t)]; }
  const t = (v - 0.75) / 0.25;
  return [Math.floor(123 + 130 * t), Math.floor(181 + 50 * t), Math.floor(78 - 50 * t)];
}

async function runAnalysis() {
  const btn = docId('btn-generate');
  btn.classList.add('loading');
  btn.innerHTML = '<span class="spinner"></span> Analyzing...';

  try {
    const res = await fetch('/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        m1: S.m1, m2: S.m2,
        s1z: S.s1z, s2z: S.s2z,
        distance: S.distance, inclination: S.inclination,
        snr: S.snr, method: 'imrphenomd',
      }),
    });
    const data = await res.json();

    if (data.status === 'error') {
      showError('analysis-await', data.error);
      return;
    }

    S.result = data;
    docId('pipe-elapsed').textContent = data.elapsed + 's';
    docId('analysis-await').style.display = 'none';

    updateKPIs(data.physics);
    updateSidebarPhysics(data.physics);
    renderStrainChart(data.waveform);
    renderQTransform(data.spectrogram, data.freq_track);
    renderSNRChart(data.snr);
    renderFreqChart(data.waveform);
    setupAudio(data.audio_base64, data.audio_duration);

  } catch (e) {
    showError('analysis-await', 'Connection failed: ' + e.message);
  } finally {
    btn.classList.remove('loading');
    btn.innerHTML = 'Generate Waveform';
  }
}

function showError(containerId, msg) {
  const el = docId(containerId);
  el.style.display = 'block';
  el.textContent = "";
  const div = document.createElement("div");
  div.className = "error-msg";
  div.textContent = "⚠️ " + msg;
  el.appendChild(div);
}

function updateKPIs(p) {
  setHtmlSafe('kpi-mc', p.chirp_mass.toFixed(2), 'M☉', 'kpi-unit');
  docId('kpi-snr').textContent = p.optimal_snr.toFixed(1);
  setHtmlSafe('kpi-erad', p.energy_radiated.toFixed(2), 'M☉c²', 'kpi-unit');
  docId('kpi-af').textContent = p.final_spin.toFixed(3);
  setHtmlSafe('kpi-dist', p.distance_mpc, 'Mpc', 'kpi-unit');
}

function updateSidebarPhysics(p) {
  setHtmlSafe('p-mtot', p.total_mass.toFixed(1), 'M☉', 'pc-unit');
  setHtmlSafe('p-mc', p.chirp_mass.toFixed(2), 'M☉', 'pc-unit');
  docId('p-eta').textContent = p.eta.toFixed(4);
  docId('p-chi').textContent = p.chi_eff.toFixed(3);
  setHtmlSafe('p-fisco', p.f_isco_hz.toFixed(1), 'Hz', 'pc-unit');
  setHtmlSafe('p-fring', p.f_ring_hz.toFixed(1), 'Hz', 'pc-unit');
  setHtmlSafe('p-fqnm', p.f_qnm_hz.toFixed(1), 'Hz', 'pc-unit');
  setHtmlSafe('p-tau', (p.tau_qnm_s * 1000).toFixed(2), 'ms', 'pc-unit');
  setHtmlSafe('p-mf', p.final_mass.toFixed(1), 'M☉', 'pc-unit');
  docId('p-af').textContent = p.final_spin.toFixed(3);
  setHtmlSafe('p-erad', p.energy_radiated.toFixed(2), 'M☉c²', 'pc-unit');
  docId('p-qf').textContent = p.quality_factor.toFixed(1);
  docId('p-hpeak').textContent = p.peak_strain.toExponential(2);
  docId('p-osnr').textContent = p.optimal_snr.toFixed(1);
  docId('p-psnr').textContent = p.peak_snr.toFixed(1);
  setHtmlSafe('p-dur', p.duration.toFixed(3), 's', 'pc-unit');
}

function renderStrainChart(wf) {
  const pts_hp = wf.time.map((t, i) => ({ x: t, y: wf.h_plus[i] }));
  const pts_hx = wf.time.map((t, i) => ({ x: t, y: wf.h_cross[i] }));
  makeChart('c-strain', 'scatter', {
    datasets: [
      { data: pts_hp, showLine: true, borderColor: SIG, borderWidth: 1.1, pointRadius: 0, tension: 0.1, label: 'h+(t)' },
      { data: pts_hx, showLine: true, borderColor: SIG2, borderWidth: 0.9, borderDash: [4, 4], pointRadius: 0, tension: 0.1, label: 'h×(t)' },
    ],
  }, chartOpts('Time [ms]', 'Strain'));
}

function renderSNRChart(snr) {
  const pts = snr.time.map((t, i) => ({ x: t, y: snr.values[i] }));
  const thresh = snr.time.map(t => ({ x: t, y: snr.threshold }));
  makeChart('c-snr', 'scatter', {
    datasets: [
      { data: pts, showLine: true, borderColor: SIG, borderWidth: 1.1, pointRadius: 0,
        fill: { target: 'origin', above: 'rgba(216,221,230,0.04)' }, tension: 0.15, label: 'ρ (t)' },
      { data: thresh, showLine: true, borderColor: ALERT, borderWidth: 0.9,
        borderDash: [5, 4], pointRadius: 0, fill: false, label: 'Threshold' },
    ],
  }, chartOpts('Time [ms]', 'SNR ρ (t)', { scales: { y: { min: 0 } } }));
}

function renderFreqChart(wf) {
  const pts = wf.time.map((t, i) => ({ x: t, y: wf.frequency[i] }));
  makeChart('c-freq', 'scatter', {
    datasets: [{
      data: pts, showLine: true, borderColor: LIVE, borderWidth: 1.1, pointRadius: 0,
      fill: { target: 'origin', above: LIVE_F }, tension: 0.3, label: 'f(t)',
    }],
  }, chartOpts('Time [ms]', 'f [Hz]', { scales: { y: { min: 0, max: 520 } } }));
}

function renderQTransform(spec, freqTrack) {
  const cv = docId('c-qtrans');
  if (!cv) return;
  const cx = cv.getContext('2d');
  const container = cv.parentElement;
  const W = cv.width = container.offsetWidth - 30;
  const H = cv.height = 186;

  cx.fillStyle = '#0c1120';
  cx.fillRect(0, 0, W, H);

  if (!spec || !spec.energy || !spec.energy.length) return;

  const { times, frequencies, energy } = spec;
  const nFreq = frequencies.length;
  const nTime = times.length;

  // Find max energy for normalization
  let maxE = 0;
  for (let i = 0; i < nFreq; i++)
    for (let j = 0; j < nTime; j++)
      if (energy[i][j] > maxE) maxE = energy[i][j];
  if (maxE === 0) return;

  const fMin = Math.max(frequencies[0], 1);
  const fMax = frequencies[nFreq - 1];
  const tMin = times[0], tMax = times[nTime - 1];
  const logFMin = Math.log(fMin), logFMax = Math.log(fMax);

  // Draw using rectangles for efficiency
  const cellW = Math.max(1, Math.ceil(W / nTime));
  for (let fi = 0; fi < nFreq; fi++) {
    const logF = Math.log(Math.max(frequencies[fi], 1));
    const py = H - ((logF - logFMin) / (logFMax - logFMin)) * H;
    const cellH = Math.max(1, H / nFreq);
    for (let ti = 0; ti < nTime; ti++) {
      const e = energy[fi][ti] / maxE;
      if (e < 0.015) continue;
      const px = ((ti) / nTime) * W;
      const rgb = ironRgb(Math.min(e, 1));
      cx.fillStyle = `rgba(${rgb[0]},${rgb[1]},${rgb[2]},${Math.min(e * 0.95, 0.95)})`;
      cx.fillRect(px, py - cellH / 2, cellW + 1, cellH + 1);
    }
  }

  // Chirp track overlay
  if (freqTrack && freqTrack.times && freqTrack.freqs) {
    cx.beginPath();
    cx.strokeStyle = 'rgba(220,228,240,0.55)';
    cx.lineWidth = 1.2;
    cx.setLineDash([5, 4]);
    let first = true;
    for (let i = 0; i < freqTrack.times.length; i++) {
      const ft = freqTrack.freqs[i];
      if (ft === null || ft <= 0 || ft < fMin || ft > fMax) continue;
      const px = ((freqTrack.times[i] - tMin) / (tMax - tMin)) * W;
      const logFt = Math.log(ft);
      const py = H - ((logFt - logFMin) / (logFMax - logFMin)) * H;
      if (first) { cx.moveTo(px, py); first = false; }
      else cx.lineTo(px, py);
    }
    cx.stroke();
    cx.setLineDash([]);
  }

  // Axis labels
  cx.fillStyle = 'rgba(160,180,220,0.4)';
  cx.font = "7px 'Space Mono',monospace";
  cx.fillText(`${Math.round(fMax)} Hz`, 3, 11);
  cx.fillText(`${Math.round(fMin)}`, 3, H - 5);

  // Color bar
  const gb = cx.createLinearGradient(0, 0, 0, H);
  gb.addColorStop(0, 'rgba(221,198,175,0.9)');
  gb.addColorStop(0.4, 'rgba(181,98,60,0.8)');
  gb.addColorStop(1, 'rgba(20,14,30,0.5)');
  cx.fillStyle = gb;
  cx.fillRect(W - 10, 0, 8, H);
  cx.fillStyle = 'rgba(160,180,220,0.4)';
  cx.fillText('1', W - 8, 10);
  cx.fillText('0', W - 8, H - 3);
}

async function runCompare() {
  const btn = docId('btn-compare');
  btn.classList.add('loading');
  btn.textContent = 'Computing...';

  const models = [];
  if (docId('chk-imr').checked) models.push('imrphenomd');
  if (docId('chk-tay').checked) models.push('taylorf2');
  if (docId('chk-pnr').checked) models.push('pn_ringdown');
  if (!models.length) {
    btn.classList.remove('loading'); btn.textContent = 'Compare Models';
    return;
  }

  try {
    const res = await fetch('/compare', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        m1: S.m1, m2: S.m2, s1z: S.s1z, s2z: S.s2z,
        distance: S.distance, inclination: S.inclination, models,
      }),
    });
    const data = await res.json();

    if (data.status === 'error') {
      showError('compare-await', data.error); return;
    }

    docId('compare-await').style.display = 'none';
    docId('cmp-charts').style.display = 'grid';
    docId('cmp-table-panel').style.display = 'block';

    const ok = Object.entries(data.models).filter(([, v]) => v.status === 'ok');

    // Strain comparison
    const strainDS = ok.map(([n, m]) => ({
      data: m.time.map((t, i) => ({ x: t, y: m.h_plus[i] })),
      showLine: true, borderColor: MODEL_COLORS[n] || '#fff',
      borderWidth: 1.2, pointRadius: 0, tension: 0.1, label: MODEL_LABELS[n] || n,
    }));
    makeChart('c-wf-cmp', 'scatter', { datasets: strainDS },
      chartOpts('Time [ms]', 'Strain h+'));
    renderLegend('cmp-strain-leg', ok.map(([n]) => [MODEL_LABELS[n], MODEL_COLORS[n]]));

    // Frequency comparison
    const freqDS = ok.map(([n, m]) => ({
      data: m.time.map((t, i) => ({ x: t, y: m.frequency[i] })),
      showLine: true, borderColor: MODEL_COLORS[n] || '#fff',
      borderWidth: 1.2, pointRadius: 0, tension: 0.3, label: MODEL_LABELS[n] || n,
    }));
    makeChart('c-fq-cmp', 'scatter', { datasets: freqDS },
      chartOpts('Time [ms]', 'f [Hz]', { scales: { y: { min: 0, max: 500 } } }));
    renderLegend('cmp-freq-leg', ok.map(([n]) => [MODEL_LABELS[n], MODEL_COLORS[n]]));

    // Parameter table
    buildCompareTable(ok);

  } catch (e) {
    showError('compare-await', 'Connection failed: ' + e.message);
  } finally {
    btn.classList.remove('loading');
    btn.textContent = 'Compare Models';
  }
}

function renderLegend(containerId, items) {
  const el = docId(containerId);
  el.innerHTML = items.map(([lbl, col]) =>
    `<div class="leg-i"><div class="leg-ln" style="background:${col}"></div>${lbl}</div>`
  ).join('');
}

function buildCompareTable(ok) {
  const thead = docId('cmp-thead');
  thead.innerHTML = '<tr><th>Parameter</th>' + ok.map(([n]) =>
    `<th style="text-align:right;color:${MODEL_COLORS[n]}">${MODEL_LABELS[n] || n}</th>`
  ).join('') + '</tr>';

  const params = [
    ['Chirp Mass (M☉)', 'chirp_mass_solar'],
    ['Total Mass (M☉)', 'total_mass_solar'],
    ['f_ISCO (Hz)', 'f_isco_hz'],
    ['Duration (s)', 'duration_seconds'],
    ['Peak Strain', 'peak_strain'],
    ['Phase Model', 'method'],
  ];
  const tbody = docId('cmp-tbody');
  tbody.innerHTML = params.map(([label, key]) =>
    `<tr><td>${label}</td>` + ok.map(([, m]) => {
      const v = m.params?.[key];
      const s = v === undefined ? '—'
        : typeof v === 'number' ? (Math.abs(v) < 0.01 ? v.toExponential(2) : v.toFixed(3))
        : v;
      return `<td style="text-align:right">${s}</td>`;
    }).join('') + '</tr>'
  ).join('');
}

async function runEstimate() {
  const btn = docId('btn-estimate');
  btn.classList.add('loading');
  btn.textContent = 'Searching...';

  try {
    const res = await fetch('/estimate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        m1_true: S.m1, m2_true: S.m2,
        s1z_true: S.s1z, s2z_true: S.s2z,
        distance: S.distance,
        m1_min: +docId('pe-m1min').value, m1_max: +docId('pe-m1max').value,
        m2_min: +docId('pe-m2min').value, m2_max: +docId('pe-m2max').value,
        coarse_step: 3.0, fine_step: 0.5,
      }),
    });
    const data = await res.json();

    if (data.status === 'error') {
      showError('pe-await', data.error); return;
    }

    docId('pe-await').style.display = 'none';
    docId('pe-results').style.display = 'block';
    docId('pe-badge').textContent = `${data.n_templates} templates · ${data.elapsed}s`;

    renderPEHeatmap(data.fine_grid, data.true_params, data.estimated_params);
    renderPEMarginals(data.fine_grid);
    renderPEChirpMassPosterior(data.fine_grid);
    renderPEWaveformOverlay(data);

  } catch (e) {
    showError('pe-await', 'Connection failed: ' + e.message);
  } finally {
    btn.classList.remove('loading');
    btn.textContent = 'Run Estimation';
  }
}

function renderPEHeatmap(grid, trueP, estP) {
  renderHeatmap('c-pe-heat', grid.match_matrix, grid.m2_values, grid.m1_values,
    viridisRgb, {
      xlabel: 'm₂ [M☉]', ylabel: 'm₁ [M☉]',
      markers: [
        { x: trueP.m2, y: trueP.m1, color: '#ff4444', symbol: 'x', label: 'True' },
        { x: estP.m2, y: estP.m1, color: LIVE, symbol: 'star', label: 'Est.' },
      ],
    });
}

function renderPEMarginals(grid) {
  // M1 marginal: best match for each m1
  const m1Vals = grid.m1_values;
  const m1Marg = m1Vals.map((_, i) => Math.max(...grid.match_matrix[i]));
  makeChart('c-post-m1', 'scatter', {
    datasets: [{
      data: m1Vals.map((v, i) => ({ x: v, y: m1Marg[i] })),
      showLine: true, borderColor: ACC, borderWidth: 1.2, pointRadius: 0,
      fill: { target: 'origin', above: ACC_F }, tension: 0.3,
    }],
  }, chartOpts('m₁ [M☉]', 'Match'));

  // M2 marginal: best match for each m2
  const m2Vals = grid.m2_values;
  const n1 = grid.match_matrix.length;
  const m2Marg = m2Vals.map((_, j) => {
    let best = 0;
    for (let i = 0; i < n1; i++) best = Math.max(best, grid.match_matrix[i][j]);
    return best;
  });
  makeChart('c-post-m2', 'scatter', {
    datasets: [{
      data: m2Vals.map((v, i) => ({ x: v, y: m2Marg[i] })),
      showLine: true, borderColor: MER, borderWidth: 1.2, pointRadius: 0,
      fill: { target: 'origin', above: MER_F }, tension: 0.3,
    }],
  }, chartOpts('m₂ [M☉]', 'Match'));
}

function renderPEChirpMassPosterior(grid) {
  // Compute chirp mass for each grid point, bin by chirp mass, weight by match
  const m1Vals = grid.m1_values, m2Vals = grid.m2_values;
  const mcVals = [];
  for (let i = 0; i < m1Vals.length; i++) {
    for (let j = 0; j < m2Vals.length; j++) {
      const m1 = m1Vals[i], m2 = m2Vals[j], M = m1 + m2;
      const eta = (m1 * m2) / (M * M);
      const mc = M * Math.pow(eta, 0.6);
      mcVals.push({ mc, match: grid.match_matrix[i][j] });
    }
  }

  const mcMin = Math.min(...mcVals.map(v => v.mc));
  const mcMax = Math.max(...mcVals.map(v => v.mc));
  const nBins = 40;
  const bins = new Float64Array(nBins);
  mcVals.forEach(({ mc, match }) => {
    const bi = Math.min(nBins - 1, Math.floor((mc - mcMin) / (mcMax - mcMin) * nBins));
    bins[bi] = Math.max(bins[bi], match);
  });

  const pts = Array.from(bins).map((v, i) => ({
    x: mcMin + (i + 0.5) * (mcMax - mcMin) / nBins, y: v,
  }));

  makeChart('c-post-mc', 'scatter', {
    datasets: [{
      data: pts, showLine: true, borderColor: RING, borderWidth: 1.2, pointRadius: 0,
      fill: { target: 'origin', above: RING_F }, tension: 0.3,
    }],
  }, chartOpts('Chirp Mass [M☉]', 'Match'));
}

function renderPEWaveformOverlay(data) {
  if (!data.observed_waveform || !data.bestfit_waveform) return;
  const obs = data.observed_waveform, fit = data.bestfit_waveform;
  makeChart('c-pe-overlay', 'scatter', {
    datasets: [
      { data: obs.time.map((t, i) => ({ x: t, y: obs.h_plus[i] })),
        showLine: true, borderColor: '#ff6666', borderWidth: 1.1, pointRadius: 0,
        tension: 0.1, label: 'Observed' },
      { data: fit.time.map((t, i) => ({ x: t, y: fit.h_plus[i] })),
        showLine: true, borderColor: LIVE, borderWidth: 1.1, borderDash: [4, 3],
        pointRadius: 0, tension: 0.1, label: 'Best Fit' },
    ],
  }, chartOpts('Time [ms]', 'Strain'));
}

async function runNetwork() {
  const btn = docId('btn-network');
  btn.classList.add('loading');
  btn.textContent = 'Computing...';

  try {
    const res = await fetch('/network', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        m1: S.m1, m2: S.m2, s1z: S.s1z, s2z: S.s2z,
        distance: S.distance, inclination: S.inclination,
        ra: +docId('net-ra').value, dec: +docId('net-dec').value,
        psi: +docId('net-psi').value,
        detectors: ['H1', 'L1', 'V1', 'K1'],
      }),
    });
    const data = await res.json();

    if (data.status === 'error') {
      showError('tp-network', data.error); return;
    }

    docId('net-badge').textContent = `Network SNR: ${data.network_snr}`;

    // Update detector cards
    const detMap = { H1: 'det-h1-snr', L1: 'det-l1-snr', V1: 'det-v1-snr', K1: 'det-k1-snr' };
    Object.entries(data.detectors).forEach(([k, d]) => {
      const el = docId(detMap[k]);
      if (el) el.textContent = d.optimal_snr.toFixed(1);
    });

    renderNetworkMap(data);
    renderTimingChart(data);

  } catch (e) {
    console.error('Network error:', e);
  } finally {
    btn.classList.remove('loading');
    btn.textContent = 'Analyze Network';
  }
}

function renderNetworkMap(data) {
  const cv = docId('c-netmap');
  if (!cv) return;
  const cx = cv.getContext('2d');
  const W = cv.width = cv.parentElement.offsetWidth - 30;
  const H = cv.height = 230;

  cx.fillStyle = '#080c14';
  cx.fillRect(0, 0, W, H);

  const margin = { left: 30, right: 15, top: 10, bottom: 20 };
  const pW = W - margin.left - margin.right;
  const pH = H - margin.top - margin.bottom;

  // Grid lines
  cx.strokeStyle = 'rgba(160,180,220,0.06)';
  cx.lineWidth = 0.5;
  for (let lat = -60; lat <= 60; lat += 30) {
    const y = margin.top + (90 - lat) / 180 * pH;
    cx.beginPath(); cx.moveTo(margin.left, y); cx.lineTo(margin.left + pW, y); cx.stroke();
  }
  for (let lon = -180; lon <= 180; lon += 60) {
    const x = margin.left + (lon + 180) / 360 * pW;
    cx.beginPath(); cx.moveTo(x, margin.top); cx.lineTo(x, margin.top + pH); cx.stroke();
  }

  // Equator
  const eqY = margin.top + pH / 2;
  cx.strokeStyle = 'rgba(160,180,220,0.12)';
  cx.beginPath(); cx.moveTo(margin.left, eqY); cx.lineTo(margin.left + pW, eqY); cx.stroke();

  // Border
  cx.strokeStyle = 'rgba(160,180,220,0.15)';
  cx.lineWidth = 1;
  cx.strokeRect(margin.left, margin.top, pW, pH);

  // Detector positions
  const dets = {
    H1: { lat: 46.455, lon: -119.408, color: ACC, glow: 'rgba(123,167,200,0.3)' },
    L1: { lat: 30.563, lon: -90.774, color: MER, glow: 'rgba(181,98,60,0.3)' },
    V1: { lat: 43.631, lon: 10.505, color: RING, glow: 'rgba(74,114,102,0.3)' },
    K1: { lat: 36.410, lon: 137.306, color: '#ffd166', glow: 'rgba(255,209,102,0.3)' },
  };

  // Draw connections between active detectors
  const activePos = {};
  Object.entries(dets).forEach(([k, d]) => {
    const x = margin.left + (d.lon + 180) / 360 * pW;
    const y = margin.top + (90 - d.lat) / 180 * pH;
    activePos[k] = { x, y };
  });

  cx.setLineDash([3, 4]);
  cx.lineWidth = 0.8;
  const activeKeys = Object.keys(activePos);
  activeKeys.forEach((a, i, arr) => {
    for (let j = i + 1; j < arr.length; j++) {
      const b = arr[j];
      cx.strokeStyle = 'rgba(123,167,200,0.15)';
      cx.beginPath();
      cx.moveTo(activePos[a].x, activePos[a].y);
      cx.lineTo(activePos[b].x, activePos[b].y);
      cx.stroke();
    }
  });
  cx.setLineDash([]);

  // Draw detectors
  Object.entries(dets).forEach(([k, d]) => {
    const { x, y } = activePos[k];
    const isActive = !!data.detectors[k];

    // Glow
    if (isActive) {
      const g = cx.createRadialGradient(x, y, 0, x, y, 18);
      g.addColorStop(0, d.glow);
      g.addColorStop(1, 'transparent');
      cx.fillStyle = g;
      cx.beginPath(); cx.arc(x, y, 18, 0, Math.PI * 2); cx.fill();
    }

    cx.beginPath(); cx.arc(x, y, isActive ? 4 : 3, 0, Math.PI * 2);
    cx.fillStyle = d.color; cx.fill();

    cx.font = "bold 8px 'Space Mono',monospace";
    cx.fillStyle = d.color;
    cx.fillText(k, x + 7, y - 5);

    if (data.detectors[k]) {
      cx.font = "7px 'Space Mono',monospace";
      cx.fillStyle = 'rgba(160,180,220,0.5)';
      cx.fillText(`SNR ${data.detectors[k].optimal_snr}`, x + 7, y + 6);
    }
  });

  // Source position
  if (data.sky_position) {
    const ra = data.sky_position.ra_deg;
    const dec = data.sky_position.dec_deg;
    const sx = margin.left + ((ra / 360 * 360 + 180) % 360) / 360 * pW;
    const sy = margin.top + (90 - dec) / 180 * pH;

    // Source glow
    const sg = cx.createRadialGradient(sx, sy, 0, sx, sy, 25);
    sg.addColorStop(0, 'rgba(181,98,60,0.3)');
    sg.addColorStop(1, 'transparent');
    cx.fillStyle = sg;
    cx.beginPath(); cx.arc(sx, sy, 25, 0, Math.PI * 2); cx.fill();

    // Cross marker
    cx.strokeStyle = MER;
    cx.lineWidth = 1.5;
    cx.beginPath(); cx.moveTo(sx - 5, sy); cx.lineTo(sx + 5, sy); cx.stroke();
    cx.beginPath(); cx.moveTo(sx, sy - 5); cx.lineTo(sx, sy + 5); cx.stroke();

    cx.font = "7px 'Space Mono',monospace";
    cx.fillStyle = MER;
    cx.fillText('SOURCE', sx + 8, sy - 3);
  }

  // Axis labels
  cx.fillStyle = 'rgba(160,180,220,0.3)';
  cx.font = "7px 'Space Mono',monospace";
  cx.textAlign = 'center';
  [-180, -120, -60, 0, 60, 120, 180].forEach(lon => {
    cx.fillText(`${lon}°`, margin.left + (lon + 180) / 360 * pW, H - 3);
  });
  cx.textAlign = 'right';
  [-60, -30, 0, 30, 60].forEach(lat => {
    cx.fillText(`${lat}°`, margin.left - 4, margin.top + (90 - lat) / 180 * pH + 3);
  });
}

function renderTimingChart(data) {
  const dets = ['H1', 'L1', 'V1', 'K1'];
  const colors = [ACC, MER, RING, '#445568'];
  const labels = [], delays = [], barColors = [];
  dets.forEach((k, i) => {
    if (data.detectors[k]) {
      labels.push(k);
      delays.push(data.detectors[k].arrival_delay_ms);
      barColors.push(colors[i]);
    }
  });

  makeChart('c-timing', 'bar', {
    labels,
    datasets: [{
      data: delays, backgroundColor: barColors.map(c => c + '44'),
      borderColor: barColors, borderWidth: 1, borderRadius: 3,
    }],
  }, {
    responsive: true, maintainAspectRatio: false,
    indexAxis: 'y',
    plugins: { legend: { display: false }, tooltip: {
      callbacks: { label: ctx => `Δt = ${ctx.parsed.x.toFixed(3)} ms` },
      backgroundColor: 'rgba(8,12,20,0.92)', bodyFont: { family: MONO, size: 9 },
      bodyColor: SIG, cornerRadius: 4,
    }},
    scales: {
      x: { grid: { color: GRID }, border: { color: 'rgba(160,180,220,0.07)' },
           ticks: { color: TICK, font: { family: MONO, size: 7 } },
           title: { display: true, text: 'Arrival Delay [ms]', color: TICK, font: { size: 8, family: MONO } } },
      y: { grid: { display: false }, border: { color: 'rgba(160,180,220,0.07)' },
           ticks: { color: TICK, font: { family: MONO, size: 9, weight: 'bold' } } },
    },
  });
}

async function runParamSpace() {
  const btn = docId('btn-parspace');
  btn.classList.add('loading');
  btn.textContent = 'Computing...';

  const quantity = docId('ps-quantity').value;

  try {
    const res = await fetch('/parameter_space', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        m1_min: 10, m1_max: 100, m2_min: 10, m2_max: 100,
        grid_size: 30, distance: S.distance,
        s1z: S.s1z, s2z: S.s2z, quantity,
      }),
    });
    const data = await res.json();

    if (data.status === 'error') {
      console.error(data.error); return;
    }

    docId('ps-badge').textContent = `${quantity} · ${data.elapsed}s`;

    if (data.grid && data.grid[quantity]) {
      renderHeatmap('c-spacetime', data.grid[quantity],
        data.grid.m2_values, data.grid.m1_values, ironRgb, {
          xlabel: 'm₂ [M☉]', ylabel: 'm₁ [M☉]',
          markers: [{ x: S.m2, y: S.m1, color: '#fff', symbol: 'cross', label: 'Current' }],
        });
    }

    renderMassSpinDiagram();
    renderChirpMassContours(data.grid);

  } catch (e) {
    console.error('Param space error:', e);
  } finally {
    btn.classList.remove('loading');
    btn.textContent = 'Explore Space';
  }
}

function renderHeatmap(canvasId, data2D, xVals, yVals, colormap, opts = {}) {
  const cv = docId(canvasId);
  if (!cv || !data2D || !data2D.length) return;
  const cx = cv.getContext('2d');
  const W = cv.parentElement.offsetWidth - 30;
  const H = parseInt(cv.getAttribute('height')) || 200;
  const dpr = window.devicePixelRatio || 1;
  cv.style.width = W + 'px';
  cv.style.height = H + 'px';
  cv.width = W * dpr;
  cv.height = H * dpr;
  cx.scale(dpr, dpr);

  cx.fillStyle = '#0c1120';
  cx.fillRect(0, 0, W, H);

  const nRows = data2D.length, nCols = data2D[0].length;
  let minV = Infinity, maxV = -Infinity;
  for (let i = 0; i < nRows; i++)
    for (let j = 0; j < nCols; j++) {
      const v = data2D[i][j];
      if (v !== null && v !== undefined && isFinite(v)) {
        minV = Math.min(minV, v);
        maxV = Math.max(maxV, v);
      }
    }
  if (maxV === minV) maxV = minV + 1;

  const m = { left: 40, right: 20, top: 10, bottom: 28 };
  const pW = W - m.left - m.right;
  const pH = H - m.top - m.bottom;
  const cellW = pW / nCols, cellH = pH / nRows;

  for (let i = 0; i < nRows; i++) {
    for (let j = 0; j < nCols; j++) {
      const v = data2D[i][j];
      if (v === null || v === undefined || !isFinite(v)) continue;
      const norm = (v - minV) / (maxV - minV);
      const rgb = colormap(Math.max(0, Math.min(1, norm)));
      cx.fillStyle = `rgb(${rgb[0]},${rgb[1]},${rgb[2]})`;
      cx.fillRect(m.left + j * cellW, m.top + (nRows - 1 - i) * cellH, cellW + 1, cellH + 1);
    }
  }

  // Axes
  cx.strokeStyle = 'rgba(160,180,220,0.2)';
  cx.lineWidth = 1;
  cx.strokeRect(m.left, m.top, pW, pH);

  cx.fillStyle = 'rgba(160,180,220,0.4)';
  cx.font = "7px 'Space Mono',monospace";
  cx.textAlign = 'center';

  const nXT = Math.min(6, nCols);
  for (let i = 0; i < nXT; i++) {
    const idx = Math.round(i * (nCols - 1) / (nXT - 1));
    cx.fillText(xVals[idx].toFixed(0), m.left + idx * cellW + cellW / 2, H - 8);
  }
  cx.textAlign = 'right';
  const nYT = Math.min(6, nRows);
  for (let i = 0; i < nYT; i++) {
    const idx = Math.round(i * (nRows - 1) / (nYT - 1));
    cx.fillText(yVals[idx].toFixed(0), m.left - 5, m.top + (nRows - 1 - idx) * cellH + cellH / 2 + 3);
  }

  if (opts.xlabel) {
    cx.textAlign = 'center';
    cx.fillText(opts.xlabel, m.left + pW / 2, H - 0);
  }
  if (opts.ylabel) {
    cx.save(); cx.translate(8, m.top + pH / 2); cx.rotate(-Math.PI / 2);
    cx.textAlign = 'center'; cx.fillText(opts.ylabel, 0, 0); cx.restore();
  }

  // Markers
  if (opts.markers) {
    opts.markers.forEach(mk => {
      const mx = m.left + ((mk.x - xVals[0]) / (xVals[xVals.length - 1] - xVals[0])) * pW;
      const my = m.top + (1 - (mk.y - yVals[0]) / (yVals[yVals.length - 1] - yVals[0])) * pH;
      cx.strokeStyle = mk.color || '#fff';
      cx.lineWidth = 2;
      const s = 6;
      cx.beginPath();
      cx.moveTo(mx - s, my - s); cx.lineTo(mx + s, my + s);
      cx.moveTo(mx + s, my - s); cx.lineTo(mx - s, my + s);
      cx.stroke();
      if (mk.label) {
        cx.font = "7px 'Space Mono',monospace";
        cx.fillStyle = mk.color || '#fff';
        cx.textAlign = 'left';
        cx.fillText(mk.label, mx + 8, my + 3);
      }
    });
  }

  // Color bar
  const cbW = 8;
  const cbX = W - cbW - 5;
  for (let i = 0; i < pH; i++) {
    const norm = 1 - i / pH;
    const rgb = colormap(norm);
    cx.fillStyle = `rgb(${rgb[0]},${rgb[1]},${rgb[2]})`;
    cx.fillRect(cbX, m.top + i, cbW, 1);
  }
  cx.fillStyle = 'rgba(160,180,220,0.4)';
  cx.font = "6px 'Space Mono',monospace";
  cx.textAlign = 'right';
  cx.fillText(maxV.toPrecision(3), cbX - 2, m.top + 6);
  cx.fillText(minV.toPrecision(3), cbX - 2, m.top + pH);
}

function renderMassSpinDiagram(m1, m2, s1z, s2z) {
  const m1Min = 5, m1Max = 100, m2Min = 5, m2Max = 100;
  const sMin = -1, sMax = 1;

  destroyChart('c-mass-spin');
  const cv = docId('c-massspin');
  if (!cv) return;
  const cx = cv.getContext('2d');
  const W = cv.parentElement.offsetWidth - 30;
  const H = 200;
  const dpr = window.devicePixelRatio || 1;
  cv.style.width = W + 'px';
  cv.style.height = H + 'px';
  cv.width = W * dpr;
  cv.height = H * dpr;
  cx.scale(dpr, dpr);

  cx.fillStyle = '#0c1120';
  cx.fillRect(0, 0, W, H);

  const m = { left: 40, right: 15, top: 15, bottom: 28 };
  const pW = W - m.left - m.right, pH = H - m.top - m.bottom;

  // Grid
  cx.strokeStyle = GRID; cx.lineWidth = 0.5;
  for (let i = 0; i <= 4; i++) {
    const y = m.top + i * pH / 4;
    cx.beginPath(); cx.moveTo(m.left, y); cx.lineTo(m.left + pW, y); cx.stroke();
    const x = m.left + i * pW / 4;
    cx.beginPath(); cx.moveTo(x, m.top); cx.lineTo(x, m.top + pH); cx.stroke();
  }

  // Known BBH events
  const events = [
    { name: 'GW150914', m: 65, chi: -0.01 },
    { name: 'GW151226', m: 21.8, chi: 0.21 },
    { name: 'GW170104', m: 51.2, chi: -0.12 },
    { name: 'GW170814', m: 53.2, chi: 0.07 },
    { name: 'GW190521', m: 150, chi: 0.08 },
  ];

  const mMin = 10, mMax = 160, chiMin = -1, chiMax = 1;
  events.forEach(ev => {
    const x = m.left + (ev.m - mMin) / (mMax - mMin) * pW;
    const y = m.top + (chiMax - ev.chi) / (chiMax - chiMin) * pH;
    cx.beginPath(); cx.arc(x, y, 4, 0, Math.PI * 2);
    cx.fillStyle = 'rgba(160,180,220,0.25)'; cx.fill();
    cx.font = "6px 'Space Mono',monospace";
    cx.fillStyle = 'rgba(160,180,220,0.35)';
    cx.textAlign = 'left';
    cx.fillText(ev.name, x + 6, y + 3);
  });

  // Current binary
  const curM = S.m1 + S.m2;
  const chi_eff = (S.m1 * S.s1z + S.m2 * S.s2z) / curM;
  const cx_ = m.left + (curM - mMin) / (mMax - mMin) * pW;
  const cy_ = m.top + (chiMax - chi_eff) / (chiMax - chiMin) * pH;
  const g = cx.createRadialGradient(cx_, cy_, 0, cx_, cy_, 12);
  g.addColorStop(0, 'rgba(123,167,200,0.3)');
  g.addColorStop(1, 'transparent');
  cx.fillStyle = g;
  cx.beginPath(); cx.arc(cx_, cy_, 12, 0, Math.PI * 2); cx.fill();
  cx.beginPath(); cx.arc(cx_, cy_, 5, 0, Math.PI * 2);
  cx.fillStyle = ACC; cx.fill();
  cx.font = "bold 7px 'Space Mono',monospace";
  cx.fillStyle = ACC;
  cx.fillText('CURRENT', cx_ + 8, cy_ + 3);

  // Axis labels
  cx.fillStyle = 'rgba(160,180,220,0.4)';
  cx.font = "7px 'Space Mono',monospace";
  cx.textAlign = 'center';
  cx.fillText('Total Mass [M☉]', m.left + pW / 2, H - 3);
  cx.save(); cx.translate(8, m.top + pH / 2); cx.rotate(-Math.PI / 2);
  cx.fillText('χ_eff', 0, 0); cx.restore();

  // Tick values
  [mMin, 50, 100, mMax].forEach(v => {
    const x = m.left + (v - mMin) / (mMax - mMin) * pW;
    cx.textAlign = 'center';
    cx.fillText(v, x, H - 12);
  });
  cx.textAlign = 'right';
  [-1, -0.5, 0, 0.5, 1].forEach(v => {
    const y = m.top + (chiMax - v) / (chiMax - chiMin) * pH;
    cx.fillText(v.toFixed(1), m.left - 4, y + 3);
  });
}

function renderChirpMassContours() {
  const cv = docId('c-skyloc');
  if (!cv) return;
  const cx = cv.getContext('2d');
  const W = cv.parentElement.offsetWidth - 30;
  const H = 200;
  const dpr = window.devicePixelRatio || 1;
  cv.style.width = W + 'px';
  cv.style.height = H + 'px';
  cv.width = W * dpr;
  cv.height = H * dpr;
  cx.scale(dpr, dpr);

  cx.fillStyle = '#0c1120';
  cx.fillRect(0, 0, W, H);

  const m = { left: 40, right: 15, top: 10, bottom: 28 };
  const pW = W - m.left - m.right, pH = H - m.top - m.bottom;

  // Draw chirp mass contour lines on (m1, m2) plane
  const m1Min = 5, m1Max = 80, m2Min = 5, m2Max = 80;
  const mcValues = [5, 10, 15, 20, 25, 30, 40, 50];

  mcValues.forEach((mc, idx) => {
    cx.beginPath();
    cx.strokeStyle = `rgba(123,167,200,${0.1 + idx * 0.05})`;
    cx.lineWidth = 0.8;
    for (let m1 = m1Min; m1 <= m1Max; m1 += 0.5) {
      for (let m2 = m2Min; m2 <= m1; m2 += 0.5) {
        const M = m1 + m2;
        const eta = (m1 * m2) / (M * M);
        const mcCalc = M * Math.pow(eta, 0.6);
        if (Math.abs(mcCalc - mc) < 0.5) {
          const x = m.left + (m1 - m1Min) / (m1Max - m1Min) * pW;
          const y = m.top + (m2Max - m2) / (m2Max - m2Min) * pH;
          cx.lineTo(x, y);
          break;
        }
      }
    }
    cx.stroke();

    // Label
    const lm1 = Math.max(mc * 2, m1Min + 5);
    const lx = m.left + (lm1 - m1Min) / (m1Max - m1Min) * pW;
    cx.font = "6px 'Space Mono',monospace";
    cx.fillStyle = 'rgba(123,167,200,0.3)';
    cx.textAlign = 'left';
    cx.fillText(`Mc=${mc}`, lx, m.top + 12 + idx * 10);
  });

  // Current point
  const curX = m.left + (S.m1 - m1Min) / (m1Max - m1Min) * pW;
  const curY = m.top + (m2Max - S.m2) / (m2Max - m2Min) * pH;
  cx.beginPath(); cx.arc(curX, curY, 5, 0, Math.PI * 2);
  cx.fillStyle = ACC; cx.fill();

  // Axes
  cx.fillStyle = 'rgba(160,180,220,0.4)';
  cx.font = "7px 'Space Mono',monospace";
  cx.textAlign = 'center';
  cx.fillText('m₁ [M☉]', m.left + pW / 2, H - 3);
  cx.save(); cx.translate(8, m.top + pH / 2); cx.rotate(-Math.PI / 2);
  cx.fillText('m₂ [M☉]', 0, 0); cx.restore();
}

function setupAudio(base64, duration) {
  if (S.audioEl) { S.audioEl.pause(); S.audioEl = null; }
  S.playing = false;
  updateAudioUI();

  docId('audio-sub').textContent = duration ? `${duration.toFixed(1)}s · Freq-shifted for audibility` : '—';

  if (base64) {
    S.audioEl = new Audio(`data:audio/wav;base64,${base64}`);
    S.audioEl.onended = () => { S.playing = false; updateAudioUI(); };
  }
}

function toggleAudio() {
  if (!S.audioEl) return;
  if (S.playing) {
    S.audioEl.pause();
    S.playing = false;
  } else {
    S.audioEl.currentTime = 0;
    S.audioEl.play().then(() => { S.playing = true; updateAudioUI(); }).catch(e => console.error(e));
  }
  updateAudioUI();
}

function updateAudioUI() {
  const btn = docId('btn-audio');
  const wv = docId('wv-bars');
  btn.textContent = S.playing ? 'Stop' : 'Play';
  btn.classList.toggle('playing', S.playing);
  wv.classList.toggle('paused', !S.playing);
}

function loadPreset() {
  S.m1 = 36; S.m2 = 29; 
  S.s1z = 0; S.s2z = 0;
  S.distance = 410; S.inclination = 0;
  
  docId('sl-m1').value = 36; docId('v-m1').textContent = '36.0 M☉';
  docId('sl-m2').value = 29; docId('v-m2').textContent = '29.0 M☉';
  docId('sl-s1z').value = 0; docId('v-s1z').textContent = '0.00';
  docId('sl-s2z').value = 0; docId('v-s2z').textContent = '0.00';
  docId('sl-dist').value = 410; docId('v-dist').textContent = '410 Mpc';
  docId('sl-inc').value = 0; docId('v-inc').textContent = '0.00 rad';
  
  updateDerivedPhysics();
}

async function fetchDetectorStatus() {
  try {
    const res = await fetch('/detector_status');
    const data = await res.json();
    ['H1', 'L1', 'V1', 'K1'].forEach(det => {
      const chip = docId('chip-' + det);
      if (chip) {
        chip.className = Object.prototype.hasOwnProperty.call(data, det) && data[det] ? 'chip on' : 'chip off';
      }
    });
    
    const isOnline = Boolean(data.H1 || data.L1 || data.V1 || data.K1);
    const dot = docId('live-dot-icon');
    const txt = docId('live-text');
    if (dot && txt) {
      if (isOnline) {
        txt.textContent = 'ONLINE';
        dot.style.background = 'var(--live)';
        dot.style.boxShadow = '0 0 5px var(--live)';
        dot.style.animation = 'livepulse 2.8s ease-in-out infinite';
      } else {
        txt.textContent = 'OFFLINE';
        dot.style.background = 'var(--alert)';
        dot.style.boxShadow = '0 0 5px var(--alert)';
        dot.style.animation = 'none';
      }
    }
  } catch (e) {
    console.error('Failed to fetch detector status:', e);
  }
}

function init() {
  fetchDetectorStatus();
  initSliders();
  updateDerivedPhysics();

  docId('btn-generate').addEventListener('click', runAnalysis);
  docId('btn-preset').addEventListener('click', loadPreset);
  docId('btn-audio').addEventListener('click', toggleAudio);
  docId('btn-compare').addEventListener('click', runCompare);
  docId('btn-estimate').addEventListener('click', runEstimate);
  docId('btn-network').addEventListener('click', runNetwork);
  docId('btn-parspace').addEventListener('click', runParamSpace);

  // Render initial decorative canvases
  renderMassSpinDiagram();
  renderChirpMassContours(null);
  renderNetworkMapStatic();
}

function renderNetworkMapStatic() {
  // Render initial network map without data
  const cv = docId('c-netmap');
  if (!cv) return;
  const cx = cv.getContext('2d');
  const W = cv.width = cv.parentElement.offsetWidth - 30;
  const H = cv.height = 230;

  cx.fillStyle = '#080c14'; cx.fillRect(0, 0, W, H);

  const m = { left: 30, right: 15, top: 10, bottom: 20 };
  const pW = W - m.left - m.right, pH = H - m.top - m.bottom;

  cx.strokeStyle = 'rgba(160,180,220,0.06)'; cx.lineWidth = 0.5;
  for (let lat = -60; lat <= 60; lat += 30) {
    const y = m.top + (90 - lat) / 180 * pH;
    cx.beginPath(); cx.moveTo(m.left, y); cx.lineTo(m.left + pW, y); cx.stroke();
  }
  for (let lon = -180; lon <= 180; lon += 60) {
    const x = m.left + (lon + 180) / 360 * pW;
    cx.beginPath(); cx.moveTo(x, m.top); cx.lineTo(x, m.top + pH); cx.stroke();
  }

  cx.strokeStyle = 'rgba(160,180,220,0.15)'; cx.lineWidth = 1;
  cx.strokeRect(m.left, m.top, pW, pH);

  const dets = {
    H1: { lat: 46.455, lon: -119.408, color: ACC },
    L1: { lat: 30.563, lon: -90.774, color: MER },
    V1: { lat: 43.631, lon: 10.505, color: RING },
    K1: { lat: 36.410, lon: 137.306, color: '#ffd166' },
  };
  Object.entries(dets).forEach(([k, d]) => {
    const x = m.left + (d.lon + 180) / 360 * pW;
    const y = m.top + (90 - d.lat) / 180 * pH;
    cx.beginPath(); cx.arc(x, y, 4, 0, Math.PI * 2);
    cx.fillStyle = d.color; cx.fill();
    cx.font = "bold 8px 'Space Mono',monospace";
    cx.fillStyle = d.color;
    cx.fillText(k, x + 7, y - 5);
  });

  cx.fillStyle = 'rgba(160,180,220,0.3)';
  cx.font = "7px 'Space Mono',monospace";
  cx.textAlign = 'center';
  [-180, -120, -60, 0, 60, 120, 180].forEach(lon => {
    cx.fillText(`${lon}°`, m.left + (lon + 180) / 360 * pW, H - 3);
  });
}

init();
