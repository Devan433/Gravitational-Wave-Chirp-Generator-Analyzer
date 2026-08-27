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

    updateKPIs(data.physics);
    updateSidebarPhysics(data.physics);
    renderStrainChart(data.waveform);
    renderQTransform(data.spectrogram, data.freq_track);
    renderSNRChart(data.snr);
    renderFreqChart(data.waveform);

  } catch (e) {
    showError('analysis-await', 'Connection failed: ' + e.message);
  } finally {
    btn.classList.remove('loading');
    btn.innerHTML = 'Generate Waveform';
  }
}

function showError(containerId, msg) {
  const el = docId(containerId);
  if (!el) {
    console.error(msg);
    alert("Error: " + msg);
    return;
  }
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


function init() {
  initSliders();
  updateDerivedPhysics();

  docId('btn-generate').addEventListener('click', runAnalysis);

}

init();
