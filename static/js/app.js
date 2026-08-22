/**
 * static/js/app.js
 * Main application logic: upload, job polling, result rendering, tabs.
 * Fixes: job_id propagation, freeze guards, PDF light theme, debounce.
 */

// ─── Toast System ─────────────────────────────────────────────────────────────
const TOAST_ICONS = {
  success: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>`,
  error:   `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>`,
  info:    `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>`,
  warning: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>`,
};

function toast(type, title, message = '', duration = 4500) {
  const container = document.getElementById('toast-container');
  if (!container) return;
  const el = document.createElement('div');
  el.className = `toast toast-${type}`;
  el.innerHTML = `
    <span class="toast-icon">${TOAST_ICONS[type] || TOAST_ICONS.info}</span>
    <div class="toast-content">
      <div class="toast-title">${title}</div>
      ${message ? `<div class="toast-msg">${message}</div>` : ''}
    </div>
  `;
  container.appendChild(el);
  setTimeout(() => {
    el.classList.add('hiding');
    el.addEventListener('animationend', () => el.remove(), { once: true });
  }, duration);
}

// ─── Global State ─────────────────────────────────────────────────────────────
let currentJobId = null;
let currentShareToken = null;
let pollInterval = null;
let currentResults = null;

// ─── Init ─────────────────────────────────────────────────────────────────────
window.addEventListener('DOMContentLoaded', () => {
  const saved = sessionStorage.getItem('autolysis_result');
  if (saved) {
    try {
      const data = JSON.parse(saved);
      currentJobId = data.job_id || null;
      currentShareToken = data.share_token || null;
      _applyResults(data);
    } catch {
      sessionStorage.removeItem('autolysis_result');
    }
  }
  initUploadZone();
  initTabs();
});

// ─── Upload Zone ──────────────────────────────────────────────────────────────
function initUploadZone() {
  const zone = document.getElementById('upload-zone');
  const fileInput = document.getElementById('file-input');
  if (!zone || !fileInput) return;

  zone.addEventListener('click', () => fileInput.click());
  zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('dragover'); });
  zone.addEventListener('dragleave', () => zone.classList.remove('dragover'));
  zone.addEventListener('drop', e => {
    e.preventDefault();
    zone.classList.remove('dragover');
    if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
  });
  fileInput.addEventListener('change', () => {
    if (fileInput.files.length) handleFile(fileInput.files[0]);
    fileInput.value = '';
  });
}

const ALLOWED_EXTS = ['csv', 'xlsx', 'xls', 'json'];
const MAX_SIZE_MB = 50;

async function handleFile(file) {
  const ext = file.name.split('.').pop().toLowerCase();
  if (!ALLOWED_EXTS.includes(ext)) {
    toast('error', 'Unsupported format', `Allowed: ${ALLOWED_EXTS.join(', ')}`);
    return;
  }
  if (file.size > MAX_SIZE_MB * 1024 * 1024) {
    toast('error', 'File too large', `Maximum size is ${MAX_SIZE_MB}MB`);
    return;
  }

  setSection('progress');
  resetProgress();

  const formData = new FormData();
  formData.append('file', file);

  try {
    const res = await fetch('/submit', { method: 'POST', body: formData });
    const data = await res.json();
    if (!res.ok || data.error) throw new Error(data.error || 'Upload failed');

    currentJobId = data.job_id;
    window.currentJobId = currentJobId; // expose globally for ChartsModule
    toast('info', 'Analysis started', 'Your file is being processed...');
    startPolling(currentJobId);
  } catch (err) {
    toast('error', 'Upload failed', err.message);
    setSection('hero');
  }
}

// ─── Job Polling ──────────────────────────────────────────────────────────────
function startPolling(jobId) {
  clearInterval(pollInterval);
  pollInterval = setInterval(() => pollStatus(jobId), 1800);
}

const STAGES = [
  { threshold: 20 }, { threshold: 45 }, { threshold: 65 },
  { threshold: 80 }, { threshold: 95 }, { threshold: 100 },
];

async function pollStatus(jobId) {
  try {
    const res = await fetch(`/status/${jobId}`);
    const job = await res.json();
    updateProgress(job.progress || 0, job.progress_label || '');
    if (job.status === 'done') {
      clearInterval(pollInterval);
      await fetchAndShowResult(jobId);
    } else if (job.status === 'error') {
      clearInterval(pollInterval);
      toast('error', 'Analysis failed', job.error_msg || 'Unknown error');
      setSection('hero');
    }
  } catch { /* network hiccup — keep polling */ }
}

function updateProgress(pct, label) {
  const bar = document.getElementById('progress-bar');
  const labelEl = document.getElementById('progress-label');
  const pctEl = document.getElementById('progress-pct');
  if (bar) bar.style.width = `${pct}%`;
  if (labelEl && label) labelEl.textContent = label;
  if (pctEl) pctEl.textContent = `${pct}%`;

  STAGES.forEach((stage, i) => {
    const dot = document.getElementById(`stage-${i}`);
    if (!dot) return;
    const parent = dot.closest('.stage');
    if (pct >= stage.threshold) {
      parent.classList.remove('active');
      parent.classList.add('done');
    } else if (pct >= (STAGES[i - 1]?.threshold || 0)) {
      parent.classList.add('active');
      parent.classList.remove('done');
    }
  });
}

function resetProgress() {
  const bar = document.getElementById('progress-bar');
  if (bar) bar.style.width = '0%';
  const labelEl = document.getElementById('progress-label');
  if (labelEl) labelEl.textContent = 'Starting analysis...';
  const pctEl = document.getElementById('progress-pct');
  if (pctEl) pctEl.textContent = '0%';
  document.querySelectorAll('.stage').forEach(s => s.classList.remove('active', 'done'));
  document.querySelector('.stage')?.classList.add('active');
}

// ─── Fetch & Show Results ─────────────────────────────────────────────────────
async function fetchAndShowResult(jobId) {
  try {
    const res = await fetch(`/result/${jobId}`);
    const data = await res.json();
    if (!res.ok || data.error) throw new Error(data.error || 'Failed to load results');

    currentResults = data;
    currentShareToken = data.share_token;
    currentJobId = jobId;
    window.currentJobId = jobId; // expose for ChartsModule

    try { sessionStorage.setItem('autolysis_result', JSON.stringify(data)); } catch {}

    _applyResults(data);
    toast('success', 'Analysis complete!', `${data.filename} has been analyzed.`);
  } catch (err) {
    toast('error', 'Failed to load results', err.message);
    setSection('hero');
  }
}

function _applyResults(data) {
  // Init chat
  if (window.ChatModule) window.ChatModule.init(data.job_id || currentJobId);
  // Init charts with job_id
  if (window.ChartsModule) window.ChartsModule.setJobId(data.job_id || currentJobId);
  showResults(data);
}

// ─── Render Results ───────────────────────────────────────────────────────────
function showResults(data) {
  setSection('results');

  const filenameEl = document.getElementById('result-filename');
  if (filenameEl) filenameEl.textContent = data.filename || 'Analysis Report';

  const metaEl = document.getElementById('result-meta');
  if (metaEl && data.created_at) {
    metaEl.textContent = `Analyzed on ${new Date(data.created_at).toLocaleString()}`;
  }

  if (data.share_token) {
    const shareInput = document.getElementById('share-url-input');
    if (shareInput) shareInput.value = `${window.location.origin}/report/${data.share_token}`;
  }

  // Markdown report
  const reportEl = document.getElementById('tab-report-content');
  if (reportEl && data.readme) {
    try { reportEl.innerHTML = marked.parse(data.readme); } catch { reportEl.textContent = data.readme; }
  }

  // Charts
  if (window.ChartsModule && data.charts) {
    window.ChartsModule.renderAll(data.charts);
  }

  // Stats table
  if (data.column_info) renderStatsTable(data.column_info);

  // Anomaly section
  if (data.anomaly_data) renderAnomalySection(data.anomaly_data);

  // Show chat FAB
  document.getElementById('chat-fab')?.classList.remove('hidden');

  // Show "New Session" button
  document.getElementById('new-session-btn')?.style &&
    (document.getElementById('new-session-btn').style.display = 'flex');

  document.getElementById('results-section')?.classList.add('animate-fade-up');
}

// ─── Stats Table ──────────────────────────────────────────────────────────────
function renderStatsTable(columnInfo) {
  const container = document.getElementById('stats-table-body');
  if (!container) return;
  container.innerHTML = '';

  Object.entries(columnInfo).forEach(([col, info]) => {
    const tr = document.createElement('tr');
    const numInfo = info.is_numeric
      ? `<span style="color:var(--clr-text-muted)">μ=${fmt(info.mean)}</span>&nbsp;<span style="color:var(--clr-text-dim)">σ=${fmt(info.std)}</span>`
      : (info.top_values ? Object.keys(info.top_values).slice(0, 3).map(v => `<code>${v}</code>`).join(', ') : '—');

    tr.innerHTML = `
      <td>
        <div class="col-name-cell" data-col="${col}" title="Click for AI deep-dive analysis">
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>
          ${col}
        </div>
      </td>
      <td><span class="badge ${info.is_numeric ? 'badge-sky' : info.is_datetime ? 'badge-mint' : 'badge-slate'}">${info.dtype}</span></td>
      <td>${(info.n_unique || 0).toLocaleString()}</td>
      <td>${(info.n_missing || 0) > 0 ? `<span style="color:var(--clr-warning)">${info.n_missing} (${info.pct_missing}%)</span>` : '<span style="color:var(--clr-mint)">None</span>'}</td>
      <td style="font-size:0.8rem">${numInfo}</td>
    `;
    container.appendChild(tr);
  });

  // Attach click handlers — fixes the onclick not working issue
  container.querySelectorAll('.col-name-cell').forEach(el => {
    el.addEventListener('click', () => {
      const col = el.dataset.col;
      if (col && window.ChartsModule) window.ChartsModule.openColumnModal(col);
    });
  });
}

// ─── Anomaly Section ──────────────────────────────────────────────────────────
function renderAnomalySection(anomalyData) {
  const container = document.getElementById('anomaly-summary');
  if (!container) return;

  if (anomalyData.error) {
    container.innerHTML = `<p class="text-muted">${anomalyData.error}</p>`;
    return;
  }

  const count = anomalyData.anomaly_count || 0;
  const pct = anomalyData.anomaly_pct || 0;
  const contam = anomalyData.contamination_used || '—';

  container.innerHTML = `
    <div style="display:flex; gap:20px; flex-wrap:wrap; margin-bottom:24px;">
      <div class="card card-pad" style="flex:1; min-width:160px; text-align:center;">
        <div style="font-size:2.2rem; font-weight:800; color:var(--clr-error)">${count.toLocaleString()}</div>
        <div class="text-muted text-sm mt-sm">Anomalies Detected</div>
      </div>
      <div class="card card-pad" style="flex:1; min-width:160px; text-align:center;">
        <div style="font-size:2.2rem; font-weight:800; color:var(--clr-peach)">${pct}%</div>
        <div class="text-muted text-sm mt-sm">of Dataset</div>
      </div>
      <div class="card card-pad" style="flex:1; min-width:160px; text-align:center;">
        <div style="font-size:1.3rem; font-weight:700; color:var(--clr-sky); font-family:var(--font-mono)">${contam}</div>
        <div class="text-muted text-sm mt-sm">Contamination Rate</div>
      </div>
    </div>
    <p class="text-muted text-sm">
      <strong style="color:var(--clr-peach)">Isolation Forest</strong> model detected rows that deviate
      significantly from the normal data distribution. These rows appear highlighted in the
      <strong>Anomaly Detection</strong> chart in the Charts tab.
    </p>
  `;
}

// ─── Tab Management ───────────────────────────────────────────────────────────
function initTabs() {
  document.querySelectorAll('[data-tab]').forEach(btn => {
    btn.addEventListener('click', () => switchTab(btn.dataset.tab));
  });
}

function switchTab(tabId) {
  document.querySelectorAll('[data-tab]').forEach(b => b.classList.toggle('active', b.dataset.tab === tabId));
  document.querySelectorAll('[data-panel]').forEach(p => p.classList.toggle('active', p.dataset.panel === tabId));
}

// ─── Section Management ───────────────────────────────────────────────────────
function setSection(name) {
  ['hero', 'progress', 'results'].forEach(s => {
    const el = document.getElementById(`${s}-section`);
    if (el) el.classList.toggle('hidden', s !== name);
  });
}

// ─── New Session ──────────────────────────────────────────────────────────────
function startNewSession() {
  if (!confirm('Clear current analysis and start fresh?')) return;
  sessionStorage.removeItem('autolysis_result');
  currentJobId = null;
  window.currentJobId = null;
  currentShareToken = null;
  currentResults = null;
  clearInterval(pollInterval);
  document.getElementById('chat-fab')?.classList.add('hidden');
  document.getElementById('new-session-btn') &&
    (document.getElementById('new-session-btn').style.display = 'none');
  if (window.ChatModule) window.ChatModule.reset();
  setSection('hero');
}

// ─── Share Link ────────────────────────────────────────────────────────────────
function copyShareLink() {
  const input = document.getElementById('share-url-input');
  if (!input || !input.value) return toast('warning', 'No share link yet', 'Run analysis first.');
  navigator.clipboard.writeText(input.value).then(() => {
    toast('success', 'Link copied!', 'Share it with your team.');
  });
}

// ─── PDF Download — Inline-styled clone (reliable on dark-theme apps) ────────
let _pdfBusy = false;

function _applyPdfStyles(wrapper) {
  // Must set every element's styles explicitly — html2canvas ignores !important CSS
  const s = (el, css) => Object.assign(el.style, css);
  const base = { color: '#1a1a1a', background: 'transparent', webkitTextFillColor: '#1a1a1a' };

  wrapper.querySelectorAll('h1').forEach(el => s(el, { ...base, fontSize: '22px', fontWeight: '800', marginBottom: '14px', marginTop: '0', lineHeight: '1.2' }));
  wrapper.querySelectorAll('h2').forEach(el => s(el, { ...base, fontSize: '16px', fontWeight: '700', marginTop: '26px', marginBottom: '10px', borderBottom: '2px solid #d0d8e0', paddingBottom: '6px', color: '#1a3550' }));
  wrapper.querySelectorAll('h3').forEach(el => s(el, { ...base, fontSize: '13px', fontWeight: '600', marginTop: '16px', marginBottom: '6px', color: '#2a4560' }));
  wrapper.querySelectorAll('p').forEach(el => s(el, { ...base, fontSize: '13px', lineHeight: '1.7', marginBottom: '10px' }));
  wrapper.querySelectorAll('strong').forEach(el => s(el, { color: '#0a1a2a', fontWeight: '700', background: 'transparent', webkitTextFillColor: '#0a1a2a' }));
  wrapper.querySelectorAll('em').forEach(el => s(el, { color: '#2a4060', fontStyle: 'italic', background: 'transparent', webkitTextFillColor: '#2a4060' }));
  wrapper.querySelectorAll('a').forEach(el => s(el, { color: '#1a6aa0', background: 'transparent', webkitTextFillColor: '#1a6aa0' }));
  wrapper.querySelectorAll('code').forEach(el => s(el, { background: '#f0f4f8', color: '#2a3a4a', fontFamily: 'Courier New, monospace', fontSize: '11px', padding: '1px 5px', borderRadius: '3px', webkitTextFillColor: '#2a3a4a' }));
  wrapper.querySelectorAll('pre').forEach(el => s(el, { background: '#f0f4f8', padding: '12px 16px', borderRadius: '6px', overflow: 'auto', marginBottom: '12px' }));
  wrapper.querySelectorAll('hr').forEach(el => s(el, { border: 'none', borderTop: '1px solid #d0d8e0', margin: '20px 0' }));
  wrapper.querySelectorAll('ul, ol').forEach(el => s(el, { ...base, fontSize: '13px', paddingLeft: '22px', marginBottom: '10px' }));
  wrapper.querySelectorAll('li').forEach(el => s(el, { ...base, fontSize: '13px', marginBottom: '4px', lineHeight: '1.6' }));
  wrapper.querySelectorAll('blockquote').forEach(el => s(el, { borderLeft: '3px solid #d0d8e0', paddingLeft: '16px', color: '#3a5060', margin: '12px 0', background: 'transparent' }));
  wrapper.querySelectorAll('table').forEach(el => s(el, { width: '100%', borderCollapse: 'collapse', marginBottom: '16px', fontSize: '12px' }));
  wrapper.querySelectorAll('th').forEach(el => s(el, { background: '#edf2f7', color: '#1a3550', fontWeight: '600', padding: '8px 12px', border: '1px solid #c8d5e0', textAlign: 'left', webkitTextFillColor: '#1a3550' }));
  wrapper.querySelectorAll('td').forEach(el => s(el, { ...base, fontSize: '12px', padding: '7px 12px', border: '1px solid #c8d5e0' }));
  wrapper.querySelectorAll('tr:nth-child(even) td').forEach(el => s(el, { background: '#f8fafc' }));
}

async function downloadPDF() {
  if (_pdfBusy) return;
  if (!currentResults || !currentResults.readme) {
    return toast('warning', 'No report yet', 'Run analysis first.');
  }

  const btn = document.getElementById('download-pdf-btn');
  _pdfBusy = true;
  if (btn) { btn.disabled = true; btn.textContent = 'Generating...'; }

  let wrapper = null;
  try {
    // Parse the markdown fresh — don't clone the dark-themed DOM element
    const htmlContent = (typeof marked !== 'undefined')
      ? marked.parse(currentResults.readme)
      : currentResults.readme;

    // Build a completely clean, isolated container with explicit white background
    wrapper = document.createElement('div');
    wrapper.id = '_pdf_tmp';
    Object.assign(wrapper.style, {
      position: 'absolute',
      left: '-9999px',
      top: '0',
      width: '780px',
      background: '#ffffff',
      color: '#1a1a1a',
      fontFamily: 'Arial, Helvetica, sans-serif',
      fontSize: '13px',
      lineHeight: '1.7',
      padding: '48px 56px',
    });
    wrapper.innerHTML = htmlContent;

    // Apply inline styles to every element — makes html2canvas render correctly
    _applyPdfStyles(wrapper);

    document.body.appendChild(wrapper);

    const filename = (currentResults.filename || 'report').replace(/\.[^/.]+$/, '');
    const opt = {
      margin: [12, 12, 12, 12],
      filename: `autolysis_${filename}.pdf`,
      image: { type: 'jpeg', quality: 0.98 },
      html2canvas: {
        scale: 2,
        useCORS: true,
        backgroundColor: '#ffffff',
        logging: false,
        removeContainer: true,
      },
      jsPDF: { unit: 'mm', format: 'a4', orientation: 'portrait' },
      pagebreak: { mode: ['avoid-all', 'css', 'legacy'] },
    };

    await html2pdf().set(opt).from(wrapper).save();
    toast('success', 'PDF downloaded!', 'Check your downloads folder.');
  } catch (err) {
    toast('error', 'PDF generation failed', err.message);
  } finally {
    wrapper?.remove();
    _pdfBusy = false;
    if (btn) {
      btn.disabled = false;
      btn.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg> Download PDF`;
    }
  }
}


// ─── Helpers ──────────────────────────────────────────────────────────────────
function fmt(val) {
  if (val === null || val === undefined) return '—';
  return typeof val === 'number'
    ? val.toLocaleString(undefined, { maximumFractionDigits: 3 })
    : val;
}

// Expose globals
window.startNewSession = startNewSession;
window.copyShareLink = copyShareLink;
window.downloadPDF = downloadPDF;
window.switchTab = switchTab;
