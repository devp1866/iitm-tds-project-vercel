/**
 * static/js/charts.js
 * Plotly chart rendering, fullscreen expand, and column drill-down modal.
 * Fixes: fullscreen view, debounced column modal, job_id propagation.
 */

(function () {
  'use strict';

  let _charts = [];
  let _jobId = null;
  let _colModalBusy = false; // prevent double-click freeze
  let _expandModal = null;   // fullscreen chart modal element

  // ─── Plotly Dark Config ──────────────────────────────────────────────────────
  const PLOTLY_CONFIG = {
    responsive: true,
    displayModeBar: true,
    displaylogo: false,
    modeBarButtonsToRemove: ['lasso2d', 'select2d'],
    toImageButtonOptions: { format: 'png', scale: 2 },
  };

  const BASE_LAYOUT = {
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    font: { family: 'Inter, sans-serif', color: '#e4f2fb', size: 11 },
    margin: { l: 48, r: 24, t: 12, b: 48 },
    xaxis: { gridcolor: 'rgba(73,101,128,0.3)', zerolinecolor: 'rgba(73,101,128,0.3)', automargin: true },
    yaxis: { gridcolor: 'rgba(73,101,128,0.3)', zerolinecolor: 'rgba(73,101,128,0.3)', automargin: true },
    legend: { font: { color: '#7bafc8' }, orientation: 'h', y: -0.2 },
    hoverlabel: { bgcolor: '#0f2030', bordercolor: '#496580', font: { color: '#e4f2fb' } },
    colorway: ['#BADFFF', '#BAFFF5', '#FFDBBB', '#496580', '#7bafc8', '#a8d8ff', '#d4fff9'],
  };

  const TYPE_LABELS = {
    heatmap: '🔥 Heatmap',
    histogram: '📊 Distribution',
    timeseries: '📈 Time Series',
    bar: '📊 Bar Chart',
    scatter: '🔵 Scatter',
    boxplot: '📦 Box Plot',
    anomaly_scatter: '⚠️ Anomaly',
    cluster_scatter: '🔵 Clusters',
    pie: '🥧 Pie',
    corr_bar: '🔗 Correlation',
  };

  // ─── Create fullscreen modal once ───────────────────────────────────────────
  function _ensureExpandModal() {
    if (_expandModal) return;
    _expandModal = document.createElement('div');
    _expandModal.id = 'chart-expand-overlay';
    _expandModal.style.cssText = `
      position:fixed; inset:0; z-index:500;
      background:rgba(10,21,32,0.95); backdrop-filter:blur(12px);
      display:none; flex-direction:column; align-items:center; justify-content:center; padding:24px;
    `;
    _expandModal.innerHTML = `
      <div style="width:100%;max-width:1200px;height:85vh;display:flex;flex-direction:column;">
        <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:12px;">
          <span id="expand-title" style="font-size:1rem;font-weight:700;color:var(--clr-text)"></span>
          <button id="expand-close-btn" style="
            background:var(--clr-bg-card); border:1px solid var(--clr-border); border-radius:var(--radius-md);
            color:var(--clr-text-muted); padding:8px 14px; cursor:pointer; font-size:0.875rem;
            display:flex; align-items:center; gap:6px;
          ">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>
            </svg>
            Close
          </button>
        </div>
        <div id="expand-chart-mount" style="flex:1;background:var(--clr-bg-card);border:1px solid var(--clr-border);border-radius:var(--radius-lg);overflow:hidden;"></div>
      </div>
    `;
    document.body.appendChild(_expandModal);

    _expandModal.addEventListener('click', e => {
      if (e.target === _expandModal) _closeExpand();
    });
    document.getElementById('expand-close-btn').addEventListener('click', _closeExpand);
    document.addEventListener('keydown', e => {
      if (e.key === 'Escape') _closeExpand();
    });
  }

  function _openExpand(chart) {
    _ensureExpandModal();
    _expandModal.style.display = 'flex';
    document.getElementById('expand-title').textContent = chart.title || 'Chart';

    const mount = document.getElementById('expand-chart-mount');
    mount.innerHTML = '';

    const layout = _mergeLayout(chart.figure?.layout, {
      height: undefined,
      autosize: true,
      margin: { l: 60, r: 40, t: 30, b: 60 },
    });
    setTimeout(() => {
      Plotly.newPlot(mount, chart.figure?.data || [], layout, {
        ...PLOTLY_CONFIG,
        displayModeBar: true,
      });
    }, 50);
  }

  function _closeExpand() {
    if (!_expandModal) return;
    _expandModal.style.display = 'none';
    const mount = document.getElementById('expand-chart-mount');
    if (mount) { Plotly.purge(mount); mount.innerHTML = ''; }
  }

  // ─── Merge layout with BASE_LAYOUT ──────────────────────────────────────────
  function _mergeLayout(figLayout, overrides = {}) {
    const merged = Object.assign({}, BASE_LAYOUT, figLayout || {}, {
      paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor: 'rgba(0,0,0,0)',
      title: undefined, // title is shown in card header
      font: BASE_LAYOUT.font,
      hoverlabel: BASE_LAYOUT.hoverlabel,
    }, overrides);

    // Always force axis theme
    merged.xaxis = Object.assign({}, BASE_LAYOUT.xaxis, figLayout?.xaxis || {});
    merged.yaxis = Object.assign({}, BASE_LAYOUT.yaxis, figLayout?.yaxis || {});
    return merged;
  }

  // ─── Render all charts ───────────────────────────────────────────────────────
  function renderAll(charts) {
    _charts = charts;
    const grid = document.getElementById('charts-grid');
    if (!grid) return;
    grid.innerHTML = '';

    if (!charts || charts.length === 0) {
      grid.innerHTML = '<p class="text-muted" style="grid-column:1/-1;padding:40px;text-align:center;">No charts generated for this dataset.</p>';
      return;
    }

    charts.forEach((chart, idx) => {
      const card = _createChartCard(chart, idx);
      grid.appendChild(card);
    });
  }

  function _createChartCard(chart, idx) {
    const id = `chart-${idx}`;
    const typeLabel = TYPE_LABELS[chart.type] || chart.type || 'Chart';

    const card = document.createElement('div');
    card.className = 'chart-card';

    card.innerHTML = `
      <div class="chart-card-header">
        <div style="flex:1;min-width:0;">
          <div class="chart-card-title">${chart.title || `Chart ${idx + 1}`}</div>
          <div class="chart-card-type">${typeLabel}</div>
        </div>
        <button class="chart-expand-btn" data-idx="${idx}" title="Expand chart" aria-label="Expand ${chart.title}">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <polyline points="15 3 21 3 21 9"/><polyline points="9 21 3 21 3 15"/>
            <line x1="21" y1="3" x2="14" y2="10"/><line x1="3" y1="21" x2="10" y2="14"/>
          </svg>
        </button>
      </div>
      <div class="chart-container" id="${id}"></div>
    `;

    // Expand button
    card.querySelector('.chart-expand-btn').addEventListener('click', () => {
      _openExpand(chart);
    });

    // Render Plotly
    setTimeout(() => {
      const el = card.querySelector(`#${id}`);
      if (!el || !chart.figure) return;
      const layout = _mergeLayout(chart.figure.layout, { height: 300 });
      Plotly.newPlot(el, chart.figure.data || [], layout, {
        ...PLOTLY_CONFIG,
        // Show simplified modebar in card, full in expand modal
        modeBarButtonsToRemove: ['lasso2d', 'select2d', 'autoScale2d', 'hoverCompareCartesian'],
      }).catch(() => {});
    }, 60 * (idx % 4)); // stagger rendering to avoid UI block

    return card;
  }

  // ─── Column Modal ────────────────────────────────────────────────────────────
  async function openColumnModal(colName) {
    if (_colModalBusy) return; // prevent double-click freeze
    if (!_jobId && !window.currentJobId) {
      console.warn('[ChartsModule] No job_id set — cannot fetch column analysis');
      return;
    }
    const jobId = _jobId || window.currentJobId;

    const overlay = document.getElementById('column-modal-overlay');
    const titleEl = document.getElementById('column-modal-title');
    const insightEl = document.getElementById('column-modal-insight');
    const chartsEl = document.getElementById('column-modal-charts');
    if (!overlay) return;

    _colModalBusy = true;
    titleEl.innerHTML = `Column: <span class="gradient-text">${colName}</span>`;
    insightEl.innerHTML = `
      <div style="display:flex;gap:5px;align-items:center;padding:10px;">
        <div class="typing-dot"></div><div class="typing-dot"></div><div class="typing-dot"></div>
        <span class="text-muted text-sm" style="margin-left:8px">Analyzing column with AI...</span>
      </div>`;
    chartsEl.innerHTML = '';
    overlay.classList.add('open');

    try {
      const res = await fetch('/column-analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ job_id: jobId, column: colName }),
      });
      const data = await res.json();

      if (data.insight) {
        insightEl.className = 'markdown-body';
        try { insightEl.innerHTML = marked.parse(data.insight); } catch { insightEl.textContent = data.insight; }
      } else {
        insightEl.className = '';
        insightEl.innerHTML = `<p class="text-muted">${data.error || 'No insight available.'}</p>`;
      }
    } catch {
      insightEl.className = '';
      insightEl.innerHTML = '<p class="text-muted">Failed to load AI insight.</p>';
    } finally {
      _colModalBusy = false;
    }

    // Show column charts from existing data
    const colCharts = _charts.filter(c =>
      c.title && c.title.toLowerCase().includes(colName.toLowerCase())
    );
    if (colCharts.length > 0) {
      chartsEl.innerHTML = '<div style="font-size:0.8rem;font-weight:600;color:var(--clr-text-muted);margin:16px 0 8px;text-transform:uppercase;letter-spacing:0.5px;">Related Charts</div>';
      colCharts.forEach((chart, i) => {
        const id = `col-chart-${i}-${Date.now()}`;
        const wrap = document.createElement('div');
        wrap.style.cssText = 'margin-top:12px; border:1px solid var(--clr-border); border-radius:var(--radius-md); overflow:hidden;';
        wrap.innerHTML = `
          <div style="padding:10px 16px;border-bottom:1px solid var(--clr-border);font-size:0.82rem;font-weight:600;color:var(--clr-text)">${chart.title}</div>
          <div id="${id}" style="height:260px"></div>
        `;
        chartsEl.appendChild(wrap);
        setTimeout(() => {
          const el = document.getElementById(id);
          if (el && chart.figure) {
            const layout = _mergeLayout(chart.figure.layout, { height: 260 });
            Plotly.newPlot(el, chart.figure.data || [], layout, PLOTLY_CONFIG).catch(() => {});
          }
        }, 100);
      });
    }
  }

  function closeColumnModal() {
    const overlay = document.getElementById('column-modal-overlay');
    if (overlay) overlay.classList.remove('open');
    // Purge Plotly instances in modal to free memory
    document.querySelectorAll('[id^="col-chart-"]').forEach(el => {
      try { Plotly.purge(el); } catch {}
    });
  }

  function setJobId(id) {
    _jobId = id;
  }

  // ─── Init: close modal on overlay click ──────────────────────────────────────
  document.addEventListener('DOMContentLoaded', () => {
    document.getElementById('column-modal-overlay')?.addEventListener('click', e => {
      if (e.target.id === 'column-modal-overlay') closeColumnModal();
    });
  });

  // Public API
  window.ChartsModule = { renderAll, openColumnModal, closeColumnModal, setJobId };
})();
