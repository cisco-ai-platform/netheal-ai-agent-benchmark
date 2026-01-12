// Copyright 2026 Cisco Systems, Inc. and its affiliates
//
// SPDX-License-Identifier: Apache-2.0

// NetHeal Web Demo frontend (vanilla JS)
// Interacts with FastAPI backend endpoints to drive the RL environment.

const byId = (id) => document.getElementById(id);

const els = {
  form: byId('reset-form'),
  seed: byId('seed'),
  maxDevices: byId('max_devices'),
  maxSteps: byId('max_steps'),
  enableHints: byId('enable_hints'),
  hintMode: byId('hint_mode'),
  hint: byId('hint'),
  stepCount: byId('step-count'),
  progress: byId('progress'),
  networkSize: byId('network-size'),
  discovered: byId('discovered'),
  lastReward: byId('last-reward'),
  terminated: byId('terminated'),
  truncated: byId('truncated'),
  rewardBreakdown: byId('reward-breakdown'),
  actions: byId('actions'),
  discoveryMatrix: byId('discovery-matrix'),
  recentDiagnostics: byId('recent-diagnostics'),
  rawInfo: byId('raw-info'),
  deviceMap: byId('device-map'),
  actionSearch: byId('action-search'),
  actionCategory: byId('action-category'),
  actionDedup: byId('action-dedup'),
  actionsCount: byId('actions-count'),
};

const API = {
  async reset(payload) {
    const res = await fetch('/api/env/reset', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!res.ok) throw new Error(await res.text());
    return res.json();
  },
  async state() {
    const res = await fetch('/api/env/state');
    if (!res.ok) throw new Error(await res.text());
    return res.json();
  },
  async actions() {
    const res = await fetch('/api/env/actions');
    if (!res.ok) throw new Error(await res.text());
    return res.json();
  },
  async step(action_id) {
    const res = await fetch('/api/env/step', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action_id }),
    });
    if (!res.ok) throw new Error(await res.text());
    return res.json();
  },
  async exportScenario() {
    const res = await fetch('/api/env/export');
    if (!res.ok) throw new Error(await res.text());
    return res.json();
  },
  async importScenario(scenarioData) {
    const res = await fetch('/api/env/import', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ scenario_data: scenarioData }),
    });
    if (!res.ok) throw new Error(await res.text());
    return res.json();
  },
};

let lastActionSpecs = [];

function renderState(state) {
  const info = state.info || {};
  const obs = state.observation || {};

  els.hint.textContent = info.user_hint || 'No hint available.';

  els.stepCount.textContent = (info.step_count ?? 0).toString();
  els.progress.textContent = (info.episode_progress ?? 0).toFixed(2);
  els.networkSize.textContent = (info.network_size ?? 0).toString();
  els.discovered.textContent = (info.discovered_devices ?? 0).toString();
  els.lastReward.textContent = (state.last_reward ?? 0).toFixed(2);
  els.terminated.textContent = (state.terminated ?? false).toString();
  els.truncated.textContent = (state.truncated ?? false).toString();
  els.rewardBreakdown.textContent = JSON.stringify(info.reward_breakdown || {}, null, 2);

  renderDiscoveryMatrix(obs.discovery_matrix, state.device_index_map || []);
  renderRecentDiagnosticsDetailed(state.recent_diagnostics_detailed || []);
  renderDeviceMap(state.device_index_map || []);
  els.rawInfo.textContent = JSON.stringify(info, null, 2);

  // Prefer detailed specs when available
  lastActionSpecs = Array.isArray(state.valid_action_specs) ? state.valid_action_specs : [];
  renderActionsWithFilters();

  renderFinalOutcome(state.final_outcome || null);
  
  // Enable export button now that we have a valid environment
  const exportBtn = byId('export-btn');
  if (exportBtn) {
    exportBtn.disabled = false;
    exportBtn.title = 'Export current scenario to JSON file';
  }
}

function renderActionsWithFilters() {
  const search = (els.actionSearch.value || '').toLowerCase().trim();
  const cat = els.actionCategory.value || 'all';
  const dedup = !!els.actionDedup.checked;

  // Work on a copy
  let specs = Array.isArray(lastActionSpecs) ? [...lastActionSpecs] : [];

  // Filter by category
  if (cat !== 'all') {
    specs = specs.filter(s => (s.category || '').toLowerCase() === cat);
  }

  // Free-text search across description, type, and parameters
  if (search) {
    specs = specs.filter(s => {
      const p = JSON.stringify(s.parameters || {});
      const text = `${s.description || ''} ${s.action_type || ''} ${p}`.toLowerCase();
      return text.includes(search);
    });
  }

  // Optional dedup (by description + parameters signature)
  if (dedup) {
    const seen = new Set();
    specs = specs.filter(s => {
      const sig = `${s.category}|${s.action_type}|${JSON.stringify(s.parameters || {})}`;
      if (seen.has(sig)) return false;
      seen.add(sig);
      return true;
    });
  }

  els.actions.innerHTML = '';
  els.actionsCount.textContent = `${specs.length} actions`;

  // Group by category for readability
  const groups = {
    topology_discovery: [],
    diagnostic: [],
    diagnosis: [],
  };
  for (const s of specs) {
    const key = (s.category || '').toLowerCase();
    if (groups[key]) groups[key].push(s);
  }

  const order = ['topology_discovery', 'diagnostic', 'diagnosis'];
  for (const groupKey of order) {
    const list = groups[groupKey] || [];
    if (!list.length) continue;
    const header = document.createElement('h3');
    header.textContent = groupKey.replace('_', ' ').replace(/\b\w/g, c => c.toUpperCase());
    header.className = 'action-group-header';
    els.actions.appendChild(header);

    const subset = list.slice(0, 200);
    const groupDiv = document.createElement('div');
    groupDiv.className = 'action-group';
    for (const spec of subset) {
      const id = spec.id;
      const type = spec.action_type;
      const params = spec.parameters || {};
      const paramsStr = Object.entries(params).map(([k,v]) => `${k}=${v}`).join(', ');
      const label = `${id}: ${type} (${paramsStr || 'no-params'})`;

      const btn = document.createElement('button');
      btn.className = 'action-btn';
      btn.textContent = label;
      btn.title = spec.description || 'Execute action';
      btn.addEventListener('click', async () => {
        try {
          const state = await API.step(id);
          renderState(state);
        } catch (e) {
          alert(`Step failed: ${e}`);
        }
      });
      groupDiv.appendChild(btn);
    }
    els.actions.appendChild(groupDiv);
  }
}

function renderDiscoveryMatrix(matrix, deviceMap) {
  els.discoveryMatrix.innerHTML = '';
  if (!Array.isArray(matrix) || !Array.isArray(deviceMap)) return;
  const n = Math.min(deviceMap.length, matrix.length);
  if (n === 0) {
    els.discoveryMatrix.textContent = 'No discovered links yet.';
    return;
  }

  // Build axis labels + matrix body
  const container = document.createElement('div');
  container.className = 'matrix-wrap';

  // Column labels
  const colsBar = document.createElement('div');
  colsBar.className = 'matrix-cols';
  const corner = document.createElement('div');
  corner.className = 'corner-cell';
  colsBar.appendChild(corner);
  for (let c = 0; c < n; c++) {
    const lab = document.createElement('div');
    lab.className = 'col-label';
    lab.textContent = deviceMap[c];
    lab.title = `#${c}`;
    colsBar.appendChild(lab);
  }
  container.appendChild(colsBar);

  // Rows with labels and cells
  for (let r = 0; r < n; r++) {
    const row = document.createElement('div');
    row.className = 'matrix-row';
    const rlab = document.createElement('div');
    rlab.className = 'row-label';
    rlab.textContent = deviceMap[r];
    rlab.title = `#${r}`;
    row.appendChild(rlab);
    for (let c = 0; c < n; c++) {
      const val = matrix[r][c];
      const cell = document.createElement('div');
      cell.className = 'cell ' + (val === 1 ? 'c' : val === -1 ? 'd' : val === 2 ? 'g' : 'u');
      cell.title = `${deviceMap[r]} → ${deviceMap[c]} : ${val}`;
      row.appendChild(cell);
    }
    container.appendChild(row);
  }
  els.discoveryMatrix.appendChild(container);
}

function renderRecentDiagnosticsDetailed(items) {
  els.recentDiagnostics.innerHTML = '';
  if (!Array.isArray(items)) return;
  if (items.length === 0) {
    els.recentDiagnostics.textContent = 'No diagnostics yet.';
    return;
  }
  const list = document.createElement('div');
  for (let i = 0; i < items.length; i++) {
    const it = items[i];
    const div = document.createElement('div');
    div.className = 'diag-row ' + (it.success ? 'ok' : 'fail');
    const dir = it.destination ? `${it.source} → ${it.destination}` : `${it.source || ''}`;
    const summary = it.summary || '';
    div.textContent = `#${i + 1} ${it.tool}: ${dir} — ${summary}`;
    list.appendChild(div);
  }
  els.recentDiagnostics.appendChild(list);
}

function renderDeviceMap(map) {
  els.deviceMap.innerHTML = '';
  if (!Array.isArray(map)) return;
  if (map.length === 0) {
    els.deviceMap.textContent = 'No devices discovered yet.';
    return;
  }
  const ul = document.createElement('ul');
  for (let i = 0; i < map.length; i++) {
    const li = document.createElement('li');
    li.textContent = `#${i} → ${map[i]}`;
    ul.appendChild(li);
  }
  els.deviceMap.appendChild(ul);
}

function renderFinalOutcome(outcome) {
  const el = document.getElementById('final-outcome');
  if (!el) return;
  if (!outcome || !outcome.final) {
    el.classList.add('hidden');
    el.classList.remove('success', 'failure');
    el.textContent = '';
    return;
  }
  const by = outcome.by || 'unknown';
  const isSuccess = by === 'diagnosis' && outcome.correct === true;
  el.classList.remove('hidden');
  el.classList.toggle('success', isSuccess);
  el.classList.toggle('failure', !isSuccess);

  if (by === 'diagnosis') {
    el.textContent = isSuccess
      ? `Diagnosis correct: ${outcome.diagnosed_fault} at ${outcome.diagnosed_location}`
      : `Diagnosis incorrect: predicted ${outcome.diagnosed_fault} at ${outcome.diagnosed_location}; GT ${outcome.ground_truth_fault} at ${outcome.ground_truth_location}`;
  } else if (by === 'timeout') {
    el.textContent = `Episode ended by timeout. Ground truth: ${outcome.ground_truth_fault} at ${outcome.ground_truth_location}`;
  } else {
    el.textContent = `Episode ended.`;
  }
}
async function doReset(evt) {
  evt.preventDefault();
  const payload = {
    seed: els.seed.value ? Number(els.seed.value) : null,
    max_devices: Number(els.maxDevices.value || 8),
    max_episode_steps: Number(els.maxSteps.value || 20),
    topology_types: null,
    enable_user_hints: els.enableHints.value === 'true',
    hint_provider_mode: els.hintMode.value || 'auto',
    user_context: {},
  };
  try {
    const state = await API.reset(payload);
    renderState(state);
  } catch (e) {
    alert(`Reset failed: ${e}`);
  }
}

els.form.addEventListener('submit', doReset);
els.actionSearch.addEventListener('input', renderActionsWithFilters);
els.actionCategory.addEventListener('change', renderActionsWithFilters);
els.actionDedup.addEventListener('change', renderActionsWithFilters);

// Export/Import handlers
const exportBtn = byId('export-btn');
const importBtn = byId('import-btn');
const importFile = byId('import-file');

console.log('Export/Import buttons:', { exportBtn, importBtn, importFile });

if (exportBtn) {
  console.log('Setting up export button listener');
  // Initially disable export button until environment is initialized
  exportBtn.disabled = true;
  exportBtn.title = 'Click "Reset Episode" first to create a scenario';
  
  exportBtn.addEventListener('click', async () => {
    console.log('Export button clicked');
    try {
      const scenario = await API.exportScenario();
      console.log('Scenario exported:', scenario);
      const blob = new Blob([JSON.stringify(scenario, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      const timestamp = new Date().toISOString().replace(/[:.]/g, '-').substring(0, 19);
      a.download = `netheal-scenario-${timestamp}.json`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
      alert('Scenario exported successfully!');
    } catch (e) {
      console.error('Export error:', e);
      if (e.toString().includes('not initialized')) {
        alert('Please click "Reset Episode" first to create a scenario before exporting.');
      } else {
        alert(`Export failed: ${e}`);
      }
    }
  });
} else {
  console.error('Export button not found!');
}

if (importBtn) {
  console.log('Setting up import button listener');
  importBtn.addEventListener('click', () => {
    console.log('Import button clicked');
    if (importFile) importFile.click();
  });
} else {
  console.error('Import button not found!');
}

if (importFile) {
  console.log('Setting up file input listener');
  importFile.addEventListener('change', async (evt) => {
    console.log('File selected');
    const file = evt.target.files?.[0];
    if (!file) return;
    
    try {
      const text = await file.text();
      const scenarioData = JSON.parse(text);
      console.log('Importing scenario:', scenarioData);
      const state = await API.importScenario(scenarioData);
      renderState(state);
      alert('Scenario imported successfully!');
      // Reset the file input so the same file can be imported again
      importFile.value = '';
    } catch (e) {
      console.error('Import error:', e);
      alert(`Import failed: ${e}`);
      importFile.value = '';
    }
  });
} else {
  console.error('File input not found!');
}

// Try to fetch state (if env already exists). If not, it's fine.
(async function init() {
  try {
    const state = await API.state();
    renderState(state);
  } catch (_) {
    // no existing env; wait for reset
  }
})();
