import { Viewer } from './viewer.js';

const $ = (id) => document.getElementById(id);
const api = {
  async get(url) { const r = await fetch(url); return r.json(); },
  async post(url, body) {
    const r = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body || {}),
    });
    return r.json();
  },
};

const state = {
  selected: [],          // [{path, name, size}]
  formats: new Set(['fbx']),
  jobs: [],
  activeJob: null,
  config: {},
  polling: null,
  browse: { path: null, mode: 'file' },
};

let viewer = null;

// ----------------------------------------------------------------- helpers
const fmtBytes = (n) => {
  if (!n) return '0 B';
  const u = ['B', 'KB', 'MB', 'GB'];
  const i = Math.min(u.length - 1, Math.floor(Math.log(n) / Math.log(1024)));
  return (n / Math.pow(1024, i)).toFixed(i ? 1 : 0) + ' ' + u[i];
};
const esc = (s) => String(s ?? '').replace(/[&<>"]/g,
  (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const status = (msg) => { $('statusText').textContent = msg; };
const baseName = (p) => String(p).split(/[\\/]/).pop();

// -------------------------------------------------------------- selection
function addSelection(items) {
  for (const it of items) {
    if (!state.selected.some((s) => s.path === it.path)) state.selected.push(it);
  }
  renderSelection();
}

function renderSelection() {
  const box = $('selectedList');
  if (!state.selected.length) {
    box.innerHTML = '';
    $('btnConvert').disabled = true;
    return;
  }
  box.innerHTML = state.selected.map((s, i) => `
    <div class="sel-item">
      <span class="nm" title="${esc(s.path)}">${esc(s.name)}</span>
      <span class="meta">${s.size ? fmtBytes(s.size) : ''}</span>
      <span class="x" data-i="${i}" title="Remove">&times;</span>
    </div>`).join('');
  box.querySelectorAll('.x').forEach((el) => {
    el.onclick = () => {
      state.selected.splice(Number(el.dataset.i), 1);
      renderSelection();
    };
  });
  $('btnConvert').disabled = state.formats.size === 0;
  $('btnConvert').textContent = state.selected.length > 1
    ? `Convert ${state.selected.length} files` : 'Convert';
}

// ---------------------------------------------------------------- formats
function renderFormats(formats) {
  $('formatGrid').innerHTML = formats.map((f) => `
    <div class="fmt ${state.formats.has(f.key) ? 'on' : ''}" data-k="${f.key}">
      <b>${f.key.toUpperCase()}</b>
      <span>${esc(f.label.replace(/^[A-Za-z]+ ?/, ''))|| f.label}</span>
      <span>${f.rig ? 'rig' : 'no rig'} &middot; ${f.morphs ? 'morphs' : 'no morphs'}</span>
    </div>`).join('');
  $('formatGrid').querySelectorAll('.fmt').forEach((el) => {
    el.onclick = () => {
      const k = el.dataset.k;
      if (state.formats.has(k)) state.formats.delete(k); else state.formats.add(k);
      el.classList.toggle('on');
      renderSelection();
    };
  });
}

// ------------------------------------------------------------------- jobs
function jobOptions() {
  const filter = $('optMorphFilter').value.trim();
  return {
    include_skeleton: $('optSkeleton').checked,
    include_morphs: $('optMorphs').checked,
    include_hd_morphs: $('optHd').checked,
    merge_meshes: $('optMerge').checked,
    pose: $('optPose').value,
    max_influences: Number($('optInfluences').value),
    scale: Number($('optScale').value),
    morph_filter: filter ? filter.split(',').map((s) => s.trim()).filter(Boolean) : [],
    copy_textures: $('optCopyTex').checked,
    embed_textures: $('optEmbed').checked,
  };
}

async function startConversion() {
  if (!state.selected.length || !state.formats.size) return;
  const res = await api.post('/api/jobs', {
    paths: state.selected.map((s) => s.path),
    formats: [...state.formats],
    options: jobOptions(),
    outputDir: $('outputDir').value,
  });
  if (res.error) { status('Error: ' + res.error); return; }
  status(`Queued ${res.jobs.length} job(s)`);
  pollJobs(true);
}

function renderJobs() {
  const list = $('jobList');
  if (!state.jobs.length) {
    list.innerHTML = '<div class="hint">No conversions yet. Pick a .duf file and hit Convert &mdash; or press <b>Load demo asset</b> to try it immediately.</div>';
    return;
  }
  list.innerHTML = state.jobs.map((j) => {
    const s = j.summary || {};
    const stats = j.status === 'done' ? `
      <div class="job-stats">
        ${s.vertices ? `<span class="chip"><b>${s.vertices.toLocaleString()}</b> verts</span>` : ''}
        ${s.polygons ? `<span class="chip"><b>${s.polygons.toLocaleString()}</b> polys</span>` : ''}
        ${s.materials ? `<span class="chip"><b>${s.materials}</b> mats</span>` : ''}
        ${s.bones ? `<span class="chip"><b>${s.bones}</b> bones</span>` : ''}
        ${s.morphs ? `<span class="chip"><b>${s.morphs}</b> morphs</span>` : ''}
      </div>` : '';
    const files = (j.outputs || []).map((o) =>
      `<a class="file-chip" href="/api/download/${j.id}?file=${encodeURIComponent(o.name)}" download>${esc(o.format.toUpperCase())} &middot; ${fmtBytes(o.bytes)}</a>`).join('');
    const all = j.outputs && j.outputs.length > 1
      ? `<a class="file-chip" href="/api/download/${j.id}" download>ZIP all</a>` : '';
    return `
      <div class="job ${state.activeJob === j.id ? 'active' : ''}" data-id="${j.id}">
        <div class="job-top">
          <span class="dot ${j.status}"></span>
          <span class="job-name" title="${esc(j.source)}">${esc(j.name)}</span>
          <span class="muted small">${j.status === 'done' ? j.duration + 's' : ''}</span>
        </div>
        ${j.status === 'running' || j.status === 'queued'
          ? `<div class="progress"><i style="width:${Math.round(j.progress * 100)}%"></i></div>` : ''}
        <div class="job-msg ${j.status === 'error' ? 'err' : ''}" style="${j.status === 'error' ? 'color:var(--err)' : ''}">${esc(j.message)}</div>
        ${stats}
        <div class="job-files">${files}${all}</div>
      </div>`;
  }).join('');

  list.querySelectorAll('.job').forEach((el) => {
    el.onclick = (ev) => {
      if (ev.target.closest('a')) return;
      selectJob(el.dataset.id);
    };
  });
}

async function selectJob(id) {
  const job = state.jobs.find((j) => j.id === id);
  if (!job) return;
  state.activeJob = id;
  renderJobs();
  renderWarnings(job);
  if (job.status === 'done' && job.hasPreview) {
    await loadPreview(job);
  }
}

function renderWarnings(job) {
  const box = $('warningBox');
  const warnings = (job && job.warnings) || [];
  if (!warnings.length) { box.innerHTML = ''; return; }
  box.innerHTML = `<div class="warn-box"><b>${warnings.length} note(s)</b><ul>${
    warnings.slice(0, 8).map((w) => `<li>${esc(w.split('\n')[0]).slice(0, 180)}</li>`).join('')
  }</ul></div>`;
}

async function loadPreview(job) {
  status('Loading preview\u2026');
  $('viewportEmpty').hidden = true;
  try {
    const stats = await viewer.load(`/api/preview/${job.id}?t=${job.finished}`);
    $('viewerTitle').textContent = job.name;
    $('viewerStats').innerHTML = [
      `<span class="chip"><b>${stats.vertices.toLocaleString()}</b> verts</span>`,
      `<span class="chip"><b>${stats.triangles.toLocaleString()}</b> tris</span>`,
      `<span class="chip"><b>${stats.materials}</b> materials</span>`,
      stats.bones ? `<span class="chip"><b>${stats.bones}</b> bones</span>` : '',
      stats.morphNames.length ? `<span class="chip"><b>${stats.morphNames.length}</b> morphs</span>` : '',
    ].join('');
    $('btnBones').disabled = !viewer.hasSkeleton();
    renderMorphSliders(stats.morphNames);
    status('Preview ready');
  } catch (err) {
    $('viewportEmpty').hidden = false;
    status('Preview failed: ' + err.message);
  }
}

function renderMorphSliders(names) {
  const panel = $('morphPanel');
  const body = $('morphSliders');
  if (!names.length) { panel.hidden = true; body.innerHTML = ''; return; }
  panel.hidden = false;
  body.innerHTML = names.slice(0, 60).map((n, i) => `
    <div class="morph-row">
      <label><b title="${esc(n)}">${esc(n)}</b><span id="mv${i}">0.00</span></label>
      <input type="range" min="0" max="1" step="0.01" value="0" data-n="${esc(n)}" data-i="${i}">
    </div>`).join('');
  body.querySelectorAll('input').forEach((el) => {
    el.oninput = () => {
      viewer.setMorph(el.dataset.n, Number(el.value));
      $('mv' + el.dataset.i).textContent = Number(el.value).toFixed(2);
    };
  });
}

async function pollJobs(force) {
  const data = await api.get('/api/jobs');
  state.jobs = data.jobs || [];
  renderJobs();
  const busy = state.jobs.some((j) => j.status === 'running' || j.status === 'queued');
  // auto-select + preview the newest finished job
  const newest = state.jobs.find((j) => j.status === 'done');
  if (newest && (force || !state.activeJob) && state.activeJob !== newest.id) {
    await selectJob(newest.id);
  }
  if (busy) {
    clearTimeout(state.polling);
    state.polling = setTimeout(() => pollJobs(true), 450);
  } else if (force) {
    const active = state.jobs.find((j) => j.id === state.activeJob);
    if (active) renderWarnings(active);
  }
}

// ----------------------------------------------------------------- modals
function openModal(title, bodyHtml, footHtml) {
  $('modalTitle').textContent = title;
  $('modalBody').innerHTML = bodyHtml;
  $('modalFoot').innerHTML = footHtml || '';
  $('modalBackdrop').hidden = false;
}
function closeModal() { $('modalBackdrop').hidden = true; }

async function openBrowser(mode) {
  state.browse.mode = mode;
  const start = state.browse.path || state.config.output_dir || '~';
  await renderBrowser(start);
}

async function renderBrowser(path) {
  const data = await api.post('/api/browse', { path, files: state.browse.mode === 'file' });
  state.browse.path = data.path;
  const rows = [];
  if (data.parent) {
    rows.push(`<div class="browse-row" data-dir="${esc(data.parent)}"><span class="ic">&uarr;</span><span>..</span></div>`);
  }
  for (const e of data.entries) {
    rows.push(e.dir
      ? `<div class="browse-row" data-dir="${esc(e.path)}"><span class="ic">&#128193;</span><span>${esc(e.name)}</span></div>`
      : `<div class="browse-row file" data-file="${esc(e.path)}" data-size="${e.size}"><span class="ic">&#9679;</span><span>${esc(e.name)}</span><span class="spacer"></span><span class="sz">${fmtBytes(e.size)}</span></div>`);
  }
  const title = state.browse.mode === 'file' ? 'Choose Daz files' : 'Choose a folder';
  const foot = state.browse.mode === 'dir'
    ? `<button class="btn ghost" id="mClose2">Cancel</button><button class="btn primary" id="mUseDir">Use this folder</button>`
    : `<button class="btn ghost" id="mClose2">Close</button>`;
  openModal(title, `
    <div class="crumb">${esc(data.path)} ${data.isContentRoot ? '<span class="chip">content root</span>' : ''}</div>
    <div class="browse-list">${rows.join('') || '<div class="browse-row muted">empty</div>'}</div>
    ${state.browse.mode === 'file' ? '<p class="hint" style="margin-top:10px">Only .duf and .dsf files are listed. Click one to add it to the queue.</p>' : ''}
  `, foot);

  document.querySelectorAll('.browse-row').forEach((el) => {
    el.onclick = () => {
      if (el.dataset.dir) renderBrowser(el.dataset.dir);
      else if (el.dataset.file) {
        addSelection([{ path: el.dataset.file, name: baseName(el.dataset.file),
                        size: Number(el.dataset.size) }]);
        status('Added ' + baseName(el.dataset.file));
        closeModal();
      }
    };
  });
  const c2 = $('mClose2'); if (c2) c2.onclick = closeModal;
  const use = $('mUseDir');
  if (use) use.onclick = () => { state.browse.onPick?.(state.browse.path); closeModal(); };
}

async function openSettings() {
  const st = await api.get('/api/state');
  state.config = st.config;
  const dirs = st.config.content_dirs || [];
  const detected = (st.detectedContentDirs || []).filter((d) => !dirs.includes(d));
  openModal('Settings', `
    <p class="hint">Daz content directories are the folders that contain <code>data/</code> and
    <code>Runtime/</code>. DazConv needs them to resolve geometry, UV sets, morphs and textures
    referenced by a scene file. Everything stays on this machine.</p>
    <h4 style="margin:14px 0 6px;font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.8px">Content directories</h4>
    <div class="dir-list" id="dirList">
      ${dirs.length ? dirs.map((d, i) => `
        <div class="dir-item"><span class="p" title="${esc(d)}">${esc(d)}</span>
        <button class="btn tiny" data-rm="${i}">remove</button></div>`).join('')
        : '<div class="hint">None configured yet.</div>'}
    </div>
    <button class="btn small ghost" id="btnAddDir" style="margin-top:6px">Add directory&hellip;</button>
    ${detected.length ? `<h4 style="margin:16px 0 6px;font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.8px">Detected on this machine</h4>
      <div class="dir-list">${detected.map((d) => `<div class="dir-item"><span class="p">${esc(d)}</span>
        <button class="btn tiny" data-add="${esc(d)}">add</button></div>`).join('')}</div>` : ''}
    <h4 style="margin:16px 0 6px;font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.8px">Output folder</h4>
    <input type="text" id="setOutput" value="${esc(st.config.output_dir || '')}" spellcheck="false" />
  `, `<button class="btn ghost" id="mCancel">Cancel</button><button class="btn primary" id="mSave">Save</button>`);

  const refresh = () => openSettings();
  document.querySelectorAll('[data-rm]').forEach((el) => {
    el.onclick = async () => {
      const next = dirs.filter((_, i) => i !== Number(el.dataset.rm));
      await api.post('/api/settings', { content_dirs: next });
      refresh();
    };
  });
  document.querySelectorAll('[data-add]').forEach((el) => {
    el.onclick = async () => {
      await api.post('/api/settings', { content_dirs: [...dirs, el.dataset.add] });
      refresh();
    };
  });
  $('btnAddDir').onclick = () => {
    state.browse.mode = 'dir';
    state.browse.onPick = async (p) => {
      await api.post('/api/settings', { content_dirs: [...dirs, p] });
      refresh();
    };
    openBrowser('dir');
  };
  $('mCancel').onclick = closeModal;
  $('mSave').onclick = async () => {
    const cfg = await api.post('/api/settings', { output_dir: $('setOutput').value });
    state.config = cfg.config;
    $('outputDir').value = cfg.config.output_dir;
    updateContentStatus(cfg.config.content_dirs);
    closeModal();
  };
}

async function openLibrary() {
  openModal('Content library', '<div class="hint">Scanning&hellip;</div>', '');
  const data = await api.get('/api/scan?limit=400');
  if (!data.assets.length) {
    $('modalBody').innerHTML = `<p class="hint">No <code>.duf</code> assets found.
      Add a Daz content directory in Settings, or press <b>Load demo asset</b>.</p>`;
    return;
  }
  $('modalBody').innerHTML = `
    <div class="crumb">${data.count} asset(s) in ${data.roots.length} root(s)</div>
    <input type="text" id="libFilter" placeholder="Filter&hellip;" style="margin-bottom:8px" />
    <div class="browse-list" id="libList">
      ${data.assets.map((a) => `<div class="browse-row file" data-file="${esc(a.path)}" data-size="${a.size}">
        <span class="ic">&#9679;</span><span>${esc(a.name)}</span><span class="spacer"></span>
        <span class="sz">${esc(a.rel.split('/').slice(0, -1).join('/'))}</span></div>`).join('')}
    </div>`;
  const bind = () => document.querySelectorAll('#libList .browse-row').forEach((el) => {
    el.onclick = () => {
      addSelection([{ path: el.dataset.file, name: baseName(el.dataset.file),
                      size: Number(el.dataset.size) }]);
      closeModal();
    };
  });
  bind();
  $('libFilter').oninput = (e) => {
    const q = e.target.value.toLowerCase();
    document.querySelectorAll('#libList .browse-row').forEach((el) => {
      el.style.display = el.textContent.toLowerCase().includes(q) ? '' : 'none';
    });
  };
}

// ---------------------------------------------------------------- uploads
async function uploadFiles(fileList) {
  const files = [...fileList].filter((f) =>
    /\.(duf|dsf|png|jpe?g|tiff?|tga|bmp)$/i.test(f.name));
  if (!files.length) { status('No Daz files in that drop'); return; }
  status(`Uploading ${files.length} file(s)\u2026`);
  const form = new FormData();
  for (const f of files) form.append(f.webkitRelativePath || f.name, f, f.name);
  const r = await fetch('/api/upload', { method: 'POST', body: form });
  const res = await r.json();
  if (res.error) { status('Upload failed: ' + res.error); return; }
  addSelection((res.paths || []).map((p) => ({ path: p, name: baseName(p), size: 0 })));
  status(`Added ${(res.paths || []).length} asset file(s)`);
}

function updateContentStatus(dirs) {
  const n = (dirs || []).length;
  $('contentDirsStatus').textContent = n
    ? `${n} content director${n === 1 ? 'y' : 'ies'}`
    : 'no content directories configured';
}

// -------------------------------------------------------------------- init
async function init() {
  viewer = new Viewer($('viewport'));

  const st = await api.get('/api/state');
  state.config = st.config;
  $('versionPill').textContent = 'v' + st.app.version;
  $('outputDir').value = st.config.output_dir || '';
  renderFormats(st.formats);
  updateContentStatus(st.config.content_dirs);
  state.jobs = st.jobs || [];
  renderJobs();

  // option wiring
  $('optInfluences').oninput = (e) => { $('influencesVal').textContent = e.target.value; };
  $('optScale').oninput = (e) => {
    $('scaleVal').textContent = Number(e.target.value).toFixed(2) + '\u00d7';
  };
  $('optEmbed').onchange = (e) => { if (e.target.checked) $('optCopyTex').checked = false; };
  $('optCopyTex').onchange = (e) => { if (e.target.checked) $('optEmbed').checked = false; };

  // source buttons
  $('btnPickFiles').onclick = () => $('fileInput').click();
  $('btnPickFolder').onclick = () => $('folderInput').click();
  $('fileInput').onchange = (e) => uploadFiles(e.target.files);
  $('folderInput').onchange = (e) => uploadFiles(e.target.files);
  $('btnBrowse').onclick = () => openBrowser('file');
  $('btnLibrary').onclick = openLibrary;
  $('btnSettings').onclick = openSettings;

  $('btnDemo').onclick = async () => {
    status('Generating demo library\u2026');
    const res = await api.post('/api/demo', {});
    state.config = res.config;
    updateContentStatus(res.config.content_dirs);
    addSelection([{ path: res.manifest.scene, name: 'DemoFigure.duf', size: 0 }]);
    status(`Demo asset ready: ${res.manifest.vertices || ''} verts, ${res.manifest.bones || ''} bones`);
  };

  $('btnConvert').onclick = startConversion;
  $('btnReveal').onclick = () => api.post('/api/reveal', { path: $('outputDir').value });
  $('btnClearJobs').onclick = async () => {
    await api.post('/api/jobs/clear', {});
    state.activeJob = null;
    pollJobs(true);
  };

  // viewer toolbar
  const toggle = (btn, fn) => {
    btn.onclick = () => { btn.classList.toggle('active'); fn(btn.classList.contains('active')); };
  };
  toggle($('btnWire'), (v) => viewer.setWireframe(v));
  toggle($('btnBones'), (v) => viewer.setSkeleton(v));
  toggle($('btnSpin'), (v) => viewer.setSpin(v));
  $('btnTex').classList.add('active');
  $('btnTex').onclick = () => {
    $('btnTex').classList.toggle('active');
    viewer.setTextures($('btnTex').classList.contains('active'));
  };
  $('btnFrame').onclick = () => viewer.frame();
  $('btnMorphReset').onclick = () => {
    viewer.resetMorphs();
    document.querySelectorAll('#morphSliders input').forEach((el) => {
      el.value = 0;
      $('mv' + el.dataset.i).textContent = '0.00';
    });
  };

  // modal
  $('modalClose').onclick = closeModal;
  $('modalBackdrop').onclick = (e) => { if (e.target === $('modalBackdrop')) closeModal(); };
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape') closeModal(); });

  // drag and drop
  const dz = $('dropzone');
  ['dragenter', 'dragover'].forEach((ev) => document.addEventListener(ev, (e) => {
    e.preventDefault(); dz.classList.add('hot');
  }));
  ['dragleave', 'drop'].forEach((ev) => document.addEventListener(ev, (e) => {
    e.preventDefault();
    if (ev === 'drop' || e.relatedTarget === null) dz.classList.remove('hot');
  }));
  document.addEventListener('drop', (e) => {
    if (e.dataTransfer && e.dataTransfer.files.length) uploadFiles(e.dataTransfer.files);
  });

  pollJobs(false);
  status('Ready');
}

init();
