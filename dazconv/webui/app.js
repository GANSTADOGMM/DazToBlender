/* dazconv web UI -- no frameworks, no CDN.
 * Talks to the Python backend over JSON and renders the preview with raw WebGL.
 */
"use strict";

const $ = (id) => document.getElementById(id);

const state = {
  file: null,          // File from drag & drop / file picker
  path: null,          // absolute path chosen from the content library
  sourceName: "",
  jobId: null,
  pollTimer: null,
  assets: [],
};

/* ------------------------------------------------------------------ api --- */
async function api(path, options) {
  const response = await fetch(path, options);
  const text = await response.text();
  let data;
  try { data = JSON.parse(text); } catch { throw new Error(text.slice(0, 300)); }
  if (!response.ok || data.ok === false) {
    throw new Error(data.error || `HTTP ${response.status}`);
  }
  return data;
}

/* -------------------------------------------------------------- options --- */
function readOptions() {
  const morphs = $("opt_morph").value
    .split(",")
    .map((entry) => entry.trim())
    .filter(Boolean)
    .map((entry) => {
      const [id, value] = entry.split("=");
      let weight = 1.0;
      if (value !== undefined && value.trim() !== "") {
        const raw = value.trim();
        weight = raw.endsWith("%") ? parseFloat(raw) / 100 : parseFloat(raw);
      }
      return { id: id.trim(), weight: Number.isFinite(weight) ? weight : 1.0 };
    });

  return {
    triangulate: $("opt_triangulate").checked,
    smooth_normals: $("opt_smooth").checked,
    flip_uv: $("opt_flipuv").checked,
    export_skeleton: $("opt_skeleton").checked,
    export_weights: $("opt_weights").checked,
    export_materials: $("opt_materials").checked,
    export_blendshapes: $("opt_blendshapes").checked,
    up_axis: $("opt_upaxis").value,
    max_influences: parseInt($("opt_influences").value, 10) || 4,
    scale: parseFloat($("opt_scale").value) || 1,
    unit_scale: parseFloat($("opt_unit").value) || 1,
    morphs,
  };
}

function applyOptions(options) {
  if (!options) return;
  const set = (id, value) => { if (value !== undefined && $(id)) $(id).value = value; };
  const check = (id, value) => { if (typeof value === "boolean" && $(id)) $(id).checked = value; };
  check("opt_triangulate", options.triangulate);
  check("opt_smooth", options.smooth_normals);
  check("opt_flipuv", options.flip_uv);
  check("opt_skeleton", options.export_skeleton);
  check("opt_weights", options.export_weights);
  check("opt_materials", options.export_materials);
  set("opt_upaxis", options.up_axis);
  set("opt_influences", options.max_influences);
  set("opt_scale", options.scale);
  set("opt_unit", options.unit_scale);
}

/* ----------------------------------------------------------- drop & pick -- */
function setSource(file, path) {
  state.file = file || null;
  state.path = path || null;
  state.sourceName = file ? file.name : (path ? path.split(/[\\/]/).pop() : "");
  const size = file ? ` &middot; ${(file.size / 1024).toFixed(1)} KB` : "";
  $("fileline").innerHTML = state.sourceName
    ? `<strong>${escapeHtml(state.sourceName)}</strong>${size}`
    : "";
  $("convert").disabled = !state.sourceName;
}

function escapeHtml(text) {
  return String(text).replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  })[c]);
}

function initDropZone() {
  const zone = $("dropzone");
  const input = $("fileinput");

  zone.addEventListener("click", () => input.click());
  zone.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") { event.preventDefault(); input.click(); }
  });
  input.addEventListener("change", () => {
    if (input.files && input.files[0]) { setSource(input.files[0], null); markSelection(null); }
  });

  for (const event of ["dragenter", "dragover"]) {
    zone.addEventListener(event, (e) => { e.preventDefault(); zone.classList.add("over"); });
  }
  for (const event of ["dragleave", "dragend", "drop"]) {
    zone.addEventListener(event, (e) => { e.preventDefault(); zone.classList.remove("over"); });
  }
  zone.addEventListener("drop", (event) => {
    const dropped = event.dataTransfer && event.dataTransfer.files;
    if (dropped && dropped.length) { setSource(dropped[0], null); markSelection(null); }
  });

  // Dropping anywhere on the page should work too.
  window.addEventListener("dragover", (e) => e.preventDefault());
  window.addEventListener("drop", (e) => {
    e.preventDefault();
    const dropped = e.dataTransfer && e.dataTransfer.files;
    if (dropped && dropped.length && !zone.contains(e.target)) {
      setSource(dropped[0], null); markSelection(null);
    }
  });
}

/* -------------------------------------------------------------- library --- */
async function scanLibrary() {
  const list = $("liblist");
  list.innerHTML = '<li class="empty">Scanning&hellip;</li>';
  try {
    const data = await api("/api/assets?limit=2000");
    state.assets = data.assets || [];
    renderLibrary("");
  } catch (error) {
    list.innerHTML = `<li class="empty">${escapeHtml(error.message)}</li>`;
  }
}

function renderLibrary(filter) {
  const list = $("liblist");
  const needle = filter.trim().toLowerCase();
  const matches = state.assets.filter(
    (asset) => !needle || asset.relative.toLowerCase().includes(needle)
  );
  list.innerHTML = "";
  if (!matches.length) {
    list.innerHTML = '<li class="empty">No matching assets.</li>';
    return;
  }
  for (const asset of matches.slice(0, 800)) {
    const item = document.createElement("li");
    const ext = asset.extension.replace(".", "");
    item.innerHTML = `<span class="ext">${escapeHtml(ext)}</span> ${escapeHtml(asset.relative)}`;
    item.title = asset.path;
    item.dataset.path = asset.path;
    item.addEventListener("click", () => {
      setSource(null, asset.path);
      markSelection(asset.path);
    });
    list.appendChild(item);
  }
}

function markSelection(path) {
  for (const item of $("liblist").querySelectorAll("li")) {
    item.classList.toggle("sel", !!path && item.dataset.path === path);
  }
}

/* -------------------------------------------------------------- convert --- */
async function convert() {
  clearResult();
  $("convert").disabled = true;
  showProgress("starting", 0, "uploading");

  try {
    let payload;
    if (state.file) {
      const form = new FormData();
      form.append("file", state.file, state.file.name);
      form.append("options", new Blob([JSON.stringify(readOptions())], { type: "application/json" }));
      payload = await api("/api/convert", { method: "POST", body: form });
    } else {
      payload = await api("/api/convert-path", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ path: state.path, options: readOptions() }),
      });
    }
    state.jobId = payload.job_id;
    pollJob();
  } catch (error) {
    showError(error.message);
    $("convert").disabled = false;
    hideProgress();
  }
}

function pollJob() {
  clearInterval(state.pollTimer);
  state.pollTimer = setInterval(async () => {
    try {
      const status = await api(`/api/job/${state.jobId}`);
      showProgress(status.stage, status.fraction, status.message);
      if (!status.done) return;
      clearInterval(state.pollTimer);
      $("convert").disabled = false;
      if (status.error) { showError(status.error); hideProgress(); return; }
      renderReport(status.report);
      await loadPreview();
      hideProgress();
    } catch (error) {
      clearInterval(state.pollTimer);
      showError(error.message);
      $("convert").disabled = false;
      hideProgress();
    }
  }, 220);
}

function showProgress(stage, fraction, message) {
  $("progress").hidden = false;
  $("fill").style.width = `${Math.round((fraction || 0) * 100)}%`;
  $("stage").textContent = `${stage || ""} ${message ? "· " + message : ""}`;
}
function hideProgress() { $("progress").hidden = true; }
function clearResult() { $("result").innerHTML = ""; $("warnings").innerHTML = ""; }
function showError(message) {
  $("result").innerHTML = `<div class="error"><strong>Failed:</strong> ${escapeHtml(message)}</div>`;
}

function renderReport(report) {
  if (!report) return;
  const scene = report.scene || {};
  const exported = report.export || {};
  const rows = [
    ["vertices", scene.vertices],
    ["polygons", scene.polygons],
    ["meshes", scene.meshes],
    ["bones", scene.bones],
    ["materials", scene.materials],
    ["morphs", scene.morphs],
    ["FBX triangles", exported.triangles],
    ["file size", exported.size_bytes ? `${(exported.size_bytes / 1024).toFixed(1)} KB` : null],
    ["time", report.elapsed_seconds ? `${report.elapsed_seconds.toFixed(2)} s` : null],
  ].filter(([, value]) => value !== null && value !== undefined);

  const stats = rows.map(([label, value]) =>
    `<dt>${label}</dt><dd>${escapeHtml(String(value))}</dd>`).join("");

  $("result").innerHTML = `
    <dl class="stats">${stats}</dl>
    <a class="download" href="/api/download/${state.jobId}" download>Download FBX</a>`;

  const warnings = report.warnings || [];
  if (warnings.length) {
    $("warnings").innerHTML =
      `<strong>${warnings.length} warning${warnings.length === 1 ? "" : "s"}</strong>` +
      `<ul>${warnings.slice(0, 60).map((w) => `<li>${escapeHtml(w)}</li>`).join("")}` +
      (warnings.length > 60 ? `<li>&hellip;and ${warnings.length - 60} more</li>` : "") + `</ul>`;
  }
}

/* ================================================================ WebGL === */
const viewer = {
  gl: null, program: null, lineProgram: null,
  mesh: null, wire: null, bones: null,
  yaw: -0.6, pitch: 0.25, distance: 300, target: [0, 90, 0],
  showWire: false, showBones: true, spin: false,
  needsDraw: true, dragging: null,
};

function initGL() {
  const canvas = $("gl");
  const gl = canvas.getContext("webgl", { antialias: true, alpha: false });
  if (!gl) {
    $("overlay").textContent = "WebGL is unavailable in this browser";
    return;
  }
  viewer.gl = gl;
  gl.enable(gl.DEPTH_TEST);
  gl.clearColor(0.05, 0.06, 0.075, 1);

  viewer.program = buildProgram(gl, MESH_VS, MESH_FS, ["aPos", "aNormal"]);
  viewer.lineProgram = buildProgram(gl, LINE_VS, LINE_FS, ["aPos"]);

  attachControls(canvas);
  requestAnimationFrame(drawLoop);
}

function buildProgram(gl, vsSource, fsSource, attributes) {
  const compile = (type, source) => {
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      throw new Error(gl.getShaderInfoLog(shader));
    }
    return shader;
  };
  const program = gl.createProgram();
  gl.attachShader(program, compile(gl.VERTEX_SHADER, vsSource));
  gl.attachShader(program, compile(gl.FRAGMENT_SHADER, fsSource));
  attributes.forEach((name, index) => gl.bindAttribLocation(program, index, name));
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    throw new Error(gl.getProgramInfoLog(program));
  }
  return program;
}

const MESH_VS = `
attribute vec3 aPos;
attribute vec3 aNormal;
uniform mat4 uMVP;
uniform mat3 uNormalMat;
varying vec3 vNormal;
void main() {
  vNormal = normalize(uNormalMat * aNormal);
  gl_Position = uMVP * vec4(aPos, 1.0);
}`;

const MESH_FS = `
precision mediump float;
varying vec3 vNormal;
uniform vec3 uColour;
void main() {
  vec3 light = normalize(vec3(0.4, 0.8, 0.6));
  float lambert = max(dot(vNormal, light), 0.0);
  float back = max(dot(-vNormal, light), 0.0) * 0.35;
  vec3 colour = uColour * (0.28 + 0.72 * lambert + back);
  gl_FragColor = vec4(colour, 1.0);
}`;

const LINE_VS = `
attribute vec3 aPos;
uniform mat4 uMVP;
uniform float uPointSize;
void main() {
  gl_PointSize = uPointSize;
  gl_Position = uMVP * vec4(aPos, 1.0);
}`;

const LINE_FS = `
precision mediump float;
uniform vec4 uLineColour;
void main() { gl_FragColor = uLineColour; }`;

/* ------------------------------------------------------------ tiny mat4 --- */
function mat4Identity() {
  return [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1];
}
function mat4Multiply(a, b) {
  const out = new Array(16);
  for (let column = 0; column < 4; column++) {
    for (let row = 0; row < 4; row++) {
      out[column * 4 + row] =
        a[row] * b[column * 4] +
        a[4 + row] * b[column * 4 + 1] +
        a[8 + row] * b[column * 4 + 2] +
        a[12 + row] * b[column * 4 + 3];
    }
  }
  return out;
}
function mat4Perspective(fovY, aspect, near, far) {
  const f = 1 / Math.tan(fovY / 2);
  const nf = 1 / (near - far);
  return [
    f / aspect, 0, 0, 0,
    0, f, 0, 0,
    0, 0, (far + near) * nf, -1,
    0, 0, 2 * far * near * nf, 0,
  ];
}
function mat4LookAt(eye, centre, up) {
  const z = normalize3(sub3(eye, centre));
  const x = normalize3(cross3(up, z));
  const y = cross3(z, x);
  return [
    x[0], y[0], z[0], 0,
    x[1], y[1], z[1], 0,
    x[2], y[2], z[2], 0,
    -dot3(x, eye), -dot3(y, eye), -dot3(z, eye), 1,
  ];
}
function sub3(a, b) { return [a[0]-b[0], a[1]-b[1], a[2]-b[2]]; }
function cross3(a, b) {
  return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]];
}
function dot3(a, b) { return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]; }
function length3(a) { return Math.hypot(a[0], a[1], a[2]); }
function normalize3(a) { const l = length3(a) || 1; return [a[0]/l, a[1]/l, a[2]/l]; }

function normalMatrix(mvp) {
  // Upper-left 3x3 of the model-view matrix; the preview applies no non-uniform
  // scale, so the MVP's rotation part is good enough for lighting.
  return [mvp[0], mvp[1], mvp[2], mvp[4], mvp[5], mvp[6], mvp[8], mvp[9], mvp[10]];
}

/* ------------------------------------------------------------- controls --- */
function attachControls(canvas) {
  let last = null;
  canvas.addEventListener("pointerdown", (event) => {
    canvas.setPointerCapture(event.pointerId);
    viewer.dragging = event.button === 2 || event.shiftKey ? "pan" : "orbit";
    last = { x: event.clientX, y: event.clientY };
  });
  canvas.addEventListener("pointermove", (event) => {
    if (!viewer.dragging || !last) return;
    const dx = event.clientX - last.x;
    const dy = event.clientY - last.y;
    last = { x: event.clientX, y: event.clientY };
    if (viewer.dragging === "orbit") {
      viewer.yaw -= dx * 0.008;
      viewer.pitch = Math.max(-1.5, Math.min(1.5, viewer.pitch + dy * 0.008));
    } else {
      const scale = viewer.distance * 0.0016;
      const right = [Math.cos(viewer.yaw), 0, -Math.sin(viewer.yaw)];
      const upv = [0, 1, 0];
      for (let i = 0; i < 3; i++) {
        viewer.target[i] -= right[i] * dx * scale + upv[i] * -dy * scale;
      }
    }
    viewer.needsDraw = true;
  });
  for (const event of ["pointerup", "pointercancel"]) {
    canvas.addEventListener(event, () => { viewer.dragging = null; last = null; });
  }
  canvas.addEventListener("contextmenu", (event) => event.preventDefault());
  canvas.addEventListener("wheel", (event) => {
    event.preventDefault();
    viewer.distance = Math.max(5, viewer.distance * (event.deltaY > 0 ? 1.1 : 0.9));
    viewer.needsDraw = true;
  }, { passive: false });
}

/* --------------------------------------------------------------- loading -- */
function makeBuffer(gl, data, type) {
  const buffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
  gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
  return buffer;
}

function setPreview(data) {
  const gl = viewer.gl;
  if (!gl) return;
  const meshes = data.meshes || [];
  if (!meshes.length) {
    viewer.mesh = null; viewer.wire = null; viewer.bones = null;
    $("overlay").textContent = data.error ? `preview unavailable: ${data.error}` : "no geometry";
    $("legend").innerHTML = "";
    viewer.needsDraw = true;
    return;
  }
  const mesh = meshes[0];

  viewer.mesh = {
    positions: makeBuffer(gl, new Float32Array(mesh.vertices), gl.FLOAT),
    normals: makeBuffer(gl, new Float32Array(mesh.normals), gl.FLOAT),
    indices: makeBuffer(gl, new Uint32Array(mesh.triangles), gl.UNSIGNED_INT),
    count: mesh.triangles.length,
    colour: mesh.colour || [0.72, 0.72, 0.74],
  };

  // Wireframe: turn the face outline list into line segments.
  const outline = mesh.wireframe || [];
  const segments = [];
  let face = [];
  for (const index of outline) {
    if (index === -1) {
      for (let i = 0; i < face.length; i++) {
        const a = face[i]; const b = face[(i + 1) % face.length];
        segments.push(...sliceVertex(mesh.vertices, a), ...sliceVertex(mesh.vertices, b));
      }
      face = [];
    } else {
      face.push(index);
    }
  }
  viewer.wire = segments.length
    ? { buffer: makeBuffer(gl, new Float32Array(segments), gl.FLOAT), count: segments.length / 3 }
    : null;

  // Skeleton: parent -> child segments.
  const byId = new Map((data.bones || []).map((bone) => [bone.id, bone]));
  const lines = [];
  const joints = [];
  for (const bone of data.bones || []) {
    const parent = bone.parent ? byId.get(bone.parent) : null;
    if (parent && parent.position) {
      lines.push(...parent.position, ...bone.position);
    }
    if (bone.end) lines.push(...bone.position, ...bone.end);
    joints.push(...bone.position);
  }
  viewer.bones = lines.length
    ? {
        lines: makeBuffer(gl, new Float32Array(lines), gl.FLOAT),
        lineCount: lines.length / 3,
        joints: makeBuffer(gl, new Float32Array(joints), gl.FLOAT),
        jointCount: joints.length / 3,
      }
    : null;

  // 32-bit indices need an extension in WebGL 1; fall back to nothing if absent.
  const ext = gl.getExtension("OES_element_index_uint");
  viewer.uintIndices = !!ext;
  if (!ext) $("overlay").textContent = "OES_element_index_uint missing -- mesh not drawn";

  fitTo(data.bounds || mesh.bounds);
  $("ovl-title").textContent = data.name || "";
  renderLegend(mesh, data);
  viewer.needsDraw = true;
}

function sliceVertex(vertices, index) {
  return [vertices[index * 3], vertices[index * 3 + 1], vertices[index * 3 + 2]];
}

function fitTo(bounds) {
  if (!bounds) return;
  const centre = [0, 1, 2].map((i) => (bounds.min[i] + bounds.max[i]) / 2);
  const size = [0, 1, 2].map((i) => bounds.max[i] - bounds.min[i]);
  viewer.target = centre;
  viewer.distance = Math.max(...size) * 1.6 + 10;
}

function renderLegend(mesh, data) {
  const stats = data.stats || {};
  const items = [
    `${mesh.vertex_count.toLocaleString()} verts`,
    `${mesh.polygon_count.toLocaleString()} polys`,
    mesh.has_uvs ? `${mesh.uv_corners.toLocaleString()} uv corners` : "no uvs",
    `${mesh.weighted_vertices.toLocaleString()} weighted`,
    `${(data.bones || []).length} bones`,
    stats.materials ? `${stats.materials} materials` : null,
    mesh.subdivision ? "subdivision surface" : null,
  ].filter(Boolean);
  $("legend").innerHTML = items.map((item) => `<span>${escapeHtml(item)}</span>`).join("");
}

/* ---------------------------------------------------------------- render -- */
function drawLoop() {
  requestAnimationFrame(drawLoop);
  if (viewer.spin) { viewer.yaw += 0.006; viewer.needsDraw = true; }
  if (!viewer.needsDraw) return;
  viewer.needsDraw = false;
  draw();
}

function draw() {
  const gl = viewer.gl;
  if (!gl) return;
  const canvas = gl.canvas;
  const width = canvas.clientWidth || canvas.width;
  const height = canvas.clientHeight || canvas.height;
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width; canvas.height = height;
  }
  gl.viewport(0, 0, canvas.width, canvas.height);
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
  if (!viewer.mesh) return;

  const eye = [
    viewer.target[0] + viewer.distance * Math.cos(viewer.pitch) * Math.sin(viewer.yaw),
    viewer.target[1] + viewer.distance * Math.sin(viewer.pitch),
    viewer.target[2] + viewer.distance * Math.cos(viewer.pitch) * Math.cos(viewer.yaw),
  ];
  const projection = mat4Perspective(0.9, canvas.width / canvas.height, 0.5, viewer.distance * 12 + 500);
  const view = mat4LookAt(eye, viewer.target, [0, 1, 0]);
  const mvp = mat4Multiply(projection, view);

  // --- shaded mesh ---
  if (viewer.uintIndices && !viewer.showWire) {
    const program = viewer.program;
    gl.useProgram(program);
    gl.uniformMatrix4fv(gl.getUniformLocation(program, "uMVP"), false, new Float32Array(mvp));
    gl.uniformMatrix3fv(gl.getUniformLocation(program, "uNormalMat"), false,
      new Float32Array(normalMatrix(view)));
    gl.uniform3fv(gl.getUniformLocation(program, "uColour"), new Float32Array(viewer.mesh.colour));
    gl.bindBuffer(gl.ARRAY_BUFFER, viewer.mesh.positions);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, viewer.mesh.normals);
    gl.enableVertexAttribArray(1);
    gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, viewer.mesh.indices);
    gl.drawElements(gl.TRIANGLES, viewer.mesh.count, gl.UNSIGNED_INT, 0);
    gl.disableVertexAttribArray(1);
  }

  // --- wireframe ---
  if (viewer.wire && (viewer.showWire || viewer.showBones === false)) {
    drawLines(viewer.wire.buffer, viewer.wire.count, gl.LINES,
      viewer.showWire ? [0.35, 0.75, 1.0, 0.55] : [0.4, 0.45, 0.5, 1.0], mvp);
  }

  // --- skeleton ---
  if (viewer.bones && viewer.showBones) {
    drawLines(viewer.bones.lines, viewer.bones.lineCount, gl.LINES, [0.35, 0.9, 0.55, 0.85], mvp);
    drawLines(viewer.bones.joints, viewer.bones.jointCount, gl.POINTS, [1.0, 0.85, 0.3, 1.0], mvp, 4.0);
  }
}

function drawLines(buffer, count, mode, colour, mvp, pointSize) {
  const gl = viewer.gl;
  const program = viewer.lineProgram;
  gl.useProgram(program);
  gl.uniformMatrix4fv(gl.getUniformLocation(program, "uMVP"), false, new Float32Array(mvp));
  gl.uniform4fv(gl.getUniformLocation(program, "uLineColour"), new Float32Array(colour));
  gl.uniform1f(gl.getUniformLocation(program, "uPointSize"), pointSize || 1.0);
  gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
  gl.enableVertexAttribArray(0);
  gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);
  gl.drawArrays(mode, 0, count);
  gl.disableVertexAttribArray(0);
}

/* --------------------------------------------------------------- preview -- */
async function loadPreview() {
  try {
    const data = await api(`/api/preview/${state.jobId}`);
    setPreview(data);
  } catch (error) {
    $("overlay").textContent = `preview failed: ${error.message}`;
  }
}

/* ------------------------------------------------------------------ boot -- */
async function boot() {
  initDropZone();
  try {
    const info = await api("/api/info");
    $("meta").textContent =
      `dazconv ${info.version}\nPython ${info.python}\n` +
      (info.content_roots.length ? `${info.content_roots.length} content root(s)` : "no content root");
    $("rootlabel").textContent = info.content_roots.length
      ? `(${info.content_roots.length} root${info.content_roots.length === 1 ? "" : "s"})`
      : "(not configured)";
    $("footerinfo").textContent =
      `inputs ${info.inputs.join(", ")}  ·  output ${info.outputs.join(", ")}  ·  ` +
      (info.content_roots[0] || "drop files to convert");
    applyOptions(info.default_options);
    if (info.content_roots.length) scanLibrary();
  } catch (error) {
    $("meta").textContent = `backend error: ${error.message}`;
  }

  $("libscan").addEventListener("click", scanLibrary);
  $("libfilter").addEventListener("input", (event) => renderLibrary(event.target.value));
  $("convert").addEventListener("click", convert);

  const toggle = (id, key) => {
    $(id).addEventListener("click", () => {
      viewer[key] = !viewer[key];
      $(id).classList.toggle("on", viewer[key]);
      viewer.needsDraw = true;
    });
  };
  toggle("btn-wire", "showWire");
  toggle("btn-bones", "showBones");
  toggle("btn-spin", "spin");
  $("btn-bones").classList.add("on");
  $("btn-fit").addEventListener("click", async () => {
    if (!state.jobId) return;
    try {
      const data = await api(`/api/preview/${state.jobId}`);
      fitTo(data.bounds || (data.meshes[0] && data.meshes[0].bounds));
      viewer.needsDraw = true;
    } catch { /* nothing loaded yet */ }
  });

  initGL();
}

document.addEventListener("DOMContentLoaded", boot);
