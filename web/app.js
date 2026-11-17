/* Myeye AI - frontend logic (vanilla JS, zero dependencies) */
"use strict";

const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];

/* --------------------------------------------------------------- lightbox */
const lb = $("#lightbox");
const lbImg = $("#lightboxImg");
function closeLightbox() {
  if (!lb || !lbImg) return;
  lb.setAttribute("hidden", "");
  lbImg.removeAttribute("src");
}
if (lb && lbImg) {
  document.addEventListener("click", (e) => {
    const img = e.target.closest("img");
    if (!img || img === lbImg) return;
    const src = img.getAttribute("src") || "";
    if (!src.startsWith("/media/")) return;
    lbImg.src = src;
    lb.removeAttribute("hidden");
  });
  lb.addEventListener("click", closeLightbox);
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") closeLightbox();
  });
}

const CLASS_HEX = {
  person: "#34d399", car: "#3b82f6", motorcycle: "#a78bfa",
  bus: "#f59e0b", truck: "#facc15", bicycle: "#22d3ee", train: "#f87171",
};
const VEHICLE_LABELS = ["car", "motorcycle", "bus", "truck"];
const COLORS_LIST = [
  ["White", "#f1f5f9"], ["Silver", "#cbd5e1"], ["Gray", "#6b7280"], ["Black", "#111827"],
  ["Red", "#ef4444"], ["Orange", "#f97316"], ["Yellow", "#eab308"], ["Green", "#22c55e"],
  ["Cyan", "#06b6d4"], ["Blue", "#3b82f6"], ["Purple", "#a855f7"], ["Brown", "#92400e"],
];
const state = {
  page: "live", stats: null, objCat: "", objColor: "", evType: "",
  objLimit: 60, polling: null,
};

/* ---------------------------------------------------------------- toasts */
function toast(msg, kind = "") {
  const t = document.createElement("div");
  t.className = `toast ${kind}`;
  t.textContent = msg;
  $("#toasts").appendChild(t);
  setTimeout(() => t.remove(), 4200);
}

async function api(path, opts) {
  const r = await fetch(path, opts);
  if (!r.ok) {
    let msg = r.statusText;
    try { msg = (await r.json()).detail || msg; } catch {}
    throw new Error(msg);
  }
  return r.json();
}

/* ------------------------------------------------------------ navigation */
$("#nav").addEventListener("click", (e) => {
  const item = e.target.closest(".nav-item");
  if (!item) return;
  $$(".nav-item").forEach((n) => n.classList.toggle("active", n === item));
  const page = item.dataset.page;
  $$(".page").forEach((p) => p.classList.toggle("active", p.id === `page-${page}`));
  $("#pageTitle").textContent = item.querySelector("span").textContent;
  state.page = page;
  refreshPage(true);
});

function refreshPage(force) {
  pollStats();
  if (state.page === "objects") loadObjects(force);
  if (state.page === "plates") loadPlates(force);
  if (state.page === "faces") loadFaces(force);
  if (state.page === "events") loadEvents(force);
  if (state.page === "settings") loadSettings();
}

/* ------------------------------------------------------------ source ctl */
$("#sourceType").addEventListener("change", () => {
  const t = $("#sourceType").value;
  $("#fileInput").style.display = t === "file" ? "" : "none";
  $("#rtspInput").style.display = t === "rtsp" ? "" : "none";
});

$("#btnStart").addEventListener("click", async () => {
  const t = $("#sourceType").value;
  try {
    let body;
    if (t === "file") {
      const f = $("#fileInput").files[0];
      if (!f) return toast("Choose a video file first", "bad");
      toast("Uploading video…");
      const fd = new FormData();
      fd.append("file", f);
      const up = await api("/api/upload", { method: "POST", body: fd });
      body = { type: "file", value: up.path };
    } else if (t === "rtsp") {
      body = { type: "rtsp", value: $("#rtspInput").value.trim() };
    } else if (t === "demo") {
      body = { type: "demo" };
    } else {
      body = { type: "webcam", value: 0 };
    }
    await api("/api/source", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    attachStream();
    toast("Source started", "ok");
  } catch (e) { toast(`Start failed: ${e.message}`, "bad"); }
});

$("#btnStop").addEventListener("click", async () => {
  try {
    await api("/api/stop", { method: "POST", headers: { "Content-Type": "application/json" }, body: "{}" });
    detachStream();
    toast("Source stopped", "ok");
  } catch (e) { toast(e.message, "bad"); }
});

$("#btnRestart").addEventListener("click", () => {
  if ($("#btnStop").disabled) return toast("No source running", "bad");
  $("#btnStop").click(); setTimeout(() => $("#btnStart").click(), 700);
});

$("#btnSnapshot").addEventListener("click", async () => {
  try {
    const r = await fetch("/api/frame");
    if (!r.ok) throw new Error("no frame");
    const b = await r.blob();
    const a = document.createElement("a");
    a.href = URL.createObjectURL(b);
    a.download = `myeye_snapshot_${Date.now()}.jpg`;
    a.click();
  } catch (e) { toast("Snapshot failed — start a source first", "bad"); }
});

let streamRetry = null;
let streamBroken = false;
function attachStream() {
  const img = $("#stream");
  clearTimeout(streamRetry);
  streamBroken = false;
  img.onerror = () => {
    // server restart / dropped MJPEG connection: reconnect while running
    streamBroken = true;
    clearTimeout(streamRetry);
    streamRetry = setTimeout(() => {
      if (state.stats && state.stats.running) attachStream();
    }, 1500);
  };
  img.src = `/api/stream?t=${Date.now()}`;
  img.style.display = "";
  $("#videoOff").style.display = "none";
}
function detachStream() {
  clearTimeout(streamRetry);
  const img = $("#stream");
  img.onerror = null;
  img.removeAttribute("src");
  img.style.display = "none";
  $("#videoOff").style.display = "";
}
document.addEventListener("visibilitychange", () => {
  if (!document.hidden && streamBroken && state.stats && state.stats.running) {
    attachStream();
  }
});

/* ------------------------------------------------------------ stats loop */
async function pollStats() {
  try {
    const s = await api("/api/stats");
    state.stats = s;
    $("#fpsPill").textContent = `FPS ${s.running ? s.fps : "—"}`;
    const pill = $("#statusPill");
    pill.className = `pill ${s.running ? "online" : "offline"}`;
    $("#statusText").textContent = s.running
      ? `Online · ${s.source.type}` : "Offline";
    $("#btnStop").disabled = !s.running;
    $("#btnStart").disabled = s.running;
    $("#srcInfo").textContent = s.running
      ? `${s.source.type}: ${s.source.value} · ${s.frame_size[0]}×${s.frame_size[1]}` : "";

    // keep the video element in sync with the backend state
    const img = $("#stream");
    if (s.running && (!img.src || !img.src.includes("/api/stream"))) attachStream();
    if (!s.running && img.src.includes("/api/stream")) detachStream();

    // active objects
    const entries = Object.entries(s.active).sort((a, b) => b[1] - a[1]);
    $("#activeList").innerHTML = entries.length
      ? entries.map(([k, v]) => `<div class="obj-row">
          <span style="display:flex;align-items:center;gap:8px">
          <i style="width:9px;height:9px;border-radius:3px;background:${CLASS_HEX[k] || "#888"}"></i>
          <b class="n">${k}</b></span><span class="c">${v} active · ${s.unique[k] || 0} unique</span>
        </div>`).join("")
      : `<div class="empty">—</div>`;

    drawTimeline($("#timelineChart"), s.timeline);
    drawClassChart($("#classChart"), s.unique);

    // nav event counter
    const plates = (s.event_counts.plate || 0) + (s.event_counts.face || 0);
    const navPill = $("#navEventCount");
    navPill.style.display = plates ? "" : "none";
    navPill.textContent = plates > 99 ? "99+" : plates;
  } catch {}
}

setInterval(() => {
  if (state.page === "live") pollStats();
}, 1000);
setInterval(() => {
  if (state.page === "events") loadEvents();
  if (state.page === "plates") loadPlates();
  if (state.page === "faces") { loadFaces(); }
  if (state.page === "objects") loadObjects();
}, 4000);

/* ---------------------------------------------------------------- charts */
function chartCtx(cv) {
  const dpr = window.devicePixelRatio || 1;
  const w = cv.clientWidth, h = cv.clientHeight;
  cv.width = w * dpr; cv.height = h * dpr;
  const ctx = cv.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return [ctx, w, h];
}

function drawTimeline(cv, points) {
  const [ctx, W, H] = chartCtx(cv);
  ctx.clearRect(0, 0, W, H);
  const pad = { l: 26, r: 8, t: 10, b: 18 };
  if (!points || points.length < 2) {
    ctx.fillStyle = "#8b94a7"; ctx.font = "12px Inter, sans-serif";
    ctx.textAlign = "center"; ctx.fillText("waiting for data…", W / 2, H / 2);
    return;
  }
  const maxV = Math.max(4, ...points.map((p) => p[1]));
  const t0 = points[0][0], t1 = points[points.length - 1][0] || t0 + 1;
  const X = (t) => pad.l + ((t - t0) / Math.max(1, t1 - t0)) * (W - pad.l - pad.r);
  const Y = (v) => H - pad.b - (v / maxV) * (H - pad.t - pad.b);

  ctx.strokeStyle = "rgba(255,255,255,.07)"; ctx.lineWidth = 1;
  ctx.fillStyle = "#8b94a7"; ctx.font = "10px Inter, sans-serif"; ctx.textAlign = "right";
  for (let i = 0; i <= 3; i++) {
    const v = (maxV / 3) * i, y = Y(v);
    ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(W - pad.r, y); ctx.stroke();
    ctx.fillText(Math.round(v), pad.l - 6, y + 3);
  }
  const grad = ctx.createLinearGradient(0, 0, 0, H);
  grad.addColorStop(0, "rgba(99,102,241,.45)"); grad.addColorStop(1, "rgba(99,102,241,0)");
  ctx.beginPath(); ctx.moveTo(X(points[0][0]), Y(points[0][1]));
  points.forEach((p) => ctx.lineTo(X(p[0]), Y(p[1])));
  ctx.lineTo(X(points[points.length - 1][0]), H - pad.b);
  ctx.lineTo(X(points[0][0]), H - pad.b); ctx.closePath();
  ctx.fillStyle = grad; ctx.fill();
  ctx.beginPath(); ctx.moveTo(X(points[0][0]), Y(points[0][1]));
  points.forEach((p) => ctx.lineTo(X(p[0]), Y(p[1])));
  ctx.strokeStyle = "#6366f1"; ctx.lineWidth = 2.2; ctx.lineJoin = "round"; ctx.stroke();
  const last = points[points.length - 1];
  ctx.beginPath(); ctx.arc(X(last[0]), Y(last[1]), 3.4, 0, 7);
  ctx.fillStyle = "#22d3ee"; ctx.fill();
}

function drawClassChart(cv, unique) {
  const entries = Object.entries(unique).sort((a, b) => b[1] - a[1]).slice(0, 6);
  const [ctx, W, H] = chartCtx(cv);
  ctx.clearRect(0, 0, W, H);
  $("#classLegend").innerHTML = entries.map(([k, v]) =>
    `<span class="li"><i style="background:${CLASS_HEX[k] || "#888"}"></i>${k} · ${v}</span>`).join("")
    || `<span class="li">no tracked objects yet</span>`;
  if (!entries.length) {
    ctx.fillStyle = "#8b94a7"; ctx.font = "12px Inter, sans-serif"; ctx.textAlign = "center";
    ctx.fillText("waiting for data…", W / 2, H / 2);
    return;
  }
  const maxV = Math.max(...entries.map((e) => e[1]));
  const bw = W / entries.length;
  entries.forEach(([k, v], i) => {
    const h = (v / maxV) * (H - 30);
    const x = i * bw + bw * 0.18, y = H - h - 16;
    const g = ctx.createLinearGradient(0, y, 0, H);
    g.addColorStop(0, CLASS_HEX[k] || "#6366f1"); g.addColorStop(1, CLASS_HEX[k] + "55");
    ctx.fillStyle = g;
    roundRect(ctx, x, y, bw * 0.64, h, 6); ctx.fill();
    ctx.fillStyle = "#e8ecf4"; ctx.font = "700 11px Inter, sans-serif"; ctx.textAlign = "center";
    ctx.fillText(v, x + bw * 0.32, y - 4);
    ctx.fillStyle = "#8b94a7"; ctx.font = "10px Inter, sans-serif";
    ctx.fillText(k.slice(0, 9), x + bw * 0.32, H - 4);
  });
}
function roundRect(ctx, x, y, w, h, r) {
  r = Math.min(r, w / 2, h / 2, 8);
  ctx.beginPath();
  ctx.moveTo(x + r, y); ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r); ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r); ctx.closePath();
}

/* ---------------------------------------------------------------- events */
function fmtThumb(row) {
  return row.thumb ? `/media/${row.thumb}` : null;
}
function fmtTime(ts) {
  const d = new Date(ts * 1000);
  return d.toLocaleDateString([], { month: "short", day: "numeric" }) + " " +
    d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}
function emptyHTML(icon, msg) {
  return `<div class="empty" style="grid-column:1/-1"><div class="icon">${icon}</div>${msg}</div>`;
}

function renderFeed(rows, el, icon, msg) {
  el.innerHTML = rows.length ? rows.map((r) => {
    const th = fmtThumb(r);
    const label = r.type === "plate" ? `<span class="mono">${r.label}</span>` : r.label;
    const extra = r.extra ? JSON.parse(r.extra) : {};
    const sub = [extra.color, extra.vehicle, extra.conf ? `conf ${extra.conf}` : null,
                 extra.sim !== null && extra.sim !== undefined ? `sim ${extra.sim}` : null]
      .filter(Boolean).join(" · ");
    return `<div class="feed-row">
      ${th ? `<img class="ev-${r.type}" src="${th}" loading="lazy" title="Click to enlarge">` : `<div class="thumb-sm" style="display:grid;place-items:center;color:#8b94a7">▣</div>`}
      <span class="tag ${r.type}">${r.type}</span>
      <div><div class="lbl">${label}</div>${sub ? `<div class="muted">${sub}</div>` : ""}</div>
      <span class="tm">${fmtTime(r.ts)}</span>
    </div>`;
  }).join("") : emptyHTML(icon, msg);
}

$$("[data-ev]").forEach((chip) => chip.addEventListener("click", () => {
  $$("[data-ev]").forEach((c) => c.classList.toggle("on", c === chip));
  state.evType = chip.dataset.ev;
  loadEvents(true);
}));

async function loadEvents(force) {
  try {
    const q = new URLSearchParams({ limit: 80 });
    if (state.evType) q.set("type", state.evType);
    const { rows } = await api(`/api/events?${q}`);
    renderFeed(rows, $("#eventFeed"), "📭", "No events yet.");
  } catch {}
}

$("#evExport").addEventListener("click", () =>
  window.open(`/api/events/export${state.evType ? `?type=${state.evType}` : ""}`, "_blank"));
$("#evClear").addEventListener("click", async () => {
  if (!confirm("Delete ALL events?")) return;
  await api("/api/events", { method: "DELETE" });
  toast("Events cleared", "ok"); loadEvents(true);
});

/* --------------------------------------------------------------- objects */
const CATEGORY_OF = (label) => (label === "person" ? "person"
  : VEHICLE_LABELS.includes(label) ? "vehicle" : "other");

function buildColorChips() {
  $("#colorChips").innerHTML = COLORS_LIST.map(([n, hex]) =>
    `<span class="chip" data-color="${n}"><i class="cdot" style="background:${hex}"></i>${n}</span>`).join("");
  $$("#colorChips .chip").forEach((c) => c.addEventListener("click", () => {
    $$("#colorChips .chip").forEach((x) => x.classList.toggle("on", x === c));
    state.objColor = state.objColor === c.dataset.color ? "" : c.dataset.color;
    state.objLimit = 60; loadObjects(true);
  }));
}
$$("[data-cat]").forEach((chip) => chip.addEventListener("click", () => {
  $$("[data-cat]").forEach((c) => c.classList.toggle("on", c === chip));
  state.objCat = chip.dataset.cat;
  state.objLimit = 60; loadObjects(true);
}));
$("#objSearch").addEventListener("input", debounce(() => { state.objLimit = 60; loadObjects(true); }, 350));
$("#objReset").addEventListener("click", () => {
  state.objCat = ""; state.objColor = ""; $("#objSearch").value = "";
  $$("[data-cat]").forEach((c) => c.classList.toggle("on", !c.dataset.cat));
  $$("#colorChips .chip").forEach((c) => c.classList.remove("on"));
  loadObjects(true);
});
$("#objMore").addEventListener("click", () => { state.objLimit += 60; loadObjects(true); });

function debounce(fn, ms) {
  let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); };
}

async function loadObjects() {
  try {
    const search = $("#objSearch").value.trim();
    const { rows } = await api(`/api/events?type=object&limit=${state.objLimit}` +
      (search ? `&search=${encodeURIComponent(search)}` : ""));
    const filtered = rows.filter((r) => {
      if (state.objCat && CATEGORY_OF(r.label) !== state.objCat) return false;
      if (state.objColor) {
        let extra = {}; try { extra = JSON.parse(r.extra || "{}"); } catch {}
        if (extra.color !== state.objColor) return false;
      }
      return true;
    });
    $("#objCount").textContent = `${filtered.length} objects${state.objCat ? ` · ${state.objCat}` : ""}`;
    $("#objCards").innerHTML = filtered.length ? filtered.map((r) => {
      let extra = {}; try { extra = JSON.parse(r.extra || "{}"); } catch {}
      const th = fmtThumb(r);
      return `<div class="obj-card">
        ${th ? `<img src="${th}" loading="lazy">` : `<div style="aspect-ratio:4/3;display:grid;place-items:center;color:#8b94a7;background:#05070c">▣</div>`}
        <div class="badges">
          <span class="badge">${r.label}${extra.track_id != null ? ` #${extra.track_id}` : ""}</span>
          ${extra.color ? `<span class="badge color-b">${extra.color}</span>` : ""}
          ${extra.conf ? `<span class="badge dim">${extra.conf}</span>` : ""}
        </div>
        <div class="t">${fmtTime(r.ts)}</div>
      </div>`;
    }).join("") : emptyHTML("🧿", "No detections match these filters.");
  } catch {}
}

/* ---------------------------------------------------------------- plates */
$("#plateSearch").addEventListener("input", debounce(() => loadPlates(true), 350));
$("#plateExport").addEventListener("click", () => window.open("/api/events/export?type=plate", "_blank"));
$("#plateClear").addEventListener("click", async () => {
  if (!confirm("Delete all plate events?")) return;
  await api("/api/events?type=plate", { method: "DELETE" });
  toast("Plate log cleared", "ok"); loadPlates(true);
});

async function loadPlates() {
  try {
    const search = $("#plateSearch").value.trim();
    const { rows } = await api(`/api/events?type=plate&limit=100` +
      (search ? `&search=${encodeURIComponent(search)}` : ""));
    $("#plateRows").innerHTML = rows.length ? rows.map((r) => {
      let extra = {}; try { extra = JSON.parse(r.extra || "{}"); } catch {}
      const th = fmtThumb(r);
      return `<tr>
        <td>${th ? `<img class="thumb-sm plate" src="${th}" loading="lazy" title="Click to enlarge">` : "—"}</td>
        <td class="mono">${r.label}</td>
        <td>${extra.vehicle || "—"}</td>
        <td>${extra.color || "—"}</td>
        <td>${extra.conf != null ? extra.conf : "—"}</td>
        <td class="muted">${fmtTime(r.ts)}</td>
      </tr>`;
    }).join("") : `<tr><td colspan="6" class="empty">No plate reads yet.</td></tr>`;
    const { plates: top } = await api("/api/top_plates");
    $("#topPlates").innerHTML = top.length ? top.map((p) =>
      `<div class="feed-row"><span class="tag plate">plate</span>
       <div class="lbl mono">${p.label}</div>
       <span class="c muted">${p.c} reads · last ${fmtTime(p.last)}</span></div>`).join("")
      : `<div class="empty">—</div>`;
    const counts = await api("/api/stats");
    $("#plateTotal").textContent = counts.event_counts.plate || 0;
    $("#plateUnique").textContent = top.length ? top.length : 0;
    $("#ocrEngine").textContent = counts.engines.ocr || "not installed";
  } catch {}
}

/* ----------------------------------------------------------------- faces */
$("#btnEnroll").addEventListener("click", async () => {
  const name = $("#faceName").value.trim();
  const files = $("#faceFiles").files;
  if (!name) return toast("Enter a name", "bad");
  if (!files.length) return toast("Choose one or more photos", "bad");
  const fd = new FormData();
  fd.append("name", name);
  [...files].forEach((f) => fd.append("files", f));
  try {
    const r = await api("/api/faces/enroll", { method: "POST", body: fd });
    toast(`Enrolled ${r.enrolled} photo(s) for ${name}`, "ok");
    $("#faceName").value = ""; $("#faceFiles").value = "";
    loadFaces(true);
  } catch (e) { toast(`Enroll failed: ${e.message}`, "bad"); }
});

$("#btnCapture").addEventListener("click", async () => {
  const name = $("#faceName").value.trim();
  if (!name) return toast("Enter a name first", "bad");
  const fd = new FormData();
  fd.append("name", name);
  try {
    await api("/api/faces/capture", { method: "POST", body: fd });
    toast(`Captured ${name} from the live stream`, "ok");
    loadFaces(true);
  } catch (e) { toast(`Capture failed: ${e.message}`, "bad"); }
});

$("#btnSearch").addEventListener("click", async () => {
  const f = $("#searchFile").files[0];
  if (!f) return toast("Choose a photo to search", "bad");
  const fd = new FormData();
  fd.append("file", f);
  try {
    const r = await api("/api/faces/search", { method: "POST", body: fd });
    $("#searchResults").innerHTML = r.matches.length
      ? r.matches.map((m) => `<div class="match-row">
          ${m.thumb ? `<img class="thumb-face" src="/media/${m.thumb}">` : `<div class="thumb-face"></div>`}
          <div style="width:120px"><b>${m.name}</b><div class="muted">${m.face_id}</div></div>
          <div class="sim-bar"><i style="width:${Math.max(0, Math.min(100, (m.similarity + 1) * 55))}%"></i></div>
          <b class="${m.match ? "ok-chip" : "muted"}">${m.similarity}</b>
        </div>`).join("")
      : `<div class="empty">${r.message || "no matches"}</div>`;
  } catch (e) { toast(`Search failed: ${e.message}`, "bad"); }
});

async function loadFaces() {
  try {
    const { rows } = await api("/api/faces");
    $("#knownCount").textContent = rows.length;
    $("#knownGrid").innerHTML = rows.length ? rows.map((f) => `
      <div class="known-card">
        <button class="del" data-del="${f.id}" title="Remove">✕</button>
        ${f.thumb ? `<img src="/media/${f.thumb}">` : `<div class="thumb-face" style="margin:auto"></div>`}
        <div class="n">${f.name}</div>
      </div>`).join("") : emptyHTML("👤", "No faces enrolled yet.");
    $$("[data-del]").forEach((b) => b.addEventListener("click", async () => {
      await api(`/api/faces/${b.dataset.del}`, { method: "DELETE" });
      loadFaces(true);
    }));
    const { rows: fev } = await api("/api/events?type=face&limit=25");
    renderFeed(fev, $("#faceEvents"), "👤", "No face events yet.");
  } catch {}
}

/* -------------------------------------------------------------- settings */
async function loadSettings() {
  try {
    const s = await api("/api/settings");
    $("#setConf").value = Math.round(s.confidence * 100);
    $("#setConfV").textContent = s.confidence.toFixed(2);
    $("#setModel").value = s.model;
    $("#setObjects").checked = s.detect_objects;
    $("#setClasses").value = s.detect_classes;
    $("#setPlates").checked = s.detect_plates;
    $("#setPlateMode").value = s.plate_mode;
    $("#setColors").checked = s.recognize_colors;
    $("#setFaces").checked = s.recognize_faces;
    $("#setFaceThr").value = Math.round(s.face_threshold * 100);
    $("#setFaceThrV").textContent = s.face_threshold.toFixed(2);
    $("#setLoop").checked = s.loop_file;

    const sys = await api("/api/system");
    const e = sys.engines;
    $("#sysInfo").innerHTML = `
      <b>Python</b> ${sys.python} · <b>OpenCV</b> ${sys.opencv}<br>
      <b>YOLO detector:</b> <span class="${e.detector ? "ok-chip" : "no-chip"}">${e.detector ? "ready" : "unavailable"}</span><br>
      <b>Plate OCR engine:</b> <span class="${e.ocr ? "ok-chip" : "no-chip"}">${e.ocr || "install easyocr / fast-plate-ocr"}</span><br>
      <b>Face engine:</b> <span class="${e.faces ? "ok-chip" : (e.faces_failed ? "no-chip" : "muted")}">${e.faces ? "ready" : (e.faces_failed ? "download failed" : "downloads on first use")}</span>
      (YuNet ${sys.face_models.yunet ? "✓" : "–"} / SFace ${sys.face_models.sface ? "✓" : "–"})<br>
      <b>Data folder:</b> ${sys.data_dir}<br>
      <b>Data size:</b> ${sys.data_size_mb} MB · <b>Demo video:</b> ${sys.demo_video ? "✓ bundled" : "missing"}`;
  } catch {}
}

async function saveSetting(patch) {
  try {
    await api("/api/settings", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(patch),
    });
    toast("Setting saved", "ok");
  } catch (e) { toast(e.message, "bad"); }
}

$("#setConf").addEventListener("input", () =>
  ($("#setConfV").textContent = ($("#setConf").value / 100).toFixed(2)));
$("#setConf").addEventListener("change", () => saveSetting({ confidence: $("#setConf").value / 100 }));
$("#setModel").addEventListener("change", () => saveSetting({ model: $("#setModel").value }));
$("#setObjects").addEventListener("change", () => saveSetting({ detect_objects: $("#setObjects").checked }));
$("#setClasses").addEventListener("change", () => saveSetting({ detect_classes: $("#setClasses").value }));
$("#setPlates").addEventListener("change", () => saveSetting({ detect_plates: $("#setPlates").checked }));
$("#setPlateMode").addEventListener("change", () => saveSetting({ plate_mode: $("#setPlateMode").value }));
$("#setColors").addEventListener("change", () => saveSetting({ recognize_colors: $("#setColors").checked }));
$("#setFaces").addEventListener("change", () => saveSetting({ recognize_faces: $("#setFaces").checked }));
$("#setFaceThr").addEventListener("input", () =>
  ($("#setFaceThrV").textContent = ($("#setFaceThr").value / 100).toFixed(2)));
$("#setFaceThr").addEventListener("change", () => saveSetting({ face_threshold: $("#setFaceThr").value / 100 }));
$("#setLoop").addEventListener("change", () => saveSetting({ loop_file: $("#setLoop").checked }));
$("#btnOpenData").addEventListener("click", () =>
  toast("Your data (models, events, faces) lives in the myeye/data folder next to the app"));

/* ------------------------------------------------------------------ init */
buildColorChips();
pollStats();
loadSettings();
