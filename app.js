// LeafDoctor: all inference runs in the browser with ONNX Runtime Web (WebAssembly). Nothing is uploaded.
ort.env.wasm.wasmPaths = "vendor/";
ort.env.wasm.numThreads = 1;   // GitHub Pages has no cross-origin isolation, so a single thread is used

const $ = (id) => document.getElementById(id);
let session = null, cfg = null, last = null;

function setNet() { $("net").textContent = navigator.onLine ? "online" : "offline ✓"; }
addEventListener("online", setNet); addEventListener("offline", setNet); setNet();

if ("serviceWorker" in navigator) navigator.serviceWorker.register("sw.js").catch(() => {});

async function init() {
  try {
    cfg = await (await fetch("model/config.json")).json();
    session = await ort.InferenceSession.create("model/model.onnx", { executionProviders: ["wasm"], graphOptimizationLevel: "all" });
    $("status").textContent = "Ready. Works offline from now on.";
    $("ver").textContent = `model ${cfg.trained_model}/${cfg.model_variant} · ${cfg.version}`;
  } catch (e) {
    $("status").textContent = "Could not load the model: " + e.message;
    console.error(e);
  }
}

function softmax(z, T) {
  const m = Math.max(...z); const e = z.map((v) => Math.exp((v - m) / T)); const s = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / s);
}

// Same preprocessing as training: centre square, keep the central 87.5 %, resize to 224, RGB in [0,1]
function preprocess(bitmap) {
  const S = cfg.input_size, frac = cfg.crop_fraction;
  const side = Math.min(bitmap.width, bitmap.height) * frac;
  const sx = (bitmap.width - side) / 2, sy = (bitmap.height - side) / 2;
  const c = $("photo"), ctx = c.getContext("2d", { willReadFrequently: true });
  ctx.imageSmoothingEnabled = true; ctx.imageSmoothingQuality = "high";
  ctx.drawImage(bitmap, sx, sy, side, side, 0, 0, S, S);
  const px = ctx.getImageData(0, 0, S, S).data, n = S * S, data = new Float32Array(3 * n);
  let lum = 0; const gray = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const r = px[4 * i] / 255, g = px[4 * i + 1] / 255, b = px[4 * i + 2] / 255;
    data[i] = r; data[n + i] = g; data[2 * n + i] = b;
    gray[i] = 0.299 * r + 0.587 * g + 0.114 * b; lum += gray[i];
  }
  // Simple photo-quality checks: brightness and sharpness (variance of the Laplacian)
  let lap = 0, lapSq = 0, cnt = 0;
  for (let y = 1; y < S - 1; y++) for (let x = 1; x < S - 1; x++) {
    const i = y * S + x, v = 4 * gray[i] - gray[i - 1] - gray[i + 1] - gray[i - S] - gray[i + S];
    lap += v; lapSq += v * v; cnt++;
  }
  const sharp = (lapSq / cnt - (lap / cnt) ** 2) * 1e4;
  return { tensor: new ort.Tensor("float32", data, [1, 3, S, S]), brightness: lum / n, sharpness: sharp };
}

function drawHeat(cam, h, w) {
  const c = $("heat"), ctx = c.getContext("2d");
  let mx = 1e-9; for (const v of cam) mx = Math.max(mx, v);
  const small = document.createElement("canvas"); small.width = w; small.height = h;
  const sctx = small.getContext("2d"), img = sctx.createImageData(w, h);
  for (let i = 0; i < h * w; i++) {
    const t = cam[i] / mx;   // simple blue→yellow→red colour map
    img.data[4 * i] = Math.round(255 * Math.min(1, 2 * t));
    img.data[4 * i + 1] = Math.round(255 * Math.min(1, 2 * (1 - Math.abs(t - 0.5))));
    img.data[4 * i + 2] = Math.round(255 * Math.max(0, 1 - 2 * t));
    img.data[4 * i + 3] = Math.round(255 * Math.min(1, 0.2 + t));
  }
  sctx.putImageData(img, 0, 0);
  ctx.clearRect(0, 0, c.width, c.height); ctx.imageSmoothingEnabled = true;
  ctx.drawImage(small, 0, 0, c.width, c.height);
}

async function analyse(file) {
  if (!file) return;
  if (!session) { $("status").textContent = "The model is still loading, please wait a moment."; return; }
  $("status").textContent = "Checking the leaf…";
  const bitmap = await createImageBitmap(file);
  const pre = preprocess(bitmap);
  const t0 = performance.now();
  const out = await session.run({ image: pre.tensor });
  const ms = Math.round(performance.now() - t0);
  const probs = softmax(Array.from(out.logits.data), cfg.temperature);
  const order = probs.map((p, i) => [p, i]).sort((a, b) => b[0] - a[0]).slice(0, 3);
  const [ch, cw] = out.cam.dims.slice(1);
  drawHeat(out.cam.data, ch, cw);

  const issues = [];
  if (pre.brightness < 0.15) issues.push("The photo is very dark.");
  if (pre.brightness > 0.9) issues.push("The photo is over-exposed.");
  if (pre.sharpness < 2) issues.push("The photo looks blurry.");
  $("quality").textContent = issues.length ? "⚠️ " + issues.join(" ") + " Try again in daylight, holding the phone steady." : "";

  const top = order[0], cls = cfg.classes[top[1]];
  const unsure = top[0] < cfg.threshold || issues.length > 0;
  const pct = (p) => Math.round(p * 100) + "%";
  let html = "";
  if (unsure) {
    html += `<div class="unsure"><b>Not sure: ask an expert.</b><br>The model is not confident enough to name this condition${issues.length ? " and the photo quality is low" : ""}. ` +
            `It may be a crop or disease it does not know. Show the leaf to your local agricultural extension officer.</div>` +
            `<div class="muted">Possible matches, for the expert only:</div>`;
  } else {
    html += `<div class="muted">Most likely</div><div class="top1">${cls.name}</div>` +
            `<div class="bar"><div style="width:${pct(top[0])}"></div></div><div class="muted">Confidence ${pct(top[0])}</div>`;
  }
  order.forEach(([p, i], k) => {
    if (!unsure && k === 0) return;
    html += `<div class="alt"><span>${cfg.classes[i].name}</span><span>${pct(p)}</span></div>`;
  });
  if (!unsure) html += `<div class="look"><b>What to look for:</b> ${window.LEAF_INFO[cls.id] || ""}</div>`;
  html += `<div class="note">Indicative only. Confirm with your local agricultural extension office before any treatment. · ${ms} ms on this device</div>`;
  $("result").innerHTML = html;
  $("resultCard").classList.remove("hidden");
  $("fbNote").textContent = "";
  $("status").textContent = "Done. Take another photo any time.";
  last = { time: new Date().toISOString(), top1: cls.name, confidence: +top[0].toFixed(3), unsure, ms };
  $("resultCard").scrollIntoView({ behavior: "smooth" });
}

// Human in the loop: an expert's verdict is stored only on this phone and can be exported to share
function loadFb() { try { return JSON.parse(localStorage.getItem("leafdoctor_fb") || "[]"); } catch { return []; } }
document.querySelectorAll("#fb button").forEach((b) => b.addEventListener("click", () => {
  if (!last) return;
  const fb = loadFb(); fb.push({ ...last, expert: b.dataset.v });
  try { localStorage.setItem("leafdoctor_fb", JSON.stringify(fb)); $("fbNote").textContent = `Saved on this phone (${fb.length} records).`; }
  catch { $("fbNote").textContent = "Could not save on this browser."; }
}));
$("export").addEventListener("click", (e) => {
  e.preventDefault(); const fb = loadFb();
  if (!fb.length) { alertless("No feedback saved yet."); return; }
  const csv = ["time,top1,confidence,unsure,ms,expert"].concat(fb.map((r) => [r.time, `"${r.top1}"`, r.confidence, r.unsure, r.ms, r.expert].join(","))).join("\n");
  const a = document.createElement("a"); a.href = URL.createObjectURL(new Blob([csv], { type: "text/csv" })); a.download = "leafdoctor_feedback.csv"; a.click();
});
function alertless(msg) { $("fbNote").textContent = msg; }
$("heatToggle").addEventListener("change", (e) => { $("heat").style.opacity = e.target.checked ? 0.45 : 0; });
["cam", "gal"].forEach((id) => $(id).addEventListener("change", (e) => { analyse(e.target.files[0]); e.target.value = ""; }));
init();
