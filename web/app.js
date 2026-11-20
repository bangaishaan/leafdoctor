"use strict";
// LeafDoctor: on-device leaf classification with ONNX Runtime Web (wasm, no CDN).
// Preprocessing matches train.py's evaluation transform: EXIF-rotate, RGB,
// direct resize to 224x224, ImageNet mean/std, CHW float32.

const SIZE = 224;
const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];

const $ = (id) => document.getElementById(id);
const statusEl = $("status");
let session = null, classes = null, calib = null, stream = null;

function setStatus(msg) { statusEl.textContent = msg; }

async function init() {
  ort.env.wasm.wasmPaths = new URL("ort/", location.href).href;
  ort.env.wasm.numThreads = 1; // GitHub Pages is not cross-origin isolated, so no threads
  try {
    const [c, k] = await Promise.all([
      fetch("model/classes.json").then((r) => r.json()),
      fetch("model/calibration.json").then((r) => r.json()),
    ]);
    classes = c; calib = k;
    session = await ort.InferenceSession.create("model/leafdoctor_int8.onnx", { executionProviders: ["wasm"] });
    setStatus("Ready. Take or upload a photo of one leaf.");
  } catch (err) {
    console.error(err);
    setStatus("Could not load the model. Reload the page while online once, then it works offline.");
  }
}

function preprocess(source, w, h) {
  const canvas = document.createElement("canvas");
  canvas.width = SIZE; canvas.height = SIZE;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(source, 0, 0, w, h, 0, 0, SIZE, SIZE);
  const px = ctx.getImageData(0, 0, SIZE, SIZE).data;
  const n = SIZE * SIZE, data = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) {
    for (let c = 0; c < 3; c++) data[c * n + i] = (px[i * 4 + c] / 255 - MEAN[c]) / STD[c];
  }
  return new ort.Tensor("float32", data, [1, 3, SIZE, SIZE]);
}

function softmax(logits, T) {
  const z = Array.from(logits, (v) => v / T);
  const m = Math.max(...z);
  const e = z.map((v) => Math.exp(v - m));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / s);
}

async function classify(source, w, h, previewUrl) {
  if (!session) { setStatus("The model is still loading…"); return; }
  setStatus("Checking…");
  const out = await session.run({ input: preprocess(source, w, h) });
  const probs = softmax(out.logits.data, calib.temperature);
  const top = probs.map((p, i) => [p, i]).sort((a, b) => b[0] - a[0]).slice(0, 3);
  render(top, previewUrl);
  setStatus("Done. Try another photo any time.");
}

function render(top, previewUrl) {
  const unsure = top[0][0] < calib.threshold;
  $("preview").src = previewUrl;
  $("unsure").hidden = !unsure;
  $("result").classList.toggle("is-unsure", unsure);
  $("result-title").textContent = unsure ? "Result: unsure" : "Most likely";
  const list = $("top3");
  list.replaceChildren(...top.map(([p, i]) => {
    const li = document.createElement("li");
    const pct = (100 * p).toFixed(p >= 0.995 ? 1 : 0);
    li.innerHTML = `<div class="row"><span></span><span>${pct}%</span></div>
      <div class="bar" role="progressbar" aria-valuemin="0" aria-valuemax="100" aria-valuenow="${pct}"><i></i></div>`;
    li.querySelector(".row span").textContent = classes[i].label;
    li.querySelector(".bar").setAttribute("aria-label", `${classes[i].label} confidence`);
    li.querySelector(".bar > i").style.width = `${Math.max(1, 100 * p)}%`;
    return li;
  }));
  const best = classes[top[0][1]];
  $("look-for").textContent = unsure
    ? `If it were ${best.label}, you might see: ${best.look_for}`
    : `What to look for (${best.label}): ${best.look_for}`;
  $("result").hidden = false;
  $("result").scrollIntoView({ behavior: "smooth", block: "start" });
}

async function handleFile(file) {
  if (!file) return;
  try {
    const bitmap = await createImageBitmap(file, { imageOrientation: "from-image" });
    await classify(bitmap, bitmap.width, bitmap.height, URL.createObjectURL(file));
    bitmap.close();
  } catch (err) {
    console.error(err);
    setStatus("Could not read that image. Try a JPEG or PNG photo.");
  }
}

async function openCamera() {
  if (!navigator.mediaDevices?.getUserMedia) { $("capture-input").click(); return; }
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" }, audio: false });
    $("video").srcObject = stream;
    await $("video").play();
    $("camera").hidden = false;
  } catch (err) {
    console.warn(err);
    $("capture-input").click(); // fall back to the phone's camera app
  }
}

function closeCamera() {
  stream?.getTracks().forEach((t) => t.stop());
  stream = null;
  $("camera").hidden = true;
}

async function snap() {
  const v = $("video");
  if (!v.videoWidth) return;
  const c = document.createElement("canvas");
  c.width = v.videoWidth; c.height = v.videoHeight;
  c.getContext("2d").drawImage(v, 0, 0);
  closeCamera();
  const blob = await new Promise((r) => c.toBlob(r, "image/jpeg", 0.92));
  await classify(c, c.width, c.height, URL.createObjectURL(blob));
}

$("camera-btn").addEventListener("click", openCamera);
$("snap-btn").addEventListener("click", snap);
$("close-camera-btn").addEventListener("click", closeCamera);
for (const id of ["file-input", "capture-input"]) {
  $(id).addEventListener("change", async (e) => {
    await handleFile(e.target.files[0]);
    e.target.value = ""; // allow choosing the same photo again
  });
}

if ("serviceWorker" in navigator) {
  window.addEventListener("load", () => navigator.serviceWorker.register("sw.js").catch((e) => console.warn(e)));
}
init();
