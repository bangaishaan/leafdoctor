# LeafDoctor: offline plant-disease checker

A phone web app that recognises 28 leaf conditions on 14 crops, **offline, in the browser**. The photo never leaves the phone. It shows the top-3 conditions with calibrated confidence, a Grad-CAM heat-map, and a **"Not sure: ask an expert"** state.

MAIB AI 221 Computer Vision with Applications, Final Project #2 (Agriculture), SP Jain School of Global Management.

**Live app:** https://YOUR-GITHUB-USERNAME.github.io/leafdoctor/

## How it works
| Part | What |
|---|---|
| Training data | PlantVillage (lab photos, plain backgrounds), 28 classes that also appear in PlantDoc |
| Field test data | PlantDoc (real field photos, CC BY 4.0): `train/` folder used for calibration, `test/` folder for the final test only |
| Baselines | HSV colour histogram + logistic regression; frozen ImageNet MobileNetV3-Small + logistic regression |
| Model | MobileNetV3-Small (ImageNet), fine-tuned with augmentation, exported to ONNX (fp32, 6.2 MB). int8 post-training quantisation was tried but lost too much accuracy (28 % → 6 % on field photos), so fp32 is deployed |
| Generative AI | SD-Turbo makes field backgrounds (soil, hands, foliage, harsh light). Lab leaves are pasted on and blended with image-to-image (strength 0.35). Model B is trained on lab + synthetic images and compared with model A (lab only) on PlantDoc |
| Calibration | Temperature scaling on field validation; the rejection threshold is chosen for ≥70 % accuracy on answered photos |
| App | Static page + ONNX Runtime Web (WebAssembly), service worker for offline use, GitHub Pages |

## Key results (PlantDoc field test, 236 photos)
| Model | Lab acc | Field acc | Field top-3 |
|---|---|---|---|
| Classical HSV histogram + LR | 83.2 % | 7.6 % | 21.6 % |
| Frozen ImageNet features + LR | 95.4 % | 25.4 % | 47.5 % |
| A: fine-tuned, lab only | 95.3 % | 31.8 % | 58.9 % |
| B: fine-tuned, lab + diffusion (deployed) | 96.6 % | 28.0 % | 58.9 % |

On the larger field validation set (2,342 photos), B beat A (26.6 % vs 21.3 %). After temperature scaling (T = 1.34, field ECE 0.160 → 0.057), the app marks 43.6 % of field photos "unsure", and accuracy on the rest is 39.9 %. Full details: `results/report_numbers.md`.

## Repository layout
```
index.html, help.html, app.js, info.js   the web app
sw.js, manifest.webmanifest, icons/      offline / installable (PWA)
vendor/                                  ONNX Runtime Web 1.20.1 (MIT), served locally so the app works offline
model/model.onnx, model/config.json      trained model, class list, temperature and threshold
training/LeafDoctor_Colab.ipynb          full training + evaluation pipeline (free Colab T4)
results/                                 results.json, report_numbers.md, figures
MODEL_CARD.md
```

## Set up and run
**Train (about 15 min on a free Colab T4):** open `training/LeafDoctor_Colab.ipynb` in Google Colab, choose Runtime → Change runtime type → T4 GPU, then Runtime → Run all. It downloads both datasets from GitHub and writes `leafdoctor_out.zip`. Copy `model/model.onnx` and `model/config.json` from the zip into `model/`.

**Run locally:** `python -m http.server 8000` in the repository folder, then open http://localhost:8000. Opening `index.html` directly from disk will not work, because the model has to be fetched over HTTP.

**Deploy:** push to a public GitHub repository, then Settings → Pages → Source: *Deploy from a branch* → `main` / `(root)` → Save. The site appears at `https://<user>.github.io/<repo>/` after about a minute.

## Licences and credits
- PlantVillage dataset: Hughes & Salathé (2015), github.com/spMohanty/PlantVillage-Dataset (CC BY-SA 3.0 as stated by the authors)
- PlantDoc dataset: Singh et al. (2020), github.com/pratikkayal/PlantDoc-Dataset, CC BY 4.0
- MobileNetV3 weights: torchvision (BSD-3-Clause)
- SD-Turbo: Stability AI, used under the Stability AI Non-Commercial Research Community licence, for synthetic training images only (not shipped in the app)
- ONNX Runtime Web: Microsoft, MIT licence
- Code in this repository: MIT

**Disclaimer:** results are indicative only. Always confirm with a local agricultural extension service before any treatment.
