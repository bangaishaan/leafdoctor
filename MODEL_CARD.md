# Model card: LeafDoctor leaf-condition classifier (fp32, model B, lab + diffusion)

**Model.** MobileNetV3-Small (ImageNet weights) fine-tuned for 28 leaf conditions on 14 crops; ONNX, 6.2 MB, temperature T = 1.34, "unsure" threshold τ = 0.24. Outputs: logits and a 7 × 7 Grad-CAM map. Runs in the browser (ONNX Runtime Web). Version 20261123-1806.

**Intended use.** Triage support for farmers and extension officers: photograph one leaf of a covered crop and get the top-3 likely conditions, with a calibrated confidence, to decide whether to call an expert.

**Out-of-scope use.** Diagnosis without an expert; choosing treatments; crops, diseases or pests not listed; whole-plant, fruit or non-plant photos; insurance or regulatory decisions.

**Training data.** PlantVillage lab photos (3,360 images, 28 classes), plus 560 synthetic field-style images made with SD-Turbo (model B only). **Calibration data:** PlantDoc train folder (2,342 field photos). **Test data:** PlantDoc test folder (236 field photos, CC BY 4.0), and a PlantVillage held-out test (1,120).

**Metrics.**

| | Lab test | Field test |
|---|---|---|
| Accuracy | 96.6 % | 28.0 % |
| Macro-F1 | 0.966 | 0.241 |
| Top-3 accuracy | 99.6 % | 58.9 % |

Field: ECE 0.160 → 0.057 after temperature scaling; 43.6 % of photos marked "unsure"; accuracy on answered photos 39.9 %. Speed: 3.1 ms per image (1 CPU thread), **[FILL IN: phone ms]** ms on a phone.

**Subgroup results (field accuracy by crop, n = 6–69).** Best: raspberry 86 %, grape 80 %, squash 50 %. Worst: potato 6 %, cherry 10 %, apple 14 %, soybean and strawberry 0 %; tomato 22 %. **Potato is the highest-risk subgroup: 6 % accurate but only 25 % flagged unsure.**

**Known failures.** Grape black rot is a catch-all for other spotted leaves; potato and tomato late blight are confused (same pathogen); healthy leaves of different crops are mixed up; several leaves in one photo, clutter, tiny lesions, dark or blurry photos. Six crops (blueberry, cherry, peach, raspberry, soybean, strawberry) have only a "healthy" class, so the model *cannot* detect disease on them and may falsely reassure. Mainly US training photos; Gulf, South Asian and African crops are absent.

**Ethics and safety.** Every result is indicative and refers to the local extension office; no treatment advice; on-device inference, no data collected; expert feedback is stored locally and exported only by the user.
