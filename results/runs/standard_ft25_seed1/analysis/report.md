# Run analysis: standard_ft25_seed1

Arm `standard`, seed 1, best epoch 21 of 25 (chosen by pv_val macro-F1), trained on Tesla T4 in 47 min.

## Class order and row alignment

| Check | Result |
|---|---|
| metrics.json classes == class_map.csv order | pass |
| best.pt class_ids == class_map.csv order | pass |
| pv_val: logits have 27 columns | pass |
| pv_val: saved labels == labels rebuilt from split_manifest (row order) | pass |
| pv_val: accuracy/macro-F1 recomputed from logits == metrics.json | pass |
| pv_test: logits have 27 columns | pass |
| pv_test: saved labels == labels rebuilt from split_manifest (row order) | pass |
| pv_test: accuracy/macro-F1 recomputed from logits == metrics.json | pass |
| field_val: logits have 27 columns | pass |
| field_val: saved labels == labels rebuilt from split_manifest (row order) | pass |
| field_val: accuracy/macro-F1 recomputed from logits == metrics.json | pass |
| field_test: logits have 27 columns | pass |
| field_test: saved labels == labels rebuilt from split_manifest (row order) | pass |
| field_test: accuracy/macro-F1 recomputed from logits == metrics.json | pass |
| pv_test: every class's recall > 0.9 (logit column i is class_map class i) | pass |

## Metrics

| Split | n | Top-1 acc | Top-3 acc | Macro-F1 | Crop-level acc |
|---|---|---|---|---|---|
| pv_test | 5403 | 0.995 | 1.000 | 0.995 | 0.998 |
| field_test | 1802 | 0.195 | 0.373 | 0.153 | 0.428 |

Crop-level accuracy counts a prediction as right when its plant species matches (e.g. any tomato class for a tomato leaf).

## Comparison with baselines

Lab = pv_test (PlantVillage), field = field_test (PlantDoc).

| Model | Lab acc | Field acc | Lab→field gap | Lab macro-F1 | Field macro-F1 | Field top-3 |
|---|---|---|---|---|---|---|
| Classical: HSV + LBP, RBF SVM | 0.927 | 0.036 | -0.892 | 0.926 | 0.011 | – |
| Frozen MobileNetV3 + logistic reg. | 0.981 | 0.198 | -0.783 | 0.980 | 0.173 | – |
| Fine-tuned MobileNetV3 (standard_ft25_seed1) | 0.995 | 0.195 | -0.800 | 0.995 | 0.153 | 0.373 |

## 5 most common confusions: pv_test

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| corn_gray_leaf_spot | corn_northern_blight | 5 | 4% |
| corn_northern_blight | corn_gray_leaf_spot | 5 | 2% |
| soybean_healthy | cherry_healthy | 3 | 1% |
| tomato_ylcv | tomato_bacterial_spot | 3 | 1% |
| tomato_late_blight | tomato_bacterial_spot | 2 | 1% |

## 5 most common confusions: field_test

| True | Predicted | Count | Share of true class |
|---|---|---|---|
| corn_northern_blight | corn_gray_leaf_spot | 67 | 50% |
| corn_common_rust | corn_gray_leaf_spot | 50 | 62% |
| raspberry_healthy | tomato_late_blight | 37 | 45% |
| potato_late_blight | tomato_late_blight | 33 | 45% |
| tomato_septoria | tomato_early_blight | 29 | 27% |

## Crop-level accuracy by species

| Crop | pv_test n | pv_test acc | field_test n | field_test acc |
|---|---|---|---|---|
| Apple | 440 | 0.995 | 191 | 0.147 |
| Bell pepper | 449 | 0.998 | 92 | 0.065 |
| Blueberry | 324 | 0.997 | 81 | 0.086 |
| Cherry | 123 | 1.000 | 40 | 0.050 |
| Corn | 549 | 1.000 | 263 | 0.871 |
| Grape | 268 | 1.000 | 93 | 0.581 |
| Peach | 71 | 1.000 | 79 | 0.000 |
| Potato | 376 | 0.997 | 155 | 0.303 |
| Raspberry | 162 | 0.994 | 83 | 0.120 |
| Soybean | 287 | 0.986 | 47 | 0.000 |
| Squash | 314 | 1.000 | 91 | 0.626 |
| Strawberry | 80 | 1.000 | 67 | 0.000 |
| Tomato | 1960 | 1.000 | 520 | 0.637 |

## Figures

- `confusion_field_test.png`: row-normalised confusion matrix on field_test.
- `field_test_grid.png`: 24 random field_test images after eval preprocessing; 4/24 correct (green = correct, red = wrong).
