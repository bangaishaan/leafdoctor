# LeafDoctor results (auto-generated)

## Lab vs field (domain shift)

| Model | Lab acc | Lab macro-F1 | Field acc | Field macro-F1 | Field top-3 | Drop (lab-field) |
|---|---|---|---|---|---|---|
| B0 classical HSV histogram + LR | 0.832 | 0.832 | 0.076 | 0.067 | 0.216 | +0.756 |
| B1 frozen ImageNet MobileNetV3-S + LR | 0.954 | 0.953 | 0.254 | 0.219 | 0.475 | +0.699 |
| A fine-tuned, lab only | 0.953 | 0.952 | 0.318 | 0.293 | 0.589 | +0.635 |
| B fine-tuned, lab + diffusion | 0.966 | 0.966 | 0.280 | 0.241 | 0.589 | +0.686 |

Deployed: model B (fp32), chosen on field validation accuracy (A 0.213 vs B 0.266).

## Calibration and rejection (field test)

- Temperature T = 1.338; ECE 0.160 → 0.057
- Threshold 0.24: 43.6% of field photos marked unsure; accuracy on the rest 0.3985

## Export

- fp32 6.2 MB, field acc 0.280, 3.1 ms/image (1 CPU thread)
- int8 3.72 MB, field acc 0.059, 6.0 ms/image (1 CPU thread)

## Per crop (field test, deployed model)

| Crop | n | Acc | Unsure share |
|---|---|---|---|
| Apple | 29 | 0.138 | 65.5% |
| Bell pepper | 17 | 0.294 | 41.2% |
| Blueberry | 11 | 0.273 | 63.6% |
| Cherry | 10 | 0.100 | 60.0% |
| Corn | 26 | 0.385 | 7.7% |
| Grape | 20 | 0.800 | 30.0% |
| Peach | 9 | 0.222 | 44.4% |
| Potato | 16 | 0.062 | 25.0% |
| Raspberry | 7 | 0.857 | 0.0% |
| Soybean | 8 | 0.000 | 87.5% |
| Squash | 6 | 0.500 | 16.7% |
| Strawberry | 8 | 0.000 | 75.0% |
| Tomato | 69 | 0.217 | 49.3% |

## Top confusions (field test)

- Corn northern leaf blight → Corn gray leaf spot: 7
- Tomato Septoria leaf spot → Grape black rot: 5
- Potato late blight → Tomato late blight: 5
- Apple cedar rust → Grape black rot: 5
- Strawberry (healthy) → Raspberry (healthy): 4
- Soybean (healthy) → Blueberry (healthy): 4
- Bell pepper bacterial spot → Grape black rot: 4
- Tomato late blight → Potato late blight: 3

## Data

{
 "plantvillage_train": 3360,
 "plantvillage_val": 560,
 "plantvillage_test": 1120,
 "plantdoc_field_val(train folder)": 2342,
 "plantdoc_field_test(test folder)": 236,
 "classes": 28,
 "field_test_per_class": {
  "Apple scab": 10,
  "Apple cedar rust": 10,
  "Apple (healthy)": 9,
  "Bell pepper bacterial spot": 9,
  "Bell pepper (healthy)": 8,
  "Blueberry (healthy)": 11,
  "Cherry (healthy)": 10,
  "Corn gray leaf spot": 4,
  "Corn northern leaf blight": 12,
  "Corn common rust": 10,
  "Grape black rot": 8,
  "Grape (healthy)": 12,
  "Peach (healthy)": 9,
  "Potato early blight": 8,
  "Potato late blight": 8,
  "Raspberry (healthy)": 7,
  "Soybean (healthy)": 8,
  "Squash powdery mildew": 6,
  "Strawberry (healthy)": 8,
  "Tomato bacterial spot": 9,
  "Tomato early blight": 9,
  "Tomato late blight": 10,
  "Tomato leaf mold": 6,
  "Tomato Septoria leaf spot": 11,
  "Tomato spider mites": 0,
  "Tomato mosaic virus": 10,
  "Tomato yellow leaf curl virus": 6,
  "Tomato (healthy)": 8
 }
}

Generative: {"model": "stabilityai/sd-turbo (text-to-image backgrounds + image-to-image harmonisation, strength 0.35)", "n_backgrounds": 64, "n_synthetic": 560, "seconds": 420.1}