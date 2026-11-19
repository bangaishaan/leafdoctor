# AGENTS.md — LeafDoctor project rules

LeafDoctor is an offline plant-disease checker built for a university computer-vision project. A user photographs a leaf and the app predicts the disease entirely on their own device.

These rules apply to all code, scripts, and changes in this repository.

## Cost and services

- Do not use paid APIs, paid services, or paid hosting.
- Do not add any feature that needs a server-side backend or a network call to run inference.

## Inference runs in the browser

- Every model runs in the user's browser with ONNX Runtime Web.
- The app must work offline after the first load.

## Model size and format

- The shipped model is an int8-quantized ONNX file under 15 MB.
- Check the size after export and quantization. A model over 15 MB does not ship.

## Deployment

- Deploy on GitHub Pages as a Progressive Web App (PWA).
- The service worker must cache the app shell and the ONNX model so the app runs offline.

## Datasets

- Train only on PlantVillage.
- Keep every image of one physical leaf in the same PlantVillage split, across the color and segmented configs. `src/prepare_data.py` enforces this.
- PlantVillage has `train`, `val` (15% of train, split by leaf) and `test`. Tune hyperparameters on PlantVillage `val`.
- PlantDoc is the field evaluation set. Never use it for training, in any form. This covers both PlantDoc's own `train` and `test` splits.
- `src/prepare_data.py` pools PlantDoc and splits it 30/70 into `field_val` and `field_test`, keeping near-duplicates together. `field_val` may be used for model selection and calibration. `field_test` is only for reporting final results.

## Classes

- Use exactly the 27 classes in [data/class_map.csv](data/class_map.csv).
- Use the `class_id` column as the label, and the `plantvillage_folder` and `plantdoc_folder` columns to map each dataset's folders to it.
- Do not add, drop, merge, or rename classes without updating `class_map.csv` first. Read the class list and its order from that file instead of hard-coding it.

## Git

- Never commit images or datasets. This covers raw downloads, extracted folders, archives, and processed copies.
- Keep data outside the repo or in git-ignored folders.
- The final int8 ONNX model may be committed, because GitHub Pages has to serve it.

## Training scripts

- Training runs on Colab or Kaggle, not locally.
- Every script must run from the command line and take arguments (for example with `argparse`) for data paths, output paths, and hyperparameters.
- Do not hard-code local or Colab-specific paths, and do not rely on notebook-only state.
