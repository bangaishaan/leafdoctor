# leafdoctor
AI-powered offline plant disease detection using computer vision

## Data and attribution

LeafDoctor is trained on PlantVillage and evaluated on PlantDoc. No dataset images are stored
in this repository; scripts in `src/` download and prepare them (see `src/prepare_data.py`).

- **PlantVillage**: Mohanty, S. P., Hughes, D. P., & Salathé, M. (2016). *Using Deep Learning for
  Image-Based Plant Disease Detection.* Frontiers in Plant Science, 7, 1419.
  [doi:10.3389/fpls.2016.01419](https://doi.org/10.3389/fpls.2016.01419).
  Images from [spMohanty/PlantVillage-Dataset](https://github.com/spMohanty/PlantVillage-Dataset),
  via [mohanty/PlantVillage](https://huggingface.co/datasets/mohanty/PlantVillage) on Hugging Face,
  licensed [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/).
- **PlantDoc**: Singh, D., Jain, N., Jain, P., Kayal, P., Kumawat, S., & Batra, N. (2020).
  *PlantDoc: A Dataset for Visual Plant Disease Detection.* CODS-COMAD 2020.
  [doi:10.1145/3371158.3371196](https://doi.org/10.1145/3371158.3371196).
  Images from [pratikkayal/PlantDoc-Dataset](https://github.com/pratikkayal/PlantDoc-Dataset),
  licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

Synthetic training images for arm C (`src/gen/diffusion_inpaint.py`) are made with
[stable-diffusion-v1-5/stable-diffusion-inpainting](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-inpainting)
(revision `8a4288a`), licensed [CreativeML OpenRAIL-M](https://huggingface.co/spaces/CompVis/stable-diffusion-license).
The leaf in each image is the original PlantVillage leaf; only the background is generated.

Figures in `results/` that show dataset images contain material from these datasets:

- `results/figures/cutpaste_preview.png` shows PlantVillage leaves cut out, rotated, scaled and placed
  on generated backgrounds. As an adaptation of PlantVillage, it is shared under CC BY-SA 3.0.
- `results/runs/*/analysis/field_test_grid.png` shows resized PlantDoc photos (CC BY 4.0).
