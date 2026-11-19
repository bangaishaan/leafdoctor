# CLAUDE.md — LeafDoctor

LeafDoctor is an offline plant-disease checker for a university computer-vision project.

## Project rules

- Do not use paid APIs or paid services.
- All models run in the user's browser (ONNX Runtime Web). Nothing runs on a server.
- The final model is an int8 ONNX file under 15 MB.
- Deploy on GitHub Pages.
- PlantDoc is never used for training.

The full rules, including datasets, classes, git, and training scripts, are in AGENTS.md:

@AGENTS.md
