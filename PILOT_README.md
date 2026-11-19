# Sobel edge-loss pilot

`main.py` and `main_resnet.py` are preserved. New runnable versions reuse
the VGG content/style loss definitions from `main.py`:

- `main_baseline.py`: content + Gram style objective (edge weight 0).
- `main_edge.py`: same objective plus the loss in `edge_loss.py`.
- `pilot.py`: shared runner; can run baseline and all weights together.

## Install and run (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# Baseline only
.\.venv\Scripts\python.exe main_baseline.py

# Sobel weights 1000, 10000, 100000
.\.venv\Scripts\python.exe main_edge.py

# Recommended: one comparison including the baseline
.\.venv\Scripts\python.exe pilot.py --edge-weights 0 1000 10000 100000

# Quick pilot, or specify several photos explicitly
.\.venv\Scripts\python.exe pilot.py --image-size 256 --steps 100
.\.venv\Scripts\python.exe pilot.py --content imgs/pilot1.jpg --edge-weights 0 100 1000
```

Defaults: all photos in `imgs/`, `style.jpg`, longest side 512, 300 optimizer
steps, style weight 1e6, content weight 1. Paths default relative to the code
directory. First run downloads pretrained VGG-19 weights (requires internet).
CUDA is selected automatically if available; use `--device cpu` to force CPU.
CPU threads default to 4 (`--threads`); weights are cached in `.torch_cache/`
unless `TORCH_HOME` is set.
CPU runs can be slow. For Colab, install requirements and use
`!python pilot.py --edge-weights 0 1000 10000 100000` from the project directory.

## Loss definition

`total = content_weight * content + style_weight * style + edge_weight * edge`

RGB images in [0, 1] are converted to grayscale with coefficients
(0.299, 0.587, 0.114). A horizontal Sobel kernel divided by 8 and its transpose
produce signed x/y gradients. Edge loss is the mean squared difference of
these two gradient maps against the fixed input photograph. Replicate padding
avoids artificial borders. The loss is applied before VGG normalization.
It matches edges everywhere, including background edges; it does not identify
text. Edge weights depend on this normalization and need pilot tuning.

## Fair comparison and outputs

Both new entry points preserve image aspect ratio and EXIF orientation, avoid
center cropping, initialize from the same photo, freeze VGG parameters, and
use exactly the same optimization budget. Each step is one L-BFGS update
(`max_iter=1`). This differs from the old script's closure-count stopping
rule, so use the new baseline for comparisons with the new Sobel version.
The original script remains available separately.

Each photo gets a directory such as `pilot_output/01_pilot1/` containing:

- `original.png`, `style.png`: processed optimization inputs.
- `baseline.png`, `edge_10.png`, etc.: generated images.
- `comparison.png`: input and outputs side by side.
- `*_loss.csv`: raw losses and weighted total, measured before each update.
- `metrics.csv`: raw final losses recomputed after the final update and clamp,
  plus runtime. `style` is Gram distance; these are not OCR readability scores.
- `config.json`: input paths, weights, device and resolution.

Repeating a configuration overwrites matching files. Use `--output-dir` to
keep experiments separate, e.g. `--output-dir pilot_output/pilot_256`.

Start with weights 0, 1000, 10000, 100000. Compare thin letter strokes, enclosed
spaces in letters, background texture and painting style in `comparison.png`.
Refine the range around a useful tradeoff, keeping size, steps and other
weights fixed. Smaller edge loss alone does not prove better readability;
inspect the images before selecting a weight. Add more photos to `imgs/`
to evaluate the same weights across them. No best weight is assumed in advance.

## Pilot already run on the supplied photo

`imgs/pilot1.jpg` with `style.jpg`, longest side 256 and 100 steps on CPU:

- `pilot_output/01_pilot1/`: weights 0, 10, 100, 1000.
- `pilot_output/strong_edges/01_pilot1/`: weights 0, 1000, 10000, 100000.

| Edge weight | Final edge MSE | Final Gram style distance |
|---|---|---|
| 0 | 0.002117 | 0.00002024 |
| 1000 | 0.002022 | 0.00002092 |
| 10000 | 0.001510 | 0.00002205 |
| 100000 | 0.0005963 | 0.00003904 |

The strongest weight reduces edge MSE by about 72% but increases Gram
distance. The comparison image shows stronger letter-edge preservation;
this is a visual pilot observation, not a measured OCR improvement.
Only one photo was supplied, so the new default sweep is a starting range,
not a validated best weight. More photos and higher-resolution runs are
needed before choosing the final weight.

## Verification

```powershell
.\.venv\Scripts\python.exe -m unittest test_edge_loss -v
```

Checks identical/constant images, nonzero edge gradients, detached targets,
CPU optimization with both objectives using a small untrained network, and
agreement of baseline losses with the original model builder. The new builder
computes style targets without forwarding through content-loss modules, so
content and style images can have different aspect ratios.
