# Camera Orientation Estimation using Vision Transformers

Estimating the orientation of a camera (the angles **pitch**, **yaw** and **roll**) from a single
**real (query) photograph** by matching it against a synthetic **360° equirectangular panorama** rendered
from a Digital Elevation Model (DEM). The work focuses on challenging mountain environments,
where appearance changes drastically with season and weather, and uses a transformer-based,
cross-modal architecture.

## Overview

Given two inputs:

1. a **synthetic 360° panorama** (DEM, equirectangular, `4096 × 2048 px`), and
2. a **real query image** (`512 × 512 px`) that captures some part of that panorama,

the model predicts where the query image is located within the panorama, expressed as three
Euler angles:

| Angle | Range            | Meaning                                                              |
|-------|------------------|----------------------------------------------------------------------|
| Pitch | `[-90, +90]°`    | Up/down. `0` = horizon, `+` looks up, `−` looks down.                 |
| Yaw   | `[-180, +180]°`  | Left/right. `0` = center, `+` to the right, `−` to the left.         |
| Roll  | `[-180, +180]°`  | In-plane rotation. `0` = aligned with horizon, `+` CW, `−` CCW.       |

Unlike the previous state-of-the-art methods (Baboud et al. 2011 and Brejcha et al. 2018),
this approach **does not require the field-of-view (FOV) as an input**, which is normally
unavailable for arbitrary internet photos. It reaches comparable accuracy to SOTA on the
GeoPose3K test split (and works even when the horizon is occluded or features are unclear),
while being weaker on the lower-resolution Venturi dataset.

## Method

The architecture (`src/model/model.py`) works as follows:

1. **Tiling.** The aspect-ratio-preserving `4096 × 2048 px` panorama is split into
   `4 × 8 = 32` perspective tiles of `512 × 512 px` (FOV `45° × 45°`) using
   [`py360convert`](https://github.com/sunset1995/py360convert).
2. **Shared Vision Encoder.** Both the 32 panorama tiles and the single query image pass
   through a shared, **fine-tuned** `PE-Spatial-Tiny` Vision Encoder
   (`timm/vit_pe_spatial_tiny_patch16_512.fb`, loaded via Hugging Face `AutoModel`).
3. **Positional information.** A learnable per-tile positional embedding plus a 1D
   sinusoidal positional embedding are added to the panorama and query patch tokens.
4. **Cross-Attention.** A shared multi-head cross-attention layer (8 heads) attends
   panorama → query and query → panorama, with residual connections and Layer Normalization.
5. **Attention Pooling.** Each branch is pooled into a single vector via a learned
   attention-pooling layer, the two vectors are concatenated and normalized.
6. **Classification heads.** Three separate MLP heads predict per-degree class logits:
   180 bins for pitch, 360 bins for yaw, 360 bins for roll. Treating the task as
   classification outperformed regression.
7. **Decoding (postprocessing).** Final angles are obtained as a circular weighted average
   over the bin centers (`circ_expect_deg`), which correctly handles wrap-around
   (e.g. `−179°` and `+179°` are 2° apart, not 358°).

**Loss:** Circular Huber Loss with `δ = 3.0`, made period-aware so that angular wrap-around
is handled correctly.

## Repository structure

```
.
├── README.md
├── requirements.txt
├── .gitignore
└── src/
    ├── model/
    │   └── model.py                  # OrientationEstimator architecture
    ├── training/
    │   ├── train.py                  # Training loop (deterministic)
    │   └── dataset.py                # OrientationDataset (panorama/query/gt triplets)
    ├── eval/
    │   ├── eval_geopose.py           # Evaluation on the GeoPose3K test split
    │   ├── eval_venturi.py           # Evaluation on the Venturi dataset (per-sequence)
    │   ├── eval_twostage_fixedpano.py# Two-stage (fixed-pano crop) refinement experiment
    │   └── performance.py            # Inference time / GPU memory profiling
    ├── preprocessing/
    │   ├── preprocess_geopose.py     # GeoPose3K -> panorama/query/gt triplets
    │   ├── preprocess_venturi.py     # Venturi -> panorama/query/gt triplets
    │   ├── preprocess_landscapear.py # LandscapeAR -> panorama/query/gt triplets
    │   └── augment.py                # Offline augmentation (6 augmentations + flip)
    ├── interpretation/
    │   ├── get_attn_maps.py          # Run a single example and visualize attention
    │   ├── model.py                  # Model variant that also returns attention maps
    │   └── vis_utils.py              # Cross-attention overlay visualization
    ├── visualization/
    │   ├── visualize_gt.py           # Overlay the query onto the panorama using GT
    │   └── py360convert_lib/         # Perspective <-> equirectangular helpers
    ├── checkpoints/
    │   └── best_model.pth            # Trained weights (best epoch on the test data)
    └── example/
        ├── panorama.jpg              # Example panorama
        ├── query.jpg                 # Example query image
        └── gt.csv                    # Example ground truth: pitch,yaw,roll,fov
```

## Installation

Requires Python 3.11 and a CUDA-capable GPU (CPU works but is impractically slow).

```bash
git clone https://github.com/martinkubicka/Camera-Orientation-Estimation-using-Vision-Transformers.git
cd Camera-Orientation-Estimation-using-Vision-Transformers

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```



The Vision Encoder is downloaded automatically from the Hugging Face Hub on first run, so
an internet connection is required the first time.

> **Note on paths.** All scripts use relative paths, so run each script from inside its own directory (e.g. `cd src/training` before running `train.py`) 

## Data format

The dataset is a flat directory of triplets, one per sample, sharing a common base name:

```
<base>_panorama.jpg   # equirectangular DEM panorama, 4096 x 2048 px
<base>_query.jpg      # real query image, 512 x 512 px
<base>_gt.csv         # single line: pitch,yaw,roll,fov   (degrees)
```

Resizing preserves the aspect ratio and pads with black borders. Inputs are normalized to
`[0, 1]` and standardized with `mean = std = (0.5, 0.5, 0.5)` (to match the Vision Encoder
the model is built on). The scripts in `src/preprocessing/` convert the raw GeoPose3K,
LandscapeAR and Venturi datasets into this format; `augment.py` produces 6 augmented
triplets per original (color/brightness/contrast/CLAHE/gamma jitter plus a paired
horizontal flip of both panorama and query).

The full training/test datasets are not publicly released due to author restrictions; see
the thesis for the sources (GeoPose3K + filtered LandscapeAR for train/validation,
GeoPose3K test split and Venturi for testing). The example in `src/example/` lets you run
inference and interpretation out of the box.

## Training

`train.py` runs a fully deterministic training loop. Because it enables
`torch.use_deterministic_algorithms(True)`, **CUDA needs a fixed cuBLAS workspace**,
otherwise PyTorch raises a runtime error for some deterministic GEMM operations. Set the
`CUBLAS_WORKSPACE_CONFIG` environment variable when launching training:

```bash
cd src/training
CUBLAS_WORKSPACE_CONFIG=:4096:8 python train.py
```

What the script expects and produces:

- **Input dataset:** `src/dataset/` (i.e. `../dataset/` relative to `src/training/`),
  containing the `*_panorama.jpg` / `*_query.jpg` / `*_gt.csv` triplets described above.
- **Split:** 90% train / 10% validation, split deterministically (`SEED = 42`).
- **Checkpoints:** written every epoch to `src/checkpoints/model_<epoch>.pth`
  (each contains `model_state` and `optim_state`).
- **Logs:** appended to `training_logs.txt` (loss and orientation estimation error per epoch).

Default training configuration:

| Setting              | Value                                                  |
|----------------------|--------------------------------------------------------|
| Optimizer            | AdamW (`β1=0.9`, `β2=0.999`, `eps=1e-8`, `wd=1e-2`)     |
| Learning rate        | `1e-5` for the Vision Encoder, `1e-4` for the rest     |
| LR schedule          | Constant (no scheduler)                                |
| Batch size           | 3                                                      |
| Precision            | Mixed precision (`autocast` + `GradScaler`)            |
| Loss                 | Circular Huber Loss (`δ = 3.0`, period-aware)          |
| Epochs               | 100 (loop); the thesis model used ~38, best at epoch 36|
| Seed                 | 42 (deterministic, cuDNN deterministic, no benchmark)  |

To **resume** from a checkpoint, uncomment the `RESUME TRAINING` block near the end of
`main()` in `train.py` (it loads `../checkpoints/best_model.pth` into both the model and
the optimizer).

**Hardware/time:** One epoch (train + validation) takes ~15 h on
8× NVIDIA A100 40 GB, or ~90 h on a single GPU. Batch size 3 needs ~40 GB of GPU memory
during training. Training was performed on the Karolina cluster.

## Evaluation

A trained checkpoint is provided at `src/checkpoints/best_model.pth`. The evaluation
scripts load it, run inference over a test directory of triplets, and report the
orientation estimation error
`e(R_gt, R_pred) = arccos((tr(R_gtᵀ R_pred) − 1) / 2)`, per-angle errors, median, and the
AUC of the error CDF (also saved as `auc.png`).

```bash
cd src/eval

# GeoPose3K test split (expects ../geopose_test/)
python eval_geopose.py

# Venturi dataset, per-sequence stats (expects ../venturi/)
python eval_venturi.py

# Two-stage refinement experiment: fixed brute-force crop of the 8K panorama
# (expects ../geopose_test/ and ../geopose_test_8k/)
python eval_twostage_fixedpano.py
```

Inference time / GPU memory profiling on the example pair:

```bash
cd src/eval
python performance.py
```

Inference needs ~4 GB of GPU memory and ~1.93 s per pair
(without pre/postprocessing). Panorama tiling via `py360convert` is the main bottleneck
(~98% of inference time) and is a clear target for future optimization.

## Model interpretation

`src/interpretation/` provides attention-map visualizations after the cross-attention
layer, showing which parts of the panorama matter for the query and vice versa (the model
mostly attends to edges and peaks). Run it on the bundled example:

```bash
cd src/interpretation
python get_attn_maps.py
```

This loads `../checkpoints/best_model.pth`, runs the `../example/` pair, displays the
"query → panorama" and "panorama → query" attention overlays, and prints the orientation
estimation error against `../example/gt.csv`.

## Visualization

`src/visualization/visualize_gt.py` overlays the query image onto the panorama using the
ground-truth angles and FOV, which is useful for sanity-checking annotations and
predictions:

```bash
cd src/visualization
python visualize_gt.py
```

## Results

Based on orientation estimation error (lower is better).

**GeoPose3K test split** (FOV-free; SOTA methods use FOV):

| Method                | Mean ↓   | Median ↓ | AUC ↑ |
|-----------------------|----------|----------|-------|
| **Ours**              | 19.38°   | 9.27°    | 0.89  |
| CF-VCC-2011 (SOTA)    | 39.25°   | 3.51°    | 0.78  |
| MatchAnything         | 32.57°   | 14.27°   | —     |
| RoMA                  | 51.47°   | 43.65°   | —     |

Per-angle on GeoPose3K: mean pitch `1.99°`, mean yaw `18.89°`, mean roll `0.71°`
(yaw is the hardest angle). On the **Venturi** dataset the method reaches a mean of
`~19.71°` (AUC `0.885`) but is outperformed by `CF-VCC-2011-m3D`.

## Limitations and future work

- Comparable to SOTA on GeoPose3K but weaker on Venturi.
- Inference is ~2 s, not real time; panorama tiling is the dominant cost.
- Training data is limited and likely biased toward the Alps region.
- FOV is still needed only for the final visual alignment, not for the prediction itself.

Promising directions include alternative architectures, multi-stage refinement, extending
to full geolocalization, and adding an FOV-estimation branch.

## Citation

```bibtex
@mastersthesis{kubicka2025orientation,
  title  = {Camera Orientation Estimation using Vision Transformers},
  author = {Kubi\v{c}ka, Martin},
  school = {Brno University of Technology, Faculty of Information Technology},
  year   = {2025},
  type   = {Master's thesis},
  note   = {Supervisor: prof. Ing. Martin \v{C}ad\'{i}k, Ph.D.}
}
```

## Acknowledgements
I would like to thank my supervisor prof. Ing. Martin Čadík Ph.D., my family and friends for guiding me through the process of creating this thesis, giving me advice along the way and passing on their knowledge.
