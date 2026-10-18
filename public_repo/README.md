# Monocular Badminton Analytics System

> **Paper:** *[Title — to be added on acceptance]*  
> **Authors:** *[Author names — withheld for double-blind review]*  
> **Venue:** *[Conference / Journal — to be added]*

---

## Overview

An end-to-end deep-learning and physics-informed system that extracts
professional-grade 3-D analytics from **single-camera** badminton footage —
no multi-camera rigs or depth sensors required.

The system provides:
- 🏸 **3D shuttle trajectory reconstruction** from monocular 2-D detections
- 🧍 **Player movement & positioning** via RTMPose skeletal tracking
- 🎯 **Automated stroke classification** across 17 stroke types

---

## Pipeline Architecture

```
                        ┌──────────────────────────────────────────────────┐
                        │              INPUT: Rally Video                   │
                        └─────────────────────┬────────────────────────────┘
                                              │
                 ┌────────────────────────────▼──────────────────────────┐
                 │          Stage 1: Court Detection & Calibration        │
                 │  • Roboflow: court corner + net-line keypoint detection │
                 │  • L-BFGS-B optimisation of camera intrinsics K        │
                 │  • PnP → R, t, projection matrix P                     │
                 └───────────────┬──────────────────────────┬────────────┘
                                 │                          │
          ┌──────────────────────▼───┐          ┌──────────▼──────────────────┐
          │   Stage 2a: Shuttle       │          │   Stage 2b: Pose Estimation  │
          │   Tracking (parallel)     │          │   (parallel)                  │
          │  • TrackNetV3 (HF Space)  │          │  • RTMPose (HF Space)         │
          │  • Per-frame (x, y, vis)  │          │  • COCO-17 keypoints          │
          │  • Handles blur / occ.    │          │  • Near / Far player split    │
          └──────────┬───────────────┘          └──────────┬───────────────────┘
                     │                                      │
          ┌──────────▼──────────────────────────────────────▼───────────────┐
          │           Stage 3: Physics-Informed 3-D Reconstruction            │
          │  • ODE: ẋ = v,  v̇ = g − Cₐ‖v‖v  (gravity + quadratic drag)    │
          │  • Joint optimisation of x₀, v₀, Cₐ via L-BFGS-B                │
          │  • Reprojection loss + 3-D anchor constraints (Eq. 4)            │
          │  ⚠  Full implementation withheld — see paper for algorithm       │
          └─────────────────────────────────────┬────────────────────────────┘
                                                │
          ┌─────────────────────────────────────▼────────────────────────────┐
          │           Stage 4: Stroke Classification (BST-CG-AP)              │
          │  • 5-D feature fusion: 2-D joints + bones + 3-D trajectory        │
          │  • Transformer architecture with cross-attention gating            │
          │  • 17-class output over ShuttleSet17 taxonomy                     │
          │  ⚠  Model weights withheld pending acceptance                    │
          └──────────────────────────────────────────────────────────────────┘
```

> **Note:** Stages 3 and 4 implement the novel contributions of the paper.
> Their core implementation is withheld in this public release to comply with
> double-blind review policy.  All code will be open-sourced on acceptance.

---

## Sample Results

### 🎯 3D Shuttle Trajectory + XYZ Coordinates

https://github.com/user-attachments/assets/trajectory_3d.mp4

> Physics-informed 3D reconstruction of the shuttlecock flight path with
> real-time XYZ world-coordinate overlay.  Each frame shows the projected
> trajectory arc and the solved (X, Y, Z) position in metres.

---

### 🏸 Shot Classification — Rally 1

https://github.com/user-attachments/assets/shot_classification_1.mp4

> BST-CG-AP stroke classification output for a full rally.
> Each shot interval is labelled with the predicted stroke type
> and hitter side (Top / Bottom player).

---

### 🏸 Shot Classification — Rally 2

https://github.com/user-attachments/assets/shot_classification_2.mp4

> A second rally demonstrating classification across multiple stroke types
> including Smash, Lob, Drop, and Net Shot variants.

---

## Repository Structure

```
public_repo/
├── run_pipeline.py          ← End-to-end CLI runner (no GUI)
├── requirements.txt
├── .env.example             ← Copy to .env and fill in your Roboflow key
├── samples/
│   ├── trajectory_3d.mp4         ← 3D shuttle trajectory demo
│   ├── shot_classification_1.mp4 ← Stroke classification demo (rally 1)
│   └── shot_classification_2.mp4 ← Stroke classification demo (rally 2)
├── tracknetv3/              ← TrackNetV3 inference scripts (public)
├── pipeline/
│   ├── config.py            ← Court geometry constants
│   └── modules/
│       ├── court_detector.py      ← Court keypoints + camera calibration (local)
│       ├── shuttle_detector.py    ← TrackNetV3 local subprocess wrapper
│       ├── pose_detector.py       ← RTMPose + YOLO local inference
│       ├── shot_segmentor.py      ← Hit-frame → interval conversion
│       ├── trajectory_estimator.py  ← Physics ODE stub (see paper §3.3)
│       └── shot_classifier.py       ← BST-CG-AP stub (see paper §3.4)
```

---

## Setup

### 1. Clone & install

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt
```

### 2. Download model weights

| Weight file | What it is | Where to put it |
|-------------|-----------|-----------------|
| `player_detector.pt` | Custom YOLO player detector | `weights/player_detector.pt` |
| `TrackNet_best.pt` | TrackNetV3 shuttle tracker | `tracknetv3/TrackNet_best.pt` |
| `InpaintNet_best.pt` | TrackNetV3 inpainting model | `tracknetv3/InpaintNet_best.pt` |

> 📥 **[Download all weights (Google Drive)](https://drive.google.com/drive/folders/1P-t5z0vrIdcS_7WFX7smh2T7SeWLFbJe)**

### 3. Configure environment (optional — only for auto court detection)

```bash
cp .env.example .env
# Edit .env with your Roboflow credentials
```

> If you prefer **manual court annotation** (click 6 points on a frame),
> you can skip the Roboflow setup entirely — see the Manual Calibration section below.

---

## Running the Pipeline

```bash
python run_pipeline.py \
    --video  path/to/rally.mp4 \
    --hits   10 45 90 130      \
    --yolo   weights/player_detector.pt \
    --output results/
```

| Argument | Description |
|----------|-------------|
| `--video` | Path to an input rally video |
| `--hits`  | Space-separated frame indices where the racket contacts the shuttle |
| `--yolo`  | Path to the YOLO player-detector checkpoint |
| `--output` | Directory for cached intermediate results (default: `results/`) |
| `--skip-classify` | Skip Stage 4 (model weights not yet public) |

Stages 2a (shuttle, TrackNetV3) and 2b (pose, RTMPose) run **in parallel**
using `ThreadPoolExecutor`. MMPose dependencies are installed automatically
on first run.

### Manual Court Calibration (no Roboflow needed)

```python
from pipeline.modules.court_detector import annotate_frame, solve_camera
import cv2

frame_path = "my_frame.jpg"
img        = cv2.imread(frame_path)
h, w       = img.shape[:2]

# Opens an interactive matplotlib window — click 6 points in order
img_pts = annotate_frame(frame_path)
calib   = solve_camera(img_pts, (w, h))
print(f"Reprojection error: {calib['reproj_error']:.3f} px")
```

---

## Supported Stroke Types (ShuttleSet17)

| # | Stroke | # | Stroke |
|---|--------|---|--------|
| 1 | Net Shot | 10 | Drop |
| 2 | Return Net | 11 | Passive Drop |
| 3 | Smash | 12 | Push |
| 4 | Wrist Smash | 13 | Rush (Kill) |
| 5 | Lob | 14 | Defensive Return Drive |
| 6 | Defensive Return Lob | 15 | Cross-court Net Shot |
| 7 | Clear | 16 | Short Service |
| 8 | Drive | 17 | Long Service |
| 9 | Back-court Drive | — | Unknown |

---

## Withheld Components

The following components contain the **novel contributions** of the paper and
are intentionally not distributed in this release:

| Component | Reason |
|-----------|--------|
| `trajectory_estimator.py` — ODE integrator & loss | Core physics contribution (§3.3) |
| `shot_classifier.py` — BST-CG-AP forward pass | Core model contribution (§3.4) |
| `stroke/bst_5d.py` — model definition | Architecture IP |
| `stroke/best_bst_5d_global.pt` — trained weights | Model IP |
| Training scripts | Experimental details under review |

All withheld components will be released together with the accepted
manuscript.

---

## Citation

> *Citation will be added on paper acceptance.*

---

## License

This repository is released under the **MIT License** for the pipeline
scaffolding code.  Model weights and architecture files are subject to a
separate research-use licence that will be published alongside the paper.
