"""
run_pipeline.py
---------------
End-to-end inference script for the Monocular Badminton Analytics System.

This script runs the four-stage pipeline described in the paper entirely
locally — no cloud APIs or Gradio required:

  Stage 1 : Court keypoint detection  →  camera calibration (K, R, t, P)
  Stage 2a: Shuttlecock tracking       (TrackNetV3, local subprocess)
  Stage 2b: Player pose estimation     (RTMPose + YOLO, local GPU/CPU)
  Stage 3 : Physics-informed 3-D trajectory reconstruction
  Stage 4 : Stroke classification      (BST-CG-AP — withheld pending publication)

Usage
-----
    python run_pipeline.py \\
        --video      path/to/rally.mp4 \\
        --hits       10 45 90 130      \\
        --yolo       weights/player_detector.pt \\
        --output     results/

Requirements
------------
    pip install -r requirements.txt
    Copy .env.example → .env  (only needed for Roboflow court detection)

    Download model weights from the link in the README and place at:
        weights/player_detector.pt
        tracknetv3/TrackNet_best.pt
        tracknetv3/InpaintNet_best.pt
"""

import argparse
import concurrent.futures
import logging
import os
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np

# Pipeline modules
from pipeline.modules.court_detector    import CourtDetector
from pipeline.modules.shuttle_detector  import ShuttleDetector
from pipeline.modules.pose_detector     import PoseDetector
from pipeline.modules.shot_segmentor    import ShotSegmentor
from pipeline.modules.trajectory_estimator import TrajectoryEstimator
from pipeline.modules.shot_classifier   import ShotClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger("run_pipeline")


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def save_cache(out_dir: Path, name: str, data) -> None:
    with open(out_dir / f"{name}.pkl", "wb") as f:
        pickle.dump(data, f)
    logger.info(f"Saved {name}.pkl  →  {out_dir}")


def load_cache(out_dir: Path, name: str):
    p = out_dir / f"{name}.pkl"
    if p.exists():
        with open(p, "rb") as f:
            return pickle.load(f)
    return None


def get_video_info(video_path: str) -> tuple[int, float, int, int]:
    """Returns (total_frames, fps, width, height)."""
    cap = cv2.VideoCapture(video_path)
    n   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return n, fps, w, h


# ---------------------------------------------------------------------------
# Stage 1 — Court detection + calibration
# ---------------------------------------------------------------------------

def stage_court(video_path: str, out_dir: Path) -> tuple[dict, dict]:
    logger.info("=== Stage 1: Court Detection & Calibration ===")
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Cannot read first frame from {video_path}")

    h, w = frame.shape[:2]
    tmp  = str(out_dir / "_tmp_frame.jpg")
    cv2.imwrite(tmp, frame)

    detector = CourtDetector()
    pts      = detector.detect(tmp)
    os.remove(tmp)

    # Sanitise NumPy int types to plain Python int for JSON serialisability
    pts = {k: (int(v[0]), int(v[1])) for k, v in pts.items()}
    calib = detector.calibrate(pts, (w, h))

    logger.info(f"  Keypoints detected : {list(pts.keys())}")
    logger.info(f"  Reprojection error : {calib['reproj_error']:.4f} px")

    save_cache(out_dir, "pts",   pts)
    save_cache(out_dir, "calib", calib)
    return pts, calib


# ---------------------------------------------------------------------------
# Stage 2a — Shuttle tracking  (run in parallel with pose)
# ---------------------------------------------------------------------------

def stage_shuttle(video_path: str, out_dir: Path) -> dict:
    logger.info("=== Stage 2a: Shuttlecock Tracking ===")
    detector = ShuttleDetector()
    data     = detector.detect(video_path)
    if data:
        save_cache(out_dir, "shuttle", data)
        logger.info(f"  Detections received : {len(data.get('detections', []))} frames")
    else:
        logger.warning("  Shuttle detection returned no data.")
    return data or {}


# ---------------------------------------------------------------------------
# Stage 2b — Pose estimation  (run in parallel with shuttle)
# ---------------------------------------------------------------------------

def stage_pose(video_path: str, pts: dict, out_dir: Path,
               yolo_path: str = "weights/player_detector.pt") -> list:
    logger.info("=== Stage 2b: Player Pose Estimation ===")
    detector = PoseDetector(yolo_path=yolo_path)
    data     = detector.detect(video_path, pts)
    if data:
        save_cache(out_dir, "pose", data)
        logger.info(f"  Pose frames received : {len(data)}")
    else:
        logger.warning("  Pose estimation returned no data.")
    return data or []


# ---------------------------------------------------------------------------
# Stage 3 — 3-D trajectory reconstruction
# ---------------------------------------------------------------------------

def stage_trajectory(
    video_path:    str,
    hit_frames:    list,
    shuttle_cache: dict,
    pose_cache:    list,
    calib_cache:   dict,
    out_dir:       Path,
) -> dict:
    logger.info("=== Stage 3: 3-D Trajectory Reconstruction ===")

    n_frames, fps, _, _ = get_video_info(video_path)
    segmentor = ShotSegmentor()
    intervals = segmentor.get_intervals(hit_frames)

    if not intervals:
        logger.warning("  No shot intervals — check your --hits argument.")
        return {}

    # Build dense 2-D shuttle array (NaN = not detected)
    shuttle_2d = np.full((n_frames, 2), np.nan)
    for det in shuttle_cache.get("detections", []):
        if det["vis"] == 1:
            shuttle_2d[det["frame"]] = [det["x"], det["y"]]

    P = np.array(calib_cache["P"])
    estimator  = TrajectoryEstimator()
    traj_3d_dict = {}

    for i, (start, end) in enumerate(intervals):
        shot_id = i + 1
        h_3d = (pose_cache[start].near.floor_pos_3d
                if pose_cache and start < len(pose_cache) else None)
        r_3d = (pose_cache[end].far.floor_pos_3d
                if pose_cache and end   < len(pose_cache) else None)

        traj_3d, err = estimator.estimate_shot(
            shot_id, shuttle_2d[start:end], P, h_3d, r_3d, fps
        )
        traj_3d_dict[shot_id] = traj_3d
        logger.info(f"  Shot {shot_id} ({start}–{end}) | error: {err:.2f} px")

    save_cache(out_dir, "traj_3d", traj_3d_dict)
    return traj_3d_dict


# ---------------------------------------------------------------------------
# Stage 4 — Stroke classification
# ---------------------------------------------------------------------------

def stage_classify(
    video_path:    str,
    hit_frames:    list,
    shuttle_cache: dict,
    pose_cache:    list,
    calib_cache:   dict,
    traj_3d_dict:  dict,
    out_dir:       Path,
) -> list:
    logger.info("=== Stage 4: Stroke Classification ===")

    n_frames, _, w, h = get_video_info(video_path)
    segmentor = ShotSegmentor()
    intervals = segmentor.get_intervals(hit_frames)

    shuttle_2d = np.full((n_frames, 2), np.nan)
    for det in shuttle_cache.get("detections", []):
        if det["vis"] == 1:
            shuttle_2d[det["frame"]] = [det["x"], det["y"]]

    classifier = ShotClassifier()          # NOTE: raises NotImplementedError in this release
    results    = classifier.classify_rally_shots(
        video_path, intervals, shuttle_2d, pose_cache, calib_cache, traj_3d_dict
    )

    save_cache(out_dir, "classifications", results)
    logger.info("  Results:")
    for row in results:
        logger.info(f"    {row}")
    return results


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(
        description="Monocular Badminton Analytics — end-to-end pipeline"
    )
    ap.add_argument("--video",  required=True,
                    help="Path to the input rally video (.mp4 / .avi etc.)")
    ap.add_argument("--hits",   required=True, type=int, nargs="+",
                    help="Sorted list of frame indices where the racket "
                         "contacts the shuttle.  Example: --hits 10 45 90 130")
    ap.add_argument("--yolo",   default="weights/player_detector.pt",
                    help="Path to custom YOLO player-detector checkpoint "
                         "(default: weights/player_detector.pt).")
    ap.add_argument("--output", default="results",
                    help="Output directory for cached data  (default: results/)")
    ap.add_argument("--skip-classify", action="store_true",
                    help="Skip Stage 4 (classification model not yet public).")
    return ap.parse_args()


def main():
    args    = parse_args()
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Video  : {args.video}")
    logger.info(f"Hits   : {args.hits}")
    logger.info(f"Output : {out_dir}")

    # ---- Stage 1 ----
    pts, calib = stage_court(args.video, out_dir)

    # ---- Stages 2a + 2b  (run concurrently) ----
    logger.info("=== Stages 2a + 2b: Tracking (parallel) ===")
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        f_shuttle = ex.submit(stage_shuttle, args.video, out_dir)
        f_pose    = ex.submit(stage_pose,    args.video, pts, out_dir, args.yolo)
        shuttle_cache = f_shuttle.result()
        pose_cache    = f_pose.result()

    # ---- Stage 3 ----
    traj_3d_dict = stage_trajectory(
        args.video, args.hits,
        shuttle_cache, pose_cache, calib,
        out_dir,
    )

    # ---- Stage 4 (optional) ----
    if not args.skip_classify:
        try:
            stage_classify(
                args.video, args.hits,
                shuttle_cache, pose_cache, calib, traj_3d_dict,
                out_dir,
            )
        except NotImplementedError as e:
            logger.warning(
                "Stage 4 skipped — BST-CG-AP weights not yet available: "
                + str(e)
            )
    else:
        logger.info("Stage 4 skipped (--skip-classify).")

    logger.info("Pipeline complete.  Results written to: %s", out_dir)


if __name__ == "__main__":
    main()
