"""
modules/pose_detector.py
------------------------
Runs RTMPose + YOLO player detection locally using MMPose / MMDet.

Dependencies (installed automatically on first run if missing):
    openmim, mmengine, mmdet, mmpose, ultralytics
    A custom YOLO player-detection checkpoint: weights/player_detector.pt
"""

import os
import subprocess
import sys
import logging
import pickle
from dataclasses import dataclass, field
from typing import Optional, List

import cv2
import numpy as np
from scipy.optimize import minimize

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Runtime dependency bootstrap (mirrors the HF Space install logic)
# ---------------------------------------------------------------------------

def _ensure_dependencies():
    try:
        import mmdet, mmpose  # noqa: F401
        return
    except ImportError:
        pass

    logger.info("Installing MMDet / MMPose (first-time setup)…")
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "setuptools==69.5.1", "wheel", "openmim"])
    import torch
    torch_v  = torch.__version__.split("+")[0]
    cuda_v   = "cu121" if torch.version.cuda and "12" in str(torch.version.cuda) else "cu118"
    url      = f"https://download.openmmlab.com/mmcv/dist/{cuda_v}/torch{torch_v}/index.html"
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "mmcv==2.1.0", "--no-build-isolation", "-f", url])
    subprocess.check_call([sys.executable, "-m", "mim", "install", "mmengine>=0.10.0"])
    subprocess.check_call([sys.executable, "-m", "mim", "install", "mmdet>=3.0.0"])
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "mmpose>=1.1.0", "--no-deps"])
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "xtcocotools", "json-tricks", "munkres"])


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PlayerPose:
    """Pose data for one player in a single frame."""
    left_ankle_px:  Optional[List[float]] = None
    right_ankle_px: Optional[List[float]] = None
    keypoints:      Optional[List]        = None   # COCO-17 keypoints [x, y, score]
    bbox:           Optional[List[float]] = None   # [x1, y1, x2, y2]
    floor_pos_3d:   Optional[List[float]] = None   # Ankle midpoint in world coords
    body_pos_3d:    Optional[List[float]] = None   # Hip midpoint in world coords
    confidence:     float                 = 0.0


@dataclass
class PoseFrame:
    """Near- and far-player poses for one video frame."""
    frame_idx: int
    near: PlayerPose = field(default_factory=PlayerPose)
    far:  PlayerPose = field(default_factory=PlayerPose)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

# COCO-17 keypoint indices used for player assignment
_L_ANKLE, _R_ANKLE, _L_HIP, _R_HIP = 15, 16, 11, 12

def _optimise_K(world_pts, image_pts, img_w, img_h):
    """L-BFGS-B optimisation of (fx, fy, cx, cy) — no square-pixel assumption."""
    def err(p):
        fx, fy, cx, cy = p
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        ok, rv, tv = cv2.solvePnP(world_pts, image_pts, K, None,
                                   flags=cv2.SOLVEPNP_SQPNP)
        if not ok:
            return 1e9
        proj, _ = cv2.projectPoints(world_pts, rv, tv, K, None)
        return float(np.mean(np.linalg.norm(proj.reshape(-1, 2) - image_pts, axis=1)))

    x0     = [img_w * 2.0, img_w * 2.0, img_w / 2.0, img_h / 2.0]
    bounds = [(img_w * 0.8, img_w * 5.0), (img_w * 0.8, img_w * 5.0),
              (img_w * 0.35, img_w * 0.65), (img_h * 0.35, img_h * 0.65)]
    res    = minimize(err, x0, method="L-BFGS-B", bounds=bounds)
    fx, fy, cx, cy = res.x
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


def _backproject_to_floor(u, v, K, rvec, tvec, target_z=0.0):
    """Back-project pixel (u, v) onto world plane Z = target_z."""
    R, _  = cv2.Rodrigues(rvec)
    ray   = np.linalg.inv(K) @ np.array([u, v, 1.0])
    Rt    = R.T
    tv    = tvec.flatten()
    den   = (Rt @ ray)[2]
    if abs(den) < 1e-10:
        return None
    lam = (target_z + (Rt @ tv)[2]) / den
    return (Rt @ (ray * lam - tv)).tolist()


# ---------------------------------------------------------------------------
# PoseDetector
# ---------------------------------------------------------------------------

# Standard badminton court anchor points (metres) — must match calibration order
_WORLD_PTS = np.array([
    [0.0, 0.0, 0.0], [6.7, 0.0, 0.0], [6.7, 13.4, 0.0],
    [0.0, 13.4, 0.0], [0.0, 6.7, 1.55], [6.7, 6.7, 1.55],
], dtype=np.float64)


class PoseDetector:
    """
    Detects COCO-17 skeletal keypoints for the near and far player in
    every frame of a rally video.

    Detection strategy
    ------------------
    1. YOLO (custom player-detector) proposes bounding boxes.
    2. RTMDet (mmdet) filters boxes by person confidence.
    3. The intersection (IoU > 0.1) is sent to RTMPose (mmpose) for keypoints.
    4. Players are assigned near/far by bounding-box bottom-y and tracked by IoU.
    5. Ankle keypoints are back-projected to world-space floor coordinates
       using the camera calibration computed from *points_dict*.

    Parameters
    ----------
    yolo_path : path to the custom YOLO player-detector checkpoint (.pt).
                Download from the link in the README.

    Batch size
    ----------
    batch_size controls how many frames are fed to RTMDet at once.
    Reduce if you run out of GPU memory.
    """

    # Public RTMDet / RTMPose checkpoint URLs (OpenMMLab)
    _DET_CFG  = "rtmdet_tiny_8xb32-300e_coco.py"
    _DET_CKPT = ("https://download.openmmlab.com/mmdetection/v3.0/rtmdet/"
                 "rtmdet_tiny_8xb32-300e_coco/"
                 "rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth")
    _POSE_CFG  = "rtmpose-m_8xb256-420e_coco-256x192.py"
    _POSE_CKPT = ("https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/"
                  "rtmpose-m_simcc-coco_pt-aic-coco_420e-256x192-d8dd5ca4_20230127.pth")

    def __init__(self, yolo_path: str = "weights/player_detector.pt"):
        _ensure_dependencies()

        import torch
        from ultralytics import YOLO
        from mmdet.apis import init_detector
        from mmpose.apis import init_model
        import mmdet, mmpose  # noqa: F401

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"PoseDetector running on: {self.device}")

        det_cfg_path = os.path.join(
            os.path.dirname(__import__("mmdet").__file__),
            f".mim/configs/rtmdet/{self._DET_CFG}"
        )
        pose_cfg_path = os.path.join(
            os.path.dirname(__import__("mmpose").__file__),
            f".mim/configs/body_2d_keypoint/rtmpose/coco/{self._POSE_CFG}"
        )

        self.det_model  = init_detector(det_cfg_path,  self._DET_CKPT,  device=self.device)
        self.pose_model = init_model(pose_cfg_path, self._POSE_CKPT, device=self.device)
        self.yolo       = YOLO(yolo_path)

    def detect(self, video_path: str, points_dict: dict,
               batch_size: int = 24) -> List[PoseFrame]:
        """
        Parameters
        ----------
        video_path   : path to the rally video.
        points_dict  : six court keypoints returned by CourtDetector.detect(),
                       keys: FarLeft, FarRight, NearLeft, NearRight,
                             UpperNetLeft, UpperNetRight.
        batch_size   : frames processed per RTMDet call.

        Returns
        -------
        List[PoseFrame] — one entry per video frame.
        """
        from mmengine.registry import init_default_scope
        from mmdet.apis import inference_detector
        from mmpose.apis import inference_topdown

        order    = ["NearLeft", "NearRight", "FarRight", "FarLeft",
                    "UpperNetLeft", "UpperNetRight"]
        img_pts  = np.array([[int(points_dict[k][0]), int(points_dict[k][1])]
                              for k in order], dtype=np.float64)

        cap = cv2.VideoCapture(video_path)
        w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Camera calibration from court keypoints
        K = _optimise_K(_WORLD_PTS, img_pts, w, h)
        _, rv, tv = cv2.solvePnP(_WORLD_PTS, img_pts, K, None,
                                  flags=cv2.SOLVEPNP_ITERATIVE)
        rv, tv = cv2.solvePnPRefineLM(_WORLD_PTS, img_pts, K, None, rv, tv)
        court_poly = img_pts[:4].astype(np.int32)

        results = []
        last_n, last_f = None, None
        f_idx = 0

        while True:
            frames = []
            for _ in range(batch_size):
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            if not frames:
                break

            # RTMDet person detection
            init_default_scope("mmdet")
            det_results = inference_detector(self.det_model, frames)

            init_default_scope("mmpose")
            for j, frame in enumerate(frames):
                pf     = PoseFrame(f_idx)
                y_boxes = (self.yolo.predict(frame, conf=0.3, verbose=False)[0]
                           .boxes.xyxy.cpu().numpy())
                inst   = det_results[j].pred_instances
                r_boxes = inst.bboxes.cpu().numpy()[
                    (inst.scores.cpu().numpy() > 0.45) &
                    (inst.labels.cpu().numpy() == 0)
                ]

                # Keep only YOLO boxes that overlap with an RTMDet person box
                final_bboxes = []
                for yb in y_boxes:
                    ious = [self._iou(yb, rb) for rb in r_boxes]
                    if ious and max(ious) > 0.1:
                        final_bboxes.append(r_boxes[np.argmax(ious)])

                if final_bboxes:
                    poses = inference_topdown(
                        self.pose_model, frame,
                        bboxes=np.array(final_bboxes)
                    )
                    cands = []
                    for k_idx, res in enumerate(poses):
                        kp = res.pred_instances.keypoints[0]
                        cn = res.pred_instances.keypoint_scores[0]
                        box = final_bboxes[k_idx]
                        if (self._is_on_court(kp, cn, box, court_poly, last_n) or
                                self._is_on_court(kp, cn, box, court_poly, last_f)):
                            ank_idx = _L_ANKLE if cn[_L_ANKLE] > cn[_R_ANKLE] else _R_ANKLE
                            f3d = _backproject_to_floor(
                                kp[ank_idx][0], kp[ank_idx][1], K, rv, tv, 0.0
                            )
                            cands.append({
                                "kp": kp.tolist(), "box": box.tolist(),
                                "cn": cn.tolist(), "f3d": f3d,
                                "conf": float(np.mean(cn[[_L_HIP, _R_HIP,
                                                           _L_ANKLE, _R_ANKLE]])),
                            })

                    if cands:
                        # Near player: largest bottom-y (closest to camera)
                        c_n = (max(cands, key=lambda x: self._iou(x["box"], last_n))
                               if last_n is not None
                               else max(cands, key=lambda x: x["box"][3]))
                        pf.near = PlayerPose(keypoints=c_n["kp"], bbox=c_n["box"],
                                             floor_pos_3d=c_n["f3d"],
                                             confidence=c_n["conf"])
                        last_n = c_n["box"]

                        # Far player: remaining candidate with smallest bottom-y
                        rem = [c for c in cands if c["box"] != c_n["box"]]
                        if rem:
                            c_f = (max(rem, key=lambda x: self._iou(x["box"], last_f))
                                   if last_f is not None
                                   else min(rem, key=lambda x: x["box"][3]))
                            pf.far = PlayerPose(keypoints=c_f["kp"], bbox=c_f["box"],
                                                floor_pos_3d=c_f["f3d"],
                                                confidence=c_f["conf"])
                            last_f = c_f["box"]

                results.append(pf)
                f_idx += 1

        cap.release()
        logger.info(f"Pose detection complete — {f_idx} frames processed.")
        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _iou(b1, b2) -> float:
        if b1 is None or b2 is None:
            return 0.0
        x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
        x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
        if x2 < x1 or y2 < y1:
            return 0.0
        it = (x2 - x1) * (y2 - y1)
        return it / ((b1[2]-b1[0])*(b1[3]-b1[1]) +
                     (b2[2]-b2[0])*(b2[3]-b2[1]) - it + 1e-6)

    def _is_on_court(self, kp, cn, box, poly, last_box) -> bool:
        if last_box is not None and self._iou(box, last_box) > 0.15:
            return True
        return any(
            cn[i] > 0.3 and
            cv2.pointPolygonTest(poly, (float(kp[i, 0]), float(kp[i, 1])), False) >= 0
            for i in [_L_ANKLE, _R_ANKLE]
        )
