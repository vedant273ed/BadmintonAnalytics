"""
modules/court_detector.py
--------------------------
Detects badminton court keypoints and calibrates the camera.

Two usage modes:

  1. **Automatic** (default in the pipeline): Uses Roboflow Inference to detect
     court corners and netline keypoints from a frame image.

  2. **Manual** (for calibration scripts): Launches an interactive matplotlib
     GUI where you click 6 court landmarks on a frame.

Camera model
------------
Intrinsics (fx, fy, cx, cy) are optimised jointly via L-BFGS-B, allowing for
non-square pixels and off-centre optical axes common in broadcast telephotos.
Extrinsics are solved via OpenCV's SOLVEPNP_ITERATIVE + Levenberg-Marquardt
sub-pixel refinement.

Dependencies
------------
  pip install inference-sdk python-dotenv opencv-python scipy matplotlib
  (Roboflow key only required in auto mode)
"""

import logging
import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Court geometry — 6 keypoints in world coordinates (metres)
# ---------------------------------------------------------------------------
# Point order:
#   0  NearLeft      1  NearRight
#   2  FarRight      3  FarLeft
#   4  UpperNetLeft  5  UpperNetRight
#
# (This order is used by both annotation GUI and Roboflow workflow.)
# ---------------------------------------------------------------------------
COURT_W, COURT_L, NET_Y, NET_H = 6.7, 13.4, 6.7, 1.55

WORLD_PTS = np.array([
    [0.0,    0.0,    0.0],   # NearLeft
    [COURT_W, 0.0,   0.0],   # NearRight
    [COURT_W, COURT_L, 0.0], # FarRight
    [0.0,    COURT_L, 0.0],  # FarLeft
    [0.0,    NET_Y,  NET_H], # UpperNetLeft
    [COURT_W, NET_Y, NET_H], # UpperNetRight
], dtype=np.float64)

POINT_LABELS = [
    "NearLeft", "NearRight", "FarRight", "FarLeft",
    "UpperNetLeft", "UpperNetRight",
]


# ---------------------------------------------------------------------------
# Calibration core  (shared by both modes)
# ---------------------------------------------------------------------------

def optimise_K(world_pts: np.ndarray, image_pts: np.ndarray,
               img_w: int, img_h: int) -> np.ndarray:
    """
    Minimise mean reprojection error over (fx, fy, cx, cy) independently.

    Freeing fy (vs. assuming square pixels) fixes systematic drift caused
    by telephoto lens pixel-aspect ratios deviating by 0.1–0.3 %.
    """
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
    bounds = [(img_w * 0.8,  img_w * 5.0),
              (img_w * 0.8,  img_w * 5.0),
              (img_w * 0.35, img_w * 0.65),
              (img_h * 0.35, img_h * 0.65)]
    res = minimize(err, x0, method="L-BFGS-B", bounds=bounds,
                   options={"ftol": 1e-10, "maxiter": 3000})
    fx, fy, cx, cy = res.x
    logger.info(f"K optimised — fx={fx:.1f} fy={fy:.1f} "
                f"cx={cx:.1f} cy={cy:.1f}  err={res.fun:.3f} px")
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


def solve_camera(image_pts: np.ndarray, img_size: tuple) -> dict:
    """
    Solve full camera model from 6 matched 2D–3D point pairs.

    Parameters
    ----------
    image_pts : (6, 2) array of pixel coordinates in POINT_LABELS order.
    img_size  : (width, height) of the source frame.

    Returns
    -------
    dict with keys: K, P, rvec, tvec, reproj_error
    """
    w, h = img_size
    K = optimise_K(WORLD_PTS, image_pts, w, h)

    ok, rvec, tvec = cv2.solvePnP(WORLD_PTS, image_pts, K, None,
                                   flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        raise RuntimeError("solvePnP failed — check point correspondence order.")

    rvec, tvec = cv2.solvePnPRefineLM(WORLD_PTS, image_pts, K, None, rvec, tvec)
    R, _  = cv2.Rodrigues(rvec)
    P     = K @ np.hstack([R, tvec])
    P    /= P[2, 3]   # normalise so P[2,3] == 1

    # Compute final reprojection error
    pts_h  = np.hstack([WORLD_PTS, np.ones((6, 1))])
    proj_h = (P @ pts_h.T).T
    proj   = proj_h[:, :2] / proj_h[:, 2:]
    err    = float(np.mean(np.linalg.norm(proj - image_pts, axis=1)))

    logger.info(f"Camera solved — reprojection error: {err:.4f} px")
    return {
        "K":           K.tolist(),
        "P":           P.tolist(),
        "rvec":        rvec.tolist(),
        "tvec":        tvec.tolist(),
        "reproj_error": err,
    }


def backproject_to_floor(u, v, K, rvec, tvec, target_z=0.0):
    """Back-project pixel (u, v) to world plane Z = target_z."""
    R, _  = cv2.Rodrigues(rvec) if not isinstance(rvec, np.ndarray) else (cv2.Rodrigues(rvec)[0], None)
    if isinstance(rvec, np.ndarray) and rvec.shape == (3, 3):
        R = rvec
    else:
        R, _ = cv2.Rodrigues(np.array(rvec))
    ray = np.linalg.inv(np.array(K)) @ np.array([u, v, 1.0])
    Rt  = R.T
    tv  = np.array(tvec).flatten()
    den = (Rt @ ray)[2]
    if abs(den) < 1e-10:
        return None
    lam = (target_z + (Rt @ tv)[2]) / den
    return (Rt @ (ray * lam - tv)).tolist()


# ---------------------------------------------------------------------------
# Mode 1 — Manual interactive annotation  (for standalone calibration)
# ---------------------------------------------------------------------------

def annotate_frame(image_path: str) -> np.ndarray:
    """
    Launch a matplotlib GUI to manually click 6 court keypoints.

    Click order (POINT_LABELS):
      1  NearLeft    2  NearRight    3  FarRight
      4  FarLeft     5  UpperNetLeft 6  UpperNetRight

    Returns (6, 2) float64 array of pixel coordinates.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot open: {image_path}")
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(figsize=(16, 10))
    ax.imshow(rgb)
    ax.set_title(
        "Court Calibration\n"
        "  1=Near-L corner   2=Near-R corner   3=Far-R corner\n"
        "  4=Far-L corner    5=Left post tip   6=Right post tip\n"
        "  Click INNER edge of white lines; post tip = top of yellow cap",
        fontsize=10, fontweight="bold", loc="left",
    )
    pts = np.array(plt.ginput(n=6, timeout=0, show_clicks=True), dtype=np.float64)
    plt.close(fig)
    return pts


# ---------------------------------------------------------------------------
# Mode 2 — Automatic keypoint detection  (Roboflow Inference)
# ---------------------------------------------------------------------------

class CourtDetector:
    """
    Detects the six court keypoints from a single frame using a Roboflow
    detection workflow (court corners + netline model), then solves the
    camera calibration.

    For manual click-based calibration, use :func:`annotate_frame` +
    :func:`solve_camera` directly — no Roboflow key required.

    Environment variables (see .env.example)
    -----------------------------------------
    ROBOFLOW_API_KEY
    ROBOFLOW_WORKSPACE
    ROBOFLOW_WORKFLOW_ID
    """

    def __init__(self):
        import dotenv
        from inference_sdk import InferenceHTTPClient
        dotenv.load_dotenv()
        self.client      = InferenceHTTPClient(
            api_url="https://serverless.roboflow.com",
            api_key=os.getenv("ROBOFLOW_API_KEY"),
        )
        self.workspace   = os.getenv("ROBOFLOW_WORKSPACE")
        self.workflow_id = os.getenv("ROBOFLOW_WORKFLOW_ID")

    def detect(self, frame_path: str) -> dict:
        """
        Run the Roboflow workflow and return detected keypoints as a dict::

            {
                "NearLeft": (x, y), "NearRight": (x, y),
                "FarLeft":  (x, y), "FarRight":  (x, y),
                "UpperNetLeft": (x, y), "UpperNetRight": (x, y),
            }
        """
        try:
            result = self.client.run_workflow(
                workspace_name=self.workspace,
                workflow_id=self.workflow_id,
                images={"image": frame_path},
                use_cache=True,
            )
            return self._parse_result(result[0])
        except Exception as e:
            logger.error(f"Roboflow court detection failed: {e}")
            return {}

    def calibrate(self, points_dict: dict, img_size: tuple) -> dict:
        """
        Solve the camera model from *points_dict* (output of :meth:`detect`).

        Returns the same dict as :func:`solve_camera`.
        """
        order    = ["NearLeft", "NearRight", "FarRight", "FarLeft",
                    "UpperNetLeft", "UpperNetRight"]
        img_pts  = np.array([[int(points_dict[k][0]), int(points_dict[k][1])]
                              for k in order], dtype=np.float64)
        return solve_camera(img_pts, img_size)

    # ------------------------------------------------------------------
    def _parse_result(self, data: dict) -> dict:
        """Extract the outermost keypoint from each Roboflow prediction box."""
        output = {}

        # --- Corner boxes ---
        corner_preds = data.get("court_corner_predictions", {}).get("predictions", [])
        if len(corner_preds) >= 4:
            centers = np.array([[p["x"], p["y"]] for p in corner_preds])

            def extreme(box_idx, crit):
                kps = corner_preds[box_idx].get("keypoints", [])
                if not kps:
                    return centers[box_idx]
                pts = np.array([[kp["x"], kp["y"]] for kp in kps])
                if crit == "min_sum":  return pts[np.argmin(pts.sum(axis=1))]
                if crit == "max_sum":  return pts[np.argmax(pts.sum(axis=1))]
                if crit == "min_diff": return pts[np.argmin(pts[:, 0] - pts[:, 1])]
                if crit == "max_diff": return pts[np.argmax(pts[:, 0] - pts[:, 1])]
                return centers[box_idx]

            # Assign boxes by geometric position
            idx_nl = np.argmin(centers.sum(axis=1))         # NearLeft  → min x+y
            idx_fr_full = np.argmax(centers.sum(axis=1))    # FarRight  → max x+y (bottom-right in image)
            others = [i for i in range(len(corner_preds))
                      if i not in [idx_nl, idx_fr_full]]

            output["NearLeft"]  = tuple(extreme(idx_nl,      "min_sum").astype(int))
            output["FarRight"]  = tuple(extreme(idx_fr_full, "max_sum").astype(int))

            if len(others) >= 2:
                s0 = centers[others[0], 0] - centers[others[0], 1]
                s1 = centers[others[1], 0] - centers[others[1], 1]
                idx_nr = others[0] if s0 > s1 else others[1]  # NearRight → max x-y
                idx_fl = others[1] if idx_nr == others[0] else others[0]
                output["NearRight"] = tuple(extreme(idx_nr, "max_diff").astype(int))
                output["FarLeft"]   = tuple(extreme(idx_fl, "min_diff").astype(int))

        # --- Net poles ---
        net_preds = data.get("predictions", {}).get("predictions", [])
        net_tops  = []
        for pred in net_preds:
            if pred["class"] == "Netline":
                kps = pred.get("keypoints", [])
                if kps:
                    top_kp = min(kps, key=lambda p: p["y"])
                    net_tops.append((int(top_kp["x"]), int(top_kp["y"])))

        if len(net_tops) >= 2:
            net_tops.sort(key=lambda p: p[0])
            output["UpperNetLeft"], output["UpperNetRight"] = net_tops[0], net_tops[1]

        return output
