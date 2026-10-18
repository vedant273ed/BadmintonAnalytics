"""
modules/shot_classifier.py
---------------------------
Classifies badminton stroke types using the BST-CG-AP fusion model described
in the paper.  The model operates on a 5-dimensional joint representation:

    • 2-D joint positions + bone vectors  (per player, COCO-17 skeleton)
    • 2-D normalised shuttle coordinates
    • 3-D normalised shuttle trajectory   (from TrajectoryEstimator)

For full architectural details, please refer to:

    [Paper title / arXiv link — to be added on acceptance]

This file exposes the public interface only.  The BST-CG-AP model definition
and trained weights are not distributed in this repository to comply with
journal double-blind review requirements.
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Stroke label mapping (Chinese ShuttleSet17 → English)
# ---------------------------------------------------------------------------

BST_TO_ENGLISH = {
    "放小球": "Net Shot",             "擋小球": "Return Net",
    "殺球":   "Smash",                "點扣":   "Wrist Smash",
    "挑球":   "Lob",                  "防守回挑": "Defensive Return Lob",
    "長球":   "Clear",                "平球":   "Drive",
    "後場抽平球": "Back-court Drive", "切球":   "Drop",
    "過渡切球": "Passive Drop",       "推球":   "Push",
    "撲球":   "Rush (Kill)",          "防守回抽": "Defensive Return Drive",
    "勾球":   "Cross-court Net Shot", "發短球": "Short Service",
    "發長球": "Long Service",         "未知球種": "Unknown",
}

COCO_BONES = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]

COURT_W, COURT_L, COURT_H = 6.1, 13.4, 5.0


class ShotClassifier:
    """
    Classifies the stroke type for each shot interval in a rally.

    Model:  BST-CG-AP (Badminton Stroke Transformer with Cross-attention
            Gating and Attention Pooling).

    Input feature vector per frame (dim = 5):
        [u/W, v/H, X/Lw, Y/Ll, Z/Lh]
        where (u,v) are 2-D shuttle pixel coordinates normalised by frame
        size, and (X,Y,Z) are the 3-D world-space coordinates normalised by
        court dimensions.

    Joint + bone representation:
        Per player: 17 × 2 joint coordinates + 17 × 2 bone vectors = 68-d
        (see Section 3.4 of the paper for the full encoding).

    Note
    ----
    The BST-CG-AP model weights and architecture (bst_5d.py) are withheld
    pending paper publication.  Weights will be released on acceptance.
    The feature extraction pipeline below is fully functional and documents
    the input format expected by the model.
    """

    def __init__(self, weight_path: str = "stroke/best_bst_5d_global.pt"):
        """
        Loads the BST-CG-AP checkpoint.

        Parameters
        ----------
        weight_path : path to the .pt checkpoint file (not distributed here).
        """
        raise NotImplementedError(
            "ShotClassifier requires the BST-CG-AP model weights and "
            "architecture file (stroke/bst_5d.py), which are not distributed "
            "in this public release.  They will be open-sourced together with "
            "the accepted manuscript.  See the paper for architectural details."
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify_rally_shots(
        self,
        video_path:    str,
        intervals:     list,
        shuttle_2d_all: np.ndarray,
        pose_cache:    list,
        calib_cache:   dict,
        traj_3d_dict:  dict,
    ) -> list:
        """
        Classify every shot interval in a rally.

        Parameters
        ----------
        video_path     : path to the source rally video (for frame dimensions).
        intervals      : list of (start_frame, end_frame) tuples from
                         ShotSegmentor.get_intervals().
        shuttle_2d_all : (total_frames, 2) array of 2-D shuttle detections;
                         NaN where the shuttle is not detected.
        pose_cache     : list[PoseFrame] — per-frame pose data from
                         PoseDetector.detect().
        calib_cache    : dict returned by CourtDetector.calibrate().
        traj_3d_dict   : dict mapping shot_id → (N, 3) trajectory array from
                         TrajectoryEstimator.estimate_shot().

        Returns
        -------
        list of [shot_label, hitter_side, stroke_type_english] rows.
        """
        raise NotImplementedError(
            "classify_rally_shots() requires the BST-CG-AP model — "
            "see __init__ docstring."
        )

    # ------------------------------------------------------------------
    # Feature extraction (fully documented — not withheld)
    # ------------------------------------------------------------------

    def _determine_hitter_side(
        self,
        hit_frame:   int,
        pose_cache:  list,
        shuttle_2d:  np.ndarray,
    ) -> str:
        """
        Determines which player (Top/Bottom) struck the shuttle based on
        proximity between each player's bounding-box centre and the shuttle
        position at the hit frame.

        Returns ``"Top"`` (far-side player) or ``"Bottom"`` (near-side).
        """
        if not pose_cache or hit_frame >= len(pose_cache):
            return "Bottom"
        pf = pose_cache[hit_frame]
        shut_pos = shuttle_2d[hit_frame]
        if np.isnan(shut_pos[0]):
            return "Bottom"

        def dist_to_shuttle(player):
            if player is None or player.bbox is None:
                return 1e9
            bx = (player.bbox[0] + player.bbox[2]) / 2
            by = (player.bbox[1] + player.bbox[3]) / 2
            return np.hypot(bx - shut_pos[0], by - shut_pos[1])

        return "Top" if dist_to_shuttle(pf.far) < dist_to_shuttle(pf.near) else "Bottom"

    def _extract_features(
        self,
        start:      int,
        end:        int,
        s2d_all:    np.ndarray,
        pose_cache: list,
        calib:      dict,
        traj_3d:    np.ndarray | None,
        W:          int,
        H:          int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Builds the three input tensors for BST-CG-AP from raw detection data.

        Returns
        -------
        jnb_seq : (100, 2, 72)  — joint + bone sequences for near/far players.
                  Channels 0:34  → normalised joint coordinates (17 × 2).
                  Channels 34:72 → bone vectors (17 × 2, see COCO_BONES).
        pos_seq : (100, 2, 2)   — normalised floor positions (X/Lw, Y/Ll).
        s5d_seq : (100, 5)      — 5-D shuttle feature (u/W, v/H, X/Lw, Y/Ll, Z/Lh).

        All sequences are resampled to exactly 100 frames via linear
        interpolation of frame indices.
        """
        jnb_seq = np.zeros((100, 2, 72), dtype=np.float32)
        pos_seq = np.zeros((100, 2, 2),  dtype=np.float32)
        s5d_seq = np.zeros((100, 5),     dtype=np.float32)

        indices = np.linspace(start, end - 1, 100).astype(int)

        for t, orig_f in enumerate(indices):
            # --- 5-D shuttle feature ---
            u, v = s2d_all[orig_f]
            s2d_norm = [u / W, v / H] if not np.isnan(u) else [0.0, 0.0]
            rel_idx  = orig_f - start
            if traj_3d is not None and rel_idx < len(traj_3d):
                s3d_norm = np.nan_to_num(traj_3d[rel_idx]) / [COURT_W, COURT_L, COURT_H]
            else:
                s3d_norm = [0.0, 0.0, 0.0]
            s5d_seq[t] = np.concatenate([s2d_norm, s3d_norm])

            # --- Joint + bone features ---
            if pose_cache and orig_f < len(pose_cache):
                pf = pose_cache[orig_f]
                for p_idx, p_attr in enumerate(["near", "far"]):
                    player = getattr(pf, p_attr)
                    if player and player.keypoints:
                        kpts = np.array(player.keypoints)[:, :2] / [W, H]
                        jnb_seq[t, p_idx, :34] = kpts.flatten()
                        bones = np.array([kpts[j1] - kpts[j2]
                                          for j1, j2 in COCO_BONES[:19]])
                        jnb_seq[t, p_idx, 34:72] = bones.flatten()
                        if player.floor_pos_3d:
                            pos_seq[t, p_idx] = (
                                np.nan_to_num(player.floor_pos_3d[:2])
                                / [COURT_W, COURT_L]
                            )
        return jnb_seq, pos_seq, s5d_seq
