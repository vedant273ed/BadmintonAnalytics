"""
modules/shuttle_detector.py
----------------------------
Runs TrackNetV3 locally to detect the shuttlecock position in every frame.

TrackNetV3 is called as a subprocess (predict_x.py) and its CSV output is
parsed into a list of per-frame detections.

Dependencies
------------
- The TrackNetV3 scripts must be in: tracknetv3/
- Download model weights from the link in the README and place them at:
      tracknetv3/TrackNet_best.pt
      tracknetv3/InpaintNet_best.pt
- pip install -r requirements.txt  (PyTorch + OpenCV)
"""

import csv
import logging
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)

# Paths to TrackNetV3 scripts and weights (relative to this file's directory)
_TRACKNET_DIR     = Path(__file__).parent.parent / "tracknetv3"
_PREDICT_SCRIPT   = _TRACKNET_DIR / "predict_x.py"
_WEIGHTS_TRACKNET = _TRACKNET_DIR / "TrackNet_best.pt"
_WEIGHTS_INPAINT  = _TRACKNET_DIR / "InpaintNet_best.pt"


class ShuttleDetector:
    """
    Runs TrackNetV3 inference on a rally video and returns per-frame
    shuttlecock detections.

    TrackNetV3 is invoked as a subprocess (predict_x.py) and the
    resulting CSV is parsed into a structured Python dict.

    Parameters
    ----------
    eval_mode  : ``"weight"`` (default, most accurate) or ``"nonoverlap"``
                 (faster, less accurate).
    batch_size : frames per TrackNetV3 forward pass.
    """

    def __init__(
        self,
        eval_mode:  str = "weight",
        batch_size: int = 16,
    ):
        if not _PREDICT_SCRIPT.exists():
            raise FileNotFoundError(
                f"TrackNetV3 predict script not found at {_PREDICT_SCRIPT}.\n"
                "Make sure the tracknetv3/ directory is present alongside "
                "this module."
            )
        if not _WEIGHTS_TRACKNET.exists() or not _WEIGHTS_INPAINT.exists():
            raise FileNotFoundError(
                "TrackNetV3 weights not found.\n"
                f"  Expected: {_WEIGHTS_TRACKNET}\n"
                f"            {_WEIGHTS_INPAINT}\n"
                "Download from the link in the README."
            )
        self.eval_mode  = eval_mode
        self.batch_size = batch_size

    def detect(self, video_path: str) -> dict:
        """
        Run TrackNetV3 on *video_path*.

        Returns
        -------
        dict with key ``"detections"``: list of per-frame dicts::

            {
                "frame": int,
                "vis":   int,    # 1 = visible, 0 = occluded / missing
                "x":     int,    # pixel x  (0 if not visible)
                "y":     int,    # pixel y  (0 if not visible)
            }

        Also includes ``"total_frames": int``.
        Returns an empty dict on failure.
        """
        request_id = str(uuid.uuid4())[:8]
        tmp_dir    = Path(f"_tmp_shuttle_{request_id}")
        tmp_dir.mkdir(parents=True, exist_ok=True)

        # Copy video to an isolated temp workspace
        tmp_video = tmp_dir / "input_video.mp4"
        shutil.copy(video_path, tmp_video)

        cmd = [
            sys.executable, str(_PREDICT_SCRIPT),
            "--video_file",      str(tmp_video.resolve()),
            "--tracknet_file",   str(_WEIGHTS_TRACKNET.resolve()),
            "--inpaintnet_file", str(_WEIGHTS_INPAINT.resolve()),
            "--save_dir",        str(tmp_dir.resolve()),
            "--eval_mode",       self.eval_mode,
            "--batch_size",      str(self.batch_size),
        ]

        logger.info(f"Running TrackNetV3 on: {video_path}")
        try:
            result = subprocess.run(
                cmd, check=True, capture_output=True, text=True
            )
            logger.debug(result.stdout)

            csv_path = tmp_dir / "input_video_ball.csv"
            if not csv_path.exists():
                logger.error("TrackNetV3 did not produce a CSV output.")
                return {}

            detections = []
            with open(csv_path, newline="") as f:
                for row in csv.DictReader(f):
                    detections.append({
                        "frame": int(row["Frame"]),
                        "vis":   int(row["Visibility"]),
                        "x":     int(float(row["X"])),
                        "y":     int(float(row["Y"])),
                    })

            logger.info(f"Shuttle detection complete — {len(detections)} frames.")
            return {"detections": detections, "total_frames": len(detections)}

        except subprocess.CalledProcessError as e:
            logger.error(f"TrackNetV3 subprocess failed:\n{e.stderr}")
            return {}
        except Exception as e:
            logger.error(f"ShuttleDetector error: {e}")
            return {}
        finally:
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir)
