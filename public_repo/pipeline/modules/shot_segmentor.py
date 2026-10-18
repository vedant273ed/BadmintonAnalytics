"""
modules/shot_segmentor.py
--------------------------
Converts a list of manually-marked hit frames into shot intervals.
"""

import logging

logger = logging.getLogger(__name__)


class ShotSegmentor:
    """
    Converts a flat list of hit-frame indices into (start, end) intervals.

    Example
    -------
    >>> seg = ShotSegmentor()
    >>> seg.get_intervals([10, 45, 90, 130])
    [(10, 45), (45, 90), (90, 130)]
    """

    def get_intervals(self, hit_frames: list) -> list:
        """
        Parameters
        ----------
        hit_frames : sorted list of frame indices where the racket contacts
                     the shuttle (obtained from the manual navigator).

        Returns
        -------
        List of (start_frame, end_frame) tuples — one per shot interval.
        """
        if len(hit_frames) < 2:
            logger.warning("Need at least 2 marked frames to form an interval.")
            return []
        return [(hit_frames[i], hit_frames[i + 1])
                for i in range(len(hit_frames) - 1)]
