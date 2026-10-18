import numpy as np

# Standard Badminton Court Dimensions (in meters)
COURT_LENGTH = 13.4
COURT_WIDTH = 6.1
NET_HEIGHT = 1.55
NET_Y = 6.7

# 3D anchor points for the 6 detected court keypoints
# Order must match detection order in court_detector.py:
# [FarLeft, FarRight, NearLeft, NearRight, UpperNetLeft, UpperNetRight]
COURT_POINTS_3D = np.array([
    [0,           COURT_LENGTH, 0],           # FarLeft
    [COURT_WIDTH, COURT_LENGTH, 0],           # FarRight
    [0,           0,           0],            # NearLeft
    [COURT_WIDTH, 0,           0],            # NearRight
    [0,           NET_Y,       NET_HEIGHT],   # UpperNetLeft
    [COURT_WIDTH, NET_Y,       NET_HEIGHT],   # UpperNetRight
], dtype=np.float32)
