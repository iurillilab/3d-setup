"""
Constants for mouse pose keypoints and skeleton structure.
"""
KEYPOINT_NAMES = [
    'nose', 'ear_lf', 'forepaw_lf', 'hindpaw_lf', 'tailbase',
    'hindpaw_rt', 'forepaw_rt', 'ear_rt', 'belly_rostral',
    'belly_caudal', 'back_caudal', 'back_mid', 'back_rostral'
]

SKELETON_EDGES = [
    (0, 1), (0, 7), (1, 7), (1, 12), (7, 12), (12, 11),
    (11, 10), (10, 4), (1, 8), (7, 6), (8, 2), (8, 6),
    (2, 9), (9, 3), (9, 6), (9, 4), (5, 9)
]



