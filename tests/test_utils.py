import numpy as np

from swingot.metrics import foscttm


def test_foscttm_for_identical_pairs():
    points = np.eye(4, dtype=float)
    assert foscttm(points, points) == 0.0
