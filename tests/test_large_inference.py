import numpy as np

from src.large_inference import TileDetection, global_nms, tile_starts


def test_tile_starts_covers_the_right_edge_once():
    assert tile_starts(10_000, 800, 480)[-1] == 9_200
    assert tile_starts(600, 800, 480) == [0]


def test_global_nms_merges_cross_tile_duplicates_by_coarse_group():
    first = TileDetection(np.array([100, 100, 200, 200], dtype=np.float32), 0.9, 0)
    duplicate_other_fine_class = TileDetection(np.array([105, 105, 205, 205], dtype=np.float32), 0.8, 1)
    other = TileDetection(np.array([500, 500, 600, 600], dtype=np.float32), 0.7, 24)
    assert len(global_nms([first, duplicate_other_fine_class, other], 0.5, "coarse")) == 2
    assert len(global_nms([first, duplicate_other_fine_class, other], 0.5, "fine")) == 3
