import warnings

import numpy as np
from matplotlib.colors import LinearSegmentedColormap

def test_deduplicate_cmap():
    # Build a cold_hot colormap
    n_colors = 4
    cmap, value = html_stat_map._deduplicate_cmap(
        'cold_hot', n_colors=n_colors, annotate=True)

    assert cmap.N == n_colors
    assert value

    # Check that there are no duplicated colors
    cmaplist = [cmap(i) for i in range(cmap.N)]
    mask = [cmaplist.count(cmap(i)) == 1 for i in range(cmap.N)]
    assert np.min(mask)

    # Build a custom colormap with only repeated colors
    n_colors = 2
    cmap_custom = LinearSegmentedColormap.from_list(
        'Custom cmap', [[0, 0, 0], [1, 1, 1], [0, 0, 0], [1, 1, 1]], 4)
    cmap, value = html_stat_map._deduplicate_cmap(
        cmap_custom, n_colors=n_colors, annotate=True)

    assert cmap.N == cmap_custom.N

    # Check that annotation were automatically turned off because it is not
    # possible to deduplicate this map
    assert value is False
