# src/hd/neighbors.py
"""
VisiumHD-only neighbor selection: pure k-NN (the `num_local` NEAREST sub-bins),
ignoring the ball-query radius.

Regular Visium keeps using the shared, UNMODIFIED
`src/utils/context_utils.build_spot_contexts_fast` (ball-query within
`local_distance`, then random-pad if fewer than `num_local` fall inside). That
radius-then-pad behavior is what reproduces the paper and MUST NOT change.

Why HD needs its own builder: HD sub-bins are dense and the paper-style radius
(e.g. 500px) holds far fewer than `num_local`, so the shared builder would pad
~99% of sub-bins with random DUPLICATES. HD instead takes the `num_local`
NEAREST real sub-bins (k-NN). This function is INJECTED from the HD train/eval
wrappers, which rebind the name `build_spot_contexts_fast` inside the dataset
module -- so no shared file on disk is edited, and regular-Visium runs (which
never import this) are unaffected.

The signature matches `build_spot_contexts_fast` exactly so it is a drop-in.
"""
import random

import numpy as np
from scipy.spatial import cKDTree


def build_spot_contexts_knn(df, num_local, num_global, local_distance):
    """Drop-in for build_spot_contexts_fast: local = the `num_local` NEAREST spots.

    `local_distance` is accepted for signature compatibility but intentionally
    IGNORED (k-NN, not ball-query). `num_global > 0` is delegated to the shared,
    unmodified builder, since HD only ever uses `num_global == 0`.
    """
    if not (num_global == 0 and num_local > 0):
        # HD never uses global context; anything else -> shared original builder.
        from src.utils.context_utils import build_spot_contexts_fast
        return build_spot_contexts_fast(df, num_local, num_global, local_distance)

    coords = df[['pxl_row_in_fullres', 'pxl_col_in_fullres']].values
    spot_ids = df['spot_id'].values
    N = len(df)
    tree = cKDTree(coords)

    all_spots = list(spot_ids)
    spot_contexts = {}

    # +1 because the nearest point to each spot is itself (distance 0); drop it below.
    k = min(num_local + 1, N)
    _, idxs = tree.query(coords, k=k, workers=-1)
    if k == 1:
        idxs = idxs[:, None]

    for i, target_id in enumerate(spot_ids):
        local_indices = [int(idx) for idx in np.atleast_1d(idxs[i])
                         if int(idx) != i and int(idx) < N]
        local_spots = [spot_ids[idx] for idx in local_indices[:num_local]]
        # Pad ONLY when the whole slide has fewer than num_local other spots
        # (never happens at HD scale; kept for parity with the shared builder).
        if len(local_spots) < num_local:
            if local_spots:
                local_spots.extend(random.choices(local_spots, k=num_local - len(local_spots)))
            else:
                other = [s for s in all_spots if s != target_id]
                local_spots = random.sample(other, min(num_local, len(other)))
        spot_contexts[target_id] = {'local': local_spots, 'global': []}

    return spot_contexts
