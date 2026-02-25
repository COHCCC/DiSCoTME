# src/utils/context_utils.py

import numpy as np
import random
from scipy.spatial import cKDTree

def build_spot_contexts_fast(df, num_local, num_global, local_distance):
    coords = df[['pxl_row_in_fullres', 'pxl_col_in_fullres']].values  # [N, 2]
    spot_ids = df['spot_id'].values                                   # [N,]
    N = len(df)

    tree = cKDTree(coords)

    spot_contexts = {}
    all_spots = list(spot_ids)

    for i, (target_id, target_pos) in enumerate(zip(spot_ids, coords)):
        # 1) local candidates (ball query) 
        idxs_local_from_ball = tree.query_ball_point(target_pos, r=local_distance)
        local_candidates_indices = [idx for idx in idxs_local_from_ball if idx != i]

        # 2) sort local candidates by distance and take top num_local
        if not local_candidates_indices:
            sorted_pairs = []
        else:
            candidate_coords = coords[local_candidates_indices]
            local_dists = np.sqrt(np.sum((candidate_coords - target_pos) ** 2, axis=1))
            sorted_pairs = sorted(zip(local_candidates_indices, local_dists), key=lambda x: x[1])

        # 取 num_local
        if len(sorted_pairs) > num_local:
            local_spots = [spot_ids[idx] for idx, dist in sorted_pairs[:num_local]]
        else:
            local_spots = [spot_ids[idx] for idx, dist in sorted_pairs]
            if num_local > 0 and len(local_spots) < num_local:
                if len(local_spots) > 0:
                    repeats_needed = num_local - len(local_spots)
                    local_spots.extend(random.choices(local_spots, k=repeats_needed))
                else:
                    other_spots_ids = [s_id for s_id in all_spots if s_id != target_id]
                    if other_spots_ids:
                        local_spots = random.sample(other_spots_ids, min(num_local, len(other_spots_ids)))
                    else:
                        local_spots = []

        # 3) global spots
        global_spots = []
        local_set = set(local_spots)
        remaining_spots_ids = [s_id for s_id in all_spots if s_id not in local_set and s_id != target_id]

        if num_global > 0:
            if len(remaining_spots_ids) > num_global:
                global_spots = random.sample(remaining_spots_ids, num_global)
            else:
                global_spots = remaining_spots_ids.copy()
                if 0 < len(global_spots) < num_global:
                    repeats_needed = num_global - len(global_spots)
                    global_spots.extend(random.choices(global_spots, k=repeats_needed))
                elif len(global_spots) == 0:
                    other_spots_ids_for_global = [s_id for s_id in all_spots if s_id != target_id]
                    if other_spots_ids_for_global:
                        global_spots = random.sample(
                            other_spots_ids_for_global, min(num_global, len(other_spots_ids_for_global))
                        )

        spot_contexts[target_id] = {
            'local': local_spots,
            'global': global_spots
        }

    return spot_contexts