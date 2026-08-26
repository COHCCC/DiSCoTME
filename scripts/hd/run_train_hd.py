#!/usr/bin/env python
"""VisiumHD training entry -- a thin wrapper around the shared, UNMODIFIED trainer.

It does three things, then calls the exact same `main()` the regular-Visium
pipeline uses, so every shared file stays byte-for-byte the GitHub original:

  1. register the HD sub-bin encoders (importing `src.hd`);
  2. inject the HD k-NN neighbor builder -- HD selects the `num_local` NEAREST
     sub-bins, whereas regular Visium keeps the shared ball-query+random-pad
     builder. We rebind the name INSIDE the dataset module, so no shared file on
     disk is edited and regular-Visium runs are unaffected;
  3. call the shared `run_train_internal_test.main()`.

Select the HD gene encoder via `--gene-encoder-type subbin_gene_encoder_n16`
(or _n9 / _mean / _mlp / ...); pass the relpos config via
`--config-path configs/hd/visiumhd_crc_dilated_n120_relpos.yaml`.
"""
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(os.path.dirname(_here))  # scripts/hd/ -> scripts -> project root
if _root not in sys.path:
    sys.path.insert(0, _root)

# 1) Register VisiumHD encoders into the shared registries.
import src.hd  # noqa: F401,E402

# 2) Inject the HD k-NN neighbor builder (no shared file edited on disk).
import src.data.dataset as _dataset  # noqa: E402
from src.hd.neighbors import build_spot_contexts_knn  # noqa: E402
_dataset.build_spot_contexts_fast = build_spot_contexts_knn

# 3) Reuse the shared, unmodified training entrypoint.
from scripts.run_train_internal_test import main  # noqa: E402

if __name__ == "__main__":
    main()
