#!/usr/bin/env python
"""VisiumHD evaluation entry -- a small, self-contained eval that:

  1. registers the HD sub-bin encoders (`import src.hd`);
  2. injects the HD k-NN neighbor builder (so eval selects neighbors the SAME
     way HD training did) -- no shared file on disk is edited;
  3. loads the model context config from a YAML (`--config-path`), so the relpos
     settings match the trained checkpoint.

The heavy lifting -- embedding extraction and PCA+KMeans clustering -- is REUSED
verbatim from the shared, UNMODIFIED `eval/run_eval_with_alpha.py`. Only the
orchestration that needs HD registration / k-NN / YAML config lives here (the
shared `main()` builds its config from a named preset and would ignore relpos).
"""
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(os.path.dirname(_here))  # eval/hd/ -> eval -> project root
if _root not in sys.path:
    sys.path.insert(0, _root)

import torch  # noqa: E402
import torchvision.transforms as T  # noqa: E402
import pandas as pd  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

# 1) register HD encoders
import src.hd  # noqa: F401,E402
# 2) inject HD k-NN neighbor builder (no shared file edited)
import src.data.dataset as _dataset  # noqa: E402
from src.hd.neighbors import build_spot_contexts_knn  # noqa: E402
_dataset.build_spot_contexts_fast = build_spot_contexts_knn

# Reuse the shared eval building blocks (unmodified file).
from eval.run_eval_with_alpha import (  # noqa: E402
    parse_args, do_pca_kmeans, extract_embeddings_with_alpha,
)
from src.data.dataset import MultiScaleContextDataset  # noqa: E402
from src.models.discotme_net import MultiScaleMultiModalModel  # noqa: E402
from src.models.dilated_blocks import DilatedConfigs  # noqa: E402


def _load_config(args):
    """HD addition: load the context config from YAML (`--config-path`) so relpos
    settings (relative_pos / pos_scale / neighbor_agg) match training; otherwise
    fall back to the named preset, then to empty."""
    if args.config_path and os.path.exists(args.config_path):
        import yaml
        with open(args.config_path, "r") as f:
            print(f"[Config] Loaded from YAML: {args.config_path}")
            return yaml.safe_load(f)
    if args.context_config in DilatedConfigs:
        print(f"[Config] Loaded preset: {args.context_config}")
        return DilatedConfigs[args.context_config].copy()
    print("[Config] WARNING: using empty context config")
    return {}


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # Build the model with the (possibly relpos) config, then load the checkpoint.
    config = _load_config(args)
    model = MultiScaleMultiModalModel(
        img_enc_name=args.image_encoder_type,
        gene_enc_name=args.gene_encoder_type,
        proj_dim=args.proj_dim,
        img_args={"backbone_type": args.image_backbone, "embed_dim": args.embed_dim, "config_dict": config},
        gene_args={"gene_dim": 2000, "embed_dim": args.embed_dim, "config_dict": config},
    )
    checkpoint = torch.load(args.model_path, map_location="cpu")
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict({k.replace("module.", ""): v for k, v in state_dict.items()})
    model.to(device)

    # Dataset: k-NN neighbors are injected above; local_distance is ignored by the
    # HD builder but passed through for signature parity.
    dataset = MultiScaleContextDataset(
        metadata_csv=args.metadata_csv, tissue_positions_csv=args.tissue_positions_csv,
        root_dir=args.data_root, num_local=args.num_local, num_global=args.num_global,
        local_distance=args.local_distance,
        transform_image=T.Compose([T.Resize((224, 224)), T.ToTensor(),
                                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    data = extract_embeddings_with_alpha(model, dataloader, device)

    # Cluster each modality at every k and save (reuses the shared helpers).
    suffix = args.output_suffix
    for k in range(args.k_min, args.k_max + 1):
        print(f"\n--- Processing k={k} ---")
        k_dir = os.path.join(args.output_dir, f"k{k}")
        os.makedirs(k_dir, exist_ok=True)
        for mode in ["image", "gene", "fused"]:
            labels = do_pca_kmeans(data[mode], k, args.pca_dim)
            pd.DataFrame({"spot_id": data["spot_ids"], "cluster": labels}).to_csv(
                os.path.join(k_dir, f"{mode}_clusters.csv"), index=False)

    # Alpha weights + raw embeddings.
    pd.DataFrame({
        "spot_id": data["spot_ids"],
        "alpha_img": data["alpha_img"].flatten(),
        "alpha_gene": data["alpha_gene"].flatten(),
    }).to_csv(os.path.join(args.output_dir, f"alpha_weights_{suffix}.csv"), index=False)

    for mode in ["image", "gene", "fused"]:
        emb_df = pd.DataFrame(data[mode])
        emb_df.insert(0, "spot_id", data["spot_ids"])
        emb_df.to_csv(os.path.join(args.output_dir, f"{mode}_embeddings_{suffix}.csv"), index=False)

    print(f"\nHD evaluation complete. Results in {args.output_dir}")


if __name__ == "__main__":
    main()
