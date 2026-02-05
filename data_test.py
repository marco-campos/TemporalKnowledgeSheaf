#!/usr/bin/env python3
"""
Quick sanity check for loading the TGB THG dataset (thgl-software).
This uses the numpy-based loader to avoid PyG dependencies.
"""

from __future__ import annotations

import sys


def main() -> int:
    try:
        from tgb.linkproppred.dataset import LinkPropPredDataset
    except Exception as exc:  # pragma: no cover - import guard
        print("Failed to import tgb. Install with: pip install py-tgb")
        print(f"Import error: {exc}")
        return 1

    name = "thgl-software"
    root = "datasets"

    print(f"Loading dataset: {name}")
    dataset = LinkPropPredDataset(name=name, root=root, preprocess=True)

    data = dataset.full_data
    print("Loaded keys:", sorted(list(data.keys())))
    print("Num nodes:", dataset.num_nodes)
    print("Num edges:", dataset.num_edges)
    print("Num relations:", getattr(dataset, "num_rels", None))

    # Edge arrays
    print("sources:", data["sources"].shape)
    print("destinations:", data["destinations"].shape)
    print("timestamps:", data["timestamps"].shape)
    print("edge_feat:", data["edge_feat"].shape)
    print("edge_label:", data["edge_label"].shape)
    print("edge_idxs:", data["edge_idxs"].shape)

    # THG-specific info (may be None if not provided)
    edge_type = getattr(dataset, "edge_type", None)
    node_type = getattr(dataset, "node_type", None)
    if edge_type is not None:
        print("edge_type:", edge_type.shape)
    if node_type is not None:
        print("node_type:", node_type.shape)

    # Splits
    print("train_mask:", dataset.train_mask.shape)
    print("val_mask:", dataset.val_mask.shape)
    print("test_mask:", dataset.test_mask.shape)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
