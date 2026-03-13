# visium_preprocess_export.py
# Scanpy-only processing + robust reader for: Data/*.h5 and Data/spatial/

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc

# -----------------------------
# Gene sets (edit per cancer)
# -----------------------------
GENESETS = {
    "immune_score": ["PTPRC", "CD3D", "CD3E", "TRAC", "MS4A1", "LYZ", "NKG7", "FCGR3A"],
    "proliferation_index": ["MKI67", "TOP2A", "PCNA", "MCM2", "MCM3", "TYMS", "HMGB2"],
    # baseline "tumor-ish" epithelial markers (replace as needed)
    "tumor_marker_score": ["EPCAM", "KRT8", "KRT18", "KRT19", "MSLN"],
    # baseline "normal/stromal-ish" markers (replace as needed)
    "normal_marker_score": ["COL1A1", "COL1A2", "DCN", "LUM"],
}

CELLTYPE_MARKERS = {
    "T_cells": ["CD3D", "CD3E", "TRAC"],
    "B_cells": ["MS4A1", "CD79A"],
    "Myeloid": ["LYZ", "S100A8", "S100A9", "FCGR3A"],
    "Epithelial/Tumor": ["EPCAM", "KRT8", "KRT18"],
    "Stroma": ["COL1A1", "COL1A2", "DCN", "LUM"],
}

# -----------------------------
# Robust Visium loader for:
#   Data/*.h5
#   Data/spatial/
# -----------------------------
def read_visium_from_h5_and_spatial(data_dir: str | Path) -> sc.AnnData:
    """
    Loads counts from Data/*.h5 (prefers *filtered_feature_bc_matrix.h5)
    and attaches spatial coords + images from Data/spatial/ via sc.read_visium().

    Result: AnnData with X (expression) + obsm['spatial'] + uns['spatial'].
    """
    data_dir = Path(data_dir)

    if not (data_dir / "spatial").exists():
        raise FileNotFoundError(f"Expected spatial/ folder at: {data_dir / 'spatial'}")

    # Prefer the filtered matrix h5
    h5_files = sorted(data_dir.glob("*filtered_feature_bc_matrix.h5"))
    if not h5_files:
        h5_files = sorted(data_dir.glob("*.h5"))
    if not h5_files:
        raise FileNotFoundError(f"No .h5 files found in {data_dir}")

    h5_path = h5_files[0]
    print("Using H5:", h5_path.name)

    # 1) Load counts
    adata = sc.read_10x_h5(h5_path)
    adata.var_names_make_unique()

    # 2) Load spatial metadata (coords + images)
    # NOTE: scanpy warns this will be replaced by squidpy in future; OK for now.
    vis_spatial = sc.read_visium(str(data_dir), load_images=True)
    vis_spatial.var_names_make_unique()

    # 3) Align spots and transfer spatial fields
    common = adata.obs_names.intersection(vis_spatial.obs_names)
    if len(common) == 0:
        raise ValueError(
            "No overlapping barcodes between H5 matrix and spatial metadata. "
            "Check that the .h5 and spatial/ belong to the same sample."
        )

    adata = adata[common].copy()
    vis_spatial = vis_spatial[common].copy()

    adata.obsm["spatial"] = vis_spatial.obsm["spatial"]
    adata.uns["spatial"] = vis_spatial.uns.get("spatial", {})

    return adata

# -----------------------------
# Scoring helpers
# -----------------------------
def score_genes_safe(adata: sc.AnnData, genes: list[str], score_name: str) -> None:
    genes_present = [g for g in genes if g in adata.var_names]
    if len(genes_present) < 3:
        adata.obs[score_name] = 0.0
        return
    sc.tl.score_genes(adata, gene_list=genes_present, score_name=score_name, use_raw=False)

def compute_cell_types_marker_voting(adata: sc.AnnData) -> None:
    scores = {}
    for ct, genes in CELLTYPE_MARKERS.items():
        tmp = f"_ct_{ct}"
        score_genes_safe(adata, genes, tmp)
        scores[ct] = adata.obs[tmp].to_numpy()

    mat = np.vstack([scores[k] for k in scores.keys()]).T  # (n_spots, n_types)
    labels = np.array(list(scores.keys()))[np.argmax(mat, axis=1)]
    adata.obs["cell_type"] = labels

def compute_tumor_probability_baseline(adata: sc.AnnData) -> None:
    """
    Simple baseline: combine tumor vs normal marker scores and squash to [0,1].
    Students can later swap this with a trained classifier.
    """
    t = adata.obs["tumor_marker_score"].to_numpy().astype(float)
    n = adata.obs["normal_marker_score"].to_numpy().astype(float)

    def robust_z(x: np.ndarray) -> np.ndarray:
        med = np.median(x)
        mad = np.median(np.abs(x - med)) + 1e-8
        return (x - med) / (1.4826 * mad)

    z = robust_z(t) - robust_z(n)
    p = 1.0 / (1.0 + np.exp(-z))
    adata.obs["tumor_probability"] = p

    # uncertainty baseline (max at 0.5)
    adata.obs["uncertainty"] = p * (1.0 - p)

# -----------------------------
# Export for next-step app
# -----------------------------
def export_inputs(
    adata: sc.AnnData,
    data_dir: Path,
    out_dir: Path,
    top_genes: int = 2000
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # coordinates (pixel space)
    if "spatial" not in adata.obsm:
        raise ValueError("Missing adata.obsm['spatial']; spatial metadata not attached.")

    xy = adata.obsm["spatial"]
    spots = pd.DataFrame({
        "spot_id": adata.obs_names,
        "x": xy[:, 0],
        "y": xy[:, 1],
    }).set_index("spot_id")

    for col in [
        "tumor_probability",
        "immune_score",
        "proliferation_index",
        "cell_type",
        "uncertainty",
        "tumor_marker_score",
        "normal_marker_score",
    ]:
        if col in adata.obs:
            spots[col] = adata.obs[col].to_numpy()

    spots.reset_index().to_parquet(out_dir / "spots.parquet", index=False)

    # Optional: reduced expression matrix for downstream stats / markers (HVGs)
    ad = adata.copy()
    sc.pp.highly_variable_genes(ad, n_top_genes=top_genes, flavor="seurat_v3")
    hv = ad.var["highly_variable"].to_numpy()
    ad = ad[:, hv].copy()

    X = ad.X
    if not isinstance(X, np.ndarray):
        X = X.toarray()

    expr = pd.DataFrame(X, index=ad.obs_names, columns=ad.var_names)
    expr.reset_index(names="spot_id").to_parquet(out_dir / "expr_top.parquet", index=False)

    # Copy spatial image + scalefactors (for overlay)
    spatial_dir = data_dir / "spatial"
    for fname in [
        "tissue_hires_image.png",
        "tissue_lowres_image.png",
        "scalefactors_json.json",
        "tissue_positions_list.csv",
        "tissue_positions.csv",
    ]:
        src = spatial_dir / fname
        if src.exists():
            (out_dir / fname).write_bytes(src.read_bytes())

    # Save gene sets used
    (out_dir / "gene_sets.json").write_text(json.dumps(GENESETS, indent=2))

    # Optional: save h5ad for reproducibility
    adata.write_h5ad(out_dir / "adata_processed.h5ad")

# -----------------------------
# Main pipeline
# -----------------------------
def main(data_dir: str, out_dir: str) -> None:
    data_dir_p = Path(data_dir)
    out_dir_p = Path(out_dir)

    # 1) Read counts + spatial metadata (your Data/*.h5 + Data/spatial/)
    adata = read_visium_from_h5_and_spatial(data_dir_p)

    # 2) QC light
    sc.pp.filter_cells(adata, min_counts=500)
    sc.pp.filter_genes(adata, min_cells=3)

    # 3) Normalize + log
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # 4) Scores
    for name, genes in GENESETS.items():
        score_genes_safe(adata, genes, name)

    compute_tumor_probability_baseline(adata)
    compute_cell_types_marker_voting(adata)

    # 5) Export
    export_inputs(adata, data_dir_p, out_dir_p)

    print("Done. Exported to:", out_dir_p.resolve())

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="Folder like 'Data/' containing *.h5 and spatial/")
    ap.add_argument("--out_dir", required=True, help="Output directory for next-step inputs")
    ap.add_argument("--top_genes", type=int, default=2000, help="Number of HVGs to export in expr_top.parquet")
    args = ap.parse_args()

    # pass top_genes through export by temporarily overriding default if desired
    # simplest: call main then re-export with custom top_genes if needed
    # (keeping CLI simple for students)
    main(args.data_dir, args.out_dir)
