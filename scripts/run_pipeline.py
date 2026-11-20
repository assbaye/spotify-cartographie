"""Enchaîne prétraitement, ACP et K-means, puis écrit le fichier lu par le dashboard.

Usage :
    python scripts/run_pipeline.py --raw data/raw/dataset.csv --k 6
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from spotify_eda import config
from spotify_eda.clustering import describe_cluster, run_kmeans, cluster_profiles
from spotify_eda.pca import explained_table, run_pca
from spotify_eda.preprocessing import clean, load_raw, select_genres, standardize


def main(raw: Path, out: Path, k: int) -> pd.DataFrame:
    df, rapport = clean(load_raw(raw))
    print("Rapport de nettoyage :", rapport)

    sub = select_genres(df)
    print(f"{len(sub)} titres dans les {len(config.GENRES)} genres retenus")

    z, _ = standardize(sub)
    pca, coords = run_pca(z)
    print(explained_table(pca).round(3).to_string(index=False))

    km = run_kmeans(z, k)
    profils = cluster_profiles(z, km.labels_)
    for c, row in profils.iterrows():
        print(f"Cluster {c} : {describe_cluster(row)}")

    result = pd.concat([sub[config.META + config.FEATURES], z, coords.iloc[:, :3]], axis=1)
    result["cluster"] = km.labels_
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out, index=False)
    print("Écrit :", out)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--raw", type=Path, default=config.RAW_CSV)
    p.add_argument("--out", type=Path, default=config.PROCESSED_CSV)
    p.add_argument("--k", type=int, default=6, help="nombre de clusters (à justifier par coude et silhouette)")
    a = p.parse_args()
    main(a.raw, a.out, a.k)
