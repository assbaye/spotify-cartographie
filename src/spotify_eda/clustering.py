"""Bloc 6 : K-means, CAH et profils d'ambiance."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from . import config


def elbow_silhouette(z: pd.DataFrame, ks=range(2, 11), sample: int = 10_000) -> pd.DataFrame:
    """Inertie J(k) (coude) et silhouette moyenne pour chaque k.

    La silhouette coûte O(n²) : on la calcule sur un échantillon.
    """
    rng = np.random.default_rng(config.RANDOM_STATE)
    idx = rng.choice(len(z), size=min(sample, len(z)), replace=False)
    lignes = []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=config.RANDOM_STATE).fit(z)
        sil = silhouette_score(z.iloc[idx], km.labels_[idx])
        lignes.append({"k": k, "inertie": km.inertia_, "silhouette": sil})
    return pd.DataFrame(lignes)


def run_kmeans(z: pd.DataFrame, k: int) -> KMeans:
    """K-means avec K-means++ et 10 lancements (limite l'effet des optima locaux)."""
    return KMeans(n_clusters=k, n_init=10, random_state=config.RANDOM_STATE).fit(z)


def run_cah(z: pd.DataFrame, n_clusters: int, sample: int = 2_000, method: str = "ward"):
    """CAH sur un échantillon. Renvoie (matrice de liage, indices, étiquettes)."""
    rng = np.random.default_rng(config.RANDOM_STATE)
    idx = rng.choice(len(z), size=min(sample, len(z)), replace=False)
    link = linkage(z.iloc[idx].to_numpy(), method=method)
    labels = fcluster(link, t=n_clusters, criterion="maxclust")
    return link, idx, labels


def cluster_profiles(z: pd.DataFrame, labels) -> pd.DataFrame:
    """Moyenne (en z-score) de chaque caractéristique par cluster."""
    return z.groupby(np.asarray(labels)).mean()


def describe_cluster(profile_row: pd.Series, seuil: float = 0.5) -> str:
    """Phrase en langage courant : caractéristiques nettement hautes ou basses."""
    haut = [c.removesuffix("_z") for c, v in profile_row.items() if v >= seuil]
    bas = [c.removesuffix("_z") for c, v in profile_row.items() if v <= -seuil]
    morceaux = []
    if haut:
        morceaux.append("élevé : " + ", ".join(haut))
    if bas:
        morceaux.append("faible : " + ", ".join(bas))
    return " ; ".join(morceaux) or "profil proche de la moyenne"


def genre_composition(genres: pd.Series, labels) -> pd.DataFrame:
    """Part de chaque genre dans chaque cluster (question 3 du sujet)."""
    return pd.crosstab(np.asarray(labels), genres.to_numpy(), normalize="index")
