"""Bloc 7 (cœur) : voisins proches et génération de playlist."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


def nearest_tracks(z: pd.DataFrame, position: int, n: int = 10) -> np.ndarray:
    """Positions des n titres les plus proches (distance euclidienne en z-score)."""
    nn = NearestNeighbors(n_neighbors=n + 1).fit(z.to_numpy())
    _, idx = nn.kneighbors(z.iloc[[position]].to_numpy())
    return idx[0][1:]  # le premier voisin est le titre lui-même


def playlist_from_cluster(
    z: pd.DataFrame, labels, cluster: int, n: int = 20, popularity: pd.Series | None = None, min_popularity: int = 0
) -> np.ndarray:
    """Positions des n titres les plus proches du centre du cluster (option : popularité minimale)."""
    labels = np.asarray(labels)
    membres = np.where(labels == cluster)[0]
    if popularity is not None:
        membres = membres[popularity.to_numpy()[membres] >= min_popularity]
    centre = z.to_numpy()[labels == cluster].mean(axis=0)
    dist = np.linalg.norm(z.to_numpy()[membres] - centre, axis=1)
    return membres[np.argsort(dist)[:n]]
