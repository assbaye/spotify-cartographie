"""Bloc 4 : ACP sur les caractéristiques sonores normalisées."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from . import config


def run_pca(z: pd.DataFrame, n_components: int | None = None) -> tuple[PCA, pd.DataFrame]:
    """ACP normée : `z` doit déjà être centrée-réduite (voir preprocessing.standardize)."""
    pca = PCA(n_components=n_components, random_state=config.RANDOM_STATE)
    coords = pd.DataFrame(pca.fit_transform(z), columns=[f"PC{i + 1}" for i in range(pca.n_components_)], index=z.index)
    return pca, coords


def explained_table(pca: PCA) -> pd.DataFrame:
    """Valeurs propres, part de variance et cumul (critère : au moins 80 %)."""
    part = pca.explained_variance_ratio_
    return pd.DataFrame(
        {
            "axe": [f"PC{i + 1}" for i in range(len(part))],
            "valeur_propre": pca.explained_variance_,
            "part": part,
            "cumul": np.cumsum(part),
        }
    )


def variable_axis_correlations(pca: PCA, features) -> pd.DataFrame:
    """Corrélation entre chaque variable et chaque axe (coordonnées du cercle des corrélations).

    Pour une ACP normée : r(variable, axe) = vecteur propre x racine(valeur propre).
    """
    r = pca.components_.T * np.sqrt(pca.explained_variance_)
    # StandardScaler divise par n, sklearn par n - 1 : on rétablit des variances exactement égales à 1
    n = pca.n_samples_
    r = r / np.sqrt(n / (n - 1))
    return pd.DataFrame(r, index=list(features), columns=[f"PC{i + 1}" for i in range(r.shape[1])])
