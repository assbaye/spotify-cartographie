"""Bloc 3 : covariance, corrélations, variables redondantes."""
from __future__ import annotations

import pandas as pd

from . import config


def correlation_matrix(df: pd.DataFrame, features=config.FEATURES) -> pd.DataFrame:
    return df[features].corr()


def covariance_matrix(df: pd.DataFrame, features=config.FEATURES) -> pd.DataFrame:
    """Covariance (division par n - 1). Dépend des unités : voir la corrélation."""
    return df[features].cov()


def redundant_pairs(corr: pd.DataFrame, seuil: float = 0.6) -> pd.DataFrame:
    """Paires de variables dont |r| dépasse le seuil (ex. energy et loudness)."""
    lignes = []
    cols = list(corr.columns)
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            r = corr.loc[a, b]
            if abs(r) >= seuil:
                lignes.append({"variable_1": a, "variable_2": b, "r": round(float(r), 3), "r2": round(float(r) ** 2, 3)})
    return pd.DataFrame(lignes).sort_values("r", key=abs, ascending=False, ignore_index=True) if lignes else pd.DataFrame(
        columns=["variable_1", "variable_2", "r", "r2"]
    )
