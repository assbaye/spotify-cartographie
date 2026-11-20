"""Bloc 5 : analyse discriminante linéaire avec le genre comme classe."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict, StratifiedKFold

from . import config


def run_lda(z: pd.DataFrame, y: pd.Series):
    """Ajuste l'ADL et renvoie (modèle, coordonnées sur les axes discriminants).

    Avec k classes, au plus k - 1 axes discriminants (voir CM3).
    """
    lda = LinearDiscriminantAnalysis()
    coords = pd.DataFrame(
        lda.fit_transform(z, y),
        columns=[f"LD{i + 1}" for i in range(lda.transform(z.iloc[:2]).shape[1])],
        index=z.index,
    )
    return lda, coords


def cross_validated_confusion(z: pd.DataFrame, y: pd.Series, n_splits: int = 5):
    """Matrice de confusion en validation croisée : honnête, contrairement à celle sur l'apprentissage."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=config.RANDOM_STATE)
    pred = cross_val_predict(LinearDiscriminantAnalysis(), z, y, cv=cv)
    labels = sorted(y.unique())
    cm = pd.DataFrame(confusion_matrix(y, pred, labels=labels), index=labels, columns=labels)
    taux = float((pred == y).mean())
    return cm, taux


def most_confused(cm: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """Paires de genres les plus confondues (part de A prédits B, et inversement)."""
    part = cm.div(cm.sum(axis=1), axis=0)
    lignes = []
    for a in cm.index:
        for b in cm.columns:
            if a < b:
                lignes.append({"genre_1": a, "genre_2": b, "confusion": float(part.loc[a, b] + part.loc[b, a])})
    return pd.DataFrame(lignes).sort_values("confusion", ascending=False, ignore_index=True).head(n)
