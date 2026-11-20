"""Bloc 2 : prétraitement (doublons, manquants, aberrations, normalisation, genres)."""
from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import StandardScaler

from . import config


def load_raw(path=config.RAW_CSV) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Le CSV Kaggle contient une colonne d'index « Unnamed: 0 »
    return df.loc[:, ~df.columns.str.startswith("Unnamed")]


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Nettoie le catalogue et renvoie (données propres, rapport chiffré)."""
    rapport = {"lignes_brutes": len(df)}

    rapport["manquants_par_colonne"] = df[config.META + config.FEATURES].isna().sum()[lambda s: s > 0].to_dict()
    df = df.dropna(subset=["track_id", "track_name"] + config.FEATURES)
    rapport["apres_manquants"] = len(df)

    # Un titre peut figurer plusieurs fois dans le MÊME genre : vrais doublons
    avant = len(df)
    df = df.drop_duplicates(subset=["track_id", "track_genre"])
    rapport["doublons_meme_genre"] = avant - len(df)

    masque_duree = df["duration_ms"].between(config.MIN_DURATION_MS, config.MAX_DURATION_MS)
    masque_tempo = df["tempo"] > 0
    rapport["duree_aberrante"] = int((~masque_duree).sum())
    rapport["tempo_nul"] = int((~masque_tempo).sum())
    df = df[masque_duree & masque_tempo]

    rapport["lignes_finales"] = len(df)
    return df.reset_index(drop=True), rapport


def select_genres(df: pd.DataFrame, genres=config.GENRES) -> pd.DataFrame:
    """Sous-ensemble pour la partie supervisée.

    Un même `track_id` présent dans plusieurs des genres retenus a une étiquette
    ambiguë : on le retire pour ne pas fausser l'ADL.
    """
    sub = df[df["track_genre"].isin(genres)]
    ambigus = sub.groupby("track_id")["track_genre"].nunique()
    ambigus = ambigus[ambigus > 1].index
    return sub[~sub["track_id"].isin(ambigus)].reset_index(drop=True)


def standardize(df: pd.DataFrame, features=config.FEATURES) -> tuple[pd.DataFrame, StandardScaler]:
    """Z-score de chaque caractéristique (moyenne 0, écart-type 1)."""
    scaler = StandardScaler()
    z = pd.DataFrame(scaler.fit_transform(df[features]), columns=[f"{f}_z" for f in features], index=df.index)
    return z, scaler
