"""Faux catalogue au format du jeu Kaggle, pour tester le code sans le vrai CSV."""
import numpy as np
import pandas as pd

from spotify_eda import config


def make_catalog(n_per_genre: int = 150, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    centres = {g: rng.normal(0, 1.2, size=len(config.FEATURES)) for g in config.GENRES}
    lignes = []
    for g, c in centres.items():
        x = c + rng.normal(0, 1, size=(n_per_genre, len(config.FEATURES)))
        d = pd.DataFrame(x, columns=config.FEATURES)
        d["track_genre"] = g
        lignes.append(d)
    df = pd.concat(lignes, ignore_index=True)
    # Remise à des échelles réalistes
    for c in ["danceability", "energy", "speechiness", "acousticness", "instrumentalness", "liveness", "valence"]:
        df[c] = 1 / (1 + np.exp(-df[c]))
    df["key"] = rng.integers(0, 12, len(df))
    df["loudness"] = -10 + 3 * df["loudness"]
    df["tempo"] = (120 + 25 * df["tempo"]).clip(lower=1)
    df["track_id"] = [f"id{i}" for i in range(len(df))]
    df["track_name"] = [f"Titre {i}" for i in range(len(df))]
    df["artists"] = "Artiste"
    df["album_name"] = "Album"
    df["popularity"] = rng.integers(0, 100, len(df))
    df["duration_ms"] = rng.integers(120_000, 360_000, len(df))
    # Quelques défauts à nettoyer
    df.loc[0, "tempo"] = 0
    df.loc[1, "duration_ms"] = 0
    df.loc[2, "energy"] = np.nan
    df = pd.concat([df, df.iloc[[5]]], ignore_index=True)  # un doublon exact
    return df
