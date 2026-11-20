"""Paramètres communs au projet : chemins, colonnes, genres retenus."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_CSV = ROOT / "data" / "raw" / "dataset.csv"
PROCESSED_CSV = ROOT / "data" / "processed" / "tracks_clean.csv"

RANDOM_STATE = 42

# Les 10 caractéristiques sonores citées dans le sujet (tonalité = key).
# À discuter dans le rapport : `key` est un entier 0-11 (variable circulaire),
# la garder ou non dans l'ACP est un choix à justifier.
FEATURES = [
    "danceability",
    "energy",
    "key",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
]

META = ["track_id", "artists", "album_name", "track_name", "popularity", "duration_ms", "track_genre"]

# 8 genres contrastés, tous présents dans le CSV (114 genres de 1 000 titres chacun).
# Le choix reste à justifier dans le rapport (voir notebook 01).
GENRES = [
    "classical",
    "heavy-metal",
    "hip-hop",
    "jazz",
    "reggae",
    "techno",
    "country",
    "sleep",
]

# Seuils de nettoyage (à justifier dans le rapport)
MIN_DURATION_MS = 30_000        # moins de 30 s : titre aberrant
MAX_DURATION_MS = 20 * 60_000   # plus de 20 min : titre aberrant
