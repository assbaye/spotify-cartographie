# Cartographie musicale Spotify

Projet d'analyse exploratoire de données (M1 Informatique, sujet 4) : les caractéristiques sonores suffisent-elles à retrouver les genres et à former des profils d'ambiance ?

## Installation

```bash
python -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate
pip install -r requirements.txt
```

## Données

Télécharger `dataset.csv` depuis [Kaggle (114 000 Spotify Songs)](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset) et le placer dans `data/raw/dataset.csv`. Le fichier n'est pas versionné.

## Exécution

```bash
python scripts/run_pipeline.py --k 6     # nettoyage, ACP, K-means, écrit data/processed/tracks_clean.csv
streamlit run app/dashboard.py           # dashboard de démonstration
pytest                                   # tests sur un faux catalogue (sans le vrai CSV)
```

Le nombre de clusters `--k` doit être justifié par le coude et la silhouette (`elbow_silhouette` dans `src/spotify_eda/clustering.py`).

## Organisation

| Dossier | Contenu | Bloc |
| --- | --- | --- |
| `src/spotify_eda/preprocessing.py` | doublons, manquants, aberrations, normalisation, choix des genres | 2 |
| `src/spotify_eda/exploration.py` | covariance, corrélations, paires redondantes | 3 |
| `src/spotify_eda/pca.py` | ACP, valeurs propres, cercle des corrélations | 4 |
| `src/spotify_eda/lda.py` | ADL, matrice de confusion en validation croisée | 5 |
| `src/spotify_eda/clustering.py` | K-means, CAH, profils, composition par genre | 6 |
| `src/spotify_eda/recommend.py`, `app/dashboard.py` | voisins proches, playlist, dashboard | 7 |
| `notebooks/` | un notebook par bloc, pour les figures et les commentaires | 2 à 6 |
| `reports/` | rapport et figures | 8 |

## Règles de travail

- `main` reste propre : on y arrive par pull request relue par un autre membre.
- Une branche par bloc (`feat/pretraitement`, `feat/acp`, `feat/clustering`...).
- Graine aléatoire fixe : `random_state = 42` (`config.py`).
- Les genres retenus et les seuils de nettoyage se changent dans `src/spotify_eda/config.py`.

## À faire

- [x] Genres retenus vérifiés : les 8 de `config.GENRES` existent dans le CSV.
- [x] `notebooks/01_pretraitement.ipynb` (à compléter avec les commentaires de chacun).
- [ ] Un notebook par bloc suivant (le code des fonctions est déjà dans `src/`).
- [ ] `data/processed/tracks_clean.csv` est provisoire (k = 6) : à régénérer avec le k justifié.
- [ ] Rapport et réponses explicites aux 4 questions du sujet.
