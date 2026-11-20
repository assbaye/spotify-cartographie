"""Tests de fumée : chaque bloc tourne sur le faux catalogue."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from synthetic import make_catalog
from spotify_eda import config
from spotify_eda.clustering import cluster_profiles, describe_cluster, elbow_silhouette, genre_composition, run_cah, run_kmeans
from spotify_eda.exploration import correlation_matrix, redundant_pairs
from spotify_eda.lda import cross_validated_confusion, most_confused, run_lda
from spotify_eda.pca import explained_table, run_pca, variable_axis_correlations
from spotify_eda.preprocessing import clean, select_genres, standardize
from spotify_eda.recommend import nearest_tracks, playlist_from_cluster


def prepare():
    df, rapport = clean(make_catalog())
    sub = select_genres(df)
    z, _ = standardize(sub)
    return df, rapport, sub, z


def test_clean_removes_defects():
    df, rapport, _, _ = prepare()
    assert rapport["tempo_nul"] == 1
    assert rapport["duree_aberrante"] == 1
    assert rapport["doublons_meme_genre"] == 1
    assert df["tempo"].min() > 0


def test_standardize_mean_zero_std_one():
    _, _, _, z = prepare()
    assert np.allclose(z.mean(), 0, atol=1e-9)
    assert np.allclose(z.std(ddof=0), 1, atol=1e-9)


def test_pca_trace_equals_number_of_variables():
    _, _, _, z = prepare()
    pca, coords = run_pca(z)
    assert abs(explained_table(pca)["valeur_propre"].sum() - len(config.FEATURES) * len(z) / (len(z) - 1)) < 1e-6
    r = variable_axis_correlations(pca, config.FEATURES)
    assert (r.pow(2).sum(axis=1).round(6) == 1).all()  # chaque variable est bien représentée par l'ensemble des axes


def test_lda_axes_at_most_k_minus_one():
    _, _, sub, z = prepare()
    lda, coords = run_lda(z, sub["track_genre"])
    assert coords.shape[1] <= len(config.GENRES) - 1
    cm, taux = cross_validated_confusion(z, sub["track_genre"])
    assert taux > 0.5 and len(most_confused(cm)) > 0


def test_clustering_and_recommendation():
    _, _, sub, z = prepare()
    table = elbow_silhouette(z, ks=range(2, 5), sample=500)
    assert list(table["k"]) == [2, 3, 4]
    km = run_kmeans(z, 4)
    profils = cluster_profiles(z, km.labels_)
    assert isinstance(describe_cluster(profils.iloc[0]), str)
    assert genre_composition(sub["track_genre"], km.labels_).shape[0] == 4
    link, idx, lab = run_cah(z, 4, sample=300)
    assert link.shape[0] == 299
    voisins = nearest_tracks(z, 0, n=5)
    assert len(voisins) == 5 and 0 not in voisins
    pl = playlist_from_cluster(z, km.labels_, 0, n=10)
    assert len(pl) == 10 and (km.labels_[pl] == 0).all()


def test_correlation_pairs():
    df, *_ = prepare()
    assert redundant_pairs(correlation_matrix(df), seuil=0.99).empty
