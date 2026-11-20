"""Dashboard de démonstration (bloc 7).

Lancer :  streamlit run app/dashboard.py
Prérequis : avoir produit data/processed/tracks_clean.csv avec scripts/run_pipeline.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import plotly.express as px
import streamlit as st

from spotify_eda import config
from spotify_eda.clustering import describe_cluster
from spotify_eda.recommend import nearest_tracks, playlist_from_cluster

st.set_page_config(page_title="Cartographie musicale Spotify", layout="wide")
st.title("Cartographie musicale : profils sonores et playlists")

if not config.PROCESSED_CSV.exists():
    st.error("Fichier introuvable : lancez d'abord `python scripts/run_pipeline.py` (voir le README).")
    st.stop()


@st.cache_data
def load(path):
    return pd.read_csv(path)


df = load(config.PROCESSED_CSV)
zcols = [f"{f}_z" for f in config.FEATURES]
z = df[zcols]
SHOW = ["track_name", "artists", "track_genre", "cluster", "popularity"]

tab_map, tab_nn, tab_playlist = st.tabs(["Carte des titres", "Titres proches", "Playlist par ambiance"])

with tab_map:
    c1, c2, c3, c4 = st.columns(4)
    colour = c1.radio("Couleur", ["track_genre", "cluster"], format_func=lambda s: "Genre" if s == "track_genre" else "Cluster")
    axes = [c for c in ["PC1", "PC2", "PC3"] if c in df.columns]
    x_axis = c2.selectbox("Axe horizontal", axes, index=0)
    y_axis = c3.selectbox("Axe vertical", axes, index=1)
    n_points = c4.slider("Points affichés", 1000, len(df), min(8000, len(df)), step=500)
    sample = df.sample(n=min(n_points, len(df)), random_state=config.RANDOM_STATE).copy()
    sample["cluster"] = sample["cluster"].astype(str)
    fig = px.scatter(
        sample, x=x_axis, y=y_axis, color=colour if colour == "track_genre" else "cluster",
        hover_data=SHOW, opacity=0.6, height=650,
    )
    st.plotly_chart(fig, use_container_width=True)

with tab_nn:
    requete = st.text_input("Rechercher un titre (nom ou artiste)")
    if requete:
        mask = df["track_name"].str.contains(requete, case=False, na=False) | df["artists"].str.contains(requete, case=False, na=False)
        trouves = df[mask].head(50)
        if trouves.empty:
            st.info("Aucun titre trouvé.")
        else:
            choix = st.selectbox(
                "Titre", trouves.index, format_func=lambda i: f"{df.loc[i, 'track_name']} : {df.loc[i, 'artists']} ({df.loc[i, 'track_genre']})"
            )
            n = st.slider("Nombre de voisins", 5, 30, 10)
            voisins = nearest_tracks(z, df.index.get_loc(choix), n)
            st.write("Titre choisi")
            st.dataframe(df.loc[[choix], SHOW])
            st.write("Titres les plus proches dans l'espace des caractéristiques sonores")
            st.dataframe(df.iloc[voisins][SHOW])

with tab_playlist:
    profils = df.groupby("cluster")[zcols].mean()
    cluster = st.selectbox("Profil d'ambiance (cluster)", sorted(profils.index))
    st.caption("Profil sonore : " + describe_cluster(profils.loc[cluster]))
    c1, c2 = st.columns(2)
    n = c1.slider("Longueur de la playlist", 5, 50, 20)
    pop = c2.slider("Popularité minimale", 0, 100, 0)
    positions = playlist_from_cluster(z, df["cluster"], cluster, n=n, popularity=df["popularity"], min_popularity=pop)
    st.dataframe(df.iloc[positions][SHOW])
    bar = px.bar(profils.loc[cluster].rename(lambda s: s.removesuffix("_z")), labels={"index": "", "value": "écart à la moyenne (z)"})
    st.plotly_chart(bar, use_container_width=True)
