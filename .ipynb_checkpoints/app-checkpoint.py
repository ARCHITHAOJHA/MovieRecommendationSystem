import streamlit as st
import pickle
import pandas as pd
import requests
from dotenv import load_dotenv
import os

load_dotenv

API_KEY = os.getenv("OMDB_API_KEY")
st.set_page_config(layout="wide")

# 🎨 UI Styling
st.markdown("""
<style>
body { background-color: #141414; color: white; }
h1 { color: #E50914; text-align: center; }

.poster img {
    border-radius: 10px;
    transition: transform 0.3s;
}
.poster img:hover {
    transform: scale(1.1);
}

.movie-title {
    text-align: center;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)

# 🎬 Title
st.markdown("<h1>🎬 Movie Recommendation System</h1>", unsafe_allow_html=True)

# ✅ Load Data
@st.cache_data
def load_data():
    movies_dict = pickle.load(open('movie_dict.pkl', 'rb'))
    movies = pd.DataFrame(movies_dict)
    similarity = pickle.load(open('similarity.pkl', 'rb'))
    return movies, similarity

movies, similarity = load_data()

# 🎬 Fetch Poster (FIXED)
@st.cache_data
def fetch_poster(title):
    try:
        url = f"http://www.omdbapi.com/?t={title}&apikey={API_KEY}"
        data = requests.get(url, timeout=5).json()

        if data.get("Response") == "True":
            poster = data.get("Poster")
            if poster and poster != "N/A":
                return poster

    except Exception as e:
        print("Error:", e)

    return "https://via.placeholder.com/300x450?text=No+Poster"

# 📺 Fetch Details
@st.cache_data
def fetch_details(title):
    try:
        url = f"http://www.omdbapi.com/?t={title}&apikey={API_KEY}"
        return requests.get(url, timeout=5).json()
    except:
        return {}

# 🎯 Recommendation Logic (FIXED TITLE)
def recommend(movie):
    idx = movies[movies['title'] == movie].index[0]
    distances = similarity[idx]

    movie_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    names, posters = [], []

    for i in movie_list:
        raw_title = movies.iloc[i[0]].title

        # 🔥 FIX: clean title (remove year like (2010))
        clean_title = raw_title.split('(')[0].strip()

        names.append(raw_title)
        posters.append(fetch_poster(clean_title))

    return names, posters

# 🎯 ONLY SELECT BOX
selected_movie = st.selectbox(
    "Select a movie",
    movies['title'].values
)

# 🚀 Recommend Button
if st.button("Recommend"):
    names, posters = recommend(selected_movie)

    st.subheader("Recommended for you 🍿")

    cols = st.columns(5)

    for i in range(5):
        with cols[i]:
            st.image(posters[i], width="stretch")
            st.write(names[i])

            # 📺 Fetch details using cleaned title
            clean_title = names[i].split('(')[0].strip()
            details = fetch_details(clean_title)

            # ⭐ Rating
            rating = details.get("imdbRating")
            st.write(f"⭐ {rating if rating != 'N/A' else 'Not available'}")