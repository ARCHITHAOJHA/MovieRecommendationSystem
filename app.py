import streamlit as st
import pickle
import pandas as pd
import requests
from dotenv import load_dotenv
import os
from urllib.parse import quote
import ast
from pathlib import Path

load_dotenv()

API_KEY = os.getenv("OMDB_API_KEY")
PLACEHOLDER_POSTER = "https://placehold.co/300x450?text=No+Poster"
WIKIPEDIA_HEADERS = {"User-Agent": "Mozilla/5.0"}
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

# 🔧 Generate pickle files from CSVs if they don't exist
def generate_data_files():
    if Path('movie_dict.pkl').exists() and Path('similarity.pkl').exists():
        return
    
    st.info("🔄 Generating data files... This may take a moment on first load.")
    
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Read CSVs
        credits = pd.read_csv('MovieRS/tmdb_5000_credits.csv')
        movies = pd.read_csv('MovieRS/tmdb_5000_movies.csv')
        
        # Merge datasets
        movies_data = movies.merge(credits, on='title')
        
        # Select and clean data
        movie_features = movies_data[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']].copy()
        movie_features.dropna(inplace=True)
        
        # Parse JSON-like strings
        def parse_json_column(obj):
            try:
                items = ast.literal_eval(obj)
                return ' '.join([item['name'] for item in items])
            except:
                return ''
        
        def parse_cast(obj):
            try:
                items = ast.literal_eval(obj)
                return ' '.join([item['name'] for item in items[:3]])  # Top 3 cast members
            except:
                return ''
        
        movie_features['genres'] = movie_features['genres'].apply(parse_json_column)
        movie_features['keywords'] = movie_features['keywords'].apply(parse_json_column)
        movie_features['cast'] = movie_features['cast'].apply(parse_cast)
        
        # Combine features for similarity calculation
        movie_features['tags'] = (
            movie_features['overview'] + ' ' + 
            movie_features['genres'] + ' ' + 
            movie_features['keywords'] + ' ' + 
            movie_features['cast']
        )
        
        # Calculate similarity matrix
        vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(movie_features['tags'])
        similarity = cosine_similarity(tfidf_matrix)
        
        # Save as pickle files
        movie_dict = movie_features[['movie_id', 'title']].reset_index(drop=True).to_dict()
        pickle.dump(movie_dict, open('movie_dict.pkl', 'wb'))
        pickle.dump(similarity, open('similarity.pkl', 'wb'))
        
        st.success("✅ Data files generated successfully!")
    except Exception as e:
        st.error(f"Error generating data files: {str(e)}")
        raise

# ✅ Load Data
@st.cache_data
def load_data():
    generate_data_files()
    movies_dict = pickle.load(open('movie_dict.pkl', 'rb'))
    movies = pd.DataFrame(movies_dict)
    similarity = pickle.load(open('similarity.pkl', 'rb'))
    return movies, similarity

movies, similarity = load_data()

# 🎬 Fetch Poster (FIXED)
def fetch_wikipedia_poster(title):
    def get_summary(page_title):
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{quote(page_title.replace(' ', '_'), safe='()_')}"
        response = requests.get(url, headers=WIKIPEDIA_HEADERS, timeout=5)

        if not response.ok:
            return None

        data = response.json()
        thumbnail = data.get("thumbnail", {}).get("source")
        description = (data.get("description") or "").lower()

        return {
            "thumbnail": thumbnail,
            "description": description,
        }

    exact_summary = get_summary(title)
    if exact_summary and exact_summary["thumbnail"]:
        if any(keyword in exact_summary["description"] for keyword in ["film", "movie", "television", "series"]):
            return exact_summary["thumbnail"]

    for search_term in (f"{title} film", f"{title} movie"):
        search_url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "list": "search",
            "srsearch": search_term,
            "format": "json",
            "srlimit": 5,
        }

        try:
            response = requests.get(search_url, params=params, headers=WIKIPEDIA_HEADERS, timeout=5)
            if not response.ok:
                continue

            results = response.json().get("query", {}).get("search", [])
            for item in results:
                candidate_title = item.get("title")
                if not candidate_title:
                    continue

                candidate_summary = get_summary(candidate_title)
                if candidate_summary and candidate_summary["thumbnail"]:
                    return candidate_summary["thumbnail"]
        except Exception:
            continue

    if exact_summary and exact_summary["thumbnail"]:
        return exact_summary["thumbnail"]

    return PLACEHOLDER_POSTER


@st.cache_data
def fetch_poster(title):
    if not API_KEY:
        return fetch_wikipedia_poster(title)

    try:
        url = f"https://www.omdbapi.com/?t={title}&apikey={API_KEY}"
        data = requests.get(url, timeout=5).json()

        if data.get("Response") == "True":
            poster = data.get("Poster")
            if poster and poster != "N/A":
                return poster

    except Exception as e:
        print("Error:", e)

    return fetch_wikipedia_poster(title)

# 📺 Fetch Details
@st.cache_data
def fetch_details(title):
    if not API_KEY:
        return {}

    try:
        url = f"https://www.omdbapi.com/?t={title}&apikey={API_KEY}"
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
            display_rating = rating if (rating and rating != 'N/A') else 'Ratings'
            st.write(f"⭐ {display_rating}")