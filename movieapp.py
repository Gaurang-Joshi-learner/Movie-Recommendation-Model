import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from difflib import get_close_matches

def load_data():
    ratings = pd.read_csv("ratings.csv")
    movies = pd.read_csv("movienames.csv")
    return ratings, movies

ratings, movies = load_data()


def compute_similarity():
    keta = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)
    movie_matrix = keta.T
    similarity_df = cosine_similarity(movie_matrix)
    movie_ids = movie_matrix.index
    cosine_sim = pd.DataFrame(similarity_df, index=movie_ids, columns=movie_ids)
    return cosine_sim

cosine_sim = compute_similarity()

movie_list = movies[['movieId', 'title']].drop_duplicates().reset_index(drop=True)
movie_name_to_id = dict(zip(movie_list['title'], movie_list['movieId']))
movie_id_to_name = dict(zip(movie_list['movieId'], movie_list['title']))

def recommend(movie_id, top_n=5, genre_filter=None):
    if movie_id not in cosine_sim.columns:
        return pd.DataFrame()
    sim = cosine_sim[movie_id].sort_values(ascending=False)
    top_ids = sim.index[1:]  # skip itself
    top_movies = movies[movies['movieId'].isin(top_ids)][['movieId', 'title', 'genres']].drop_duplicates()
    top_movies['similarity'] = [sim[mid] for mid in top_movies['movieId']]
    if genre_filter:
        top_movies = top_movies[top_movies['genres'].str.contains(genre_filter, case=False, na=False)]
    return top_movies.head(top_n)

# 🌟 Streamlit UI
st.set_page_config(page_title="🎬 Movie Recommender", layout="wide")
st.title("🎬 Movie Recommendation System")
st.markdown("Get movies similar to your favorite one!")

st.sidebar.header("Settings")
num_recs = st.sidebar.slider("Number of Recommendations", 1, 10, 5)

genre_options = sorted(set(g for sublist in movies['genres'].dropna().str.split('|') for g in sublist))
selected_genre = st.sidebar.selectbox("Optional: Filter by Genre", ["All"] + genre_options)

st.markdown("### 🎥 Search for a Movie")

search_movie = st.text_input("🔍 Type a Movie Name:")

if st.button("🎬 Recommend") and search_movie:
    matches = get_close_matches(search_movie, movie_list['title'], n=1, cutoff=0.3)
    if matches:
        selected_movie = matches[0]
        mid = movie_name_to_id[selected_movie]
        genre_filter = None if selected_genre == "All" else selected_genre
        recs = recommend(mid, num_recs, genre_filter)

        st.subheader(f"🎥 Recommendations similar to **{selected_movie}**:")
        if recs.empty:
            st.warning("No recommendations found for the selected criteria.")
        else:
            for _, row in recs.iterrows():
                st.markdown(f"**{row['title']}**")
                st.progress(row['similarity'])
                st.caption(f"Genres: {row['genres']}")
    else:
        st.error("❌ No close matches found. Please refine your search.")
