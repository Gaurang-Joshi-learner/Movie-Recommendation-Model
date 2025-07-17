import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


ratings = pd.read_csv("ratings.csv")
movies = pd.read_csv("movienames.csv")

keta = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)
movie_matrix = keta.T


similarity_df = cosine_similarity(movie_matrix)
movie_ids = movie_matrix.index
cosine_sim = pd.DataFrame(similarity_df, index=movie_ids, columns=movie_ids)

# Recommend function
def recommend(movie_id, top_n=5, genre_filter=None):
    if movie_id not in cosine_sim.columns:
        return pd.DataFrame()
    sim = cosine_sim[movie_id].sort_values(ascending=False)
    top_ids = sim.index[1:]  # skip itself
    top_movies = movies[movies['movieId'].isin(top_ids)][['movieId', 'title', 'genres']]
    top_movies['similarity'] = [sim[mid] for mid in top_movies['movieId']]
    if genre_filter:
        top_movies = top_movies[top_movies['genres'].str.contains(genre_filter, case=False)]
    return top_movies.head(top_n)

# Streamlit UI
st.set_page_config(page_title="🎬 Movie Recommender", layout="wide")
st.title("🎬 Movie Recommendation System")
st.markdown("Get movies similar to your favorite one!")


st.sidebar.header("Settings")
num_recs = st.sidebar.slider("Number of Recommendations", 1, 10, 5)
genre_options = sorted(set(g for sublist in movies['genres'].str.split('|') for g in sublist))
selected_genre = st.sidebar.selectbox("Optional: Filter by Genre", ["All"] + genre_options)


movie_list = movies[['movieId', 'title']].drop_duplicates().reset_index(drop=True)
movie_name_to_id = dict(zip(movie_list['title'], movie_list['movieId']))

selected_movie = st.selectbox("Choose a Movie:", movie_list['title'])

if st.button("Recommend"):
    mid = movie_name_to_id[selected_movie]
    genre_filter = None if selected_genre == "All" else selected_genre
    recs = recommend(mid, num_recs, genre_filter)
    if recs.empty:
        st.warning("No recommendations found for the selected criteria.")
    else:
        st.subheader("🎥 Recommendations:")
        for _, row in recs.iterrows():
            st.markdown(f"**{row['title']}**")
            st.progress(row['similarity'])
            st.caption(f"Genres: {row['genres']}")

