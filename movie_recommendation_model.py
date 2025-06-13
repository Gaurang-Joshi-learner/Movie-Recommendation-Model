import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
df=pd.read_csv('movie.csv')
print(df.columns)
keta=df.pivot(index='userId',columns='movieId',values='rating').fillna(0)
movie_matrix=keta.T
similarity_df=cosine_similarity(movie_matrix)
movie_ids=movie_matrix.index
cosine_sim=pd.DataFrame(similarity_df,index=movie_ids,columns=movie_ids)

def movie_similarity(movie_id,top_n=3):
    if movie_id not in cosine_sim.columns:
        return f"movie id{movie_id} not found"
    sim=cosine_sim[movie_id].sort_values(ascending=False)
    return sim[1:top_n+1]
print(movie_similarity(101))