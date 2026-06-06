import os
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

class MovieRecommenderEngine:
    def __init__(self, ratings_path="data/ratings.csv", movies_path="data/movies.csv"):
        if not os.path.exists(ratings_path) and os.path.exists("ratings.csv"):
            ratings_path = "ratings.csv"
        if not os.path.exists(movies_path) and os.path.exists("movies.csv"):
            movies_path = "movies.csv"
            
        print(f"Loading ratings from: {ratings_path}")
        print(f"Loading movies from: {movies_path}")

        self.ratings = pd.read_csv(ratings_path)
        self.movies = pd.read_csv(movies_path)
        
        genre_map = {
            '0': 'Action', '1': 'Adventure', '2': 'Animation', "3": "Children's",
            '4': 'Comedy', '5': 'Crime', '6': 'Documentary', '7': 'Drama',
            '8': 'Fantasy', '9': 'Film-Noir', '10': 'Horror', '11': 'Musical',
            '12': 'Mystery', '13': 'Romance', '14': 'Sci-Fi', '15': 'Thriller',
            '16': 'War', '17': 'Western'
        }

        def convert_genres(genre_string):
            if pd.isna(genre_string) or not str(genre_string).strip():
                return "Unknown"
            parts = [str(g).strip() for g in str(genre_string).split(',')]
            mapped = [genre_map.get(p, p) for p in parts if p]
            return " | ".join(mapped) if mapped else "Unknown"

        self.movies['movie_genres'] = self.movies['movie_genres'].apply(convert_genres)

        ratings_sorted = self.ratings.sort_values("timestamp")
        split = int(len(ratings_sorted) * 0.8)
        self.train_df = ratings_sorted.iloc[:split].copy()
        
        self.train_matrix = self.train_df.pivot_table(
            index="user_id", columns="movie_id", values="user_rating"
        ).fillna(0)
        
        self.user_ids = sorted(self.train_matrix.index.tolist())
        
        self.user_means = self.train_matrix.replace(0, np.nan).mean(axis=1)
        matrix_centered = self.train_matrix.subtract(self.user_means, axis=0)
        matrix_centered[self.train_matrix == 0] = 0
        
        sparse_matrix = csr_matrix(matrix_centered.values)
        k = min(50, sparse_matrix.shape[0] - 1, sparse_matrix.shape[1] - 1)
        U, sigma, Vt = svds(sparse_matrix, k=k)
        
        U = U[:, ::-1]
        sigma = sigma[::-1]
        Vt = Vt[::-1, :]
        
        sigma_diag = np.diag(sigma)
        predicted_centered = U @ sigma_diag @ Vt
        predicted_ratings = predicted_centered + self.user_means.values.reshape(-1, 1)
        
        self.predicted_df = pd.DataFrame(
            predicted_ratings, index=self.train_matrix.index, columns=self.train_matrix.columns
        )
        
        self.movie_titles = dict(zip(self.movies["movie_id"], self.movies["movie_title"]))
        self.movie_genres = dict(zip(self.movies["movie_id"], self.movies["movie_genres"]))
        self.movie_posters = dict(zip(self.movies["movie_id"], self.movies["poster_url"]))

        popular_movie_ids = self.ratings['movie_id'].value_counts().index.tolist()
        representative_ids = []
        for mid in popular_movie_ids:
            if mid in self.movie_genres and mid in self.movie_posters:
                poster = self.movie_posters[mid]
                if pd.notna(poster) and str(poster).strip().startswith("http"):
                    if len(representative_ids) < 30:
                        representative_ids.append(mid)

        self.representative_movies = []
        for mid in representative_ids[:30]:
            poster = self.movie_posters.get(mid, "")
            if pd.isna(poster) or not poster:
                poster = "https://images.unsplash.com/photo-1594909122845-11baa439b7bf?q=80&w=400"
            self.representative_movies.append({
                "movie_id": mid,
                "title": self.movie_titles.get(mid, f"Movie {mid}"),
                "genres": self.movie_genres.get(mid, "Unknown"),
                "poster": poster
            })

    def get_user_ids(self):
        return self.user_ids

    def get_representative_movies(self):
        return self.representative_movies

    def recommend(self, user_id, n=12):
        if user_id not in self.predicted_df.index:
            return []
        seen_movie_ids = set(self.train_df[self.train_df["user_id"] == user_id]["movie_id"])
        user_predictions = self.predicted_df.loc[user_id]
        user_predictions = user_predictions[~user_predictions.index.isin(seen_movie_ids)]
        top_movie_ids = user_predictions.nlargest(n).index.tolist()
        
        recommendations = []
        for mid in top_movie_ids:
            title = self.movie_titles.get(mid, f"Movie {mid}")
            genres = self.movie_genres.get(mid, "Unknown")
            poster = self.movie_posters.get(mid, "")
            if pd.isna(poster) or not poster:
                poster = "https://images.unsplash.com/photo-1594909122845-11baa439b7bf?q=80&w=400"
            recommendations.append({
                "movie_id": mid, "title": title, "genres": genres, "poster": poster
            })
        return recommendations

    def recommend_for_new_user(self, selected_titles, n=12):
        if not selected_titles:
            return []
        title_to_id = {v: k for k, v in self.movie_titles.items()}
        selected_ids = [title_to_id[t] for t in selected_titles if t in title_to_id]
        if not selected_ids:
            return []
            
        similar_users = self.ratings[
            self.ratings['movie_id'].isin(selected_ids) & (self.ratings['user_rating'] >= 4)
        ]['user_id'].unique()
        
        if len(similar_users) == 0:
            top_movie_ids = self.ratings['movie_id'].value_counts().nlargest(n).index.tolist()
        else:
            recs_df = self.ratings[
                self.ratings['user_id'].isin(similar_users) & 
                (~self.ratings['movie_id'].isin(selected_ids)) &
                (self.ratings['user_rating'] >= 4)
            ]
            top_movie_ids = recs_df['movie_id'].value_counts().nlargest(n).index.tolist()
            
        recommendations = []
        for mid in top_movie_ids[:n]:
            poster = self.movie_posters.get(mid, "")
            if pd.isna(poster) or not poster:
                poster = "https://images.unsplash.com/photo-1594909122845-11baa439b7bf?q=80&w=400"
            recommendations.append({
                "movie_id": mid, "title": self.movie_titles.get(mid, f"Movie {mid}"),
                "genres": self.movie_genres.get(mid, "Unknown"), "poster": poster
            })
        return recommendations

    def get_top_rated_by_user(self, user_id, n=5):
        user_history = self.ratings[self.ratings["user_id"] == user_id]
        if user_history.empty:
            return []
        
        top_history = user_history.sort_values(by=["user_rating", "timestamp"], ascending=[False, False]).head(n)
        top_rated = []
        for _, row in top_history.iterrows():
            mid = int(row["movie_id"])
            rating = row["user_rating"]
            title = self.movie_titles.get(mid, f"Movie {mid}")
            genres = self.movie_genres.get(mid, "Unknown")
            poster = self.movie_posters.get(mid, "")
            if pd.isna(poster) or not poster:
                poster = "https://images.unsplash.com/photo-1594909122845-11baa439b7bf?q=80&w=400"
                
            top_rated.append({
                "movie_id": mid, "title": title, "genres": genres, "poster": poster, "rating": rating
            })
        return top_rated