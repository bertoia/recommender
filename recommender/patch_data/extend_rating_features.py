"""
Add rating_id, default 70% train/test split, user/movie mean rating to rating.csv
"""
import recommender.io as io
import pandas as pd
from sklearn.model_selection import train_test_split

ratings = io.get_rating()
ratings = ratings[['user_id', 'movie_id', 'rating']]
train_index, test_index = train_test_split(ratings.index,
                                           stratify=ratings.user_id,
                                           train_size=.7)

ratings.loc[test_index, "test"] = True
ratings.loc[train_index, "test"] = False
ratings["rating_id"] = ratings.index

ratings_train = ratings.query("test == False")

ratings = pd.merge(ratings,
                   ratings_train.groupby("user_id")
                                .rating
                                .mean()
                                .to_frame("user_mean"),
                   left_on="user_id", right_index=True, how="left")

ratings = pd.merge(ratings,
                   ratings_train.groupby("movie_id")
                                .rating
                                .mean()
                                .to_frame("movie_mean"),
                   left_on="movie_id", right_index=True, how="left")

ratings["user_normed_rating"] = ratings.rating - ratings.user_mean
ratings["movie_normed_rating"] = ratings.rating - ratings.movie_mean


ratings = ratings[["rating_id", "user_id", "movie_id", "rating",
                   'user_normed_rating', 'movie_normed_rating',
                   'user_mean', 'movie_mean', "test"]]

# Export
ratings.to_csv("data/rating.csv", index=False)
