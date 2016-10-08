from recommender.validation import per_user_prediction
import recommender.io as io
import GPy
import os
import pandas as pd

# -------- #
# Settings #
# -------- #
# per user model settings
MOVIE_FEATURES = ['year', 'rtAllCriticsRating', 'rtAudienceRating',
                  'rtAllCriticsNumReviews', 'rtAudienceNumRatings',
                  'rtAllCriticsScore', 'rtAudienceScore',
                  '0', '1', '2', '3', '4', '5', '6', '7']

KERNEL = GPy.kern.RBF(input_dim=len(MOVIE_FEATURES),
                      variance=1., ARD=True)

RESPONSE = "user_normed_rating"
RESPONSE_MEAN = "user_mean"

# export settings
EXPORT = True
EXPORT_PATH = r"data\predictions"
EXPORT_FILE_NAME = "per_user_word2vec_movie_rbf_pred.csv"

# ------- #
# Dataset #
# ------- #
ratings = io.load_rating_movie("easy")

# import movie_vector and filter movies with no vector
movie_vector = pd.read_csv(r"word2vec\movie_vectors.csv")
movie_vector.rename(columns={"id": "movie_id"}, inplace=True)
movie_vector = movie_vector[["movie_id", '0', '1', '2', '3', '4', '5', '6', '7']]

ratings = pd.merge(ratings, movie_vector, how="left", on="movie_id")
ratings = ratings.fillna(0)

# ------ #
# result #
# ------ #
print(EXPORT_FILE_NAME)
result_per_user = per_user_prediction(ratings, MOVIE_FEATURES, KERNEL,
                                      RESPONSE, RESPONSE_MEAN)

# ------ #
# Export #
# ------ #
if EXPORT:
    result_per_user.to_csv(os.path.join(EXPORT_PATH, EXPORT_FILE_NAME),
                           index=False)

print(EXPORT_FILE_NAME + " completed!")
