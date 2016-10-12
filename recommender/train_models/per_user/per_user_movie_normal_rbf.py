from recommender.validation import per_user_prediction
import recommender.io as io
import GPy
import os

# -------- #
# Settings #
# -------- #
# per user model settings
MOVIE_FEATURES = ['year', 'rtAllCriticsRating', 'rtAudienceRating',
                  'rtAllCriticsNumReviews', 'rtAudienceNumRatings',
                  'rtAllCriticsScore', 'rtAudienceScore','genre_Action',
                  'genre_Adventure', 'genre_Animation', "genre_Children's",
                  'genre_Comedy', 'genre_Crime', 'genre_Documentary',
                  'genre_Drama', 'genre_Fantasy', 'genre_Film-Noir',
                  'genre_Horror', 'genre_Musical', 'genre_Mystery',
                  'genre_Romance', 'genre_Sci-Fi', 'genre_Thriller',
                  'genre_War', 'genre_Western']

KERNEL = GPy.kern.RBF(input_dim=len(MOVIE_FEATURES), ARD=True)

RESPONSE = "user_normed_rating"
RESPONSE_MEAN = "user_mean"

# export settings
EXPORT = True
EXPORT_PATH = r"data\predictions"
EXPORT_FILE_NAME = "per_user_movie_normal_rbf_pred.csv"

# ------- #
# Dataset #
# ------- #
ratings = io.load_rating_movie("normal")

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
