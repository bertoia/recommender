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
                  'rtAllCriticsScore', 'rtAudienceScore']

KERNEL = GPy.kern.Linear(input_dim=len(MOVIE_FEATURES), ARD=True)

RESPONSE = "user_normed_rating"
RESPONSE_MEAN = "user_mean"

# export settings
EXPORT = True
EXPORT_PATH = r"data\predictions"
EXPORT_FILE_NAME = "per_user_movie_easy_linear_pred.csv"

# ------- #
# Dataset #
# ------- #
ratings = io.load_rating_movie("easy")

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
