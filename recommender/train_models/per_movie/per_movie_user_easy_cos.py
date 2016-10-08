from recommender.validation import per_movie_prediction
import recommender.io as io
import GPy
import os

# -------- #
# Settings #
# -------- #
# per movie model settings
USER_FEATURES = ['gender', 'age']

KERNEL = GPy.kern.Cosine(input_dim=len(USER_FEATURES),
                         variance=1., ARD=True)

RESPONSE = "movie_normed_rating"
RESPONSE_MEAN = "movie_mean"

# export settings
EXPORT = True
EXPORT_PATH = r"data\predictions"
EXPORT_FILE_NAME = "per_movie_user_easy_cos_pred.csv"

# ------- #
# Dataset #
# ------- #
ratings = io.load_rating_user("easy")

# ------ #
# result #
# ------ #
print(EXPORT_FILE_NAME)
result_per_movie = per_movie_prediction(ratings, USER_FEATURES, KERNEL,
                                        RESPONSE, RESPONSE_MEAN)

# ------ #
# Export #
# ------ #
if EXPORT:
    result_per_movie.to_csv(os.path.join(EXPORT_PATH, EXPORT_FILE_NAME),
                            index=False)

print(EXPORT_FILE_NAME + " completed!")
