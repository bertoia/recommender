"""
Calculate the validation scores for all models
"""
import pandas as pd
from recommender.validation import get_validation_scores_per_model,\
                                   get_validation_score_combine_var,\
                                   get_validation_score_combine_simple
import recommender.io as io
import os

pd.options.display.width = 150
pd.options.mode.chained_assignment = None

# Paths of results
directory = r"data\predictions"
per_user_result_files = [os.path.join(directory, file) for file in os.listdir(directory)
                         if file.endswith(".csv") and file.startswith("per_user")]

per_movie_result_files = [os.path.join(directory, file) for file in os.listdir(directory)
                          if file.endswith(".csv") and file.startswith("per_movie")]

ratings = io.get_rating()
ratings_test = ratings.query("test == True")
ratings_train = ratings.query("test == False")

# per_movie models
per_movie_scores = []
for movie_path in per_movie_result_files:
    per_movie_result = pd.merge(ratings_test, pd.read_csv(movie_path))
    per_movie_result = per_movie_result[~per_movie_result.movie_est.isnull()]
    per_movie_result["movie_est"][per_movie_result.movie_est > 5] = 5
    per_movie_result["movie_est"][per_movie_result.movie_est < 1] = 1
    scores = get_validation_scores_per_model(per_movie_result["rating"],
                                             per_movie_result["movie_est"])
    scores.name = "mov_" + os.path.basename(movie_path)[10:-9]
    per_movie_scores.append(scores)

per_movie_scores = pd.DataFrame(per_movie_scores)

# per_user models
per_user_scores = []
for user_path in per_user_result_files:
    per_user_result = pd.merge(ratings_test, pd.read_csv(user_path))
    per_user_result = per_user_result[~per_user_result.user_est.isnull()]
    per_user_result["user_est"][per_user_result.user_est > 5] = 5
    per_user_result["user_est"][per_user_result.user_est < 1] = 1
    scores = get_validation_scores_per_model(per_user_result["rating"],
                                             per_user_result["user_est"])
    scores.name = "usr_" + os.path.basename(user_path)[9:-9]
    per_user_scores.append(scores)

per_user_scores = pd.DataFrame(per_user_scores)

# combined model
per_combi_scores = []
for user_path in per_user_result_files:
    for movie_path in per_movie_result_files:
        per_movie_result = pd.merge(ratings_test, pd.read_csv(movie_path))
        per_user_result = pd.merge(ratings_test, pd.read_csv(user_path))
        per_user_result["user_est"][per_user_result.user_est > 5] = 5
        per_user_result["user_est"][per_user_result.user_est < 1] = 1
        per_movie_result["movie_est"][per_movie_result.movie_est > 5] = 5
        per_movie_result["movie_est"][per_movie_result.movie_est < 1] = 1
        combi_result = pd.merge(per_movie_result[["rating", "rating_id", 'movie_est', 'movie_est_var']],
                                per_user_result[["rating_id", 'user_est', 'user_est_var']],
                                on="rating_id")
        combi_result["movie_est"] = combi_result["movie_est"].fillna(combi_result.user_est)
        combi_result["movie_est_var"] = combi_result["movie_est_var"].fillna(combi_result.user_est_var)
        combi_result["user_est"] = combi_result["user_est"].fillna(combi_result.movie_est)
        combi_result["user_est_var"] = combi_result["user_est_var"].fillna(combi_result.movie_est_var)
        scores = get_validation_score_combine_var(combi_result["rating"],
                                                  combi_result["user_est"],
                                                  combi_result["movie_est"],
                                                  combi_result["user_est_var"],
                                                  combi_result["movie_est_var"])
        scores["usr_m"] = "usr_" + os.path.basename(user_path)[9:-9]
        scores["mov_m"] = "mov_" + os.path.basename(movie_path)[10:-9]
        scores["combi_type"] = "var_t1"

        scores2 = get_validation_score_combine_simple(combi_result["rating"],
                                                      combi_result["user_est"],
                                                      combi_result["movie_est"])
        scores2["usr_m"] = "usr_" + os.path.basename(user_path)[9:-9]
        scores2["mov_m"] = "mov_" + os.path.basename(movie_path)[10:-9]
        scores2["combi_type"] = "simple_avg"

        per_combi_scores.append(scores)
        per_combi_scores.append(scores2)

per_combi_scores = pd.DataFrame(per_combi_scores)

# ------ #
# Scores #
# ------ #
print(per_movie_scores.sort_values("explained_var_score", ascending=False))
print()
print(per_user_scores.sort_values("explained_var_score", ascending=False))
print()
print(per_combi_scores.sort_values("explained_var_score", ascending=False))

