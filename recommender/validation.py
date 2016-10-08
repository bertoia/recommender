import pandas as pd
import numpy as np
import GPy
from sklearn import preprocessing
from sklearn.metrics import explained_variance_score, mean_absolute_error,\
                            mean_squared_error, median_absolute_error,\
                            r2_score


# --------------- #
# Per user model #
# --------------- #
def per_user_prediction(ratings, movie_features, kernel, response="user_normed_rating",
                        response_mean="user_mean"):
    """

    :param ratings: rating dataframe
    :param movie_features: list of movie features
    :param kernel: kernel for per user model
    :param response: column name of y in ratings
    :param response_mean: column name of response scale
    :return: result dataframe
    :returns pd.DataFrame
    """
    ratings_full = ratings[["rating_id", "user_id", "test"] + movie_features + [response]]
    user_ids = sorted(list(ratings_full.user_id.unique()))

    mm_scaler = preprocessing.MinMaxScaler()

    for user_id in user_ids:
        if user_id % 100 == 0:
            print(user_id)
        ratings_user = ratings_full[ratings_full.user_id == user_id]
        ratings_train_user = ratings_user.query("test == False")
        train_x, train_y = ratings_train_user[movie_features], ratings_train_user[[response]]
        test_x, test_y = ratings_user[movie_features], ratings_user[[response]]

        train_x = mm_scaler.fit_transform(train_x)
        test_x = mm_scaler.transform(test_x)

        # User GP Model
        try:
            m = GPy.models.GPRegression(train_x, train_y, kernel)
            m.optimize()
        except:
            continue

        Yhat, variance = m.predict(test_x)

        ratings_full.loc[ratings_user.index, "user_est"] = Yhat
        ratings_full.loc[ratings_user.index, "user_est_var"] = variance

    ratings_full["user_est"] = ratings_full["user_est"] + ratings[response_mean]

    return ratings_full[["rating_id", "user_est", "user_est_var"]]


# -------------- #
# Per movie model #
# -------------- #
def per_movie_prediction(ratings, user_features, kernel, response="movie_normed_rating",
                         response_mean="movie_mean"):
    """

    :param ratings: rating dataframe
    :param user_features: list of movie features
    :param kernel: kernel for per user model
    :param response: column name of y in ratings
    :param response_mean: column name of response scale
    :return: result dataframe
    :returns pd.DataFrame
    """
    ratings_full = ratings[["rating_id", "movie_id", "test"] + user_features + [response]]
    movie_ids = sorted(list(ratings_full.movie_id.unique()))

    mm_scaler = preprocessing.MinMaxScaler()

    for movie_id in movie_ids:
        if movie_id % 100 == 0:
            print(movie_id)
        ratings_movie = ratings_full[ratings_full.movie_id == movie_id]
        ratings_train_movie = ratings_movie.query("test == False")
        train_x, train_y = ratings_train_movie[user_features], ratings_train_movie[[response]]
        test_x, test_y = ratings_movie[user_features], ratings_movie[[response]]

        if test_x.shape[0] == 0 or train_x.shape[0] == 0:
            continue

        train_x = mm_scaler.fit_transform(train_x)
        test_x = mm_scaler.transform(test_x)

        # Movie GP Model
        try:
            m = GPy.models.GPRegression(train_x, train_y, kernel)
            m.optimize()
        except:
            continue

        Yhat, variance = m.predict(test_x)

        ratings_full.loc[ratings_movie.index, "movie_est"] = Yhat
        ratings_full.loc[ratings_movie.index, "movie_est_var"] = variance

    ratings_full["movie_est"] = ratings_full["movie_est"] + ratings[response_mean]

    return ratings_full[["rating_id", "movie_est", "movie_est_var"]]


# ------- #
# Metrics #
# ------- #
def get_validation_score_combine_var(y_true, y_pred, y_pred2, y_var, y_var2):
    y_pred_combi = ((y_pred * y_var2) + (y_pred2 * y_var)) / (y_var + y_var2)
    return get_validation_scores_per_model(y_true, y_pred_combi)


def get_validation_score_combine_simple(y_true, y_pred, y_pred2):
    y_pred_combi = (y_pred + y_pred2) / 2
    return get_validation_scores_per_model(y_true, y_pred_combi)


def get_validation_scores_per_model(y_true, y_pred):
    score = dict()
    score["explained_var_score"] = explained_variance_score(y_true, y_pred)
    score["r2_score"] = r2_score(y_true, y_pred)
    score["mean_abs_err"] = mean_absolute_error(y_true, y_pred)
    score["median_abs_err"] = median_absolute_error(y_true, y_pred)
    score["mean_sqr_err"] = mean_squared_error(y_true, y_pred)
    return pd.Series(score)
