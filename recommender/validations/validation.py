from recommender import io
import pandas as pd
import numpy as np
import GPy
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score


# TODO: response mean scaling and skew correction
def get_per_user_model_validation_result(users, movies, ratings,
                                         per_user_kernel, per_movie_kernel,
                                         movie_features, user_features,
                                         train_test_ratio):
    """

    :param users: user dataframe
    :param movies: movie dataframe
    :param ratings: rating dataframe
    :param per_user_kernel: kernel for per user model
    :param per_movie_kernel: kernel for per movie model
    :param movie_features: list of movie features
    :param user_features: list of user features
    :param train_test_ratio: train test ratio
    :return: result dataframe
    :returns pd.DataFrame
    """

    # ---------------- #
    # Train test split #
    # ---------------- #
    ratings = pd.merge(ratings, movies, on="movie_id")
    ratings = pd.merge(ratings, users, on="user_id")

    ratings = ratings[["user_id", "movie_id"] +
                      movie_features + user_features + ["rating"]]
    ratings_test, ratings_train = train_test_split(ratings,
                                                   test_size=train_test_ratio,
                                                   stratify=ratings[["user_id"]])

    # -------------- #
    # Per user model #
    # -------------- #
    user_ids = sorted(list(ratings.user_id.unique()))
    user_models = ratings.groupby("user_id").size().to_frame("num_reviews")
    user_models["num_test"] = ratings_test.groupby("user_id").size()
    user_models["num_train"] = ratings_train.groupby("user_id").size()
    user_models["model"] = None

    mm_scaler = preprocessing.MinMaxScaler()
    # ~ 20mins to train
    for user_id in user_ids:
        if user_id % 100 == 0:
            print(user_id)
        rating_test_user = ratings_test[ratings_test.user_id == user_id]
        rating_train_user = ratings_train[ratings_train.user_id == user_id]
        test_x, test_y = rating_test_user[movie_features], rating_test_user[
            ["rating"]]
        train_x, train_y = rating_train_user[movie_features], rating_train_user[
            ["rating"]]

        train_x = mm_scaler.fit_transform(train_x)
        test_x = mm_scaler.transform(test_x)

        # User GP Model
        m = GPy.models.GPRegression(train_x, train_y, per_user_kernel)
        m.optimize()

        Yhat, variance = m.predict(test_x)
        # user_models.set_value(user_id, "model", m)

        ratings_test.loc[rating_test_user.index, "est_user"] = Yhat
        ratings_test.loc[rating_test_user.index, "est_user_var"] = variance

    # --------------- #
    # Per movie model #
    # --------------- #
    movie_ids = sorted(list(ratings.movie_id.unique()))
    movie_models = ratings.groupby("movie_id").size().to_frame("num_reviews")
    movie_models["num_test"] = ratings_test.groupby("movie_id").size()
    movie_models["num_train"] = ratings_train.groupby("movie_id").size()
    movie_models["model"] = None

    mm_scaler = preprocessing.MinMaxScaler()

    # ~ 30mins to train
    for movie_id in movie_ids:
        if movie_id % 100 == 0:
            print(movie_id)
        rating_test_movie = ratings_test[ratings_test.movie_id == movie_id]
        rating_train_movie = ratings_train[ratings_train.movie_id == movie_id]
        test_x, test_y = rating_test_movie[user_features], rating_test_movie[
            ["rating"]]
        train_x, train_y = rating_train_movie[user_features],\
                           rating_train_movie[["rating"]]

        if test_x.shape[0] == 0 or train_x.shape[0] == 0:
            continue

        train_x = mm_scaler.fit_transform(train_x)
        test_x = mm_scaler.transform(test_x)

        # Movie GP Model
        m = GPy.models.GPRegression(train_x, train_y, per_movie_kernel)
        m.optimize()

        Yhat, variance = m.predict(test_x)
        # user_models.set_value(movie_id, "model", m)

        ratings_test.loc[rating_test_movie.index, "est_movie"] = Yhat
        ratings_test.loc[rating_test_movie.index, "est_movie_var"] = variance

    # ------ #
    # Result #
    # ------ #
    result = ratings_test[["user_id", "movie_id", "rating",
                           "est_user", "est_movie",
                           "est_user_var", "est_movie_var"]]

    # 46 movies have no model due to lack of data
    result["est_movie"] = result["est_movie"].fillna(result.est_user)
    result["est_movie_var"] = result["est_movie_var"].fillna(
        result.est_user_var)

    return

def get_validation_score(result):
    # Metric 1: Explained Variance Score
    # Percentage of variance explained by model. 1 is good. 0 is bad.
    score = {"explained variance": dict()}  # pd.DataFrame(columns=["per_user", "per_movie", "avg", "weighted_avg_1"])

    # per user model
    score["explained variance"]["per_user"] = explained_variance_score(result["rating"],
                                                                       result["est_user"])

    # per movie model: 0.22776182757516739
    explained_variance_score(result["rating"],
                             result["est_movie"])

    # avg of movie/user models: 0.26453133595803247
    explained_variance_score(result["rating"],
                             (result["est_movie"] + result["est_user"]) / 2)

    # weighted avg of movie/user models: 0.26915533848006079
    # One quick and dirty way to do the variance based weighted avg.
    explained_variance_score(result["rating"],
                             (result["est_movie"] * result["est_user_var"] +
                              result["est_user"] * result["est_movie_var"]) /
                             (result["est_movie_var"] + result["est_user_var"]))


# ------ #
# Export #
# ------ #
result.to_csv("data/result_movie_easy_user_easy_genre_prob.csv", index=False)
