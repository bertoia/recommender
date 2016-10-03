from recommender import io
import pandas as pd
import numpy as np
import GPy
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score

# -------- #
# Settings #
# -------- #
TRAIN_TEST_RATIO = .7

movie = io.get_movie_attr("easy")
user = io.get_user_attr("easy")

MOVIE_FEATURES = ['year', 'rtAllCriticsRating', 'rtAudienceRating',
                  'rtAllCriticsNumReviews', 'rtAudienceNumRatings',
                  'rtAllCriticsScore', 'rtAudienceScore']
USER_FEATURES = ["age", "gender"]


# ---------- #
# Train test #
# ---------- #


rating_raw = io.get_rating()
rating = pd.merge(rating_raw, movie, on="movie_id")
rating = pd.merge(rating, user, on="user_id")

rating = rating[["user_id", "movie_id"] +
                MOVIE_FEATURES + USER_FEATURES + ["rating"]]
rating_test, rating_train = train_test_split(rating,
                                             test_size=TRAIN_TEST_RATIO,
                                             stratify=rating[["user_id"]])

# -------------- #
# Per user model #
# -------------- #
user_ids = list(rating.user_id.unique())
user_models = rating.groupby("user_id").size().to_frame("num_reviews")
user_models["num_test"] = rating_test.groupby("user_id").size()
user_models["num_train"] = rating_train.groupby("user_id").size()
user_models["model"] = None

mm_scaler = preprocessing.MinMaxScaler()
kernel = GPy.kern.RBF(input_dim=len(MOVIE_FEATURES),
                      variance=1.,
                      lengthscale=1.)

# ~ 20mins to train
for user_id in user_ids:
    if user_id % 100 == 0: print(user_id)
    rating_test_user = rating_test[rating_test.user_id == user_id]
    rating_train_user = rating_train[rating_train.user_id == user_id]
    test_x, test_y = rating_test_user[MOVIE_FEATURES], rating_test_user[["rating"]]
    train_x, train_y = rating_train_user[MOVIE_FEATURES], rating_train_user[["rating"]]

    train_x = mm_scaler.fit_transform(train_x)
    test_x = mm_scaler.transform(test_x)

    # User GP Model
    m = GPy.models.GPRegression(train_x, train_y, kernel)
    m.optimize()

    Yhat, variance = m.predict(test_x)
    # user_models.set_value(user_id, "model", m)

    rating_test.loc[rating_test_user.index, "est_user"] = Yhat
    rating_test.loc[rating_test_user.index, "est_user_var"] = variance

# --------------- #
# Per movie model #
# --------------- #
movie_ids = sorted(list(rating.movie_id.unique()))
movie_models = rating.groupby("movie_id").size().to_frame("num_reviews")
movie_models["num_test"] = rating_test.groupby("movie_id").size()
movie_models["num_train"] = rating_train.groupby("movie_id").size()
movie_models["model"] = None

mm_scaler = preprocessing.MinMaxScaler()
kernel = GPy.kern.RBF(input_dim=len(USER_FEATURES),
                      variance=1.,
                      lengthscale=1.)
# ~ 30mins to train
for movie_id in movie_ids:
    print(movie_id)

    rating_test_movie = rating_test[rating_test.movie_id == movie_id]
    rating_train_movie = rating_train[rating_train.movie_id == movie_id]
    test_x, test_y = rating_test_movie[USER_FEATURES], rating_test_movie[["rating"]]
    train_x, train_y = rating_train_movie[USER_FEATURES], rating_train_movie[["rating"]]

    if test_x.shape[0] == 0 or train_x.shape[0] == 0:
        continue

    train_x = mm_scaler.fit_transform(train_x)
    test_x = mm_scaler.transform(test_x)

    # Movie GP Model
    m = GPy.models.GPRegression(train_x, train_y, kernel)
    m.optimize()

    Yhat, variance = m.predict(test_x)
    # user_models.set_value(movie_id, "model", m)

    rating_test.loc[rating_test_movie.index, "est_movie"] = Yhat
    rating_test.loc[rating_test_movie.index, "est_movie_var"] = variance

# ------ #
# Result #
# ------ #
result = rating_test[["user_id", "movie_id", "rating",
                      "est_user", "est_movie",
                      "est_user_var", "est_movie_var"]]

# 46 movies have no model due to lack of data
result["est_movie"] = result["est_movie"].fillna(result.est_user)
result["est_movie_var"] = result["est_movie_var"].fillna(result.est_user_var)

# Percentage of variance explained by model. 1 is good. 0 is bad.

# per user model: 0.14148969349865204
explained_variance_score(result["rating"],
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
result.to_csv("data/result_movie_easy_user_easy.csv", index=False)
