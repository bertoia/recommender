import pandas as pd
import os

# ----- #
# Movie #
# ----- #
movie = pd.read_csv(r"data/movielen/movies.dat", sep="::")
movie["genres"] = movie["genres"].apply(lambda x: tuple(x.split("|")))
movie["year"] = pd.to_numeric(movie["title"].apply(lambda x: x[-5:-1]))
movie["title"] = movie["title"].apply(lambda x: x[:-7])
movie.describe()  # 3883 rows

# prep tomato dataset
movie_tomato = pd.read_csv(r"data/tomato/movies.dat", sep=r"\t")
movie_tomato = movie_tomato.query("id<=3952").reset_index(drop=True)
cols = ['id',
        'rtAllCriticsRating', 'rtAllCriticsNumReviews', 'rtAllCriticsScore',
        'rtAudienceRating', 'rtAudienceNumRatings', 'rtAudienceScore']
movie_tomato = movie_tomato[cols]  #3661 rows

# join directors and countries
directors = pd.read_csv(r"data/tomato/movie_directors.dat", sep=r"\t")
countries = pd.read_csv(r"data/tomato/movie_countries.dat", sep=r"\t")
actors = pd.read_csv(r"data/tomato/movie_actors.dat", sep=r"\t")
actors = actors.query("ranking<=5")
actors = actors.groupby("movieID").head(5)\
               .groupby("movieID").aggregate(lambda x: tuple(x))

movie_tomato = pd.merge(movie_tomato,
                        directors[["movieID", "directorName"]],
                        how="inner", left_on="id", right_on="movieID")  # 3645 rows

movie_tomato.drop("movieID", axis=1, inplace=True)

movie_tomato = pd.merge(movie_tomato, countries,
                        how="inner", left_on="id", right_on="movieID")  # 3645 rows
movie_tomato.drop("movieID", axis=1, inplace=True)

movie_tomato = pd.merge(movie_tomato, actors[["actorName"]],
                        how="left", left_on="id", right_index=True)  # 3645 rows

# Combine MovieLen and tomato
movie_final = pd.merge(movie, movie_tomato,
                       how="inner", left_on="movie_id", right_on="id")  # 3645 rows

# movie_final.columns
# Index(['movie_id', 'title', 'genres', 'year', 'id',
#        'rtAllCriticsRating', 'rtAllCriticsNumReviews', 'rtAllCriticsScore',
#        'rtAudienceRating', 'rtAudienceNumRatings', 'rtAudienceScore',
#        'directorName', 'country', 'actorName'],
#        dtype='object')

########
# User #
########
user = pd.read_csv(r"data/movielen/users.dat", sep="::")
user.drop("zip_code", axis=1, inplace=True)
user["age"][user["age"] == 1] = 12  # under 18
user["age"][user["age"] == 18] = 21  # 18-24
user["age"][user["age"] == 25] = 30  # 25-34
user["age"][user["age"] == 35] = 40  # 35-45
user["age"][user["age"] == 45] = 47  # 45-49
user["age"][user["age"] == 50] = 53  # 50-55
user["age"][user["age"] == 56] = 60  # 56+

occupation_dict = {
    0:  "other",
    1:  "academic/educator",
    2:  "artist",
    3:  "clerical/admin",
    4:  "college/grad student",
    5:  "customer service",
    6:  "doctor/health care",
    7:  "executive/managerial",
    8:  "farmer",
    9:  "homemaker",
    10:  "K-12 student",
    11:  "lawyer",
    12:  "programmer",
    13:  "retired",
    14:  "sales/marketing",
    15:  "scientist",
    16:  "self-employed",
    17:  "technician/engineer",
    18:  "tradesman/craftsman",
    19:  "unemployed",
    20:  "writer"
}

user["occupation"] = user["occupation"].\
                        apply(lambda x: occupation_dict[x])

user_final = user

# ------- #
# Ratings #
# ------- #

# TODO: Filter away ratings for removed movie_id
rating = pd.read_csv(r"data/movielen/ratings.dat", sep="::")
rating.drop("timestamp", axis=1, inplace=True)

rating_final = rating


# ------ #
# Export #
# ------ #

movie_final.to_csv("data\movie.csv", index=False)
user_final.to_csv("data\user.csv", index=False)
rating_final.to_csv("data\rating.csv", index=False)


# ------ #
# Import #
# ------ #
# TODO: Import tuple as tuple instead of string
mv = pd.read_csv(r"data\movie.csv", encoding="ISO-8859-1")
rt = pd.read_csv(r"data\rating.csv")
ur = pd.read_csv(r"data\user.csv")
