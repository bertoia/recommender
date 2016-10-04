import pandas as pd

# ----- #
# Movie #
# ----- #
movie = pd.read_csv(r"data/movielen/movies.dat", sep="::")
movie["genres"] = movie["genres"].apply(lambda x: tuple(x.split("|")))
movie["year"] = pd.to_numeric(movie["title"].apply(lambda x: x[-5:-1]))
movie["title"] = movie["title"].apply(lambda x: x[:-7])
movie.describe()  # 3883 rows

genres = ["Action", "Adventure", "Animation", "Children's", "Comedy",
          "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir",
          "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller",
          "War", "Western"]

for genre in genres:
    movie["genre_"+genre] = movie["genres"].apply(lambda x: 1 if genre in x else 0)

# prep tomato dataset
movie_tomato = pd.read_csv(r"data/tomato/movies.dat", sep=r"\t")
movie_tomato = movie_tomato.query("id<=3952").reset_index(drop=True)

# remove movies with missing values
movie_tomato = movie_tomato[movie_tomato["rtAllCriticsRating"] != r"\N"]
cols = ['id',
        'rtAllCriticsRating', 'rtAllCriticsNumReviews', 'rtAllCriticsScore',
        'rtAudienceRating', 'rtAudienceNumRatings', 'rtAudienceScore']

# numerify cols
numerify_cols = ['rtAllCriticsRating', 'rtAllCriticsNumReviews',
                 'rtAllCriticsScore', 'rtAudienceRating',
                 'rtAudienceNumRatings', 'rtAudienceScore']

for num_col in numerify_cols:
    movie_tomato[num_col] = pd.to_numeric(movie_tomato[num_col])

movie_tomato = movie_tomato[cols]  # 3661 rows

# join directors and countries
directors = pd.read_csv(r"data/tomato/movie_directors.dat", sep=r"\t")
countries = pd.read_csv(r"data/tomato/movie_countries.dat", sep=r"\t")
actors = pd.read_csv(r"data/tomato/movie_actors.dat", sep=r"\t")
actors = actors.query("ranking<=5")
actors = actors.groupby("movieID").head(5)\
               .groupby("movieID").aggregate(lambda x: tuple(x))

movie_tomato = pd.merge(movie_tomato, directors[["movieID", "directorName"]],
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

movie_final.drop("id", axis=1, inplace=True)

# add genre buckets
genre_buckets = pd.read_csv(r"data/movies_with_genre_buckets.csv", encoding="ISO-8859-1")
genre_buckets.drop("title", axis=1, inplace=True)
movie_final = pd.merge(movie_final, genre_buckets,
                       how="inner", left_on="movie_id", right_on="movie_id")


# ------------- #
# Movie Exports #
# ------------- #
# column types
id_cols = ['movie_id', 'title']
numeric_cols = ['year', 'rtAllCriticsRating', 'rtAudienceRating',
                'rtAllCriticsNumReviews', 'rtAudienceNumRatings',
                'rtAllCriticsScore', 'rtAudienceScore']
genre_bucket_cols = ['drama', 'comedy', 'crime', 'action', 'thriller',
                     'horror', 'fantasy', 'family', 'animation']
categorical_cols = ['directorName', 'country']
multi_categorical_cols = ["genres", "actorName"]
genre_cols = ["genre_"+genre for genre in genres]


# difficulty combinations
# Easy:   kernel supports numeric inputs only (7 features) + genre bucket probabilities (9 features)
# Normal: kernel supports numeric inputs only (25 features)
# Hard:   kernel supports numeric and categorical inputs (director, country)
# Brutal: kernel supports numeric and multi-categorical input (genres, actors)
easy_cols = id_cols + numeric_cols + genre_bucket_cols
normal_cols = id_cols + numeric_cols + genre_cols
hard_cols = id_cols + numeric_cols + categorical_cols
brutal_cols = id_cols + numeric_cols + categorical_cols + multi_categorical_cols

# export
movie_final[easy_cols].to_csv(r"data\movie_easy.csv", index=False)
movie_final[normal_cols].to_csv(r"data\movie_normal.csv", index=False)
movie_final[hard_cols].to_csv(r"data\movie_hard.csv", index=False)
movie_final[brutal_cols].to_csv(r"data\movie_brutal.csv", index=False)


# ---- #
# User #
# ---- #
user = pd.read_csv(r"data/movielen/users.dat", sep="::")
user.drop("zip_code", axis=1, inplace=True)
# revalue user age to reflect user group's mean
user["age"][user["age"] == 1] = 12  # under 18
user["age"][user["age"] == 18] = 21  # 18-24
user["age"][user["age"] == 25] = 30  # 25-34
user["age"][user["age"] == 35] = 40  # 35-45
user["age"][user["age"] == 45] = 47  # 45-49
user["age"][user["age"] == 50] = 53  # 50-55
user["age"][user["age"] == 56] = 60  # 56+

# Convert gender to numeric
user["gender"][user["gender"] == "M"] = 1
user["gender"][user["gender"] == "F"] = 0
user["gender"] = pd.to_numeric(user["gender"])

# convert occupation to actual values
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

for occ in occupation_dict.values():
    user["occ_" + occ] = user["occupation"].apply(lambda x: 1 if occ in x else 0)

user["occupation"] = user["occupation"].\
                        apply(lambda x: occupation_dict[x])

# ------------ #
# user Exports #
# ------------ #
occupations_cols = ["occ_"+occupation for occupation in occupation_dict.values()]

# difficulty combinations
# Easy:   kernel supports numeric inputs only (2 features)
# Normal: kernel supports numeric inputs only (25 features)
# Hard:   kernel supports numeric and categorical input (occupation)
easy_cols = ['user_id', 'gender', 'age']
normal_cols = ['user_id', 'gender', 'age'] + occupations_cols
hard_cols = ['user_id', 'gender', 'age', 'occupation']

user[easy_cols].to_csv(r'data\user_easy.csv', index=False)
user[normal_cols].to_csv(r'data\user_normal.csv', index=False)
user[hard_cols].to_csv(r'data\user_hard.csv', index=False)


# ------- #
# Ratings #
# ------- #
rating = pd.read_csv(r"data/movielen/ratings.dat", sep="::")
rating = pd.merge(rating, movie_final[["movie_id"]], how="inner")
rating.drop("timestamp", axis=1, inplace=True)

# --------------- #
# Ratings Exports #
# --------------- #
rating.to_csv(r"data\rating.csv", index=False)
