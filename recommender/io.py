import pandas as pd


def get_movie_attr(difficulty="easy"):
    """
    difficulty selection for movie.csv
    easy:   kernel supports numeric inputs only (7 features)
    normal: kernel supports numeric inputs only (25 features)
    hard:   kernel supports numeric and categorical inputs (director, country)
    brutal: kernel supports numeric and multi-categorical input (genres, actors)
    """
    return pd.read_csv(r"data\movie_"+difficulty+".csv", encoding="ISO-8859-1")


def get_user_attr(difficulty="easy"):
    """
    difficulty selection for user
    easy:   kernel supports numeric inputs only (2 features)
    normal: kernel supports numeric inputs only (25 features)
    hard:   kernel supports numeric and categorical input (occupation)
    """

    return pd.read_csv(r"data\user_"+difficulty+".csv")


def get_rating():
    return pd.read_csv(r"data\rating.csv")


if __name__ == '__main__':
    """Example of a per user model"""
    import GPy
    import pandas as pd

    movies = get_movie_attr(difficulty="easy")
    users = get_user_attr(difficulty="easy")
    ratings = get_rating()

    ratings_movies = pd.merge(ratings, movies, how="left")
    ratings_movies_user_2334 = ratings_movies.query("user_id==2334")

    # use .as_matrix() to convert pandas dataframe to numpy array
    X = ratings_movies_user_2334[["rtAllCriticsRating"]].as_matrix()
    Y = ratings_movies_user_2334[["rating"]].as_matrix()

    m = GPy.models.GPRegression(X, Y)
    m.plot()
