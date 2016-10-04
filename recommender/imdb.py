import pandas as pd
import requests
from multiprocessing.dummy import Pool as ThreadPool


def generate_urls(movies_df):
    """
    Returns a list of omdbapi urls and their associated index, title, movie_id
    """
    res = []
    print("Total:", len(movies_df))
    for movie_id, row in movies.iterrows():
        imdb_id = row["imdbID"]
        url = "http://www.omdbapi.com/?i=tt{0}&plot=short&r=json".format('%07d' % imdb_id)  # IMDB IDs are 7 chars
        res.append((movie_id, row["title"], imdb_id, url))
    return res


def get_genre(movie):
    """
    Helper method for multi-threading
    """
    movie_id = movie[0]
    title = movie[1]
    imdb_id = movie[2]
    url = movie[3]
    print(movie_id, title, imdb_id)

    genre = requests.get(url).json().get("Genre", "")
    return movie + (genre, )


df = pd.read_csv('movies.dat', sep="\t", encoding="ISO-8859-1", index_col=0)
movies = df[["title", "imdbID"]]
urls = generate_urls(movies)

pool = ThreadPool(10)  # 10 threads
results = pool.map(get_genre, urls)

pool.close()
pool.join()

genres = [row[4] for row in sorted(results, key=lambda x: x[0])]
movies["genre"] = genres

movies.to_csv(r"data/movies_with_imdb_genres.csv")
