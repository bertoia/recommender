import pandas as pd


class Probabilities:
    def __init__(self, df):
        self.genre_counts = {}
        self.associated_genre_counts = {}
        self.probs = {}

        self._calculate_counts(df)
        self._calculate_probs()

    def _calculate_counts(self, df):
        for index, row in df.iterrows():
            try:
                genre_list = [x.strip() for x in row["genre"].split(",")]
            except AttributeError:
                continue

            for genre in genre_list:
                self.increment_count(self.genre_counts, genre)  # Update genre count

                if genre not in self.associated_genre_counts:  # Make a new dictionary to count associated genres
                    self.associated_genre_counts[genre] = {}

                for associated_genre in genre_list:
                    self.increment_count(self.associated_genre_counts[genre], associated_genre)

    def _calculate_probs(self):
        """
        Uses Witten-Bell smoothing to ensure non-zero probabilities
        """
        num_genres = len(self.genre_counts)

        for given_genre in self.associated_genre_counts:

            num_associated_genres = len(self.associated_genre_counts[given_genre])
            num_non_associated_genres = num_genres - num_associated_genres

            for genre in self.associated_genre_counts:
                # print(genre, given_genre)
                count = self.associated_genre_counts[given_genre].get(genre, 0)
                if count == 0:
                    self.probs[(genre, given_genre)] = num_associated_genres / \
                                                       (num_non_associated_genres *
                                                        (num_associated_genres + self.genre_counts[given_genre]))
                else:
                    self.probs[(genre, given_genre)] = count / \
                                                       (num_associated_genres + self.genre_counts[given_genre])

    def closeness(self, genre, given_genre):
        """
        Finds the "closeness" of a genre given a genre, P(genre | given_genre)
        """
        return self.probs[(genre, given_genre)]

    def summary(self):
        """
        Prints out a summary of the closeness between all genres, sorted by descending probability
        """
        lines = []
        for genre in self.genre_counts:
            for given_genre in self.genre_counts:
                lines.append(("{0} given {1}".format(genre, given_genre),
                              self.closeness(genre, given_genre)))

        for prob in sorted(lines, key=lambda x: x[1], reverse=True):
            print("{0: <30}".format(prob[0]), prob[1])

    def closeness_to_buckets(self, buckets, genres):
        """
        Given a list of genres, returns a list of the same size as buckets that has
        the closeness of the set of genres to each genre (each bucket).
        """
        closeness = []
        for bucket in buckets:
            closeness.append(max([self.closeness(genre, bucket) for genre in genres]))
        return closeness

    @staticmethod
    def increment_count(dictionary, key):
        dictionary[key] = dictionary.get(key, 0) + 1


genre_buckets = ["Drama", "Comedy", "Crime", "Action", "Thriller",
                 "Horror", "Fantasy", "Family", "Animation"]
df = pd.read_csv(r"movies_with_imdb_genres.csv", encoding="ISO-8859-1")
p = Probabilities(df)

rows = []
for __, row in df.iterrows():
    try:
        genre_list = [x.strip() for x in row["genre"].split(",")]
        rows.append(p.closeness_to_buckets(genre_buckets, genre_list))
    except AttributeError:
        rows.append([])

genre_features = pd.DataFrame(rows, columns=[x.lower() for x in genre_buckets])

combined = pd.concat([df, genre_features], axis=1)
combined.rename(columns={"d": "movie_id"}, inplace=True)
combined.drop(["imdbID", "genre"], axis=1, inplace=True)
combined.to_csv("movies_with_genre_buckets.csv", index=False)
