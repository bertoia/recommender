import os
import re

class MovieCorpus(object):
    def __init__(self, dirname, fname):
        self.dirname = dirname
        self.fname = fname

    def __iter__(self):
        for line in open(os.path.join(self.dirname, self.fname)):
            alpha = re.split('[>"]', line)
            alpha = map(lambda x: x.strip('\xa0 '), alpha)
            alpha = filter(lambda x: re.search('\w+', x), alpha)
            yield list(alpha)

class AltMovieCorpus(MovieCorpus):
    def __iter__(self):
        for line in open(os.path.join(self.dirname, self.fname)):
            alpha = re.split(';', line)
            beta = []
            for x in alpha:
                if ',' in x:
                    beta.extend(x.split(','))
                else:
                    beta.append(x)
            beta = map(lambda x: x.strip('\n').lower(), beta)
            yield list(beta)

