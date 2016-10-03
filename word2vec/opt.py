import moviecorpus as mclib
import gensim

class Test():
    def __init__(self, word1, word2):
        self.word1 = word1
        self.word2 = word2

class Optimiser():
    def __init__(self, dirname, fname):
        self.corpus = mclib.MovieCorpus(dirname, fname)

    def optimise(self, params={}):
        loop = len(params['size'])
        sz = params.get('size') if params.get('size') else [12]
        al = params.get('alpha') if params.get('alpha') else [0.002]
        mt = params.get('min_count') if params.get('min_count') else [1]
        wk = params.get('workers') if params.get('workers') else [8]
        it = params.get('iter') if params.get('iter') else [150]
        bw = params.get('batch_words') if params.get('batch_words') else [20]
        bigram_trans = gensim.models.Phrases(self.corpus)
        # 6 nested loops, what?!
        # maybe there's a better way
        # not sure if we can use scikit-learn
        for s in sz:
            for a in al:
                for m in mt:
                    for w in wk:
                        for t in it:
                            for b in bw:
                                print('###### TRAIN PARAMS - size={} alpha={} min_count={} workers={} iter={} batch_words={} ######'.format(s,a,m,w,t,b))
                                model = gensim.models.Word2Vec(bigram_trans[self.corpus],
                                    size=s, alpha=a, min_count=m,
                                    workers=w, iter=t, batch_words=b)
                                for test in self.testset:
                                    print(test.word1, test.word2)
                                    print(model.similarity(test.word1, test.word2))

    def populate(self, test_file):
        self.testset = []
        with open(test_file, 'r') as t:
            for line in t:
                l = line.split(',')
                self.testset.append(Test(l[0], l[1].strip('\n')))

if __name__ == '__main__':
    dirname = '.'
    fname = 'corpus.txt'
    o = Optimiser(dirname, fname)
    tname = 'test_sim.txt'
    o.populate(tname)
    params = {}

    ###################
    # VARIABLE PARAMS #
    ###################
    params['size'] = [4, 8, 12, 16]
    params['iter'] = [400, 800, 1000]
    params['batch_words'] = [15, 20]
    ###################

    o.optimise(params)
