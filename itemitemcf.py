import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

test_pairs=[[2048, 788],[2048, 36955],[2048, 77],[1024, 462],[1024, 393],[1024, 36955],[1024, 77],[1024, 268]]

class UserUserCf:
    DOC_ID = 'docId'
    USER_ID = 'userId'
    RATING = 'rating'
    TITLE = 'title'
    def __init__(self,ratings_path='data/ratings.csv',titles_path='data/movie-titles.csv',kneighs=20):
        self.load_ratings(ratings_path)
        self.calc_correlations()
        self.load_titles(titles_path)
        self.kneighs=kneighs

    def load_ratings(self,path):
        r = pd.read_csv(path,names=[self.USER_ID,self.DOC_ID,self.RATING])
        self.ratings = r.pivot(self.USER_ID,self.DOC_ID,self.RATING)  # Users x Items
        self.user_means = self.ratings.mean(axis=1)
        self.centered = (self.ratings.T - self.user_means) # Items x Users

    def set_kneighs(self, kneighs):
        self.kneighs = kneighs

    def load_titles(self,path):
        self.titles = pd.read_csv(path,names=[self.DOC_ID,self.TITLE],index_col=0)

    def calc_correlations(self):
        cmatrix = cosine_similarity(self.centered.fillna(0))
        cmatrix[cmatrix<0] = 0
        self.correlations = pd.DataFrame(cmatrix, index=self.centered.index, columns= self.centered.index)

    def get_neighbours(self,user_ratings,user,item):
        other_items = self.correlations[item].drop(item)[user_ratings.index]
        return other_items.order(ascending=False)[:self.kneighs]
        
    def score(self,user,item):
        user_ratings = self.ratings.T[user].drop(item).dropna()
        neighs = self.get_neighbours(user_ratings,user,item)
        return user_ratings[neighs.index].dot(neighs) / np.abs(neighs).sum()

    def process(self,pairs,kneighs=20):
        for user, item in pairs:
            s = self.score(user,item)
            print "{},{},{},{}".format(user,item,round(s,4),self.titles.ix[item][0])

UserUserCf().process(test_pairs)