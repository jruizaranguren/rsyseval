import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class UserUserCf:
    DOC_ID = 'docId'
    USER_ID = 'userId'
    RATING = 'rating'
    TITLE = 'title'
    def __init__(self,ratings_path='data/ratings.csv',titles_path='data/movie-titles.csv'):
        self.load_ratings(ratings_path)
        self.calc_correlations()
        self.load_titles(titles_path)

    def load_ratings(self,path):
        r = pd.read_csv(path,names=[self.USER_ID,self.DOC_ID,self.RATING])
        r = r.pivot(self.USER_ID,self.DOC_ID,self.RATING)
        self.user_means = r.mean(axis=1)
        self.ratings = (r.T - self.user_means).T

    def load_titles(self,path):
        self.titles = pd.read_csv(path,names=[self.DOC_ID,self.TITLE],index_col=0)

    def calc_correlations(self):
        cmatrix = cosine_similarity(self.ratings.fillna(0))
        self.correlations = pd.DataFrame(cmatrix, index=self.ratings.index, columns=self.ratings.index)
        
    def get_neighbours(self,item_ratings,item,user):
        other_users = self.correlations[user].drop(user)[item_ratings.index]
        return other_users.order(ascending=False)[:30]

    def score(self,user,item):
        item_ratings = self.ratings[item].drop(user).dropna()
        neighs = self.get_neighbours(item_ratings, item, user)
        user_mean = self.user_means.ix[user]
        prediction = item_ratings[neighs.index].dot(neighs) / np.abs(neighs).sum()
        return user_mean + prediction

    def process(self,pairs):
        for user, item in pairs:
            s = self.score(user,item)
            print "{},{},{},{}".format(user,item,round(s,4),self.titles.ix[item][0])

#test_pairs=[[2048, 788],[2048, 36955],[2048, 77],[1024, 462],[1024, 393],[1024, 36955],[1024, 77],[1024, 268]]
