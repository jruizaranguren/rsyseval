import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

test_pairs=[[2048, 788],[2048, 36955],[2048, 77],[1024, 462],[1024, 393],[1024, 36955],[1024, 77],[1024, 268]]
assignment_pairs=[[2137,786],[2137,77],[2137,1422],[2137,107],[2137,2164],[5043,10020],[5043,4327],[5043,8587],[5043,36658],[5043,1572],[2059,568],[2059,122],[2059,862],[2059,812],[2059,557],[1024,9802],[1024,857],[1024,238],[1024,424],[1024,955],[1914,640],[1914,568],[1914,786],[1914,672],[1914,857]]
assignment_basket = [462,788]

class ItemItemCf:
    DOC_ID = 'docId'
    USER_ID = 'userId'
    RATING = 'rating'
    TITLE = 'title'
    def __init__(self,ratings_path='data/ratings.csv',titles_path='data/movie-titles.csv',fold=None,kneighs=20):
        self.load_ratings(ratings_path,fold)
        self.calc_correlations()
        self.load_titles(titles_path)
        self.kneighs=kneighs

    def load_ratings(self,path,fold):
        if fold is None:
            r = pd.read_csv(path,names=[self.USER_ID,self.DOC_ID,self.RATING]) 
            self.ratings = r.pivot(self.USER_ID,self.DOC_ID,self.RATING)  # Users x Items
        else:
            self.ratings = fold
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

    def find_similar(self, items, topk=5):
        similars = self.correlations[items].drop(items).sum(axis=1).order(ascending=False)[:topk]
        for item, score in similars.iteritems():
            print "{},{},{}".format(item,round(score,4),self.titles.ix[item][0])

    def recommend(self,pairs,kneighs=20):
        scores = []
        for user, item in pairs:
            s = self.score(user,item)
            scores.append(s)
            print "{},{},{},{}".format(user,item,round(s,4),self.titles.ix[item][0])

        return scores

    

