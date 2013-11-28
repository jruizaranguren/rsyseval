import pandas as pd
import numpy as np

test_pairs=[[2048, 788],[2048, 36955],[2048, 77],[1024, 462],[1024, 393],[1024, 36955],[1024, 77],[1024, 268]]

class SvdCf:
    DOC_ID = 'docId'
    USER_ID = 'userId'
    RATING = 'rating'
    TITLE = 'title'
    def __init__(self,ratings_path='data/ratings.csv',titles_path='data/movie-titles.csv',fold=None,mean_type='user'):
        self.mean_type = mean_type
        self.load_ratings(ratings_path,fold)
        self.load_titles(titles_path)
        self.model()
        
    def load_ratings(self,path,fold):
        r = pd.read_csv(path,names=[self.USER_ID,self.DOC_ID,self.RATING]) if fold is None else fold
        self.ratings = r.pivot(self.USER_ID,self.DOC_ID,self.RATING)  # Users x Items
        shape = self.ratings.shape

        if self.mean_type == 'user':
            self.baseline = self.ratings.mean(axis=1)
            self.centered = (self.ratings.T - self.baseline).T # Users x Items
        elif self.mean_type == 'item':
            self.baseline = self.ratings.mean(axis=0)
            self.centered = self.ratings - self.baseline
        elif self.mean_type == 'global':
            self.baseline = self.ratings.sum().sum() / self.ratings.count().sum()
            self.centered = self.ratings - self.baseline
        else:
            # User-Item personalized mean
            pass
        self.centered.fillna(0,inplace=True)

    def model(self):
        U,s,V = np.linalg.svd(self.centered,full_matrices=False)
        xtrans = (U[:,:10].dot(np.diag(s[:10]))).dot(V[:10,:])
        offset = pd.DataFrame(xtrans,index= self.ratings.index, columns= self.ratings.columns)
        if self.mean_type == 'user':
            self.rpred = (self.baseline + offset.T).T
        elif self.mean_type == 'item':
            self.rpred = self.baseline + offset
        elif self.mean_type == 'global':
            self.rpred = self.baseline + offset

    def predict(self,pairs):
        for user, item in pairs:
            s = self.rpred[item].ix[user]
            print "{},{},{},{}".format(user,item,round(s,4),self.titles.ix[item][0])
            
    def load_titles(self,path):
        self.titles = pd.read_csv(path,names=[self.DOC_ID,self.TITLE],index_col=0)


