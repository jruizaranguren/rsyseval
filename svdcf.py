import pandas as pd
import numpy as np

test_pairs=[[2048, 788],[2048, 36955],[2048, 77],[1024, 462],[1024, 393],[1024, 36955],[1024, 77],[1024, 268]]
ass_pairs = [[4035,9741],[4035,114],[4035,278],[4035,275],[4035,7443],[2836,2501],[2836,640],[2836,857],[2836,581],[2836,453],[4282,1422],[4282,8587],[4282,238],[4282,1894],[4282,788],[1222,105],[1222,854],[1222,2501],[1222,786],[1222,36657],[2624,2024],[2624,2164],[2624,36955],[2624,36658],[2624,752]]
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
            item_mean = self.ratings.mean(axis=0)
            r1 = self.ratings - item_mean
            user_mean = pd.DataFrame(r1.mean(axis=1),index=self.ratings.index)
            self.baseline = np.zeros(self.ratings.shape)
            self.baseline += item_mean
            self.baseline = (self.baseline.T + user_mean).T
            
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

#s = SvdCf(mean_type='last')

