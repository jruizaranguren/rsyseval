import pandas as pd
import numpy as np

class SvdCf:
    DOC_ID = 'docId'
    USER_ID = 'userId'
    RATING = 'rating'
    TITLE = 'title'
    def __init__(self,ratings_path='data/ratings.csv',titles_path='data/movie-titles.csv',fold=None,mean_type='user'):
        self.mean_type = mean_type
        self.load_ratings(ratings_path,fold)
        self.load_titles(titles_path)

    def load_ratings(self,path,fold):
        r = pd.read_csv(path,names=[self.USER_ID,self.DOC_ID,self.RATING]) if fold is None else fold
        self.ratings = r.pivot(self.USER_ID,self.DOC_ID,self.RATING)  # Users x Items

        if self.mean_type == 'user':
            self.centered = (self.ratings.T - self.ratings.mean(axis=1)).T # Users x Items
        elif self.mean_type == 'item':
            self.centered = self.ratings - self.ratings.mean(axis=0)
            pass
        elif self.mean_type == 'global':
            self.centered = self.ratings - self.load_ratings.mean().mean()
        else:
            # User-Item personalized mean
            pass
            
    def load_titles(self,path):
        self.titles = pd.read_csv(path,names=[self.DOC_ID,self.TITLE],index_col=0)
