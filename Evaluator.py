import pandas as pd
import numpy as np
from Fold import *
from sklearn import cross_validation

class Evaluator():
    DOC_ID = 'docId'
    USER_ID = 'userId'
    RATING = 'rating'

    def __init__(self,path='data/ratings.csv'):
        self.load_ratings(path)
        self.kfold()

    def load_ratings(self,path):
        self.r = pd.read_csv(path,names=[self.USER_ID,self.DOC_ID,self.RATING])
        self.ratings = self.r.pivot(self.USER_ID,self.DOC_ID,self.RATING)
        self.user_means = self.r.mean(axis=1)

    def normalized_ratings(self):
        return (self.ratings.T - self.user_means).T

    def kfold(self,n_folds = 5):
        skf = cross_validation.StratifiedKFold(self.r.userId,n_folds)
        self.folds = [Fold(train_ix,test_ix) for train_ix, test_ix in skf]

    def get_metrics(self):
        pass

    def get_algorithms(self):
        pass

    def write_to_csv(self,evals):
        pass

    def eval(self):
        folds = split_folds(load_frame(path))

        fmetrics = get_metrics()

        evals = []
        for algorithm in get_algorithms():
            result = algorithm(folds)
            evals.append([metric(result) for metric in fmetrics])

        write_to_csv(evals)



    
