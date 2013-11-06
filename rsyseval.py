import pandas as pd
import numpy as np
from rstructures import *
from ralgorithm import *
from sklearn import cross_validation

class Evaluator():
    """Loads evaluation experiment: data, algorithms and metrics in order to compare recommenders performance"""
    DOC_ID = 'docId'
    USER_ID = 'userId'
    RATING = 'rating'

    def __init__(self,path='data/ratings.csv',n_folds=5):
        self.load_ratings(path)
        self.kfold(n_folds)

    def load_ratings(self,path):
        self.r = pd.read_csv(path,names=[self.USER_ID,self.DOC_ID,self.RATING])
        self.ratings = self.r.pivot(self.USER_ID,self.DOC_ID,self.RATING)
        self.users = self.r.userId.unique()
        self.user_means = self.r.mean(axis=1)

    def normalized_ratings(self):
        return (self.ratings.T - self.user_means).T

    def kfold(self,n_folds):
        skf = cross_validation.StratifiedKFold(self.r.userId,n_folds)
        self.folds = [Fold(train_ix,test_ix) for train_ix, test_ix in skf]

    def set_metrics(self,metrics):
        self.metrics = metrics

    def set_algorithms(self,algorithms):
        self.algorithms = algorithms

    def write_to_csv(self,evals):
        pass

    def eval(self):
        evals = []
        for algorithm in self.algorithms:
            rec_exec = algorithm(folds,self.r,self.users)
            evals.append([metric(rec_exec) for metric in self.metrics])

        write_to_csv(evals)


    
