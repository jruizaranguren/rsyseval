import pandas as pd
import numpy as np
from sklearn import cross_validation

class Evaluator():
    DOC_ID = 'docId'
    USER_ID = 'userId'
    RATING = 'rating'

    def __init__(self,path='data/ratings.csv'):
        self.load_ratings(path)

    def load_ratings(self,path):
        self.r = pd.read_csv(path,names=[self.USER_ID,self.DOC_ID,self.RATING])
        self.ratings = r.pivot(self.USER_ID,self.DOC_ID,self.RATING)
        self.user_means = self.r.mean(axis=1)

    def normalized_ratings(self):
        return (self.ratings.T - self.user_means).T

    def split_folds(self,n_folds = 5):
        skf = cross_validation.StratifiedKFold(self.r,self.r.index,n_folds)

        #for train_index, test_index in skf:
	       # print("TRAIN:", train_index, "TEST:", test_index)
	       # X_train, X_test = r[train_index], r[test_index]

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



    
