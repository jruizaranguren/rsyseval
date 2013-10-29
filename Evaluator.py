import pandas as pd
import numpy as np

class Evaluator():
    DOC_ID = 'docId'
    USER_ID = 'userId'
    RATING = 'rating'

    def __init__(self,path='data/ratings.csv'):
        self.load_ratings(path)

    def load_ratings(self,path):
        r = pd.read_csv(path,names=[self.USER_ID,self.DOC_ID,self.RATING])
        self.ratings = r.pivot(self.USER_ID,self.DOC_ID,self.RATING)
        self.user_means = r.mean(axis=1)

    def normalized_ratings(self):
        return (self.ratings.T - self.user_means).T

    def split_folds(frame,nfolds = 5):
        pass

    def get_metrics():
        pass

    def get_algorithms():
        pass

    def write_to_csv(evals):
        pass

    def eval():
        folds = split_folds(load_frame(path))

        fmetrics = get_metrics()

        evals = []
        for algorithm in get_algorithms():
            result = algorithm(folds)
            evals.append([metric(result) for metric in fmetrics])

        write_to_csv(evals)



    
