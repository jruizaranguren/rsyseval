import pandas as pd
import numpy as np

def load_frame(path='ratings.csv'):
    pass

def split_folds(frame,nfolds = 5):
    pass

def get_metrics():
    pass

def get_algorithms():
    pass

def write_to_csv(evals):
    pass

def process(path='ratings.csv'):
    folds = split_folds(load_frame(path))

    fmetrics = get_metrics()

    evals = []
    for algorithm in get_algorithms():
        result = algorithm(folds)
        evals.append([metric(result) for metric in fmetrics])

    write_to_csv(evals)



    
