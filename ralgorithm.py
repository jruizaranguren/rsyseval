from rstructures import *

def global_mean(folds,ratings):
    """Makes predictions using Global mean rating. Does not make recommendation list."""
    for fold in folds:
        prediction = ratings.ix[fold.test_ix]
        prediction.rating = ratings.ix[fold.train_ix].rating.mean()
        yield Recexec(prediction,None)
        #prediction = test_ix  fold   train_ix,test_ix
    


