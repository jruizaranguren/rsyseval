from rstructures import *
from numpy import NaN

def global_mean(folds,ratings,users):
    """Makes predictions using Global mean rating. Does not make recommendation list."""
    for fold in folds:
        prediction = ratings.ix[fold.test_ix]
        prediction.rating = ratings.ix[fold.train_ix].rating.mean()
        yield Recexec(prediction,None)
        
def popularity(folds,ratings,users):
    """Makes recommendations based on item popularity, i.e. number of ratings. Does not make score predictions."""
    for fold in folds:
        popularity = ratings.ix[fold.train_ix].docId.value_counts()
        test = ratings.ix[fold.test_ix].pivot('userId','docId','rating')
        test[test>0] = 1
        test = test * popularity

        recommendations = {user:test.ix[user].order(ascending=False)[0:10].index for user in users if user in test.index}
        recommendations.update({user:NaN for user in users if user not in test.index})
      
        yield Recexec(None,recommendations)
        
    
         