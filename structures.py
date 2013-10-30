class Fold():
    """Stores the train and test index of a fold for a data set"""
    def __init__(self,train_ix,test_ix):
        self.train_ix = train_ix
        self.test_ix = test_ix

class Recexec():
    """Stores the results of the execution of a recommender algorithm, with predictions per user/item and recommendations list"""
    def __init__(self,predictions,recommendations):
        self.predictions = predictions
        self.recommendations = recommendations


