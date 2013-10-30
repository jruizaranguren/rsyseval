class Fold(object):
    """Stores the train and test index of a fold for a data set"""

    def __init__(self,train_ix,test_ix):
        self.train_ix = train_ix
        self.test_ix = test_ix


