from model.classical_methods.ncm import NCMMethod
import os.path as ops
import pickle

class NaiveBayesMethod(NCMMethod):
    def __init__(self, args, is_regression):
        super().__init__(args, is_regression)
        
    def construct_model(self, model_config = None):
        from sklearn.naive_bayes import GaussianNB
        self.model = GaussianNB()
    