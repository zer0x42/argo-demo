import pickle
import pandas
import numpy as np
from definitions import ROOT_DIR


class Endpoint(object):
    def __init__(self):
        self.dummy_model = pickle.load(open(ROOT_DIR + "/serialized/dummy_model.pkl", 'rb'))

    def predict(self, x, feature_names):
        prediction = self.dummy_model.predict(x)

        return prediction