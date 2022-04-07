################################################################################
#
#
#
#       _model.py
#
#       Description: This file contains a Model class used for keeping track 
#           of the model type, hyperparameters, features, and any additional 
#           pre processing required for a machine learning model.
#
#       Class:
#
#           Model
#
#       Class Fields:
#
#           Field_Name
#
#       Class Methods:
#
#           __init__(self)
#
#
################################################################################

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression, TweedieRegressor
from sklearn.preprocessing import PolynomialFeatures

from sklearn.metrics import mean_squared_error, r2_score

################################################################################

class Model:

    ################################################################################

    def __init__(self, model, train, features, target):
        self.model = model
        self.features = features
        self.target = target

        self.model.fit(train[self.features], train[self.target])

    ################################################################################

    def make_predictions(self, df):
        return self.model.predict(df[self.features])    