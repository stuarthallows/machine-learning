""" Imputers that impute missing values
"""
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class MostFrequentImputer(BaseEstimator, TransformerMixin):
    """ Impute most frequent value for string columns.
    """
    """ Inspired from stackoverflow.com/questions/25239958
    """
    def fit(self, X, y=None):
        self.most_frequent = pd.Series([X[c].value_counts().index[0] for c in X],
                                       index=X.columns)
        return self
    def transform(self, X, y=None):
        return X.fillna(self.most_frequent)


class DataFrameSelector(BaseEstimator, TransformerMixin):
    """ A class to select numerical or categorical columns
    """
    """ Scikit-Learn doesn't handle DataFrames yet
    """
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names]
