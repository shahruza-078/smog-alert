"""ml/baselines.py  [A]  The 'smart baseline': tomorrow will be like today.

PersistenceRegressor predicts tomorrow's PM2.5 as the value of the feature `pm25_today`
(the mean of today's hours before the cutoff). It is a scikit-learn estimator, so it works
inside a Pipeline, in cross-validation, and can be saved with joblib.

It reads the feature BY NAME, so give it a DataFrame (our features do that). Do not put a
scaler in front of it, or it would return scaled numbers.
"""
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin


class PersistenceRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, feature="pm25_today"):
        self.feature = feature

    def fit(self, X, y):
        self._check(X)
        # used only when today's value is missing
        self.fallback_ = float(np.nanmedian(np.asarray(y, dtype=float)))
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        self._check(X)
        values = X[self.feature].to_numpy(dtype=float)
        return np.where(np.isnan(values), self.fallback_, values)

    def _check(self, X):
        if not hasattr(X, "columns") or self.feature not in X.columns:
            raise ValueError(f"PersistenceRegressor needs a DataFrame with a '{self.feature}' column")
