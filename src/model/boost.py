import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class CustomBoostingClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, estimators):
        self.estimators = estimators
        self.alphas = []

    def fit(self, x, y):
        sample_weights = np.ones(len(y)) / len(y)
        self.classes_ = np.unique(y) if y.ndim == 1 else np.unique(y.toarray())

        for estimator in self.estimators:
            model = estimator.fit(x, y, sample_weight=sample_weights)
            y_pred = model.predict(x)

            error = np.sum(sample_weights * (y_pred != y)) / np.sum(sample_weights)

            alpha = 0.5 * np.log((1 - error) / max(error, 1e-10))
            self.alphas.append(alpha)

            sample_weights *= np.exp(-alpha * y * y_pred)
            sample_weights /= np.sum(sample_weights)

        return self

    def predict(self, x):
        final_preds = np.zeros(x.shape[0])

        for alpha, model in zip(self.alphas, self.estimators):
            final_preds += alpha * model.predict(x)

        return np.sign(final_preds)
