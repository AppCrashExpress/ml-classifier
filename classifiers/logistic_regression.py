import numpy as np
from sklearn.preprocessing import OneHotEncoder

class LogisticRegression:
    def __init__(self, lr=0.01, max_epoch=100000):
        self._lr = lr
        self._max_epoch = max_epoch
        self._feat_count = 0
        self._weights = None
    
    # Will only accept two numerical classes
    # I am so done
    def train(self, feats, labels):
        if feats.ndim == 1:
            feats = feats[np.newaxis, :]

        if self._weights is None:
            self._feat_count = feats.shape[1]
            self._weights = np.zeros(self._feat_count)
        elif self._feat_count != feats.shape[1]:
            err = f"Feature count does not match previous count {self._feat_count}"
            raise ValueError(err)

        weights = self._weights
        
        for _ in range(self._max_epoch):
            weighted_feats = np.dot(feats, weights)
            predicts = self._sigmoid(weighted_feats)
            grad = self._gradient(feats, predicts, labels)
            weights -= self._lr * grad

        self._weights = weights
                
    def predict(self, X):
        return self._sigmoid( np.dot(X, self._weights) ).round()

    def _sigmoid(self, weighted_feats):
        return 1 / (1 + np.exp(-weighted_feats))

    def _gradient(self, feats, p, labels):
        return np.dot(feats.T, (p - labels)) / labels.size
