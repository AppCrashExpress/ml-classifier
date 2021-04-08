import numpy as np
from numpy.linalg import norm
from collections  import Counter

class KNN:
    def __init__(self, k_count):
        self._k_count = k_count
        self._train_data = None
        self._train_goal = None

    def train(self, train_data, train_goal):
        if self._train_data is None or self._train_goal is None:
            self._train_data = train_data
            self._train_goal = train_goal
            return self

        if train_data.ndim != 2:
            raise ValueError("Training data should be two-dimensional")
        if train_goal.ndim != 1:
            raise ValueError("Training goal should be one-dimensional")

        if train_data.shape[0] != train_goal.shape[0]:
            raise ValueError("Training data rows do not match goals")

        self._train_data = np.concatenate((self._train_data, train_data))
        self._train_goal = np.concatenate((self._train_goal, train_goal))

    def predict(self, data):
        if self._train_data is None or self._train_goal is None:
            raise UnboundLocalError("No training data to estimate on")

        if data.ndim != 2:
            raise ValueError("Data should be two-dimensional")

        if data.shape[1] != self._train_data.shape[1]:
            raise ValueError(f"Feature count in data should be {self._train_data.shape[1]}")

        preds = np.zeros((data.shape[0]))

        for index, sample in enumerate(data):
            dtype = [('dist', 'float'), ('label', 'int')]
            labeled_distances = np.fromiter(((self._distance(sample, t_data), y) for t_data, y in zip(self._train_data, self._train_goal)), dtype)
            labeled_distances.sort(order='dist')
            
            closest = [pair[1] for pair in labeled_distances[:self._k_count]]
            occur_count = Counter(closest)
            
            preds[index] = occur_count.most_common(1)[0][0]

        return preds
            


    def _distance(self, feat_vec1, feat_vec2):
        if feat_vec1.shape != feat_vec2.shape:
            raise ValueError("Shapes of feature vectors do not match")
        
        return norm(feat_vec2 - feat_vec1)

