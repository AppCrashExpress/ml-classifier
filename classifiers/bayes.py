import numpy as np
from collections import defaultdict

class Bayes:
    def __init__(self):
        self._feats_count = defaultdict(int)
        self._feats_total = defaultdict(int)

        self._goals_count = defaultdict(int)
        self._goals_total = 0
        self._goals = None

        self._category_count = 0

    def train(self, train_data, train_goals):
        if train_data.ndim != 2:
            raise ValueError("Expected data to be two-dimensional")

        if train_goals.ndim != 1:
            raise ValueError("Expected goals to be one-dimensional")

        if train_data.shape[0] != train_goals.shape[0]:
            raise ValueError("Data row and goal counts do not match")

        if self._category_count == 0:
            self._category_count = train_data.shape[1]
        else:
            if self._category_count != train_data.shape[1]:
                raise ValueError("Number of categories in given data does not match existing number")

        for data_row, goal in zip(train_data, train_goals):
            self._goals_count[goal] += 1

            for category, data in enumerate(data_row):
                self._feats_count[(data, category, goal)] += 1
                self._feats_total[(category, goal)] += 1

        self._goals_total += train_goals.shape[0]

        if self._goals is not None:
            self._goals = np.concatenate((self._goals, train_goals))
        else:
            self._goals = train_goals

        self._goals = np.unique(self._goals)
            
    def predict(self, samples):
        if self._category_count == 0:
            raise UnboundLocalError("No training data to estimate on")

        if samples.ndim != 2:
            raise ValueError("Data should be two-dimensional")

        if samples.shape[1] != self._category_count:
            raise ValueError(f"Feature count in data should be {self._category_count}")

        categories = [0] * (samples.shape[0])

        for sample_no, sample_row in enumerate(samples):
            probability_array = np.zeros(self._goals.shape[0])

            for index, goal in enumerate(self._goals):
                goal_prob = self._calc_goal_chance(goal)
                
                feats_prob = 1
                for category, sample_val in enumerate(sample_row):
                    feats_prob *= self._calc_feat_chance(sample_val, category, goal)

                probability_array[index] = goal_prob * feats_prob

            max_chance_index = np.argmax(probability_array)
            categories[sample_no] = self._goals[max_chance_index]

        return categories



    def _calc_goal_chance(self, goal):
        return self._goals_count[goal] / self._goals_total

    def _calc_feat_chance(self, data, category, goal):
        return self._feats_count[(data, category, goal)] / self._feats_total[(category, goal)]
