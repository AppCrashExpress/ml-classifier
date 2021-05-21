import numpy as np
from sklearn.preprocessing import LabelEncoder

class CART:
    class _DataPointer:
        def __init__(self, column, value):
            self.column = column
            self.value  = value

    class _Node:
        def __init__(self, false_node, true_node, pointer):
            self.false_node = false_node
            self.true_node  = true_node
            self.pointer    = pointer

    class _Leaf:
        def __init__(self, prediction):
            self.prediction = prediction


    def __init__(self):
        self._root_node = None

    def train(self, feats, labels):
        self._root_node = self._build_tree(feats, labels)
        
    def predict(self, feats):
        if self._root_node is None:
            raise RuntimeError("Model is untrained")

        predicts = []

        for feat_list in feats:
            predicts.append(self._match(feat_list, self._root_node))

        return predicts

    def _build_tree(self, feats, labels):
        info_gain, pointer = self._find_best_split(feats, labels)

        if info_gain == 0:
            return self._Leaf(self._most_common_class(labels))

        false_feats, false_labels, true_feats, true_labels = self._partite(feats, labels, pointer)

        false_node = self._build_tree(false_feats, false_labels)
        true_node  = self._build_tree(true_feats,  true_labels)
        
        return self._Node(false_node, true_node, pointer)

    def _find_best_split(self, feats, labels):
        max_info_gain = 0
        pointer = None
        
        current_impurity = self._gini(labels)

        for col_i, col in enumerate(feats.T):
            unique_vals = np.unique(col)

            for u_val in unique_vals :
                split_point = self._DataPointer(col_i, u_val)
                _, false_labels, _, true_labels = self._partite(feats, labels, split_point)

                if len(false_labels) == 0 or len(true_labels) == 0:
                    continue

                info_gain = self._info_gain(false_labels, true_labels, current_impurity)

                if info_gain > max_info_gain:
                    max_info_gain = info_gain
                    pointer = split_point

        return max_info_gain, pointer

    def _partite(self, feats, labels, split_point):
        column = split_point.column
        value  = split_point.value

        false_indecies = feats[:, column] <  value
        true_indecies  = feats[:, column] >= value

        false_feats = feats[false_indecies, :]
        true_feats  = feats[true_indecies, :]

        false_labels = labels[false_indecies]
        true_labels  = labels[true_indecies]

        return false_feats, false_labels, true_feats, true_labels
        
    def _match(self, feats, node):
        if isinstance(node, self._Leaf):
            return node.prediction

        to_true = feats[node.pointer.column] >= node.pointer.value

        if to_true:
            return self._match(feats, node.true_node)
        else:
            return self._match(feats, node.false_node)

    def _info_gain(self, false_labels, true_labels, parent_impurity):
        false_c = len(false_labels)
        true_c  = len(true_labels)
        total = true_c + false_c

        false_gini = self._gini(false_labels) * (false_c / total)
        true_gini  = self._gini(true_labels) * (true_c / total)

        return parent_impurity - false_gini - true_gini

    def _gini(self, classes):
        probs = self._count_classes(classes) / len(classes)

        return 1 - np.sum(probs ** 2)

    def _count_classes(self, classes):
        return np.bincount(classes)

    def _most_common_class(self, labels):
        return np.bincount(labels).argmax()

class RandomForest:
    def __init__(self, n_trees):
        self._trees = [CART() for _ in range(n_trees)]

    def train(self, feats, labels):
        sample_size = len(feats)

        for tree in self._trees:
            sorted_idxs = range(sample_size)
            sample_idxs = np.random.choice(sorted_idxs, sample_size, replace=True)
            feats_sample = feats[sample_idxs, :]
            labels_sample = labels[sample_idxs]
            tree.train(feats_sample, labels_sample)

    def predict(self, feats):
        def find_most_common(row):
            return np.bincount(row).argmax()

        predicts = []
        for tree in self._trees:
            predicts.append(tree.predict(feats))

        votes_per_sample = np.array(predicts).T
        votes = np.apply_along_axis(find_most_common, 1, votes_per_sample)

        return votes.flatten()
