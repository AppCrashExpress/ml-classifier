import numpy as np
from collections import namedtuple
from sklearn.preprocessing import LabelEncoder

class CART:
    def __init__(self, max_depth=1000):
        self._max_depth = max_depth
        self._root_node = None
        self._DataPointer = namedtuple("DataPointer", ["col", "val"])
        self._Node = namedtuple("Node", ["false_node", "true_node", "pointer"])
        self._Leaf = namedtuple("Leaf", ["prediction"])

    def train(self, feats, labels):
        self._root_node = self._build_tree(feats, labels)
        
    def predict(self, feats):
        if self._root_node is None:
            raise RuntimeError("Model is untrained")

        predicts = []

        for _, feat_list in feats.iterrows():
            predicts.append(self._match(feat_list, self._root_node))

        return predicts


    def _build_tree(self, feats, labels):
        info_gain, pointer = self._find_best_split(feats, labels)

        if info_gain == 0:
            return self._Leaf(labels.value_counts().idxmax())

        false_feats, false_labels, true_feats, true_labels = self._partite(feats, labels, pointer)

        false_node = self._build_tree(false_feats, false_labels)
        true_node  = self._build_tree(true_feats,  true_labels)
        
        return self._Node(false_node, true_node, pointer)

    def _find_best_split(self, feats, labels):
        max_info_gain = 0
        pointer = self._DataPointer(None, None)

        current_impurity = self._gini(labels)

        for col in feats:
            unique_vals = feats[col].unique()
            for u_val in unique_vals:

                split_point = self._DataPointer(col, u_val)

                _, false_labels, _, true_labels = self._partite(feats, labels, split_point)

                if len(false_labels) == 0 or len(true_labels) == 0:
                    continue

                info_gain = self._info_gain(false_labels, true_labels, current_impurity)

                if info_gain > max_info_gain:
                    max_info_gain = info_gain 
                    pointer = self._DataPointer(col, u_val)

        return max_info_gain, pointer

    def _gini(self, classes):
        probs = self._count_probabilities(classes)
        return 1 - np.sum(probs ** 2)

    def _count_probabilities(self, classes):
        encoded_classes = LabelEncoder().fit_transform(classes)
        return np.bincount(encoded_classes) / encoded_classes.size

    def _partite(self, feats, labels, pointer):
        column = pointer.col
        value  = pointer.val
        false_indecies = feats.loc[ feats[column] != value ].index
        true_indecies  = feats.loc[ feats[column] == value ].index

        false_labels = labels.loc[false_indecies]
        true_labels  = labels.loc[true_indecies]

        false_feats = feats.loc[false_indecies]
        true_feats  = feats.loc[true_indecies]

        return false_feats, false_labels, true_feats, true_labels

    def _info_gain(self, false_labels, true_labels, parent_impurity):
        false_c = len(false_labels)
        true_c  = len(true_labels)
        total = true_c + false_c

        false_gini = self._gini(false_labels) * (false_c / total)
        true_gini  = self._gini(true_labels) * (true_c / total)

        return parent_impurity - false_gini - true_gini

    def _match(self, feats, node):
        if isinstance(node, self._Leaf):
            return node.prediction

        if feats[node.pointer.col] == node.pointer.val:
            return self._match(feats, node.true_node)
        else: 
            return self._match(feats, node.false_node)
