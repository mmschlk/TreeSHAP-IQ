from typing import Union

import numpy as np
from collections import namedtuple
try:
    from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
except ImportError:
    pass


tree_model = namedtuple("Tree", ["children_left", "children_right", "features", "thresholds", "leaf_predictions", "sample_weights"])


def convert_tree(tree: Union[DecisionTreeRegressor, DecisionTreeClassifier]) -> tree_model:
    """Convert sklearn tree to a tree_model namedtuple.

    Args:
        tree (Union[DecisionTreeRegressor, DecisionTreeClassifier]): sklearn tree object.

    Returns:
        tree_model: namedtuple with the following fields:
            children_left (np.ndarray): array of left children indices.
            children_right (np.ndarray): array of right children indices.
            features (np.ndarray): array of feature indices.
            thresholds (np.ndarray): array of feature thresholds.
            leaf_predictions (np.ndarray): array of leaf values.
            sample_weights (np.ndarray): array of sample weights.
    """
    leaf_predictions = tree.tree_.value.squeeze(axis=1).squeeze()
    sample_weights = tree.tree_.weighted_n_node_samples
    return tree_model(
        children_left=tree.tree_.children_left,
        children_right=tree.tree_.children_right,
        features=tree.tree_.feature,
        thresholds=tree.tree_.threshold,
        leaf_predictions=leaf_predictions,
        sample_weights=sample_weights,
    )
