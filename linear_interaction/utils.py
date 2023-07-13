from typing import Union

import numpy as np
import itertools
from collections import namedtuple
try:
    from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
except ImportError:
    pass

# TODO change the namedtuple definition
tree_model = namedtuple("Tree", [
    "children_left", "children_right", "features", "thresholds",
    "sample_weights", "parents", "empty_rule_predictions", "ancestors", "edge_heights"
])


def _get_parent_array(
        children_left: np.ndarray[int],
        children_right: np.ndarray[int]
) -> np.ndarray[int]:
    """Combines the left and right children of the tree to a parent array. The parent of the
        root node is -1.

    Args:
        children_left (np.ndarray[int]): The left children of the tree. Leaf nodes are -1.
        children_right (np.ndarray[int]): The right children of the tree. Leaf nodes are -1.

    Returns:
        np.ndarray[int]: The parent array of the tree. The parent of the root node is -1.
    """
    parent_array = np.full_like(children_left, -1)
    non_leaf_indices = np.logical_or(children_left != -1, children_right != -1)
    parent_array[children_left[non_leaf_indices]] = np.where(non_leaf_indices)[0]
    parent_array[children_right[non_leaf_indices]] = np.where(non_leaf_indices)[0]
    return parent_array


def _get_conditional_sample_weights(
        sample_count: np.ndarray[int],
        parent_array: np.ndarray[int],
) -> np.ndarray[float]:
    """Get the conditional sample weights for the tree at each decision node.

    The conditional sample weights are the probabilities of going left or right at each decision
        node. The probabilities are computed by the number of instances going through each node
        divided by the number of instances going through the parent node. The conditional sample
        weights of the root node is 1.

    Args:
        sample_count (np.ndarray[int]): The count of the instances going through each node.
        parent_array (np.ndarray[int]): The parent array denoting the id of the parent node for
            each node in the tree. The parent of the root node is -1 or otherwise specified.

    Returns:
        np.ndarray[float]: The conditional sample weights of the nodes.
    """
    conditional_sample_weights = np.zeros_like(sample_count, dtype=float)
    conditional_sample_weights[0] = 1
    parent_sample_count = sample_count[parent_array[1:]]
    conditional_sample_weights[1:] = sample_count[1:] / parent_sample_count
    return conditional_sample_weights


def _recursively_copy_tree(
        children_left: np.ndarray[int],
        children_right: np.ndarray[int],
        parents: np.ndarray[int],
        features: np.ndarray[int],
        n_features: int
) -> tuple[np.ndarray[int], np.ndarray[int]]:
    """Traverse the tree and recursively copy edge information from the tree. Get the feature
        ancestor nodes for each node in the tree. An ancestor node is the last observed node that
        has the same feature as the current node in the path.The ancestor of the root node is
        -1. The ancestor nodes are found through recursion from the root node.

    Args:
        children_left (np.ndarray[int]): The left children of the tree. Leaf nodes are -1.
        children_right (np.ndarray[int]): The right children of the tree. Leaf nodes are -1.
        features (np.ndarray[int]): The feature id of each node in the tree. Leaf nodes are -2.

    Returns:
        ancestor_nodes (np.ndarray[int]): The ancestor nodes for each node in the tree.
        edge_heights (np.ndarray[int]): The edge heights for each node in the tree.
    """

    ancestor_nodes: np.ndarray[int] = np.full_like(children_left, -1, dtype=int)
    edge_heights: np.ndarray[int] = np.full_like(children_left,-1, dtype=int)

    def _recursive_search(
            node_id: int,
            seen_features: np.ndarray[bool],
            last_feature_nodes: np.ndarray[int]
    ):
        """Recursively search for the ancestor node of the current node.

        Args:
            node_id (int): The current node id.
            seen_features (np.ndarray[bool]): The boolean array denoting whether a feature has been
                seen in the path.
            last_feature_nodes (np.ndarray[int]): The last observed node that has the same feature
                as the current node in the path.

        Returns:
            edge_height (int): The edge height of the current node.
        """

        feature_id = features[parents[node_id]]
        if seen_features[feature_id]:
            ancestor_nodes[node_id] = last_feature_nodes[feature_id]
        seen_features[feature_id] = True
        last_feature_nodes[feature_id] = node_id
        if children_left[node_id] > -1: #node is not a leaf
            edge_height_left = _recursive_search(children_left[node_id], seen_features.copy(), last_feature_nodes.copy())
            edge_height_right = _recursive_search(children_right[node_id], seen_features.copy(), last_feature_nodes.copy())
            edge_heights[node_id] = max(edge_height_left, edge_height_right)
        else:  # is a leaf node edge height corresponds to the number of features seen on the way
            edge_heights[node_id] = np.sum(seen_features)
        return edge_heights[node_id]

    init_seen_features = np.zeros(n_features, dtype=bool)
    init_last_feature_nodes = np.full(n_features, -1, dtype=int)
    _recursive_search(children_left[0], init_seen_features.copy(), init_last_feature_nodes.copy())
    _recursive_search(children_right[0], init_seen_features.copy(), init_last_feature_nodes.copy())
    return ancestor_nodes, edge_heights


def convert_tree(tree: Union[DecisionTreeRegressor, DecisionTreeClassifier]) -> tree_model:
    """Convert sklearn tree to a tree_model namedtuple.

    Args:
        tree (Union[DecisionTreeRegressor, DecisionTreeClassifier]): sklearn tree object.

    Returns:
        tree_model: TODO add description
    """
    children_left: np.ndarray[int] = tree.tree_.children_left  # -1 for leaf nodes
    children_right: np.ndarray[int] = tree.tree_.children_right  # -1 for leaf nodes

    parents: np.ndarray[int] = _get_parent_array(children_left, children_right)  # -1 for root node

    features: np.ndarray[int] = tree.tree_.feature  # -2 for leaf nodes
    thresholds: np.ndarray[float] = tree.tree_.threshold  # -2 for leaf nodes

    sample_weights_tree = tree.tree_.weighted_n_node_samples

    marginal_probabilities = sample_weights_tree / np.max(sample_weights_tree) # marginal probabilities of each node
    sample_weights: np.ndarray[float] = _get_conditional_sample_weights(sample_weights_tree, parents)


    leaf_predictions = tree.tree_.value.squeeze(axis=1).squeeze()

    empty_rule_predictions = leaf_predictions * marginal_probabilities  # predictions of empty rules

    ancestors, edge_heights = _recursively_copy_tree(children_left, children_right, features)

    return tree_model(
        children_left=children_left,
        children_right=children_right,
        features=features,
        thresholds=thresholds,
        sample_weights=sample_weights,
        parents=parents,
        empty_rule_predictions=empty_rule_predictions,
        ancestors=ancestors,
        edge_heights=edge_heights,
    )



def powerset(iterable, min_size=-1, max_size=None):
    """Return a powerset of the iterable with optional size limits.

    Args:
        iterable (iterable): Iterable.
        min_size (int, optional): Minimum size of the subsets. Defaults to -1.
        max_size (int, optional): Maximum size of the subsets. Defaults to None.

    Returns:
        iterable: Powerset of the iterable.
    """
    if max_size is None and min_size > -1:
        max_size = min_size
    s = list(iterable)
    if max_size is None:
        max_size = len(s)
    else:
        max_size = min(max_size, len(s))
    return itertools.chain.from_iterable(
        itertools.combinations(s, r) for r in range(max(min_size, 0), max_size + 1))
