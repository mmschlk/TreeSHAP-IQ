from typing import Union

import numpy as np
import itertools

try:
    from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
except ImportError:
    pass


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
