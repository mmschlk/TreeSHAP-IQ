import sys
from dataclasses import dataclass
from typing import Union

import numpy as np


def safe_isinstance(obj, class_path_str):
    """
    Acts as a safe version of isinstance without having to explicitly
    import packages which may not exist in the users environment.

    Checks if obj is an instance of type specified by class_path_str.

    Parameters
    ----------
    obj: Any
        Some object you want to test against
    class_path_str: str or list
        A string or list of strings specifying full class paths
        Example: `sklearn.ensemble.RandomForestRegressor`

    Returns
    --------
    bool: True if isinstance is true and the package exists, False otherwise
    """
    if isinstance(class_path_str, str):
        class_path_strs = [class_path_str]
    elif isinstance(class_path_str, list) or isinstance(class_path_str, tuple):
        class_path_strs = class_path_str
    else:
        class_path_strs = ['']

    # try each module path in order
    for class_path_str in class_path_strs:
        if "." not in class_path_str:
            raise ValueError("class_path_str must be a string or list of strings specifying a full \
                module path to a class. Eg, 'sklearn.ensemble.RandomForestRegressor'")

        # Splits on last occurence of "."
        module_name, class_name = class_path_str.rsplit(".", 1)

        # here we don't check further if the model is not imported, since we shouldn't have
        # an object of that types passed to us if the model the type is from has never been
        # imported. (and we don't want to import lots of new modules for no reason)
        if module_name not in sys.modules:
            continue

        module = sys.modules[module_name]

        # Get class
        _class = getattr(module, class_name, None)

        if _class is None:
            continue

        if isinstance(obj, _class):
            return True

    return False


@dataclass
class TreeModel:
    """A dataclass for storing the information of a tree model."""
    children_left: np.ndarray[int]
    children_right: np.ndarray[int]
    features: np.ndarray[int]
    thresholds: np.ndarray[float]
    values: np.ndarray[float]
    node_sample_weight: np.ndarray[float]

    def __getitem__(self, item):
        return getattr(self, item)


def convert_tree_estimator(
        tree_model,
        scaling: float = 1.
) -> Union[TreeModel, list[TreeModel]]:
    """Converts a tree estimator to a dictionary or a list of dictionaries.

    Args:
        tree_model: The tree estimator to be converted.
        scaling (float): The scaling factor to be applied to the leaf values. Must be in range
            (0, inf+]. Defaults to 1.

    Returns:
        Union[TreeModel, list[TreeModel]]: The converted tree estimator as either a mapping of node
            information or a list of mappings for each tree in an ensemble. The dictionary will
            contain the following mappings.

    """
    if safe_isinstance(tree_model, "sklearn.tree.DecisionTreeRegressor"):
        return TreeModel(
            children_left=tree_model.tree_.children_left,
            children_right=tree_model.tree_.children_right,
            features=tree_model.tree_.feature,
            thresholds=tree_model.tree_.threshold,
            values=tree_model.tree_.value.reshape(-1, 1).copy() * scaling,
            node_sample_weight=tree_model.tree_.weighted_n_node_samples
        )

    if safe_isinstance(tree_model, "sklearn.ensemble.GradientBoostingRegressor"):
        learning_rate = tree_model.learning_rate
        return [
            convert_tree_estimator(tree, scaling=learning_rate)
            for tree in tree_model.estimators_[:, 0]
        ]

    # TODO add support for classification by providing class label
    raise NotImplementedError(f"Conversion of {type(tree_model)} is not supported.")
