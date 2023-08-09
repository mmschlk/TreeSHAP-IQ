import sys
from dataclasses import dataclass
from typing import Union
from copy import deepcopy

import numpy as np


def safe_isinstance(obj, class_path_str):
    # Copied from shap repo
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
    empty_prediction: float = None

    def __getitem__(self, item):
        return getattr(self, item)


def convert_tree_estimator(
        tree_model,
        scaling: float = 1.,
        class_label: int = None,
        empty_prediction: float = None
) -> Union[TreeModel, list[TreeModel]]:
    """Converts a tree estimator to a dictionary or a list of dictionaries.

    Args:
        tree_model: The tree estimator to be converted.
        scaling (float): The scaling factor to be applied to the leaf values. Must be in range
            (0, inf+]. Defaults to 1.
        class_label (int): The class label to be explained. Only applicable for classification
            problems. Defaults to None.
        empty_prediction (float): The prediction of the tree model when the input is empty.
            Defaults to None.

    Returns:
        Union[TreeModel, list[TreeModel]]: The converted tree estimator as either a mapping of node
            information or a list of mappings for each tree in an ensemble. The dictionary will
            contain the following mappings.

    """
    if safe_isinstance(tree_model, "sklearn.tree.DecisionTreeRegressor") or \
            safe_isinstance(tree_model, "sklearn.tree.DecisionTreeClassifier"):
        tree_values = tree_model.tree_.value.reshape(-1, 1).copy() * scaling
        if class_label is not None:
            tree_values = tree_values[:, class_label]
        return TreeModel(
            children_left=tree_model.tree_.children_left,
            children_right=tree_model.tree_.children_right,
            features=tree_model.tree_.feature,
            thresholds=tree_model.tree_.threshold,
            values=tree_values,
            node_sample_weight=tree_model.tree_.weighted_n_node_samples,
            empty_prediction=empty_prediction
        )

    if safe_isinstance(tree_model, "sklearn.ensemble.GradientBoostingRegressor") or \
            safe_isinstance(tree_model, "sklearn.ensemble.GradientBoostingClassifier"):
        learning_rate = tree_model.learning_rate
        if empty_prediction is None:
            if safe_isinstance(tree_model.init_, ["sklearn.ensemble.MeanEstimator", "sklearn.ensemble.gradient_boosting.MeanEstimator"]):
                empty_prediction = deepcopy(tree_model.init_.mean)
            elif safe_isinstance(tree_model.init_, ["sklearn.ensemble.QuantileEstimator", "sklearn.ensemble.gradient_boosting.QuantileEstimator"]):
                empty_prediction = deepcopy(tree_model.init_.quantile)
            elif safe_isinstance(tree_model.init_, "sklearn.dummy.DummyRegressor"):
                empty_prediction = deepcopy(tree_model.init_.constant_[0])
            else:
                assert False, "Unsupported init model type: " + str(type(tree_model.init_))
            empty_prediction /= len(tree_model.estimators_)  # we distribute the empty prediction to all trees equally
        return [
            # GradientBoostedClassifier contains DecisionTreeRegressor as base_estimators
            convert_tree_estimator(tree, scaling=learning_rate, class_label=None, empty_prediction=None)
            for tree in tree_model.estimators_[:, 0]
        ]
    if safe_isinstance(tree_model, "sklearn.ensemble.RandomForestRegressor") or \
            safe_isinstance(tree_model, "sklearn.ensemble.RandomForestClassifier"):
        scaling = 1.0 / len(tree_model.estimators_)
        return [
            convert_tree_estimator(tree, scaling=scaling, class_label=class_label)
            for tree in tree_model.estimators_
        ]

    # TODO add support for classification by providing class label
    raise NotImplementedError(f"Conversion of {type(tree_model)} is not supported.")
