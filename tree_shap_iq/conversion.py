import copy
import sys
from dataclasses import dataclass
from typing import Union, Optional
from copy import deepcopy

import numpy as np
from scipy.special import binom, logit, expit

#from tree_shap_iq.shap_code import XGBTreeModelLoader

import scipy.special
import sys
import struct
from packaging import version


def safe_isinstance(obj, class_path_str):
    # Copied from shap repo
    """
    Acts as a safe version of isinstance without having to explicitly
    import packages which may not exist in the user's environment.

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
class EdgeTree:
    """A dataclass for storing the information of a parsed tree."""
    parents: np.ndarray[int]
    ancestors: np.ndarray[int]
    ancestor_nodes:  dict[int, np.ndarray[int]]
    p_e_values: np.ndarray[float]
    p_e_storages: np.ndarray[float]
    split_weights: np.ndarray[float]
    empty_predictions: np.ndarray[float]
    edge_heights: np.ndarray[int]
    max_depth: int
    last_feature_node_in_path: np.ndarray[int]
    interaction_height_store: dict[int, np.ndarray[int]]

    def __getitem__(self, item):
        return getattr(self, item)


@dataclass
class TreeModel:
    """A dataclass for storing the information of a tree model."""
    children_left: np.ndarray[int]
    children_right: np.ndarray[int]
    features: np.ndarray[int]
    thresholds: np.ndarray[float]
    values: np.ndarray[float]
    node_sample_weight: np.ndarray[float]
    empty_prediction: Optional[float] = None

    def __getitem__(self, item):
        return getattr(self, item)


def convert_tree_estimator(
        tree_model,
        scaling: float = 1.,
        class_label: int = None,
        empty_prediction: float = None,
        output_type: str = "raw"
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
        if safe_isinstance(tree_model, "sklearn.tree.DecisionTreeClassifier") and class_label is None:
            class_label = 1
        tree_values = tree_model.tree_.value.copy() * scaling
        if class_label is not None:
            # turn node values into probabilities
            if len(tree_values.shape) == 3:
                tree_values = tree_values / np.sum(tree_values, axis=2, keepdims=True)
                tree_values = tree_values[:, 0, class_label]
            else:
                tree_values = tree_values / np.sum(tree_values, axis=1, keepdims=True)
                tree_values = tree_values[:, class_label]
            if output_type == "probability":
                tree_values = tree_values
        tree_values = tree_values.flatten()
        return TreeModel(
            children_left=tree_model.tree_.children_left,
            children_right=tree_model.tree_.children_right,
            features=tree_model.tree_.feature,
            thresholds=tree_model.tree_.threshold,
            values=tree_values,
            node_sample_weight=tree_model.tree_.weighted_n_node_samples,
            empty_prediction=empty_prediction
        )

    if safe_isinstance(tree_model, "sklearn.ensemble.GradientBoostingClassifier"):
        learning_rate = tree_model.learning_rate
        if safe_isinstance(tree_model.init_, [
            "sklearn.ensemble.LogOddsEstimator",
            "sklearn.ensemble.gradient_boosting.LogOddsEstimator"
        ]):
            empty_prediction = deepcopy(tree_model.init_.prior[class_label])
            # TODO not validated
        elif safe_isinstance(tree_model.init_, "sklearn.dummy.DummyClassifier"):
            empty_prediction = logit(tree_model.init_.class_prior_[1])
        else:
            assert False, "Unsupported init model type: " + str(type(tree_model.init_))
        return [
            # GradientBoostedClassifier contains DecisionTreeRegressor as base_estimators
            convert_tree_estimator(
                tree, scaling=learning_rate, class_label=None, empty_prediction=empty_prediction
            ) for tree in tree_model.estimators_[:, 0]
        ]

    if safe_isinstance(tree_model, "sklearn.ensemble.GradientBoostingRegressor"):
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
            convert_tree_estimator(tree, scaling=learning_rate, class_label=None, empty_prediction=empty_prediction)
            for tree in tree_model.estimators_[:, 0]
        ]

    if safe_isinstance(tree_model, "sklearn.ensemble.RandomForestRegressor") or \
            safe_isinstance(tree_model, "sklearn.ensemble.RandomForestClassifier"):
        scaling = 1.0 / len(tree_model.estimators_)
        output_type = None if not safe_isinstance(tree_model, "sklearn.ensemble.RandomForestClassifier") else output_type
        return [
            convert_tree_estimator(tree, scaling=scaling, class_label=class_label, output_type=output_type)
            for tree in tree_model.estimators_
        ]

    if safe_isinstance(tree_model, "xgboost.sklearn.XGBClassifier"):
        import xgboost
        model = tree_model.get_booster()
        xgb_loader = XGBTreeModelLoader(model)
        trees = xgb_loader.get_trees()
        base_offset = xgb_loader.base_score
        return copy.deepcopy(trees)

    if safe_isinstance(tree_model, "xgboost.sklearn.XGBRegressor"):
        import xgboost
        model = tree_model.get_booster()
        xgb_loader = XGBTreeModelLoader(model)
        trees = xgb_loader.get_trees()
        base_offset = xgb_loader.base_score
        return copy.deepcopy(trees)

    raise NotImplementedError(f"Conversion of {type(tree_model)} is not supported.")


def extract_edge_information_from_tree(
        children_left: np.ndarray[int],
        children_right: np.ndarray[int],
        features: np.ndarray[int],
        node_sample_weight: np.ndarray[float],
        values: np.ndarray[float],
        n_nodes: int,
        n_features: int,
        max_interaction: int,
        subset_updates_pos_store: dict[int, np.ndarray[int]]
):
    """Extracts edge information recursively from the tree information.

    Parses the tree recursively to create an edge-based representation of the tree. It
    precalculates the p_e and p_e_ancestors of the interaction subsets up to order
    'max_interaction'.

    Args:
        children_left (np.ndarray[int]): The left children of each node. Leaf nodes are denoted
            with -1.
        children_right (np.ndarray[int]): The right children of each node. Leaf nodes are denoted
            with -1.
        features (np.ndarray[int]): The feature used for splitting at each node. Leaf nodes have
            the value -2.
        node_sample_weight (np.ndarray[float]): The sample weights of the tree.
        values (np.ndarray[float]): The output values at the leaf values of the tree.
        n_nodes (int): The number of nodes in the tree.
        n_features (int): The number of features of the dataset.
        max_interaction (int, optional): The maximum interaction order to be computed. An
            interaction order of 1 corresponds to the Shapley value. Any value higher than 1
            computes the Shapley interactions values up to that order. Defaults to 1 (i.e. SV).
        subset_updates_pos_store (dict[int, np.ndarray[int]]): A dictionary containing the
            interaction subsets for each feature given an interaction order.

    Returns:
        EdgeTree: A dataclass containing the edge information of the tree.
    """
    # variables to be filled with recursive function
    parents = np.full(n_nodes, -1, dtype=int)
    ancestors: np.ndarray[int] = np.full(n_nodes, -1, dtype=int)

    ancestor_nodes: dict[int, np.ndarray[int]] = {}

    p_e_values: np.ndarray[float] = np.ones(n_nodes, dtype=float)
    p_e_storages: np.ndarray[float] = np.ones((n_nodes, n_features), dtype=float)
    split_weights: np.ndarray[float] = np.ones(n_nodes, dtype=float)
    empty_predictions: np.ndarray[float] = np.zeros(n_nodes, dtype=float)
    edge_heights: np.ndarray[int] = np.full_like(children_left, -1, dtype=int)
    max_depth: list[int] = [0]
    interaction_height_store = {i: np.zeros((n_nodes, int(binom(n_features, i))), dtype=int) for
                                i in range(1, max_interaction + 1)}

    features_last_seen_in_tree: dict[int, int] = {}

    last_feature_node_in_path: np.ndarray[bool] = np.full_like(children_left, False, dtype=bool)

    def recursive_search(
            node_id: int = 0,
            depth: int = 0,
            prod_weight: float = 1.,
            seen_features: np.ndarray[int] = None
    ):
        """Traverses the tree recursively and collects all relevant information.

        Args:
            node_id (int): The current node id.
            depth (int): The depth of the current node.
            prod_weight (float): The product of the node weights on the path to the current
                node.
            seen_features (np.ndarray[int]): The features seen on the path to the current node. Maps the
                feature id to the node id where the feature was last seen on the way.

        Returns:
            int: The edge height of the current node.
        """
        # if root node, initialize seen_features and p_e_storage
        if seen_features is None:
            seen_features: np.ndarray[int] = np.full(n_features, -1,
                                                     dtype=int)  # maps feature_id to ancestor node_id

        # update the maximum depth of the tree
        max_depth[0] = max(max_depth[0], depth)

        # set the parents of the children nodes
        left_child, right_child = children_left[node_id], children_right[node_id]
        is_leaf = left_child == -1
        if not is_leaf:
            parents[left_child], parents[right_child] = node_id, node_id
            features_last_seen_in_tree[features[node_id]] = node_id

        # if root_node, step into the tree and end recursion
        if node_id == 0:
            edge_heights_left = recursive_search(left_child, depth + 1, prod_weight,
                                                 seen_features.copy())
            edge_heights_right = recursive_search(right_child, depth + 1, prod_weight,
                                                  seen_features.copy())
            edge_heights[node_id] = max(edge_heights_left, edge_heights_right)
            return edge_heights[node_id]  # final return ending the recursion

        # node is not root node follow the path and compute weights

        ancestor_nodes[node_id] = seen_features.copy()

        # get the feature id of the current node
        feature_id = features[parents[node_id]]

        # Assume it is the last occurrence of feature
        last_feature_node_in_path[node_id] = True

        # compute prod_weight with node samples
        n_sample = node_sample_weight[node_id]
        n_parent = node_sample_weight[parents[node_id]]
        weight = n_sample / n_parent
        split_weights[node_id] = weight
        prod_weight *= weight

        # calculate the p_e value of the current node
        p_e = 1 / weight

        # copy parent height information
        for order in range(1, max_interaction + 1):
            interaction_height_store[order][node_id] = interaction_height_store[order][
                parents[node_id]].copy()
        # correct if feature was seen before
        if seen_features[feature_id] > -1:  # feature has been seen before in the path
            ancestor_id = seen_features[feature_id]  # get ancestor node with same feature
            ancestors[node_id] = ancestor_id  # store ancestor node
            last_feature_node_in_path[ancestor_id] = False  # correct previous assumption
            p_e *= p_e_values[ancestor_id]  # add ancestor weight to p_e
        else:
            for order in range(1, max_interaction + 1):
                interaction_height_store[order][node_id][
                    subset_updates_pos_store[order][feature_id]] += 1

        # store the p_e value of the current node
        p_e_values[node_id] = p_e
        p_e_storages[node_id] = p_e_storages[parents[node_id]].copy()
        p_e_storages[node_id][feature_id] = p_e

        # update seen features with current node
        seen_features[feature_id] = node_id

        # update the edge heights
        if not is_leaf:  # if node is not a leaf, continue recursion
            edge_heights_left = recursive_search(left_child, depth + 1, prod_weight,
                                                 seen_features.copy())
            edge_heights_right = recursive_search(right_child, depth + 1, prod_weight,
                                                  seen_features.copy())
            edge_heights[node_id] = max(edge_heights_left, edge_heights_right)
        else:  # if node is a leaf, end recursion
            edge_heights[node_id] = np.sum(seen_features > -1)
            empty_predictions[node_id] = prod_weight * values[node_id]
        return edge_heights[node_id]  # return upwards in the recursion

    _ = recursive_search()
    edge_tree = EdgeTree(
        parents=parents,
        ancestors=ancestors,
        ancestor_nodes=ancestor_nodes,
        p_e_values=p_e_values,
        p_e_storages=p_e_storages,
        split_weights=split_weights,
        empty_predictions=empty_predictions,
        edge_heights=edge_heights,
        max_depth=max_depth[0],
        last_feature_node_in_path=last_feature_node_in_path,
        interaction_height_store=interaction_height_store
    )
    return edge_tree

class XGBTreeModelLoader(object):
    """ This loads an XGBoost model directly from a raw memory dump.

    We can't use the JSON dump because due to numerical precision issues those
    tree can actually be wrong when feature values land almost on a threshold.
    """

    def __init__(self, xgb_model):
        # new in XGBoost 1.1, 'binf' is appended to the buffer
        self.buf = xgb_model.save_raw()
        if self.buf.startswith(b'binf'):
            self.buf = self.buf[4:]
        self.pos = 0

        # load the model parameters
        self.base_score = self.read('f')
        self.num_feature = self.read('I')
        self.num_class = self.read('i')
        self.contain_extra_attrs = self.read('i')
        self.contain_eval_metrics = self.read('i')
        self.read_arr('i', 29)  # reserved
        self.name_obj_len = self.read('Q')
        self.name_obj = self.read_str(self.name_obj_len)
        self.name_gbm_len = self.read('Q')
        self.name_gbm = self.read_str(self.name_gbm_len)

        # new in XGBoost 1.0 is that the base_score is saved untransformed (https://github.com/dmlc/xgboost/pull/5101)
        # so we have to transform it depending on the objective
        import xgboost
        if version.parse(xgboost.__version__).major >= 1:
            if self.name_obj in ["binary:logistic", "reg:logistic"]:
                self.base_score = scipy.special.logit(self.base_score)  # pylint: disable=no-member

        assert self.name_gbm == "gbtree", "Only the 'gbtree' model type is supported, not '%s'!" % self.name_gbm

        # load the gbtree specific parameters
        self.num_trees = self.read('i')
        self.num_roots = self.read('i')
        self.num_feature = self.read('i')
        self.pad_32bit = self.read('i')
        self.num_pbuffer_deprecated = self.read('Q')
        self.num_output_group = self.read('i')
        self.size_leaf_vector = self.read('i')
        self.read_arr('i', 32)  # reserved

        # load each tree
        self.num_roots = np.zeros(self.num_trees, dtype=np.int32)
        self.num_nodes = np.zeros(self.num_trees, dtype=np.int32)
        self.num_deleted = np.zeros(self.num_trees, dtype=np.int32)
        self.max_depth = np.zeros(self.num_trees, dtype=np.int32)
        self.num_feature = np.zeros(self.num_trees, dtype=np.int32)
        self.size_leaf_vector = np.zeros(self.num_trees, dtype=np.int32)
        self.node_parents = []
        self.node_cleft = []
        self.node_cright = []
        self.node_sindex = []
        self.node_info = []
        self.loss_chg = []
        self.sum_hess = []
        self.base_weight = []
        self.leaf_child_cnt = []
        for i in range(self.num_trees):

            # load the per-tree params
            self.num_roots[i] = self.read('i')
            self.num_nodes[i] = self.read('i')
            self.num_deleted[i] = self.read('i')
            self.max_depth[i] = self.read('i')
            self.num_feature[i] = self.read('i')
            self.size_leaf_vector[i] = self.read('i')

            # load the nodes
            self.read_arr('i', 31)  # reserved
            self.node_parents.append(np.zeros(self.num_nodes[i], dtype=np.int32))
            self.node_cleft.append(np.zeros(self.num_nodes[i], dtype=np.int32))
            self.node_cright.append(np.zeros(self.num_nodes[i], dtype=np.int32))
            self.node_sindex.append(np.zeros(self.num_nodes[i], dtype=np.uint32))
            self.node_info.append(np.zeros(self.num_nodes[i], dtype=np.float32))
            for j in range(self.num_nodes[i]):
                self.node_parents[-1][j] = self.read('i')
                self.node_cleft[-1][j] = self.read('i')
                self.node_cright[-1][j] = self.read('i')
                self.node_sindex[-1][j] = self.read('I')
                self.node_info[-1][j] = self.read('f')

            # load the stat nodes
            self.loss_chg.append(np.zeros(self.num_nodes[i], dtype=np.float32))
            self.sum_hess.append(np.zeros(self.num_nodes[i], dtype=np.float32))
            self.base_weight.append(np.zeros(self.num_nodes[i], dtype=np.float32))
            self.leaf_child_cnt.append(np.zeros(self.num_nodes[i], dtype=int))
            for j in range(self.num_nodes[i]):
                self.loss_chg[-1][j] = self.read('f')
                self.sum_hess[-1][j] = self.read('f')
                self.base_weight[-1][j] = self.read('f')
                self.leaf_child_cnt[-1][j] = self.read('i')

    def get_trees(self, data=None, data_missing=None):
        shape = (self.num_trees, self.num_nodes.max())
        self.children_default = np.zeros(shape, dtype=int)
        self.features = np.zeros(shape, dtype=int)
        self.thresholds = np.zeros(shape, dtype=np.float32)
        self.values = np.zeros((shape[0], shape[1], 1), dtype=np.float32)
        trees = []
        for i in range(self.num_trees):
            for j in range(self.num_nodes[i]):
                if np.right_shift(self.node_sindex[i][j], np.uint32(31)) != 0:
                    self.children_default[i, j] = self.node_cleft[i][j]
                else:
                    self.children_default[i, j] = self.node_cright[i][j]
                self.features[i, j] = self.node_sindex[i][j] & (
                        (np.uint32(1) << np.uint32(31)) - np.uint32(1))
                if self.node_cleft[i][j] >= 0:
                    # Xgboost uses < for thresholds where shap uses <=
                    # Move the threshold down by the smallest possible increment
                    self.thresholds[i, j] = np.nextafter(self.node_info[i][j], - np.float32(np.inf))
                else:
                    self.values[i, j] = self.node_info[i][j]

            l = len(self.node_cleft[i])
            trees.append(TreeModel(
                children_left=self.node_cleft[i],
                children_right=self.node_cright[i],
                features=self.features[i, :l],
                thresholds=self.thresholds[i, :l],
                values=self.values[i, :l].flatten(),
                node_sample_weight=self.sum_hess[i]
            ))
        return trees

    def read(self, dtype):
        size = struct.calcsize(dtype)
        val = struct.unpack(dtype, self.buf[self.pos:self.pos + size])[0]
        self.pos += size
        return val

    def read_arr(self, dtype, n_items):
        format = "%d%s" % (n_items, dtype)
        size = struct.calcsize(format)
        val = struct.unpack(format, self.buf[self.pos:self.pos + size])[0]
        self.pos += size
        return val

    def read_str(self, size):
        val = self.buf[self.pos:self.pos + size].decode('utf-8')
        self.pos += size
        return val

    def print_info(self):

        print("--- global parmeters ---")
        print("base_score =", self.base_score)
        print("num_feature =", self.num_feature)
        print("num_class =", self.num_class)
        print("contain_extra_attrs =", self.contain_extra_attrs)
        print("contain_eval_metrics =", self.contain_eval_metrics)
        print("name_obj_len =", self.name_obj_len)
        print("name_obj =", self.name_obj)
        print("name_gbm_len =", self.name_gbm_len)
        print("name_gbm =", self.name_gbm)
        print()
        print("--- gbtree specific parameters ---")
        print("num_trees =", self.num_trees)
        print("num_roots =", self.num_roots)
        print("num_feature =", self.num_feature)
        print("pad_32bit =", self.pad_32bit)
        print("num_pbuffer_deprecated =", self.num_pbuffer_deprecated)
        print("num_output_group =", self.num_output_group)
        print("size_leaf_vector =", self.size_leaf_vector)
