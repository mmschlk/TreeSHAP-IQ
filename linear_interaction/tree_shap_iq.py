"""This module contains the linear TreeSHAP class, which is a reimplementation of the original TreeSHAP algorithm."""
import time

import numpy as np
from numpy.polynomial import Polynomial
from numpy.polynomial.polynomial import polydiv
from scipy.special import binom

from utils import _recursively_copy_tree, _get_parent_array, powerset


class TreeSHAPIQExplainer:

    def __init__(
            self,
            tree_model: dict,
            max_interaction_order: int = 1,
            observational: bool = True,
            background_dataset: np.ndarray = None,
            n_features: int = None,
            root_node_id: int = 0,
    ):
        """The TreeSHAPIQExplainer class. This class is a reimplementation of the original Linear
            TreeSHAP algorithm for the case of interaction order 1 (i.e. Shapley value). If the
            interaction order is higher than 1, the algorithm compute Shapley interaction values up
            to that order. The algorithm can be used for both the ´observational´ and
            ´interventional´ Shapley approach. The interventional Shapley approach requires a
            background dataset to be provided.

        Args:
            tree_model (dict): The tree model to be explained. The tree model must be a dictionary
                with the following keys:
                    - children_left: np.ndarray[int] - The left children of each node.
                        Leaf nodes are denoted with -1.
                    - children_right: np.ndarray[int] - The right children of each node.
                        Leaf nodes are denoted with -1.
                    - features: np.ndarray[int] - The feature used for splitting at each node.
                        Leaf nodes have the value -2.
                    - thresholds: np.ndarray[float] - The threshold used for splitting at each node.
                        Leaf nodes have the value -2.
                    - values: np.ndarray[float] - The output values at the leaf values of the tree.
                        The values for decision nodes are not required.
                    - node_sample_weight: np.ndarray[float] - The sample weights of the tree. Only
                        required for observational Shapley interactions.
            max_interaction_order (int, optional): The maximum interaction order to be computed. An
                interaction order of 1 corresponds to the Shapley value. Any value higher than 1
                computes the Shapley interactions values up to that order. Defaults to 1 (i.e. SV).
            observational (bool, optional): Whether to compute the Shapley interactions for the
                observational or interventional Shapley approach. Defaults to True.
            background_dataset (np.ndarray, optional): The background dataset to be used for the
                interventional Shapley approach. The dataset is used to determine how often each
                child node is reached by the background dataset without splitting before. Defaults
                to None.
            n_features (int, optional): The number of features of the dataset. If no value is
                provided, the number of features is determined by the maximum feature id in the tree
                model. Defaults to None.
            root_node_id (int, optional): The root node id of the tree. Defaults to 0.
        """
        # get the node attributes from the tree_model definition
        self.children_left: np.ndarray[int] = tree_model["children_left"]  # -1 for leaf nodes
        self.children_right: np.ndarray[int] = tree_model["children_right"]  # -1 for leaf nodes
        self.parents: np.ndarray[int] = _get_parent_array(self.children_left, self.children_right)  # -1 for the root node
        self.features: np.ndarray = tree_model["features"]  # -2 for leaf nodes
        self.thresholds: np.ndarray = tree_model["thresholds"]  # -2 for leaf nodes

        # get the number of nodes and the node ids
        self.n_nodes: int = len(self.children_left)
        self.nodes: np.ndarray = np.arange(self.n_nodes)
        # get the leaf and node masks
        self.leaf_mask: np.ndarray = self.children_left == -1
        self.node_mask: np.ndarray = ~self.leaf_mask

        # set the root node id and number of features
        self.root_node_id: int = root_node_id
        self.n_features: int = n_features
        if n_features is None:
            self.n_features: int = max(self.features) + 1

        # get the leaf predictions and the observational or interventional sample weights
        self.values = tree_model["values"]
        self.interaction_order = max_interaction_order
        if observational:
            node_sample_weight = tree_model["node_sample_weight"]
        else:
            # TODO check if the leaf_pred are correct or incorporate TreeSHAP
            node_sample_weight = self._compute_interventional_node_sample_weights(background_dataset)
        self.weights, self.leaf_predictions = self._recursively_compute_weights(
            children_left=self.children_left,
            children_right=self.children_right,
            node_sample_weight=node_sample_weight,
            values=self.values,
            observational=observational
        )

        self.ancestor_nodes, self.edge_heights, self.has_ancestors = _recursively_copy_tree(
            self.children_left, self.children_right, self.parents, self.features, self.n_features,
            max_interaction_order)

        # initialize an array to store the summary polynomials
        self.summary_polynomials: np.ndarray = np.empty(self.n_nodes, dtype=Polynomial)

        # initialize an array to store the shapley values
        self.shapley_values: np.ndarray = np.zeros(self.n_features, dtype=float)

        # get empty prediction of model
        # self.empty_prediction = self.probabilities * self.leaf_predictions
        self.empty_prediction: float = float(np.sum(self.leaf_predictions[self.leaf_mask]))

        # stores the interaction scores up to a given order
        self.max_order: int = max_interaction_order
        self.shapley_interactions: np.ndarray[float] = np.zeros(int(binom(self.n_features, self.max_order)), dtype=float)
        self.shapley_interactions_lookup: dict = {}
        self.subset_ancestors: dict = {}
        self.subset_updates_pos: dict = {}  # stores position of interactions that include feature i
        self.subset_updates: dict = {}  # stores interactions that include feature i
        self._precalculate_interaction_ancestors()

    def explain(
            self,
            x: np.ndarray,
            order: int = 1
    ) -> np.ndarray[float]:
        """Computes the Shapley Interaction values for a given instance x and interaction order.
            This function is the main explanation function of this class.

        Args:
            x (np.ndarray): Instance to be explained.
            order (int, optional): Order of the interactions. Defaults to 1.

        Returns:
            np.ndarray[float]: Shapley Interaction values. The shape of the array is (n_features,
                order).
        """
        assert order <= self.max_order, f"Order {order} is larger than the maximum interaction " \
                                        f"order {self.max_order}."
        self.shapley_interactions = np.zeros(int(binom(self.n_features, order)), dtype=float)
        # get an array index by the nodes
        initial_polynomial = Polynomial([1.])
        # call the recursive function to compute the shapley values
        self._compute_shapley_values(x, 0, initial_polynomial, order)
        return self.shapley_interactions.copy()

    def _compute_shapley_values(
            self,
            x: np.ndarray,
            node_id: int,
            path_summary_poly: Polynomial,
            order: int,
            p_e_storage: np.ndarray[float] = None,
            went_left: bool = None,
    ):
        # to store the p_e(x) of the features
        p_e_storage: np.ndarray[float] = np.ones(self.n_features, dtype=float) if p_e_storage is None else p_e_storage

        # get node / edge information
        parent_id = self.parents[node_id]
        feature_id = self.features[parent_id]
        feature_threshold = self.thresholds[parent_id]
        edge_weight = self.weights[node_id]
        # backup ancestor p_e for updates of shapley interactions
        p_e_ancestor = p_e_storage[feature_id].copy()

        # the node had an edge before, so we need to update the summary polynomial
        if node_id is not self.root_node_id:

            # polynomial correction if feature_id has been observed before (has an ancestor)
            if self.has_ancestors[node_id]:
                # remove previous polynomial factor, to extend with current updated factor
                path_summary_poly = Polynomial(polydiv(path_summary_poly.coef, Polynomial([p_e_ancestor, 1]).coef)[0])

            # compute current p_e, i.e. update from previous for this feature_id and extend summary polynomial
            p_e_current = self._get_p_e(x, feature_id, edge_weight, p_e_storage[feature_id], feature_threshold, went_left)
            p_e_storage[feature_id] = p_e_current
            path_summary_poly = path_summary_poly * Polynomial([p_e_current, 1])

        # if node is a leaf (base case of the recursion)
        if self.leaf_mask[node_id]:
            self.summary_polynomials[node_id] = path_summary_poly * self.leaf_predictions[node_id]
        else:  # if the node is a decision node then we continue the recursion down the tree
            left_child_id = self.children_left[node_id]
            self._compute_shapley_values(
                x=x,
                node_id=left_child_id,
                path_summary_poly=path_summary_poly,
                order=order,
                p_e_storage=p_e_storage.copy(),
                went_left=True,
            )
            right_child_id = self.children_right[node_id]
            self._compute_shapley_values(
                x=x,
                node_id=right_child_id,
                path_summary_poly=path_summary_poly,
                order=order,
                p_e_storage=p_e_storage.copy(),
                went_left=False,
            )
            # add the summary polynomials of the left and right child nodes together
            added_polynomial = self._special_polynomial_addition(
                p1=self.summary_polynomials[left_child_id],
                p2=self.summary_polynomials[right_child_id]
            )
            # store the summary polynomial of the current node
            self.summary_polynomials[node_id] = added_polynomial

        if node_id is not self.root_node_id:
            # TODO describe verbally the q_S variables
            q_S, Q_S, Q_S_ancestor, q_S_ancestor = {}, {}, {}, {}
            for pos, S in zip(self.subset_updates_pos[feature_id], self.subset_updates[feature_id]):
                # compute interaction factor and polynomial for aggregation below
                q_S[S] = self._compute_p_e_interaction(S, p_e_storage)
                Q_S[S] = self._compute_poly_interaction(S, p_e_storage)
                # update interactions for all interactions that contain feature_id
                quotient = Polynomial(polydiv(self.summary_polynomials[node_id].coef, Q_S[S].coef)[0])
                psi = self._psi(quotient)
                self.shapley_interactions[pos] += q_S[S] * psi

                # if the node has an ancestor, we need to update the interactions for the ancestor
                ancestor_node_id = self.subset_ancestors[node_id][pos]
                if ancestor_node_id > -1:
                    p_e_storage_ancestor = p_e_storage.copy()
                    p_e_storage_ancestor[feature_id] = p_e_ancestor
                    q_S_ancestor[S] = self._compute_p_e_interaction(S, p_e_storage_ancestor)
                    Q_S_ancestor[S] = self._compute_poly_interaction(S, p_e_storage_ancestor)
                    d_e = self.edge_heights[node_id]
                    d_e_ancestor = self.edge_heights[ancestor_node_id]
                    psi_factor = Polynomial([1, 1]) ** (d_e_ancestor - d_e)
                    psi_product = self.summary_polynomials[node_id] * psi_factor
                    quotient_ancestor = Polynomial(polydiv(psi_product.coef, Q_S_ancestor[S].coef)[0])
                    psi_ancestor = self._psi(quotient_ancestor)
                    self.shapley_interactions[pos] -= q_S_ancestor[S] * psi_ancestor

    def explain_brute_force(
            self,
            x: np.ndarray,
            max_order: int = 1
    ):
        self.shapley_values: np.ndarray = np.zeros(self.n_features, dtype=float)

        # Evaluate model for every subset
        counter_subsets = 0
        subset_values = np.zeros(2 ** self.n_features)
        position_lookup = {}
        for S in powerset(range(self.n_features)):
            subset_values[counter_subsets] = self._naive_shapley_recursion(x, S, 0, 1)
            position_lookup[S] = counter_subsets
            counter_subsets += 1

        # Aggregate subsets for the shapley values / interactions
        shapley_interactions = {}
        shapley_interactions_lookup = {}
        for order in range(max_order + 1):
            if order == 0:
                shapley_interactions[order] = subset_values[position_lookup[()]]
            else:
                si = self._compute_shapley_from_subsets(subset_values, position_lookup, order)
                shapley_interactions[order], shapley_interactions_lookup[order] = si
        return shapley_interactions, shapley_interactions_lookup

    def _compute_shapley_from_subsets(
            self,
            subset_values: np.ndarray,
            position_lookup: dict,
            order: int = 1
    ):
        features = range(self.n_features)
        shapley_interactions = np.zeros(int(binom(self.n_features, order)))
        shapley_interactions_lookup = {}
        counter_interactions = 0
        for S in powerset(features, order):
            temp_values = 0
            for T in powerset(set(features) - set(S)):
                weight_T = 1 / (binom(self.n_features - order, len(T)) * (self.n_features - order + 1))
                for L in powerset(S):
                    subset = tuple(sorted(L + T))
                    pos = position_lookup[subset]
                    temp_values += weight_T * (-1) ** (order - len(L)) * subset_values[pos]
            shapley_interactions[counter_interactions] = temp_values
            shapley_interactions_lookup[counter_interactions] = S
            counter_interactions += 1
        return shapley_interactions, shapley_interactions_lookup

    def _naive_shapley_recursion(
            self,
            x: np.ndarray[float],
            S: tuple,
            node_id: int,
            weight: float
    ):
        threshold = self.thresholds[node_id]
        feature_id = self.features[node_id]
        if self.leaf_mask[node_id]:
            return self.values[node_id] * weight
        else:
            if feature_id in S:
                if x[feature_id] <= threshold:
                    subset_val_right = 0
                    subset_val_left = self._naive_shapley_recursion(x, S, self.children_left[node_id], weight)
                else:
                    subset_val_left = 0
                    subset_val_right = self._naive_shapley_recursion(x, S, self.children_right[node_id], weight)
            else:
                subset_val_left = self._naive_shapley_recursion(
                    x, S, self.children_left[node_id], weight * self.weights[self.children_left[node_id]])
                subset_val_right = self._naive_shapley_recursion(
                    x, S, self.children_right[node_id], weight * self.weights[self.children_right[node_id]])
        return subset_val_left + subset_val_right

    def _precalculate_interaction_ancestors(self):
        """Calculates the position of the ancestors of the interactions for the tree for a given
        order of interactions."""

        # stores position of interactions
        counter_interaction = 0

        for node_id in self.ancestor_nodes:
            self.subset_ancestors[node_id] = np.full(int(binom(self.n_features, self.max_order)), -1, dtype=int)
        for S in powerset(range(self.n_features), self.max_order):
            self.shapley_interactions_lookup[S] = counter_interaction
            for node_id in self.ancestor_nodes:
                subset_ancestor = -1
                for i in S:
                    subset_ancestor = max(subset_ancestor, self.ancestor_nodes[node_id][i])
                self.subset_ancestors[node_id][counter_interaction] = subset_ancestor
            counter_interaction += 1

        for i in range(self.n_features):
            subsets = []
            positions = np.zeros(int(binom(self.n_features - 1, self.max_order - 1)), dtype=int)
            pos_counter = 0
            for S in powerset(range(self.n_features), min_size=self.max_order, max_size=self.max_order):
                if i in S:
                    positions[pos_counter] = self.shapley_interactions_lookup[S]
                    subsets.append(S)
                    pos_counter += 1
            self.subset_updates_pos[i] = positions
            self.subset_updates[i] = subsets

    @staticmethod
    def _compute_poly_interaction(S: tuple, p_e_values: np.ndarray[float]) -> Polynomial:
        """Computes Q_S (interaction polynomial) given p_e values"""
        poly_interaction = Polynomial([1.])
        for i in S:
            poly_interaction = poly_interaction * Polynomial([p_e_values[i], 1])
        return poly_interaction

    @staticmethod
    def _compute_p_e_interaction(S: tuple, p_e_values: np.ndarray[float]) -> float:
        """Computes q_S (interaction factor) given p_e values"""
        p_e_interaction = 0
        for L in powerset(set(S)):
            p_e_prod = 1
            for j in L:
                p_e_prod *= p_e_values[j]
            p_e_interaction += (-1) ** (len(S) - len(L)) * p_e_prod
        return p_e_interaction

    def _compute_interventional_node_sample_weights(
            self,
            background_dataset: np.ndarray[float]
    ) -> np.ndarray[float]:
        """Computes the interventional sample weights with a background dataset of shape (n_samples,
            n_features).

         The interventional sample weights are computed by counting how many data points from the
            background dataset go left and right at each decision node and thus reach either the
            left or right child node.

        Args:
            background_dataset (np.ndarray[float]): A background dataset of shape (n_samples,
                n_features).

        Returns:
            np.ndarray[int]: The interventional sample weights (split probabilities) for each node
                in the tree.
        """

        active_edges: np.ndarray = np.zeros((background_dataset.shape[0], self.n_nodes), dtype=bool)
        left_active = background_dataset[:, self.features[self.node_mask]] <= self.thresholds[self.node_mask]
        active_edges[:, self.children_left[self.node_mask]] = left_active
        right_active = background_dataset[:, self.features[self.node_mask]] > self.thresholds[self.node_mask]
        active_edges[:, self.children_right[self.node_mask]] = right_active
        active_edges[:, 0] = True
        sample_count = np.sum(active_edges, axis=0)
        sample_weights = sample_count / sample_count[0]
        return sample_weights

    @staticmethod
    def _recursively_compute_weights(
            children_left: np.ndarray[int],
            children_right: np.ndarray[int],
            node_sample_weight: np.ndarray[int],
            values: np.ndarray[float],
            observational: bool = True,
    ) -> tuple[np.ndarray[float], np.ndarray[float]]:
        """Traverse the tree and recursively copy edge information from the tree. Get the feature
            ancestor nodes for each node in the tree. An ancestor node is the last observed node that
            has the same feature as the current node in the path.The ancestor of the root node is
            -1. The ancestor nodes are found through recursion from the root node.

        Args:
            children_left (np.ndarray[int]): The left children of the tree. Leaf nodes are -1.
            children_right (np.ndarray[int]): The right children of the tree. Leaf nodes are -1.
            node_sample_weight (np.ndarray[int]): The number of samples in each node.
            values (np.ndarray[float]): The values of each node in the tree.
            observational (bool): Flag to indicate whether to use observational or interventional
                sample weights. Defaults to True (observational).

        Returns:
            ancestor_nodes (np.ndarray[int]): The ancestor nodes for each node in the tree.
            edge_heights (np.ndarray[int]): The edge heights for each node in the tree.
        """
        leaf_predictions: np.ndarray[float] = np.zeros(children_left.shape, dtype=float)
        if observational:
            weights: np.ndarray[float] = np.ones(children_left.shape, dtype=float)
            n_total_samples = node_sample_weight[0]
        else:
            weights: np.ndarray[float] = node_sample_weight.copy()
            path_probabilities: np.ndarray[float] = np.zeros(children_left.shape, dtype=float)
            path_probabilities[0] = 1.

        def _recursive_compute(node_id: int):
            """Recursively search for the ancestor node of the current node."""
            if children_left[node_id] > -1:  # not a leaf node
                weights[children_left[node_id]] = node_sample_weight[children_left[node_id]] / node_sample_weight[node_id]
                _recursive_compute(children_left[node_id])
                weights[children_right[node_id]] = node_sample_weight[children_right[node_id]] / node_sample_weight[node_id]
                _recursive_compute(children_right[node_id])
            else:  # is a leaf node multiply weights (R^v_\emptyset) by leaf_prediction
                leaf_predictions[node_id] = node_sample_weight[node_id] / n_total_samples * values[node_id]

        def _recursive_compute_int(node_id: int):
            """Recursively search for the ancestor node of the current node."""
            if children_left[node_id] > -1:  # not a leaf node
                path_probabilities[children_left[node_id]] = path_probabilities[node_id] * weights[children_left[node_id]]
                _recursive_compute_int(children_left[node_id])
                path_probabilities[children_right[node_id]] = path_probabilities[node_id] * weights[children_right[node_id]]
                _recursive_compute_int(children_right[node_id])
            else:  # is a leaf node multiply weights (R^v_\emptyset) by leaf_prediction
                leaf_predictions[node_id] = path_probabilities[node_id] * values[node_id]

        if observational:
            _recursive_compute(0)
        else:
            _recursive_compute_int(0)
        return weights, leaf_predictions

    @staticmethod
    def _get_binomial_polynomial(degree: int) -> Polynomial:
        """Get a reciprocal binomial polynomial with degree d.

        The reciprocal binomial polynomial is defined as `\sum_{i=0}^{d} binom(d, i)^{-1} * x^{i}`.

        Args:
            degree (int): The degree of the binomial polynomial.

        Returns:
            Polynomial: The reciprocal binomial polynomial.
        """
        return Polynomial([1 / binom(degree, i) for i in range(degree + 1)])

    def _special_polynomial_addition(self, p1: Polynomial, p2: Polynomial) -> Polynomial:
        """Add two polynomials of different degrees.

        Args:
            p1 (Polynomial): The first polynomial with degree d1.
            p2 (Polynomial): The second polynomial with degree d2.

        Returns:
            Polynomial: The sum of the two polynomials with degree max(d1, d2).
        """
        power = p1.degree() - p2.degree()
        if power < 0:
            return self._special_polynomial_addition(p2, p1)
        # a polynomial of (1 + x) with degree d1 - d2
        p3 = Polynomial([1, 1]) ** power
        return p1 + p2 * p3

    def _psi(self, polynomial: Polynomial) -> float:
        """Compute the psi function for a polynomial with degree d.

        The psi function is computed by the inner product of the polynomial with a reciprocal
            binomial polynomial of degree d. This quotient is then divided by the degree d + 1.

        Args:
            polynomial (Polynomial): The polynomial with degree d.

        Returns:
            float: The psi function for the polynomial.
        """
        binomial_polynomial = self._get_binomial_polynomial(polynomial.degree())
        inner_product = np.inner(polynomial.coef, binomial_polynomial.coef)
        return inner_product / (polynomial.degree() + 1)

    @staticmethod
    def _get_p_e(
            x: np.ndarray,
            feature_id: int,
            edge_weight: float,
            p_e_previous: float,
            feature_threshold: int,
            went_left: bool
    ) -> float:
        """Get the weight p_e of the decision for the feature given the instance.

        Args:
            x: The input data.
            feature_id (int): The id of the feature of the edge.
            edge_weight (float): The weight of the edge.
            feature_threshold (float): The id of the parent node of the edge.
            went_left (bool): Whether the instance went left or right at the parent node.

        Returns:
            float: The weight of the decision for the feature given the instance.
        """
        activation = 0
        if went_left:
            if x[feature_id] <= feature_threshold:
                activation = 1
        else:
            if x[feature_id] > feature_threshold:
                activation = 1
        p_e = p_e_previous * 1 / edge_weight
        return activation * p_e

    @staticmethod
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

    @staticmethod
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


if __name__ == "__main__":
    DO_TREE_SHAP = False
    DO_PLOTTING = True
    DO_OBSERVATIONAL = False
    DO_GROUND_TRUTH = False

    INTERACTION_ORDER = 1

    from linear_interaction.utils import convert_tree

    if DO_TREE_SHAP:
        try:
            from shap import TreeExplainer
        except ImportError:
            print("TreeSHAP not available. Please install shap package.")
            DO_TREE_SHAPE = False

    from sklearn.datasets import make_regression
    from sklearn.tree import DecisionTreeRegressor, plot_tree
    import matplotlib.pyplot as plt
    import numpy as np
    from copy import deepcopy

    # fix random seed for reproducibility
    random_seed = 10
    np.random.seed(random_seed)

    # create dummy regression dataset and fit tree model
    X, y = make_regression(1000, n_features=10, random_state=random_seed)
    clf = DecisionTreeRegressor(max_depth=5, random_state=random_seed).fit(X, y)

    # convert the tree to be usable like in TreeSHAP
    # tree_model = convert_tree(clf)

    x_input = X[:1]
    print("Output f(x):", clf.predict(x_input)[0])

    if DO_PLOTTING:
        plt.figure(dpi=150)
        plot_tree(clf, node_ids=True, proportion=True)
        plt.savefig("tree.pdf")

    # TreeSHAP -------------------------------------------------------------------------------------

    my_thresholds = clf.tree_.threshold.copy()
    # my_thresholds[32] = 0.5

    my_features = clf.tree_.feature.copy()
    # my_features[0] = 8
    # my_features[32] = 6
    # my_features[26] = 8
    # my_features[33] = 8
    # my_features[33] = 8
    # my_features[48] = 1
    # my_features[57] = 4
    # my_features[35] = 7

    tree_dict = {
        "children_left": clf.tree_.children_left.copy(),
        "children_right": clf.tree_.children_right.copy(),
        "children_default": clf.tree_.children_left.copy(),
        "features": my_features,
        "thresholds": my_thresholds,
        "values": clf.tree_.value.reshape(-1, 1).copy(),
        "node_sample_weight": clf.tree_.weighted_n_node_samples.copy()
    }

    model = {
        "trees": [tree_dict]
    }

    if DO_TREE_SHAP:
        # explain the tree with observational TreeSHAP
        start_time = time.time()
        if DO_OBSERVATIONAL:
            explainer_shap = TreeExplainer(deepcopy(model), feature_perturbation="tree_path_dependent")
        else:
            explainer_shap = TreeExplainer(deepcopy(model), feature_perturbation="interventional", data=X[:50])

        sv_tree_shap = explainer_shap.shap_values(x_input)
        time_elapsed = time.time() - start_time

        print("TreeSHAP")
        print("TreeSHAP - time elapsed  ", time_elapsed)
        print("TreeSHAP - SVs (obs.)    ", sv_tree_shap)
        print("TreeSHAP - sum SVs (obs.)", sv_tree_shap.sum() + explainer_shap.expected_value)
        print("TreeSHAP - empty pred    ", explainer_shap.expected_value)
        print()

    # LinearTreeSHAP -------------------------------------------------------------------------------
    # explain the tree with LinearTreeSHAP
    start_time = time.time()
    explainer = TreeSHAPIQExplainer(
        tree_model=deepcopy(tree_dict), n_features=x_input.shape[1], observational=True
    )
    sv_linear_tree_shap = explainer.explain(x_input[0], INTERACTION_ORDER)
    time_elapsed = time.time() - start_time

    print("Linear")
    print("Linear - time elapsed    ", time_elapsed)
    print("Linear - SVs (obs.)      ", sv_linear_tree_shap)
    print("Linear - sum SVs (obs.)  ", sv_linear_tree_shap.sum() + explainer.empty_prediction)
    print("Linear - time elapsed    ", time_elapsed)
    print("Linear - empty pred      ", explainer.empty_prediction)
    print()

    if not DO_OBSERVATIONAL:
        start_time = time.time()
        explainer = TreeSHAPIQExplainer(
           tree_model=deepcopy(tree_dict), n_features=x_input.shape[1], observational=False, background_dataset=X[:50]
        )
        sv_linear_tree_shap = explainer.explain(x_input[0], INTERACTION_ORDER)
        time_elapsed = time.time() - start_time

        print("Linear")
        print("Linear - time elapsed    ", time_elapsed)
        print("Linear - SVs (int.)      ", sv_linear_tree_shap)
        print("Linear - sum SVs (int.)  ", sv_linear_tree_shap.sum() + explainer.empty_prediction)
        print("Linear - time elapsed    ", time_elapsed)
        print("Linear - empty pred      ", explainer.empty_prediction)
        print()

    # Ground Truth Brute Force ---------------------------------------------------------------------
    # compute the ground truth interactions with brute force
    if DO_GROUND_TRUTH:

        start_time = time.time()
        gt_results = explainer.explain_brute_force(x_input[0], INTERACTION_ORDER)
        ground_truth_shap_int, ground_truth_shap_int_pos = gt_results
        time_elapsed = time.time() - start_time

        print("Ground Truth")
        print("GT - time elapsed        ", time_elapsed)
        print()
        for ORDER in range(INTERACTION_ORDER + 1):
            print(f"GT - order {ORDER} SIs         ", ground_truth_shap_int[ORDER])
            print(f"GT - order {ORDER} sum SIs     ", ground_truth_shap_int[ORDER].sum())
            print()

        # debug order =2
        if len(sv_linear_tree_shap) == binom(x_input.shape[1], 2):
            mismatch = np.where((ground_truth_shap_int[2] - sv_linear_tree_shap) ** 2 > 0.001)
            for key in mismatch[0]:
                print(ground_truth_shap_int_pos[2][key])
