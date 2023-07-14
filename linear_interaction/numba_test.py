"""This module contains the linear TreeSHAP class, which is a reimplementation of the original TreeSHAP algorithm."""
import time

from numba import jit

from utils import tree_model, powerset, _recursively_copy_tree

from collections import namedtuple


import numpy as np
from numpy.polynomial import Polynomial
from numpy.polynomial.polynomial import polydiv, polymul
from scipy.special import binom


class LinearTreeSHAPExplainer:

    def __init__(self, tree_model: dict, n_features: int = None, root_node_id: int = 0):
        # get the node attributes from the tree_model definition
        self.children_left: np.ndarray[int] = tree_model["children_left"]  # -1 for leaf nodes
        self.children_right: np.ndarray[int] = tree_model["children_right"]  # -1 for leaf nodes
        self.parents: np.ndarray[int] = _get_parent_array(self.children_left,
                                                          self.children_right)  # -1 for the root node
        self.features: np.ndarray = tree_model["features"]  # -2 for leaf nodes
        self.thresholds: np.ndarray = tree_model["thresholds"]  # -2 for leaf nodes
        # not really needed
        self.node_sample_weight = tree_model["node_sample_weight"]
        self.values = tree_model["values"]

        self.ancestor_nodes, self.edge_heights = _recursively_copy_tree(self.children_left,
                                                                        self.children_right,
                                                                        self.parents,
                                                                        self.features,
                                                                        n_features
                                                                        )

        self.weights, self.leaf_predictions = self._recursively_compute_weights(self.children_left,
                                                                                self.children_right,
                                                                                self.node_sample_weight,
                                                                                self.values
                                                                                )
        # set the root node id and number of features
        self.root_node_id: int = root_node_id
        self.n_features: int = n_features
        if n_features is None:
            self.n_features: int = len(np.unique(self.features))

        # get the number of nodes and the node ids
        self.n_nodes: int = len(self.children_left)
        self.nodes: np.ndarray = np.arange(self.n_nodes)

        # get the leaf and node masks
        self.leaf_mask: np.ndarray = self.children_left == -1
        self.node_mask: np.ndarray = ~self.leaf_mask

        # initialize an array to store the summary polynomials
        self.summary_polynomials: np.ndarray = np.empty(self.n_nodes, dtype=Polynomial)

        # initialize an array to store the shapley values
        self.shapley_values: np.ndarray = np.zeros(self.n_features, dtype=float)

        # get empty prediction of model
        # self.empty_prediction = self.probabilities * self.leaf_predictions
        self.empty_prediction: float = float(np.sum(self.leaf_predictions[self.leaf_mask]))

    def _compute_shapley_from_subsets(self, subset_values: np.ndarray, position_lookup: dict,
                                      order: int = 1):
        features = range(self.n_features)
        shapley_interactions = np.zeros(int(binom(self.n_features, order)))
        shapley_interactions_lookup = {}
        counter_interactions = 0
        for S in powerset(features, order):
            temp_values = 0
            for T in powerset(set(features) - set(S)):
                weight_T = 1 / (
                            binom(self.n_features - order, len(T)) * (self.n_features - order + 1))
                for L in powerset(S):
                    subset = tuple(sorted(L + T))
                    pos = position_lookup[subset]
                    temp_values += weight_T * (-1) ** (order - len(L)) * subset_values[pos]
            shapley_interactions[counter_interactions] = temp_values
            shapley_interactions_lookup[counter_interactions] = S
            counter_interactions += 1
        return shapley_interactions, shapley_interactions_lookup

    def explain(self, x: np.ndarray, order: int = 1):
        # get an array index by the nodes
        initial_polynomial = Polynomial([1.])

        # Stores interactions
        self.shapley_interactions: np.ndarray = np.zeros(int(binom(self.n_features, order)),
                                                         dtype=float)
        # Stores position of interactions
        self.shapley_interactions_lookup = {}
        counter_interaction = 0
        for S in powerset(range(self.n_features), order):
            self.shapley_interactions_lookup[S] = counter_interaction
            counter_interaction += 1

        # Stores position of interactions that include feature i
        self.subset_updates = {}
        for i in range(self.n_features):
            positions = np.zeros(int(binom(self.n_features - 1, order - 1)))
            pos_counter = 0
            for S in powerset(range(self.n_features), min_size=order, max_size=order):
                if i in S:
                    positions[pos_counter] = self.shapley_interactions_lookup[S]
                    pos_counter += 1
            self.subset_updates[i] = positions

        self._compute_shapley_values(x, 0, initial_polynomial, order)
        # get an array indexed by the features
        return self.shapley_interactions.copy()

    def _compute_shapley_values(
            self,
            x: np.ndarray,
            node_id: int,
            path_summary_poly: Polynomial,
            order: int,
            p_e_of_feature_ancestor: np.ndarray[float] = None,
            p_e_storage: np.ndarray[float] = None,
            went_left: bool = None,
    ):
        # to store the p_e(x) of the feature ancestors when the feature was seen last in the path
        p_e_of_feature_ancestor: np.ndarray[float] = np.ones(self.n_features,
                                                             dtype=float) if p_e_of_feature_ancestor is None else p_e_of_feature_ancestor
        # to store the p_e(x) of the features
        p_e_storage: np.ndarray[float] = np.ones(self.n_features,
                                                 dtype=float) if p_e_storage is None else p_e_storage

        # the node had an edge before, so we need to update the summary polynomial
        if node_id is not self.root_node_id:
            # get node / edge information
            parent_id = self.parents[node_id]
            feature_id = self.features[parent_id]
            feature_threshold = self.thresholds[parent_id]
            edge_weight = self.weights[node_id]

            ancestor_node_id = self.ancestor_nodes[node_id]
            if ancestor_node_id > -1:
                # if feature has an ancestor
                p_e_ancestor = p_e_of_feature_ancestor[feature_id]
                p_e_ancestor_storage = p_e_of_feature_ancestor.copy()

            p_e = _get_p_e(x, feature_id, edge_weight, p_e_of_feature_ancestor,
                                feature_threshold, went_left)
            p_e_storage[feature_id] = p_e
            p_e_of_feature_ancestor[feature_id] = p_e
            path_summary_poly = path_summary_poly * Polynomial([p_e, 1])

            if ancestor_node_id > -1:
                path_summary_poly = Polynomial(
                    polydiv(path_summary_poly.coef, Polynomial([p_e_ancestor, 1]).coef)[0])

        # if node is a leaf (base case of the recursion)
        if self.leaf_mask[node_id]:
            self.summary_polynomials[node_id] = path_summary_poly * self.leaf_predictions[node_id]
        else:
            # if the node is a decision node then we continue the recursion down the tree
            # if self.children_left[node_id]>-1:
            left_child_id = self.children_left[node_id]
            self._compute_shapley_values(
                x=x,
                node_id=left_child_id,
                path_summary_poly=path_summary_poly,
                order=order,
                p_e_of_feature_ancestor=p_e_of_feature_ancestor.copy(),
                p_e_storage=p_e_storage.copy(),
                went_left=True,
            )
            right_child_id = self.children_right[node_id]
            self._compute_shapley_values(
                x=x,
                node_id=right_child_id,
                path_summary_poly=path_summary_poly,
                order=order,
                p_e_of_feature_ancestor=p_e_of_feature_ancestor.copy(),
                p_e_storage=p_e_storage.copy(),
                went_left=False,
            )
            # add the summary polynomials of the left and right child nodes together
            added_polynomial = _special_polynomial_addition(
                p1=self.summary_polynomials[left_child_id],
                p2=self.summary_polynomials[right_child_id]
            )
            # store the summary polynomial of the current node
            self.summary_polynomials[node_id] = added_polynomial
            # self.summary_polynomials_degree[node_id] = added_polynomial.degree()

        if node_id is not self.root_node_id:

            N = set(range(self.n_features))
            N_minus_feature = N - {feature_id}

            for S in powerset(N_minus_feature, min_size=order - 1, max_size=order - 1):
                S_plus_feature = tuple(sorted(S + (feature_id,)))
                S_pos = self.shapley_interactions_lookup[S_plus_feature]
                poly_denom = Polynomial([1.])
                for i in S_plus_feature:
                    poly_denom = poly_denom * Polynomial([p_e_storage[feature_id], 1])
                quotient = Polynomial(
                    polydiv(self.summary_polynomials[node_id].coef, poly_denom.coef)[0])
                psi = _psi(quotient)

                S_p_e = 0
                for L in powerset(set(S_plus_feature)):
                    p_e_prod = 1
                    for j in L:
                        p_e_prod *= p_e_storage[j]
                    S_p_e += (-1) ** (order - len(L)) * p_e_prod
                self.shapley_interactions[S_pos] += (S_p_e) * psi

            # Update Shapley values
            # quotient = Polynomial(
            #    polydiv(self.summary_polynomials[node_id].coef, Polynomial([p_e, 1]).coef)[0])
            # psi = self._psi(quotient)
            # self.shapley_interactions[feature_id] += (p_e - 1) * psi

            # the part below is only needed if the feature was already encountered in the path
            if ancestor_node_id > -1:
                d_e = self.edge_heights[node_id]
                d_e_ancestor = self.edge_heights[self.ancestor_nodes[node_id]]
                psi_factor = Polynomial([1, 1]) ** (d_e_ancestor - d_e)
                psi_product = self.summary_polynomials[node_id] * psi_factor
                psi_denominator = Polynomial([p_e_ancestor, 1])
                quotient_ancestor = Polynomial(polydiv(psi_product.coef, psi_denominator.coef)[0])
                psi_ancestor = _psi(quotient_ancestor)
                self.shapley_interactions[feature_id] -= (p_e_ancestor - 1) * psi_ancestor

    def _recursively_compute_weights(self,
                                     children_left: np.ndarray[int],
                                     children_right: np.ndarray[int],
                                     node_sample_weight: np.ndarray[int],
                                     values: np.ndarray[float],
                                     background_dataset: np.ndarray[float] = None
                                     ) -> tuple[np.ndarray[float], np.ndarray[float]]:
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

        # Use observations to compute weights (observational approach)
        weights: np.ndarray[float] = np.ones(children_left.shape, dtype=float)
        leaf_predictions: np.ndarray[float] = np.zeros(children_left.shape, dtype=float)

        if background_dataset is not None:
            # Use background dataset to compute weights (interventional approach)
            # TODO: process background_dataset to compute weights
            print("not yet implemented")
        else:

            max_tree_depth = 0
            n_total_samples = node_sample_weight[0]

            def _recursive_compute(
                    node_id: int
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
                if children_left[node_id] > -1:  # not a leaf node
                    weights[children_left[node_id]] = node_sample_weight[children_left[node_id]] / \
                                                      node_sample_weight[
                                                          node_id]
                    _recursive_compute(children_left[node_id])
                    weights[children_right[node_id]] = node_sample_weight[children_right[node_id]] / \
                                                       node_sample_weight[node_id]
                    _recursive_compute(children_right[node_id])
                else:  # is a leaf node multiply weights (R^v_\emptyset) by leaf_prediction
                    leaf_predictions[node_id] = node_sample_weight[node_id] / n_total_samples * \
                                                values[node_id]

            _recursive_compute(0)
        return weights, leaf_predictions


@jit(nopython=False)
def _get_binomial_polynomial(degree: int) -> Polynomial:
    """Get a reciprocal binomial polynomial with degree d.

    The reciprocal binomial polynomial is defined as `\sum_{i=0}^{d} binom(d, i)^{-1} * x^{i}`.

    Args:
        degree (int): The degree of the binomial polynomial.

    Returns:
        Polynomial: The reciprocal binomial polynomial.
    """
    return Polynomial([1 / binom(degree, i) for i in range(degree + 1)])


@jit(nopython=False)
def _special_polynomial_addition(p1: Polynomial, p2: Polynomial) -> Polynomial:
    """Add two polynomials of different degrees.

    Args:
        p1 (Polynomial): The first polynomial with degree d1.
        p2 (Polynomial): The second polynomial with degree d2.

    Returns:
        Polynomial: The sum of the two polynomials with degree max(d1, d2).
    """
    power = p1.degree() - p2.degree()
    if power < 0:
        return _special_polynomial_addition(p2, p1)
    # a polynomial of (1 + x) with degree d1 - d2
    p3 = Polynomial([1, 1]) ** power
    return p1 + p2 * p3


@jit(nopython=False)
def _psi(polynomial: Polynomial) -> float:
    """Compute the psi function for a polynomial with degree d.

    The psi function is computed by the inner product of the polynomial with a reciprocal
        binomial polynomial of degree d. This quotient is then divided by the degree d + 1.

    Args:
        polynomial (Polynomial): The polynomial with degree d.

    Returns:
        float: The psi function for the polynomial.
    """
    binomial_polynomial = _get_binomial_polynomial(polynomial.degree())
    inner_product = np.inner(polynomial.coef, binomial_polynomial.coef)
    inner_product = inner_product / (polynomial.degree() + 1)
    return inner_product


@jit(nopython=True)
def _get_p_e(
        x: np.ndarray,
        feature_id: int,
        edge_weight: float,
        p_e_of_feature_ancestor: np.ndarray[float],
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
    p_e = p_e_of_feature_ancestor[feature_id] * 1 / edge_weight
    return activation * p_e


@jit(nopython=True)
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


@jit(nopython=True)
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
    from sklearn.datasets import make_regression
    from sklearn.tree import DecisionTreeRegressor
    import numpy as np

    # fix random seed for reproducibility
    random_seed = 10
    np.random.seed(random_seed)

    # create dummy regression dataset and fit tree model
    X, y = make_regression(1000, n_features=10, random_state=random_seed)
    clf = DecisionTreeRegressor(max_depth=10, random_state=random_seed).fit(X, y)

    # convert the tree to be usable like in TreeSHAP
    # tree_model = convert_tree(clf)

    x_input = X[:1]
    print("Output f(x):", clf.predict(x_input)[0])

    # TreeSHAP -------------------------------------------------------------------------------------

    tree_dict = {
        "children_left": clf.tree_.children_left.copy(),
        "children_right": clf.tree_.children_right.copy(),
        "children_default": clf.tree_.children_left.copy(),
        "features": clf.tree_.feature.copy(),
        "thresholds": clf.tree_.threshold.copy(),
        "values": clf.tree_.value.reshape(-1, 1).copy(),
        "node_sample_weight": clf.tree_.weighted_n_node_samples.copy(),
    }

    model = {
        "trees": [tree_dict]
    }

    # LinearTreeSHAP -------------------------------------------------------------------------------
    # explain the tree with LinearTreeSHAP
    start_time = time.time()
    explainer = LinearTreeSHAPExplainer(tree_model=tree_dict, n_features=x_input.shape[1])
    sv_linear_tree_shap = explainer.explain(x_input[0], 1)
    time_elapsed = time.time() - start_time

    print("Linear")
    print("Linear - time elapsed    ", time_elapsed)
    print("Linear - SVs (obs.)      ", sv_linear_tree_shap)
    print("Linear - sum SVs (obs.)  ", sv_linear_tree_shap.sum() + explainer.empty_prediction)
    print("Linear - empty pred      ", explainer.empty_prediction)
    print()

    start_time = time.time()
    explainer = LinearTreeSHAPExplainer(tree_model=tree_dict, n_features=x_input.shape[1])
    sv_linear_tree_shap = explainer.explain(x_input[0], 1)
    time_elapsed = time.time() - start_time

    print("Linear")
    print("Linear - time elapsed    ", time_elapsed)
    print("Linear - SVs (obs.)      ", sv_linear_tree_shap)
    print("Linear - sum SVs (obs.)  ", sv_linear_tree_shap.sum() + explainer.empty_prediction)
    print("Linear - empty pred      ", explainer.empty_prediction)
    print()

    start_time = time.time()
    explainer = LinearTreeSHAPExplainer(tree_model=tree_dict, n_features=x_input.shape[1])
    sv_linear_tree_shap = explainer.explain(x_input[0], 1)
    time_elapsed = time.time() - start_time

    print("Linear")
    print("Linear - time elapsed    ", time_elapsed)
    print("Linear - SVs (obs.)      ", sv_linear_tree_shap)
    print("Linear - sum SVs (obs.)  ", sv_linear_tree_shap.sum() + explainer.empty_prediction)
    print("Linear - empty pred      ", explainer.empty_prediction)
    print()
