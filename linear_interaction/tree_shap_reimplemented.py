"""This module contains the linear TreeSHAP class, which is a reimplementation of the original TreeSHAP algorithm."""
import time

from utils import tree_model

from collections import namedtuple

import numpy as np
from numpy.polynomial import Polynomial
from numpy.polynomial.polynomial import polydiv, polymul
from scipy.special import binom


class LinearTreeSHAPExplainer:

    def __init__(self, tree_model: tree_model, n_features: int = None, root_node_id: int = 0):
        # get the node attributes from the tree_model definition
        self.children_left: np.ndarray[int] = tree_model.children_left  # -1 for leaf nodes
        self.children_right: np.ndarray[int] = tree_model.children_right  # -1 for leaf nodes
        self.parents: np.ndarray[int] = tree_model.parents  # -1 for root node
        self.features: np.ndarray[int] = tree_model.features  # -2 for leaf nodes
        self.thresholds: np.ndarray[int] = tree_model.thresholds  # -2 for leaf nodes
        self.edge_heights: np.ndarray[int] = tree_model.edge_heights  # -1 for leaf nodes
        self.ancestors: np.ndarray[int] = tree_model.ancestors  # -1 for all nodes without an ancestor
        self.empty_rule_predictions: np.ndarray[float] = tree_model.empty_rule_predictions  # 0 for leaf nodes
        self.sample_weights: np.ndarray[float] = tree_model.sample_weights  # in [0, 1] for all nodes

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

        self.p_e_ancestor = np.zeros(self.n_nodes, dtype=float)

        # initialize an array to store the summary polynomials
        self.summary_polynomials: np.ndarray = np.empty(self.n_nodes, dtype=Polynomial)
        self.summary_polynomials_degree: np.ndarray[int] = np.zeros(self.n_nodes, dtype=int)

        # initialize an array to store the shapley values
        self.shapley_values: np.ndarray = np.zeros(self.n_features, dtype=float)

        # get empty prediction of model
        self.empty_prediction: float = float(np.sum(self.empty_rule_predictions[self.leaf_mask]))

    def explain(self, x: np.ndarray):
        # get an array index by the nodes
        initial_polynomial = Polynomial([1.])
        self._compute_summary_polynomials(x, self.root_node_id, initial_polynomial)
        #self._compute_shapley_values(x, self.root_node_id, initial_polynomial)
        # get an array indexed by the features
        self._aggregate_shapley(x, self.root_node_id)
        return self.shapley_values.copy()

    def _compute_shapley_values(
            self,
            x: np.ndarray,
            node_id: int,
            path_summary_poly: Polynomial,
            p_e_of_feature_ancestor: np.ndarray[float] = None,
            seen_features: np.ndarray[bool] = None,
            #seen_feature_sv: np.ndarray[bool] = None,
            went_left: bool = None,
            seen_features_degree: np.ndarray[int] = None,
            #height_of_feature_ancestors: dict = {},
            #p_e_of_feature_ancestors: dict = {}
            depth: int = 0
    ):
        # to store the p_e(x) of the feature ancestors when the feature was seen last in the path
        p_e_of_feature_ancestor: np.ndarray[float] = np.ones(self.n_features, dtype=float) if p_e_of_feature_ancestor is None else p_e_of_feature_ancestor

        # to store the degrees of the summary polynom when the feature was seen last in the path
        seen_features_degree: np.ndarray[int] = np.ones(self.n_features, dtype=int) if seen_features_degree is None else seen_features_degree

        # to store if a feature was already seen in the path in constant space
        seen_features: np.ndarray[bool] = np.zeros(self.n_features, dtype=bool) if seen_features is None else seen_features

        #seen_features_sv: np.ndarray[bool] = np.zeros(self.n_features, dtype=bool) if seen_feature_sv is None else seen_feature_sv

        # get node / edge information
        parent_id = self.parents[node_id]
        feature_id = self.features[parent_id]
        feature_threshold = self.thresholds[parent_id]
        edge_weight = self.sample_weights[node_id]
        p_e = self._get_p_e(x, feature_id, edge_weight, p_e_of_feature_ancestor, feature_threshold, went_left)

        # the node had an edge before, so we need to update the summary polynomial
        if node_id is not self.root_node_id:
            path_summary_poly = path_summary_poly * Polynomial([p_e, 1])
            # check weather the feature has been observed before in the path
            if seen_features[feature_id]:
                p_e_ancestor = p_e_of_feature_ancestor[feature_id]
                path_summary_poly = Polynomial(
                    polydiv(path_summary_poly.coef, Polynomial([p_e_ancestor, 1]).coef)[0]
                )

            #store previous feature info
            #seen_features[feature_id] = True
            #p_e_of_feature_ancestor[feature_id] = p_e
            #seen_features_degree[feature_id] = path_summary_poly.degree()

            # if node is a leaf (base case of the recursion)
            if self.leaf_mask[node_id]:
                leaf_prediction = self.leaf_predictions[node_id]
                leaf_prediction *= self.probabilities[node_id]
                self.summary_polynomials[node_id] = path_summary_poly * leaf_prediction
                self.summary_polynomials_degree[node_id] = self.summary_polynomials[node_id].degree()

        # if the node is a decision node then we continue the recursion down the tree
        if self.node_mask[node_id]:

            seen_features_input = seen_features.copy()
            p_e_of_feature_ancestor_input = p_e_of_feature_ancestor.copy()
            seen_features_degree_input = seen_features_degree.copy()

            if not node_id == self.root_node_id:
                seen_features_input[feature_id] = True
                p_e_of_feature_ancestor_input[feature_id] = p_e
                seen_features_degree_input[feature_id] = path_summary_poly.degree()
                #if seen_features[feature_id]:
                #    seen_features_degree_input[feature_id] += 1

            left_child_id = self.children_left[node_id]
            self._compute_shapley_values(
                x=x,
                node_id=left_child_id,
                path_summary_poly=path_summary_poly.copy(),
                p_e_of_feature_ancestor=p_e_of_feature_ancestor_input,
                seen_features=seen_features_input,
                went_left=True,
                seen_features_degree=seen_features_degree_input,
                #height_of_feature_ancestors=height_of_feature_ancestors,
                #p_e_of_feature_ancestors=p_e_of_feature_ancestors,
                depth=depth + 1
            )
            right_child_id = self.children_right[node_id]
            self._compute_shapley_values(
                x=x,
                node_id=right_child_id,
                path_summary_poly=path_summary_poly.copy(),
                p_e_of_feature_ancestor=p_e_of_feature_ancestor_input,
                seen_features=seen_features_input,
                went_left=False,
                seen_features_degree=seen_features_degree_input,
                #height_of_feature_ancestors=height_of_feature_ancestors,
                #p_e_of_feature_ancestors=p_e_of_feature_ancestors
                depth=depth + 1
            )
            # add the summary polynomials of the left and right child nodes together
            added_polynomial = self._special_polynomial_addition(
                p1=self.summary_polynomials[left_child_id],
                p2=self.summary_polynomials[right_child_id]
            )
            # store the summary polynomial of the current node
            self.summary_polynomials[node_id] = added_polynomial
            self.summary_polynomials_degree[node_id] = added_polynomial.degree()

        if node_id is not self.root_node_id:
            # Update Shapley values
            quotient = Polynomial(
                polydiv(self.summary_polynomials[node_id].coef, Polynomial([p_e, 1]).coef)[0])
            psi = self._psi(quotient)
            self.shapley_values[feature_id] += (p_e - 1) * psi

            # the part below is only needed if the feature was already encountered in the path
            if seen_features[feature_id]:
                #p_e_ancestor = p_e_of_feature_ancestor[feature_id]

                #degree_of_summary_poly = self.summary_polynomials[node_id].degree()
                #degree_of_ancestor = seen_features_degree[feature_id]

                # get the numerator of psi
                #psi_numerator = self.summary_polynomials[node_id].coef
                #for i in range(degree_of_summary_poly - degree_of_ancestor):
                #    psi_numerator = polydiv(psi_numerator, Polynomial([1, 1]).coef)[0]

                # get the denominator of psi
                #psi_denominator = Polynomial([p_e_ancestor, 1])

                # get psi for ancestor
                #psi_quotient = Polynomial(polydiv(psi_numerator, psi_denominator.coef)[0])
                #psi_ancestor = self._psi(psi_quotient)

                #self.shapley_values[feature_id] -= (p_e_ancestor - 1) * psi_ancestor

                # maybe mistake in paper?
                p_e_ancestor = p_e_of_feature_ancestor[feature_id]
                degree_of_ancestor = seen_features_degree[feature_id]
                #degree_of_summary_poly = self.summary_polynomials[node_id].degree()

                psi_denominator = Polynomial([1, 1]) ** (depth - degree_of_ancestor)
                psi_denominator = Polynomial([p_e_ancestor, 1]) * psi_denominator
                quotient_ancestor = Polynomial(polydiv(self.summary_polynomials[node_id].coef, psi_denominator.coef)[0])
                psi_ancestor = self._psi(quotient_ancestor)
                self.shapley_values[feature_id] -= (p_e_ancestor - 1) * psi_ancestor

    def _compute_summary_polynomials(
            self,
            x: np.ndarray,
            node_id: int,
            path_summary_poly: Polynomial,
            went_left: bool = True,
    ):
        """Compute the summary polynomials for each node in the tree.

        The summary polynomials are recursively computed in a top-down fashion while traversing
            the tree from the root node to the leaf nodes. The summary polynomials are computed for
            each node in the path from the root node to the current node. The summary polynomials
            are stored in a list indexed by the node id.

        Args:
            x (np.ndarray): The input data.
            node_id (int): The id of the node to compute the summary polynomials for.
            path_summary_poly (Polynomial): The current summary polynomial in the path from the root
                node to the current node.
            went_left (bool, optional): Whether the current node was reached by going left or right
                from the parent node. Defaults to True (i.e. left).
        """

        if node_id is not self.root_node_id:
            # get the edge information with the node as the head
            parent_id = self.parents[node_id]
            feature_id = self.features[parent_id]
            feature_threshold = self.thresholds[parent_id]
            edge_weight = self.sample_weights[node_id]

            # get factor p_e of the edge
            p_e = self._get_p_e(x, feature_id, edge_weight, feature_threshold, went_left)
            path_summary_poly = path_summary_poly * Polynomial([p_e, 1])

            # check weather the feature has been observed before in the path
            if self.ancestors[node_id] > 0:
                # get the ancestor node id
                ancestor_node_id = self.ancestors[node_id]
                # the value in feature_path_weights contains p_e of an ancestor
                p_e_ancestor = feature_path_weights[feature_id]
                quotient_polynomial = Polynomial(polydiv(path_summary_poly.coef, Polynomial([p_e_ancestor, 1]).coef)[0])
                path_summary_poly = quotient_polynomial

            feature_path_weights[feature_id] = p_e

            # if the current node is a leaf node, scale the summary polynomial with the leaf output
            if self.leaf_mask[node_id]:
                leaf_prediction = self.leaf_predictions[node_id]
                leaf_prediction *= self.probabilities[node_id]
                self.summary_polynomials[node_id] = path_summary_poly * leaf_prediction
                self.summary_polynomials_degree[node_id] = self.summary_polynomials[node_id].degree()

        # compute the summary polynomials for the left and right child nodes
        left_child_id, right_child_id = self.children_left[node_id], self.children_right[node_id]
        if self.node_mask[node_id]:  # if the node is not a leaf node (recursion stop condition)
            self._compute_summary_polynomials(
                x=x,
                node_id=left_child_id,
                path_summary_poly=path_summary_poly.copy(),
                feature_path_weights=feature_path_weights.copy(),
                seen_features=seen_features.copy(),
                went_left=True,
                ancestor_node_ids=ancestor_node_ids.copy()
            )
            self._compute_summary_polynomials(
                x=x,
                node_id=right_child_id,
                path_summary_poly=path_summary_poly.copy(),
                feature_path_weights=feature_path_weights.copy(),
                seen_features=seen_features.copy(),
                went_left=False,
                ancestor_node_ids=ancestor_node_ids.copy()
            )
            # add the summary polynomials of the left and right child nodes together
            added_polynomial = self._special_polynomial_addition(
                p1=self.summary_polynomials[left_child_id],
                p2=self.summary_polynomials[right_child_id]
            )
            # store the summary polynomial of the current node
            self.summary_polynomials[node_id] = added_polynomial
            self.summary_polynomials_degree[node_id] = self.summary_polynomials[node_id].degree()

    def _aggregate_shapley(
            self,
            x: np.ndarray,
            node_id: int,
            went_left: bool = None,
            height_of_feature_ancestors: np.ndarray[int] = None,
            feature_path_weights: np.ndarray[float] = None,
            seen_features: np.ndarray[bool] = None,
            depth: int = 0
    ):
        """ Aggregate the Shapley values for each feature in the tree.

        Args:
            x (np.ndarray): The input data.
            node_id (int): The id of the node to compute the summary polynomials for.
            went_left (bool, optional): Whether the current node was reached by going left or right
                from the parent node. Defaults to True (i.e. left).
            depth_in_tree (int, optional): The height of the current node in the tree.
                Defaults to 0.
            height_of_feature_ancestors (np.ndarray[int], optional): The height of the feature
                ancestors in the tree. Defaults to None.
            feature_path_weights (np.ndarray[float], optional): The current feature path weights
                in the path from the root node to the current node. Defaults to None.
            seen_features (np.ndarray[bool], optional): Whether a feature has been seen in the
                path from the root node to the current node. Defaults to None.
        """
        if feature_path_weights is None:
            feature_path_weights: np.ndarray[float] = np.ones(self.n_features, dtype=float)

        if height_of_feature_ancestors is None:
            height_of_feature_ancestors: np.ndarray[int] = np.full(self.n_features, -1, dtype=int)

        if seen_features is None:
            seen_features: np.ndarray[bool] = np.zeros(self.n_features, dtype=bool)

        # get the edge information with the node as the head
        parent_id = self.parents[node_id]
        feature_id = self.features[parent_id]
        edge_weight = self.sample_weights[node_id]
        feature_threshold = self.thresholds[parent_id]

        # get weight of the edge
        p_e = self._get_p_e(x, feature_id, edge_weight, feature_path_weights, feature_threshold, went_left)

        # if node has children (recursion case)
        if not self.leaf_mask[node_id]:
            seen_features_input = seen_features.copy()
            seen_features_input[self.features[self.parents[node_id]]] = True

            feature_path_weights_input = feature_path_weights.copy()
            feature_path_weights_input[feature_id] = p_e

            left_child_id = self.children_left[node_id]
            height_of_feature_ancestors_input = height_of_feature_ancestors.copy()
            height_of_feature_ancestors_input[feature_id] = self.summary_polynomials[node_id].degree()
            self._aggregate_shapley(
                x=x,
                node_id=left_child_id,
                went_left=True,
                feature_path_weights=feature_path_weights_input,
                seen_features=seen_features_input,
                height_of_feature_ancestors=height_of_feature_ancestors_input,
                depth=depth + 1
            )

            right_child_id = self.children_right[node_id]
            height_of_feature_ancestors_input = height_of_feature_ancestors.copy()
            height_of_feature_ancestors_input[feature_id] = self.summary_polynomials[node_id].degree()
            self._aggregate_shapley(
                x=x,
                node_id=right_child_id,
                went_left=False,
                feature_path_weights=feature_path_weights_input,
                seen_features=seen_features_input,
                height_of_feature_ancestors=height_of_feature_ancestors_input,
                depth=depth + 1
            )

        # if v is not the root
        if node_id is not self.root_node_id:
            quotient_pos = polydiv(self.summary_polynomials[node_id].coef, Polynomial([p_e, 1]).coef)[0]
            quotient_pos = Polynomial(quotient_pos)
            psi_pos = self._psi(quotient_pos)
            self.shapley_values[feature_id] += (p_e - 1) * psi_pos

            if seen_features[feature_id]:
                # the value in feature_path_weights contains p_e of the ancestor
                p_e_ancestor = feature_path_weights[feature_id]
                height_of_ancestor = height_of_feature_ancestors[feature_id]
                #depth = self.summary_polynomials[node_id].degree()

                degree_of_ancestor = self.summary_polynomials_degree[self.ancestor_nodes[node_id]]
                degree_of_node = self.summary_polynomials_degree[node_id]

                psi_numerator = Polynomial([1, 1]) ** (degree_of_ancestor - degree_of_node)
                psi_numerator = self.summary_polynomials[node_id] * psi_numerator
                psi_denominator = Polynomial([p_e_ancestor, 1])
                quotient_ancestor = Polynomial(
                    polydiv(psi_numerator.coef, psi_denominator.coef)[0])
                psi_ancestor = self._psi(quotient_ancestor)
                self.shapley_values[feature_id] -= (p_e_ancestor - 1) * psi_ancestor

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
        p_e = feature_path_weights[feature_id] * 1 / edge_weight
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
    DO_PLOTTING = False

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

    # fix random seed for reproducibility
    random_seed = 10
    np.random.seed(random_seed)

    # create dummy regression dataset and fit tree model
    X, y = make_regression(1000, n_features=10, random_state=random_seed)
    clf = DecisionTreeRegressor(max_depth=5, random_state=random_seed).fit(X, y)

    # convert the tree to be usable like in TreeSHAP
    tree_model = convert_tree(clf)

    x_input = X[:1]
    print("Output f(x):", clf.predict(x_input)[0])

    if DO_PLOTTING:
        plt.figure(dpi=150)
        plot_tree(clf)
        plt.savefig("tree.pdf")

    # TreeSHAP -------------------------------------------------------------------------------------

    tree_dict = {
        "children_left": tree_model.children_left.copy(),
        "children_right": tree_model.children_right.copy(),
        "children_default": tree_model.children_left.copy(),
        "features": tree_model.features.copy(),
        "thresholds": tree_model.thresholds.copy(),
        "values": tree_model.leaf_predictions.reshape(-1, 1).copy(),
        "node_sample_weight": tree_model.sample_weights.copy(),
    }

    model = {
        "trees": [tree_dict]
    }

    if DO_TREE_SHAP:
        # explain the tree with observational TreeSHAP
        start_time = time.time()
        explainer_shap = TreeExplainer(model, feature_perturbation="tree_path_dependent")
        sv_tree_shap = explainer_shap.shap_values(x_input)
        time_elapsed = time.time() - start_time
        print("TreeSHAP - SVs (obs.)    ", sv_tree_shap)
        print("TreeSHAP - sum SVs (obs.)", sv_tree_shap.sum() + explainer_shap.expected_value)
        print("TreeSHAP - time elapsed  ", time_elapsed)
        print("TreeSHAP - empty pred    ", explainer_shap.expected_value)

    # LinearTreeSHAP -------------------------------------------------------------------------------

    start_time = time.time()
    explainer = LinearTreeSHAPExplainer(tree_model=tree_model, n_features=x_input.shape[1])
    sv_linear_tree_shap = explainer.explain(x_input[0])
    time_elapsed = time.time() - start_time
    print("Linear - SVs (obs.)      ", sv_linear_tree_shap)
    print("Linear - sum SVs (obs.)  ", sv_linear_tree_shap.sum() + explainer.empty_prediction)
    print("Linear - time elapsed    ", time_elapsed)
    print("Linear - empty pred      ", explainer.empty_prediction)
