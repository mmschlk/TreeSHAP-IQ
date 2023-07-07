"""This module contains the linear TreeSHAP class, which is a reimplementation of the original TreeSHAP algorithm."""
import numpy as np
from numpy.polynomial import Polynomial
from scipy.special import binom


class LinearTreeSHAPExplainer:

    def __init__(self, tree_model, n_features, root_node_id: int = 0):
        # get the node attributes from the tree_model definition
        self.children_left: np.ndarray = tree_model.children_left  # -1 for leaf nodes
        self.children_right: np.ndarray = tree_model.children_right  # -1 for leaf nodes
        self.leaf_predictions: np.ndarray = tree_model.leaf_predictions  # zero for non-leaf nodes
        self.features: np.ndarray = tree_model.features  # -2 for leaf nodes
        self.thresholds: np.ndarray = tree_model.thresholds  # -2 for leaf nodes
        self.sample_weights: np.ndarray = tree_model.sample_weights  # in [0, 1] for all nodes

        self.parents: np.ndarray = tree_model.parents  # -1 for the root node

        # set the root node id and number of features
        self.root_node_id: int = root_node_id
        self.n_features: int = n_features

        # get the number of nodes and the node ids
        self.n_nodes: int = len(self.children_left)
        self.nodes: np.ndarray = np.arange(self.n_nodes)

        # get the leaf and node masks
        self.leaf_mask: np.ndarray = self.children_left == -1
        self.node_mask: np.ndarray = ~self.leaf_mask

        # initialize an array to store the summary polynomials
        self.summary_polynomials: np.ndarray = np.empty_like(self.n_nodes, dtype=Polynomial)

        # initialize an array to store the shapley values
        self.shapley_values: np.ndarray = np.empty_like(self.n_features, dtype=float)

        self.edges: dict[tuple[int, int]] = {}
        self.features_edges: dict[int, set] = {feature: set() for feature in range(self.n_features)}

    def explain(self, x: np.ndarray):
        active_edges = self._get_active_edges(x)
        # get an array index by the nodes
        initial_polynomial = Polynomial([1.])
        self._compute_summary_polynomials(x, self.root_node_id, initial_polynomial)
        # get an array indexed by the features
        self._aggregate_shapley(x, self.root_node_id)
        return self.shapley_values

    def _compute_summary_polynomials(
            self,
            x: np.ndarray,
            node_id: int,
            path_summary_poly: Polynomial,
            feature_path_weights: np.ndarray[float] = None,
            height_in_tree: int = 0,
            seen_features: np.ndarray[bool] = None,
            went_left: bool = True
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
            feature_path_weights (np.ndarray[float], optional): The current feature path weights
                in the path from the root node to the current node. Defaults to None.
            height_in_tree (int, optional): The height of the current node in the tree.
                Defaults to 0.
            seen_features (np.ndarray[bool], optional): Whether a feature has been seen in the
                path from the root node to the current node. Defaults to None.
            went_left (bool, optional): Whether the current node was reached by going left or right
                from the parent node. Defaults to True (i.e. left).
        """
        # is used to store weight information for this path per feature
        if feature_path_weights is None:
            feature_path_weights: np.ndarray = np.ones(self.n_features, dtype=float)

        if seen_features is None:
            seen_features: np.ndarray = np.zeros(self.n_features, dtype=bool)

        # contains the summary polynomials for each node (G in the paper)
        # if node is a leaf (base case of the recursion)
        if self.leaf_mask[node_id]:
            # TODO might be wrong in the paper they do matrix multiply
            leaf_prediction = self.leaf_predictions[node_id]
            leaf_prediction *= np.prod(feature_path_weights[seen_features])
            self.summary_polynomials[node_id] = path_summary_poly * leaf_prediction
        else:  # node is not a leaf, and we traverse the tree (recursion case)
            # if node is not the root node then there is an edge with a feature here
            if node_id is not self.root_node_id:

                # get the edge information with the node as the head
                parent_id = self.parents[node_id]
                feature_id = self.features[parent_id]
                edge_weight = self.sample_weights[node_id]

                # get weight of the edge
                p_e = self._get_p_e(x, feature_id, edge_weight, feature_path_weights, parent_id, went_left)
                path_summary_poly = path_summary_poly + Polynomial([p_e, 1])

                # check weather the feature has been observed before in the path
                if seen_features[feature_id]:
                    # the value in feature_path_weights contains p_e of the ancestor
                    p_e_ancestor = feature_path_weights[feature_id]
                    path_summary_poly /= Polynomial([p_e_ancestor, 1])
                    feature_path_weights[feature_id] *= p_e
                else:
                    feature_path_weights[feature_id] = p_e
                    seen_features[feature_id] = True

            # compute the summary polynomials for the left and right child nodes
            left_child_id, right_child_id = self.children_left[node_id], self.children_right[
                node_id]
            self._compute_summary_polynomials(
                x=x,
                node_id=left_child_id,
                path_summary_poly=path_summary_poly,
                feature_path_weights=feature_path_weights,
                height_in_tree=height_in_tree + 1,
                seen_features=seen_features,
                went_left=True
            )
            self._compute_summary_polynomials(
                x=x,
                node_id=right_child_id,
                path_summary_poly=path_summary_poly,
                feature_path_weights=feature_path_weights,
                height_in_tree=height_in_tree + 1,
                seen_features=seen_features,
                went_left=False
            )
            # add the summary polynomials of the left and right child nodes together
            added_polynomial = self._special_polynomial_addition(
                p1=self.summary_polynomials[left_child_id],
                p2=self.summary_polynomials[right_child_id]
            )
            # store the summary polynomial of the current node
            self.summary_polynomials[node_id] = added_polynomial

    def _aggregate_shapley(
            self,
            x: np.ndarray,
            node_id: int,
            went_left: bool = None,
            height_in_tree: int = 0,
            height_of_feature_ancestors: np.ndarray = None,
            feature_path_weights: np.ndarray[float] = None,
            seen_features: np.ndarray[bool] = None
    ):
        """ Aggregate the Shapley values for each feature in the tree.

        Args:
            x (np.ndarray): The input data.
            node_id (int): The id of the node to compute the summary polynomials for.
            went_left (bool, optional): Whether the current node was reached by going left or right
                from the parent node. Defaults to True (i.e. left).
            height_in_tree (int, optional): The height of the current node in the tree.
                Defaults to 0.
            height_of_feature_ancestors (np.ndarray[int], optional): The height of the feature
                ancestors in the tree. Defaults to None.
            feature_path_weights (np.ndarray[float], optional): The current feature path weights
                in the path from the root node to the current node. Defaults to None.
            seen_features (np.ndarray[bool], optional): Whether a feature has been seen in the
                path from the root node to the current node. Defaults to None.
        """
        if feature_path_weights is None:
            feature_path_weights: np.ndarray = np.ones(self.n_features, dtype=float)

        if height_of_feature_ancestors is None:
            height_of_feature_ancestors: np.ndarray = np.full(self.n_features, -1, dtype=int)

        if seen_features is None:
            seen_features: np.ndarray = np.zeros(self.n_features, dtype=bool)

        # if node has children (recursion case)
        if not self.leaf_mask[node_id]:
            # get children node ids
            left_child_id, right_child_id = self.children_left[node_id], self.children_right[
                node_id]
            self._aggregate_shapley(x=x, node_id=left_child_id, went_left=True)
            self._aggregate_shapley(x=x, node_id=right_child_id, went_left=False)

        # if v is not the root
        if node_id is not self.root_node_id:

            # get the edge information with the node as the head
            parent_id = self.parents[node_id]
            feature_id = self.features[parent_id]
            edge_weight = self.sample_weights[node_id]

            # get weight of the edge
            p_e = self._get_p_e(x, feature_id, edge_weight, feature_path_weights, parent_id, went_left)

            # check weather the feature has been observed before in the path
            if seen_features[feature_id]:
                # the value in feature_path_weights contains p_e of the ancestor
                p_e_ancestor = feature_path_weights[feature_id]
                feature_path_weights[feature_id] *= p_e
                height_of_ancestor = height_of_feature_ancestors[feature_id]
            else:
                p_e_ancestor = 1
                height_of_ancestor = 0  # TODO could be wrong
                feature_path_weights[feature_id] = p_e
                seen_features[feature_id] = True
                height_of_feature_ancestors[feature_id] = height_in_tree

            psi = self._psi(self.summary_polynomials[node_id] / Polynomial([p_e, 1]))
            self.shapley_values[feature_id] += (edge_weight - 1) * psi

            psi_numerator = self.summary_polynomials[node_id] * Polynomial([1, 1])
            psi_numerator **= (height_of_ancestor - height_in_tree)
            psi_denominator = Polynomial([p_e_ancestor, 1])
            psi = self._psi(psi_numerator / psi_denominator)
            self.shapley_values[feature_id] -= (p_e_ancestor - 1) * psi

    @staticmethod
    def _get_binomial_polynom(degree: int) -> Polynomial:
        """Get a reciprocal binomial polynomial with degree d.

        The reciprocal binomial polynomial is defined as `\sum_{i=0}^{d} binom(d, i)^{-1} * x^{i}`.

        Args:
            degree (int): The degree of the binomial polynomial.

        Returns:
            Polynomial: The reciprocal binomial polynomial.
        """
        return Polynomial([1 / binom(degree, i) for i in range(degree + 1)])

    @staticmethod
    def _special_polynomial_addition(p1: Polynomial, p2: Polynomial) -> Polynomial:
        """Add two polynomials of different degrees.

        Args:
            p1 (Polynomial): The first polynomial with degree d1.
            p2 (Polynomial): The second polynomial with degree d2.

        Returns:
            Polynomial: The sum of the two polynomials with degree max(d1, d2).
        """
        # a polynomial of (1 + x) with degree d1 - d2
        p3 = Polynomial([1, 1]) ** (p1.degree() - p2.degree())
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
        binomial_polynomial = self._get_binomial_polynom(polynomial.degree())
        inner_product = np.inner(polynomial.coef, binomial_polynomial.coef)
        return inner_product / (polynomial.degree() + 1)

    def _get_p_e(
            self,
            x: np.ndarray,
            feature_id: int,
            edge_weight: float,
            feature_path_weights: np.ndarray[float],
            parent_id: int,
            went_left: bool
    ) -> float:
        """Get the weight p_e of the decision for the feature given the instance.

        Args:
            x: The input data.
            feature_id (int): The id of the feature of the edge.
            edge_weight (float): The weight of the edge.
            feature_path_weights (np.ndarray[float]): The weights of the feature paths.
            parent_id (int): The id of the parent node of the edge.
            went_left (bool): Whether the instance went left or right at the parent node.

        Returns:
            float: The weight of the decision for the feature given the instance.
        """
        activation = 0
        if went_left:
            if x[feature_id] <= self.features[parent_id]:
                activation = 1
        else:
            if x[feature_id] > self.features[parent_id]:
                activation = 1
        p_e = activation * edge_weight * feature_path_weights[feature_id]
        return 1 / p_e





if __name__ == "__main__":
    from linear_interaction.utils import copy_tree

    from sklearn.datasets import make_regression
    from sklearn.tree import DecisionTreeRegressor, plot_tree
    import matplotlib.pyplot as plt
    import numpy as np

    # fix random seed for reproducibility
    random_seed = 10
    np.random.seed(random_seed)

    # create dummy regression dataset and fit tree model
    x, y = make_regression(1000, n_features=10)
    clf = DecisionTreeRegressor(max_depth=6, random_state=random_seed).fit(x, y)

    # plt.figure(dpi=800)
    # plot_tree(clf)
    # plt.savefig("tree.pdf")
    # plt.show()

    # copy the tree
    tree_converted = copy_tree(clf.tree_)

    # print(tree_converted)

    explainer = LinearTreeSHAPExplainer(tree_model=tree_converted, n_features=x.shape[1])
    explainer.explain(x[:2])
