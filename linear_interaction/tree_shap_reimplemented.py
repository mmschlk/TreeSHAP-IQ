"""This module contains the linear TreeSHAP class, which is a reimplementation of the original TreeSHAP algorithm."""
import numpy as np
from numpy.polynomial import Polynomial
from scipy.special import binom


class LinearTreeSHAPExplainer:

    def __init__(self, tree_model, n_features):
        self.root_node_id: int = 0
        self.n_nodes: int = tree_model.num_nodes
        self.n_features: int = n_features
        self.max_depth: int = tree_model.max_depth

        # node attributes
        self.nodes = np.arange(self.n_nodes)
        self.children_left: np.ndarray = tree_model.children_left  # -1 for leaf nodes
        self.children_right: np.ndarray = tree_model.children_right  # -1 for leaf nodes
        self.node_parents: np.ndarray = tree_model.parents  # -1 for the root node
        self.leaf_predictions: np.ndarray = tree_model.leaf_predictions  # zero for non-leaf nodes
        self.split_features: np.ndarray = tree_model.features  # -1 for nodes
        self.thresholds = tree_model.thresholds
        self.leaf_mask = self.children_left == -1
        self.node_mask = ~self.leaf_mask

        self.summary_polynomials = np.empty_like(self.n_nodes, dtype=Polynomial)
        self.shapley_values = np.empty_like(self.n_features, dtype=float)

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
            path_summary_poly: Polynomial
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
        """
        # contains the summary polynomials for each node (G in the paper)
        # if node is a leaf (base case of the recursion)
        if self.leaf_mask[node_id]:
            self.summary_polynomials[node_id] = path_summary_poly * self.leaf_predictions[node_id]
        else:  # node is not a leaf, and we traverse the tree (recursion case)
            # if node is not the root node then there is an edge with a feature here
            if node_id is not self.root_node_id:
                # get the edge with the node as the head
                edge_id = self._get_edge(node_id)  # e in the paper
                # add the edge factor to the summary polynomial of this path
                edge_weight = self._get_edge_weight(x, edge_id)  # p_e(x) in the paper
                path_summary_poly = path_summary_poly + Polynomial([edge_weight, 1])
                ancestor_edge_id = self._get_ancestor_edge(edge_id)  # e' in the paper
                if ancestor_edge_id is not None:
                    path_summary_poly /= Polynomial([self._get_edge_weight(x, ancestor_edge_id), 1])
            # compute the summary polynomials for the left and right child nodes
            left_child_id, right_child_id = self.children_left[node_id], self.children_right[
                node_id]
            self._compute_summary_polynomials(x=x, node_id=left_child_id,
                                              path_summary_poly=path_summary_poly)
            self._compute_summary_polynomials(x=x, node_id=right_child_id,
                                              path_summary_poly=path_summary_poly)
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
            node_id: int
    ):
        # if node has children (recursion case)
        if not self.leaf_mask[node_id]:
            # get children node ids
            left_child_id, right_child_id = self.children_left[node_id], self.children_right[
                node_id]
            self._aggregate_shapley(x=x, node_id=left_child_id)
            self._aggregate_shapley(x=x, node_id=right_child_id)
        # if v is not the root
        if node_id is not self.root_node_id:
            # get edge with node_id as head
            parent_id = self.node_parents[node_id]
            edge_feature_id = self.split_features[parent_id]
            edge_weight = self._get_edge_weight(x, node_id)
            psi = self._psi(self.summary_polynomials[node_id] / Polynomial([edge_weight, 1]))
            self.shapley_values[edge_feature_id] += (edge_weight - 1) * psi
            ancestor_edge_id = self._get_ancestor_edge(node_id)
            ancestor_edge_weight = self._get_edge_weight(x, ancestor_edge_id)
            psi_numerator = self.summary_polynomials[node_id] * Polynomial([1, 1])
            psi_numerator **= (self._get_edge_height(ancestor_edge_id) - self._get_edge_height(node_id))
            psi_denominator = Polynomial([ancestor_edge_weight, 1])
            psi = self._psi(psi_numerator / psi_denominator)
            self.shapley_values[edge_feature_id] -= (ancestor_edge_weight - 1) * psi

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

    def _get_active_edges(self, x: np.ndarray) -> np.ndarray:
        """Get an array of active edges for each instance in x. An active edge is an edge that is
            traversed by the instance.

        Args:
            x: The input data.

        Returns:
            An array of active edges for each instance in x.
        """
        active_edges: np.ndarray = np.zeros((x.shape[0], self.children_left.shape[0]), dtype=bool)
        left_active = x[:, self.split_features[self.node_mask]] <= self.split_features[
            self.node_mask]
        active_edges[:, self.children_left[self.node_mask]] = left_active
        right_active = x[:, self.split_features[self.node_mask]] > self.split_features[
            self.node_mask]
        active_edges[:, self.children_right[self.node_mask]] = right_active
        return active_edges


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
