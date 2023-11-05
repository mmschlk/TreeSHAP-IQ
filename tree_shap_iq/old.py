"""This module contains the TreeSHAP-IQ class."""
import typing

import numpy as np
from numpy.polynomial import Polynomial
from numpy.polynomial.polynomial import polydiv
from scipy.special import binom

from tree_shap_iq.conversion import TreeModel

try:
    from .utils import _get_parent_array, powerset
except ImportError:
    from utils import _get_parent_array, powerset


class TreeShapIQ:

    def __init__(
            self,
            tree_model: typing.Union[dict, TreeModel],
            max_interaction_order: int = 1,
            observational: bool = True,
            background_dataset: np.ndarray = None,
            n_features: int = None,
            root_node_id: int = 0,
            interaction_type: str = "SII"
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
        self.interaction_type = interaction_type
        # get the node attributes from the tree_model definition
        self.children_left: np.ndarray[int] = tree_model["children_left"]  # -1 for leaf nodes
        self.children_right: np.ndarray[int] = tree_model["children_right"]  # -1 for leaf nodes
        self.parents: np.ndarray[int] = _get_parent_array(self.children_left,
                                                          self.children_right)  # -1 for the root node
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
            self.node_sample_weight = tree_model["node_sample_weight"]
        else:
            #raise NotImplementedError(
            #    "Interventional Shapley interactions are not implemented yet.")
            self.node_sample_weight = self._get_interventional_sample_weights(background_dataset)

        # precompute subsets that include each feature and their positions
        self.max_order: int = max_interaction_order
        self.shapley_interactions: np.ndarray[float] = np.zeros(
            int(binom(self.n_features, self.max_order)), dtype=float)
        self.shapley_interactions_lookup: dict = self._generate_interactions_lookup(self.n_features, self.max_order)
        self.subset_updates, self.subset_updates_pos = self._precompute_subsets_with_feature()

        edge_tree: tuple = self.extract_edge_information_from_tree(max_interaction_order)
        self.parents, self.ancestors, self.ancestor_nodes, self.p_e_values, self.p_e_storages, self.weights, self.empty_predictions, self.edge_heights, self.max_depth, self.last_feature_node_in_path, self.interaction_height = edge_tree
        self.has_ancestors = self.ancestors > -1  # TODO ancestor_nodes noch falsch
        self.p_e_values_1 = np.ones(self.n_nodes, dtype=float)

        # initialize an array to store the summary polynomials
        self.summary_polynomials: np.ndarray = np.empty(self.n_nodes, dtype=Polynomial)

        # initialize an array to store the shapley values
        self.shapley_values: np.ndarray = np.zeros(self.n_features, dtype=float)

        # get empty prediction of model
        self.empty_prediction: float = float(np.sum(self.empty_predictions[self.leaf_mask]))
        if tree_model["empty_prediction"] is not None:
            self.empty_prediction = tree_model["empty_prediction"]

        # stores the interaction scores up to a given order
        self.subset_ancestors: dict = {}
        self._precalculate_interaction_ancestors()

        # improved calculations
        self.interpolation_size = self.n_features
        self.D = np.polynomial.chebyshev.chebpts2(self.interpolation_size)
        self.D_powers = self.cache(self.D)
        self.D_powers_IP = self.cache(-self.D)
        self.Ns = self.get_N(self.D)
        self.Ns_id = self.get_N_id(self.D)
        self.activation = np.zeros_like(self.children_left, dtype=bool)

        # new for improved calculations
        self.activations: np.ndarray[bool] = np.zeros(self.n_nodes, dtype=bool)

    def explain(
            self,
            x: np.ndarray,
            order: int = 1,
            mode: str = "superfast"
    ) -> np.ndarray[float]:
        """Computes the Shapley Interaction values for a given instance x and interaction order.
            This function is the main explanation function of this class.

        Args:
            x (np.ndarray): Instance to be explained.
            order (int, optional): Order of the interactions. Defaults to 1.
            mode (str, optional): The mode of the computation. The following modes are supported:
                - "original": The original LinearTreeSHAP algorithm generalized to interactions.
                - "leaf": The original LinearTreeSHAP algorithm generalized to interactions, but
                    only computes the Shapley interactions in the leaf nodes.
                - "interpolation-original": The original LinearTreeSHAP algorithm generalized to
                    interactions, but uses interpolation to compute the summary polynomials.
                - "interpolation-leaf": The original LinearTreeSHAP algorithm generalized to
                    interactions, but uses interpolation to compute the summary polynomials and only
                    computes the Shapley interactions in the leaf nodes.
                - "superfast": The improved LinearTreeSHAP algorithm generalized to interactions.
            Defaults to "superfast".
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
        if mode == "original":
            self._compute_original(x, 0, initial_polynomial)
        elif mode == "leaf":
            self._compute_only_in_leaf(x, 0, initial_polynomial)
        elif mode == "interpolation-original":
            self._compute_shapley_values_interpolation(x, 0, original=True)
        elif mode == "interpolation-leaf":
            self._compute_shapley_values_interpolation(x, 0, original=False)
        elif mode == "superfast":
            self.D_order = np.polynomial.chebyshev.chebpts2(self.max_depth)
            self._compute_shapley_values_superfast(x, 0)
        else:
            raise ValueError(f"Mode {mode} is not supported.")
        return self.shapley_interactions.copy()

    def _compute_only_in_leaf(
            self,
            x: np.ndarray,
            node_id: int,
            path_summary_poly: Polynomial,
            p_e_storage: np.ndarray[float] = None,
    ):
        # reset activations for new calculations
        if node_id == 0:
            self.activations = np.zeros(self.n_nodes, dtype=bool)

        if p_e_storage is None:
            p_e_storage = np.ones(self.n_features, dtype=float)

        # get node / edge information
        left_child, right_child = self.children_left[node_id], self.children_right[node_id]
        ancestor_id = self.ancestors[node_id]
        parent_id = self.parents[node_id]
        feature_id = self.features[parent_id]
        child_edge_feature = self.features[node_id]
        feature_threshold = self.thresholds[node_id]
        activations = self.activations
        is_leaf = left_child < 0

        # if node is not a leaf -> set activations for children nodes accordingly
        if not is_leaf:
            if x[child_edge_feature] <= feature_threshold:
                activations[left_child], activations[right_child] = True, False
            else:
                activations[left_child], activations[right_child] = False, True

        # if node is not the root node -> calculate the summary polynomials
        if node_id is not self.root_node_id:

            # set the activations of the current node in relation to the ancestor (for setting p_e to zero)
            if self.has_ancestors[node_id]:
                activations[node_id] &= activations[ancestor_id]

            # if node is active get the correct p_e value
            p_e_current = self.p_e_values[node_id] if activations[node_id] else 0.
            p_e_storage[feature_id] = p_e_current

            # update summary polynomial
            path_summary_poly = path_summary_poly * Polynomial([p_e_current, 1])

            # remove previous polynomial factor if node has ancestors
            if self.has_ancestors[node_id]:
                p_e_ancestor = self.p_e_values[ancestor_id] if activations[ancestor_id] else 0.
                path_summary_poly = Polynomial(
                    polydiv(path_summary_poly.coef, Polynomial([p_e_ancestor, 1]).coef)[0])

        # if node is leaf -> add the empty prediction to the summary polynomial and store it
        # if node is leaf -> compute the shapley interactions for the leaf node
        if is_leaf:
            self.summary_polynomials[node_id] = path_summary_poly * self.empty_predictions[node_id]
            q_S, Q_S = {}, {}
            for S, pos in zip(self.shapley_interactions_lookup,
                              self.shapley_interactions_lookup.values()):
                # compute interaction factor and polynomial for aggregation below
                q_S[S] = self._compute_p_e_interaction_fast(S, p_e_storage)
                if q_S[S] != 0:
                    Q_S[S] = self._compute_poly_interaction(S, p_e_storage)
                    # update interactions for all interactions that contain feature_id
                    quotient = Polynomial(
                        polydiv(self.summary_polynomials[node_id].coef, Q_S[S].coef)[0])
                    psi = self._psi(quotient)
                    self.shapley_interactions[pos] += q_S[S] * psi

        else:  # not a leaf -> continue recursion
            self._compute_only_in_leaf(x, left_child, path_summary_poly, p_e_storage.copy())
            self._compute_only_in_leaf(x, right_child, path_summary_poly, p_e_storage.copy())
            # combine children summary polynomials
            added_polynomial = self._special_polynomial_addition(
                p1=self.summary_polynomials[left_child],
                p2=self.summary_polynomials[right_child]
            )
            # store the summary polynomial of the current node
            self.summary_polynomials[node_id] = added_polynomial

    def _compute_shapley_values_superfast(
            self,
            x: np.ndarray,
            node_id: int,
            SP_down: np.ndarray[float] = None,
            SP_up: np.ndarray[float] = None,
            IP_down: np.ndarray[float] = None,
            QP_down: np.ndarray[float] = None,
            depth: int = 0
    ):
        # reset activations for new calculations
        if node_id == 0:
            self.activations = np.zeros(self.n_nodes, dtype=bool)

        if SP_down is None:
            SP_down = np.zeros((self.max_depth + 1, self.interpolation_size))
            SP_down[0, :] = 1
        if SP_up is None:
            SP_up = np.zeros((self.max_depth + 1, self.interpolation_size))
        if IP_down is None:
            IP_down = np.zeros((self.max_depth + 1,
                                int(binom(self.n_features, self.interaction_order)),
                                self.interpolation_size))
            IP_down[0, :] = 1
        if QP_down is None:
            QP_down = np.zeros((self.max_depth + 1,
                                int(binom(self.n_features, self.interaction_order)),
                                self.interpolation_size))
            QP_down[0, :] = 1

        # get node / edge information
        left_child, right_child = self.children_left[node_id], self.children_right[node_id]
        ancestor_id = self.ancestors[node_id]
        parent_id = self.parents[node_id]
        feature_id = self.features[parent_id]
        child_edge_feature = self.features[node_id]
        feature_threshold = self.thresholds[node_id]
        activations = self.activations
        is_leaf = left_child < 0
        current_height, left_height, right_height = self.edge_heights[node_id], \
            self.edge_heights[left_child], \
            self.edge_heights[right_child]
        if feature_id > -1:
            interaction_sets = self.subset_updates_pos[feature_id]

        # if node is not a leaf -> set activations for children nodes accordingly
        if not is_leaf:
            if x[child_edge_feature] <= feature_threshold:
                activations[left_child], activations[right_child] = True, False
            else:
                activations[left_child], activations[right_child] = False, True

        # if node is not the root node -> calculate the summary polynomials
        if node_id is not self.root_node_id:
            # set the activations of the current node in relation to the ancestor (for setting p_e to zero)
            if self.has_ancestors[node_id]:
                activations[node_id] &= activations[ancestor_id]
            # if node is active get the correct p_e value
            p_e_current = self.p_e_values[node_id] if activations[node_id] else 0.
            # update summary polynomial
            SP_down[depth] = SP_down[depth - 1] * (self.D + p_e_current)
            # update other polynomials
            QP_down[depth, :] = QP_down[depth - 1, :].copy()
            IP_down[depth, :] = IP_down[depth - 1, :].copy()
            QP_down[depth, interaction_sets] = QP_down[depth, interaction_sets] * (
                        self.D + p_e_current)
            IP_down[depth, interaction_sets] = IP_down[depth, interaction_sets] * (
                        -self.D + p_e_current)
            # remove previous polynomial factor if node has ancestors
            if self.has_ancestors[node_id]:
                p_e_ancestor = self.p_e_values[ancestor_id] if activations[ancestor_id] else 0.
                SP_down[depth] = SP_down[depth] / (self.D + p_e_ancestor)
                QP_down[depth, interaction_sets] = QP_down[depth, interaction_sets] / (
                            self.D + p_e_ancestor)
                IP_down[depth, interaction_sets] = IP_down[depth, interaction_sets] / (
                            -self.D + p_e_ancestor)
            else:
                p_e_ancestor = 1.  # no ancestor
        # if node is leaf -> add the empty prediction to the summary polynomial and store it
        if is_leaf:
            SP_up[depth] = SP_down[depth] * self.empty_predictions[node_id]

        else:  # not a leaf -> continue recursion
            self._compute_shapley_values_superfast(x, left_child, SP_down, SP_up, IP_down, QP_down,
                                                   depth + 1)
            SP_up[depth] = SP_up[depth + 1] * self.D_powers[current_height - left_height]
            self._compute_shapley_values_superfast(x, right_child, SP_down, SP_up, IP_down, QP_down,
                                                   depth + 1)
            SP_up[depth] += SP_up[depth + 1] * self.D_powers[current_height - right_height]

        if node_id is not self.root_node_id:
            interactions_seen = interaction_sets[
                self.interaction_height[node_id][interaction_sets] == self.interaction_order]
            if len(interactions_seen) > 0:
                # TODO ÄÄÄÄM :D
                self.shapley_interactions[interactions_seen] += np.dot(IP_down[depth, interactions_seen], self.Ns_id[self.interpolation_size, :]) * self._psi_superfast(SP_up[depth, :], self.D_powers[self.n_features-current_height], QP_down[depth, interactions_seen], self.Ns, self.n_features - self.interaction_order)
            # Ancestor handling
            ancestor_node_id = self.subset_ancestors[node_id][
                interaction_sets]  # ancestors of interactions
            if np.mean(ancestor_node_id) > -1:  # if there exist ancestors
                ancestor_node_id_exists = ancestor_node_id > -1
                interactions_with_ancestor = interaction_sets[
                    ancestor_node_id_exists]  # interactions with ancestors
                interactions_ancestors = ancestor_node_id[
                    ancestor_node_id_exists]  # ancestors of interactions with ancestor
                # check if all features have been observed for this interaction, otherwise the update is zero
                cond_interaction_seen = self.interaction_height[parent_id][
                                            interactions_with_ancestor] == self.interaction_order
                interactions_with_ancestor_to_update = interactions_with_ancestor[
                    cond_interaction_seen]
                if len(interactions_with_ancestor_to_update):
                    ancestor_heights = self.edge_heights[
                        interactions_ancestors[cond_interaction_seen]]
                    # TODO ÄÄÄÄM :D
                    self.shapley_interactions[interactions_with_ancestor_to_update] -= np.dot(IP_down[depth - 1, interactions_with_ancestor_to_update], self.Ns_id[self.interpolation_size, :]) * self._psi_superfast(SP_up[depth], self.D_powers[self.n_features - current_height], QP_down[depth - 1, interactions_with_ancestor_to_update], self.Ns, self.n_features - self.interaction_order)

    def _compute_shapley_values_interpolation(
            self,
            x: np.ndarray,
            node_id: int,
            original: bool = True,
            recursion_down: np.ndarray[float] = None,
            recursion_up: np.ndarray[float] = None,
            p_e_storage: np.ndarray[float] = None,
            depth: int = 0
    ):
        # reset activations for new calculations
        if node_id == 0:
            self.activations = np.zeros(self.n_nodes, dtype=bool)

        if p_e_storage is None:
            p_e_storage = np.ones(self.n_features, dtype=float)
        if recursion_down is None:
            recursion_down = np.zeros((self.max_depth + 1, self.max_depth))
            recursion_down[0, :] = 1
        if recursion_up is None:
            recursion_up = np.zeros((self.max_depth + 1, self.max_depth))

        # get node / edge information
        left_child, right_child = self.children_left[node_id], self.children_right[node_id]
        ancestor_id = self.ancestors[node_id]
        parent_id = self.parents[node_id]
        feature_id = self.features[parent_id]
        child_edge_feature = self.features[node_id]
        feature_threshold = self.thresholds[node_id]
        activations = self.activations
        is_leaf = left_child < 0
        current_height, left_height, right_height = self.edge_heights[node_id], self.edge_heights[
            left_child], self.edge_heights[right_child]

        # if node is not a leaf -> set activations for children nodes accordingly
        if not is_leaf:
            if x[child_edge_feature] <= feature_threshold:
                activations[left_child], activations[right_child] = True, False
            else:
                activations[left_child], activations[right_child] = False, True

        # if node is not the root node -> calculate the summary polynomials
        if node_id is not self.root_node_id:
            # set the activations of the current node in relation to the ancestor (for setting p_e to zero)
            if self.has_ancestors[node_id]:
                activations[node_id] &= activations[ancestor_id]
            # if node is active get the correct p_e value
            p_e_current = self.p_e_values[node_id] if activations[node_id] else 0.
            p_e_storage[feature_id] = p_e_current
            # update summary polynomial
            recursion_down[depth] = recursion_down[depth - 1] * (self.D + p_e_current)
            # remove previous polynomial factor if node has ancestors
            if self.has_ancestors[node_id]:
                p_e_ancestor = self.p_e_values[ancestor_id] if activations[ancestor_id] else 0.
                recursion_down[depth] = recursion_down[depth] / (self.D + p_e_ancestor)
            else:
                p_e_ancestor = 1.  # no ancestor
        # if node is leaf -> add the empty prediction to the summary polynomial and store it
        if is_leaf:
            recursion_up[depth] = recursion_down[depth] * self.empty_predictions[node_id]
            if not original:
                q_S, Q_S = {}, {}
                for S, pos in zip(self.shapley_interactions_lookup,
                                  self.shapley_interactions_lookup.values()):
                    # compute interaction factor and polynomial for aggregation below
                    q_S[S] = self._compute_p_e_interaction(S, p_e_storage)
                    if q_S[S] != 0:
                        Q_S[S] = self._compute_poly_interaction_fast(S, p_e_storage)
                        test = self._compute_poly_interaction(S, p_e_storage)
                        # update interactions for all interactions that contain feature_id
                        self.shapley_interactions[pos] += q_S[S] * self._psi_fast(
                            recursion_up[depth], self.D_powers[0], Q_S[S], self.Ns,
                            current_height - len(S))

        else:  # not a leaf -> continue recursion
            self._compute_shapley_values_interpolation(x, left_child, original, recursion_down, recursion_up,
                                                       p_e_storage.copy(), depth + 1)
            recursion_up[depth] = recursion_up[depth + 1] * self.D_powers[
                current_height - left_height]
            self._compute_shapley_values_interpolation(x, right_child, original, recursion_down,
                                                       recursion_up, p_e_storage.copy(), depth + 1)
            recursion_up[depth] += recursion_up[depth + 1] * self.D_powers[
                current_height - right_height]

        # upward computation of the shapley interactions
        # if node is not the root node -> update the shapley interactions
        if original:
            if node_id is not self.root_node_id:
                q_S, Q_S, Q_S_ancestor, q_S_ancestor = {}, {}, {}, {}
                for pos, S in zip(self.subset_updates_pos[feature_id],
                                  self.subset_updates[feature_id]):
                    # compute interaction factor and polynomial for aggregation below
                    q_S[S] = self._compute_p_e_interaction(S, p_e_storage)
                    if q_S[S] != 0:
                        Q_S[S] = self._compute_poly_interaction_fast(S, p_e_storage)
                        self.shapley_interactions[pos] += q_S[S] * self._psi_fast(
                            recursion_up[depth], self.D_powers[0], Q_S[S], self.Ns,
                            current_height - len(S))
                    # if the node has an ancestor, we need to update the interactions for the ancestor
                    ancestor_node_id = self.subset_ancestors[node_id][pos]
                    if ancestor_node_id > -1:
                        ancestor_height = self.edge_heights[ancestor_node_id]
                        p_e_storage_ancestor = p_e_storage.copy()
                        p_e_storage_ancestor[feature_id] = p_e_ancestor
                        q_S_ancestor[S] = self._compute_p_e_interaction(S, p_e_storage_ancestor)
                        if q_S_ancestor[S] != 0:
                            Q_S_ancestor[S] = self._compute_poly_interaction_fast(S,
                                                                                  p_e_storage_ancestor)
                            self.shapley_interactions[pos] -= q_S_ancestor[S] * self._psi_fast(
                                recursion_up[depth],
                                self.D_powers[ancestor_height - current_height], Q_S_ancestor[S],
                                self.Ns, ancestor_height - len(S))

    def _compute_original(
            self,
            x: np.ndarray,
            node_id: int,
            path_summary_poly: Polynomial,
            p_e_storage: np.ndarray[float] = None,
    ):
        # reset activations for new calculations
        if node_id == 0:
            self.activations = np.zeros(self.n_nodes, dtype=bool)

        if p_e_storage is None:
            p_e_storage = np.ones(self.n_features, dtype=float)

        # get node / edge information
        left_child, right_child = self.children_left[node_id], self.children_right[node_id]
        ancestor_id = self.ancestors[node_id]
        parent_id = self.parents[node_id]
        feature_id = self.features[parent_id]
        child_edge_feature = self.features[node_id]
        feature_threshold = self.thresholds[node_id]
        activations = self.activations
        is_leaf = left_child < 0
        p_e_ancestor = p_e_storage[feature_id].copy()

        # if node is not a leaf -> set activations for children nodes accordingly
        if not is_leaf:
            if x[child_edge_feature] <= feature_threshold:
                activations[left_child], activations[right_child] = True, False
            else:
                activations[left_child], activations[right_child] = False, True

        # if node is not the root node -> calculate the summary polynomials
        if node_id is not self.root_node_id:

            # set the activations of the current node in relation to the ancestor (for setting p_e to zero)
            if self.has_ancestors[node_id]:
                activations[node_id] &= activations[ancestor_id]

            # if node is active get the correct p_e value
            p_e_current = self.p_e_values[node_id] if activations[node_id] else 0.
            p_e_storage[feature_id] = p_e_current

            # update summary polynomial
            path_summary_poly = path_summary_poly * Polynomial([p_e_current, 1])

            # remove previous polynomial factor if node has ancestors
            if self.has_ancestors[node_id]:
                p_e_ancestor = self.p_e_values[ancestor_id] if activations[ancestor_id] else 0.
                path_summary_poly = Polynomial(
                    polydiv(path_summary_poly.coef, Polynomial([p_e_ancestor, 1]).coef)[0])

        # if node is leaf -> add the empty prediction to the summary polynomial and store it
        if is_leaf:
            self.summary_polynomials[node_id] = path_summary_poly * self.empty_predictions[node_id]
        else:  # not a leaf -> continue recursion
            self._compute_only_in_leaf(x, left_child, path_summary_poly, p_e_storage.copy())
            self._compute_only_in_leaf(x, right_child, path_summary_poly, p_e_storage.copy())
            # combine children summary polynomials
            added_polynomial = self._special_polynomial_addition(
                p1=self.summary_polynomials[left_child],
                p2=self.summary_polynomials[right_child]
            )
            # store the summary polynomial of the current node
            self.summary_polynomials[node_id] = added_polynomial

        if node_id is not self.root_node_id:
            q_S, Q_S, Q_S_ancestor, q_S_ancestor = {}, {}, {}, {}
            for pos, S in zip(self.subset_updates_pos[feature_id], self.subset_updates[feature_id]):
                # compute interaction factor and polynomial for aggregation below
                q_S[S] = self._compute_p_e_interaction(S, p_e_storage)
                if q_S[S] != 0:
                    Q_S[S] = self._compute_poly_interaction(S, p_e_storage)
                    # update interactions for all interactions that contain feature_id
                    quotient = Polynomial(
                        polydiv(self.summary_polynomials[node_id].coef, Q_S[S].coef)[0])
                    psi = self._psi(quotient)
                    self.shapley_interactions[pos] += q_S[S] * psi

                # if the node has an ancestor, we need to update the interactions for the ancestor
                ancestor_node_id = self.subset_ancestors[node_id][pos]
                if ancestor_node_id > -1:
                    p_e_storage_ancestor = p_e_storage.copy()
                    p_e_storage_ancestor[feature_id] = p_e_ancestor
                    q_S_ancestor[S] = self._compute_p_e_interaction(S, p_e_storage_ancestor)
                    if q_S_ancestor[S] != 0:
                        Q_S_ancestor[S] = self._compute_poly_interaction(S, p_e_storage_ancestor)
                        d_e = self.edge_heights[node_id]
                        d_e_ancestor = self.edge_heights[ancestor_node_id]
                        psi_factor = Polynomial([1, 1]) ** (d_e_ancestor - d_e)
                        psi_product = self.summary_polynomials[node_id] * psi_factor
                        quotient_ancestor = Polynomial(
                            polydiv(psi_product.coef, Q_S_ancestor[S].coef)[0])
                        psi_ancestor = self._psi(quotient_ancestor)
                        self.shapley_interactions[pos] -= q_S_ancestor[S] * psi_ancestor

    def extract_edge_information_from_tree(
            self,
            max_interaction: int = 1,
            min_interaction: int = 1
    ):
        """Extracts edge information recursively from the tree information.

        Parses the tree recursively to create an edge-based representation of the tree. It
        precalculates the p_e and p_e_ancestors of the interaction subsets up to order
        'max_interaction'.

        Args:
            max_interaction (int, optional): The maximum interaction order to be computed. An
                interaction order of 1 corresponds to the Shapley value. Any value higher than 1
                computes the Shapley interactions values up to that order. Defaults to 1 (i.e. SV).
            min_interaction (int, optional): The minimum interaction order to be computed. Defaults
                to 1.

        Returns:
            tuple: A tuple containing the following elements:
                - parents: np.ndarray[int] - The parents of each node.
                - ancestors: np.ndarray[int] - The ancestors of each node.
                - ancestor_nodes: dict[int, np.ndarray[int]] - A dictionary mapping the node id to
                    the ancestor nodes of the node for each feature.
                - p_e_values: np.ndarray[float] - The p_e values of each node.
                - p_e_storages: np.ndarray[float] - The p_e values of each node for each feature.
                - split_weights: np.ndarray[float] - The weights of each node.
                - empty_predictions: np.ndarray[float] - The empty predictions of each node.
                - edge_heights: np.ndarray[int] - The edge heights of each node.
                - max_depth: int - The maximum depth of the tree.
        """
        children_left = self.children_left
        children_right = self.children_right
        features = self.features
        n_features = self.n_features
        node_sample_weight = self.node_sample_weight
        values = self.values
        n_nodes = self.n_nodes

        # variables to be filled with recursive function
        parents = np.full(n_nodes, -1, dtype=int)
        ancestors: np.ndarray[int] = np.full(self.n_nodes, -1, dtype=int)

        ancestor_nodes: dict[int, np.ndarray[int]] = {}

        p_e_values: np.ndarray[float] = np.ones(n_nodes, dtype=float)
        p_e_storages: np.ndarray[float] = np.ones((n_nodes, n_features), dtype=float)
        split_weights: np.ndarray[float] = np.ones(n_nodes, dtype=float)
        empty_predictions: np.ndarray[float] = np.zeros(n_nodes, dtype=float)
        edge_heights: np.ndarray[int] = np.full_like(children_left, -1, dtype=int)
        max_depth: list[int] = [0]
        interaction_height = np.zeros((n_nodes, int(binom(self.n_features, max_interaction))),
                                      dtype=int)

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
                seen_features: np.ndarray[int] = np.full(n_features, -1, dtype=int)  # maps feature_id to ancestor node_id

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
            interaction_height[node_id] = interaction_height[parents[node_id]].copy()
            # correct if feature was seen before
            if seen_features[feature_id] > -1:  # feature has been seen before in the path
                ancestor_id = seen_features[feature_id]  # get ancestor node with same feature
                ancestors[node_id] = ancestor_id  # store ancestor node
                last_feature_node_in_path[ancestor_id] = False  # correct previous assumption
                p_e *= p_e_values[ancestor_id]  # add ancestor weight to p_e
            else:
                interaction_height[node_id][self.subset_updates_pos[feature_id]] += 1

            # store the p_e value of the current node
            p_e_values[node_id] = p_e
            p_e_storages[node_id] = p_e_storages[parents[node_id]].copy()
            p_e_storages[node_id][feature_id] = p_e

            # update seen features with current node
            seen_features[feature_id] = node_id

            # TODO precompute what we can for the interactions

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
        return parents, ancestors, ancestor_nodes, p_e_values, p_e_storages, split_weights, empty_predictions, edge_heights, \
        max_depth[0], last_feature_node_in_path, interaction_height

    @staticmethod
    def _all_interactions(
            feature_id: int,
            n_features: int,
            size: int = 1
    ) -> tuple:
        """Yields all combinations of size 'size' that contain the feature with id 'feature_id' as a
        genartor."""
        # TODO might break with size == 1
        remaining_features = set(range(n_features)) - {feature_id}
        for S in powerset(remaining_features, max_size=size - 1):
            yield tuple(sorted(S + (feature_id,)))
        return

    def explain_brute_force(
            self,
            x: np.ndarray,
            max_order: int = 1
    ):
        """Computes the Shapley values and interactions using brute force method (enumeration)."""
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
                #weight_T = 1 / (
                #            binom(self.n_features - order, len(T)) * (self.n_features - order + 1))
                weight_T = self._get_subset_weight(len(T),len(S))
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
                    subset_val_left = self._naive_shapley_recursion(x, S,
                                                                    self.children_left[node_id],
                                                                    weight)
                else:
                    subset_val_left = 0
                    subset_val_right = self._naive_shapley_recursion(x, S,
                                                                     self.children_right[node_id],
                                                                     weight)
            else:
                subset_val_left = self._naive_shapley_recursion(
                    x, S, self.children_left[node_id],
                    weight * self.weights[self.children_left[node_id]])
                subset_val_right = self._naive_shapley_recursion(
                    x, S, self.children_right[node_id],
                    weight * self.weights[self.children_right[node_id]])
        return subset_val_left + subset_val_right

    @staticmethod
    def _generate_interactions_lookup(n_features, max_order):
        counter_interaction = 0  # stores position of interactions
        shapley_interactions_lookup: dict = {}
        for S in powerset(range(n_features), max_order):
            shapley_interactions_lookup[S] = counter_interaction
            counter_interaction += 1
        return shapley_interactions_lookup

    def _precompute_subsets_with_feature(self):
        subset_updates_pos: dict = {}  # stores position of interactions that include feature i
        subset_updates: dict = {}  # stores interactions that include feature i
        # TODO: precompute separately, optimize runtime, compute within tree recursion
        for i in range(self.n_features):
            subsets = []
            positions = np.zeros(int(binom(self.n_features - 1, self.max_order - 1)), dtype=int)
            pos_counter = 0
            for S in powerset(range(self.n_features), min_size=self.max_order,
                              max_size=self.max_order):
                if i in S:
                    positions[pos_counter] = self.shapley_interactions_lookup[S]
                    subsets.append(S)
                    pos_counter += 1
            subset_updates_pos[i] = positions
            subset_updates[i] = subsets
        return subset_updates, subset_updates_pos

    def _precalculate_interaction_ancestors(self):
        """Calculates the position of the ancestors of the interactions for the tree for a given
        order of interactions."""

        # stores position of interactions
        counter_interaction = 0

        for node_id in self.nodes[1:]:  # for all nodes except the root node
            self.subset_ancestors[node_id] = np.full(int(binom(self.n_features, self.max_order)),
                                                     -1, dtype=int)
        for S in powerset(range(self.n_features), self.max_order):
            # self.shapley_interactions_lookup[S] = counter_interaction
            for node_id in self.nodes[1:]:  # for all nodes except the root node
                subset_ancestor = -1
                for i in S:
                    subset_ancestor = max(subset_ancestor, self.ancestor_nodes[node_id][i])
                self.subset_ancestors[node_id][counter_interaction] = subset_ancestor
            counter_interaction += 1

    @staticmethod
    def _compute_poly_interaction(S: tuple, p_e_values: np.ndarray[float]) -> Polynomial:
        """Computes Q_S (interaction polynomial) given p_e values"""
        poly_interaction = Polynomial([1.])
        for i in S:
            poly_interaction = poly_interaction * Polynomial([p_e_values[i], 1])
        return poly_interaction

    def _compute_poly_interaction_fast(self, S: tuple, p_e_values: np.ndarray[float]) -> np.ndarray:
        """Computes Q_S (interaction polynomial) given p_e values"""
        rslt = np.ones((self.max_depth))
        for i in S:
            rslt *= (self.D + p_e_values[i])
        return rslt

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

    @staticmethod
    def _compute_p_e_interaction_fast(S: tuple, p_e_values: np.ndarray[float]) -> float:
        """Computes q_S (interaction factor) given p_e values"""
        poly = Polynomial([1.])
        for j in S:
            poly *= Polynomial([p_e_values[j], -1])
        return (-1) ** (len(S) + 1) * np.sum(poly.coef)

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

    def _psi_fast(self, E, D_power, quotient_poly, Ns, degree):
        n = Ns[degree + 1, :degree + 1]
        return ((E * D_power / quotient_poly)[:degree + 1]).dot(n) / (degree + 1)

    def _psi_superfast(self, E, D_power, quotient_poly, Ns, degree):
        d = degree + 1
        n = Ns[d, :d]
        return ((E * D_power / quotient_poly)[:, :d]).dot(n) / d

    def _psi_superfast_ancestor(self, E, D_power, quotient_poly, Ns, degree):
        # Variant of _psi_superfast that can deal with multiple inputs in degree
        # TODO: check if improvement can be done for matrix multiplication instead of diag
        return np.diag((E * D_power / quotient_poly).dot(Ns[degree + 1].T)) / (degree + 1)

    def get_N(self, D):
        depth = D.shape[0]
        Ns = np.zeros((depth + 1, depth))
        for i in range(1, depth + 1):
            Ns[i, :i] = np.linalg.inv(np.vander(D[:i]).T).dot(i*np.array([self._get_subset_weight(j,self.interaction_order) for j in range(i)]))
        return Ns

    def get_N_id(self, D):
        depth = D.shape[0]
        Ns_id = np.zeros((depth + 1, depth))
        for i in range(1, depth + 1):
            Ns_id[i, :i] = np.linalg.inv(np.vander(D[:i]).T).dot(np.ones((i)))
        return Ns_id

    def get_norm_weight(self, M):
        return np.array([binom(M, i) for i in range(M + 1)])


    def _get_subset_weight(self,t,order):
        if self.interaction_type == "SII":
            return 1/((self.n_features-order+1)*binom(self.n_features-order,t))
        if self.interaction_type == "STI":
            return self.interaction_order/(self.n_features*binom(self.n_features-1,t))
        if self.interaction_type == "FSI":
            return np.math.factorial(2*self.interaction_order-1)/np.math.factorial(self.interaction_order-1)**2*np.math.factorial(self.interaction_order+t-1)*np.math.factorial(self.n_features-t-1)/np.math.factorial(self.n_features+self.interaction_order-1)

    def cache(self, D):
        return np.vander(D + 1).T[::-1]

    def _get_interventional_sample_weights(self, background_dataset: np.ndarray):
        """Computes the interventional sample weights for the background dataset.

        The interventional sample weights are computed by passing the whole background dataset
            through each decision node in the tree independently of the rest of the tree and
            multiplying the weights of the edges that are traversed.

        Args:
            background_dataset (np.ndarray): The background dataset.

        Returns:
            np.ndarray: The interventional sample weights for the background dataset.
        """
        weights = np.zeros(self.n_nodes, dtype=float)
        n_samples = len(background_dataset)
        weights[0] = 1.
        for node_id in self.nodes:
            if self.leaf_mask[node_id]:
                continue
            feature_id = self.features[node_id]
            threshold = self.thresholds[node_id]
            left_child, right_child = self.children_left[node_id], self.children_right[node_id]
            left_mask = background_dataset[:, feature_id] <= threshold
            left_proportion = np.sum(left_mask) / n_samples
            if left_proportion <= 0:
                left_proportion = 1e-10
            if left_proportion >= 1:
                left_proportion = 1 - 1e-10
            right_proportion = 1 - left_proportion
            weights[left_child] = weights[node_id] * left_proportion
            weights[right_child] = weights[node_id] * right_proportion
        return weights
