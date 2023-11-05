"""This module contains the TreeSHAP-IQ class."""
from typing import Union

import numpy as np
from scipy.special import binom

from tree_shap_iq.conversion import TreeModel, EdgeTree, extract_edge_information_from_tree

try:
    from .utils import _get_parent_array, powerset
except ImportError:
    from utils import _get_parent_array, powerset


class TreeShapIQ:

    def __init__(
            self,
            tree_model: Union[dict, TreeModel],
            max_interaction_order: int = 1,
            observational: bool = True,
            background_dataset: np.ndarray = None,
            n_features: int = None,
            interaction_type: str = "SII"
    ):
        """The TreeSHAPIQExplainer class. This class is a reimplementation of the original Linear
            TreeSHAP algorithm for the case of interaction order 1 (i.e. Shapley value). If the
            interaction order is higher than 1, the algorithm compute Shapley interaction values up
            to that order. The algorithm can be used for both the ´observational´ and
            ´interventional´ Shapley approach. The interventional Shapley approach requires a
            background dataset to be provided.

        Args:
            tree_model (Union[dict, TreeModel]): The tree model to be explained. If the tree model
                is a dictionary it must include the following keys:
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
        """
        self.interaction_type = interaction_type
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
        self.root_node_id: int = 0
        self.n_features: int = n_features
        if n_features is None:
            self.n_features: int = max(self.features) + 1

        # get the leaf predictions and the observational or interventional sample weights
        self.values = tree_model["values"]
        self.max_interaction_order = max_interaction_order
        if observational:
            self.node_sample_weight = tree_model["node_sample_weight"]
        else:
            self.node_sample_weight = self._get_interventional_sample_weights(background_dataset)

        # precompute subsets that include each feature and their positions
        self.max_order: int = max_interaction_order

        self.shapley_interactions_lookup_store: dict[int, dict[tuple, int]] = {}
        self.subset_updates_pos_store: dict = {}
        for order in range(1, self.max_order + 1):
            shapley_interactions_lookup: dict = self._generate_interactions_lookup(self.n_features, order)
            self.shapley_interactions_lookup_store[order] = shapley_interactions_lookup
            _, subset_updates_pos = self._precompute_subsets_with_feature(interaction_order=order, n_features=self.n_features, shapley_interactions_lookup=shapley_interactions_lookup)
            self.subset_updates_pos_store[order] = subset_updates_pos

        edge_tree: EdgeTree = extract_edge_information_from_tree(
            children_left=self.children_left,
            children_right=self.children_right,
            features=self.features,
            node_sample_weight=self.node_sample_weight,
            values=self.values,
            max_interaction=max_interaction_order,
            n_features=self.n_features,
            n_nodes=self.n_nodes,
            subset_updates_pos_store=self.subset_updates_pos_store
        )
        self.parents = edge_tree.parents
        self.ancestors = edge_tree.ancestors
        self.ancestor_nodes = edge_tree.ancestor_nodes
        self.p_e_values = edge_tree.p_e_values
        self.p_e_storages = edge_tree.p_e_storages
        self.split_weights = edge_tree.split_weights
        self.empty_predictions = edge_tree.empty_predictions
        self.edge_heights = edge_tree.edge_heights
        self.max_depth = edge_tree.max_depth
        self.last_feature_node_in_path = edge_tree.last_feature_node_in_path
        self.interaction_height_store = edge_tree.interaction_height_store

        self.has_ancestors = self.ancestors > -1

        # get empty prediction of model
        self.empty_prediction: float = float(np.sum(self.empty_predictions[self.leaf_mask]))
        if tree_model["empty_prediction"] is not None:
            self.empty_prediction = tree_model["empty_prediction"]

        # stores the interaction scores up to a given order
        self.subset_ancestors_store: dict = {}
        self.D_store: dict = {}
        self.D_powers_store: dict = {}
        self.Ns_id_store: dict = {}
        self.Ns_store: dict = {}
        if self.interaction_type == "SII":
            #SP is of order at most d_max
            self.n_interpolation_size = min(self.max_depth,self.n_features)
        else:
            #SP is always of order n_features
            self.n_interpolation_size = self.n_features

        for order in range(1, self.max_order + 1):
            subset_ancestors: dict[int, np.ndarray] = self._precalculate_interaction_ancestors(
                interaction_order=order, n_features=self.n_features)
            self.subset_ancestors_store[order] = subset_ancestors
            #self.D_store[order] = np.polynomial.chebyshev.chebpts2(self.max_depth)
            self.D_store[order] = np.polynomial.chebyshev.chebpts2(self.n_interpolation_size)
            self.D_powers_store[order] = self.cache(self.D_store[order])
            if self.interaction_type == "SII":
                self.Ns_store[order] = self.get_N(self.D_store[order])
            else:
                self.Ns_store[order] = self.get_N_cii(self.D_store[order],order)
            self.Ns_id_store[order] = self.get_N_id(self.D_store[order])

        # new for improved calculations
        self.activations: np.ndarray[bool] = np.zeros(self.n_nodes, dtype=bool)

    def _prepare_variables_for_order(self, interaction_order: int):
        """Retrieves the precomputed variables for a given interaction order. This function is
            called before the recursive explanation function is called.

        Args:
            interaction_order (int): The interaction order for which the storage variables should be
                loaded.
        """
        self.subset_updates_pos = self.subset_updates_pos_store[interaction_order]
        self.subset_ancestors = self.subset_ancestors_store[interaction_order]
        self.D = self.D_store[interaction_order]
        self.D_powers = self.D_powers_store[interaction_order]
        self.interaction_height = self.interaction_height_store[interaction_order]
        self.Ns_id = self.Ns_id_store[interaction_order]
        self.Ns = self.Ns_store[interaction_order]
        self.subset_updates_pos = self.subset_updates_pos_store[interaction_order]

    def explain(
            self,
            x: np.ndarray,
            order: int = None,
            min_order: int = 1
    ) -> dict[int, np.ndarray[float]]:
        """Computes the Shapley Interaction values for a given instance x and interaction order.
            This function is the main explanation function of this class.

        Args:
            x (np.ndarray): Instance to be explained.
            order (int, optional): Order of the interactions. Defaults to max_order from init.
            min_order (int, optional): Minimum order of the interactions. Defaults to 1.

        Returns:
            np.ndarray[float]: Shapley Interaction values. The shape of the array is (n_features,
                order).
        """
        if order is None:
            order = self.max_order
        assert order <= self.max_order, f"Order {order} is larger than the maximum interaction " \
                                        f"order {self.max_order}."
        interactions = {}
        for order in range(min_order, order + 1):
            self.shapley_interactions = np.zeros(int(binom(self.n_features, order)), dtype=float)
            self._prepare_variables_for_order(interaction_order=order)
            # call the recursive function to compute the shapley values
            if self.interaction_type == "SII":
                self._compute_interactions(x, 0, interaction_order=order)
            else:
                self._compute_interactions_cii(x, 0, interaction_order=order)

            interactions[order] = self.shapley_interactions.copy()
        return interactions

    def _compute_interactions(
            self,
            x: np.ndarray,
            node_id: int,
            SP_down: np.ndarray[float] = None,
            SP_up: np.ndarray[float] = None,
            IP_down: np.ndarray[float] = None,
            QP_down: np.ndarray[float] = None,
            depth: int = 0,
            interaction_order: int = 1
    ):
        # reset activations for new calculations
        if node_id == 0:
            self.activations = np.zeros(self.n_nodes, dtype=bool)

        if SP_down is None:
            SP_down = np.zeros((self.max_depth + 1, self.n_interpolation_size))
            SP_down[0, :] = 1
        if SP_up is None:
            SP_up = np.zeros((self.max_depth + 1, self.n_interpolation_size))
        if IP_down is None:
            IP_down = np.zeros((self.max_depth + 1,
                                int(binom(self.n_features, interaction_order)),
                                self.n_interpolation_size))
            IP_down[0, :] = 1
        if QP_down is None:
            QP_down = np.zeros((self.max_depth + 1,
                                int(binom(self.n_features, interaction_order)),
                                self.n_interpolation_size))
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
            self._compute_interactions(x, left_child, SP_down, SP_up, IP_down, QP_down,
                                       depth + 1, interaction_order)
            SP_up[depth] = SP_up[depth + 1] * self.D_powers[current_height - left_height]
            self._compute_interactions(x, right_child, SP_down, SP_up, IP_down, QP_down,
                                       depth + 1, interaction_order)
            SP_up[depth] += SP_up[depth + 1] * self.D_powers[current_height - right_height]

        if node_id is not self.root_node_id:
            interactions_seen = interaction_sets[
                self.interaction_height[node_id][interaction_sets] == interaction_order]
            if len(interactions_seen) > 0:
                self.shapley_interactions[interactions_seen] += np.dot(IP_down[depth, interactions_seen], self.Ns_id[self.n_interpolation_size, :self.n_interpolation_size]) * self._psi_superfast(SP_up[depth, :], self.D_powers[0], QP_down[depth, interactions_seen], self.Ns, current_height - interaction_order)
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
                                            interactions_with_ancestor] == interaction_order
                interactions_with_ancestor_to_update = interactions_with_ancestor[
                    cond_interaction_seen]
                if len(interactions_with_ancestor_to_update):
                    ancestor_heights = self.edge_heights[
                        interactions_ancestors[cond_interaction_seen]]
                    self.shapley_interactions[interactions_with_ancestor_to_update] -= np.dot(IP_down[depth - 1, interactions_with_ancestor_to_update], self.Ns_id[self.n_interpolation_size, :self.n_interpolation_size]) * self._psi_superfast_ancestor(SP_up[depth], self.D_powers[ancestor_heights - current_height], QP_down[depth - 1, interactions_with_ancestor_to_update], self.Ns, ancestor_heights - interaction_order)


    def _compute_interactions_cii(
            self,
            x: np.ndarray,
            node_id: int,
            SP_down: np.ndarray[float] = None,
            SP_up: np.ndarray[float] = None,
            IP_down: np.ndarray[float] = None,
            QP_down: np.ndarray[float] = None,
            depth: int = 0,
            interaction_order: int = 1
    ):
        # reset activations for new calculations
        if node_id == 0:
            self.activations = np.zeros(self.n_nodes, dtype=bool)

        if SP_down is None:
            SP_down = np.zeros((self.max_depth + 1, self.n_interpolation_size))
            SP_down[0, :] = 1
        if SP_up is None:
            SP_up = np.zeros((self.max_depth + 1, self.n_interpolation_size))
        if IP_down is None:
            IP_down = np.zeros((self.max_depth + 1,
                                int(binom(self.n_features, interaction_order)),
                                self.n_interpolation_size))
            IP_down[0, :] = 1
        if QP_down is None:
            QP_down = np.zeros((self.max_depth + 1,
                                int(binom(self.n_features, interaction_order)),
                                self.n_interpolation_size))
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
            self._compute_interactions_cii(x, left_child, SP_down, SP_up, IP_down, QP_down,
                                       depth + 1, interaction_order)
            SP_up[depth] = SP_up[depth + 1] * self.D_powers[current_height - left_height]
            self._compute_interactions_cii(x, right_child, SP_down, SP_up, IP_down, QP_down,
                                       depth + 1, interaction_order)
            SP_up[depth] += SP_up[depth + 1] * self.D_powers[current_height - right_height]

        if node_id is not self.root_node_id:
            interactions_seen = interaction_sets[
                self.interaction_height[node_id][interaction_sets] == interaction_order]
            if len(interactions_seen) > 0:
                self.shapley_interactions[interactions_seen] += np.dot(IP_down[depth, interactions_seen],self.Ns_id[self.n_interpolation_size,:self.n_interpolation_size]) * self._psi_superfast(SP_up[depth, :], self.D_powers[self.n_features-current_height], QP_down[depth, interactions_seen], self.Ns,self.n_features - interaction_order)
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
                                            interactions_with_ancestor] == interaction_order
                interactions_with_ancestor_to_update = interactions_with_ancestor[
                    cond_interaction_seen]
                if len(interactions_with_ancestor_to_update):
                    ancestor_heights = self.edge_heights[
                        interactions_ancestors[cond_interaction_seen]]
                    self.shapley_interactions[interactions_with_ancestor_to_update] -= np.dot(IP_down[depth - 1, interactions_with_ancestor_to_update], self.Ns_id[self.n_interpolation_size, :self.n_interpolation_size]) * self._psi_superfast(SP_up[depth], self.D_powers[self.n_features - current_height], QP_down[depth - 1, interactions_with_ancestor_to_update], self.Ns, self.n_features - interaction_order)


    @staticmethod
    def _generate_interactions_lookup(n_features, max_order):
        counter_interaction = 0  # stores position of interactions
        shapley_interactions_lookup: dict = {}
        for S in powerset(range(n_features), max_order):
            shapley_interactions_lookup[S] = counter_interaction
            counter_interaction += 1
        return shapley_interactions_lookup

    @staticmethod
    def _precompute_subsets_with_feature(
            n_features: int,
            interaction_order: int,
            shapley_interactions_lookup: dict[tuple, int]
    ):
        subset_updates_pos: dict = {}  # stores position of interactions that include feature i
        subset_updates: dict = {}  # stores interactions that include feature i
        for i in range(n_features):
            subsets = []
            positions = np.zeros(int(binom(n_features - 1, interaction_order - 1)), dtype=int)
            pos_counter = 0
            for S in powerset(range(n_features), min_size=interaction_order, max_size=interaction_order):
                if i in S:
                    positions[pos_counter] = shapley_interactions_lookup[S]
                    subsets.append(S)
                    pos_counter += 1
            subset_updates_pos[i] = positions
            subset_updates[i] = subsets
        return subset_updates, subset_updates_pos

    def _precalculate_interaction_ancestors(self, interaction_order, n_features):
        """Calculates the position of the ancestors of the interactions for the tree for a given
        order of interactions."""

        # stores position of interactions
        counter_interaction = 0
        subset_ancestors: dict[int, np.ndarray[int]] = {}

        for node_id in self.nodes[1:]:  # for all nodes except the root node
            subset_ancestors[node_id] = np.full(int(binom(n_features, interaction_order)), -1, dtype=int)
        for S in powerset(range(n_features), interaction_order):
            # self.shapley_interactions_lookup[S] = counter_interaction
            for node_id in self.nodes[1:]:  # for all nodes except the root node
                subset_ancestor = -1
                for i in S:
                    subset_ancestor = max(subset_ancestor, self.ancestor_nodes[node_id][i])
                subset_ancestors[node_id][counter_interaction] = subset_ancestor
            counter_interaction += 1
        return subset_ancestors

    @staticmethod
    def _psi_superfast(E, D_power, quotient_poly, Ns, degree):
        d = degree + 1
        n = Ns[d, :d]
        return ((E * D_power / quotient_poly)[:, :d]).dot(n) / d

    @staticmethod
    def _psi_superfast_ancestor(E, D_power, quotient_poly, Ns, degree):
        d = degree + 1
        # Variant of _psi_superfast that can deal with multiple inputs in degree
        # TODO: check if improvement can be done for matrix multiplication instead of diag
        return np.diag((E * D_power / quotient_poly).dot(Ns[d].T)) / (d)

    def get_N(self,D):
        depth = D.shape[0]
        Ns = np.zeros((depth + 1, depth))
        for i in range(1, depth + 1):
            Ns[i, :i] = np.linalg.inv(np.vander(D[:i]).T).dot(1. / self.get_norm_weight(i - 1))
        return Ns

    def get_N_cii(self, D, order):
        depth = D.shape[0]
        Ns = np.zeros((depth + 1, depth))
        for i in range(1, depth + 1):
            Ns[i, :i] = np.linalg.inv(np.vander(D[:i]).T).dot(i*np.array([self._get_subset_weight(j,order) for j in range(i)]))
        return Ns

    def _get_subset_weight(self,t,order):
        if self.interaction_type == "SII":
            return 1/(self.n_features*binom(self.n_features-order,t))
        if self.interaction_type == "STI":
            return self.max_order/(self.n_features*binom(self.n_features-1,t))
        if self.interaction_type == "FSI":
            return np.math.factorial(2*self.max_order-1)/np.math.factorial(self.max_order-1)**2*np.math.factorial(self.max_order+t-1)*np.math.factorial(self.n_features-t-1)/np.math.factorial(self.n_features+self.max_order-1)

    @staticmethod
    def get_N_id(D):
        depth = D.shape[0]
        Ns_id = np.zeros((depth + 1, depth))
        for i in range(1, depth + 1):
            Ns_id[i, :i] = np.linalg.inv(np.vander(D[:i]).T).dot(np.ones((i)))
        return Ns_id

    @staticmethod
    def get_norm_weight(M):
        return np.array([binom(M, i) for i in range(M + 1)])

    @staticmethod
    def cache(D):
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

    def explain_brute_force(
            self,
            x: np.ndarray,
            max_order: int = 1
    ):
        """Computes the Shapley values and interactions using brute force method (enumeration)."""

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
        for S in powerset(features, min_size=order):
            temp_values = 0
            for T in powerset(set(features) - set(S)):
                weight_T = self._get_subset_weight(len(T), len(S))
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
                    weight * self.split_weights[self.children_left[node_id]])
                subset_val_right = self._naive_shapley_recursion(
                    x, S, self.children_right[node_id],
                    weight * self.split_weights[self.children_right[node_id]])
        return subset_val_left + subset_val_right
