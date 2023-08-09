"""This module contains the TreeSHAP-IQ class."""
from typing import Union
from dataclasses import dataclass

import numpy as np
from scipy.special import binom

from linear_interaction.conversion import TreeModel

try:
    from .utils import _get_parent_array, powerset
except ImportError:
    from utils import _get_parent_array, powerset


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


class TreeShapIQ:

    def __init__(
            self,
            tree_model: Union[dict, TreeModel],
            max_interaction_order: int = 1,
            observational: bool = True,
            background_dataset: np.ndarray = None,
            n_features: int = None
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

        edge_tree: EdgeTree = self.extract_edge_information_from_tree(max_interaction=max_interaction_order)
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
        for order in range(1, self.max_order + 1):
            subset_ancestors: dict[int, np.ndarray] = self._precalculate_interaction_ancestors(
                interaction_order=order, n_features=self.n_features)
            self.subset_ancestors_store[order] = subset_ancestors
            self.D_store[order] = np.polynomial.chebyshev.chebpts2(self.max_depth)
            self.D_powers_store[order] = self.cache(self.D_store[order])
            self.Ns_store[order] = self.get_N(self.D_store[order])
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
            order: int = 1,
            min_order: int = 1
    ) -> dict[int, np.ndarray[float]]:
        """Computes the Shapley Interaction values for a given instance x and interaction order.
            This function is the main explanation function of this class.

        Args:
            x (np.ndarray): Instance to be explained.
            order (int, optional): Order of the interactions. Defaults to 1.
            min_order (int, optional): Minimum order of the interactions. Defaults to 1.

        Returns:
            np.ndarray[float]: Shapley Interaction values. The shape of the array is (n_features,
                order).
        """
        assert order <= self.max_order, f"Order {order} is larger than the maximum interaction " \
                                        f"order {self.max_order}."
        interactions = {}
        for order in range(min_order, order + 1):
            self.shapley_interactions = np.zeros(int(binom(self.n_features, order)), dtype=float)
            self._prepare_variables_for_order(interaction_order=order)
            # call the recursive function to compute the shapley values
            self._compute_interactions(x, 0, interaction_order=order)
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
            SP_down = np.zeros((self.max_depth + 1, self.max_depth))
            SP_down[0, :] = 1
        if SP_up is None:
            SP_up = np.zeros((self.max_depth + 1, self.max_depth))
        if IP_down is None:
            IP_down = np.zeros((self.max_depth + 1,
                                int(binom(self.n_features, interaction_order)),
                                self.max_depth))
            IP_down[0, :] = 1
        if QP_down is None:
            QP_down = np.zeros((self.max_depth + 1,
                                int(binom(self.n_features, interaction_order)),
                                self.max_depth))
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
                # TODO ÄÄÄÄM :D
                self.shapley_interactions[interactions_seen] += np.dot(IP_down[depth, interactions_seen], self.Ns_id[self.max_depth, :self.max_depth]) * self._psi_superfast(SP_up[depth, :], self.D_powers[0], QP_down[depth, interactions_seen], self.Ns, current_height - interaction_order)
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
                    # TODO ÄÄÄÄM :D
                    self.shapley_interactions[interactions_with_ancestor_to_update] -= np.dot(IP_down[depth - 1, interactions_with_ancestor_to_update], self.Ns_id[self.max_depth, :self.max_depth]) * self._psi_superfast_ancestor(SP_up[depth], self.D_powers[ancestor_heights - current_height], QP_down[depth - 1, interactions_with_ancestor_to_update], self.Ns, ancestor_heights - interaction_order)

    def extract_edge_information_from_tree(
            self,
            max_interaction: int = 1
    ):
        """Extracts edge information recursively from the tree information.

        Parses the tree recursively to create an edge-based representation of the tree. It
        precalculates the p_e and p_e_ancestors of the interaction subsets up to order
        'max_interaction'.

        Args:
            max_interaction (int, optional): The maximum interaction order to be computed. An
                interaction order of 1 corresponds to the Shapley value. Any value higher than 1
                computes the Shapley interactions values up to that order. Defaults to 1 (i.e. SV).

        Returns:
            EdgeTree: A dataclass containing the edge information of the tree.
        """
        children_left = self.children_left
        children_right = self.children_right
        features = self.features
        n_features = self.n_features
        node_sample_weight = self.node_sample_weight
        values = self.values
        n_nodes = self.n_nodes
        subset_updates_pos_store = self.subset_updates_pos_store

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
        interaction_height_store = {i: np.zeros((n_nodes, int(binom(n_features, i))), dtype=int) for i in range(1, max_interaction + 1)}
        #interaction_height = np.zeros((n_nodes, int(binom(n_features, max_interaction))), dtype=int)

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
            for order in range(1, max_interaction + 1):
                interaction_height_store[order][node_id] = interaction_height_store[order][parents[node_id]].copy()
            # correct if feature was seen before
            if seen_features[feature_id] > -1:  # feature has been seen before in the path
                ancestor_id = seen_features[feature_id]  # get ancestor node with same feature
                ancestors[node_id] = ancestor_id  # store ancestor node
                last_feature_node_in_path[ancestor_id] = False  # correct previous assumption
                p_e *= p_e_values[ancestor_id]  # add ancestor weight to p_e
            else:
                for order in range(1, max_interaction + 1):
                    interaction_height_store[order][node_id][subset_updates_pos_store[order][feature_id]] += 1

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
        # Variant of _psi_superfast that can deal with multiple inputs in degree
        # TODO: check if improvement can be done for matrix multiplication instead of diag
        return np.diag((E * D_power / quotient_poly).dot(Ns[degree + 1].T)) / (degree + 1)

    def get_N(self, D):
        depth = D.shape[0]
        Ns = np.zeros((depth + 1, depth))
        for i in range(1, depth + 1):
            Ns[i, :i] = np.linalg.inv(np.vander(D[:i]).T).dot(1. / self.get_norm_weight(i - 1))
        return Ns

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
