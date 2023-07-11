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
            # depth = self.summary_polynomials[node_id].degree()

            degree_of_ancestor = self.summary_polynomials_degree[self.ancestor_nodes[node_id]]
            degree_of_node = self.summary_polynomials_degree[node_id]

            psi_numerator = Polynomial([1, 1]) ** (degree_of_ancestor - degree_of_node)
            psi_numerator = self.summary_polynomials[node_id] * psi_numerator
            psi_denominator = Polynomial([p_e_ancestor, 1])
            quotient_ancestor = Polynomial(
                polydiv(psi_numerator.coef, psi_denominator.coef)[0])
            psi_ancestor = self._psi(quotient_ancestor)
            self.shapley_values[feature_id] -= (p_e_ancestor - 1) * psi_ancestor
