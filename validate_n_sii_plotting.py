"""This module contains a unit test for the plotting of the n-SII values."""
import numpy as np
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor

from tree_shap_iq.conversion import convert_tree_estimator
from plotting.plot_n_sii import plot_n_sii, transform_interactions_in_n_shapley
from tree_shap_iq import TreeShapIQ

if __name__ == "__main__":

    MAX_INTERACTION_ORDER = 8
    DO_OBSERVATIONAL = True

    # fix random seed for reproducibility
    random_seed = 10
    np.random.seed(random_seed)

    # create dummy regression dataset and fit tree model
    X, y = make_regression(1000, n_features=10)
    x_explain = X[:1][0]
    n_features = X.shape[-1]

    # fit a tree model
    model = RandomForestRegressor(max_depth=10, random_state=random_seed, n_estimators=10)
    model.fit(X, y)
    print("Output f(x):", model.predict(x_explain.reshape(1, -1))[0])

    # convert the tree
    list_of_trees = convert_tree_estimator(model)

    # explain the tree with TreeSHAP-IQ
    sii_values_dict: dict[int, np.ndarray] = {order: 0. for order in range(1, MAX_INTERACTION_ORDER + 1)}  # will be filled with the SII values for each interaction order
    empty_prediction = 0.
    for tree_model in list_of_trees:
        sii_estimator = TreeShapIQ(
            tree_model=tree_model,
            n_features=n_features,
            max_interaction_order=MAX_INTERACTION_ORDER,
            observational=DO_OBSERVATIONAL,
            background_dataset=X
        )
        sii_scores: dict[int, np.ndarray] = sii_estimator.explain(
            x=x_explain,
            order=MAX_INTERACTION_ORDER
        )
        sii_values_dict = {
            order: sii_values_dict[order] + sii_scores[order]
            for order in range(1, MAX_INTERACTION_ORDER + 1)
        }
        empty_prediction += sii_estimator.empty_prediction

    # transform the SII values in n-Shapley values
    n_shapley_values = transform_interactions_in_n_shapley(
        interaction_values=sii_values_dict,
        n=MAX_INTERACTION_ORDER,
        n_features=n_features,
        reduce_one_dimension=True
    )
    n_shapley_values_pos, n_shapley_values_neg = n_shapley_values

    sum_value: float = empty_prediction
    for order in range(1, MAX_INTERACTION_ORDER + 1):
        sum_value += sum(list(n_shapley_values_pos[order].values()))
        sum_value += sum(list(n_shapley_values_neg[order].values()))
    print("Sum of n-Shapley values", sum_value)

    # plot the n-Shapley values
    fig, axis = plot_n_sii(
        n_shapley_values_pos=n_shapley_values_pos,
        n_shapley_values_neg=n_shapley_values_neg,
        feature_names=np.arange(n_features),
        n_sii_order=MAX_INTERACTION_ORDER
    )
    fig.show()
