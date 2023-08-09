"""This module is used to run the experiment on the california-housing dataset."""
import time

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import GradientBoostingRegressor
from tqdm import tqdm

from plotting.plot_n_sii import transform_interactions_in_n_shapley, plot_n_sii
from tree_shap_iq.conversion import convert_tree_estimator
from tree_shap_iq import TreeShapIQ


if __name__ == "__main__":

    RANDOM_STATE = 42

    MAX_INTERACTION_ORDER = 6
    EXPLANATION_INDEX = 0

    DO_OBSERVATIONAL = False
    DO_TREE_SHAP = True
    if DO_TREE_SHAP:
        from shap import TreeExplainer

    # load the california housing dataset ----------------------------------------------------------
    data = fetch_california_housing(as_frame=True)
    X, y = data.data, data.target
    n_features = X.shape[-1]
    n_samples = len(X)
    feature_names = data["feature_names"]
    # abbreviate feature names and take only first 5 chars and add a .
    feature_names = [feature[:5] + "." for feature in feature_names]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.7, shuffle=True, random_state=RANDOM_STATE)
    explanation_id = X_test.index[EXPLANATION_INDEX]
    # get explanation datapoint
    x_explain = np.asarray(X_test.iloc[EXPLANATION_INDEX].values)
    y_true_label = y_test.iloc[EXPLANATION_INDEX]
    # transform data to numpy arrays
    X, X_train, X_test, y_train, y_test = X.values, X_train.values, X_test.values, y_train.values, y_test.values

    # fit a tree model -----------------------------------------------------------------------------

    model = GradientBoostingRegressor(
        max_depth=3, learning_rate=0.1, min_samples_leaf=5, n_estimators=100, max_features=1.0,
        #max_depth=10, learning_rate=0.1, min_samples_leaf=5, n_estimators=100, max_features=1.0,
        random_state=RANDOM_STATE
    )
    model.fit(X_train, y_train)
    print("R^2 on test data", model.score(X_test, y_test))
    model_output = model.predict(x_explain.reshape(1, -1))
    print("Model output", model_output, "True label", y_true_label)

    # convert the tree -----------------------------------------------------------------------------

    list_of_trees = convert_tree_estimator(model)

    # explain with TreeShapIQ observational --------------------------------------------------------
    print("\nTreeShapIQ explanations (observational) ------------------")
    explanation_time, empty_prediction = 0, 0.
    sii_values_dict: dict[int, np.ndarray] = {order: 0. for order in range(1, MAX_INTERACTION_ORDER + 1)}  # will be filled with the SII values for each interaction order
    for i, tree_model in tqdm(enumerate(list_of_trees), total=len(list_of_trees)):
        explainer = TreeShapIQ(
            tree_model=tree_model,
            max_interaction_order=MAX_INTERACTION_ORDER,
            n_features=n_features,
            observational=True,
            background_dataset=X
        )
        start_time = time.time()
        explanation_scores: dict[int, np.ndarray] = explainer.explain(x=x_explain)
        explanation_time += time.time() - start_time
        sii_values_dict = {
            order: sii_values_dict[order] + explanation_scores[order]
            for order in range(1, MAX_INTERACTION_ORDER + 1)
        }
        empty_prediction += explainer.empty_prediction
    print("Time taken", explanation_time)
    print(sii_values_dict[1])
    print("Empty prediction", empty_prediction)
    print("Sum", np.sum(sii_values_dict[1]) + empty_prediction)

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
    fig_obs, axis_obs = plot_n_sii(
        n_shapley_values_pos=n_shapley_values_pos,
        n_shapley_values_neg=n_shapley_values_neg,
        feature_names=feature_names,
        n_sii_order=MAX_INTERACTION_ORDER
    )

    # explain with TreeShapIQ interventional -------------------------------------------------------
    print("\nTreeShapIQ explanations (interventional) ------------------")
    explanation_time, empty_prediction = 0, 0.
    sii_values_dict: dict[int, np.ndarray] = {order: 0. for order in range(1, MAX_INTERACTION_ORDER + 1)}  # will be filled with the SII values for each interaction order
    for i, tree_model in tqdm(enumerate(list_of_trees), total=len(list_of_trees)):
        explainer = TreeShapIQ(
            tree_model=tree_model,
            max_interaction_order=MAX_INTERACTION_ORDER,
            n_features=n_features,
            observational=False,
            background_dataset=X
        )
        start_time = time.time()
        explanation_scores: dict[int, np.ndarray] = explainer.explain(x=x_explain)
        explanation_time += time.time() - start_time
        sii_values_dict = {
            order: sii_values_dict[order] + explanation_scores[order]
            for order in range(1, MAX_INTERACTION_ORDER + 1)
        }
        empty_prediction += explainer.empty_prediction
    print("Time taken", explanation_time)
    print(sii_values_dict[1])
    print("Empty prediction", empty_prediction)
    print("Sum", np.sum(sii_values_dict[1]) + empty_prediction)

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
    fig_int, axis_int = plot_n_sii(
        n_shapley_values_pos=n_shapley_values_pos,
        n_shapley_values_neg=n_shapley_values_neg,
        feature_names=feature_names,
        n_sii_order=MAX_INTERACTION_ORDER
    )

    # make plots content and show the plots --------------------------------------------------------
    ylim_obs = axis_obs.get_ylim()
    ylim_int = axis_int.get_ylim()
    axis_obs.set_ylim([min(ylim_obs[0], ylim_int[0]), max(ylim_obs[1], ylim_int[1])])
    axis_int.set_ylim([min(ylim_obs[0], ylim_int[0]), max(ylim_obs[1], ylim_int[1])])

    axis_obs.set_title(f"n-SII (observational) for instance {explanation_id}")
    axis_int.set_title(f"n-SII (interventional) for instance {explanation_id}")

    fig_int.show()
    fig_obs.show()

    # explain with TreeShap ------------------------------------------------------------------------
    if DO_TREE_SHAP:
        print("\nTreeShap explanations (observational) ------------------")
        explainer_shap = TreeExplainer(model, feature_perturbation="tree_path_dependent")
        sv_shap = explainer_shap.shap_values(x_explain).copy()
        empty_prediction = explainer_shap.expected_value
        print(sv_shap)
        print("Empty prediction", empty_prediction)
        print("Sum", np.sum(sv_shap) + empty_prediction)

        print("\nTreeShap explanations (interventional) ------------------")
        explainer_shap = TreeExplainer(model, feature_perturbation="interventional", data=X_test[:50])
        sv_shap = explainer_shap.shap_values(x_explain).copy()
        empty_prediction = explainer_shap.expected_value
        print(sv_shap)
        print("Empty prediction", empty_prediction)
        print("Sum", np.sum(sv_shap) + empty_prediction)
