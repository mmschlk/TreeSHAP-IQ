import time

import numpy as np
from shap import TreeExplainer
from sklearn.datasets import make_regression
from sklearn.ensemble import GradientBoostingRegressor
from tqdm import tqdm

from linear_interaction.conversion import convert_tree_estimator
from linear_interaction.tree_shap_iq import TreeShapIQ

if __name__ == "__main__":
    DO_TREE_SHAP = False
    DO_OBSERVATIONAL = True
    DO_GROUND_TRUTH = False

    COMPUTATION_MODE = "fast-original"  # "original", "only_in_leaf", "only_in_leaf_fast"

    RANDOM_SEED = 42
    INTERACTION_ORDER = 2

    X, y = make_regression(1000, n_features=100, random_state=RANDOM_SEED)
    model = GradientBoostingRegressor(max_depth=50, random_state=RANDOM_SEED, n_estimators=50)
    model.fit(X, y)

    # explanation datapoint
    x_explain = X[0]
    model_output = model.predict(x_explain.reshape(1, -1))
    print("Model output", model_output)

    list_of_trees = convert_tree_estimator(model)

    print("\nTreeShapIQ explanations ------------------")

    ensemble_explanations = []
    empty_predictions = []
    explanation_time = 0
    for i, tree_model in tqdm(enumerate(list_of_trees), total=len(list_of_trees)):
        explainer = TreeShapIQ(
            tree_model=tree_model,
            max_interaction_order=INTERACTION_ORDER,
            n_features=X.shape[1],
            observational=DO_OBSERVATIONAL
        )
        start_time = time.time()
        explanation_scores = explainer.explain(x=x_explain, order=INTERACTION_ORDER, mode=COMPUTATION_MODE)
        explanation_time += time.time() - start_time
        ensemble_explanations.append(explanation_scores.copy())
        empty_prediction = empty_predictions.append(explainer.empty_prediction)
    ensemble_explanations = np.sum(np.asarray(ensemble_explanations), axis=0)
    empty_predictions = np.asarray(empty_predictions)
    print("TreeShapIQ explanations")
    print("Time taken", explanation_time)
    print(ensemble_explanations)
    print("Sum", np.sum(ensemble_explanations))
    print("Empty predictions for all base models", empty_predictions)
    print("Sum of empty predictions", np.sum(empty_predictions))

    difference = np.sum(ensemble_explanations) - model_output

    if DO_GROUND_TRUTH:
        ensemble_explanations = []
        empty_predictions = []
        explanation_time = 0
        print("\nGround Truth ------------------")
        for i, tree_model in tqdm(enumerate(list_of_trees), total=len(list_of_trees)):
            explainer = TreeShapIQ(
                tree_model=tree_model,
                max_interaction_order=INTERACTION_ORDER,
                n_features=X.shape[1],
                observational=DO_OBSERVATIONAL
            )
            start_time = time.time()
            gt_result = explainer.explain_brute_force(x_explain, INTERACTION_ORDER)
            explanation_time += time.time() - start_time
            ground_truth_shap_int, ground_truth_shap_int_pos = gt_result

            ensemble_explanations.append(ground_truth_shap_int[INTERACTION_ORDER].copy())
        ensemble_explanations = np.array(ensemble_explanations)
        ensemble_explanations = np.sum(ensemble_explanations, axis=0)
        print("Ground Truth explanations")
        print("Time taken", explanation_time)
        print(ensemble_explanations)
        print("Sum", np.sum(ensemble_explanations))

    if DO_TREE_SHAP:
        print("\nTreeSHAP explanations ------------------")
        # explain the tree with observational TreeSHAP
        if DO_OBSERVATIONAL:
            explainer_shap = TreeExplainer(model, feature_perturbation="tree_path_dependent")
        else:
            explainer_shap = TreeExplainer(model, feature_perturbation="interventional", data=X[:50])
        sv_shap = explainer_shap.shap_values(x_explain).copy()
        empty_prediction = explainer_shap.expected_value
        print("TreeSHAP explanations")
        print(sv_shap)
        print("Sum", np.sum(sv_shap))
        print("Empty prediction", empty_prediction)
        print("Sum with empty prediction", np.sum(sv_shap) + empty_prediction)
