"""This module is used to run the experiment on the california-housing dataset."""
import time

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import GradientBoostingRegressor
from tqdm import tqdm

from tree_shap_iq.conversion import convert_tree_estimator
from tree_shap_iq import TreeShapIQ


if __name__ == "__main__":

    RANDOM_STATE = 42

    INTERACTION_ORDER = 1
    EXPLANATION_ID = 1

    DO_OBSERVATIONAL = True
    DO_TREE_SHAP = True
    if DO_TREE_SHAP:
        from shap import TreeExplainer

    data = fetch_california_housing()
    X, y = data.data, data.target
    n_features = X.shape[-1]
    n_samples = len(X)
    feature_names = data["feature_names"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True, random_state=RANDOM_STATE)

    model = GradientBoostingRegressor(random_state=RANDOM_STATE, n_estimators=50)
    model.fit(X_train, y_train)

    print("R^2 on test data", model.score(X_test, y_test))

    # explain the model on a single datapoint
    x_explain = np.asarray(X_test[EXPLANATION_ID])
    y_true_label = y_test[EXPLANATION_ID]
    model_output = model.predict(x_explain.reshape(1, -1))
    print("Model output", model_output, "True label", y_true_label)

    list_of_trees = convert_tree_estimator(model)

    # explain with TreeShapIQ
    print("\nTreeShapIQ explanations ------------------")
    explanation_time, ensemble_explanations, empty_predictions = 0, [], []
    for i, tree_model in tqdm(enumerate(list_of_trees), total=len(list_of_trees)):
        explainer = TreeShapIQ(
            tree_model=tree_model,
            max_interaction_order=INTERACTION_ORDER,
            n_features=x_explain.shape[0],
            observational=DO_OBSERVATIONAL
        )
        start_time = time.time()
        explanation_scores = explainer.explain(x=x_explain, order=INTERACTION_ORDER)[INTERACTION_ORDER]
        explanation_time += time.time() - start_time
        ensemble_explanations.append(explanation_scores.copy())
        empty_prediction = empty_predictions.append(explainer.empty_prediction)
    ensemble_explanations = np.sum(np.asarray(ensemble_explanations), axis=0)
    empty_prediction = sum(np.asarray(empty_predictions))
    print("Time taken", explanation_time)
    print(ensemble_explanations)
    print("Empty prediction", empty_prediction)
    print("Sum", np.sum(ensemble_explanations) + empty_prediction)

    # explain with TreeShap

    if DO_TREE_SHAP:
        print("\nTreeShap explanations ------------------")
        if DO_OBSERVATIONAL:
            explainer_shap = TreeExplainer(model, feature_perturbation="tree_path_dependent")
        else:
            explainer_shap = TreeExplainer(model, feature_perturbation="interventional",
                                           data=X_test[:50])
        sv_shap = explainer_shap.shap_values(x_explain).copy()
        empty_prediction = explainer_shap.expected_value
        print(sv_shap)
        print("Empty prediction", empty_prediction)
        print("Sum", np.sum(sv_shap) + empty_prediction)
