"""This module is used to run the experiment on the qsar-biodeg dataset."""
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from tqdm import tqdm

from linear_interaction.conversion import convert_tree_estimator
from linear_interaction.tree_shap_iq import TreeShapIQ

if __name__ == "__main__":

    RANDOM_STATE = 42
    INTERACTION_ORDER = 3
    DO_OBSERVATIONAL = True
    DO_TREE_SHAP = False
    if DO_TREE_SHAP:
        from shap import TreeExplainer

    # read qsar_biodeg.csv file
    data = pd.read_csv("data/qsar_biodeg.csv", sep=",")
    X_data = data.drop(columns=["Class"])
    y_data = data["Class"]

    # make a train test split with sklearn
    X_train, X_test, y_train, y_test = train_test_split(
        X_data, y_data, test_size=0.2, random_state=RANDOM_STATE)

    # fit an GradientBoostingClassifier model to the train data
    model = GradientBoostingClassifier(max_depth=20, random_state=RANDOM_STATE, n_estimators=10)
    model.fit(X_train, y_train)

    # evaluate the model on the test data
    print("Accuracy on test data", model.score(X_test, y_test))

    # explain the model on a single datapoint
    x_explain = np.asarray(X_test.iloc[0])
    y_true_label = y_test.iloc[0]
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
        explanation_scores = explainer.explain(x=x_explain, order=INTERACTION_ORDER)
        explanation_time += time.time() - start_time
        ensemble_explanations.append(explanation_scores.copy())
        empty_prediction = empty_predictions.append(explainer.empty_prediction)
    ensemble_explanations = np.sum(np.asarray(ensemble_explanations), axis=0)
    empty_predictions = np.asarray(empty_predictions)
    print("TreeShapIQ explanations")
    print("Time taken", explanation_time)
    print(ensemble_explanations)
    print("Sum", np.sum(ensemble_explanations))

    # explain with TreeShap

    if DO_TREE_SHAP:
        print("\nTreeShap explanations ------------------")
        if DO_OBSERVATIONAL:
            explainer_shap = TreeExplainer(model, feature_perturbation="tree_path_dependent")
        else:
            explainer_shap = TreeExplainer(model, feature_perturbation="interventional", data=X_test[:50])
        sv_shap = explainer_shap.shap_values(x_explain).copy()
        empty_prediction = explainer_shap.expected_value
        print("TreeSHAP explanations")
        print(sv_shap)
        print("Sum", np.sum(sv_shap))
