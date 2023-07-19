import time

import numpy as np

from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier
from tqdm import tqdm

from linear_interaction.conversion import convert_tree_estimator
from linear_interaction.tree_shap_iq import TreeShapIQ

if __name__ == "__main__":
    DO_TREE_SHAP = True
    if DO_TREE_SHAP:
        from shap import TreeExplainer
    DO_OBSERVATIONAL = True
    DO_GROUND_TRUTH = False

    RANDOM_SEED = 42
    INTERACTION_ORDER = 1
    N_CLASSES = 2
    if N_CLASSES > 2 and DO_TREE_SHAP:
        print("TreeShap does not support multiclass classification with GradientBoostingClassifier."
              "Setting 'DO_TREE_SHAP' to False.")
        DO_TREE_SHAP = False

    X, y = make_classification(1000, n_features=15, random_state=RANDOM_SEED, n_classes=N_CLASSES, n_informative=5)
    model = GradientBoostingClassifier(max_depth=10, random_state=RANDOM_SEED, n_estimators=10)
    model.fit(X, y)

    # explanation datapoint
    x_explain = X[0]
    y_true_label = y[0]
    model_output = model.predict_proba(x_explain.reshape(1, -1))
    print("Model output", model_output, "True label", y[0])
    list_of_trees = convert_tree_estimator(model, class_label=y_true_label)

    print("TreeShapIQ explanations ------------------")

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
    #print("Empty prediction", empty_predictions)
    #print("Sum with empty prediction", np.sum(ensemble_explanations) + np.sum(empty_predictions))

    difference = np.sum(ensemble_explanations) - model_output

    if DO_GROUND_TRUTH:
        ensemble_explanations = []
        explanation_time = 0
        print("Ground Truth ------------------")
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
        print("TreeSHAP explanations ------------------")
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
        #print("Empty prediction", empty_prediction)
        #print("Sum with empty prediction", np.sum(sv_shap) + empty_prediction)
