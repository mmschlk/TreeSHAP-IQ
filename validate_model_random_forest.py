import time
from copy import deepcopy

from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
import numpy as np

from tree_shap_iq.conversion import convert_tree_estimator
from tree_shap_iq.base import TreeShapIQ

if __name__ == "__main__":
    DO_TREE_SHAP = True
    DO_OBSERVATIONAL = True

    INTERACTION_ORDER = 1

    if DO_TREE_SHAP:
        try:
            from shap import TreeExplainer
        except ImportError:
            print("TreeSHAP not available. Please install shap package.")
            DO_TREE_SHAPE = False

    # fix random seed for reproducibility
    random_seed = 10
    np.random.seed(random_seed)

    # create dummy regression dataset and fit tree model
    X, y = make_regression(1000, n_features=15)
    n_features = X.shape[-1]
    model = RandomForestRegressor(max_depth=10, random_state=random_seed, n_estimators=10)
    model.fit(X, y)

    x_explain = X[:1][0]
    print("Output f(x):", model.predict(x_explain.reshape(1, -1))[0])

    # convert the tree
    list_of_trees = convert_tree_estimator(model)

    # explain the tree with LinearTreeSHAP
    print("\nTreeShapIQ explanations ------------------")
    explanation_time, ensemble_explanations, empty_predictions = 0, [], []
    for tree in list_of_trees:
        explainer = TreeShapIQ(
            tree_model=deepcopy(tree), n_features=n_features, observational=DO_OBSERVATIONAL,
            max_interaction_order=INTERACTION_ORDER, background_dataset=X
        )
        start_time = time.time()
        explanation_scores = explainer.explain(x=x_explain, order=INTERACTION_ORDER)[INTERACTION_ORDER]
        explanation_time += time.time() - start_time
        ensemble_explanations.append(explanation_scores.copy())
        empty_prediction = empty_predictions.append(explainer.empty_prediction)
        time_elapsed = time.time() - start_time
    ensemble_explanations = np.sum(np.asarray(ensemble_explanations), axis=0)
    empty_predictions = np.asarray(empty_predictions)
    print("Time taken", explanation_time)
    print(ensemble_explanations)
    print("empty", sum(empty_predictions))
    print("Sum", np.sum(ensemble_explanations) + sum(empty_predictions))

    if DO_TREE_SHAP:
        # explain the tree with observational TreeSHAP
        start_time = time.time()
        if DO_OBSERVATIONAL:
            explainer_shap = TreeExplainer(deepcopy(model), feature_perturbation="tree_path_dependent")
        else:
            explainer_shap = TreeExplainer(deepcopy(model), feature_perturbation="interventional", data=X[:100])
        sv_tree_shap = explainer_shap.shap_values(x_explain)
        time_elapsed = time.time() - start_time
        empty_prediction = explainer_shap.expected_value
        print("TreeSHAP explanations")
        print(sv_tree_shap)
        print("Sum", np.sum(sv_tree_shap) + empty_prediction)
        print("Time taken", time_elapsed)
        print("Empty prediction", empty_prediction)
