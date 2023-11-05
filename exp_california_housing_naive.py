"""This module is used to run the experiment on the california-housing dataset."""
import time

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from tqdm import tqdm
from xgboost import XGBRegressor

from experiment_main import run_main_experiment
from tree_shap_iq import TreeShapIQ
from tree_shap_iq.conversion import convert_tree_estimator

if __name__ == "__main__":

    # Settings used for the force plot in the paper
    # random_state = 42
    # MAX_INTERACTION_ORDER = 2,3
    # EXPLANATION_INDEX = 2
    # force_limits = (0.4, 6.8)
    # model = GradientBoostingRegressor(max_depth=10, learning_rate=0.1, min_samples_leaf=5, n_estimators=100, max_features=1.0)

    random_state = 42
    MAX_INTERACTION_ORDER = 2
    EXPLANATION_INDEX = 2
    n_dummy_features = 4
    dataset_name: str = "California"

    model_flag: str = "DT"  # "XGB" or "RF", "DT", "GBT", None
    if model_flag is not None:
        print("Model:", model_flag)

    # load the california housing dataset and pre-process ------------------------------------------
    data = fetch_california_housing(as_frame=True)
    X, y = data.data, data.target
    dummy_names = []
    for dummy_i in range(n_dummy_features):
        dummy_name = "dummy_"+str(dummy_i)
        X[dummy_name] = 0.
        dummy_names.append(dummy_name)
    n_features = X.shape[-1]
    n_samples = len(X)
    feature_names = data["feature_names"] + dummy_names

    # train test split and get explanation datapoint -----------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.7, shuffle=True, random_state=random_state)
    explanation_id = X_test.index[EXPLANATION_INDEX]

    # get explanation datapoint and index
    x_explain = np.asarray(X_test.iloc[EXPLANATION_INDEX].values)
    y_true_label = y_test.iloc[EXPLANATION_INDEX]

    # transform data to numpy arrays
    X, X_train, X_test, y_train, y_test = (
        X.values, X_train.values, X_test.values, y_train.values, y_test.values
    )

    print("n_features", n_features, "n_samples", n_samples)

    # fit a tree model -----------------------------------------------------------------------------

    if model_flag == "RF":
        model: RandomForestRegressor = RandomForestRegressor(random_state=random_state, n_estimators=5, max_depth=10)
    elif model_flag == "DT":
        model: DecisionTreeRegressor = DecisionTreeRegressor(random_state=random_state)
    elif model_flag == "GBT":
        model: GradientBoostingRegressor = GradientBoostingRegressor(max_depth=10, learning_rate=0.1, min_samples_leaf=5, n_estimators=100, max_features=1.0, random_state=random_state)
    else:
        model: XGBRegressor = XGBRegressor(random_state=random_state)

    model.fit(X_train, y_train)
    print("R^2 on test data", model.score(X_test, y_test))

    # run the experiment --------------------------------------------------------------------------

    tree_model = convert_tree_estimator(model, class_label=None, output_type="raw")
    if type(tree_model) != list:
        tree_model = [tree_model]

    explanation_time = 0.
    empty_prediction = 0.
    sii_values_dict: dict[int, np.ndarray] = {
        order: 0. for order in range(1, MAX_INTERACTION_ORDER + 1)
    }

    explanation_time_naive = 0.
    empty_prediction_naive = 0.
    sii_values_naive_dict: dict[int, np.ndarray] = {
        order: 0. for order in range(1, MAX_INTERACTION_ORDER + 1)
    }

    for i, tree_ in tqdm(enumerate(tree_model), total=len(tree_model)):
        explainer = TreeShapIQ(
            tree_model=tree_,
            max_interaction_order=MAX_INTERACTION_ORDER,
            observational=True,
            n_features=n_features,
            interaction_type="SII"
        )
        start_time = time.time()
        explanation_scores: dict[int, np.ndarray] = explainer.explain(x=x_explain)
        explanation_time += time.time() - start_time
        sii_values_dict = {
            order: sii_values_dict[order] + explanation_scores[order]
            for order in range(1, MAX_INTERACTION_ORDER + 1)
        }
        empty_prediction += explainer.empty_prediction

        # naive_implementation ---------------------------------------------------------------------

        explainer_naive = TreeShapIQ(
            tree_model=tree_,
            max_interaction_order=MAX_INTERACTION_ORDER,
            observational=True,
            n_features=n_features,
            interaction_type="SII"
        )
        #explainer_naive.values = explainer.values - explainer_naive.empty_prediction
        start_time = time.time()
        explanation_scores, _ = explainer_naive.explain_brute_force(
            x=x_explain,
            max_order=MAX_INTERACTION_ORDER
        )
        explanation_time_naive += time.time() - start_time
        sii_values_naive_dict = {
            order: sii_values_naive_dict[order] + explanation_scores[order]
            for order in range(1, MAX_INTERACTION_ORDER + 1)
        }
        empty_prediction_naive += explanation_scores[0]

    print("Comparison TreeSHAP-IQ and Naive")
    print("Dataset:", dataset_name, "# features", n_features, "# dummy features", n_dummy_features)
    print("Order:", MAX_INTERACTION_ORDER, "Model:", model_flag, "# of trees in model:", len(tree_model))

    print("TreeSHAP-IQ")
    print("Time", explanation_time)
    #print(sii_values_dict)
    print("Empty prediction:", empty_prediction)
    print()

    print("Naive")
    print("Time", explanation_time_naive)
    #print(sii_values_naive_dict)
    print("Empty prediction:", empty_prediction_naive)
