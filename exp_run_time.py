"""This module is used to run the run-time experiment."""
import copy
import time

from tqdm import tqdm
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from tree_shap_iq import TreeShapIQ
from tree_shap_iq.conversion import convert_tree_estimator

if __name__ == "__main__":

    random_state: int = 42
    explanation_index: int = 0

    n_iterations: int = 10

    max_interaction_order_params: list[int] = [2]
    n_features_params: list[int] = [10, 20, 30, 40]

    # init data storage ----------------------------------------------------------------------------

    data_storage = []

    # get cross-product of max_interaction_order and n_features
    all_params = [(a, b) for a in max_interaction_order_params for b in n_features_params]
    for max_interaction_order, n_features in all_params:

        print(f"Running experiment for max_interaction_order={max_interaction_order} "
              f"and n_features={n_features}")

        # create a synth regression dataset with n_features ----------------------------------------

        X, y = make_regression(
            n_samples=2_000,
            n_features=n_features,
            n_informative=n_features,
            n_targets=1,
            bias=0.0,
            effective_rank=None,
            tail_strength=0.5,
            noise=0.0,
            shuffle=True,
            coef=False,
            random_state=random_state
        )

        # train test split and get explanation datapoint -------------------------------------------

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=0.7, shuffle=True, random_state=random_state)

        x_explain = np.asarray(X_test[explanation_index])
        y_true_label = y_test[explanation_index]

        # init model data storages -----------------------------------------------------------------

        dt_storage = {
            "model_id": "DT",
            "n_features": n_features,
            "interaction_order": max_interaction_order,
        }

        rf_storage = {
            "model_id": "RF",
            "n_features": n_features,
            "interaction_order": max_interaction_order,
        }

        xgb_storage = {
            "model_id": "XGB",
            "n_features": n_features,
            "interaction_order": max_interaction_order,
        }

        # run-time analysis for a decision tree model ----------------------------------------------

        model: DecisionTreeRegressor = DecisionTreeRegressor(
            max_depth=15,
            random_state=random_state
        )
        model.fit(X_train, y_train)
        depth = model.get_depth()
        n_nodes = model.tree_.node_count

        dt_storage["max_depth"] = int(depth)
        dt_storage["n_nodes"] = int(n_nodes)

        print(f"DT max Depth:", depth, ",# Nodes:", n_nodes)

        # convert tree
        tree_model = convert_tree_estimator(model)

        time.sleep(0.5)
        for iteration in tqdm(range(1, n_iterations + 1), total=n_iterations):
            iteration_storage = copy.deepcopy(dt_storage)
            explainer = TreeShapIQ(
                tree_model=copy.deepcopy(tree_model),
                max_interaction_order=max_interaction_order,
                n_features=n_features,
                observational=True,
                interaction_type="SII"
            )
            start_time = time.time()
            _ = explainer.explain(x_explain, max_interaction_order, min_order=max_interaction_order)
            elapsed_time = time.time() - start_time
            iteration_storage["elapsed_time"] = elapsed_time
            data_storage.append(copy.deepcopy(iteration_storage))

        print("Finished DT")

        # run-time analysis for a random forest model ----------------------------------------------

        model: RandomForestRegressor = RandomForestRegressor(
            n_estimators=10,
            max_depth=10,
            random_state=random_state
        )
        model.fit(X_train, y_train)
        # get max depth of base estimators
        max_depth, node_count = 0, 0
        for estimator in model.estimators_:
            max_depth = max(max_depth, int(estimator.get_depth()))
            node_count += estimator.tree_.node_count

        rf_storage["max_depth"] = int(max_depth)
        rf_storage["n_nodes"] = int(node_count)

        print(f"RF max Depth:", max_depth, ",# Nodes:", node_count)

        # convert tree
        tree_model = convert_tree_estimator(model)

        time.sleep(0.5)
        for iteration in tqdm(range(1, n_iterations + 1), total=n_iterations):
            iteration_storage = copy.deepcopy(rf_storage)
            elapsed_time = 0
            for tree_estimator in tree_model:
                explainer = TreeShapIQ(
                    tree_model=copy.deepcopy(tree_estimator),
                    max_interaction_order=max_interaction_order,
                    n_features=n_features,
                    observational=True,
                    interaction_type="SII"
                )
                start_time = time.time()
                _ = explainer.explain(x_explain, max_interaction_order, min_order=max_interaction_order)
                elapsed_time += time.time() - start_time
            iteration_storage["elapsed_time"] = elapsed_time
            data_storage.append(copy.deepcopy(iteration_storage))

        print("Finished RF")

        # run-time analysis for a xgboost model ----------------------------------------------------

        model: XGBRegressor = XGBRegressor(
            n_estimators=10,
            max_depth=10,
            random_state=random_state
        )
        model.fit(X_train, y_train)

        # convert tree
        tree_model = convert_tree_estimator(model)

        max_depth = model.max_depth
        node_count = sum([len(estimator.children_left) for estimator in tree_model])

        xgb_storage["max_depth"] = int(max_depth)
        xgb_storage["n_nodes"] = int(node_count)

        print(f"XGB max Depth:", max_depth, ",# Nodes:", node_count)

        time.sleep(0.5)
        for iteration in tqdm(range(1, n_iterations + 1), total=n_iterations):
            iteration_storage = copy.deepcopy(xgb_storage)
            elapsed_time = 0
            for tree_estimator in tree_model:
                explainer = TreeShapIQ(
                    tree_model=copy.deepcopy(tree_estimator),
                    max_interaction_order=max_interaction_order,
                    n_features=n_features,
                    observational=True,
                    interaction_type="SII"
                )
                start_time = time.time()
                _ = explainer.explain(x_explain, max_interaction_order, min_order=max_interaction_order)
                elapsed_time += time.time() - start_time
            iteration_storage["elapsed_time"] = elapsed_time
            data_storage.append(copy.deepcopy(iteration_storage))

        print("Finished XGB")

    # save data -----------------------------------------------------------------------------------
    print("Finished all experiments")
    data_df = pd.DataFrame(data_storage)
    data_df.to_csv("run_time.csv", index=False)
