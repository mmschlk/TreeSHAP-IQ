"""This module is used to run the complexity (run-time) experiment."""
import copy
import time

from sklearn.preprocessing import OrdinalEncoder
from tqdm import tqdm
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


from tree_shap_iq import TreeShapIQ
from tree_shap_iq.conversion import convert_tree_estimator

if __name__ == "__main__":

    random_state: int = 42

    n_iterations: int = 10
    max_interaction_order_params: list[int] = [1, 2, 3, 4, 5, 6]
    max_depth_params = [35]

    # get data -------------------------------------------------------------------------------------

    explanation_index: int = 0

    data = pd.read_csv("data/adult.csv")
    data = data.dropna()
    y = data["label"]
    data = data.drop(columns=["label"])

    num_feature_names = [
        'age', 'capital-gain', 'capital-loss', 'hours-per-week', 'fnlwgt'
    ]
    cat_feature_names = [
        'workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex',
        'native-country', 'education-num'
    ]
    data[num_feature_names] = data[num_feature_names].apply(pd.to_numeric)
    data[cat_feature_names] = OrdinalEncoder().fit_transform(data[cat_feature_names])
    data.dropna(inplace=True)

    X = data
    n_features = X.shape[-1]
    n_samples = len(X)
    feature_names = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.7, shuffle=True, random_state=random_state)
    explanation_id = X_test.index[explanation_index]

    # get explanation datapoint and index
    x_explain = np.asarray(X_test.iloc[explanation_index].values)
    y_true_label = y_test.iloc[explanation_index]

    # transform data to numpy arrays
    X, X_train, X_test, y_train, y_test = (
        X.values, X_train.values, X_test.values, y_train.values, y_test.values
    )

    print("n_features", n_features, "n_samples", n_samples)

    # init data storage ----------------------------------------------------------------------------

    data_storage = []

    # get cross-product of max_interaction_order and n_features
    all_params = [(a, b) for a in max_interaction_order_params for b in max_depth_params]
    for max_interaction_order, max_depth in all_params:

        print(f"Running experiment for max_interaction_order={max_interaction_order} "
              f"and max_depth={max_depth}")

        # init model data storages -----------------------------------------------------------------

        dt_storage = {
            "model_id": "DT",
            "n_features": n_features,
            "interaction_order": max_interaction_order
        }

        # run-time analysis for a decision tree model ----------------------------------------------

        model: DecisionTreeClassifier = DecisionTreeClassifier(
            max_depth=max_depth,
            random_state=random_state,
            min_samples_leaf=2
        )
        model.fit(X, y)

        # convert tree
        tree_model = convert_tree_estimator(model)

        # get depth and # nodes and # leafs
        depth = model.get_depth()
        n_nodes = model.tree_.node_count
        n_decision_nodes = n_nodes - model.tree_.n_leaves
        n_leaves = model.tree_.n_leaves

        # count the number of left and right children (value >in tree_model

        dt_storage["depth"] = int(depth)
        dt_storage["n_nodes"] = int(n_nodes)
        dt_storage["n_decision_nodes"] = int(n_decision_nodes)
        dt_storage["n_leaves"] = int(n_leaves)
        dt_storage["leaves_times_depth"] = int(n_leaves * depth)

        print(f"DT max Depth:", depth, ",# Nodes:", n_nodes, ",# Decision Nodes:",
              n_decision_nodes, ",# Leaves:", n_leaves)

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

    # save data -----------------------------------------------------------------------------------
    print("Finished all experiments")
    data_df = pd.DataFrame(data_storage)
    data_df.to_csv(f"run_time_interaction_{str(depth)}.csv", index=False)
