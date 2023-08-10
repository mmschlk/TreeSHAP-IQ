"""This module is used to run the experiment on the california-housing dataset."""
import time

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, \
    GradientBoostingClassifier
from sklearn.preprocessing import OrdinalEncoder
from tqdm import tqdm


from plotting.network import draw_interaction_network
from tree_shap_iq.conversion import convert_tree_estimator
from plotting.plot_n_sii import transform_interactions_in_n_shapley, plot_n_sii
from tree_shap_iq import TreeShapIQ


if __name__ == "__main__":
    RANDOM_STATE = 42

    MAX_INTERACTION_ORDER = 2
    EXPLANATION_INDEX = 1

    SAVE_FIGURES = True

    # load the german credit risk dataset from disc ------------------------------------------------
    data = pd.read_csv("data/german_credit_risk.csv")
    X = data.drop(columns=["GoodCredit"])
    y = data["GoodCredit"]
    n_features = X.shape[-1]
    n_samples = len(X)

    # data preprocessing
    cat_columns = [
        "checkingstatus", "history", "purpose", "savings", "employ", "status", "others", "property",
        "otherplans", "housing", "job", "tele", "foreign"
    ]
    X[cat_columns] = OrdinalEncoder().fit_transform(X[cat_columns])
    X = X.astype(float)
    y = y.replace({1: 1, 2: 0})

    feature_names = list(X.columns)
    # abbreviate feature names and take only first 5 chars and add a .

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.7, shuffle=True, random_state=RANDOM_STATE)
    explanation_id = X_test.index[EXPLANATION_INDEX]
    # get explanation datapoint
    x_explain = np.asarray(X_test.iloc[EXPLANATION_INDEX].values)
    y_true_label = y_test.iloc[EXPLANATION_INDEX]
    # transform data to numpy arrays
    X, X_train, X_test, y_train, y_test = X.values, X_train.values, X_test.values, y_train.values, y_test.values

    feature_names_abbrev = [feature[:4] for feature in feature_names]
    feature_names_values = [feature + f".\n(" + str(round(x_explain[i], 2)) + ")" for i, feature in enumerate(feature_names_abbrev)]

    # fit a tree model -----------------------------------------------------------------------------

    model = GradientBoostingClassifier(
        # max_depth=3, learning_rate=0.1, min_samples_leaf=5, n_estimators=100, max_features=1.0,
        max_depth=5, learning_rate=0.1, min_samples_leaf=5, n_estimators=100, max_features=1.0,
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
    print("Empty prediction", empty_prediction)
    print("Sum", np.sum(sii_values_dict[1]) + empty_prediction)

    # transform the SII values in n-Shapley values
    n_sii_values = transform_interactions_in_n_shapley(
        interaction_values=sii_values_dict,
        n=MAX_INTERACTION_ORDER,
        n_features=n_features,
        reduce_one_dimension=False
    )

    print(np.round(n_sii_values[1], 2))
    print(np.round(n_sii_values[2], 2))

    # plot the n-Shapley values
    fig, axis = draw_interaction_network(
        first_order_values=n_sii_values[1],
        second_order_values=n_sii_values[2],
        feature_names=feature_names_values,
        n_features=n_features,
    )

    # make plots content and show the plots --------------------------------------------------------
    axis.set_title(f"German Credit: n-SII network plot for instance {explanation_id}")

    if SAVE_FIGURES:
        fig.savefig(f"plots/network_german_credit_risk_obs_{explanation_id}.pdf", bbox_inches="tight")

    fig.tight_layout()
    fig.show()
