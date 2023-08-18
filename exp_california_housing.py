"""This module is used to run the experiment on the california-housing dataset."""
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from experiment_main import run_main_experiment


if __name__ == "__main__":

    # Settings used for the force plot in the paper
    # random_state = 42
    # MAX_INTERACTION_ORDER = 2,3
    # EXPLANATION_INDEX = 2
    # force_limits = (0.4, 6.8)
    # model = GradientBoostingRegressor(max_depth=10, learning_rate=0.1, min_samples_leaf=5, n_estimators=100, max_features=1.0)

    random_state = 42
    MAX_INTERACTION_ORDER = 2
    EXPLANATION_INDEX = 3
    SAVE_FIGURES = True
    dataset_name: str = "California"

    force_limits = (0.4, 6.8)

    model_flag: str = "GBT"  # "XGB" or "RF", "DT", "GBT", None
    if model_flag is not None:
        print("Model:", model_flag)

    # load the california housing dataset and pre-process ------------------------------------------
    data = fetch_california_housing(as_frame=True)
    X, y = data.data, data.target
    n_features = X.shape[-1]
    n_samples = len(X)
    feature_names = data["feature_names"]

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
        model: RandomForestRegressor = RandomForestRegressor(random_state=random_state, n_estimators=20, max_depth=10)
    elif model_flag == "DT":
        model: DecisionTreeRegressor = DecisionTreeRegressor(random_state=random_state)
    elif model_flag == "GBT":
        model: GradientBoostingRegressor = GradientBoostingRegressor(max_depth=10, learning_rate=0.1, min_samples_leaf=5, n_estimators=100, max_features=1.0, random_state=random_state)
    else:
        model: XGBRegressor = XGBRegressor(random_state=random_state)

    model.fit(X_train, y_train)
    print("R^2 on test data", model.score(X_test, y_test))

    # run the experiment --------------------------------------------------------------------------

    run_main_experiment(
        model=model,
        x_explain=x_explain,
        y_true_label=y_true_label,
        explanation_id=explanation_id,
        max_interaction_order=MAX_INTERACTION_ORDER,
        n_features=n_features,
        feature_names=feature_names,
        dataset_name=dataset_name,
        background_dataset=X,
        observational=True,
        save_figures=SAVE_FIGURES,
        force_limits=force_limits,
        model_flag=model_flag
    )
