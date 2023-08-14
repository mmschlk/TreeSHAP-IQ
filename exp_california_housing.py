"""This module is used to run the experiment on the california-housing dataset."""
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import GradientBoostingRegressor

from experiment_main import run_main_experiment


if __name__ == "__main__":

    # Settings used for the force plot in the paper
    # RANDOM_STATE = 42
    # MAX_INTERACTION_ORDER = 2,3
    # EXPLANATION_INDEX = 2
    # force_limits = (0.4, 6.8)
    # model = GradientBoostingRegressor(max_depth=10, learning_rate=0.1, min_samples_leaf=5, n_estimators=100, max_features=1.0)

    RANDOM_STATE = 42
    MAX_INTERACTION_ORDER = 3
    EXPLANATION_INDEX = 2
    SAVE_FIGURES = True
    dataset_name: str = "California"

    force_limits = (0.4, 6.8)

    # load the california housing dataset and pre-process ------------------------------------------
    data = fetch_california_housing(as_frame=True)
    X, y = data.data, data.target
    n_features = X.shape[-1]
    n_samples = len(X)
    feature_names = data["feature_names"]

    # train test split and get explanation datapoint -----------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.7, shuffle=True, random_state=RANDOM_STATE)
    explanation_id = X_test.index[EXPLANATION_INDEX]

    # get explanation datapoint and index
    x_explain = np.asarray(X_test.iloc[EXPLANATION_INDEX].values)
    y_true_label = y_test.iloc[EXPLANATION_INDEX]

    # transform data to numpy arrays
    X, X_train, X_test, y_train, y_test = (
        X.values, X_train.values, X_test.values, y_train.values, y_test.values
    )

    # fit a tree model -----------------------------------------------------------------------------

    model = GradientBoostingRegressor(
        max_depth=10, learning_rate=0.1, min_samples_leaf=5, n_estimators=100, max_features=1.0,
        random_state=RANDOM_STATE
    )
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
        force_limits=force_limits
    )
