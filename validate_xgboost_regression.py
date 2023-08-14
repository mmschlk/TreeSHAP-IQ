"""This module is used to run the experiment on the german-credit-risk dataset."""
import numpy as np
from sklearn.datasets import fetch_california_housing

from sklearn.model_selection import train_test_split
import xgboost as xgb

from experiment_main import run_main_experiment


if __name__ == "__main__":
    RANDOM_STATE = 42
    max_interaction_order = 3
    EXPLANATION_INDEX = 2
    save_figures = False
    dataset_name: str = "California"

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

    # get a xgboost model
    model: xgb.XGBClassifier = xgb.XGBRegressor()

    # fit the model
    model.fit(X_train, y_train)

    # print the accuracy
    print("Accuracy on test data", model.score(X_test, y_test))

    # run the experiment --------------------------------------------------------------------------

    run_main_experiment(
        model=model,
        x_explain=x_explain,
        y_true_label=y_true_label,
        explanation_id=explanation_id,
        max_interaction_order=max_interaction_order,
        n_features=n_features,
        feature_names=feature_names,
        dataset_name=dataset_name,
        background_dataset=X,
        observational=True,
        save_figures=save_figures,
        classification=False,
        show_plots=True
    )
