"""This module is used to run the experiment on the bike sharing dataset."""
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.datasets import fetch_openml

from xgboost import XGBRegressor

from experiment_main import run_main_experiment


if __name__ == "__main__":

    dataset_name: str = "Bike"
    classification: bool = False
    random_state: int = 42

    max_interaction_order: int = 4
    explanation_index: int = 0

    save_figures: bool = False

    # load the bike sharing dataset from openml and pre-process ------------------------------------

    data, y = fetch_openml(data_id=42713, return_X_y=True, as_frame=True)
    feature_names = list(data.columns)
    num_feature_names = ['hour', 'temp', 'feel_temp', 'humidity', 'windspeed']
    cat_feature_names = ['season', 'year', 'month', 'holiday', 'weekday', 'workingday', 'weather']
    data[num_feature_names] = data[num_feature_names].apply(pd.to_numeric)
    data[cat_feature_names] = OrdinalEncoder().fit_transform(data[cat_feature_names])
    data = pd.DataFrame(data, columns=feature_names)
    data.dropna(inplace=True)

    X = data
    n_features = X.shape[-1]
    n_samples = len(X)
    feature_names = list(X.columns)

    # train test split and get explanation datapoint -----------------------------------------------

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

    # fit a tree model -----------------------------------------------------------------------------

    model: XGBRegressor = XGBRegressor(random_state=random_state)
    model.fit(X_train, y_train)
    print("R^2 Score on test Data", model.score(X_test, y_test))

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
        classification=classification,
        show_plots=True
    )
