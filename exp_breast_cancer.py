"""This module is used to run the experiment on the breast cancer detection dataset."""
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from xgboost import XGBClassifier

from experiment_main import run_main_experiment


if __name__ == "__main__":

    dataset_name: str = "Breast Cancer"
    classification: bool = True
    random_state: int = 42

    max_interaction_order: int = 1
    explanation_index: int = 1

    save_figures: bool = False

    # load the breast cancer and pre-process -------------------------------------------------------
    dataset = load_breast_cancer()
    feature_names = list(dataset.feature_names)
    X = pd.DataFrame(dataset.data, columns=feature_names)
    y = pd.Series(dataset.target, name="target")

    n_features = X.shape[-1]
    n_samples = len(X)
    X = X.astype(float)

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

    model: XGBClassifier = XGBClassifier()
    model.fit(X_train, y_train)
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
        classification=classification,
        show_plots=True
    )
