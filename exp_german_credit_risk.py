"""This module is used to run the experiment on the german-credit-risk dataset."""
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import OrdinalEncoder

from experiment_main import run_main_experiment


if __name__ == "__main__":

    RANDOM_STATE = 42

    MAX_INTERACTION_ORDER = 2
    EXPLANATION_INDEX = 1

    SAVE_FIGURES = False
    dataset_name: str = "German Credit"

    # load the german credit risk dataset from disc and pre-process --------------------------------
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

    model = GradientBoostingClassifier(
        max_depth=3, learning_rate=0.1, min_samples_leaf=5, n_estimators=10, max_features=1.0,
        #max_depth=5, learning_rate=0.1, min_samples_leaf=5, n_estimators=100, max_features=1.0,
        random_state=RANDOM_STATE
    )
    model.fit(X_train, y_train)
    print("Accuracy", model.score(X_test, y_test))

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
    )
