"""This module is used to run the experiment on the titanic sklearn dataset."""
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder

from experiment_main import run_main_experiment


if __name__ == "__main__":

    RANDOM_STATE = 42
    MAX_INTERACTION_ORDER = 2
    EXPLANATION_INDEX = 1
    SAVE_FIGURES = True
    dataset_name: str = "Titanic"

    force_limits = None

    # load the titanic dataset from disc and pre-process -------------------------------------------
    data = pd.read_csv("data/titanic.csv")
    X = data.drop(columns=["Survived"])
    y = data["Survived"]

    # drop rows with missing values
    X = X.dropna()
    y = y[X.index]

    # ordinal encode categorical features
    cat_feature_names = [
        "Sex", "Ticket", "Cabin", "Embarked"
    ]
    X[cat_feature_names] = OrdinalEncoder().fit_transform(X[cat_feature_names])

    # drop PassengerId, Name
    X = X.drop(columns=["PassengerId", "Name"])

    # convert to float
    X = X.astype(float)

    # get the feature names
    feature_names = list(X.columns)
    n_features = X.shape[-1]
    n_samples = len(X)

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
    model = RandomForestClassifier(random_state=RANDOM_STATE)
   # model = DecisionTreeClassifier(random_state=RANDOM_STATE)
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
        force_limits=force_limits,
        sv_dim=1,
        classification=True,
        output_type="probability",
    )
