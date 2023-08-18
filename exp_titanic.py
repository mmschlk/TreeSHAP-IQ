"""This module is used to run the experiment on the titanic sklearn dataset."""
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from experiment_main import run_main_experiment


if __name__ == "__main__":

    random_state = 42

    MAX_INTERACTION_ORDER = 3

    SAVE_FIGURES = False
    dataset_name: str = "Titanic"

    EXPLANATION_INDEX = 10
    explanation_direction = 0

    force_limits = None

    model_flag: str = "GBT"  # "XGB" or "RF", "DT", "GBT", None
    if model_flag is not None:
        print("Model:", model_flag)

    # load the titanic dataset from disc and pre-process -------------------------------------------
    data = pd.read_csv("data/titanic.csv")
    X = data.drop(columns=["Survived", "PassengerId", "Name"])
    y = data["Survived"]

    # ordinal encode categorical features
    cat_feature_names = [
        "Sex", "Ticket", "Cabin", "Embarked"
    ]
    X[cat_feature_names] = OrdinalEncoder().fit_transform(X[cat_feature_names])
    X[cat_feature_names] = SimpleImputer(strategy='most_frequent').fit_transform(X[cat_feature_names])
    num_feature_names = list(set(X.columns) - set(cat_feature_names))
    X[num_feature_names] = X[num_feature_names].apply(pd.to_numeric)
    X[num_feature_names] = SimpleImputer(strategy='median').fit_transform(X[num_feature_names])
    X = X.astype(float)

    # get the feature names
    feature_names = list(X.columns)
    n_features = X.shape[-1]
    n_samples = len(X)

    print("n_features", n_features, "n_samples", n_samples)

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

    # fit a tree model -----------------------------------------------------------------------------

    if model_flag == "RF":
        model: RandomForestClassifier = RandomForestClassifier(random_state=random_state, n_estimators=20, max_depth=10)
    elif model_flag == "DT":
        model: DecisionTreeClassifier = DecisionTreeClassifier(random_state=random_state, max_depth=10)
    elif model_flag == "GBT":
        model: GradientBoostingClassifier = GradientBoostingClassifier(random_state=random_state)
    else:
        model: XGBClassifier = XGBClassifier(random_state=random_state)
    model.fit(X_train, y_train)
    print("Accuracy on test data", model.score(X_test, y_test))

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
        sv_dim=explanation_direction,
        classification=True,
        output_type="probability",
        model_flag=model_flag,
        class_label=explanation_direction
    )
