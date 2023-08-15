"""This module is used to run the experiment on the german-credit-risk dataset."""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBClassifier

from experiment_main import run_main_experiment


if __name__ == "__main__":

    # settings for the network plot image in the paper
    # random_state = 42
    # max_interaction_order = 2
    # explanation_index = 1
    # from sklearn.ensemble import GradientBoostingClassifier
    # model = GradientBoostingClassifier(
    #   max_depth=5, learning_rate=0.1, min_samples_leaf=5, n_estimators=100, max_features=1.0,
    #   random_state=random_state
    # )

    # settings for the n-SII plot in the experiments section
    # random_state = 42
    # max_interaction_order = 7
    # explanation_index = 1
    # model = XGBClassifier(random_state=random_state)

    dataset_name: str = "German Credit"
    classification: bool = True
    random_state: int = 42

    max_interaction_order: int = 2
    explanation_index: int = 1

    save_figures: bool = True

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

    model = GradientBoostingClassifier(
       max_depth=5, learning_rate=0.1, min_samples_leaf=5, n_estimators=100, max_features=1.0,
       random_state=random_state
    )
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
