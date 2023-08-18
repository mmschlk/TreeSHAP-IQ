"""This module is used to run the experiment on the adult census dataset."""
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from xgboost import XGBClassifier

from experiment_main import run_main_experiment


if __name__ == "__main__":

    dataset_name: str = "Adult Census"
    classification: bool = True
    random_state: int = 42

    max_interaction_order: int = 2
    explanation_index: int = 1

    save_figures: bool = False

    model_flag: str = "GBT"  # "XGB" or "RF", "DT", "GBT", None
    if model_flag is not None:
        print("Model:", model_flag)

    # load the german credit risk dataset from disc and pre-process --------------------------------

    data = pd.read_csv("data/adult.csv")
    data = data.dropna()
    y = data["label"]
    data = data.drop(columns=["label"])

    num_feature_names = [
        'age', 'capital-gain', 'capital-loss', 'hours-per-week', 'fnlwgt'
    ]
    cat_feature_names = [
        'workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex',
        'native-country', 'education-num'
    ]
    data[num_feature_names] = data[num_feature_names].apply(pd.to_numeric)
    data[cat_feature_names] = OrdinalEncoder().fit_transform(data[cat_feature_names])
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

    if model_flag == "RF":
        model: RandomForestClassifier = RandomForestClassifier(random_state=random_state, n_estimators=20, max_depth=10)
    elif model_flag == "DT":
        model: DecisionTreeClassifier = DecisionTreeClassifier(random_state=random_state)
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
        max_interaction_order=max_interaction_order,
        n_features=n_features,
        feature_names=feature_names,
        dataset_name=dataset_name,
        background_dataset=X,
        observational=True,
        save_figures=save_figures,
        classification=classification,
        show_plots=True,
        model_flag=model_flag
    )
