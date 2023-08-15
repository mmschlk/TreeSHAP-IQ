"""This module is used to run the experiment on the bank-marketing dataset."""
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.datasets import fetch_openml

from xgboost import XGBClassifier

from experiment_main import run_main_experiment


OPEN_ML_BANK_MARKETING_RENAME_MAPPER = {
    'V1': 'age',
    'V2': 'job',
    'V3': 'marital',
    'V4': 'education',
    'V5': 'default',
    'V6': 'balance',
    'V7': 'housing',
    'V8': 'loan',
    'V9': 'contact',
    'V10': 'day',
    'V11': 'month',
    'V12': 'duration',
    'V13': 'campaign',
    'V14': 'pdays',
    'V15': 'previous',
    'V16': 'poutcome'
}


if __name__ == "__main__":

    dataset_name: str = "Bank"
    classification: bool = True
    random_state: int = 42

    max_interaction_order: int = 2
    explanation_index: int = 1

    save_figures: bool = False

    # load the bank-marketing dataset from openml and pre-process ----------------------------------

    data, y = fetch_openml(data_id=1461, return_X_y=True, as_frame=True)

    # transform y from '1' and '2' to 0 and 1
    y = y.apply(lambda x: 1 if x == '2' else 0)

    data = data.rename(columns=OPEN_ML_BANK_MARKETING_RENAME_MAPPER)
    num_feature_names = ['age', 'balance', 'duration', 'pdays', 'previous']
    cat_feature_names = [
        'job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'day', 'month',
        'campaign', 'poutcome'
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
