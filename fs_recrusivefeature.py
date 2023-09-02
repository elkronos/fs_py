import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.model_selection import train_test_split, StratifiedKFold

def fs_recrusivefeature(data, response_var_column, estimator=RandomForestClassifier(), seed=123, n_splits=5, test_size=0.2):
    """
    Perform Recursive Feature Elimination with Cross-Validation (RFECV) on the provided data.
    
    Parameters:
    - data (pandas.DataFrame): The input data containing both features and the response variable.
    - response_var_column (str): The name of the column in `data` which serves as the response variable.
    - estimator (sklearn estimator, optional): The estimator to use for feature ranking. Defaults to RandomForestClassifier.
    - seed (int, optional): Random seed for reproducibility. Defaults to 123.
    - n_splits (int, optional): Number of splits for StratifiedKFold cross-validation. Defaults to 5.
    - test_size (float, optional): Proportion of the data to be used as test data. Should be between 0 and 1. Defaults to 0.2.

    Returns:
    - dict: A dictionary containing:
        * optimal_number_of_variables (int): Optimal number of features selected by RFECV.
        * optimal_variables (list): List of optimal feature names.
        * variable_importance (pandas.DataFrame): DataFrame of features and their importance scores.
        * resampling_results (list): Cross-validation scores for each number of features.
        * estimator (sklearn estimator): The trained estimator from the RFECV process.
    """
    
    if response_var_column not in data.columns:
        raise ValueError(f"The column '{response_var_column}' does not exist in the provided data.")

    if len(data.columns) == 1:
        raise ValueError("Data must contain features in addition to the response variable.")

    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")
    
    if not isinstance(n_splits, int) or n_splits < 2:
        raise ValueError("n_splits must be an integer greater than or equal to 2.")
    
    X = data.drop(response_var_column, axis=1)
    y = data[response_var_column]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
    
    cv = StratifiedKFold(n_splits=n_splits)
    rfe = RFECV(estimator=estimator, step=1, cv=cv, scoring="accuracy")
    rfe.fit(X_train, y_train)
    
    try:
        feature_importances = rfe.estimator_.feature_importances_
    except AttributeError:
        feature_importances = ["Not available"] * len(X.columns)
    
    var_importance = pd.DataFrame({
        'variable': X.columns,
        'importance': feature_importances
    }).sort_values(by="importance", ascending=False)
    
    return {
        'optimal_number_of_variables': rfe.n_features_,
        'optimal_variables': list(X.columns[rfe.support_]),
        'variable_importance': var_importance,
        'resampling_results': rfe.grid_scores_,
        'estimator': rfe.estimator_
    }