import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC, SVR
from sklearn.metrics import confusion_matrix, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from typing import Union

# Preprocess data
def split_data(data: pd.DataFrame, target: str, task: str, test_size: float = 0.3, 
               stratify: str = "auto", seed: int = None) -> tuple:
    """
    Splits the dataframe into train and test datasets based on the target column.

    Parameters:
    - data (pd.DataFrame): Input dataframe.
    - target (str): Name of the target column.
    - task (str): Either 'classification' or 'regression'.
    - test_size (float): Fraction of data to be used for testing.
    - stratify (str): Controls stratification. 'auto' for automatic stratification.
    - seed (int): Seed for random state to ensure reproducibility.

    Returns:
    - tuple: Train-test split datasets (X_train, X_test, y_train, y_test).
    """
    
    if target not in data.columns:
        raise ValueError(f"Target column '{target}' not found in DataFrame.")

    X = data.drop(columns=[target])
    y = data[target]

    if stratify == "auto":
        stratify = None  # Default as None
        if task == "classification":
            unique_classes = y.unique()
            if len(unique_classes) == 2:  # binary classification
                stratify = y
            elif len(unique_classes) > 2:
                # Checking class distribution
                class_counts = y.value_counts(normalize=True)
                if all(class_counts > 0.1):
                    stratify = y
                # stratify remains None for classes with < 10% representation

    return train_test_split(X, y, test_size=test_size, stratify=stratify, random_state=seed)

# Tune grid
def get_tuning_grid(task: str, model_type: str = 'svm') -> dict:
    """
    Provides hyperparameters grid based on task and model type.

    Parameters:
    - task (str): Either 'classification' or 'regression'.
    - model_type (str): Model type.

    Returns:
    - dict: Hyperparameters tuning grid.
    """
    
    if task == "classification":
        return {
            'svm__C': np.logspace(-3, 3, 7),
            'svm__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'svm__degree': [2, 3, 4],  # for poly kernel
            'svm__gamma': ['scale', 'auto']
        }
    else:  # regression
        return {
            'svm__C': np.logspace(-3, 3, 7),
            'svm__epsilon': np.linspace(0.01, 1, 10),
            'svm__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],  # Added 'sigmoid'
            'svm__degree': [2, 3, 4],
            'svm__gamma': ['scale', 'auto']
        }

# Performance metrics
def compute_performance(y_test: pd.Series, predictions: pd.Series, task: str) -> Union[np.ndarray, float]:
    """
    Computes performance metrics based on task type.

    Parameters:
    - y_test (pd.Series): True target values.
    - predictions (pd.Series): Predicted values.
    - task (str): Either 'classification' or 'regression'.

    Returns:
    - Union[np.ndarray, float]: Confusion matrix (classification) or R2 score (regression).
    """

    if task == "classification":
        return confusion_matrix(y_test, predictions)
    else:
        return r2_score(y_test, predictions)

# Primary SVM function
def fs_svm(data: pd.DataFrame, target: str, task: str, nfolds: int = 5, 
           tune_grid: dict = None, seed: int = None, preprocessor=None, 
           scoring: str = "default", model_type: str = 'svm', 
           model_params: dict = None) -> dict:
    """
    Perform feature selection using SVM and return results.

    Parameters:
    - data (pd.DataFrame): Input dataframe.
    - target (str): Target column name.
    - task (str): Either 'classification' or 'regression'.
    - nfolds (int): Number of cross-validation folds.
    - tune_grid (dict): Custom hyperparameters tuning grid.
    - seed (int): Seed for reproducibility.
    - preprocessor: Data preprocessor (default is StandardScaler).
    - scoring (str): Scoring metric (default is "accuracy" for classification or "r2" for regression).
    - model_type (str): Model type.
    - model_params (dict): Additional model parameters.

    Returns:
    - dict: Contains best model, predictions, performance metrics, etc.
    """
    
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Data must be a DataFrame.")
    
    if target not in data.columns:
        raise ValueError(f"Target column '{target}' not found in the data DataFrame.")

    if scoring == "default":
        if task == "classification":
            scoring = "accuracy"
        else:
            scoring = "r2"

    if preprocessor and not (hasattr(preprocessor, 'fit') and hasattr(preprocessor, 'transform')):
        raise ValueError("Provided preprocessor must have both 'fit' and 'transform' methods.")

    if model_type != 'svm':
        raise ValueError("Currently only SVM model type is supported.")

    X_train, X_test, y_train, y_test = split_data(data, target, task, seed=seed)

    if not preprocessor:
        preprocessor = StandardScaler()

    if task == "classification":
        try:
            model = SVC(**(model_params or {}))
        except TypeError as e:
            raise ValueError(f"Invalid parameters provided for SVC: {e}")
    else:
        try:
            model = SVR(**(model_params or {}))
        except TypeError as e:
            raise ValueError(f"Invalid parameters provided for SVR: {e}")

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('svm', model)
    ])

    if not tune_grid:
        tune_grid = get_tuning_grid(task)

    grid_search = GridSearchCV(pipeline, tune_grid, cv=nfolds, scoring=scoring, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    predictions = grid_search.predict(X_test)

    performance = compute_performance(y_test, predictions, task)

    return {
        'best_model': grid_search.best_estimator_,
        'predictions': predictions,
        'performance': performance,
        'best_params': grid_search.best_params_,
        'score': grid_search.best_score_
    }
