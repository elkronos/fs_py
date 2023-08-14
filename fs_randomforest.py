# Standard Library
import os
import logging
from typing import List, Dict, Tuple, Union, Any

# Third-party Libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (confusion_matrix, accuracy_score, f1_score, 
                             precision_score, recall_score, r2_score, 
                             mean_squared_error, mean_absolute_error)
from imblearn.over_sampling import SMOTE
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import (train_test_split, GridSearchCV, 
                                     RandomizedSearchCV, KFold)
from joblib import dump, load

# Setting up logging
logging.basicConfig(level=logging.INFO)

def scale_features(X_train, X_test):
    """Scale features using StandardScaler."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled


def handle_categorical_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Handle categorical features by encoding them with LabelEncoder.
    
    Args:
        data (pd.DataFrame): Input data frame with potential categorical features.
    
    Returns:
        pd.DataFrame: Transformed data frame with encoded categorical features.
    """
    for col in data.columns:
        if data[col].dtype == 'object':
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col])
    return data


def balance_data(data: pd.DataFrame, target: str, balance_method: str, seed: int) -> pd.DataFrame:
    """
    Balance the dataset using various methods such as SMOTE, undersampling, or oversampling.
    
    Args:
        data (pd.DataFrame): Input data frame.
        target (str): Target column name.
        balance_method (str): Method to use for balancing ('SMOTE', 'under', 'over').
        seed (int): Random seed for reproducibility.
        
    Returns:
        pd.DataFrame: Balanced data frame.
    """
    if balance_method == 'SMOTE':
        sm = SMOTE(random_state=seed)
        X, y = sm.fit_resample(data.drop(columns=target), data[target])
        return pd.concat([X, y], axis=1)
    elif balance_method in ['under', 'over']:
        class_counts = data[target].value_counts()
        n_samples = class_counts.min() if balance_method == 'under' else class_counts.max()
        temp_data = []
        for _class in class_counts.index:
            _data = data[data[target] == _class]
            _data_resampled = resample(_data, replace=(balance_method == 'over'),
                                       n_samples=n_samples, random_state=seed)
            temp_data.append(_data_resampled)
        return pd.concat(temp_data)
    else:
        raise ValueError(f"Unknown balance method: {balance_method}")


def hyperparam_tune(rf, X_train, y_train, hyperparam_tuning, k_fold):
    """
    Hyperparameter tuning for the RandomForest model.
    
    Args:
        rf: Initialized RandomForest model.
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        hyperparam_tuning (Dict): Dictionary specifying type of tuning ('grid' or 'random') and the corresponding parameters.
        k_fold (int): Number of cross-validation folds.
        
    Returns:
        Tuple: Best model and its parameters.
    """
    if hyperparam_tuning['type'] == 'grid':
        search = GridSearchCV(rf, hyperparam_tuning['param_grid'], cv=k_fold or 3)
    elif hyperparam_tuning['type'] == 'random':
        search = RandomizedSearchCV(rf, hyperparam_tuning['param_distributions'], cv=k_fold or 3)
    else:
        raise ValueError(f"Unknown hyperparameter tuning type: {hyperparam_tuning['type']}")
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_


def compute_metrics(y_test, predictions, task_type: str, average_type: str):
    """
    Compute metrics based on the task type (classification or regression).
    
    Args:
        y_test (pd.Series): True target values.
        predictions (pd.Series): Predicted values.
        task_type (str): 'classification' or 'regression'.
        average_type (str): Average type for multi-class classification ('binary', 'micro', 'macro', 'weighted').
    
    Returns:
        Dict: Computed metrics.
    """
    metrics = {}
    try:
        if task_type == "classification":
            metrics = {
                "accuracy": accuracy_score(y_test, predictions),
                "f1": f1_score(y_test, predictions, average=average_type),
                "precision": precision_score(y_test, predictions, average=average_type),
                "recall": recall_score(y_test, predictions, average=average_type)
            }
        else:
            metrics = {
                "r2": r2_score(y_test, predictions),
                "mse": mean_squared_error(y_test, predictions),
                "mae": mean_absolute_error(y_test, predictions)
            }
    except ValueError as ve:
        logging.error(f"ValueError encountered in compute_metrics: {ve}")
    return metrics

def prepare_data(data, features, target, balance_method, task_type, seed):
    """Prepare data by handling categorical features and balancing, if necessary."""
    if features:
        data = data[features + [target]]
    
    data = handle_categorical_features(data)

    if balance_method:
        if task_type == 'regression':
            raise ValueError("Balancing techniques are not appropriate for regression tasks.")
        data = balance_data(data, target, balance_method, seed)
    return data

def load_hyperparams_from_config(hyperparam_config_file):
    """Load hyperparameters from a configuration file."""
    import json
    with open(hyperparam_config_file, 'r') as f:
        config = json.load(f)
    return config

def train_and_evaluate_rf(data, target, rf_params, n_estimators, task_type, stratify_option, split_ratio, seed, k_fold, hyperparam_tuning, scale_data, n_jobs, average_type, save_model_path=None):
    """Train the random forest model and evaluate its performance."""
    unique_classes = data[target].nunique()
    if task_type == "classification" and unique_classes > 2 and average_type == 'binary':
        logging.warning("Binary averaging type is not suitable for multiclass classification. Switching to 'macro'.")
        average_type = 'macro'

    stratify = data[target] if (task_type == "classification" and stratify_option and unique_classes > 1) else None

    X_train, X_test, y_train, y_test = train_test_split(
        data.drop(columns=target), data[target], test_size=1-split_ratio, stratify=stratify, random_state=seed
    )

    if scale_data:
        X_train, X_test = scale_features(X_train, X_test)

    model_class = RandomForestClassifier if task_type == "classification" else RandomForestRegressor
    rf = model_class(n_estimators=n_estimators, n_jobs=n_jobs, random_state=seed, **rf_params)

    if hyperparam_tuning:
        rf, _ = hyperparam_tune(rf, X_train, y_train, hyperparam_tuning, k_fold)
    
    if k_fold:
        kf = KFold(n_splits=k_fold)
        for train_index, _ in kf.split(X_train):
            rf.fit(X_train.iloc[train_index], y_train.iloc[train_index])
    else:
        rf.fit(X_train, y_train)

    predictions = rf.predict(X_test)

    results = compute_metrics(y_test, predictions, task_type, average_type)
    
    results.update({
        'model': rf,
        'predictions': predictions,
        'feature_importances': rf.feature_importances_
    })

    if task_type == "classification":
        results['confusion_matrix'] = confusion_matrix(y_test, predictions).tolist()
    elif task_type == "regression":
        residuals = y_test - predictions
        results['residuals'] = residuals.tolist()

    if save_model_path:
        dump(rf, save_model_path)
        results['saved_model_path'] = save_model_path

    return results

def fs_randomforest(data: pd.DataFrame, target: str, features: List[str] = None, 
                    balance_method=None, task_type=None, seed=None, 
                    stratify_option=None, split_ratio=0.8, 
                    hyperparam_config_file: str = None,
                    load_model_path: str = None, custom_metrics: Dict[str, Any] = None, 
                    rf_params: Dict = {}, n_estimators: int = 100, 
                    k_fold: Union[int, None] = None, hyperparam_tuning: Dict = None, 
                    scale_data: bool = True, n_jobs: int = -1, 
                    average_type: str = 'binary', save_model_path: str = None) -> Dict:
    """
    Main function to prepare data, train, and evaluate a random forest model.

    Note: Added the missing parameters to the function signature.
    """
    if data is None or target is None:
        raise ValueError("Both data and target must be provided.")

    if target not in data.columns:
        raise ValueError(f"Target '{target}' not found in provided dataframe columns.")

    if load_model_path:
        if not os.path.exists(load_model_path):
            raise ValueError(f"Model path {load_model_path} does not exist.")
        rf = load(load_model_path)
        return {'model': rf}

    data = prepare_data(data, features, target, balance_method, task_type, seed)

    if hyperparam_config_file:
        config = load_hyperparams_from_config(hyperparam_config_file)
        # Update parameters based on the loaded configuration
        n_estimators = config.get('n_estimators', n_estimators)
        # Additional hyperparameters can be updated similarly

    results = train_and_evaluate_rf(data, target, rf_params, n_estimators, task_type, stratify_option, split_ratio, seed, k_fold, hyperparam_tuning, scale_data, n_jobs, average_type, save_model_path)

    # Custom Metrics
    if custom_metrics:
        for metric_name, metric_func in custom_metrics.items():
            results[metric_name] = metric_func(y_test, predictions)

    return results