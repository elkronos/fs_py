import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV, Lasso
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from warnings import catch_warnings, simplefilter, warn
from sklearn.exceptions import ConvergenceWarning

def fs_lasso(X, y, alphas=None, alpha_lasso=None, nfolds=5, standardize=True, scaler_type="standard", custom_scaler=None, use_cv=True, max_iter=1000, feature_names=None, return_scaler=False, return_model=False, random_state=None):
    """
    Feature selection using Lasso or LassoCV regression.

    Parameters:
    - X (array-like or DataFrame): Feature matrix. Should not contain NaN values.
    - y (array-like): 1-dimensional target vector or single column DataFrame. Should not contain NaN values.
    - alphas (list of floats, optional): List of alpha values to try for LassoCV. If None and use_cv=True, LassoCV will determine the best values.
    - alpha_lasso (float, optional): Regularization strength for Lasso. Must be provided if use_cv=False.
    - nfolds (int, optional): Number of folds for cross-validation when use_cv=True. Default is 5.
    - standardize (bool, optional): Whether to standardize the input data using the specified scaler_type. Default is True. Ignored if a custom scaler is provided.
    - scaler_type (str, optional): Type of scaler to use when standardizing input data. Options are "standard" for StandardScaler and "minmax" for MinMaxScaler. Default is "standard". Ignored if custom_scaler is provided.
    - custom_scaler (scikit-learn scaler, optional): Custom scaler to standardize the input data. If provided, standardize and scaler_type are ignored.
    - use_cv (bool, optional): Whether to use cross-validated Lasso (LassoCV) or regular Lasso. Default is True.
    - max_iter (int, optional): Maximum number of iterations for the Lasso solver. Default is 1000.
    - feature_names (list of str, optional): List of feature names for the input data, useful if X is a numpy array. Default is None, and column indices are used.
    - return_scaler (bool, optional): Whether to return the scaler object. Default is False.
    - return_model (bool, optional): Whether to return the fitted Lasso model. Default is False.
    - random_state (int, optional): Random state for reproducibility when using LassoCV. Default is None.

    Returns:
    - DataFrame: A DataFrame containing the variable names and their importances.
    - Scaler (optional): If return_scaler=True, the function returns the fitted scaler.
    - Model (optional): If return_model=True, the function returns the fitted Lasso model.
      If both return_scaler and return_model are True, the order of the return tuple is (importance, scaler, model).

    Notes:
    Ensure that the assumptions for the Lasso regression model are met:
    - Linear relationship between predictors and target.
    - No multicollinearity among predictors.
    - Homoscedasticity: constant variance of errors.
    - Errors are independent.
    
    """
    
    # Handle missing data
    if X.isnull().any().any() or y.isnull().any():
        raise ValueError("X and y should not have missing values.")
    
    # Validate data types for X and y
    if not isinstance(X, (np.ndarray, pd.DataFrame)) or not isinstance(y, (np.ndarray, pd.Series, pd.DataFrame)):
        raise TypeError("X should be a DataFrame or array. y should be a Series, DataFrame, or array.")
    
    # Check y dimensionality
    if len(y.shape) > 1 and y.shape[1] > 1:
        raise ValueError("y should be 1-dimensional.")
    
    # If y is DataFrame, convert it to series
    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]
    
    # Only scale X, given the complications in rescaling y
    if standardize and not custom_scaler:
        if scaler_type == "standard":
            scaler_obj = StandardScaler()
        elif scaler_type == "minmax":
            scaler_obj = MinMaxScaler()
        else:
            raise ValueError("Invalid scaler_type. Options are 'standard' or 'minmax'.")
        X = scaler_obj.fit_transform(X)
    elif custom_scaler:
        scaler_obj = custom_scaler
        X = scaler_obj.fit_transform(X)

    with catch_warnings(record=True) as w:
        simplefilter("always", category=ConvergenceWarning)
        
        if use_cv:
            if alpha_lasso:
                warn("alpha_lasso will be ignored when using LassoCV.")
            model = LassoCV(alphas=alphas, cv=nfolds, max_iter=max_iter, random_state=random_state).fit(X, y)
        else:
            if alpha_lasso is None:
                raise ValueError("For non-CV Lasso, alpha_lasso must be specified.")
            model = Lasso(alpha=alpha_lasso, max_iter=max_iter).fit(X, y)
        
        # Log convergence warnings
        for warning in w:
            if issubclass(warning.category, ConvergenceWarning):
                warn(str(warning.message))

    # Get variable names
    if isinstance(X, pd.DataFrame):
        names = X.columns
    elif feature_names is not None:
        names = feature_names
    else:
        names = [f"X{i}" for i in range(X.shape[1])]

    # Create importance dataframe
    importance = pd.DataFrame({'Variable': names, 'Importance': model.coef_})
    importance = importance.sort_values(by='Importance', ascending=False)
    
    if return_scaler and return_model:
        return importance, scaler_obj, model
    elif return_scaler:
        return importance, scaler_obj
    elif return_model:
        return importance, model
    else:
        return importance
