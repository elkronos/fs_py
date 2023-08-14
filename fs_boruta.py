import pandas as pd
import numpy as np
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from typing import Union, Optional, Dict, Any, List

def fs_boruta(data: pd.DataFrame, target_var: str, seed: Optional[int] = None, 
              maxRuns: int = 250, num_cores: int = 1, cutoff_features: Optional[int] = None, 
              cutoff_cor: float = 0.7, verbose: int = 2, 
              rf_params: Optional[Dict[str, Any]] = None, return_selector: bool = True,
              handle_nan: str = 'drop', remove_both: bool = False, 
              custom_imputer: Optional[SimpleImputer] = None) -> Dict[str, Union[list, BorutaPy, np.ndarray]]:
    """
    Feature selection using the Boruta algorithm.

    Parameters:
    data: DataFrame containing features and target variable.
    target_var: Name of the target variable column in the dataframe.
    seed: Random seed for reproducibility. Defaults to None.
    maxRuns: Maximum number of iterations for Boruta. Defaults to 250.
    num_cores: Number of cores to use for the Random Forest classifier. Defaults to 1.
    cutoff_features: Limit the number of top features to select. If None, all are considered.
    cutoff_cor: Correlation threshold for removing correlated features. Defaults to 0.7.
    verbose: Verbosity mode for Boruta (0: silent, 1: progress, 2: debug). Defaults to 2.
    rf_params: Additional parameters to be passed to the RandomForestClassifier. Defaults to None.
    return_selector: Whether to return the Boruta selector object. Default is True.
    handle_nan: Specifies how to handle NaN values ('drop', 'mean', 'median', 'mode'). Defaults to 'drop'.
    remove_both: If True, removes both correlated features. If False, just one. Defaults to False.
    custom_imputer: An optional sklearn SimpleImputer instance to handle missing values. Overrides handle_nan if provided.

    Returns:
    Dictionary containing:
    - 'selected_features': List of selected features by Boruta.
    - 'rejected_features': List of rejected features.
    - 'importance_scores': Feature importance scores if available.
    - 'boruta_selector': Boruta selector object (if return_selector=True).
    """
    
    # Ensure the target variable is in the dataframe
    if target_var not in data.columns:
        raise ValueError(f"'{target_var}' not found in the dataframe columns.")
    
    # Create a deep copy of the data to prevent modification of the original dataframe
    data = data.copy()
    
    # Handle missing values
    if custom_imputer:
        data = custom_imputer.transform(data)
    else:
        if handle_nan == 'drop':
            data.dropna(inplace=True)
        elif handle_nan in ['mean', 'median']:
            imputer = SimpleImputer(strategy=handle_nan)
            data = imputer.fit_transform(data)
        elif handle_nan == 'mode':
            for col in data.columns:
                data[col].fillna(data[col].mode().iloc[0], inplace=True)
        else:
            raise ValueError("Invalid handle_nan option.")
            
    if data.empty:
        raise ValueError("Dataframe is empty after handling NaN values.")
    
    # Separate features and target
    X_df = data.drop(columns=[target_var])
    y = data[target_var].values
    
    # Prepare RandomForest with default or provided parameters
    default_rf_params = {
        'n_jobs': num_cores,
        'class_weight': 'balanced',
        'max_depth': 5,
        'random_state': seed
    }
    if rf_params:
        default_rf_params.update(rf_params)

    rf = RandomForestClassifier(**default_rf_params)
    
    # Initialize and run Boruta
    boruta_selector = BorutaPy(rf, n_estimators='auto', verbose=verbose, random_state=seed, max_iter=maxRuns)
    boruta_selector.fit(X_df.values, y)
    
    # Extract selected and rejected features
    feature_names = X_df.columns.tolist()
    selected_features = np.array(feature_names)[boruta_selector.support_].tolist()
    rejected_features = [feat for feat in feature_names if feat not in selected_features]
    
    # Handle highly correlated features
    correlation_matrix = X_df[selected_features].corr(method='spearman').abs()
    upper_tri = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
    to_remove = [column for column in upper_tri.columns if any(upper_tri[column] > cutoff_cor)]
    
    if remove_both:
        correlated_pairs = upper_tri.stack().reset_index()
        correlated_pairs.columns = ['feature_1', 'feature_2', 'correlation']
        correlated_pairs = correlated_pairs[correlated_pairs.correlation > cutoff_cor]
        to_remove.extend(correlated_pairs['feature_1'].unique().tolist())
    
    # Remove the correlated features
    selected_features = [feat for feat in selected_features if feat not in to_remove]

    # Limit number of features if cutoff_features is provided
    if cutoff_features:
        feature_ranks = boruta_selector.ranking_
        ranked_features = sorted(selected_features, key=lambda x: feature_ranks[feature_names.index(x)])
        selected_features = ranked_features[:cutoff_features]
    
    # Extract feature importance if available
    feature_scores = boruta_selector.feature_importances_ if hasattr(boruta_selector, 'feature_importances_') else None

    # Prepare result dictionary
    result = {
        'selected_features': selected_features,
        'rejected_features': rejected_features,
        'importance_scores': feature_scores
    }
    
    if return_selector:
        result['boruta_selector'] = boruta_selector

    return result
