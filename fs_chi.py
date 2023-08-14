import pandas as pd
from scipy.stats import chi2_contingency
import numpy as np
import warnings

def compute_chi2(data, feature, target_col, correct, min_freq=5):
    """Helper function to compute chi-square."""
    cont_table = pd.crosstab(data[feature], data[target_col])
        
    if cont_table.isna().any().any():
        warnings.warn(f"Missing values detected in feature {feature}. Setting p-value as NaN.")
        return np.nan, None, None
        
    if cont_table.min().min() < min_freq:
        warnings.warn(f"Frequency less than {min_freq} detected in the contingency table for feature {feature}. Setting p-value as NaN.")
        return np.nan, None, None
        
    chi2, p, _, expected = chi2_contingency(cont_table, correction=correct)
    return p, chi2, expected

def fs_chi(data, target_col, sig_level=0.05, correct=True, apply_bonferroni=True, min_freq=5):
    """
    Perform feature selection using chi-square test.
    ...

    Parameters:
    ...
    min_freq: int, optional
        Minimum frequency count for cells in the contingency table. Defaults to 5.

    Returns:
    A dictionary containing:
    - `significant_features`: A list of the statistically significant categorical feature names.
    - `p_values`: A dictionary indicating the p-values from the chi-square tests for each feature.
    - `chi2_stats`: A dictionary indicating the chi2 statistic for each feature.
    - `expected_freqs`: A dictionary indicating the expected frequencies for each feature.
    """

    if not isinstance(data, pd.DataFrame):
        raise ValueError("data must be a pandas DataFrame.")
    if target_col not in data.columns:
        raise ValueError("target_col must be a column in data.")
    if not data[target_col].dtype == pd.CategoricalDtype():
        raise ValueError("target_col must be of categorical type.")

    # Convert columns with object datatype to category
    data = data.apply(lambda col: col.astype('category') if col.dtype == 'O' else col)
    
    feature_cols = [col for col in data.columns if col != target_col and data[col].dtype == pd.CategoricalDtype()]

    chi_square_results = {}
    chi2_stats = {}
    expected_freqs = {}

    for feature in feature_cols:
        p_val, chi2_stat, expected_freq = compute_chi2(data, feature, target_col, correct, min_freq=min_freq)
        chi_square_results[feature] = p_val
        chi2_stats[feature] = chi2_stat
        expected_freqs[feature] = expected_freq

    adj_sig_level = sig_level / len(feature_cols) if apply_bonferroni else sig_level
    sig_features = [feature for feature, p in chi_square_results.items() if not np.isnan(p) and p < adj_sig_level]
    
    return {
        "significant_features": sig_features, 
        "p_values": chi_square_results,
        "chi2_stats": chi2_stats,
        "expected_freqs": expected_freqs
    }
