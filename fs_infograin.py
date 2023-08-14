import pandas as pd
import numpy as np

def entropy(x):
    if len(x) == 0:  # Handle empty array
        return 0
    values, freq = np.unique(x, return_counts=True)
    prob = freq / len(x)
    return -np.sum(prob * np.log2(prob))

def calculate_bins(x):
    n = len(x)
    if n == 0:  # Handle empty array
        return 1
    iqr_x = np.subtract(*np.percentile(x, [75, 25]))
    iqr_x = 1 if iqr_x == 0 else iqr_x  # Ensure non-zero IQR
    fd_bin_width = 2 * iqr_x / (n**(1/3))  # Freedman-Diaconis rule
    sturges_bins = np.ceil(np.log2(n) + 1)    # Sturges' rule
    bins = np.ceil((x.max() - x.min()) / fd_bin_width) if ((x.max() - x.min()) / fd_bin_width) > 1 else sturges_bins
    return int(bins)

def fs_infogain(df, target):
    df = df.copy()  # Deep copy to avoid side effects

    if target not in df.columns:
        raise ValueError("The target variable is not found in the provided dataframe.")
    
    date_cols = df.select_dtypes(include=[np.datetime64]).columns
    for col in date_cols:
        df = df.assign(**{
            f"{col}_year": df[col].dt.year,
            f"{col}_month": df[col].dt.month,
            f"{col}_day": df[col].dt.day
        })
    df.drop(columns=date_cols, inplace=True)  # Drop original date columns

    info_gain = []
    column_names = df.columns
    entropy_target = entropy(df[target])
    total_rows = len(df)
    
    for col in column_names:
        if col != target:
            if np.issubdtype(df[col].dtype, np.number):
                n_bins = calculate_bins(df[col])
                df[col], _ = pd.cut(df[col], bins=n_bins, labels=False, retbins=True, include_lowest=True)
            
            entropy_attribute = sum(
                len(subset) / total_rows * entropy(subset[target])
                for _, subset in df.groupby(col)
            )
            gain = entropy_target - entropy_attribute
            info_gain.append(gain)
    
    result = pd.DataFrame({
        "Variable": [col for col in column_names if col != target],
        "InfoGain": info_gain
    })
    
    return result

def calculate_information_gain_multiple(dfs_list):
    results_list = []
    for i, df in enumerate(dfs_list, 1):
        result = fs_infogain(df, "target")
        result["Origin"] = f"Data_Frame_{i}"
        results_list.append(result)
    all_results = pd.concat(results_list, axis=0, ignore_index=True)
    return all_results