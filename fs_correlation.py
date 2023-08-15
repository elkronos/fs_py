import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import kendalltau, pearsonr, spearmanr

class InvalidInputError(Exception):
    pass

def distance_correlation(X, Y):
    """ Compute the distance correlation function """
    n = len(X)
    a = np.outer(X, X) - np.mean(X)
    b = np.outer(Y, Y) - np.mean(Y)
    A = a - a.mean(axis=0)[None, :] - a.mean(axis=1)[:, None] + a.mean()
    B = b - b.mean(axis=0)[None, :] - b.mean(axis=1)[:, None] + b.mean()

    dcov2_xy = (A * B).sum() / float(n * n)
    dcov2_xx = (A * A).sum() / float(n * n)
    dcov2_yy = (B * B).sum() / float(n * n)
    dcor = np.sqrt(dcov2_xy) / np.sqrt(np.sqrt(dcov2_xx) * np.sqrt(dcov2_yy))
    
    return dcor

correlation_methods = {
    "pearson": pearsonr,
    "spearman": spearmanr,
    "kendall": kendalltau,
    "distance": distance_correlation
}

def compute_correlation(x, y, method):
    """ Compute correlation for data using specified method """
    if method in correlation_methods:
        return correlation_methods[method](x, y)[0]
    raise InvalidInputError(f"Invalid method {method}. Available methods are {list(correlation_methods.keys())}.")

def rank_by_correlation_magnitude(corr_matrix):
    """ Rank columns by average correlation magnitude """
    mean_corr = corr_matrix.abs().mean(axis=0)
    return mean_corr.sort_values(ascending=False)

def fs_correlation(data, threshold, method="pearson", plot_heatmap=False):
    """
    Advanced Correlation-based feature selection.
    """
    if not isinstance(data, pd.DataFrame):
        raise InvalidInputError("The 'data' argument must be a pandas DataFrame.")
    
    if not (0 <= threshold <= 1):
        raise InvalidInputError("The 'threshold' argument must be a value between 0 and 1.")
    
    if method not in ["pearson", "spearman", "kendall", "distance"]:
        raise InvalidInputError(f"Invalid correlation method. Specify one of {list(correlation_methods.keys())}.")

    # Use a copy instead of modifying the original data in place
    data_copy = data.dropna()

    # Compute correlation matrix
    corr_matrix = data_copy.corr(method=lambda x, y: compute_correlation(x, y, method))
    
    # Extract upper triangle of the correlation matrix without the diagonal
    upper_triangle = np.triu(corr_matrix, k=1)
    high_corr_pairs = np.argwhere(np.abs(upper_triangle) > threshold)
    
    # Handle high correlation pairs
    to_remove = set()
    mean_corr_values = rank_by_correlation_magnitude(corr_matrix)
    
    for pair in high_corr_pairs:
        # Select the feature with the highest average correlation magnitude
        if mean_corr_values[data_copy.columns[pair[0]]] > mean_corr_values[data_copy.columns[pair[1]]]:
            to_remove.add(data_copy.columns[pair[1]])
        else:
            to_remove.add(data_copy.columns[pair[0]])

    selected_vars = [col for col in data_copy.columns if col not in to_remove]

    # Plotting functionality
    if plot_heatmap:
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title("Correlation Matrix Heatmap")
        plt.show()

    return {
        "corr_matrix": corr_matrix, 
        "selected_vars": selected_vars,
        "ranked_features": rank_by_correlation_magnitude(corr_matrix)
    }

# Helper function to check multicollinearity
def check_multicollinearity(corr_matrix, threshold=0.8):
    """
    Check for multicollinearity in the features.
    Returns pairs of features that have a correlation greater than the threshold.
    """
    multicollinear_pairs = []
    upper_triangle = np.triu(corr_matrix, k=1)
    pairs = np.argwhere(np.abs(upper_triangle) > threshold)
    for pair in pairs:
        multicollinear_pairs.append((corr_matrix.columns[pair[0]], corr_matrix.columns[pair[1]]))
    return multicollinear_pairs

# Example usage
if __name__ == "__main__":
    from seaborn import load_dataset
    mtcars = load_dataset("mpg")
    
    result = fs_correlation(mtcars, 0.7, method="distance", plot_heatmap=True)
    print(result["corr_matrix"])
    print(result["selected_vars"])
    print(result["high_corr_pairs"])
    print(result["ranked_features"])

    print("\nMulticollinear Pairs:", check_multicollinearity(result["corr_matrix"], threshold=0.8))
