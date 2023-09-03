import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

def perform_pca(data, num_pc=2, scale_data=True):
    """
    Perform Principal Component Analysis (PCA) on the provided data.

    Parameters:
    - data (pd.DataFrame): The data on which to perform PCA. It should contain numeric columns.
    - num_pc (int, optional): Number of principal components to extract. Defaults to 2.
    - scale_data (bool, optional): Whether to scale the data before performing PCA. Defaults to True.

    Returns:
    - dict: A dictionary containing:
        * pc_loadings: Loadings of the principal components.
        * pc_scores: Scores of the data on the principal components.
        * var_explained: Variance explained by each principal component.
        * pca_df: A dataframe with principal component scores alongside any non-numeric columns from the original data.
    """
    
    if data is None or data.shape[0] <= 1 or data.select_dtypes(include=[np.number]).shape[1] == 0:
        raise ValueError("Invalid input data for PCA. Ensure it has more than one row, contains numeric columns, and is not None.")
    
    if not isinstance(num_pc, int) or num_pc <= 0:
        raise ValueError("num_pc must be a positive integer.")
    
    max_num_pc = min(data.shape[1], data.shape[0])
    if num_pc > max_num_pc:
        raise ValueError(f"Number of principal components ({num_pc}) cannot exceed the number of features ({data.shape[1]}) or rows ({data.shape[0]}).")

    label_cols = data.select_dtypes(exclude=[np.number]).columns
    features = data.drop(columns=label_cols).copy()
    
    if scale_data:
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
    else:
        features_scaled = features.values
    
    pca = PCA(n_components=num_pc)
    pc_scores = pca.fit_transform(features_scaled)
    pc_scores_df = pd.DataFrame(data=pc_scores, columns=['PC' + str(i+1) for i in range(num_pc)])
    pca_df = pd.concat([pc_scores_df, data[label_cols]], axis=1)
    
    return {
        'pc_loadings': pca.components_,
        'pc_scores': pc_scores,
        'var_explained': pca.explained_variance_ratio_,
        'pca_df': pca_df
    }

def plot_pca(pca_result, label_col=None):
    """
    Plot the results of PCA.

    Parameters:
    - pca_result (dict): The result dictionary obtained from the perform_pca function.
    - label_col (str, optional): The name of the column to be used as a label for coloring points. If not provided or if the column doesn't exist, points won't be colored by any category.

    Returns:
    - None: Displays a scatter plot based on num_pc and prints variance explained by the principal components.
    """
    required_keys = ['pca_df', 'var_explained']
    for key in required_keys:
        if key not in pca_result:
            raise KeyError(f"Key '{key}' not found in pca_result dictionary. Make sure to pass a valid pca_result from the perform_pca function.")
    
    pca_df = pca_result['pca_df']
    var_explained = pca_result['var_explained']
    
    if len(var_explained) == 1:
        sns.histplot(data=pca_df, x='PC1', kde=True, color='blue').set(title='Distribution of PC1 Scores', xlabel=f"PC1 ({var_explained[0]*100:.2f}%)")
        sns.despine()
    else:
        if label_col and label_col in pca_df.columns:
            unique_labels = len(pca_df[label_col].unique())
            palette = sns.color_palette("hsv", unique_labels) if unique_labels > 8 else 'Set1'
            hue = label_col
        else:
            palette = None
            hue = None

        sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue=hue, palette=palette).set(title='PCA Plot', xlabel=f"PC1 ({var_explained[0]*100:.2f}%)", ylabel=f"PC2 ({var_explained[1]*100:.2f}%)")
        sns.despine()

    plt.show()
    
    for i, var in enumerate(var_explained):
        print(f"Variance explained by PC{i+1}: {var*100:.2f}%")

def scree_plot(pca_result):
    """
    Plot a scree plot of the variance explained by each principal component.

    Parameters:
    - pca_result (dict): The result dictionary obtained from the perform_pca function.

    Returns:
    - None: Displays the scree plot.
    """
    explained_variance = pca_result['var_explained']
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=range(1, len(explained_variance) + 1), y=explained_variance)
    plt.title("Scree Plot: Variance Explained by Each Component")
    plt.ylabel("Proportion of Variance Explained")
    plt.xlabel("Principal Component")
    sns.despine()
    plt.show()
