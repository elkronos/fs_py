import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split, HalvingGridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PCA
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from statsmodels.formula.api import dmatrices
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Configuration constants
RANDOM_STATE = 42
TEST_SIZE = 0.2
Z_THRESHOLD = 3
VIF_THRESHOLD = 10

def remove_outliers(data, z_threshold=Z_THRESHOLD):
    """Removes outliers based on z-scores.
    
    Parameters:
        data (pd.DataFrame): The data to be processed.
        z_threshold (float): The z-score threshold.

    Returns:
        pd.DataFrame: Data without outliers.
    """
    z_scores = np.abs(stats.zscore(data.select_dtypes(include=[np.number])))
    return data[(z_scores < z_threshold).all(axis=1)]

def prepare_data(data, formula):
    """Prepares data by separating predictors and response.
    
    Parameters:
        data (pd.DataFrame): Input data.
        formula (str): Patsy formula for constructing design matrices.

    Returns:
        pd.DataFrame, pd.Series: Processed predictors and response.
    """
    y, X = dmatrices(formula, data=data, return_type='dataframe')
    imputer = SimpleImputer(strategy='mean')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    return X_imputed, y

def apply_scaling(X_train, X_test, scale_method):
    """Applies specified scaling to the data.
    
    Parameters:
        X_train (pd.DataFrame): Training data.
        X_test (pd.DataFrame): Test data.
        scale_method (str): Scaling method ('standard' or 'minmax').

    Returns:
        np.array, np.array, Scaler: Scaled training and test data, and the scaler used.
    """
    if scale_method == 'standard':
        scaler = StandardScaler()
    elif scale_method == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Invalid scaling method: {scale_method}")

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, scaler

def fs_elastic(data, formula, l1_ratios=None, cv=5, scale_method='standard', alphas=None, visualize=False):
    """Feature selection using ElasticNet.
    
    Parameters:
        data (pd.DataFrame): The input data.
        formula (str): Patsy formula for constructing design matrices.
        l1_ratios (list): List of l1_ratios.
        cv (int): Cross-validation folds.
        scale_method (str): Scaling method ('standard' or 'minmax').
        alphas (list): List of alpha values.
        visualize (bool): Whether to visualize the regularization path.

    Returns:
        dict: Results including best model and parameters.
    """
    X, y = prepare_data(data, formula)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    
    # Handle outliers and high collinearity on training set only
    X_train = remove_outliers(X_train)
    high_vif_features = get_high_vif_features(X_train)
    if high_vif_features:
        print(f"Dropping high VIF features: {high_vif_features}")
    X_train.drop(high_vif_features, axis=1, inplace=True)
    
    X_train, X_test, scaler = apply_scaling(X_train, X_test, scale_method)
    
    pipeline = Pipeline([
        ('elasticnet', ElasticNet(tol=0.01, max_iter=10000, random_state=RANDOM_STATE))
    ])
    
    if not alphas:
        alphas = np.logspace(-5, 2, 50)

    if not l1_ratios:
        l1_ratios = np.linspace(0.1, 1.0, 10)

    param_grid = {'elasticnet__alpha': alphas, 'elasticnet__l1_ratio': l1_ratios}
    search = HalvingGridSearchCV(pipeline, param_grid, cv=cv, n_jobs=-1, verbose=1, random_state=RANDOM_STATE, scoring='neg_mean_squared_error')
    search.fit(X_train, y_train)

    if visualize:
        plot_regularization_path(l1_ratios, alphas, X_train, y_train, X.columns)
    
    coef = search.best_estimator_.named_steps['elasticnet'].coef_
    plot_feature_importance(coef, X.columns)

    return {
        'model': search,
        'best_estimator': search.best_estimator_,
        'coef': coef,
        'best_alpha': search.best_params_['elasticnet__alpha'],
        'best_l1_ratio': search.best_params_['elasticnet__l1_ratio']
    }

def plot_regularization_path(l1_ratios, alphas, X_train, y_train, feature_names):
    """Plots the regularization path for ElasticNet.
    
    Parameters:
        l1_ratios (list): List of l1_ratios.
        alphas (list): List of alpha values.
        X_train (np.array): Training data.
        y_train (np.array): Training response.
        feature_names (list): Names of the features.

    Returns:
        None: Displays a plot.
    """
    alphas_matrix, l1_ratios_matrix = np.meshgrid(alphas, l1_ratios)
    elastic = ElasticNet(max_iter=10000)
    coefs = np.apply_along_axis(lambda args: elastic.set_params(alpha=args[0], l1_ratio=args[1]).fit(X_train, y_train).coef_,
                                0, np.stack([alphas_matrix, l1_ratios_matrix]))

    ax = plt.gca()
    for i, coef in enumerate(coefs):
        plt.plot(np.log10(alphas), coef, label=f'L1 ratio = {l1_ratios[i]}')

    plt.xlabel('Log(alpha)')
    plt.ylabel('Coefficients')
    plt.title('ElasticNet Regularization Path')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

def plot_feature_importance(coef, feature_names, title='Feature Importance'):
    """Plots feature importance.

    Parameters:
        coef (list): Coefficients for features.
        feature_names (list): Names of the features.
        title (str): Title of the plot.

    Returns:
        None: Displays a plot.
    """
    feature_importance = pd.Series(coef, index=feature_names).sort_values(ascending=True)
    feature_importance.plot(kind='barh', figsize=(8, 6))
    plt.title(title)
    plt.xlabel('Coefficient Value')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.show()

def get_high_vif_features(data, threshold=VIF_THRESHOLD):
    """Identifies features with high Variance Inflation Factor (VIF).
    
    Parameters:
        data (pd.DataFrame): The data.
        threshold (float): VIF threshold.

    Returns:
        list: Features with VIF above the threshold.
    """
    vif_data = pd.DataFrame()
    vif_data["Feature"] = data.columns
    vif_data["VIF"] = [variance_inflation_factor(data.values, i) for i in range(data.shape[1])]
    
    return vif_data[vif_data["VIF"] > threshold]["Feature"].tolist()