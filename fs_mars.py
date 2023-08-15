import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, learning_curve, RepeatedKFold
from sklearn.metrics import mean_squared_error, accuracy_score
from pyearth import Earth

def fs_mars(data, response_name="response", p=0.8, degree=list(range(1, 4)), 
            nprune=[5, 10, 15], task='regression', search="grid", n_iter=10, 
            n_splits=5, n_repeats=1, seed=123, scale_data=True, save_plot=False, 
            parallel_processing=False, **kwargs):
    """
    Train and evaluate a MARS (Multivariate Adaptive Regression Splines) model on a dataset.
    
    Parameters:
    - data: pd.DataFrame
        The input dataset containing both features and the response variable.
        
    - response_name: str, default="response"
        The name of the column in `data` that represents the response variable.
        
    - p: float, default=0.8
        The proportion of the dataset to be used for training. The remainder will be used for testing.
        
    - degree: list of int, default=[1,2,3]
        The list of degrees for the MARS model to be considered during the search.
        
    - nprune: list of int, default=[5, 10, 15]
        A list specifying the maximum number of terms to be considered in the pruned model.
        
    - task: str, default='regression'
        The type of machine learning task. Supported tasks are 'regression' and 'classification'.
        
    - search: str, default="grid"
        The method used for hyperparameter search. Supported methods are 'grid' for grid search and 'random' for randomized search.
        
    - n_iter: int, default=10
        Number of iterations for randomized search. Ignored if `search` is set to "grid".
        
    - n_splits: int, default=5
        The number of splits for the cross-validation.
        
    - n_repeats: int, default=1
        Number of times cross-validator needs to be repeated.
        
    - seed: int, default=123
        Random seed for reproducibility.
        
    - scale_data: bool, default=True
        Whether to scale the data using StandardScaler before training.
        
    - save_plot: bool, default=False
        If set to True and task is 'regression', a scatter plot of true vs predicted values will be saved as 'prediction_plot_regression.png'.
        
    - parallel_processing: bool, default=False
        If set to True, the search will use all available cores for parallel processing. If False, the search will be sequential.
        
    - **kwargs:
        Additional arguments passed to GridSearchCV or RandomizedSearchCV.

    Returns:
    Dictionary with:
    - 'model': The trained model object.
    - Performance metric: 'rmse' for regression tasks or 'accuracy' for classification tasks.
    - 'scaler': The trained scaler object, if scaling was applied. None otherwise.
    - 'feature_importances': (Optional) A list of importances of features if the model supports it.
    - 'learning_curve': A dictionary with learning curve data to analyze overfitting. It contains 'train_sizes', 'train_scores', and 'valid_scores'.
    
    Exceptions:
    - ValueError: Raised if an incorrect task name is provided or the specified response column is missing in the data.
    """
    
    # Validating task
    if task not in ['regression', 'classification']:
        raise ValueError("Invalid task. Choose either 'regression' or 'classification'.")
    
    if not response_name in data.columns:
        raise ValueError(f"'{response_name}' not found in data columns.")
    
    # Separate features and target
    X = data.drop(response_name, axis=1)
    y = data[response_name]
    
    # Data scaling
    scaler = None
    if scale_data:
        scaler = StandardScaler().fit(X)  
        X = pd.DataFrame(scaler.transform(X), columns=X.columns, index=X.index)
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=p, random_state=seed)
    
    param_grid = {'max_degree': degree, 'max_terms': nprune}
    
    # Scoring metric choice
    scoring = 'neg_mean_squared_error' if task == 'regression' else 'accuracy'
    
    model = Earth()
    
    # Grid or Randomized Search
    if search == "grid":
        searcher = GridSearchCV(model, param_grid, scoring=scoring,
                                cv=RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed), 
                                n_jobs=-1 if parallel_processing else None,  # Parallel processing
                                **kwargs)
    elif search == "random":
        searcher = RandomizedSearchCV(model, param_grid, scoring=scoring, n_iter=n_iter, 
                                      cv=RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed),
                                      n_jobs=-1 if parallel_processing else None,  # Parallel processing
                                      **kwargs)
    else:
        raise ValueError("Unsupported search method. Supported methods are 'grid' and 'random'.")
    
    # Fitting the model
    searcher.fit(X_train, y_train)
    best_model = searcher.best_estimator_
    
    # Predictions
    y_pred = best_model.predict(X_test)
    
    # Model evaluation and visualization
    results = {'model': best_model, 'scaler': scaler}

    if task == 'regression':
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        results['rmse'] = rmse
        
        # Visualization
        if save_plot:
            import matplotlib.pyplot as plt
            plt.scatter(y_test, y_pred, alpha=0.5)
            plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
            plt.xlabel('True Values')
            plt.ylabel('Predictions')
            plt.title('True vs Predicted Values')
            plt.savefig('prediction_plot_regression.png')
    else:
        acc = accuracy_score(y_test, y_pred)
        results['accuracy'] = acc
        
        # (Potential) Feature Importances
        if hasattr(best_model, "feature_importances_"):
            results['feature_importances'] = best_model.feature_importances_

    # Learning curve
    train_sizes, train_scores, valid_scores = learning_curve(best_model, X, y, cv=RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed))
    results['learning_curve'] = {'train_sizes': train_sizes, 'train_scores': train_scores, 'valid_scores': valid_scores}
    
    return results
