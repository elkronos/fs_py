# fs_py
 Feature selection functions for python. This repo is still under development, but suggestions are welcome! These functions are currently undergoing a full UAT. Please use at your own discretion.


# fs_bayes Script Summary

- **Script Name**: fs_bayes
- **Modules Imported**:
  - `pystan`: For Bayesian statistics and probabilistic programming.
  - `pandas`: To manipulate and analyze data.
  - `numpy`: For numerical computations.
  - `itertools`: For creating iterators for efficient looping.
  - `warnings`: To capture and manage warnings during runtime.

- **Classes and Methods**:
  - `class bayesian_regression`:
    - `__init__(self)`: Initializes and compiles the Stan model with the given STAN_CODE.
    - `parse_formula(self, formula_str)`: Extracts response and predictor variables from a formula string.
    - `fit_model(self, data, formula_str, iter=4000, **kwargs)`: Fits the Bayesian linear regression model using the parsed formula and input data.
    - `calculate_metrics(data, response, fitted_values)`: Static method to calculate metrics (MAE, RMSE) based on the model and data.
    - `fs_bayes(self, data, response_col, predictor_cols, date_col=None, use_waic=True, **kwargs)`: Conducts feature selection for Bayesian linear regression, evaluates different predictor combinations, and identifies the best model based on WAIC or a different criterion.

- **Features**:
  - It supports input in the form of a formula string to specify the response and predictor variables.
  - Can optionally incorporate date information into the feature selection process.
  - Offers flexibility in setting the iteration count and additional parameters for the `fit_model` method through `**kwargs`.

- **Usage**:
  - This script is used to perform Bayesian linear regression with feature selection, identifying the best model based on user-specified criteria (like WAIC) and calculating performance metrics (MAE and RMSE).
  - Can be utilized to handle regression problems where Bayesian inference is necessary to estimate the parameters and uncertainties.


# fs_boruta Script Summary

- **Script Name**: fs_boruta
- **Modules Imported**:
  - `pandas`: To manipulate and analyze data.
  - `numpy`: For numerical computations and array manipulations.
  - `boruta`: To perform feature selection using the Boruta algorithm.
  - `sklearn.ensemble`: To use the RandomForestClassifier as the base estimator for Boruta.
  - `sklearn.impute`: To handle missing values using SimpleImputer.
  - `typing`: To annotate the types of function parameters and return values.

- **Function**:
  - `fs_boruta(data, target_var, seed=None, maxRuns=250, num_cores=1, cutoff_features=None, cutoff_cor=0.7, verbose=2, rf_params=None, return_selector=True, handle_nan='drop', remove_both=False, custom_imputer=None)`: 
    - **Parameters**:
      - `data`: The input data as a pandas DataFrame.
      - `target_var`: The target variable for feature selection.
      - Various other parameters to control the feature selection process including options for handling missing values, configuring the random forest classifier, and adjusting the correlation cutoff.
    - **Functionality**: 
      - Performs feature selection using the Boruta algorithm.
      - Handles missing values based on the parameter `handle_nan` or using a custom imputer.
      - Removes highly correlated features based on the `cutoff_cor` parameter.
      - Can limit the number of selected features using the `cutoff_features` parameter.
    - **Returns**:
      - A dictionary containing selected and rejected features, importance scores, and optionally the Boruta selector object.

- **Features**:
  - Provides flexibility with numerous parameters to control the Boruta feature selection process.
  - Allows for reproducibility through the `seed` parameter.
  - Can return the Boruta selector object for further analysis or use.

- **Usage**:
  - This script is used to automate feature selection for machine learning models, particularly useful in pre-processing stages of data pipeline to help build models with significant features.
  - Helpful in handling missing values through various strategies such as dropping, mean, median, or mode imputation.


# fs_chi Script Summary

- **Script Name**: fs_chi
- **Modules Imported**:
  - `pandas`: For data manipulation and analysis.
  - `scipy.stats`: To compute the chi-square test.
  - `numpy`: For numerical computations and handling NaN values.
  - `warnings`: To issue warnings in case of detected issues during the computation.

- **Functions**:
  - `compute_chi2(data, feature, target_col, correct, min_freq=5)`:
    - **Parameters**:
      - `data`: The input data as a pandas DataFrame.
      - `feature`: The feature column to compute the chi-square test on.
      - `target_col`: The target column for the chi-square test.
      - `correct`: Boolean indicating whether to apply Yates’ correction for continuity.
      - `min_freq`: Minimum frequency count for cells in the contingency table, default is 5.
    - **Functionality**: 
      - Computes the chi-square statistic, p-value, and expected frequencies for a given feature against a target column.
      - Issues warnings and returns NaN if missing values are detected or minimum frequency condition is not met.
    - **Returns**:
      - p-value, chi-square statistic, and expected frequencies of the test.

  - `fs_chi(data, target_col, sig_level=0.05, correct=True, apply_bonferroni=True, min_freq=5)`:
    - **Parameters**:
      - `data`: The input data as a pandas DataFrame.
      - `target_col`: The target column for the chi-square tests.
      - `sig_level`: Significance level for the test, default is 0.05.
      - `correct`: Boolean indicating whether to apply Yates’ correction, default is True.
      - `apply_bonferroni`: Boolean indicating whether to apply Bonferroni correction for multiple testing, default is True.
      - `min_freq`: Minimum frequency count for cells in the contingency table, default is 5.
    - **Functionality**: 
      - Performs feature selection using the chi-square test for each categorical feature against a target column.
      - Converts object data types to category and applies the chi-square test using the helper function `compute_chi2`.
    - **Returns**:
      - A dictionary containing significant feature names, p-values, chi2 statistics, and expected frequencies for each feature.

- **Features**:
  - Provides a comprehensive result including p-values, chi2 statistics, and expected frequencies for each feature.
  - Implements error handling and data type conversions to ensure appropriate application of the chi-square test.
  - Offers options for statistical corrections including Yates' continuity correction and Bonferroni correction for multiple testing.

- **Usage**:
  - This script is used for feature selection by identifying statistically significant categorical features through chi-square tests, which is generally applied in the data preprocessing step before training a machine learning model.
  - It can help to identify and retain only the significant features for model training, potentially improving model performance and interpretability.


# fs_correlation Script Summary

- **Script Name**: fs_correlation
- **Modules Imported**:
  - `pandas`: For data manipulation and analysis.
  - `numpy`: For numerical operations.
  - `seaborn` and `matplotlib.pyplot`: For data visualization, especially heatmaps.
  - `scipy.stats`: For different correlation methods - Pearson, Spearman, and Kendall.

- **Classes**:
  - `InvalidInputError(Exception)`: Custom exception class for handling invalid inputs.

- **Functions**:
  - `distance_correlation(X, Y)`:
    - Computes the distance correlation between two variables.
    - **Returns**: The distance correlation value.
    
  - `compute_correlation(x, y, method)`:
    - Computes correlation for two variables based on the specified method (Pearson, Spearman, Kendall, Distance).
    - **Returns**: The correlation value.
    
  - `rank_by_correlation_magnitude(corr_matrix)`:
    - Ranks the columns of a correlation matrix based on the magnitude of their average correlations.
    - **Returns**: Columns ranked by their average correlation magnitude.
    
  - `fs_correlation(data, threshold, method="pearson", plot_heatmap=False)`:
    - Computes the correlation matrix for input data and selects variables based on a given correlation threshold.
    - Can optionally plot a heatmap of the correlation matrix.
    - **Returns**: A dictionary containing the correlation matrix, selected variables, and ranked features based on their average correlation magnitude.
    
  - `check_multicollinearity(corr_matrix, threshold=0.8)`:
    - Checks for multicollinearity in the features based on a given threshold.
    - **Returns**: Pairs of features that have a correlation greater than the threshold.
    
- **Global Variables**:
  - `correlation_methods`: A dictionary containing different correlation methods (functions) mapped to their respective names.

- **Features**:
  - Provides methods to compute various types of correlations including Pearson, Spearman, Kendall, and Distance correlation.
  - Can visualize the correlation matrix with a heatmap.
  - Assists in feature selection by eliminating features with high correlations.
  - Includes functionality to identify multicollinear feature pairs, which is essential in linear modeling.
  
- **Usage**:
  - This script can be employed for feature selection and multicollinearity checking in datasets where features might be highly correlated. It's useful in the preprocessing step before training regression models or other linear algorithms sensitive to multicollinearity.


# fs_elastic Script Summary

- **Script Name**: fs_elastic
- **Modules Imported**:
  - `numpy`: For numerical computations and handling arrays.
  - `pandas`: To manipulate and analyze data.
  - `scipy`: For statistical computations including computing z-scores.
  - `sklearn`: For machine learning components like model selection, pipeline, linear models, preprocessing, and hyperparameter tuning.
  - `matplotlib.pyplot`: For visualization and plotting graphs.
  - `statsmodels`: To create design matrices and calculate variance inflation factor.

- **Global Variables**:
  - `RANDOM_STATE`: To set the seed for random number generation (Value: 42).
  - `TEST_SIZE`: To specify the proportion of the data to be used as the test set (Value: 0.2).
  - `Z_THRESHOLD`: The threshold for z-scores to identify outliers (Value: 3).
  - `VIF_THRESHOLD`: The threshold for variance inflation factor to identify multicollinearity (Value: 10).

- **Functions**:
  - `remove_outliers(data, z_threshold=Z_THRESHOLD)`:
    - Removes outliers based on the z-score threshold.
    - **Returns**: Data without outliers.

  - `prepare_data(data, formula)`:
    - Separates predictors and response and imputes missing values using mean strategy.
    - **Returns**: Processed predictors and response.

  - `apply_scaling(X_train, X_test, scale_method)`:
    - Scales the data using either standard scaling or min-max scaling.
    - **Returns**: Scaled training and test data, along with the scaler used.

  - `fs_elastic(data, formula, l1_ratios=None, cv=5, scale_method='standard', alphas=None, visualize=False)`:
    - Main function to perform feature selection using ElasticNet and identifies the best model and parameters through cross-validation.
    - **Returns**: A dictionary containing results including the best model and parameters.

  - `plot_regularization_path(l1_ratios, alphas, X_train, y_train, feature_names)`:
    - Visualizes the ElasticNet regularization path for different alpha and l1_ratio values.
    - **Returns**: None (Displays a plot).

  - `plot_feature_importance(coef, feature_names, title='Feature Importance')`:
    - Plots the feature importance based on the coefficients of the ElasticNet model.
    - **Returns**: None (Displays a plot).

  - `get_high_vif_features(data, threshold=VIF_THRESHOLD)`:
    - Identifies features with high variance inflation factor (VIF).
    - **Returns**: List of features with high VIF.

- **Features**:
  - Implements a comprehensive feature selection process using the ElasticNet algorithm.
  - Provides visualization tools to analyze the effect of regularization on feature coefficients and to understand feature importance.
  - Helps in handling outliers and multicollinearity in the dataset effectively.
  - Utilizes HalvingGridSearchCV for efficient hyperparameter tuning.

- **Usage**:
  - This script is utilized for feature selection in machine learning pipelines, especially when working with linear models. It helps in preprocessing data by removing outliers, scaling features, and selecting important features through regularization.


# fs_info Script Summary

- **Script Name**: fs_info
- **Modules Imported**:
  - `pandas` (as `pd`): For data manipulation and analysis.
  - `numpy` (as `np`): To support numerical and mathematical operations.

- **Functions**:
  - `entropy(x)`:
    - Computes the entropy of a given array `x`.
    - **Returns**: The entropy value.
    
  - `calculate_bins(x)`:
    - Calculates the number of bins for histogram using Freedman-Diaconis rule and Sturges' rule.
    - **Returns**: The number of bins calculated based on the data distribution.
    
  - `fs_infogain(df, target)`:
    - Computes the information gain for each variable in the dataframe with respect to the target variable.
    - Extracts year, month, and day from date columns and removes original date columns.
    - **Returns**: A DataFrame containing each variable and its corresponding information gain value.
    
  - `calculate_information_gain_multiple(dfs_list)`:
    - Computes information gain for multiple dataframes in the list and tags each result with an "Origin" denoting the data frame it came from.
    - **Returns**: A DataFrame containing the results of information gain calculations for all dataframes in the list.

- **Features**:
  - Computes the entropy of given variables to calculate information gain.
  - Dynamically calculates histogram bins based on data distribution, facilitating optimal discretization of numerical variables.
  - Processes date columns by extracting relevant date components (year, month, day).
  - Enables batch processing of multiple dataframes to compute information gain simultaneously, saving time and effort in feature selection processes.

- **Usage**:
  - This script is pivotal for feature selection processes where information gain is a determinant for selecting relevant features.
  - Suitable for preprocessing stages in data science pipelines, aiding in the identification and selection of significant features based on information gain metrics.
  - Can be extended for batch processing of multiple dataframes, facilitating parallel feature selection across different datasets.


# fs_lasso Script Summary

- **Script Name**: fs_lasso
- **Modules Imported**:
  - `numpy` (as `np`): To support numerical and mathematical operations.
  - `pandas` (as `pd`): For data manipulation and analysis.
  - `LassoCV`, `Lasso` from `sklearn.linear_model`: To perform Lasso and cross-validated Lasso regression.
  - `StandardScaler`, `MinMaxScaler` from `sklearn.preprocessing`: To standardize the input data.
  - `catch_warnings`, `simplefilter`, `warn` from `warnings`: To handle and log warnings.
  - `ConvergenceWarning` from `sklearn.exceptions`: To catch specific warnings related to convergence.

- **Functions**:
  - `fs_lasso(...)`:
    - **Purpose**: Performs feature selection using Lasso or LassoCV regression, providing the importance of each feature.
    - **Parameters**:
      - Various options for handling input data, specifying model parameters, and configuring output.
    - **Returns**:
      - A DataFrame with feature importances.
      - Optionally, the fitted scaler and/or the fitted model.
    - **Notes**: Contains specific details and instructions about ensuring proper conditions for using Lasso regression.

- **Features**:
  - Validates input data and handles possible issues with missing values and data types.
  - Standardizes input data using specified or custom scaler.
  - Provides flexibility to choose between Lasso and LassoCV, allowing user-defined alpha values or automated cross-validation to find the best alpha.
  - Offers options to return the scaler and/or model used, facilitating further analysis or usage in pipelines.
  - Implements robust warning handling to catch and log convergence warnings.

- **Usage**:
  - This script is a utility for feature selection, especially in the context of regression analysis, helping identify the most important features in a dataset.
  - Suitable for integrating into data preprocessing pipelines where feature selection is a crucial step.
  - Can be used for detailed analysis of model performance and feature importance in research or model development projects.
  

# fs_mars Script Summary

- **Script Name**: fs_mars
- **Modules Imported**:
  - `numpy` (as `np`): Supports various numerical and mathematical operations.
  - `pandas` (as `pd`): Utilized for data manipulation and analysis.
  - Various functions from `sklearn.preprocessing` and `sklearn.model_selection`: Employed for data preprocessing, model training, and hyperparameter tuning.
  - `mean_squared_error`, `accuracy_score` from `sklearn.metrics`: Employed to calculate performance metrics for the model.
  - `Earth` from `pyearth`: Utilized as the primary model for training, which supports Multivariate Adaptive Regression Splines (MARS).

- **Functions**:
  - `fs_mars(...)`:
    - **Purpose**: To train and evaluate a MARS model on a dataset, providing options for hyperparameter tuning and data preprocessing.
    - **Parameters**:
      - Accepts a variety of parameters to customize data input, model training process, and output, including options for specifying task type, hyperparameter search method, cross-validation settings, and more.
    - **Returns**:
      - Outputs a dictionary containing the trained model, a performance metric (either RMSE or accuracy), the trained scaler object (if data scaling was applied), and learning curve data to evaluate overfitting.
    - **Exceptions**: Triggers ValueError if an incorrect task name is provided or the specified response column is not present in the data.

- **Features**:
  - Provides flexibility in defining the machine learning task as either regression or classification and allows configuring the hyperparameter search method as either grid or random.
  - Incorporates options for data scaling using `StandardScaler` and for parallel processing during the hyperparameter search to expedite computations.
  - Implements error handling mechanisms to manage incorrect task names or missing response columns.
  - Facilitates saving a plot that visualizes true versus predicted values during regression analysis.
  - Supports computation of feature importances for classification tasks, given the model supports it.

- **Usage**:
  - Functions as a utility tool for creating and evaluating MARS models, aiding in tasks such as hyperparameter tuning and data preprocessing.
  - Suitable for integration into data analysis pipelines, especially where regression or classification tasks are conducted using MARS models.
  - Apt for usage in both research and development projects aiming for a detailed analysis of model performance, feature importances, and potential overfitting issues.


# fs_pca Script Summary

- **Script Name**: fs_pca
- **Modules Imported**:
  - `pandas` (as `pd`): Used for data handling and manipulation.
  - `numpy` (as `np`): Supports various numerical and mathematical operations.
  - `PCA` from `sklearn.decomposition`: The primary class used to perform Principal Component Analysis (PCA).
  - `StandardScaler` from `sklearn.preprocessing`: Utilized for standardizing the dataset (if required) before PCA.
  - `seaborn` (as `sns`): A data visualization library used for plotting the results of PCA.
  - `matplotlib.pyplot` (as `plt`): Used in conjunction with seaborn to enhance plots and display them.

- **Functions**:
  - `perform_pca(data, num_pc=2, scale_data=True)`:
    - **Purpose**: To perform Principal Component Analysis on the input dataset.
    - **Parameters**:
      - `data` (pd.DataFrame): The input data for PCA.
      - `num_pc` (int, optional): The number of principal components to extract.
      - `scale_data` (bool, optional): Flag to decide whether to scale the data before performing PCA.
    - **Returns**:
      - A dictionary containing the PCA loadings, scores, variance explained, and a DataFrame containing PCA scores alongside non-numeric columns from the original data.
    - **Exceptions**: Raises ValueErrors for invalid input data, non-positive `num_pc`, and when `num_pc` exceeds the number of features or rows.

  - `plot_pca(pca_result, label_col=None)`:
    - **Purpose**: To visualize the results of PCA using scatter plots or histograms.
    - **Parameters**:
      - `pca_result` (dict): The dictionary output from the `perform_pca` function.
      - `label_col` (str, optional): Column name to use for labeling data points in the scatter plot.
    - **Returns**:
      - None: Displays a plot and prints the variance explained by each principal component.
    - **Exceptions**: Raises KeyErrors if essential keys are not found in the `pca_result` dictionary.

  - `scree_plot(pca_result)`:
    - **Purpose**: To visualize the proportion of variance explained by each principal component through a scree plot.
    - **Parameters**:
      - `pca_result` (dict): The dictionary output from the `perform_pca` function.
    - **Returns**:
      - None: Displays a scree plot showing the variance explained by each principal component.

- **Features**:
  - Comprehensive PCA analysis including data preparation (scaling and extracting numeric columns), PCA computation, and visualization through scatter plots, histograms, and scree plots.
  - Implementations for error handling to manage incorrect inputs and ensure consistent output.
  - Utilizes seaborn and matplotlib for high-quality data visualizations, with options for coloring data points based on labels.
  - A modular approach with separate functions for performing PCA and visualizations, allowing for flexible usage in data analysis pipelines.

- **Usage**:
  - Suitable for exploratory data analysis and dimensionality reduction in data science projects.
  - Functions can be utilized in a modular manner, easily integrating within larger data analysis pipelines.
  - Can assist in understanding the underlying patterns and structure of the data through PCA visualizations.


# fs_randomforest Script Summary

- **Script Name**: fs_randomforest
- **Modules Imported**:
  - `os`: For interacting with the OS, especially regarding file paths.
  - `logging`: To set up logging for the script.
  - Various types from `typing`: To define type hints for function signatures.
  - Several modules from `sklearn`: For data preprocessing, model building, and evaluation.
  - `pandas` (as `pd`): For data manipulation and analysis.
  - `imblearn.over_sampling.SMOTE`: To balance the data using Synthetic Minority Over-sampling Technique.
  - `joblib`: To save and load trained models.

- **Functions**:
  - `scale_features(...)`:
    - **Purpose**: Scales the features using StandardScaler.
    - **Parameters**: Training and testing data.
    - **Returns**: Scaled training and testing data.

  - `handle_categorical_features(...)`:
    - **Purpose**: Handles categorical features by encoding them using LabelEncoder.
    - **Parameters**: Input data frame.
    - **Returns**: Transformed data frame with encoded categorical features.

  - `balance_data(...)`:
    - **Purpose**: Balances the dataset using various methods such as SMOTE, undersampling, or oversampling.
    - **Parameters**: Input data frame, target column name, balance method, and random seed.
    - **Returns**: Balanced data frame.

  - `hyperparam_tune(...)`:
    - **Purpose**: Performs hyperparameter tuning for the RandomForest model using grid search or random search.
    - **Parameters**: RandomForest model, training features and target, hyperparameter tuning configuration, number of cross-validation folds.
    - **Returns**: Best model and its parameters.

  - `compute_metrics(...)`:
    - **Purpose**: Computes classification or regression metrics based on the task type.
    - **Parameters**: True target values, predicted values, task type, and average type for classification metrics.
    - **Returns**: Dictionary of computed metrics.

  - `prepare_data(...)`:
    - **Purpose**: Prepares data by handling categorical features and balancing the data if necessary.
    - **Parameters**: Data, selected features, target column, balance method, task type, and random seed.
    - **Returns**: Prepared data.

  - `load_hyperparams_from_config(...)`:
    - **Purpose**: Loads hyperparameters from a specified configuration file.
    - **Parameters**: Configuration file path.
    - **Returns**: Dictionary of loaded configuration.

  - `train_and_evaluate_rf(...)`:
    - **Purpose**: Trains the random forest model and evaluates its performance.
    - **Parameters**: Various parameters including data, target column, random forest parameters, task type, stratification option, data split ratio, cross-validation folds, scaling option, etc.
    - **Returns**: Dictionary with evaluation results and model details.

  - `fs_randomforest(...)`:
    - **Purpose**: Main function to prepare data, train, and evaluate a random forest model.
    - **Parameters**: Various parameters including data, target column, selected features, balance method, task type, random seed, stratification option, data split ratio, etc.
    - **Returns**: Dictionary with model details or evaluation results.

- **Features**:
  - Handles categorical features effectively by transforming them into numerical values.
  - Offers data balancing capabilities using different techniques such as SMOTE, undersampling, and oversampling.
  - Supports hyperparameter tuning using grid search or random search.
  - Allows for k-fold cross-validation.
  - Facilitates saving and loading of trained models for later use.

- **Usage**:
  - This script is a comprehensive tool for building and evaluating random forest models, suitable for both classification and regression tasks.
  - Can be integrated into data pipelines to streamline the model building process.
  - Useful for researchers and practitioners aiming to build robust random forest models with various data preprocessing and evaluation options.


# fs_recrusivefeature Script Summary

- **Script Name**: fs_recrusivefeature
- **Modules Imported**:
  - `pandas` (as `pd`): Used for data manipulation and analysis.
  - `RandomForestClassifier` from `sklearn.ensemble`: A machine learning classifier used as the default estimator for feature ranking.
  - `RFECV` from `sklearn.feature_selection`: Performs Recursive Feature Elimination with Cross-Validation to select the most important features.
  - `train_test_split`, `StratifiedKFold` from `sklearn.model_selection`: Used for splitting the data into training and test sets and for stratified k-fold cross-validation, respectively.

- **Functions**:
  - `fs_recrusivefeature(...)`:
    - **Purpose**: To perform feature selection by recursively eliminating less important features using the specified or default estimator and identifying the optimal number of features through cross-validation.
    - **Parameters**:
      - `data`: Input DataFrame containing both features and the response variable.
      - `response_var_column`: Column name of the response variable in the input data.
      - `estimator`: (Optional) Estimator to use for feature ranking. Default is `RandomForestClassifier`.
      - `seed`: (Optional) Random seed for reproducibility. Default is 123.
      - `n_splits`: (Optional) Number of splits for cross-validation. Default is 5.
      - `test_size`: (Optional) Proportion of the data to be used as test data, between 0 and 1. Default is 0.2.
    - **Returns**:
      - A dictionary containing the optimal number of features, the names of the optimal features, the importance of each feature, cross-validation scores, and the trained estimator.
    - **Notes**: Includes several validation checks to ensure proper input data format and parameters, raising appropriate errors if necessary.

- **Features**:
  - Validates the presence of the response variable column and ensures there are more than one column in the input data.
  - Validates the `test_size` and `n_splits` parameters to ensure they are within appropriate ranges.
  - Utilizes stratified k-fold cross-validation to maintain the proportion of the response variable in each fold.
  - Returns a dataframe detailing the importance of each feature as determined by the estimator.

- **Usage**:
  - This script serves as a utility for feature selection, assisting in identifying the most important features in a dataset for modeling purposes.
  - Suitable for integration into data preprocessing pipelines where feature selection is a vital step.
  - Can be used for detailed analysis of feature importance in research or model development projects, offering insights into which features contribute most to predictive performance.


# fs_stepwise Script Summary

- **Script Name**: fs_stepwise
- **Modules Imported**:
  - `pandas` (as `pd`): Utilized for data manipulation and analysis.
  - `numpy` (as `np`): Used for numerical operations.
  - `KFold`, `cross_val_score` from `sklearn.model_selection`: For cross-validation.
  - `StandardScaler` from `sklearn.preprocessing`: For data scaling.
  - `BaseEstimator`, `clone` from `sklearn.base`: To create and manipulate base models.
  - `LinearRegression` from `sklearn.linear_model`: Used as a default model.
  - `dump`, `load` from `joblib`: To save and load model states.

- **Functions**:
  - `StepwiseRegression(...)`: Initializes stepwise regression object.
    - **Purpose**: Implements feature selection using forward and backward elimination steps.
    - **Parameters**:
      - `dependent_var`: The dependent variable's column name.
      - `base_model`: Base model to use (default is `LinearRegression`).
      - `step_type`: Type of stepwise selection ("forward", "backward", "both").
      - Other options to control behavior, such as verbosity, scaling, and monitoring.
    - **Returns**: Initialized `StepwiseRegression` object.

  - `fit(data: pd.DataFrame)`: Fits the stepwise regression model.
    - **Purpose**: Executes forward and/or backward selection to find optimal features.
    - **Parameters**: Training data as a pandas DataFrame.
    - **Returns**: Trained `StepwiseRegression` object.

  - `predict(X_new: pd.DataFrame, model_features=None)`: Predicts using the trained model.
    - **Purpose**: Makes predictions on new data using selected features.
    - **Parameters**:
      - `X_new`: New data as a pandas DataFrame.
      - `model_features`: Features to use for prediction (default is best selected features).
    - **Returns**: Predicted values.

  - Private methods like `_forward_selection`, `_backward_elimination`, `_cross_val`, and `_scale_data` perform internal steps.

- **Features**:
  - Implements stepwise regression with forward and backward selection.
  - Supports custom or default base models for regression.
  - Option to scale data and monitor standard deviation during selection.
  - Provides methods to fit, predict, save, and load model states.
  - Handles column mismatches and missing data during prediction.

- **Usage**:
  - Useful for automating feature selection in regression tasks.
  - Integrates well into preprocessing pipelines for efficient feature selection.
  - Offers flexibility in selecting the best subset of features based on forward and backward elimination.
  - Allows saving and loading model states for reproducibility and sharing.


# fs_svd Script Summary

- **Script Name**: fs_svd
- **Modules Imported**:
  - `numpy` (as `np`): Utilized for numerical operations.
  - `pandas` (as `pd`): Used for data manipulation.
  - `Enum`, `auto` from `enum`: Used for defining enums.

- **Enumerations**:
  - `ScaleType`:
    - Purpose: Enumerates scaling types (BOTH, CENTER, SCALE).
    - Values: `BOTH`, `CENTER`, `SCALE`.

- **Functions**:
  - `validate_inputs(...)`: Validates input data and parameters for `fs_svd` function.
    - **Parameters**:
      - `matrix_data`: Input matrix data (numpy.ndarray or pandas.DataFrame).
      - `scale_input`: Desired scaling type (BOTH, CENTER, or SCALE).
      - `n_singular_values`: Desired number of singular values.
    - **Raises**:
      - `TypeError` if input data is not valid.
      - `ValueError` for improper data dimensions or invalid singular value count.

  - `center_matrix(...)`: Centers input matrix by subtracting the mean.
    - **Parameters**: Input matrix data.
    - **Returns**: Centered matrix.

  - `scale_matrix(...)`: Scales input matrix to have unit variance.
    - **Parameters**: Input matrix data.
    - **Returns**: Scaled matrix.

  - `apply_scaling(...)`: Applies desired scaling (centering and/or scaling) to matrix.
    - **Parameters**:
      - `matrix_data`: Input matrix data.
      - `scale_input`: Desired scaling type.
    - **Returns**: Scaled matrix.
    - **Raises**: `ValueError` for unrecognized scaling types.

  - `variance_explained(...)`: Computes variance explained by singular values.
    - **Parameters**: Array of singular values.
    - **Returns**: Array of explained variances.

  - `reduced_matrix(...)`: Gets matrix using reduced dimensions.
    - **Parameters**:
      - `u`: Left singular vectors.
      - `s`: Singular values.
      - `vh`: Right singular vectors.
    - **Returns**: Matrix in reduced dimensions.

  - `fs_svd(...)`: Computes SVD for given matrix with optional scaling.
    - **Parameters**:
      - `matrix_data`: Input matrix data.
      - `scale_input`: Desired scaling type.
      - `n_singular_values`: Desired number of singular values.
      - Other options for output format, NaN handling, etc.
    - **Returns**: Dictionary with SVD results.
    - **Raises**: `ValueError` for unrecognized output data type or invalid singular values.

- **Usage**:
  - Computes SVD of a matrix with flexible scaling options.
  - Handles missing values using specified methods.
  - Supports truncation of singular values for dimensionality reduction.
  - Can return results in desired data format (numpy.ndarray or pandas.DataFrame).
  - Suitable for data preprocessing, dimensionality reduction, and feature extraction.


# fs_svm Script Summary

- **Script Name**: fs_svm
- **Modules Imported**:
  - `numpy` (as `np`): Utilized for numerical operations.
  - `pandas` (as `pd`): Used for data manipulation.
  - `train_test_split`, `GridSearchCV` from `sklearn.model_selection`: For data splitting and hyperparameter tuning.
  - `SVC`, `SVR` from `sklearn.svm`: For support vector classification and regression.
  - `confusion_matrix`, `r2_score` from `sklearn.metrics`: For performance metrics.
  - `StandardScaler` from `sklearn.preprocessing`: For data scaling.
  - `Pipeline` from `sklearn.pipeline`: For constructing a processing pipeline.
  - `Union` from `typing`: For type hints.

- **Functions**:
  - `split_data(...)`: Splits dataframe into train and test datasets.
    - **Parameters**:
      - `data`: Input dataframe.
      - `target`: Target column name.
      - `task`: Task type (classification or regression).
      - Other options for test size, stratification, and seed.
    - **Returns**: Train-test split datasets (X_train, X_test, y_train, y_test).

  - `get_tuning_grid(...)`: Provides hyperparameters grid based on task and model type.
    - **Parameters**:
      - `task`: Task type (classification or regression).
      - `model_type`: Model type.
    - **Returns**: Hyperparameters tuning grid.

  - `compute_performance(...)`: Computes performance metrics based on task type.
    - **Parameters**:
      - `y_test`: True target values.
      - `predictions`: Predicted values.
      - `task`: Task type (classification or regression).
    - **Returns**: Confusion matrix (classification) or R2 score (regression).

  - `fs_svm(...)`: Performs feature selection using SVM and returns results.
    - **Parameters**:
      - `data`: Input dataframe.
      - `target`: Target column name.
      - `task`: Task type (classification or regression).
      - Other options for cross-validation folds, hyperparameters tuning, etc.
    - **Returns**: Dictionary with best model, predictions, performance metrics, etc.

- **Usage**:
  - Splits data into train and test sets based on specified target column.
  - Supports both classification and regression tasks.
  - Constructs SVM models with hyperparameter tuning using pipelines.
  - Computes and reports performance metrics like confusion matrix and R2 score.
  - Suitable for automated model selection and evaluation.


# fs_variance Script Summary

- **Script Name**: fs_variance
- **Modules Imported**:
  - `numpy` (as `np`): Used for numerical operations.
  - `pandas` (as `pd`): Used for data manipulation.
  - `warnings`: Utilized for handling warnings.
  - `Union` from `typing`: Used for type hints.

- **Function**:
  - `fs_variance(...)`: Applies variance thresholding to a numeric dataset.
    - **Parameters**:
      - `data`: Numeric dataset (numpy array or pandas DataFrame).
      - `threshold`: Variance threshold value.
      - `return_type`: Return type of processed data (array or dataframe).
    - **Returns**:
      - Processed data (numeric array or DataFrame).
      - Returns `None` if no features have variance above the threshold.
    - **Raises**:
      - ValueError for invalid threshold, return_type, or data input.
      - Raises warnings if no features have variance above the threshold.

- **Usage**:
  - Used to preprocess numeric datasets by removing low-variance features.
  - Helps improve efficiency and model performance by reducing noise.
  - Suitable for use in data cleaning and feature selection pipelines.
  - Warns if no features meet the variance threshold, indicating potential issues.


# Contact
- email: napoleonic_bores@proton.me
- discord: elkronos
