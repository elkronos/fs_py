import pystan
import pandas as pd
import numpy as np
import itertools
import warnings

# Stan code template
STAN_CODE = """
data {
    int<lower=0> N;
    vector[N] y;
    matrix[N, p] X;
}
parameters {
    vector[p] beta;
    real alpha;
    real<lower=0> sigma;
}
model {
    y ~ normal(X * beta + alpha, sigma);
}
"""

class bayesian_regression:
    
    def __init__(self):
        # Compile the model during initialization
        self.compiled_model = pystan.StanModel(model_code=STAN_CODE)

    def parse_formula(self, formula_str):
        """Extract response and predictors from formula string."""
        response, predictors_str = formula_str.split('~')
        predictors = [p.strip() for p in predictors_str.strip().split('+')]
        return response.strip(), predictors

    def fit_model(self, data, formula_str, iter=4000, **kwargs):
        """Fit the Bayesian linear regression model."""
        response, predictors = self.parse_formula(formula_str)
        
        # Exclude the intercept from the design matrix
        X = data[predictors].copy()
        
        stan_data = {
            'N': len(data),
            'y': data[response],
            'X': X,
            'p': len(predictors)
        }

        try:
            fit = self.compiled_model.sampling(data=stan_data, iter=iter, **kwargs)
            return fit
        except Exception as e:
            # Capture and print the exception message
            warnings.warn(f"Error for Model: {formula_str}. Exception: {str(e)}")
            return None

    @staticmethod
    def calculate_metrics(data, response, fitted_values):
        """Calculate metrics based on the model and data."""
        residuals = data[response] - fitted_values
        mae = np.mean(np.abs(residuals))
        rmse = np.sqrt(np.mean(residuals ** 2))
        return mae, rmse

    def fs_bayes(self, data, response_col, predictor_cols, date_col=None, use_waic=True, **kwargs):
        """Feature selection for Bayesian linear regression."""
        
        predictors_temp = predictor_cols.copy()
        if date_col:
            data['week'] = pd.to_datetime(data[date_col]).dt.week
            predictors_temp.append('week')

        predictor_combinations = list(itertools.chain(*[
            itertools.combinations(predictors_temp, r) for r in range(1, len(predictors_temp) + 1)
        ]))
        
        best_model = None
        best_waic = float('inf') if use_waic else None
        
        for predictor_comb in predictor_combinations:
            formula_str = f"{response_col} ~ {' + '.join(predictor_comb)}"
            model = self.fit_model(data, formula_str, **kwargs)
            if model and use_waic:
                waic_val = -2 * model.loo()
                if waic_val < best_waic:
                    best_model = model
                    best_waic = waic_val
        
        params = best_model.extract()
        beta_mean = params['beta'].mean(axis=0)
        alpha_mean = params['alpha'].mean()
        predictors = [col for col in data.columns if col != response_col]
        fitted_values = data[predictors].dot(beta_mean) + alpha_mean

        mae, rmse = self.calculate_metrics(data, response_col, fitted_values)
        
        print("Best Model:", best_model)
        if use_waic:
            print("Best WAIC:", best_waic)
        print("MAE:", mae)
        print("RMSE:", rmse)
        
        return {
            "Model": best_model,
            "Data": data,
            "MAE": mae,
            "RMSE": rmse
        }
