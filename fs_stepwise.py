import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, clone
from sklearn.linear_model import LinearRegression
from joblib import dump, load

class StepwiseRegression(BaseEstimator):
    def __init__(self, 
                 dependent_var: str, 
                 base_model: BaseEstimator = None, 
                 step_type: str = "both", 
                 seed: int = None, 
                 verbose: bool = False, 
                 n_splits: int = 10, 
                 threshold: float = 0.01, 
                 scale_data: bool = True, 
                 monitor_std_dev: bool = False):

        if base_model and (not hasattr(base_model, 'fit') or not hasattr(base_model, 'predict')):
            raise ValueError("The provided model does not adhere to the scikit-learn regressor interface.")
        elif not base_model:
            base_model = LinearRegression()

        self.dependent_var = dependent_var
        self.base_model = base_model
        self.step_type = step_type
        self.seed = seed
        self.verbose = verbose
        self.n_splits = n_splits
        self.threshold = threshold
        self.scale_data = scale_data
        self.monitor_std_dev = monitor_std_dev
        self.models = {}

        if self.seed:
            np.random.seed(self.seed)

    def fit(self, data: pd.DataFrame):
        if self.n_splits > len(data):
            raise ValueError("n_splits cannot be greater than the number of rows in the data.")
        
        self.original_data = data.copy()
        self.data = data.dropna()
        
        if self.data.columns.difference([self.dependent_var]).empty:
            raise ValueError("The dataset must contain at least one feature apart from the dependent variable.")

        y = self.data[self.dependent_var]
        X = self.data.drop(columns=self.dependent_var)

        if self.scale_data:
            X, y, self.scaler_X, self.scaler_y = self._scale_data(X, y)
        else:
            self.scaler_X, self.scaler_y = None, None

        if self.verbose:
            print("Starting fit process...")

        if self.step_type in ["forward", "both"]:
            self._forward_selection(X, y)
        if self.step_type in ["backward", "both"]:
            self._backward_elimination(X, y)

        best_model = self._get_best_model()
        self.base_model.fit(X[best_model['features']], y)

        return self

    def predict(self, X_new: pd.DataFrame, model_features=None):
        unseen_columns = set(X_new.columns) - set(self.original_data.columns)
        missing_columns = set(self.original_data.columns) - set(X_new.columns)
        
        if unseen_columns:
            raise ValueError(f"Unseen columns in the input: {unseen_columns}. Please match the columns with the training data.")
        if missing_columns:
            raise ValueError(f"Missing columns in the input: {missing_columns}. Please ensure all columns from the training data are present.")
        
        if self.scale_data:
            X_new = pd.DataFrame(self.scaler_X.transform(X_new), columns=X_new.columns)

        if not model_features:
            best_model = self._get_best_model()
            model_features = best_model['features']

        X_new = X_new[model_features]

        predictions = self.base_model.predict(X_new)

        if self.scale_data:
            predictions = self.scaler_y.inverse_transform(predictions.reshape(-1, 1)).flatten()

        return predictions

    def _forward_selection(self, X, y):
        features = list(X.columns)
        selected_features = []
        last_score = -np.inf
        iteration = 0

        while True:
            iteration += 1
            remaining_features = list(set(features) - set(selected_features))
            best_score = -np.inf
            best_feature = None
            best_std_dev = None

            for feature in remaining_features:
                X_subset = X[selected_features + [feature]]
                score, std_dev = self._cross_val(X_subset, y)

                if score > best_score:
                    best_score = score
                    best_feature = feature
                    best_std_dev = std_dev

            if best_score <= last_score or (best_score - last_score < self.threshold):
                break

            selected_features.append(best_feature)
            self.models[tuple(selected_features)] = {
                'features': selected_features,
                'score': best_score,
                'std_dev': best_std_dev
            }

            last_score = best_score

            if self.verbose:
                print(f"Iteration {iteration}: Features selected: {selected_features} with score {best_score}")

    def _backward_elimination(self, X, y):
        selected_features = list(X.columns)
        last_score = -np.inf
        iteration = 0

        while len(selected_features) > 1:
            iteration += 1
            best_score = -np.inf
            feature_to_remove = None
            best_std_dev = None

            for feature in selected_features:
                features_minus_current = list(set(selected_features) - {feature})
                X_subset = X[features_minus_current]
                score, std_dev = self._cross_val(X_subset, y)

                if score > best_score:
                    best_score = score
                    feature_to_remove = feature
                    best_std_dev = std_dev

            if best_score <= last_score or (best_score - last_score < self.threshold):
                break

            selected_features.remove(feature_to_remove)
            self.models[tuple(selected_features)] = {
                'features': selected_features,
                'score': best_score,
                'std_dev': best_std_dev
            }

            last_score = best_score

            if self.verbose:
                print(f"Iteration {iteration}: Features selected after removal: {selected_features} with score {best_score}")

    def _cross_val(self, X, y):
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)
        model = clone(self.base_model)
        scores = cross_val_score(model, X, y, cv=kf)

        if self.monitor_std_dev:
            return scores.mean(), scores.std()
        return scores.mean(), None

    def _scale_data(self, X, y):
        scaler_X = StandardScaler().fit(X)
        scaler_y = StandardScaler().fit(y.values.reshape(-1, 1))

        X_scaled = pd.DataFrame(scaler_X.transform(X), columns=X.columns)
        y_scaled = scaler_y.transform(y.values.reshape(-1, 1)).flatten()
        
        return X_scaled, y_scaled, scaler_X, scaler_y

    def _get_best_model(self):
        sorted_models = sorted(self.models.values(), key=lambda x: x['score'], reverse=True)
        return sorted_models[0]

    def save_state(self, filename: str):
        data = {
            'models': self.models,
            'base_model': self.base_model,
            'original_data': self.original_data,
            'scaler_X': self.scaler_X,
            'scaler_y': self.scaler_y
        }
        dump(data, filename)

    @classmethod
    def load_state(cls, filename: str, dependent_var: str):
        loaded_data = load(filename)
        if not all(key in loaded_data for key in ['models', 'base_model', 'original_data', 'scaler_X', 'scaler_y']):
            raise ValueError("The saved state appears to be incomplete or corrupted.")

        stepwise_obj = cls(dependent_var=dependent_var)
        stepwise_obj.models = loaded_data['models']
        stepwise_obj.base_model = loaded_data['base_model']
        stepwise_obj.original_data = loaded_data['original_data']
        stepwise_obj.scaler_X = loaded_data['scaler_X']
        stepwise_obj.scaler_y = loaded_data['scaler_y']

        return stepwise_obj