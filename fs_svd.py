import numpy as np
import pandas as pd
from enum import Enum, auto
from typing import Union

class ScaleType(Enum):
    BOTH = auto()
    CENTER = auto()
    SCALE = auto()

def validate_inputs(matrix_data: Union[np.ndarray, pd.DataFrame], scale_input: ScaleType, n_singular_values: int):
    """
    Validate the input data and parameters for fs_svd function.
    
    Parameters:
    - matrix_data (Union[np.ndarray, pd.DataFrame]): Input matrix data for SVD computation.
    - scale_input (ScaleType): Desired scaling type (BOTH, CENTER, or SCALE).
    - n_singular_values (int): Desired number of singular values for the computation.

    Raises:
    - TypeError: If input data is not numpy.ndarray or pandas.DataFrame.
    - ValueError: If the data is not 2D or n_singular_values is invalid.
    """
    if not isinstance(matrix_data, (np.ndarray, pd.DataFrame)):
        raise TypeError("Input data must be of type numpy.ndarray or pandas.DataFrame.")
    if matrix_data.ndim != 2:
        raise ValueError("Input must be a matrix (2D data).")
    if n_singular_values is not None:
        if not isinstance(n_singular_values, int) or n_singular_values <= 0:
            raise ValueError("n_singular_values must be a positive integer.")

def center_matrix(matrix_data: np.ndarray) -> np.ndarray:
    """
    Center the input matrix by subtracting the mean.
    
    Parameters:
    - matrix_data (np.ndarray): Input matrix data to be centered.

    Returns:
    - np.ndarray: Centered matrix.
    """
    return matrix_data - np.mean(matrix_data, axis=0)

def scale_matrix(matrix_data: np.ndarray) -> np.ndarray:
    """
    Scale the input matrix to have unit variance.
    
    Parameters:
    - matrix_data (np.ndarray): Input matrix data to be scaled.

    Returns:
    - np.ndarray: Scaled matrix.
    """
    std_dev = np.std(matrix_data, axis=0)
    scaled = matrix_data / np.where(std_dev == 0, 1, std_dev)
    return scaled

def apply_scaling(matrix_data: np.ndarray, scale_input: ScaleType) -> np.ndarray:
    """
    Apply the desired scaling (centering and/or scaling) to the matrix.
    
    Parameters:
    - matrix_data (np.ndarray): Input matrix data to be scaled.
    - scale_input (ScaleType): Desired scaling type (BOTH, CENTER, or SCALE).

    Returns:
    - np.ndarray: Appropriately scaled matrix.

    Raises:
    - ValueError: If the scaling type is not recognized.
    """
    if scale_input == ScaleType.BOTH:
        return scale_matrix(center_matrix(matrix_data))
    elif scale_input == ScaleType.CENTER:
        return center_matrix(matrix_data)
    elif scale_input == ScaleType.SCALE:
        return scale_matrix(matrix_data)
    else:
        raise ValueError(f"Invalid scale_input value: {scale_input}")

def variance_explained(singular_values: np.ndarray) -> np.ndarray:
    """
    Compute the variance explained by the singular values.
    
    Parameters:
    - singular_values (np.ndarray): Array of singular values.

    Returns:
    - np.ndarray: Variance explained by each singular value.
    """
    return np.square(singular_values) / np.sum(np.square(singular_values))

def reduced_matrix(u: np.ndarray, s: np.ndarray, vh: np.ndarray) -> np.ndarray:
    """
    Get the matrix using reduced dimensions.
    
    Parameters:
    - u (np.ndarray): Left singular vectors from the SVD computation.
    - s (np.ndarray): Singular values from the SVD computation.
    - vh (np.ndarray): Right singular vectors from the SVD computation.

    Returns:
    - np.ndarray: Matrix in reduced dimensions.
    """
    return np.dot(u * s, vh)

def fs_svd(matrix_data: Union[np.ndarray, pd.DataFrame], scale_input: ScaleType = ScaleType.BOTH, n_singular_values: int = None, return_dtype: type = None, return_original_matrix: bool = False, nan_fill_method: str = None) -> dict:
    """
    Compute Singular Value Decomposition for a given matrix with optional scaling.
    
    Parameters:
    - matrix_data (Union[np.ndarray, pd.DataFrame]): Input matrix data for SVD computation.
    - scale_input (ScaleType, optional): Desired scaling type (BOTH, CENTER, or SCALE). Defaults to BOTH.
    - n_singular_values (int, optional): Desired number of singular values for the computation. Defaults to None.
    - return_dtype (type, optional): Desired type (pandas.DataFrame, numpy.ndarray) of the output data. Defaults to None.
    - return_original_matrix (bool, optional): If True, includes the original (or transformed) matrix in the output. Defaults to False.
    - nan_fill_method (str, optional): Method to handle NaN values ('mean' or 'zero'). Defaults to None.

    Returns:
    - dict: Dictionary containing the results of the SVD computation.

    Raises:
    - ValueError: If the return_dtype is not recognized or if there's an issue with the number of singular values.
    """
    original_dtype = type(matrix_data)
    if return_dtype not in [pd.DataFrame, np.ndarray, None]:
        raise ValueError("return_dtype must be either pandas.DataFrame, numpy.ndarray, or None.")

    return_dtype = return_dtype or original_dtype
    column_names, index_names = None, None
    
    # If DataFrame, retain column and index names
    if isinstance(matrix_data, pd.DataFrame):
        column_names = matrix_data.columns
        index_names = matrix_data.index
        matrix_data = matrix_data.values

    # Validate input parameters
    validate_inputs(matrix_data, scale_input, n_singular_values)
    
    # Ensure we're not modifying the original data
    matrix_data = matrix_data.copy()

    # Check for missing values
    if np.isnan(matrix_data).any():
        if nan_fill_method == 'mean':
            matrix_data = np.where(np.isnan(matrix_data), np.nanmean(matrix_data, axis=0), matrix_data)
        elif nan_fill_method == 'zero':
            matrix_data[np.isnan(matrix_data)] = 0
        else:
            raise ValueError("Input matrix contains missing values. Consider using the 'nan_fill_method' parameter to handle NaN values.")
    
    # Apply scaling
    matrix_data = apply_scaling(matrix_data, scale_input)
    
    # Compute SVD
    u, s, vh = np.linalg.svd(matrix_data, full_matrices=False)
    
    # Truncate SVD if needed
    if n_singular_values is not None:
        if n_singular_values > min(matrix_data.shape):
            raise ValueError(f"n_singular_values ({n_singular_values}) exceeds the number of singular values in the matrix ({min(matrix_data.shape)}).")
        
        s = s[:n_singular_values]
        u = u[:, :n_singular_values]
        vh = vh[:n_singular_values, :]
    
    result = {
        'singular_values': s,
        'left_singular_vectors': u,
        'right_singular_vectors': vh.T,
        'variance_explained': variance_explained(s)
    }

    if n_singular_values is not None:
        result['reduced_matrix'] = reduced_matrix(u, s, vh)
    if return_original_matrix:
        result['original_matrix'] = matrix_data

    # Convert back to DataFrame if the original input was a DataFrame
    if return_dtype == pd.DataFrame:
        result['singular_values'] = pd.Series(s)
        result['left_singular_vectors'] = pd.DataFrame(u, columns=range(1, len(s) + 1), index=index_names)
        result['right_singular_vectors'] = pd.DataFrame(vh.T, columns=column_names, index=range(1, len(s) + 1))

    return result