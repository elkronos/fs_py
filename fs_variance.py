import numpy as np
import pandas as pd
import warnings
from typing import Union

def fs_variance(data: Union[np.ndarray, pd.DataFrame], threshold: float, 
                return_type: str = 'array') -> Union[np.ndarray, pd.DataFrame, None]:
    """
    Apply variance thresholding to a numeric dataset.

    This function applies variance thresholding to a numeric dataset, removing features
    whose variances are not above a certain threshold. The function raises a warning 
    if no features have variance above the threshold and returns None. It will also
    raise an exception if the inputs are not as expected.

    Parameters:
    -----------
    data : Union[numpy.ndarray, pd.DataFrame]
        A numeric dataset. All columns should be numeric.

    threshold : float
        A non-negative value specifying the variance threshold. Features
        with variances above this threshold are kept in the dataset.

    return_type : str, optional (default='array')
        Return type of the processed data. It can be 'array' or 'dataframe'. 
        If 'dataframe' and input is a numpy array, an error is raised.
        Raises a ValueError if the return_type is not 'array' or 'dataframe'.

    Returns:
    --------
    Union[numpy.ndarray, pd.DataFrame, None]
        A numeric array or DataFrame containing the thresholded data. 
        If no features have variance above the threshold, the function returns None.

    Raises:
    -------
    ValueError
        - If threshold is not a non-negative numeric value.
        - If not all columns of the DataFrame are numeric.
        - If return_type is 'dataframe' and the input data is a numpy array.
        - If return_type is not 'array' or 'dataframe'.
        - If data is neither a numpy array nor a DataFrame.
    """

    # Check threshold input
    if not isinstance(threshold, (int, float)) or threshold < 0:
        raise ValueError("Threshold should be a non-negative numeric value.")
    
    # Check return type
    if return_type not in ['array', 'dataframe']:
        raise ValueError("return_type should be either 'array' or 'dataframe'.")
    
    data_is_dataframe = isinstance(data, pd.DataFrame)  # Store this to avoid re-checking later
    
    # Check data input and ensure all columns are numeric for DataFrame
    if data_is_dataframe:
        if not all(data.dtypes.apply(np.issubdtype, args=(np.number,))):
            raise ValueError("All columns of DataFrame should be numeric.")
        
        if return_type == 'array':
            data = data.values
    elif isinstance(data, np.ndarray):
        if return_type == 'dataframe':
            raise ValueError("Cannot return DataFrame when input is numpy array.")
    else:
        raise ValueError("Data should be a numpy array or a DataFrame.")
    
    # Calculate the variances for each feature
    variances = np.var(data, axis=0)
    
    # Identify the columns/features with variances above the threshold
    valid_features = variances > threshold
    
    # Check if any variances exceed the threshold
    if not valid_features.any():
        warnings.warn("No features have variance above the threshold.")
        return None
    
    # Retain features with variances above the threshold
    thresholded_data = data[:, valid_features]
    
    # Convert back to DataFrame if necessary
    if return_type == 'dataframe' and data_is_dataframe:
        thresholded_data = pd.DataFrame(thresholded_data, columns=np.array(data.columns)[valid_features])
    
    return thresholded_data
