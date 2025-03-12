import numpy as np
from typing import Union
from utils.types import Real, assert_positive_real_number

def convert_db_to_lin(x: Union[np.ndarray, float], x_max: Real = None) -> np.ndarray:
    """Convert dB values to linear scale with numerical safeguards.
    
    Args:
        x: Input values in dB
        x_max: Optional maximum linear value for clipping (prevents overflow)
    
    Returns:
        Linear scale values
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Handle extreme values to prevent overflow/underflow
    with np.errstate(invalid='ignore', over='ignore'):
        lin = 10 ** (x / 20)
        
        # Clip unreasonably large values if requested
        if x_max is not None:
            assert_positive_real_number(x=x_max)
            lin = np.clip(lin, a_min=None, a_max=x_max)
            
        # Replace infs with max finite value
        lin = np.nan_to_num(lin, nan=0.0, posinf=np.nanmax(lin[lin != np.inf]))
    
    return lin

def convert_lin_to_db(x: Union[np.ndarray, float], 
                     x_min: Real = 1e-12, 
                     x_max: Real = None) -> np.ndarray:
    """Convert linear values to dB scale with numerical safeguards.
    
    Args:
        x: Input linear values
        x_min: Minimum value to prevent log(0) (default: 1e-12)
        x_max: Optional maximum value for clipping (prevents underflow)
    
    Returns:
        dB scale values
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Input validation and clipping
    assert_positive_real_number(x=x_min)
    if x_max is not None:
        assert_positive_real_number(x=x_max)
        x = np.clip(x, a_min=x_min, a_max=x_max)
    else:
        x = np.maximum(x, x_min)
    
    # Handle invalid values and compute dB
    with np.errstate(divide='ignore', invalid='ignore'):
        db = 20 * np.log10(x)
        
        # Replace invalid values
        db = np.nan_to_num(db, nan=-np.inf, posinf=None, neginf=-np.inf)
    
    return db