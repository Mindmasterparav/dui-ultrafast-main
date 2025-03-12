import warnings
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline


def compute_fwhm_line(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Full Width at Half Maximum (FWHM) of a line profile."""
    # Convert to 1D arrays and remove singleton dimensions
    y = np.asarray(y).squeeze().flatten()
    x = np.asarray(x).squeeze().flatten()

    # Validate input shapes
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError(
            f"Inputs must be 1D after squeezing. Got x: {x.shape}, y: {y.shape}"
        )
    if x.size != y.size:
        raise ValueError(
            f"x ({x.size} elements) and y ({y.size} elements) must have same length"
        )
    if x.size < 10:  # Minimum points for spline interpolation
        warnings.warn("Insufficient data points for accurate FWHM calculation")
        return np.nan

    # Handle edge cases
    y = np.nan_to_num(y, copy=False, nan=0.0, posinf=np.nanmax(y), neginf=np.nanmin(y))
    if np.all(y == 0):
        warnings.warn("All-zero line profile, returning NaN")
        return np.nan

    try:
        # Normalize y to [0, 1] for better numerical stability
        y_norm = y - np.min(y)
        y_norm = y_norm / np.max(y_norm)
        
        # Fit spline with boundary checks
        spl = InterpolatedUnivariateSpline(
            x=x, 
            y=y_norm, 
            k=min(3, len(x)-1),  # Ensure k <= len(x)-1
            ext='zeros'
        )
        
        # Find peak using spline derivatives
        spl_der = spl.derivative(n=1)
        x_roots = spl_der.roots()
        
        if x_roots.size == 0:
            # Fallback to max value if no roots found
            peak_idx = np.argmax(y_norm)
            x_max = x[peak_idx]
            y_max = y_norm[peak_idx]
        else:
            # Evaluate spline at derivative roots
            y_roots = spl(x_roots)
            peak_idx = np.argmax(y_roots)
            x_max = x_roots[peak_idx]
            y_max = y_roots[peak_idx]

        # Calculate FWHM using linear interpolation for better stability
        half_max = y_max / 2
        crossing_points = np.where(y_norm >= half_max)[0]
        
        if crossing_points.size < 2:
            warnings.warn("Insufficient crossings for FWHM calculation")
            return np.nan
            
        left_idx = max(0, crossing_points[0] - 1)
        right_idx = min(len(x)-1, crossing_points[-1] + 1)
        
        # Linear interpolation for precise crossing points
        x_left = np.interp(half_max, 
                         y_norm[left_idx:crossing_points[0]+1], 
                         x[left_idx:crossing_points[0]+1])
        x_right = np.interp(half_max, 
                          y_norm[crossing_points[-1]:right_idx+1][::-1], 
                          x[crossing_points[-1]:right_idx+1][::-1])

        return abs(x_right - x_left)

    except Exception as e:
        warnings.warn(f"FWHM computation failed: {str(e)}")
        return np.nan