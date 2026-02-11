"""
Standardized meteorological verification metrics.
"""
import numpy as np
from typing import Optional, Tuple

###################################################
### Parameters

# int16 range for clipping NE values
INT16_MIN = -32768
INT16_MAX = 32767

# Available domain-aggregated metrics
AVAILABLE_DOMAIN_METRICS = ('ne', 'ane', 'rmse', 'bias', 'pod', 'far', 'csi', 'gss', 'fbias')

# Available metrics
AVAILABLE_METRICS = ('ne', 'ane', 'rse', 'bias', 'pod', 'far', 'csi', 'gss', 'fbias')

# Available metrics mapping for info retrieval
# (Moved from evaluate.py to keep metadata with implementation)

def _get_metric_info(metric: str) -> dict:
    """Get metadata for a metric."""
    info = {
        'ne': {
            'dtype': np.int16,
            'units': 'percent',
            'long_name': 'Normalised Error',
            'standard_name': 'normalised_error',
        },
        'ane': {
            'dtype': np.int16,
            'units': 'percent',
            'long_name': 'Mean Absolute Normalised Error',
            'standard_name': 'mean_absolute_normalised_error',
        },
        'rse': {
            'dtype': np.float32,
            'units': 'same as variable',
            'long_name': 'Root Mean Square Error',
            'standard_name': 'root_mean_square_error',
        },
        'bias': {
            'dtype': np.float32,
            'units': 'same as variable',
            'long_name': 'Mean Error (Bias)',
            'standard_name': 'mean_error',
        },
        'pod': {
            'dtype': np.float32,
            'units': '1',
            'long_name': 'Probability of Detection',
            'standard_name': 'probability_of_detection',
        },
        'far': {
            'dtype': np.float32,
            'units': '1',
            'long_name': 'False Alarm Ratio',
            'standard_name': 'false_alarm_ratio',
        },
        'csi': {
            'dtype': np.float32,
            'units': '1',
            'long_name': 'Critical Success Index',
            'standard_name': 'critical_success_index',
        },
        'gss': {
            'dtype': np.float32,
            'units': '1',
            'long_name': 'Gilbert Skill Score',
            'standard_name': 'gilbert_skill_score',
        },
        'fbias': {
            'dtype': np.float32,
            'units': '1',
            'long_name': 'Frequency Bias',
            'standard_name': 'frequency_bias',
        },
    }
    return info.get(metric, {})

def _get_domain_metric_info(metric: str) -> dict:
    """Get metadata for a domain-aggregated metric."""
    info = {
        'ne': {
            'dtype': np.float64,
            'units': 'percent',
            'long_name': 'Domain-aggregated Normalised Error',
            'standard_name': 'domain_normalised_error',
        },
        'ane': {
            'dtype': np.float64,
            'units': 'percent',
            'long_name': 'Domain-aggregated Absolute Normalised Error',
            'standard_name': 'domain_absolute_normalised_error',
        },
        'rmse': {
            'dtype': np.float64,
            'units': 'same as variable',
            'long_name': 'Domain-aggregated Root Mean Square Error',
            'standard_name': 'domain_root_mean_square_error',
        },
        'bias': {
            'dtype': np.float64,
            'units': 'same as variable',
            'long_name': 'Domain-aggregated Mean Error (Bias)',
            'standard_name': 'domain_mean_error',
        },
        'pod': {
            'dtype': np.float64,
            'units': '1',
            'long_name': 'Domain-aggregated Probability of Detection',
            'standard_name': 'domain_probability_of_detection',
        },
        'far': {
            'dtype': np.float64,
            'units': '1',
            'long_name': 'Domain-aggregated False Alarm Ratio',
            'standard_name': 'domain_false_alarm_ratio',
        },
        'csi': {
            'dtype': np.float64,
            'units': '1',
            'long_name': 'Domain-aggregated Critical Success Index',
            'standard_name': 'domain_critical_success_index',
        },
        'gss': {
            'dtype': np.float64,
            'units': '1',
            'long_name': 'Domain-aggregated Gilbert Skill Score',
            'standard_name': 'domain_gilbert_skill_score',
        },
        'fbias': {
            'dtype': np.float64,
            'units': '1',
            'long_name': 'Domain-aggregated Frequency Bias',
            'standard_name': 'domain_frequency_bias',
        },
    }
    return info.get(metric, {})

##################################################
### Continuous Metrics

def compute_ne(
    source_data: np.ndarray,
    test_data: np.ndarray,
    epsilon: float = 1e-10,
) -> np.ndarray:
    """
    Compute normalised error between source and test data, returning int16.
    NE = ((test - source) / source) * 100
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        ne = ((test_data - source_data) / source_data) * 100

    mask = np.abs(source_data) < epsilon
    ne[mask] = 0.0

    ne = np.nan_to_num(ne, nan=0.0, posinf=INT16_MAX, neginf=INT16_MIN)
    ne = np.clip(ne, INT16_MIN, INT16_MAX)

    return np.round(ne).astype(np.int16)

def compute_ane(
    source_data: np.ndarray,
    test_data: np.ndarray,
    epsilon: float = 1e-10,
) -> np.ndarray:
    """
    Compute absolute normalised error between source and test data, returning int16.
    ANE = |((test - source) / source)| * 100
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        ane = np.abs((test_data - source_data) / source_data) * 100

    mask = np.abs(source_data) < epsilon
    ane[mask] = 0.0

    ane = np.nan_to_num(ane, nan=0.0, posinf=INT16_MAX, neginf=0.0)
    ane = np.clip(ane, 0, INT16_MAX)

    return np.round(ane).astype(np.int16)

def compute_rse(
    source_data: np.ndarray,
    test_data: np.ndarray,
) -> np.ndarray:
    """
    Compute root squared error between source and test data.
    RSE = sqrt((test - source)^2)
    """
    rse = np.sqrt((test_data - source_data) ** 2)
    return rse.astype(np.float32)

def compute_bias(
    source_data: np.ndarray,
    test_data: np.ndarray,
) -> np.ndarray:
    """
    Compute Mean Error (Bias) between source and test data.
    Bias = test - source
    """
    return (test_data - source_data).astype(np.float32)

def compute_pearson_correlation(
    source_data: np.ndarray,
    test_data: np.ndarray,
) -> float:
    """Compute Pearson correlation coefficient between two arrays."""
    if source_data.size < 2:
        return np.nan
    return np.corrcoef(source_data.flatten(), test_data.flatten())[0, 1]

##################################################
### Domain-aggregated metric functions

def compute_ne_domain(
    source_data: np.ndarray,
    test_data: np.ndarray,
    mask: np.ndarray = None,
    epsilon: float = 1e-10,
) -> np.ndarray:
    """Compute domain-aggregated normalised error for each timestep."""
    if mask is not None:
        source_masked = np.where(mask, source_data, 0.0)
        test_masked = np.where(mask, test_data, 0.0)
    else:
        source_masked = source_data
        test_masked = test_data

    source_sum = np.sum(source_masked, axis=(1, 2))
    test_sum = np.sum(test_masked, axis=(1, 2))

    with np.errstate(divide='ignore', invalid='ignore'):
        ne = ((test_sum - source_sum) / source_sum) * 100

    ne = np.where(np.abs(source_sum) < epsilon, 0.0, ne)
    ne = np.nan_to_num(ne, nan=0.0, posinf=0.0, neginf=0.0)

    return ne

def compute_ane_domain(
    source_data: np.ndarray,
    test_data: np.ndarray,
    mask: np.ndarray = None,
    epsilon: float = 1e-10,
) -> np.ndarray:
    """Compute domain-aggregated absolute normalised error for each timestep."""
    ne = compute_ne_domain(source_data, test_data, mask, epsilon)
    return np.abs(ne)

def compute_rmse_domain(
    source_data: np.ndarray,
    test_data: np.ndarray,
    mask: np.ndarray = None,
) -> np.ndarray:
    """Compute domain-aggregated root mean square error for each timestep."""
    squared_error = (test_data - source_data) ** 2

    if mask is not None:
        n_cells = np.sum(mask)
        squared_error_masked = np.where(mask, squared_error, 0.0)
        mse = np.sum(squared_error_masked, axis=(1, 2)) / n_cells
    else:
        mse = np.mean(squared_error, axis=(1, 2))

    rmse = np.sqrt(mse)
    return rmse

def compute_bias_domain(
    source_data: np.ndarray,
    test_data: np.ndarray,
    mask: np.ndarray = None,
) -> np.ndarray:
    """Compute domain-aggregated Mean Error (Bias) for each timestep."""
    error = test_data - source_data
    if mask is not None:
        n_cells = np.sum(mask)
        error_masked = np.where(mask, error, 0.0)
        bias = np.sum(error_masked, axis=(1, 2)) / n_cells
    else:
        bias = np.mean(error, axis=(1, 2))
    return bias

##################################################
### Categorical / Contingency Table Metrics

class ContingencyTable:
    """
    A class representing a 2x2 contingency table for binary events.
    Matches MET Grid-Stat/Point-Stat output categories.
    """
    def __init__(self, hits: float, false_alarms: float, misses: float, correct_negatives: float):
        self.hits = hits  # A
        self.false_alarms = false_alarms  # B
        self.misses = misses  # C
        self.correct_negatives = correct_negatives  # D
        self.total = hits + false_alarms + misses + correct_negatives

    @classmethod
    def from_data(cls, source_data: np.ndarray, test_data: np.ndarray, threshold: float):
        """Create a contingency table from forecast and observation data using a threshold."""
        source_yes = source_data >= threshold
        test_yes = test_data >= threshold
        
        hits = np.sum(source_yes & test_yes)
        false_alarms = np.sum((~source_yes) & test_yes)
        misses = np.sum(source_yes & (~test_yes))
        correct_negatives = np.sum((~source_yes) & (~test_yes))
        
        return cls(hits, false_alarms, misses, correct_negatives)

    def pod(self) -> float:
        """Probability of Detection (Hit Rate). POD = Hits / (Hits + Misses)."""
        denom = self.hits + self.misses
        return self.hits / denom if denom > 0 else np.nan

    def far(self) -> float:
        """False Alarm Ratio. FAR = False Alarms / (Hits + False Alarms)."""
        denom = self.hits + self.false_alarms
        return self.false_alarms / denom if denom > 0 else np.nan

    def csi(self) -> float:
        """Critical Success Index (Threat Score). CSI = Hits / (Hits + False Alarms + Misses)."""
        denom = self.hits + self.false_alarms + self.misses
        return self.hits / denom if denom > 0 else np.nan

    def gss(self) -> float:
        """
        Gilbert Skill Score (Equitable Threat Score).
        GSS = (Hits - Hits_random) / (Hits + False Alarms + Misses - Hits_random)
        where Hits_random = (Hits + Misses) * (Hits + False Alarms) / Total
        """
        hits_random = ((self.hits + self.misses) * (self.hits + self.false_alarms)) / self.total if self.total > 0 else 0
        num = self.hits - hits_random
        den = self.hits + self.false_alarms + self.misses - hits_random
        return num / den if den != 0 else np.nan

    def bias(self) -> float:
        """Frequency Bias. Bias = (Hits + False Alarms) / (Hits + Misses)."""
        denom = self.hits + self.misses
        return (self.hits + self.false_alarms) / denom if denom > 0 else np.nan
