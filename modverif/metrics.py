"""
Standardized meteorological verification metrics.
"""
from typing import Tuple

import numpy as np
from scipy.ndimage import uniform_filter

###################################################
### Parameters

# int16 range for clipping NE values
INT16_MIN = -32768
INT16_MAX = 32767

# Available domain-aggregated metrics
AVAILABLE_DOMAIN_METRICS = ('ne', 'ane', 'rmse', 'bias', 'pearson', 'pod', 'far', 'csi', 'gss', 'fbias')

# Available metrics (cell-level)
AVAILABLE_METRICS = ('ne', 'ane', 'rse', 'bias', 'mae', 'pod', 'far', 'csi', 'gss', 'fbias')

# Available station metrics (per-station over time)
AVAILABLE_STATION_METRICS = ('bias', 'mae', 'rmse', 'ne', 'ane', 'pearson')

# Available vector wind metrics
AVAILABLE_WIND_METRICS = ('vector_rmse', 'speed_bias', 'direction_bias')

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
        'mae': {
            'dtype': np.float32,
            'units': 'same as variable',
            'long_name': 'Mean Absolute Error',
            'standard_name': 'mean_absolute_error',
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
        'pearson': {
            'dtype': np.float64,
            'units': '1',
            'long_name': 'Domain-aggregated Pearson Correlation',
            'standard_name': 'domain_pearson_correlation',
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

def compute_mae(
    source_data: np.ndarray,
    test_data: np.ndarray,
) -> np.ndarray:
    """
    Compute Mean Absolute Error between source and test data.
    MAE = |test - source|
    """
    return np.abs(test_data - source_data).astype(np.float32)


def compute_pearson_correlation(
    source_data: np.ndarray,
    test_data: np.ndarray,
) -> float:
    """Compute Pearson correlation coefficient between two arrays."""
    if source_data.size < 2:
        return np.nan
    return np.corrcoef(source_data.flatten(), test_data.flatten())[0, 1]


def compute_mean_bias(model: np.ndarray, obs: np.ndarray) -> float:
    """Mean error averaged over time. model and obs are 1D (n_times,)."""
    return float(np.nanmean(model - obs))


def compute_mae_1d(model: np.ndarray, obs: np.ndarray) -> float:
    """Mean absolute error averaged over time."""
    return float(np.nanmean(np.abs(model - obs)))


def compute_rmse_1d(model: np.ndarray, obs: np.ndarray) -> float:
    """RMSE over a 1D time series."""
    return float(np.sqrt(np.nanmean((model - obs) ** 2)))


def compute_ne_1d(
    model: np.ndarray,
    obs: np.ndarray,
    epsilon: float = 1e-10,
) -> float:
    """Normalised error for 1D paired arrays (as percentage)."""
    obs_sum = np.nansum(obs)
    if np.abs(obs_sum) < epsilon:
        return 0.0
    return float((np.nansum(model) - obs_sum) / obs_sum * 100)


def compute_ane_1d(
    model: np.ndarray,
    obs: np.ndarray,
    epsilon: float = 1e-10,
) -> float:
    """Absolute normalised error for 1D paired arrays (as percentage)."""
    return abs(compute_ne_1d(model, obs, epsilon))


def compute_residual_skill_score(resid: np.ndarray, resid_base: np.ndarray) -> float:
    """
    Skill of one set of residuals against a baseline's, as a fraction of RMSE removed.

    ``1 - rmse(resid) / rmse(resid_base)``. Positive means the model beat the baseline; zero means it
    matched it; negative means the baseline was better, which is the outcome worth reporting rather
    than hiding.

    Two things to note before comparing this with a number from elsewhere:

    * It takes **residuals**, not ``(model, obs)`` pairs, unlike the rest of this module. The
      baseline is often something with no per-point prediction to subtract -- a leave-one-out mean,
      a climatology -- so residuals are the only common currency.
    * It is an **RMSE** ratio, not the more common MSE-ratio skill score (MSESS). The two are not
      interchangeable: an MSE ratio of 0.5 is an RMSE ratio of about 0.29.

    Parameters
    ----------
    resid : np.ndarray
        Residuals of the method under test.
    resid_base : np.ndarray
        Residuals of the reference method.

    Returns
    -------
    float
        Skill score, or NaN if the baseline has zero RMSE (nothing to improve on).
    """
    def _rmse(a):
        # Strict mean, not nanmean: a NaN here means the caller's residuals are wrong, and it should
        # surface as NaN rather than be silently dropped from the denominator.
        return float(np.sqrt(np.mean(np.asarray(a) ** 2)))

    rb = _rmse(resid_base)
    return (1.0 - _rmse(resid) / rb) if rb > 0 else np.nan


def compute_lagged_correlation(
    model: np.ndarray,
    obs: np.ndarray,
    max_lag: int = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cross-correlation between model and observation time series at multiple lags.

    A positive optimal lag means the model leads (event arrives early);
    a negative optimal lag means the model lags (event arrives late).

    Parameters
    ----------
    model : np.ndarray
        1D model time series (n_times,).
    obs : np.ndarray
        1D observation time series (n_times,).
    max_lag : int, optional
        Maximum lag (in timesteps) to evaluate in both directions.
        Default is n_times // 4.

    Returns
    -------
    lags : np.ndarray
        Integer lag values from -max_lag to +max_lag.
    correlations : np.ndarray
        Pearson correlation at each lag.
    """
    valid = ~(np.isnan(model) | np.isnan(obs))
    m = model[valid]
    o = obs[valid]
    n = len(m)

    if n < 3:
        lags = np.array([0])
        return lags, np.array([np.nan])

    if max_lag is None:
        max_lag = n // 4
    max_lag = min(max_lag, n - 2)

    lags = np.arange(-max_lag, max_lag + 1)
    correlations = np.full(len(lags), np.nan)

    for i, lag in enumerate(lags):
        if lag >= 0:
            m_slice = m[:n - lag]
            o_slice = o[lag:]
        else:
            m_slice = m[-lag:]
            o_slice = o[:n + lag]
        if len(m_slice) < 2:
            continue
        correlations[i] = np.corrcoef(m_slice, o_slice)[0, 1]

    return lags, correlations


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


def compute_pearson_domain(
    source_data: np.ndarray,
    test_data: np.ndarray,
    mask: np.ndarray = None,
) -> np.ndarray:
    """Pearson correlation over the spatial domain per timestep."""
    results = np.zeros(source_data.shape[0])
    for t in range(source_data.shape[0]):
        s = source_data[t].flatten()
        tt = test_data[t].flatten()
        if mask is not None:
            m = mask.flatten()
            s, tt = s[m], tt[m]
        results[t] = np.corrcoef(s, tt)[0, 1] if len(s) > 1 else np.nan
    return results

##################################################
### Fractions Skill Score (FSS)

def compute_fraction_field(
    binary_field: np.ndarray,
    neighborhood_size: int,
) -> np.ndarray:
    """
    Compute fraction of True cells within a square neighborhood.

    Parameters
    ----------
    binary_field : np.ndarray
        2D bool array (y, x).
    neighborhood_size : int
        Odd integer for neighborhood window size.

    Returns
    -------
    np.ndarray
        2D float array of fractions.
    """
    return uniform_filter(binary_field.astype(np.float64), size=neighborhood_size, mode='constant')


def compute_fss(
    source_data: np.ndarray,
    test_data: np.ndarray,
    threshold: float,
    neighborhood_size: int,
    mask: np.ndarray = None,
) -> float:
    """
    Compute Fractions Skill Score for a single 2D field.

    Parameters
    ----------
    source_data : np.ndarray
        2D reference field (y, x).
    test_data : np.ndarray
        2D forecast/test field (y, x).
    threshold : float
        Binary event threshold.
    neighborhood_size : int
        Odd integer for neighborhood window size.
    mask : np.ndarray, optional
        2D boolean mask. Only masked cells contribute.

    Returns
    -------
    float
        FSS value in [0, 1].
    """
    source_binary = source_data >= threshold
    test_binary = test_data >= threshold

    frac_source = compute_fraction_field(source_binary, neighborhood_size)
    frac_test = compute_fraction_field(test_binary, neighborhood_size)

    if mask is not None:
        frac_source = frac_source[mask]
        frac_test = frac_test[mask]
    else:
        frac_source = frac_source.flatten()
        frac_test = frac_test.flatten()

    mse_frac = np.mean((frac_test - frac_source) ** 2)
    mse_ref = np.mean(frac_test ** 2) + np.mean(frac_source ** 2)

    if mse_ref == 0.0:
        return 1.0
    return float(1.0 - mse_frac / mse_ref)


def compute_fss_multi_scale(
    source_data: np.ndarray,
    test_data: np.ndarray,
    threshold: float,
    neighborhood_sizes: list = None,
    mask: np.ndarray = None,
) -> dict:
    """
    Compute FSS across multiple neighborhood sizes.

    Parameters
    ----------
    source_data : np.ndarray
        2D reference field (y, x).
    test_data : np.ndarray
        2D forecast/test field (y, x).
    threshold : float
        Binary event threshold.
    neighborhood_sizes : list[int], optional
        Neighborhood sizes. Default: [1, 3, 5, 9, 17, 33, 65].
    mask : np.ndarray, optional
        2D boolean mask.

    Returns
    -------
    dict[int, float]
        Mapping neighborhood_size -> FSS.
    """
    if neighborhood_sizes is None:
        neighborhood_sizes = [1, 3, 5, 9, 17, 33, 65]
    return {n: compute_fss(source_data, test_data, threshold, n, mask) for n in neighborhood_sizes}


##################################################
### Vector Wind Metrics

def compute_vector_rmse(
    source_u: np.ndarray,
    source_v: np.ndarray,
    test_u: np.ndarray,
    test_v: np.ndarray,
) -> float:
    """
    Vector RMSE: sqrt(mean((du)^2 + (dv)^2)).

    Parameters
    ----------
    source_u, source_v : np.ndarray
        Reference U/V wind components (2D or 1D).
    test_u, test_v : np.ndarray
        Test U/V wind components (2D or 1D).

    Returns
    -------
    float
        Scalar vector RMSE.
    """
    du = (test_u - source_u).flatten()
    dv = (test_v - source_v).flatten()
    return float(np.sqrt(np.nanmean(du ** 2 + dv ** 2)))


def compute_wind_speed_bias(
    source_u: np.ndarray,
    source_v: np.ndarray,
    test_u: np.ndarray,
    test_v: np.ndarray,
) -> float:
    """
    Wind speed bias: mean(test_speed - source_speed).

    Returns
    -------
    float
        Mean wind speed bias.
    """
    source_speed = np.sqrt(source_u ** 2 + source_v ** 2).flatten()
    test_speed = np.sqrt(test_u ** 2 + test_v ** 2).flatten()
    return float(np.nanmean(test_speed - source_speed))


def compute_wind_direction_bias(
    source_u: np.ndarray,
    source_v: np.ndarray,
    test_u: np.ndarray,
    test_v: np.ndarray,
) -> float:
    """
    Mean angular difference in wind direction (degrees).
    Uses circular statistics to handle wraparound.

    Returns
    -------
    float
        Mean directional bias in degrees [-180, 180].
    """
    source_dir = np.arctan2(-source_u, -source_v)
    test_dir = np.arctan2(-test_u, -test_v)
    diff = test_dir - source_dir
    # Wrap to [-pi, pi]
    diff = np.arctan2(np.sin(diff), np.cos(diff))
    return float(np.nanmean(np.degrees(diff.flatten())))


##################################################
### Diurnal Cycle Analysis

def compute_diurnal_stats(
    times: np.ndarray,
    model: np.ndarray,
    obs: np.ndarray,
    metric: str = 'bias',
    utc_offset: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute a metric grouped by hour-of-day.

    Parameters
    ----------
    times : np.ndarray
        Array of datetime64 values.
    model : np.ndarray
        1D model values (n_times,).
    obs : np.ndarray
        1D observation values (n_times,).
    metric : str
        One of 'bias', 'rmse', 'mae', 'pearson'.
    utc_offset : float
        Hours to add to UTC to get local time.

    Returns
    -------
    hours : np.ndarray
        Integer array of shape (24,).
    values : np.ndarray
        Float64 array of shape (24,).
    """
    # Convert times to hour-of-day
    epoch = np.datetime64('1970-01-01T00:00:00')
    seconds = (times - epoch) / np.timedelta64(1, 's')
    hours_of_day = ((seconds / 3600 + utc_offset) % 24).astype(int)

    result_hours = np.arange(24)
    result_values = np.full(24, np.nan)

    for h in range(24):
        mask = hours_of_day == h
        if np.sum(mask) == 0:
            continue
        m = model[mask]
        o = obs[mask]
        valid = ~(np.isnan(m) | np.isnan(o))
        m, o = m[valid], o[valid]
        if len(m) == 0:
            continue
        if metric == 'bias':
            result_values[h] = np.mean(m - o)
        elif metric == 'rmse':
            result_values[h] = np.sqrt(np.mean((m - o) ** 2))
        elif metric == 'mae':
            result_values[h] = np.mean(np.abs(m - o))
        elif metric == 'pearson':
            result_values[h] = np.corrcoef(m, o)[0, 1] if len(m) > 1 else np.nan

    return result_hours, result_values


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
        if self.total > 0:
            hits_random = ((self.hits + self.misses) * (self.hits + self.false_alarms)) / self.total
        else:
            hits_random = 0
        num = self.hits - hits_random
        den = self.hits + self.false_alarms + self.misses - hits_random
        return num / den if den != 0 else np.nan

    def bias(self) -> float:
        """Frequency Bias. Bias = (Hits + False Alarms) / (Hits + Misses)."""
        denom = self.hits + self.misses
        return (self.hits + self.false_alarms) / denom if denom > 0 else np.nan
