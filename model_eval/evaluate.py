"""
Functions for evaluating WRF model outputs.
"""
import pathlib
from datetime import date, datetime
from typing import Union

import h5py
import numpy as np
import rechunkit

# from model_eval.cyclone import (
#     CyclonePosition,
#     _compute_sea_level_pressure,
#     _estimate_cyclone_radius,
#     _find_pressure_minimum,
#     _grid_distances_km,
#     _haversine_distance,
# )
from cyclone import (
    CyclonePosition,
    _compute_sea_level_pressure,
    _estimate_cyclone_radius,
    _find_pressure_minimum,
    _grid_distances_km,
    _haversine_distance,
)

# from model_eval.wrfio import NetCDF4Writer, WRFFile
from wrfio import NetCDF4Writer, WRFFile

###################################################
### Parameters

# int16 range for clipping NE values
INT16_MIN = -32768
INT16_MAX = 32767

# NetCDF4 dimension scale marker
DIMENSION_LIST = 'DIMENSION_LIST'
CLASS = 'CLASS'
NAME = 'NAME'
REFERENCE_LIST = 'REFERENCE_LIST'

# Available domain-aggregated metrics
AVAILABLE_DOMAIN_METRICS = ('ne', 'ane', 'rmse')

# Available metrics
AVAILABLE_METRICS = ('ne', 'ane', 'rse')

time_units_dict = {
    'M': 'months',
    'D': 'days',
    'h': 'hours',
    'm': 'minutes',
    's': 'seconds',
    'ms': 'milliseconds',
    'us': 'microseconds',
    'ns': 'nanoseconds',
    }

inv_time_units_dict = {value: key for key, value in time_units_dict.items()}

##################################################
### Functions


def parse_cf_dates(units, dtype_encoded):
    """

    """
    if ' since ' in units:
        freq, start_date = units.split(' since ')
        freq_code = inv_time_units_dict[freq]
        origin_date = np.datetime64(start_date, freq_code)
        unix_date = np.datetime64('1970-01-01', freq_code)
        # origin_diff = (unix_date - origin_date).astype(dtype_encoded)
        units = f'{freq} since {str(unix_date)}'
        if freq_code not in ('M', 'D', 'h', 'm'):
            dtype_encoded = np.dtype('int64')
        dtype_decoded = origin_date.dtype
    else:
        dtype_decoded = dtype_encoded
        origin_date = None

    return units, dtype_decoded, dtype_encoded, origin_date


def _get_metric_info(metric: str) -> dict:
    """
    Get metadata for a metric.

    Parameters
    ----------
    metric : str
        Metric name ('ne', 'ane', 'rse').

    Returns
    -------
    dict
        Dictionary with 'dtype', 'units', 'long_name', 'standard_name' keys.
    """
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
    }
    return info[metric]


def find_wrfout_files(
    folder: pathlib.Path,
    domain: int,
    start_date: date = None,
    end_date: date = None,
) -> dict[date, pathlib.Path]:
    """
    Scan a folder for WRF output files matching a specific domain and date range.

    Expects filenames in the format: wrfout_d{domain}_{date}_{time}
    For example: wrfout_d04_2020-09-30_00:00:00

    Parameters
    ----------
    folder : pathlib.Path
        Directory containing wrfout_* files.
    domain : int
        WRF domain number to filter (e.g., 4 for d04).
    start_date : date, optional
        Start date (inclusive). If None, no lower bound.
    end_date : date, optional
        End date (inclusive). If None, no upper bound.

    Returns
    -------
    dict[date, pathlib.Path]
        Mapping of run dates to file paths.
    """
    domain_pattern = f'd{domain:02d}'
    files = {}
    for file in folder.iterdir():
        if not file.is_file():
            continue
        if not file.name.startswith('wrfout_'):
            continue
        parts = file.name.split('_')
        if len(parts) < 4:
            continue
        domain_str = parts[1]
        date_str = parts[2]
        if domain_str != domain_pattern:
            continue
        try:
            run_date = date.fromisoformat(date_str)
        except ValueError:
            continue
        # Apply date filters
        if start_date is not None and run_date < start_date:
            continue
        if end_date is not None and run_date > end_date:
            continue
        files[run_date] = file
    return files


def compute_ne(
    source_data: np.ndarray,
    test_data: np.ndarray,
    epsilon: float = 1e-10,
) -> np.ndarray:
    """
    Compute normalised error between source and test data, returning int16.

    NE = ((test - source) / source) * 100

    Values are clipped to int16 range [-32768, 32767] percent.

    Parameters
    ----------
    source_data : np.ndarray
        Reference/baseline model data.
    test_data : np.ndarray
        Test model data to compare against source.
    epsilon : float
        Small value to avoid division by zero. Values where |source| < epsilon
        will have NE set to 0.

    Returns
    -------
    np.ndarray
        Normalised error as percentage (int16, clipped to ±32767%).
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        ne = ((test_data - source_data) / source_data) * 100

    # Handle division by zero/small values
    mask = np.abs(source_data) < epsilon
    ne[mask] = 0.0

    # Replace any remaining inf/nan and clip to int16 range
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

    Values are clipped to int16 range [0, 32767] percent.

    Parameters
    ----------
    source_data : np.ndarray
        Reference/baseline model data.
    test_data : np.ndarray
        Test model data to compare against source.
    epsilon : float
        Small value to avoid division by zero. Values where |source| < epsilon
        will have ANE set to 0.

    Returns
    -------
    np.ndarray
        Absolute normalised error as percentage (int16, clipped to 0-32767%).
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        ane = np.abs((test_data - source_data) / source_data) * 100

    # Handle division by zero/small values
    mask = np.abs(source_data) < epsilon
    ane[mask] = 0.0

    # Replace any remaining inf/nan and clip to int16 range (positive only)
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

    This is computed element-wise (per grid cell per timestep), preserving
    the original array shape. For a single RSE value over a region or time
    period, further aggregation would be needed.

    Parameters
    ----------
    source_data : np.ndarray
        Reference/baseline model data.
    test_data : np.ndarray
        Test model data to compare against source.

    Returns
    -------
    np.ndarray
        Root squared error in same units as input (float32).
    """
    rse = np.sqrt((test_data - source_data) ** 2)
    return rse.astype(np.float32)


##################################################
### Domain-aggregated metric functions


def compute_ne_domain(
    source_data: np.ndarray,
    test_data: np.ndarray,
    mask: np.ndarray = None,
    epsilon: float = 1e-10,
) -> np.ndarray:
    """
    Compute domain-aggregated normalised error for each timestep.

    NE_domain = ((sum(test) - sum(source)) / sum(source)) * 100

    This aggregates over the spatial domain first, then computes the
    normalised error. Useful when cell-by-cell comparison is inappropriate
    due to spatial alignment differences.

    Parameters
    ----------
    source_data : np.ndarray
        Reference/baseline model data with shape (time, y, x).
    test_data : np.ndarray
        Test model data with shape (time, y, x).
    mask : np.ndarray, optional
        2D boolean mask (y, x). True = include cell. If None, use all cells.
    epsilon : float
        Small value to avoid division by zero.

    Returns
    -------
    np.ndarray
        Normalised error as percentage for each timestep (float64, shape: (time,)).
    """
    if mask is not None:
        # Apply mask - set masked cells to 0 for summation
        source_masked = np.where(mask, source_data, 0.0)
        test_masked = np.where(mask, test_data, 0.0)
    else:
        source_masked = source_data
        test_masked = test_data

    # Sum over spatial dimensions (axes 1 and 2)
    source_sum = np.sum(source_masked, axis=(1, 2))
    test_sum = np.sum(test_masked, axis=(1, 2))

    # Compute normalised error
    with np.errstate(divide='ignore', invalid='ignore'):
        ne = ((test_sum - source_sum) / source_sum) * 100

    # Handle division by zero
    ne = np.where(np.abs(source_sum) < epsilon, 0.0, ne)
    ne = np.nan_to_num(ne, nan=0.0, posinf=0.0, neginf=0.0)

    return ne


def compute_ane_domain(
    source_data: np.ndarray,
    test_data: np.ndarray,
    mask: np.ndarray = None,
    epsilon: float = 1e-10,
) -> np.ndarray:
    """
    Compute domain-aggregated absolute normalised error for each timestep.

    ANE_domain = |((sum(test) - sum(source)) / sum(source))| * 100

    Parameters
    ----------
    source_data : np.ndarray
        Reference/baseline model data with shape (time, y, x).
    test_data : np.ndarray
        Test model data with shape (time, y, x).
    mask : np.ndarray, optional
        2D boolean mask (y, x). True = include cell. If None, use all cells.
    epsilon : float
        Small value to avoid division by zero.

    Returns
    -------
    np.ndarray
        Absolute normalised error as percentage for each timestep (float64, shape: (time,)).
    """
    ne = compute_ne_domain(source_data, test_data, mask, epsilon)
    return np.abs(ne)


def compute_rmse_domain(
    source_data: np.ndarray,
    test_data: np.ndarray,
    mask: np.ndarray = None,
) -> np.ndarray:
    """
    Compute domain-aggregated root mean square error for each timestep.

    RMSE_domain = sqrt(mean((test - source)^2))

    This computes the RMSE across all spatial cells at each timestep.

    Parameters
    ----------
    source_data : np.ndarray
        Reference/baseline model data with shape (time, y, x).
    test_data : np.ndarray
        Test model data with shape (time, y, x).
    mask : np.ndarray, optional
        2D boolean mask (y, x). True = include cell. If None, use all cells.

    Returns
    -------
    np.ndarray
        RMSE in same units as input for each timestep (float64, shape: (time,)).
    """
    squared_error = (test_data - source_data) ** 2

    if mask is not None:
        # Apply mask - only include masked cells in mean
        n_cells = np.sum(mask)
        squared_error_masked = np.where(mask, squared_error, 0.0)
        mse = np.sum(squared_error_masked, axis=(1, 2)) / n_cells
    else:
        mse = np.mean(squared_error, axis=(1, 2))

    rmse = np.sqrt(mse)
    return rmse


def _get_domain_metric_info(metric: str) -> dict:
    """
    Get metadata for a domain-aggregated metric.

    Parameters
    ----------
    metric : str
        Metric name ('ne', 'ane', 'rmse').

    Returns
    -------
    dict
        Dictionary with 'dtype', 'units', 'long_name', 'standard_name' keys.
    """
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
    }
    return info[metric]


def _make_netcdf4_dimension(h5file: h5py.File, name: str, size: int, data: np.ndarray = None) -> h5py.Dataset:
    """
    Create a NetCDF4-compliant dimension scale.

    Parameters
    ----------
    h5file : h5py.File
        Open HDF5 file handle.
    name : str
        Dimension name.
    size : int
        Dimension size.
    data : np.ndarray, optional
        Coordinate data. If None, creates an empty dimension.

    Returns
    -------
    h5py.Dataset
        The dimension scale dataset.
    """
    if data is not None:
        dim_ds = h5file.create_dataset(name, data=data)
    else:
        dim_ds = h5file.create_dataset(name, shape=(size,), dtype='f4')

    # Mark as dimension scale (NetCDF4 convention)
    dim_ds.attrs[CLASS] = np.bytes_('DIMENSION_SCALE')
    dim_ds.attrs[NAME] = np.bytes_(name)

    return dim_ds


def _attach_dimension_scales(dataset: h5py.Dataset, dim_datasets: list[h5py.Dataset]):
    """
    Attach dimension scales to a dataset (NetCDF4 convention).

    Uses h5py's dimension scale API directly rather than manually creating
    DIMENSION_LIST attributes.

    Parameters
    ----------
    dataset : h5py.Dataset
        The dataset to attach dimensions to.
    dim_datasets : list[h5py.Dataset]
        List of dimension scale datasets, one per dimension.
    """
    for i, dim_ds in enumerate(dim_datasets):
        dim_ds.make_scale(dim_ds.name.split('/')[-1])
        dataset.dims[i].attach_scale(dim_ds)


def _get_wrf_proj4(attrs: h5py.AttributeManager) -> str:
    """
    Construct a PROJ4 string from WRF global attributes.

    Parameters
    ----------
    attrs : h5py.AttributeManager
        Attributes from a WRF output file.

    Returns
    -------
    str
        PROJ4 string or None if projection unknown.
    """
    map_proj = attrs.get('MAP_PROJ')
    if map_proj is None:
        return None

    # Common WRF sphere radius
    r = attrs.get('EARTH_RADIUS', 6370000.0)
    proj_base = f"+a={r} +b={r} +no_defs"

    if map_proj == 1:  # Lambert Conformal
        truelat1 = attrs.get('TRUELAT1')
        truelat2 = attrs.get('TRUELAT2')
        stand_lon = attrs.get('STAND_LON')
        moad_cen_lat = attrs.get('MOAD_CEN_LAT')
        return f"+proj=lcc +lat_1={truelat1} +lat_2={truelat2} +lat_0={moad_cen_lat} +lon_0={stand_lon} {proj_base}"
    elif map_proj == 2:  # Polar Stereographic
        truelat1 = attrs.get('TRUELAT1')
        stand_lon = attrs.get('STAND_LON')
        return f"+proj=stere +lat_ts={truelat1} +lat_0=90 +lon_0={stand_lon} +k=1 +x_0=0 +y_0=0 {proj_base}"
    elif map_proj == 3:  # Mercator
        truelat1 = attrs.get('TRUELAT1')
        stand_lon = attrs.get('STAND_LON')
        return f"+proj=merc +lat_ts={truelat1} +lon_0={stand_lon} +x_0=0 +y_0=0 {proj_base}"
    elif map_proj == 6:  # Cylindrical Equidistant
        stand_lon = attrs.get('STAND_LON')
        moad_cen_lat = attrs.get('MOAD_CEN_LAT')
        return f"+proj=longlat +lon_0={stand_lon} +lat_0={moad_cen_lat} {proj_base}"

    return None


def _find_latlon_bounds(
    h5file: h5py.File,
    bounds: tuple[float, float, float, float],
) -> tuple[slice, slice]:
    """
    Find y, x index slices that correspond to lat/lon bounds.

    Parameters
    ----------
    h5file : h5py.File
        Open WRF HDF5 file containing XLAT and XLONG variables.
    bounds : tuple
        (min_lat, max_lat, min_lon, max_lon) defining the region of interest.

    Returns
    -------
    tuple[slice, slice]
        (y_slice, x_slice) for subsetting data to the region.

    Raises
    ------
    ValueError
        If XLAT/XLONG not found or no grid cells fall within bounds.
    """
    if 'XLAT' not in h5file or 'XLONG' not in h5file:
        raise ValueError("XLAT and XLONG variables required for lat/lon bounds subsetting")

    min_lat, max_lat, min_lon, max_lon = bounds

    # XLAT and XLONG are typically (time, y, x), use first timestep
    xlat = h5file['XLAT'][0, :, :]
    xlong = h5file['XLONG'][0, :, :]

    # Find cells within bounds
    mask = (xlat >= min_lat) & (xlat <= max_lat) & (xlong >= min_lon) & (xlong <= max_lon)

    if not np.any(mask):
        raise ValueError(
            f"No grid cells found within bounds: lat=[{min_lat}, {max_lat}], lon=[{min_lon}, {max_lon}]"
        )

    # Find bounding box of valid cells
    y_indices, x_indices = np.where(mask)
    y_slice = slice(y_indices.min(), y_indices.max() + 1)
    x_slice = slice(x_indices.min(), x_indices.max() + 1)

    return y_slice, x_slice


def evaluate_models_cell(
    source_folder: Union[str, pathlib.Path],
    test_folder: Union[str, pathlib.Path],
    output_path: Union[str, pathlib.Path],
    domain: int,
    variables: list[str],
    metrics: Union[str, list[str]] = 'ne',
    region: Union[tuple[float, float, float, float], np.ndarray, None] = None,
    start_date: Union[str, date] = None,
    end_date: Union[str, date] = None,
    epsilon: float = 1e-10,
    max_memory_bytes: int = 2**29,
) -> pathlib.Path:
    """
    Evaluate two WRF model runs by computing cell-by-cell error metrics.

    This function compares WRF output files (netCDF4/HDF5 format) from two model runs,
    computing one or more error metrics at each grid cell.

    Processing is done one timestep at a time (full spatial extent) using rechunkit
    to handle arbitrarily-chunked source files efficiently.

    Parameters
    ----------
    source_folder : str or pathlib.Path
        Path to folder containing source/reference WRF output files.
    test_folder : str or pathlib.Path
        Path to folder containing test WRF output files to evaluate.
    output_path : str or pathlib.Path
        Path for output NetCDF4 file containing evaluation results.
    domain : int
        WRF domain number to evaluate (e.g., 4 for d04 files).
    variables : list[str]
        List of WRF variable names to evaluate (e.g., ['T2', 'Q2', 'U10']).
    metrics : str or list[str]
        Metric(s) to compute. Available metrics:
        - 'ne': Normalised Error = ((test - source) / source) * 100 [int16, percent]
        - 'ane': Absolute Normalised Error = |NE| [int16, percent]
        - 'rse': Root Squared Error = sqrt((test - source)^2) [float32, same units]
        Can be a single string or list of strings. Default is 'ne'.
    region : tuple or np.ndarray, optional
        Spatial region to evaluate. Can be either:
        - tuple of 4 floats (min_lat, max_lat, min_lon, max_lon): Extract a rectangular
          region based on lat/lon bounds. Requires XLAT and XLONG in WRF files.
        - 2D numpy boolean array: Mask array where True indicates cells to include.
          Must match spatial dimensions (n_y, n_x). Masked cells are set to fill value.
        If None (default), evaluate the entire domain.
    start_date : str or date, optional
        Start date (inclusive) for evaluation period. Can be ISO format string
        (e.g., '2020-09-30') or date object. If None, no lower bound.
    end_date : str or date, optional
        End date (inclusive) for evaluation period. Can be ISO format string
        (e.g., '2020-10-15') or date object. If None, no upper bound.
    epsilon : float
        Small value to avoid division by zero in NE/ANE calculation.
    max_memory_bytes : int
        Maximum memory for rechunkit to use during chunk processing (default 512 MB).

    Returns
    -------
    pathlib.Path
        Path to the output file.

    Raises
    ------
    FileNotFoundError
        If source or test folder does not exist.
    ValueError
        If no matching files found, dates don't align, or invalid metric specified.

    Notes
    -----
    Output NetCDF4 structure:
        Dimensions: time, y, x
        Variables:
            /{variable}_{metric}  - error array with dimensions (time, y, x)
            e.g., T2_ne, T2_ane, T2_rse
    """
    source_folder = pathlib.Path(source_folder)
    test_folder = pathlib.Path(test_folder)
    output_path = pathlib.Path(output_path)

    # Normalize metrics to a list
    if isinstance(metrics, str):
        metrics = [metrics]
    metrics = [m.lower() for m in metrics]

    # Validate metrics
    for m in metrics:
        if m not in AVAILABLE_METRICS:
            raise ValueError(f"Unknown metric '{m}'. Available metrics: {AVAILABLE_METRICS}")

    # Parse date strings if provided
    if isinstance(start_date, str):
        start_date = date.fromisoformat(start_date)
    if isinstance(end_date, str):
        end_date = date.fromisoformat(end_date)

    if not source_folder.exists():
        raise FileNotFoundError(f"Source folder not found: {source_folder}")
    if not test_folder.exists():
        raise FileNotFoundError(f"Test folder not found: {test_folder}")

    # Find matching files within date range
    source_files = find_wrfout_files(source_folder, domain, start_date, end_date)
    test_files = find_wrfout_files(test_folder, domain, start_date, end_date)

    if not source_files:
        raise ValueError(f"No wrfout files found for domain {domain} in {source_folder}")
    if not test_files:
        raise ValueError(f"No wrfout files found for domain {domain} in {test_folder}")

    # Find common dates
    common_dates = sorted(set(source_files.keys()) & set(test_files.keys()))
    if not common_dates:
        raise ValueError("No common dates found between source and test folders")

    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Parse region parameter
    y_slice = slice(None)
    x_slice = slice(None)
    spatial_mask = None  # 2D boolean mask (after slicing)
    use_mask = False

    if region is not None:
        if isinstance(region, np.ndarray):
            # 2D mask array
            if region.ndim != 2:
                raise ValueError(f"Spatial mask must be 2D, got {region.ndim}D array")
            spatial_mask = region.astype(bool)
            use_mask = True
        elif isinstance(region, (list, tuple)) and len(region) == 4:
            # Lat/lon bounds - will resolve to slices after opening first file
            pass
        else:
            raise ValueError(
                "region must be either a tuple of 4 floats (min_lat, max_lat, min_lon, max_lon) "
                "or a 2D numpy boolean array"
            )

    # Scan all files to get shape info, total timesteps, and coordinate data
    n_times = 0
    n_y_full = None
    n_x_full = None
    n_y = None
    n_x = None
    dx = None
    dy = None
    proj4 = None
    all_times = []
    all_raw_times = []

    for run_date in common_dates:
        source_file = source_files[run_date]
        test_file = test_files[run_date]
        with h5py.File(source_file, 'r') as h5s, h5py.File(test_file, 'r') as h5t:
            # Validate variables exist (only need to check first file pair fully)
            if n_y_full is None:
                for var in variables:
                    if var not in h5s:
                        raise ValueError(f"Variable '{var}' not found in {source_file}")
                    if var not in h5t:
                        raise ValueError(f"Variable '{var}' not found in {test_file}")

            ref_var = h5s[variables[0]]
            test_var = h5t[variables[0]]

            if n_y_full is None:
                n_y_full = ref_var.shape[1]
                n_x_full = ref_var.shape[2]

                # Resolve region to slices/mask on first file
                if region is not None:
                    if isinstance(region, (list, tuple)) and len(region) == 4:
                        # Lat/lon bounds - find corresponding indices
                        y_slice, x_slice = _find_latlon_bounds(h5s, tuple(region))

                    if spatial_mask is not None:
                        # Validate mask shape matches full domain
                        if spatial_mask.shape != (n_y_full, n_x_full):
                            raise ValueError(
                                f"Mask shape {spatial_mask.shape} does not match "
                                f"domain shape ({n_y_full}, {n_x_full})"
                            )

                # Calculate output dimensions
                if y_slice != slice(None):
                    n_y = y_slice.stop - y_slice.start
                else:
                    n_y = n_y_full
                if x_slice != slice(None):
                    n_x = x_slice.stop - x_slice.start
                else:
                    n_x = n_x_full

                # Get grid spacing and projection from global attributes
                dx = h5s.attrs.get('DX')
                dy = h5s.attrs.get('DY')
                proj4 = _get_wrf_proj4(h5s.attrs)

            # Use minimum timesteps from source and test for this date
            file_n_times = min(ref_var.shape[0], test_var.shape[0])
            n_times += file_n_times

            # Collect times from source file
            if 'Times' in h5s:
                times_data = h5s['Times'][:file_n_times]
                all_raw_times.append(times_data)
                for t_row in times_data:
                    if isinstance(t_row, (bytes, str)):
                        t_str = t_row.decode('utf-8') if isinstance(t_row, bytes) else t_row
                    else:
                        t_str = b"".join(t_row).decode('utf-8')

                    t_str = t_str.replace('_', 'T')
                    try:
                        dt = np.datetime64(t_str)
                        # Convert to hours since 1970-01-01 for hourly data precision
                        hours = (dt - np.datetime64('1970-01-01')) / np.timedelta64(1, 'h')
                        all_times.append(hours)
                    except ValueError:
                        all_times.append(np.nan)

    time_values = np.array(all_times, dtype='f8') if all_times else None
    # Calculate coordinate arrays with correct offset for subsetted regions
    if dx is not None:
        x_start = x_slice.start if x_slice != slice(None) else 0
        x_values = ((np.arange(n_x) + x_start) * dx).astype('f4')
    else:
        x_values = None
    if dy is not None:
        y_start = y_slice.start if y_slice != slice(None) else 0
        y_values = ((np.arange(n_y) + y_start) * dy).astype('f4')
    else:
        y_values = None

    with NetCDF4Writer(output_path) as nc:
        # Set global attributes
        nc.set_global_attrs(
            source_folder=str(source_folder),
            test_folder=str(test_folder),
            domain=domain,
        )
        if proj4:
            nc.h5.attrs['proj4'] = np.bytes_(proj4)

        # Create dimensions
        time_ds = nc.create_time_dimension(n_times, data=time_values)
        y_ds, x_ds = nc.create_spatial_dimensions(n_y, n_x, y_data=y_values, x_data=x_values)
        dim_scales = [time_ds, y_ds, x_ds]

        # Store region information in output file attributes
        if region is not None:
            if isinstance(region, (list, tuple)) and len(region) == 4:
                nc.h5.attrs['region_type'] = np.bytes_('latlon_bounds')
                nc.h5.attrs['region_min_lat'] = region[0]
                nc.h5.attrs['region_max_lat'] = region[1]
                nc.h5.attrs['region_min_lon'] = region[2]
                nc.h5.attrs['region_max_lon'] = region[3]
            elif use_mask:
                nc.h5.attrs['region_type'] = np.bytes_('spatial_mask')
                # Store the mask in the output file
                mask_ds = nc.create_variable(
                    'spatial_mask',
                    shape=(n_y, n_x),
                    data=spatial_mask.astype(np.int8),
                    dtype='i1',
                    long_name='Spatial mask (1=included, 0=excluded)',
                    flag_values=np.array([0, 1], dtype=np.int8),
                    flag_meanings='excluded included',
                )
                nc.attach_scales(mask_ds, [y_ds, x_ds])

        # Create output datasets for each variable and metric combination
        out_datasets = {}
        for var in variables:
            for metric in metrics:
                metric_info = _get_metric_info(metric)
                ds_name = f'{var}_{metric}'

                # Determine fill value for masked data
                fill_value = None
                if use_mask:
                    if metric_info['dtype'] == np.float32:
                        fill_value = np.float32(np.nan)
                    else:
                        fill_value = np.iinfo(metric_info['dtype']).min

                out_ds = nc.create_variable(
                    ds_name,
                    shape=(n_times, n_y, n_x),
                    dtype=metric_info['dtype'],
                    units=metric_info['units'],
                    long_name=f"{metric_info['long_name']} for {var}",
                    standard_name=metric_info['standard_name'],
                    fill_value=fill_value,
                    chunks=(1, n_y, n_x),
                )
                nc.attach_scales(out_ds, dim_scales)
                out_datasets[(var, metric)] = out_ds

        # Process each variable
        for var in variables:
            time_offset = 0
            for run_date in common_dates:
                source_file = source_files[run_date]
                test_file = test_files[run_date]

                with h5py.File(source_file, 'r') as h5s, h5py.File(test_file, 'r') as h5t:
                    if var not in h5s:
                        raise ValueError(f"Variable '{var}' not found in {source_file}")
                    if var not in h5t:
                        raise ValueError(f"Variable '{var}' not found in {test_file}")

                    source_ds = h5s[var]
                    test_ds = h5t[var]

                    # Check spatial dimensions match
                    if source_ds.shape[1:] != test_ds.shape[1:]:
                        raise ValueError(
                            f"Spatial shape mismatch for {var} on {run_date.isoformat()}: "
                            f"source {source_ds.shape[1:]} vs test {test_ds.shape[1:]}"
                        )

                    # Use minimum number of timesteps from both files
                    n_timesteps = min(source_ds.shape[0], test_ds.shape[0])
                    shape = (n_timesteps, source_ds.shape[1], source_ds.shape[2])
                    source_chunks = source_ds.chunks or source_ds.shape
                    test_chunks = test_ds.chunks or test_ds.shape

                    # Target chunks: 1 timestep, full spatial extent
                    target_chunks = (1, shape[1], shape[2])

                    # Create rechunkers for source and test data (limited to common timesteps)
                    source_rechunker = rechunkit.rechunker(
                        lambda idx: source_ds[idx],
                        shape,
                        source_ds.dtype,
                        source_ds.dtype.itemsize,
                        source_chunks,
                        target_chunks,
                        max_memory_bytes,
                    )
                    test_rechunker = rechunkit.rechunker(
                        lambda idx: test_ds[idx],
                        shape,
                        test_ds.dtype,
                        test_ds.dtype.itemsize,
                        test_chunks,
                        target_chunks,
                        max_memory_bytes,
                    )

                    # Process each timestep
                    for (source_slices, source_data), (_, test_data) in zip(
                        source_rechunker, test_rechunker
                    ):
                        # Apply spatial subsetting if using lat/lon bounds
                        if y_slice != slice(None) or x_slice != slice(None):
                            source_data = source_data[:, y_slice, x_slice]
                            test_data = test_data[:, y_slice, x_slice]

                        # Adjust time slice for output position
                        out_time_start = time_offset + source_slices[0].start
                        out_time_stop = time_offset + source_slices[0].stop
                        out_slices = (slice(out_time_start, out_time_stop), slice(None), slice(None))

                        # Compute and store each metric
                        for metric in metrics:
                            if metric == 'ne':
                                result = compute_ne(source_data, test_data, epsilon)
                            elif metric == 'ane':
                                result = compute_ane(source_data, test_data, epsilon)
                            elif metric == 'rse':
                                result = compute_rse(source_data, test_data)

                            # Apply spatial mask if provided
                            if use_mask and spatial_mask is not None:
                                metric_info = _get_metric_info(metric)
                                if metric_info['dtype'] == np.float32:
                                    fill_value = np.nan
                                else:
                                    # Use minimum value for int16 as fill
                                    fill_value = np.iinfo(metric_info['dtype']).min
                                # Broadcast mask to match result shape (time, y, x)
                                result = np.where(spatial_mask, result, fill_value)

                            out_datasets[(var, metric)][out_slices] = result

                    time_offset += n_timesteps

    return output_path


def evaluate_models_domain(
    source_folder: Union[str, pathlib.Path],
    test_folder: Union[str, pathlib.Path],
    output_path: Union[str, pathlib.Path],
    domain: int,
    variables: list[str],
    metrics: Union[str, list[str]] = 'ne',
    region: Union[tuple[float, float, float, float], np.ndarray, None] = None,
    start_date: Union[str, date] = None,
    end_date: Union[str, date] = None,
    epsilon: float = 1e-10,
    max_memory_bytes: int = 2**29,
) -> pathlib.Path:
    """
    Evaluate two WRF model runs using domain-aggregated metrics.

    Unlike evaluate_models() which computes cell-by-cell metrics, this function
    first aggregates values over the spatial domain at each timestep, then
    computes the metric. This is useful when:

    - Cell-by-cell comparison is inappropriate due to spatial alignment differences
    - You want to compare bulk/integrated quantities (e.g., total precipitation)
    - You need a single metric value per timestep for time series analysis

    For example, domain-aggregated NE is computed as:
        NE = ((sum(test) - sum(source)) / sum(source)) * 100

    Parameters
    ----------
    source_folder : str or pathlib.Path
        Path to folder containing source/reference WRF output files.
    test_folder : str or pathlib.Path
        Path to folder containing test WRF output files to evaluate.
    output_path : str or pathlib.Path
        Path for output NetCDF4 file containing evaluation results.
    domain : int
        WRF domain number to evaluate (e.g., 4 for d04 files).
    variables : list[str]
        List of WRF variable names to evaluate (e.g., ['T2', 'Q2', 'U10']).
    metrics : str or list[str]
        Metric(s) to compute. Available metrics:
        - 'ne': Normalised Error = ((sum(test) - sum(source)) / sum(source)) * 100
        - 'ane': Absolute Normalised Error = |NE|
        - 'rmse': Root Mean Square Error = sqrt(mean((test - source)^2))
        Can be a single string or list of strings. Default is 'ne'.
    region : tuple or np.ndarray, optional
        Spatial region to aggregate over. Can be either:
        - tuple of 4 floats (min_lat, max_lat, min_lon, max_lon): Rectangular region.
        - 2D numpy boolean array: Mask where True indicates cells to include.
        If None (default), aggregate over the entire domain.
    start_date : str or date, optional
        Start date (inclusive) for evaluation period.
    end_date : str or date, optional
        End date (inclusive) for evaluation period.
    epsilon : float
        Small value to avoid division by zero in NE/ANE calculation.
    max_memory_bytes : int
        Maximum memory for rechunkit to use during chunk processing.

    Returns
    -------
    pathlib.Path
        Path to the output file.

    Notes
    -----
    Output NetCDF4 structure:
        Dimensions: time, metric
        Coordinates:
            time   - timestep coordinate (hours since 1970-01-01)
            metric - integer index (0, 1, 2, ...) with flag_meanings attribute
        Variables:
            /{variable}  - 2D array with dimensions (time, metric)
            e.g., T2[time, metric], RAINNC[time, metric]

    The metric coordinate uses integer indices for compatibility with tools
    like Panoply. The metric names are stored in the 'flag_meanings' attribute
    of the metric variable (e.g., "ne ane rmse").

    Example access with xarray:
        ds = xr.open_dataset('output.nc')
        ds['T2'].isel(metric=0)  # Get first metric (e.g., NE)
        ds['T2'].isel(metric=1)  # Get second metric (e.g., ANE)

        # Get metric names from attribute
        metric_names = ds['metric'].attrs['flag_meanings'].split()
    """
    source_folder = pathlib.Path(source_folder)
    test_folder = pathlib.Path(test_folder)
    output_path = pathlib.Path(output_path)

    # Normalize metrics to a list
    if isinstance(metrics, str):
        metrics = [metrics]
    metrics = [m.lower() for m in metrics]

    # Validate metrics
    for m in metrics:
        if m not in AVAILABLE_DOMAIN_METRICS:
            raise ValueError(f"Unknown metric '{m}'. Available domain metrics: {AVAILABLE_DOMAIN_METRICS}")

    # Parse date strings if provided
    if isinstance(start_date, str):
        start_date = date.fromisoformat(start_date)
    if isinstance(end_date, str):
        end_date = date.fromisoformat(end_date)

    if not source_folder.exists():
        raise FileNotFoundError(f"Source folder not found: {source_folder}")
    if not test_folder.exists():
        raise FileNotFoundError(f"Test folder not found: {test_folder}")

    # Find matching files within date range
    source_files = find_wrfout_files(source_folder, domain, start_date, end_date)
    test_files = find_wrfout_files(test_folder, domain, start_date, end_date)

    if not source_files:
        raise ValueError(f"No wrfout files found for domain {domain} in {source_folder}")
    if not test_files:
        raise ValueError(f"No wrfout files found for domain {domain} in {test_folder}")

    # Find common dates
    common_dates = sorted(set(source_files.keys()) & set(test_files.keys()))
    if not common_dates:
        raise ValueError("No common dates found between source and test folders")

    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Parse region parameter
    y_slice = slice(None)
    x_slice = slice(None)
    spatial_mask = None

    if region is not None:
        if isinstance(region, np.ndarray):
            if region.ndim != 2:
                raise ValueError(f"Spatial mask must be 2D, got {region.ndim}D array")
            spatial_mask = region.astype(bool)
        elif isinstance(region, (list, tuple)) and len(region) == 4:
            # Will resolve to slices after opening first file
            region = tuple(region)
        else:
            raise ValueError(
                "region must be either a tuple of 4 floats (min_lat, max_lat, min_lon, max_lon) "
                "or a 2D numpy boolean array"
            )

    # Scan all files to get shape info and total timesteps
    n_times = 0
    n_y_full = None
    n_x_full = None
    proj4 = None
    all_times = []

    for run_date in common_dates:
        source_file = source_files[run_date]
        test_file = test_files[run_date]
        with h5py.File(source_file, 'r') as h5s, h5py.File(test_file, 'r') as h5t:
            if n_y_full is None:
                for var in variables:
                    if var not in h5s:
                        raise ValueError(f"Variable '{var}' not found in {source_file}")
                    if var not in h5t:
                        raise ValueError(f"Variable '{var}' not found in {test_file}")

            ref_var = h5s[variables[0]]
            test_var = h5t[variables[0]]

            if n_y_full is None:
                n_y_full = ref_var.shape[1]
                n_x_full = ref_var.shape[2]

                # Resolve region to slices on first file
                if isinstance(region, tuple):
                    y_slice, x_slice = _find_latlon_bounds(h5s, region)

                elif spatial_mask is not None:
                    if spatial_mask.shape != (n_y_full, n_x_full):
                        raise ValueError(
                            f"Mask shape {spatial_mask.shape} does not match "
                            f"domain shape ({n_y_full}, {n_x_full})"
                        )

                # dx = h5s.attrs.get('DX')
                # dy = h5s.attrs.get('DY')
                proj4 = _get_wrf_proj4(h5s.attrs)

            file_n_times = min(ref_var.shape[0], test_var.shape[0])
            n_times += file_n_times

            # Collect times from source file
            if 'Times' in h5s:
                times_data = h5s['Times'][:file_n_times]
                for t_row in times_data:
                    if isinstance(t_row, (bytes, str)):
                        t_str = t_row.decode('utf-8') if isinstance(t_row, bytes) else t_row
                    else:
                        t_str = b"".join(t_row).decode('utf-8')
                    t_str = t_str.replace('_', 'T')
                    try:
                        dt = np.datetime64(t_str)
                        hours = (dt - np.datetime64('1970-01-01')) / np.timedelta64(1, 'h')
                        all_times.append(hours)
                    except ValueError:
                        all_times.append(np.nan)

    time_values = np.array(all_times, dtype='f8') if all_times else None

    with NetCDF4Writer(output_path) as nc:
        # Set global attributes
        nc.set_global_attrs(
            source_folder=str(source_folder),
            test_folder=str(test_folder),
            domain=domain,
            aggregation_type='domain',
        )
        if proj4:
            nc.h5.attrs['proj4'] = np.bytes_(proj4)

        # Store region information
        if region is not None:
            if isinstance(region, (list, tuple)) and len(region) == 4:
                nc.h5.attrs['region_type'] = np.bytes_('latlon_bounds')
                nc.h5.attrs['region_min_lat'] = region[0]
                nc.h5.attrs['region_max_lat'] = region[1]
                nc.h5.attrs['region_min_lon'] = region[2]
                nc.h5.attrs['region_max_lon'] = region[3]
            elif spatial_mask is not None:
                nc.h5.attrs['region_type'] = np.bytes_('spatial_mask')

        # Create dimensions
        time_ds = nc.create_time_dimension(n_times, data=time_values)
        metric_ds = nc.create_metric_dimension(metrics)

        # Build metric index lookup
        n_metrics = len(metrics)
        metric_indices = {m: i for i, m in enumerate(metrics)}

        # Create output datasets for each variable with shape (time, metric)
        out_datasets = {}
        for var in variables:
            out_ds = nc.create_variable(
                var,
                shape=(n_times, n_metrics),
                dtype='f4',
                long_name=f"Domain-aggregated evaluation metrics for {var}",
            )
            nc.attach_scales(out_ds, [time_ds, metric_ds])
            out_datasets[var] = out_ds

        # Process each variable
        for var in variables:
            time_offset = 0
            for run_date in common_dates:
                source_file = source_files[run_date]
                test_file = test_files[run_date]

                with h5py.File(source_file, 'r') as h5s, h5py.File(test_file, 'r') as h5t:
                    source_ds = h5s[var]
                    test_ds = h5t[var]

                    # Check spatial dimensions match
                    if source_ds.shape[1:] != test_ds.shape[1:]:
                        raise ValueError(
                            f"Spatial shape mismatch for {var} on {run_date.isoformat()}: "
                            f"source {source_ds.shape[1:]} vs test {test_ds.shape[1:]}"
                        )

                    n_timesteps = min(source_ds.shape[0], test_ds.shape[0])
                    shape = (n_timesteps, source_ds.shape[1], source_ds.shape[2])
                    source_chunks = source_ds.chunks or source_ds.shape
                    test_chunks = test_ds.chunks or test_ds.shape

                    # Target chunks: 1 timestep, full spatial extent
                    target_chunks = (1, shape[1], shape[2])

                    source_rechunker = rechunkit.rechunker(
                        lambda idx: source_ds[idx],
                        shape,
                        source_ds.dtype,
                        source_ds.dtype.itemsize,
                        source_chunks,
                        target_chunks,
                        max_memory_bytes,
                    )
                    test_rechunker = rechunkit.rechunker(
                        lambda idx: test_ds[idx],
                        shape,
                        test_ds.dtype,
                        test_ds.dtype.itemsize,
                        test_chunks,
                        target_chunks,
                        max_memory_bytes,
                    )

                    # Accumulate data for this file to compute domain metrics
                    for (source_slices, source_data), (_, test_data) in zip(
                        source_rechunker, test_rechunker
                    ):
                        # Apply spatial subsetting if using lat/lon bounds
                        if y_slice != slice(None) or x_slice != slice(None):
                            source_data = source_data[:, y_slice, x_slice]
                            test_data = test_data[:, y_slice, x_slice]

                        # Compute domain-aggregated metrics
                        out_time_start = time_offset + source_slices[0].start
                        out_time_stop = time_offset + source_slices[0].stop

                        for metric in metrics:
                            if metric == 'ne':
                                result = compute_ne_domain(source_data, test_data, spatial_mask, epsilon)
                            elif metric == 'ane':
                                result = compute_ane_domain(source_data, test_data, spatial_mask, epsilon)
                            elif metric == 'rmse':
                                result = compute_rmse_domain(source_data, test_data, spatial_mask)

                            metric_idx = metric_indices[metric]
                            out_datasets[var][out_time_start:out_time_stop, metric_idx] = result

                    time_offset += n_timesteps

    return output_path


def evaluate_cyclones(
    source_path: Union[str, pathlib.Path],
    test_path: Union[str, pathlib.Path],
    output_path: Union[str, pathlib.Path],
    variables: list[str],
    metrics: Union[str, list[str]] = 'ne',
    start_lat: float = None,
    start_lon: float = None,
    search_radius_km: float = 500.0,
    pressure_threshold_pa: float = 400.0,
    max_cyclone_radius_km: float = 1000.0,
    smoothing_sigma: float = None,
    epsilon: float = 1e-10,
) -> pathlib.Path:
    """
    Evaluate two WRF models containing the same cyclone.

    Tracks the cyclone independently in both source and test models, then computes
    domain-aggregated metrics over each model's own cyclone region. This allows
    comparison of cyclone characteristics even when positions differ.

    The function outputs:
    - Track data for both models (lat, lon, central_pressure, radius per timestep)
    - Track comparison metrics (position difference, pressure difference, radius difference)
    - Domain-aggregated evaluation metrics for specified variables

    Parameters
    ----------
    source_path : str or pathlib.Path
        Path to source/reference WRF output file.
    test_path : str or pathlib.Path
        Path to test WRF output file to evaluate.
    output_path : str or pathlib.Path
        Path for output NetCDF4 file.
    variables : list[str]
        List of WRF variable names to evaluate (e.g., ['RAINNC', 'T2', 'U10']).
    metrics : str or list[str]
        Metric(s) to compute. Available: 'ne', 'ane', 'rmse'. Default is 'ne'.
    start_lat : float, optional
        Initial search latitude for cyclone tracking. If None, uses global
        pressure minimum at t=0.
    start_lon : float, optional
        Initial search longitude for cyclone tracking.
    search_radius_km : float
        Radius in km to search for pressure minimum at each timestep.
        Default is 500 km.
    pressure_threshold_pa : float
        Pressure increase from center that defines cyclone edge.
        Default is 400 Pa (4 hPa).
    max_cyclone_radius_km : float
        Maximum cyclone radius to consider. Default is 1000 km.
    smoothing_sigma : float, optional
        Standard deviation for Gaussian smoothing of SLP field.
        If None, no smoothing is applied.
    epsilon : float
        Small value to avoid division by zero. Default is 1e-10.

    Returns
    -------
    pathlib.Path
        Path to the output file.

    Notes
    -----
    Output NetCDF4 structure:
        Dimensions: time, metric

        Cyclone track variables (shape: time):
            source_latitude, source_longitude, source_pressure, source_radius
            test_latitude, test_longitude, test_pressure, test_radius

        Track comparison variables (shape: time):
            position_difference_km  - Distance between cyclone centers
            pressure_difference     - Test pressure minus source pressure (Pa)
            radius_difference       - Test radius minus source radius (km)

        Evaluation variables (shape: time, metric):
            {variable}  - Domain-aggregated metrics for each variable

    The evaluation uses each model's own cyclone region:
    - Source data is aggregated over the source cyclone area
    - Test data is aggregated over the test cyclone area
    - Metrics compare these aggregated values
    """
    source_path = pathlib.Path(source_path)
    test_path = pathlib.Path(test_path)
    output_path = pathlib.Path(output_path)

    # Normalize metrics to a list
    if isinstance(metrics, str):
        metrics = [metrics]
    metrics = [m.lower() for m in metrics]

    for m in metrics:
        if m not in AVAILABLE_DOMAIN_METRICS:
            raise ValueError(f"Unknown metric '{m}'. Available: {AVAILABLE_DOMAIN_METRICS}")

    with WRFFile(source_path) as wrf_s, WRFFile(test_path) as wrf_t:
        # Validate required variables for SLP calculation
        required_vars = ['PSFC', 'HGT', 'T2', 'XLAT', 'XLONG']
        for var in required_vars:
            if not wrf_s.has_variable(var):
                raise ValueError(f"Required variable '{var}' not found in {source_path}")
            if not wrf_t.has_variable(var):
                raise ValueError(f"Required variable '{var}' not found in {test_path}")

        # Validate evaluation variables
        for var in variables:
            if not wrf_s.has_variable(var):
                raise ValueError(f"Variable '{var}' not found in {source_path}")
            if not wrf_t.has_variable(var):
                raise ValueError(f"Variable '{var}' not found in {test_path}")

        # Get dimensions
        n_times = min(wrf_s.n_times, wrf_t.n_times)

        # Get grids
        xlat_s = wrf_s.xlat
        xlong_s = wrf_s.xlong
        xlat_t = wrf_t.xlat
        xlong_t = wrf_t.xlong

        # Get time values
        time_values = wrf_s.time_values[:n_times] if wrf_s.time_values is not None else None

        # Track cyclones and compute metrics
        source_positions = []
        test_positions = []
        current_lat_s = start_lat
        current_lon_s = start_lon
        current_lat_t = start_lat
        current_lon_t = start_lon

        # Storage for metrics
        n_metrics = len(metrics)
        var_results = {var: np.zeros((n_times, n_metrics), dtype=np.float32) for var in variables}

        for t in range(n_times):
            # --- Track source cyclone ---
            slp_s = wrf_s.get_slp(t, smoothing_sigma=smoothing_sigma)

            if t == 0 and current_lat_s is None:
                y_idx_s, x_idx_s, min_p_s = _find_pressure_minimum(slp_s, xlat_s, xlong_s)
            else:
                y_idx_s, x_idx_s, min_p_s = _find_pressure_minimum(
                    slp_s, xlat_s, xlong_s,
                    search_lat=current_lat_s,
                    search_lon=current_lon_s,
                    search_radius_km=search_radius_km,
                )

            center_lat_s = float(xlat_s[y_idx_s, x_idx_s])
            center_lon_s = float(xlong_s[y_idx_s, x_idx_s])

            radius_s = _estimate_cyclone_radius(
                slp_s, xlat_s, xlong_s, y_idx_s, x_idx_s,
                pressure_threshold_pa=pressure_threshold_pa,
                max_radius_km=max_cyclone_radius_km,
            )

            source_positions.append(CyclonePosition(
                time_index=t,
                y_index=y_idx_s,
                x_index=x_idx_s,
                latitude=center_lat_s,
                longitude=center_lon_s,
                central_pressure=min_p_s,
                radius_km=radius_s,
            ))

            current_lat_s = center_lat_s
            current_lon_s = center_lon_s

            # --- Track test cyclone ---
            slp_t = wrf_t.get_slp(t, smoothing_sigma=smoothing_sigma)

            if t == 0 and current_lat_t is None:
                y_idx_t, x_idx_t, min_p_t = _find_pressure_minimum(slp_t, xlat_t, xlong_t)
            else:
                y_idx_t, x_idx_t, min_p_t = _find_pressure_minimum(
                    slp_t, xlat_t, xlong_t,
                    search_lat=current_lat_t,
                    search_lon=current_lon_t,
                    search_radius_km=search_radius_km,
                )

            center_lat_t = float(xlat_t[y_idx_t, x_idx_t])
            center_lon_t = float(xlong_t[y_idx_t, x_idx_t])

            radius_t = _estimate_cyclone_radius(
                slp_t, xlat_t, xlong_t, y_idx_t, x_idx_t,
                pressure_threshold_pa=pressure_threshold_pa,
                max_radius_km=max_cyclone_radius_km,
            )

            test_positions.append(CyclonePosition(
                time_index=t,
                y_index=y_idx_t,
                x_index=x_idx_t,
                latitude=center_lat_t,
                longitude=center_lon_t,
                central_pressure=min_p_t,
                radius_km=radius_t,
            ))

            current_lat_t = center_lat_t
            current_lon_t = center_lon_t

            # --- Create masks for cyclone regions ---
            distances_s = _grid_distances_km(xlat_s, xlong_s, center_lat_s, center_lon_s)
            mask_s = distances_s <= radius_s

            distances_t = _grid_distances_km(xlat_t, xlong_t, center_lat_t, center_lon_t)
            mask_t = distances_t <= radius_t

            # --- Compute metrics for each variable ---
            for var in variables:
                source_data = wrf_s.get_variable(var, t)
                test_data = wrf_t.get_variable(var, t)

                for m_idx, metric in enumerate(metrics):
                    if metric == 'ne':
                        # Aggregate each over its own mask, then compute NE
                        source_sum = np.sum(np.where(mask_s, source_data, 0.0))
                        test_sum = np.sum(np.where(mask_t, test_data, 0.0))
                        if np.abs(source_sum) < epsilon:
                            result = 0.0
                        else:
                            result = ((test_sum - source_sum) / source_sum) * 100
                    elif metric == 'ane':
                        source_sum = np.sum(np.where(mask_s, source_data, 0.0))
                        test_sum = np.sum(np.where(mask_t, test_data, 0.0))
                        if np.abs(source_sum) < epsilon:
                            result = 0.0
                        else:
                            result = np.abs((test_sum - source_sum) / source_sum) * 100
                    elif metric == 'rmse':
                        # For RMSE, compute mean over each region then difference
                        n_cells_s = np.sum(mask_s)
                        n_cells_t = np.sum(mask_t)
                        source_mean = np.sum(np.where(mask_s, source_data, 0.0)) / max(n_cells_s, 1)
                        test_mean = np.sum(np.where(mask_t, test_data, 0.0)) / max(n_cells_t, 1)
                        result = np.abs(test_mean - source_mean)

                    var_results[var][t, m_idx] = result

    # --- Write output file using NetCDF4Writer ---
    with NetCDF4Writer(output_path) as nc:
        nc.set_global_attrs(
            source_file=str(source_path),
            test_file=str(test_path),
            evaluation_type='cyclone',
        )

        # Create dimensions
        time_ds = nc.create_time_dimension(n_times, data=time_values)
        metric_ds = nc.create_metric_dimension(metrics)

        # --- Source track variables ---
        source_lat_ds = nc.create_variable(
            'source_latitude',
            shape=(n_times,),
            data=np.array([p.latitude for p in source_positions], dtype=np.float32),
            units='degrees_north',
            long_name='Source cyclone center latitude',
            compress=False,
        )
        nc.attach_scales(source_lat_ds, [time_ds])

        source_lon_ds = nc.create_variable(
            'source_longitude',
            shape=(n_times,),
            data=np.array([p.longitude for p in source_positions], dtype=np.float32),
            units='degrees_east',
            long_name='Source cyclone center longitude',
            compress=False,
        )
        nc.attach_scales(source_lon_ds, [time_ds])

        source_pressure_ds = nc.create_variable(
            'source_pressure',
            shape=(n_times,),
            data=np.array([p.central_pressure for p in source_positions], dtype=np.float32),
            units='Pa',
            long_name='Source cyclone central sea level pressure',
            compress=False,
        )
        nc.attach_scales(source_pressure_ds, [time_ds])

        source_radius_ds = nc.create_variable(
            'source_radius',
            shape=(n_times,),
            data=np.array([p.radius_km for p in source_positions], dtype=np.float32),
            units='km',
            long_name='Source cyclone radius',
            compress=False,
        )
        nc.attach_scales(source_radius_ds, [time_ds])

        # --- Test track variables ---
        test_lat_ds = nc.create_variable(
            'test_latitude',
            shape=(n_times,),
            data=np.array([p.latitude for p in test_positions], dtype=np.float32),
            units='degrees_north',
            long_name='Test cyclone center latitude',
            compress=False,
        )
        nc.attach_scales(test_lat_ds, [time_ds])

        test_lon_ds = nc.create_variable(
            'test_longitude',
            shape=(n_times,),
            data=np.array([p.longitude for p in test_positions], dtype=np.float32),
            units='degrees_east',
            long_name='Test cyclone center longitude',
            compress=False,
        )
        nc.attach_scales(test_lon_ds, [time_ds])

        test_pressure_ds = nc.create_variable(
            'test_pressure',
            shape=(n_times,),
            data=np.array([p.central_pressure for p in test_positions], dtype=np.float32),
            units='Pa',
            long_name='Test cyclone central sea level pressure',
            compress=False,
        )
        nc.attach_scales(test_pressure_ds, [time_ds])

        test_radius_ds = nc.create_variable(
            'test_radius',
            shape=(n_times,),
            data=np.array([p.radius_km for p in test_positions], dtype=np.float32),
            units='km',
            long_name='Test cyclone radius',
            compress=False,
        )
        nc.attach_scales(test_radius_ds, [time_ds])

        # --- Track comparison variables ---
        position_diff = np.array([
            _haversine_distance(
                source_positions[t].latitude, source_positions[t].longitude,
                test_positions[t].latitude, test_positions[t].longitude
            )
            for t in range(n_times)
        ], dtype=np.float32)

        pos_diff_ds = nc.create_variable(
            'position_difference_km',
            shape=(n_times,),
            data=position_diff,
            units='km',
            long_name='Distance between cyclone centers',
            compress=False,
        )
        nc.attach_scales(pos_diff_ds, [time_ds])

        pressure_diff = np.array([
            test_positions[t].central_pressure - source_positions[t].central_pressure
            for t in range(n_times)
        ], dtype=np.float32)

        pressure_diff_ds = nc.create_variable(
            'pressure_difference',
            shape=(n_times,),
            data=pressure_diff,
            units='Pa',
            long_name='Test minus source central pressure',
            compress=False,
        )
        nc.attach_scales(pressure_diff_ds, [time_ds])

        radius_diff = np.array([
            test_positions[t].radius_km - source_positions[t].radius_km
            for t in range(n_times)
        ], dtype=np.float32)

        radius_diff_ds = nc.create_variable(
            'radius_difference',
            shape=(n_times,),
            data=radius_diff,
            units='km',
            long_name='Test minus source cyclone radius',
            compress=False,
        )
        nc.attach_scales(radius_diff_ds, [time_ds])

        # --- Evaluation metric variables ---
        for var in variables:
            out_ds = nc.create_variable(
                var,
                shape=(n_times, n_metrics),
                data=var_results[var],
                long_name=f'Cyclone-region aggregated metrics for {var}',
            )
            nc.attach_scales(out_ds, [time_ds, metric_ds])

    return output_path