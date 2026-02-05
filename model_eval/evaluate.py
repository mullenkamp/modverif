"""
Functions for evaluating WRF model outputs.
"""
import pathlib
from datetime import date, datetime
from typing import Union

import h5py
import numpy as np
import rechunkit

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


def evaluate_models(
    source_folder: Union[str, pathlib.Path],
    test_folder: Union[str, pathlib.Path],
    output_path: Union[str, pathlib.Path],
    domain: int,
    variables: list[str],
    metrics: Union[str, list[str]] = 'ne',
    start_date: Union[str, date] = None,
    end_date: Union[str, date] = None,
    epsilon: float = 1e-10,
    max_memory_bytes: int = 2**29,
) -> pathlib.Path:
    """
    Evaluate two WRF model runs by computing error metrics for each variable and timestep.

    This function compares WRF output files (netCDF4/HDF5 format) from two model runs,
    computing one or more error metrics.

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

    # Scan all files to get shape info, total timesteps, and coordinate data
    n_times = 0
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
            if n_y is None:
                for var in variables:
                    if var not in h5s:
                        raise ValueError(f"Variable '{var}' not found in {source_file}")
                    if var not in h5t:
                        raise ValueError(f"Variable '{var}' not found in {test_file}")

            ref_var = h5s[variables[0]]
            test_var = h5t[variables[0]]

            if n_y is None:
                n_y = ref_var.shape[1]
                n_x = ref_var.shape[2]
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
    x_values = (np.arange(n_x) * dx).astype('f4') if dx is not None else None
    y_values = (np.arange(n_y) * dy).astype('f4') if dy is not None else None
    raw_times_values = np.concatenate(all_raw_times, axis=0) if all_raw_times else None

    with h5py.File(output_path, 'w') as h5out:
        # Set NetCDF4 conventions
        h5out.attrs['Conventions'] = np.bytes_('CF-1.8')
        h5out.attrs['history'] = np.bytes_(f'Created {datetime.now().isoformat()} by model_eval')
        h5out.attrs['source_folder'] = np.bytes_(str(source_folder))
        h5out.attrs['test_folder'] = np.bytes_(str(test_folder))
        h5out.attrs['domain'] = domain
        if proj4:
            h5out.attrs['proj4'] = np.bytes_(proj4)

        # Create dimensions as NetCDF4 dimension scales
        time_ds = _make_netcdf4_dimension(h5out, 'time', n_times, data=time_values)
        time_ds.attrs['units'] = np.bytes_('hours since 1970-01-01')
        time_ds.attrs['calendar'] = np.bytes_('proleptic_gregorian')
        time_ds.attrs['standard_name'] = np.bytes_('time')

        if raw_times_values is not None:
            # Create WRF-style Times variable
            if raw_times_values.ndim == 2:
                char_dim_size = raw_times_values.shape[1]
                if 'DateStrLen' not in h5out:
                    _make_netcdf4_dimension(h5out, 'DateStrLen', char_dim_size)
                h5out.create_dataset('Times', data=raw_times_values, dtype='S1')
            else:
                h5out.create_dataset('Times', data=raw_times_values)

        y_ds = _make_netcdf4_dimension(h5out, 'y', n_y, data=y_values)
        y_ds.attrs['standard_name'] = np.bytes_('projection_y_coordinate')
        y_ds.attrs['units'] = np.bytes_('m')

        x_ds = _make_netcdf4_dimension(h5out, 'x', n_x, data=x_values)
        x_ds.attrs['standard_name'] = np.bytes_('projection_x_coordinate')
        x_ds.attrs['units'] = np.bytes_('m')

        dim_scales = [time_ds, y_ds, x_ds]

        # Create output datasets for each variable and metric combination
        out_datasets = {}
        for var in variables:
            for metric in metrics:
                metric_info = _get_metric_info(metric)
                ds_name = f'{var}_{metric}'
                out_ds = h5out.create_dataset(
                    ds_name,
                    shape=(n_times, n_y, n_x),
                    dtype=metric_info['dtype'],
                    chunks=(1, n_y, n_x),
                    compression='gzip',
                    compression_opts=4,
                )
                out_ds.attrs['units'] = np.bytes_(metric_info['units'])
                out_ds.attrs['long_name'] = np.bytes_(f"{metric_info['long_name']} for {var}")
                out_ds.attrs['standard_name'] = np.bytes_(metric_info['standard_name'])

                # Attach dimension scales
                _attach_dimension_scales(out_ds, dim_scales)

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
                        # Adjust time slice for output position
                        out_time_start = time_offset + source_slices[0].start
                        out_time_stop = time_offset + source_slices[0].stop
                        out_slices = (slice(out_time_start, out_time_stop), source_slices[1], source_slices[2])

                        # Compute and store each metric
                        for metric in metrics:
                            if metric == 'ne':
                                result = compute_ne(source_data, test_data, epsilon)
                            elif metric == 'ane':
                                result = compute_ane(source_data, test_data, epsilon)
                            elif metric == 'rse':
                                result = compute_rse(source_data, test_data)

                            out_datasets[(var, metric)][out_slices] = result

                    time_offset += n_timesteps

    return output_path