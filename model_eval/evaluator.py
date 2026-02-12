"""
Centralized orchestrator for model evaluation.
"""
import pathlib
from datetime import date
from typing import Union, Optional, List, Tuple

import h5py
import numpy as np
import rechunkit

from model_eval.metrics import (
    AVAILABLE_DOMAIN_METRICS,
    AVAILABLE_METRICS,
    ContingencyTable,
    _get_domain_metric_info,
    _get_metric_info,
    compute_ane,
    compute_ane_domain,
    compute_bias,
    compute_bias_domain,
    compute_ne,
    compute_ne_domain,
    compute_rmse_domain,
    compute_rse,
)
from model_eval.wrfio import NetCDF4Writer

def find_wrfout_files(
    folder: pathlib.Path,
    domain: int,
    start_date: date = None,
    end_date: date = None,
) -> dict[date, pathlib.Path]:
    """
    Scan a folder for WRF output files matching a specific domain and date range.
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

def _get_wrf_proj4(attrs) -> str:
    """Construct a PROJ4 string from WRF global attributes."""
    map_proj = attrs.get('MAP_PROJ')
    if map_proj is None:
        return None

    r = attrs.get('EARTH_RADIUS', 6370000.0)
    proj_base = f"+a={r} +b={r} +no_defs"

    if map_proj == 1:  # Lambert Conformal
        return f"+proj=lcc +lat_1={attrs.get('TRUELAT1')} +lat_2={attrs.get('TRUELAT2')} +lat_0={attrs.get('MOAD_CEN_LAT')} +lon_0={attrs.get('STAND_LON')} {proj_base}"
    elif map_proj == 2:  # Polar Stereographic
        return f"+proj=stere +lat_ts={attrs.get('TRUELAT1')} +lat_0=90 +lon_0={attrs.get('STAND_LON')} +k=1 +x_0=0 +y_0=0 {proj_base}"
    elif map_proj == 3:  # Mercator
        return f"+proj=merc +lat_ts={attrs.get('TRUELAT1')} +lon_0={attrs.get('STAND_LON')} +x_0=0 +y_0=0 {proj_base}"
    elif map_proj == 6:  # Cylindrical Equidistant
        return f"+proj=longlat +lon_0={attrs.get('STAND_LON')} +lat_0={attrs.get('MOAD_CEN_LAT')} {proj_base}"

    return None

def _find_latlon_bounds(h5file, bounds: Tuple[float, float, float, float]) -> Tuple[slice, slice]:
    """Find y, x index slices that correspond to lat/lon bounds."""
    if 'XLAT' not in h5file or 'XLONG' not in h5file:
        raise ValueError("XLAT and XLONG variables required for lat/lon bounds subsetting")

    min_lat, max_lat, min_lon, max_lon = bounds
    xlat = h5file['XLAT'][0, :, :]
    xlong = h5file['XLONG'][0, :, :]

    mask = (xlat >= min_lat) & (xlat <= max_lat) & (xlong >= min_lon) & (xlong <= max_lon)
    if not np.any(mask):
        raise ValueError(f"No grid cells found within bounds: lat=[{min_lat}, {max_lat}], lon=[{min_lon}, {max_lon}]")

    y_indices, x_indices = np.where(mask)
    return slice(y_indices.min(), y_indices.max() + 1), slice(x_indices.min(), x_indices.max() + 1)

class WRFEvaluator:
    """
    Orchestrates the evaluation of WRF model outputs between two or more runs.

    This class centralizes the setup phase of model evaluation, including file discovery,
    temporal alignment (finding common dates/times), and spatial subsetting (lat/lon bounds
    or boolean masks). It provides a unified processing engine that handles large-scale
    data efficiently using rechunking.

    Attributes
    ----------
    source_dir : pathlib.Path
        Directory containing reference WRF output files.
    test_dir : pathlib.Path
        Directory containing test WRF output files.
    domain : int
        WRF domain number to evaluate (e.g., 4 for d04).
    region : tuple or np.ndarray, optional
        Spatial region to evaluate. Can be either:
        - tuple of 4 floats (min_lat, max_lat, min_lon, max_lon): Extract a rectangular
          region based on lat/lon bounds. Requires XLAT and XLONG in WRF files.
        - 2D numpy boolean array: Mask array where True indicates cells to include.
          Must match spatial dimensions (n_y, n_x).
        If None (default), evaluate the entire domain.
    start_date : str or date, optional
        Start date (inclusive) for evaluation period. ISO format string or date object.
    end_date : str or date, optional
        End date (inclusive) for evaluation period. ISO format string or date object.
    """

    def __init__(
        self,
        source_dir: Union[str, pathlib.Path],
        test_dir: Union[str, pathlib.Path],
        domain: int,
        region: Union[Tuple[float, float, float, float], np.ndarray, None] = None,
        start_date: Union[str, date, None] = None,
        end_date: Union[str, date, None] = None,
    ):
        """
        Initialize the evaluation context.

        Parameters
        ----------
        source_dir : str or pathlib.Path
            Path to folder containing source/reference WRF output files.
        test_dir : str or pathlib.Path
            Path to folder containing test WRF output files to evaluate.
        domain : int
            WRF domain number to filter (e.g., 4 for d04).
        region : tuple or np.ndarray, optional
            Spatial region to evaluate. Can be either:
            - tuple of 4 floats (min_lat, max_lat, min_lon, max_lon): Extract a rectangular
              region based on lat/lon bounds. Requires XLAT and XLONG in WRF files.
            - 2D numpy boolean array: Mask array where True indicates cells to include.
              Must match spatial dimensions (n_y, n_x).
            If None (default), evaluate the entire domain.
        start_date : str or date, optional
            Start date (inclusive) for evaluation period. ISO format string or date object.
        end_date : str or date, optional
            End date (inclusive) for evaluation period. ISO format string or date object.

        Raises
        ------
        FileNotFoundError
            If source or test folder does not exist.
        ValueError
            If no matching files found, no common dates found, or invalid region provided.
        """
        self.source_dir = pathlib.Path(source_dir)
        self.test_dir = pathlib.Path(test_dir)
        self.domain = domain
        self.region = region
        
        # Parse date strings if provided
        if isinstance(start_date, str):
            start_date = date.fromisoformat(start_date)
        if isinstance(end_date, str):
            end_date = date.fromisoformat(end_date)
            
        if not self.source_dir.exists():
            raise FileNotFoundError(f"Source folder not found: {self.source_folder}")
        if not self.test_dir.exists():
            raise FileNotFoundError(f"Test folder not found: {self.test_folder}")

        # 1. Discover and match files
        self.source_files = find_wrfout_files(self.source_dir, domain, start_date, end_date)
        self.test_files = find_wrfout_files(self.test_dir, domain, start_date, end_date)

        if not self.source_files:
            raise ValueError(f"No wrfout files found for domain {domain} in {source_dir}")
        if not self.test_files:
            raise ValueError(f"No wrfout files found for domain {domain} in {test_dir}")

        self.common_dates = sorted(set(self.source_files.keys()) & set(self.test_files.keys()))
        if not self.common_dates:
            raise ValueError("No common dates found between source and test folders")

        # 2. Extract shared metadata and resolve spatial context using the first file
        self._initialize_context()

    def _initialize_context(self):
        """Pre-calculates slices, masks, and metadata from the first common file pair."""
        first_date = self.common_dates[0]
        with h5py.File(self.source_files[first_date], 'r') as h5s:
            # Grid spacing and projection
            self.dx = h5s.attrs.get('DX')
            self.dy = h5s.attrs.get('DY')
            self.proj4 = _get_wrf_proj4(h5s.attrs)
            
            # Dimensions
            for k in h5s.keys():
                if isinstance(h5s[k], h5py.Dataset) and h5s[k].ndim >= 3:
                    self.n_y_full = h5s[k].shape[1]
                    self.n_x_full = h5s[k].shape[2]
                    break

            # Resolve region to slices/mask
            self.y_slice = slice(None)
            self.x_slice = slice(None)
            self.spatial_mask = None
            self.use_mask = False

            if self.region is not None:
                if isinstance(self.region, np.ndarray):
                    if self.region.ndim != 2:
                        raise ValueError(f"Spatial mask must be 2D, got {self.region.ndim}D array")
                    self.spatial_mask = self.region.astype(bool)
                    self.use_mask = True
                    if self.spatial_mask.shape != (self.n_y_full, self.n_x_full):
                        raise ValueError(f"Mask shape {self.spatial_mask.shape} does not match domain shape ({self.n_y_full}, {self.n_x_full})")
                elif isinstance(self.region, (list, tuple)) and len(self.region) == 4:
                    self.y_slice, self.x_slice = _find_latlon_bounds(h5s, tuple(self.region))
                else:
                    raise ValueError("region must be either a tuple of 4 floats (min_lat, max_lat, min_lon, max_lon) or a 2D numpy boolean array")

            # Final output dimensions
            self.n_y = (self.y_slice.stop - self.y_slice.start) if self.y_slice != slice(None) else self.n_y_full
            self.n_x = (self.x_slice.stop - self.x_slice.start) if self.x_slice != slice(None) else self.n_x_full

        # 3. Aggregate temporal info across all files
        self.time_values, self.n_times = self._collect_temporal_metadata()

    def _collect_temporal_metadata(self):
        """Scans all files to calculate total timesteps and coordinate values."""
        all_times = []
        n_total_times = 0
        for run_date in self.common_dates:
            with h5py.File(self.source_files[run_date], 'r') as h5s, h5py.File(self.test_files[run_date], 'r') as h5t:
                for any_var in h5s.keys():
                    if isinstance(h5s[any_var], h5py.Dataset) and h5s[any_var].ndim >= 3:
                        break
                
                file_n_times = min(h5s[any_var].shape[0], h5t[any_var].shape[0])
                n_total_times += file_n_times
                
                if 'Times' in h5s:
                    times_data = h5s['Times'][:file_n_times]
                    for t_row in times_data:
                        t_str = t_row.decode('utf-8') if isinstance(t_row, bytes) else b"".join(t_row).decode('utf-8')
                        t_str = t_str.replace('_', 'T')
                        try:
                            dt = np.datetime64(t_str)
                            hours = (dt - np.datetime64('1970-01-01')) / np.timedelta64(1, 'h')
                            all_times.append(hours)
                        except ValueError:
                            all_times.append(np.nan)
        
        return np.array(all_times, dtype='f8') if all_times else None, n_total_times

    def evaluate_cell(
        self, 
        output_path: Union[str, pathlib.Path], 
        variables: List[str], 
        metrics: Union[str, List[str]] = 'ne', 
        threshold: float = None,
        epsilon: float = 1e-10,
        max_memory_bytes: int = 2**29
    ) -> pathlib.Path:
        """
        Perform high-resolution cell-by-cell spatial evaluation.

        Compares WRF output files from two model runs, computing error metrics at each 
        individual grid cell. Results are stored in a 3D NetCDF4 file (time, y, x).

        Parameters
        ----------
        output_path : str or pathlib.Path
            Path for output NetCDF4 file containing evaluation results.
        variables : list[str]
            List of WRF variable names to evaluate (e.g., ['T2', 'U10']).
        metrics : str or list[str]
            Metric(s) to compute. 
            Continuous: 'ne', 'ane', 'rse', 'bias'.
            Categorical (requires threshold): 'pod', 'far', 'csi', 'fbias'.
            Can be a single string or list of strings. Default is 'ne'.
        threshold : float, optional
            Threshold value used for categorical metrics. Required if 'pod', 'far', etc. are used.
        epsilon : float
            Small value to avoid division by zero in normalized metrics (NE/ANE).
        max_memory_bytes : int
            Maximum memory for rechunkit to use during chunk processing (default 512 MB).

        Returns
        -------
        pathlib.Path
            Path to the generated output NetCDF4 file.

        Notes
        -----
        Output NetCDF4 structure:
            Dimensions: time, y, x
            Variables: /{variable}_{metric} (time, y, x)
        """
        return self._run_engine(output_path, variables, metrics, threshold, epsilon, max_memory_bytes, agg_type='cell')

    def evaluate_domain(
        self, 
        output_path: Union[str, pathlib.Path], 
        variables: List[str], 
        metrics: Union[str, List[str]] = 'ne', 
        threshold: float = None,
        epsilon: float = 1e-10,
        max_memory_bytes: int = 2**29
    ) -> pathlib.Path:
        """
        Perform domain-aggregated evaluation for time series analysis.

        First aggregates values over the spatial domain (or sub-region) at each timestep, 
        then computes the metric. This provides a single metric value per timestep, 
        useful for bulk performance analysis and time series comparisons.

        Parameters
        ----------
        output_path : str or pathlib.Path
            Path for output NetCDF4 file containing evaluation results.
        variables : list[str]
            List of WRF variable names to evaluate.
        metrics : str or list[str]
            Metric(s) to compute. 
            Continuous: 'ne', 'ane', 'rmse', 'bias'.
            Categorical (requires threshold): 'pod', 'far', 'csi', 'gss', 'fbias'.
            Can be a single string or list of strings. Default is 'ne'.
        threshold : float, optional
            Threshold value used for categorical metrics. Required if 'pod', 'far', etc. are used.
        epsilon : float
            Small value to avoid division by zero.
        max_memory_bytes : int
            Maximum memory for rechunkit to use during chunk processing.

        Returns
        -------
        pathlib.Path
            Path to the generated output NetCDF4 file.

        Notes
        -----
        Output NetCDF4 structure:
            Dimensions: time, metric
            Variables: /{variable} (time, metric)
            The 'metric' coordinate stores the name of each calculated metric.
        """
        return self._run_engine(output_path, variables, metrics, threshold, epsilon, max_memory_bytes, agg_type='domain')

    def _run_engine(self, output_path, variables, metrics, threshold, epsilon, max_memory_bytes, agg_type):
        """
        The unified processing core for both cell and domain evaluations.

        This internal method handles the heavy lifting:
        1. Validates requested metrics against the aggregation type.
        2. Prepares spatial coordinate arrays (X/Y) with correct offsets.
        3. Initializes the output NetCDF4 file with global metadata and dimensions.
        4. Streams data from WRF files using rechunkit for memory efficiency.
        5. Dispatches data to specific metric functions in model_eval.metrics.
        6. Writes results to the output file timestep-by-timestep.

        Parameters
        ----------
        output_path : pathlib.Path
            Path to the output file.
        variables : list[str]
            List of WRF variables to process.
        metrics : list[str]
            List of metrics to calculate.
        threshold : float or None
            Value for categorical thresholding.
        epsilon : float
            Small value for division safety.
        max_memory_bytes : int
            Memory limit for rechunking operations.
        agg_type : str
            Either 'cell' or 'domain'.
        """
        output_path = pathlib.Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(variables, str):
            variables = [variables]

        if isinstance(metrics, str):
            metrics = [metrics]
        metrics = [m.lower() for m in metrics]

        # Validate metrics
        allowed = AVAILABLE_METRICS if agg_type == 'cell' else AVAILABLE_DOMAIN_METRICS
        for m in metrics:
            if m not in allowed:
                raise ValueError(f"Unknown metric '{m}'. Available metrics: {allowed}")

        # Prepare coordinate arrays
        x_values = ((np.arange(self.n_x) + (self.x_slice.start or 0)) * self.dx).astype('f4') if self.dx else None
        y_values = ((np.arange(self.n_y) + (self.y_slice.start or 0)) * self.dy).astype('f4') if self.dy else None

        with NetCDF4Writer(output_path) as nc:
            # Global Attributes
            nc.set_global_attrs(
                source_folder=str(self.source_dir),
                test_folder=str(self.test_dir),
                domain=self.domain,
                aggregation_type=agg_type
            )
            if self.proj4:
                nc.h5.attrs['proj4'] = np.bytes_(self.proj4)
            
            if self.region is not None:
                if isinstance(self.region, (list, tuple)) and len(self.region) == 4:
                    nc.h5.attrs['region_type'] = np.bytes_('latlon_bounds')
                    nc.h5.attrs['region_min_lat'] = self.region[0]
                    nc.h5.attrs['region_max_lat'] = self.region[1]
                    nc.h5.attrs['region_min_lon'] = self.region[2]
                    nc.h5.attrs['region_max_lon'] = self.region[3]
                elif self.use_mask:
                    nc.h5.attrs['region_type'] = np.bytes_('spatial_mask')

            # Dimensions
            time_ds = nc.create_time_dimension(self.n_times, data=self.time_values)
            if agg_type == 'cell':
                y_ds, x_ds = nc.create_spatial_dimensions(self.n_y, self.n_x, y_data=y_values, x_data=x_values)
                dim_scales = [time_ds, y_ds, x_ds]
            else:
                metric_ds = nc.create_metric_dimension(metrics)
                dim_scales = [time_ds, metric_ds]

            # Variable Creation (and mask creation if needed)
            if self.use_mask:
                # Store the mask in the output file
                mask_ds = nc.create_variable(
                    'spatial_mask',
                    shape=(self.n_y, self.n_x),
                    data=self.spatial_mask.astype(np.int8),
                    dtype='i1',
                    long_name='Spatial mask (1=included, 0=excluded)',
                    flag_values=np.array([0, 1], dtype=np.int8),
                    flag_meanings='excluded included',
                )
                # Coordinates for mask
                if agg_type == 'cell':
                    nc.attach_scales(mask_ds, [dim_scales[1], dim_scales[2]])

            out_datasets = {}
            for var in variables:
                if agg_type == 'cell':
                    for metric in metrics:
                        info = _get_metric_info(metric)
                        fill = np.nan if info['dtype'] == np.float32 else np.iinfo(info['dtype']).min if self.use_mask else None
                        ds = nc.create_variable(
                            f"{var}_{metric}", shape=(self.n_times, self.n_y, self.n_x),
                            dtype=info['dtype'], units=info['units'], long_name=f"{info['long_name']} for {var}",
                            standard_name=info['standard_name'], fill_value=fill, chunks=(1, self.n_y, self.n_x)
                        )
                        nc.attach_scales(ds, dim_scales)
                        out_datasets[(var, metric)] = ds
                else:
                    ds = nc.create_variable(
                        var, shape=(self.n_times, len(metrics)), dtype='f4',
                        long_name=f"Domain-aggregated metrics for {var}"
                    )
                    nc.attach_scales(ds, dim_scales)
                    out_datasets[var] = ds

            # Main processing loop
            for var in variables:
                time_offset = 0
                for run_date in self.common_dates:
                    with h5py.File(self.source_files[run_date], 'r') as h5s, h5py.File(self.test_files[run_date], 'r') as h5t:
                        try:
                            source_ds, test_ds = h5s[var], h5t[var]
                        except KeyError:
                            raise ValueError(f"Variable '{var}' not found in file")
                        
                        # Check spatial dimensions match
                        if source_ds.shape[1:] != test_ds.shape[1:]:
                            raise ValueError(
                                f"Spatial shape mismatch for {var} on {run_date.isoformat()}: "
                                f"source {source_ds.shape[1:]} vs test {test_ds.shape[1:]}"
                            )
                        
                        n_file_times = min(source_ds.shape[0], test_ds.shape[0])
                        shape = (n_file_times, source_ds.shape[1], source_ds.shape[2])
                        
                        source_rechunker = rechunkit.rechunker(lambda idx: source_ds[idx], shape, source_ds.dtype, source_ds.chunks or shape, (1, shape[1], shape[2]), max_memory_bytes)
                        test_rechunker = rechunkit.rechunker(lambda idx: test_ds[idx], shape, test_ds.dtype, test_ds.chunks or shape, (1, shape[1], shape[2]), max_memory_bytes)

                        for (slices, s_chunk), (_, t_chunk) in zip(source_rechunker, test_rechunker):
                            # Apply Subsetting
                            s_chunk = s_chunk[:, self.y_slice, self.x_slice]
                            t_chunk = t_chunk[:, self.y_slice, self.x_slice]
                            
                            out_idx = time_offset + slices[0].start
                            
                            if agg_type == 'cell':
                                for metric in metrics:
                                    res = self._compute_cell_metric(s_chunk, t_chunk, metric, threshold, epsilon)
                                    if self.use_mask:
                                        res = np.where(self.spatial_mask, res, out_datasets[(var, metric)].fillvalue)
                                    out_datasets[(var, metric)][out_idx:out_idx+s_chunk.shape[0]] = res
                            else:
                                for m_idx, metric in enumerate(metrics):
                                    res = self._compute_domain_metric(s_chunk, t_chunk, metric, threshold, epsilon)
                                    out_datasets[var][out_idx:out_idx+s_chunk.shape[0], m_idx] = res
                                    
                        time_offset += n_file_times
        return output_path

    def _compute_cell_metric(self, s_chunk, t_chunk, metric, threshold, epsilon):
        if metric == 'ne': return compute_ne(s_chunk, t_chunk, epsilon)
        if metric == 'ane': return compute_ane(s_chunk, t_chunk, epsilon)
        if metric == 'rse': return compute_rse(s_chunk, t_chunk)
        if metric == 'bias': return compute_bias(s_chunk, t_chunk)
        
        # Categorical cell-by-cell
        if threshold is None: raise ValueError(f"Threshold required for {metric}")
        s_yes, t_yes = s_chunk >= threshold, t_chunk >= threshold
        if metric == 'pod': return np.where(s_yes, t_yes.astype(np.float32), np.nan)
        if metric == 'far': return np.where(t_yes, (~s_yes).astype(np.float32), np.nan)
        if metric == 'csi': return np.where(s_yes | t_yes, (s_yes & t_yes).astype(np.float32), np.nan)
        return (s_yes & t_yes).astype(np.float32)

    def _compute_domain_metric(self, s_chunk, t_chunk, metric, threshold, epsilon):
        if metric == 'ne': return compute_ne_domain(s_chunk, t_chunk, self.spatial_mask, epsilon)
        if metric == 'ane': return compute_ane_domain(s_chunk, t_chunk, self.spatial_mask, epsilon)
        if metric == 'rmse': return compute_rmse_domain(s_chunk, t_chunk, self.spatial_mask)
        if metric == 'bias': return compute_bias_domain(s_chunk, t_chunk, self.spatial_mask)
        
        # Categorical domain-aggregated
        if threshold is None: raise ValueError(f"Threshold required for {metric}")
        results = []
        for i in range(s_chunk.shape[0]):
            s_step, t_step = s_chunk[i], t_chunk[i]
            if self.use_mask:
                s_step, t_step = s_step[self.spatial_mask], t_step[self.spatial_mask]
            ct = ContingencyTable.from_data(s_step, t_step, threshold)
            if metric == 'pod': results.append(ct.pod())
            elif metric == 'far': results.append(ct.far())
            elif metric == 'csi': results.append(ct.csi())
            elif metric == 'gss': results.append(ct.gss())
            elif metric == 'fbias': results.append(ct.bias())
        return results
