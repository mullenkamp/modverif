"""
Centralized orchestrator for model evaluation using cfdb datasets.
"""
import pathlib
from typing import Union, List, Tuple

import cfdb
import numpy as np

from modverif.metrics import (
    AVAILABLE_DOMAIN_METRICS,
    AVAILABLE_METRICS,
    AVAILABLE_WIND_METRICS,
    ContingencyTable,
    compute_ane,
    compute_ane_domain,
    compute_bias,
    compute_bias_domain,
    compute_diurnal_stats,
    compute_fss_multi_scale,
    compute_mae,
    compute_ne,
    compute_ne_domain,
    compute_pearson_domain,
    compute_rmse_domain,
    compute_rse,
    compute_vector_rmse,
    compute_wind_direction_bias,
    compute_wind_speed_bias,
)


class Evaluator:
    """
    Orchestrates the evaluation of model outputs between two runs.

    Accepts cfdb datasets as input and computes cell-level or domain-aggregated
    error metrics. Data should be pre-converted to cfdb format using cfdb-ingest
    or similar tools before evaluation.

    Parameters
    ----------
    source : str or pathlib.Path
        Path to cfdb dataset containing source/reference model output.
    test : str or pathlib.Path
        Path to cfdb dataset containing test model output to evaluate.
    region : tuple or np.ndarray, optional
        Spatial region to evaluate. Can be either:
        - tuple of 4 floats (min_lon, min_lat, max_lon, max_lat): Bounding box
          in WGS84 degrees, transformed to the dataset's CRS for subsetting.
        - 2D numpy boolean array: Mask array where True indicates cells to include.
          Must match spatial dimensions (n_y, n_x).
        If None (default), evaluate the entire domain.
    start_time : str or np.datetime64, optional
        Start time (inclusive) for evaluation period.
    end_time : str or np.datetime64, optional
        End time (inclusive) for evaluation period.
    """

    def __init__(
        self,
        source: Union[str, pathlib.Path],
        test: Union[str, pathlib.Path],
        region: Union[Tuple[float, float, float, float], np.ndarray, None] = None,
        start_time: Union[str, np.datetime64, None] = None,
        end_time: Union[str, np.datetime64, None] = None,
    ):
        self.source_path = pathlib.Path(source)
        self.test_path = pathlib.Path(test)
        self.region = region
        self.start_time = np.datetime64(start_time) if isinstance(start_time, str) else start_time
        self.end_time = np.datetime64(end_time) if isinstance(end_time, str) else end_time

        if not self.source_path.exists():
            raise FileNotFoundError(f"Source dataset not found: {self.source_path}")
        if not self.test_path.exists():
            raise FileNotFoundError(f"Test dataset not found: {self.test_path}")

        self._initialize_context()

    def _initialize_context(self):
        """Extract shared metadata and resolve spatial/temporal subsetting."""
        with cfdb.open_dataset(self.source_path) as ds_s, cfdb.open_dataset(self.test_path) as ds_t:
            # Read coordinate arrays
            s_time = ds_s['time'].data
            t_time = ds_t['time'].data
            s_y = ds_s['y'].data
            t_y = ds_t['y'].data
            s_x = ds_s['x'].data
            t_x = ds_t['x'].data

            # Validate spatial dimensions match
            if len(s_y) != len(t_y) or len(s_x) != len(t_x):
                raise ValueError(
                    f"Spatial dimension mismatch: source ({len(s_y)}, {len(s_x)}) "
                    f"vs test ({len(t_y)}, {len(t_x)})"
                )

            # Find common times
            common_times = np.intersect1d(s_time, t_time)
            if len(common_times) == 0:
                raise ValueError("No common timesteps found between source and test datasets")

            # Apply time filtering
            if self.start_time is not None:
                common_times = common_times[common_times >= self.start_time]
            if self.end_time is not None:
                common_times = common_times[common_times <= self.end_time]
            if len(common_times) == 0:
                raise ValueError("No timesteps remain after time filtering")

            # Build time index maps
            self._source_time_indices = np.searchsorted(s_time, common_times)
            self._test_time_indices = np.searchsorted(t_time, common_times)

            # Store full coordinate arrays and dimensions
            self.time_values = common_times
            self.n_times = len(common_times)
            self.y_values = s_y.copy()
            self.x_values = s_x.copy()

            # Grid spacing from coordinate differences
            self.dy = float(np.diff(s_y[:2])[0]) if len(s_y) > 1 else None
            self.dx = float(np.diff(s_x[:2])[0]) if len(s_x) > 1 else None

            # CRS
            self.crs = ds_s.crs

            # Height coordinate
            if 'height' in ds_s.coord_names:
                self.height_values = ds_s['height'].data
            else:
                self.height_values = None

            # Resolve spatial subsetting
            self.y_slice = slice(None)
            self.x_slice = slice(None)
            self.spatial_mask = None
            self.use_mask = False
            self.n_y_full = len(s_y)
            self.n_x_full = len(s_x)

            if self.region is not None:
                if isinstance(self.region, np.ndarray):
                    if self.region.ndim != 2:
                        raise ValueError(f"Spatial mask must be 2D, got {self.region.ndim}D array")
                    self.spatial_mask = self.region.astype(bool)
                    self.use_mask = True
                    if self.spatial_mask.shape != (self.n_y_full, self.n_x_full):
                        raise ValueError(
                            f"Mask shape {self.spatial_mask.shape} does not match "
                            f"domain shape ({self.n_y_full}, {self.n_x_full})"
                        )
                elif isinstance(self.region, (list, tuple)) and len(self.region) == 4:
                    self.y_slice, self.x_slice = self._bbox_to_slices(
                        self.region, s_y, s_x, self.crs
                    )
                else:
                    raise ValueError(
                        "region must be either a tuple of 4 floats "
                        "(min_lon, min_lat, max_lon, max_lat) or a 2D numpy boolean array"
                    )

            # Final output dimensions
            if self.y_slice != slice(None):
                self.n_y = self.y_slice.stop - self.y_slice.start
                self.y_values = s_y[self.y_slice]
            else:
                self.n_y = self.n_y_full

            if self.x_slice != slice(None):
                self.n_x = self.x_slice.stop - self.x_slice.start
                self.x_values = s_x[self.x_slice]
            else:
                self.n_x = self.n_x_full

    @staticmethod
    def _bbox_to_slices(bbox, y_coords, x_coords, crs):
        """Convert a WGS84 bounding box to y/x index slices."""
        import pyproj

        min_lon, min_lat, max_lon, max_lat = bbox
        transformer = pyproj.Transformer.from_crs('EPSG:4326', crs, always_xy=True)

        # Transform corners and edge midpoints for curved projections
        sample_lons = [min_lon, max_lon, min_lon, max_lon,
                       (min_lon + max_lon) / 2, (min_lon + max_lon) / 2, min_lon, max_lon]
        sample_lats = [min_lat, min_lat, max_lat, max_lat,
                       min_lat, max_lat, (min_lat + max_lat) / 2, (min_lat + max_lat) / 2]
        proj_x, proj_y = transformer.transform(sample_lons, sample_lats)

        x_min, x_max = min(proj_x), max(proj_x)
        y_min, y_max = min(proj_y), max(proj_y)

        x_mask = (x_coords >= x_min) & (x_coords <= x_max)
        y_mask = (y_coords >= y_min) & (y_coords <= y_max)

        x_indices = np.where(x_mask)[0]
        y_indices = np.where(y_mask)[0]

        if len(x_indices) == 0 or len(y_indices) == 0:
            raise ValueError(
                f"No grid cells found within bounds: "
                f"lon=[{min_lon}, {max_lon}], lat=[{min_lat}, {max_lat}]"
            )

        return (
            slice(int(y_indices[0]), int(y_indices[-1]) + 1),
            slice(int(x_indices[0]), int(x_indices[-1]) + 1),
        )

    def evaluate_cell(
        self,
        output_path: Union[str, pathlib.Path],
        variables: List[str],
        metrics: Union[str, List[str]] = 'ne',
        threshold: float = None,
        epsilon: float = 1e-10,
    ) -> pathlib.Path:
        """
        Perform high-resolution cell-by-cell spatial evaluation.

        Compares model outputs from two runs, computing error metrics at each
        individual grid cell. Results are stored in a cfdb dataset with shape
        (time, y, x) per variable-metric combination.

        Parameters
        ----------
        output_path : str or pathlib.Path
            Path for output cfdb dataset containing evaluation results.
        variables : list[str]
            List of cfdb variable names to evaluate (e.g., ['air_temperature', 'u_wind']).
        metrics : str or list[str]
            Metric(s) to compute.
            Continuous: 'ne', 'ane', 'rse', 'bias'.
            Categorical (requires threshold): 'pod', 'far', 'csi', 'fbias'.
            Default is 'ne'.
        threshold : float, optional
            Threshold value used for categorical metrics.
        epsilon : float
            Small value to avoid division by zero in normalized metrics.

        Returns
        -------
        pathlib.Path
            Path to the generated output cfdb dataset.
        """
        return self._run_engine(output_path, variables, metrics, threshold, epsilon, agg_type='cell')

    def evaluate_domain(
        self,
        output_path: Union[str, pathlib.Path],
        variables: List[str],
        metrics: Union[str, List[str]] = 'ne',
        threshold: float = None,
        epsilon: float = 1e-10,
    ) -> pathlib.Path:
        """
        Perform domain-aggregated evaluation for time series analysis.

        Aggregates values over the spatial domain at each timestep, then
        computes the metric. Results are stored in a cfdb dataset with shape
        (time, metric) per variable.

        Parameters
        ----------
        output_path : str or pathlib.Path
            Path for output cfdb dataset containing evaluation results.
        variables : list[str]
            List of cfdb variable names to evaluate.
        metrics : str or list[str]
            Metric(s) to compute.
            Continuous: 'ne', 'ane', 'rmse', 'bias'.
            Categorical (requires threshold): 'pod', 'far', 'csi', 'gss', 'fbias'.
            Default is 'ne'.
        threshold : float, optional
            Threshold value used for categorical metrics.
        epsilon : float
            Small value to avoid division by zero.

        Returns
        -------
        pathlib.Path
            Path to the generated output cfdb dataset.
        """
        return self._run_engine(output_path, variables, metrics, threshold, epsilon, agg_type='domain')

    def _run_engine(self, output_path, variables, metrics, threshold, epsilon, agg_type):
        """
        Unified processing core for both cell and domain evaluations.

        Parameters
        ----------
        output_path : pathlib.Path
            Path to the output cfdb dataset.
        variables : list[str]
            List of cfdb variable names to process.
        metrics : list[str]
            List of metrics to calculate.
        threshold : float or None
            Value for categorical thresholding.
        epsilon : float
            Small value for division safety.
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

        with (
            cfdb.open_dataset(self.source_path) as ds_s,
            cfdb.open_dataset(self.test_path) as ds_t,
            cfdb.open_dataset(output_path, 'n', dataset_type='grid') as ds_out,
        ):
            # Create output coordinates
            time_coord = ds_out.create.coord.time(data=self.time_values)

            if agg_type == 'cell':
                y_coord = ds_out.create.coord.y(data=self.y_values.astype('float32'))
                x_coord = ds_out.create.coord.x(data=self.x_values.astype('float32'))
                out_coord_names = ('time', 'y', 'x')
                out_chunk_shape = (1, self.n_y, self.n_x)
            else:
                # For domain metrics, create a generic coordinate for the metric dimension
                metric_indices = np.arange(len(metrics), dtype='int32')
                metric_coord = ds_out.create.coord.generic(
                    'metric', data=metric_indices, dtype='int32'
                )
                metric_coord.attrs['flag_meanings'] = ' '.join(metrics)
                out_coord_names = ('time', 'metric')
                out_chunk_shape = (1, len(metrics))

            # Set CRS and attributes
            if self.crs is not None:
                if agg_type == 'cell':
                    ds_out.create.crs.from_user_input(self.crs, x_coord='x', y_coord='y')

            ds_out.attrs['source_path'] = str(self.source_path)
            ds_out.attrs['test_path'] = str(self.test_path)
            ds_out.attrs['aggregation_type'] = agg_type

            # Store spatial mask in output if used
            if self.use_mask and agg_type == 'cell':
                mask_var = ds_out.create.data_var.generic(
                    'spatial_mask', ('y', 'x'), dtype='int8', chunk_shape=(self.n_y, self.n_x)
                )
                mask_var[:] = self.spatial_mask.astype(np.int8)
                mask_var.attrs['long_name'] = 'Spatial mask (1=included, 0=excluded)'

            # Create output data variables
            out_vars = {}
            for var in variables:
                if agg_type == 'cell':
                    for metric in metrics:
                        out_name = f"{var}_{metric}"
                        out_var = ds_out.create.data_var.generic(
                            out_name, out_coord_names, dtype='float32',
                            chunk_shape=out_chunk_shape,
                        )
                        out_var.attrs['long_name'] = f"{metric.upper()} for {var}"
                        out_vars[(var, metric)] = out_var
                else:
                    out_var = ds_out.create.data_var.generic(
                        var, out_coord_names, dtype='float32',
                        chunk_shape=out_chunk_shape,
                    )
                    out_var.attrs['long_name'] = f"Domain-aggregated metrics for {var}"
                    out_vars[var] = out_var

            # Determine height index for each variable
            # cfdb-ingest produces (time, height, y, x); we select the first height level
            height_idx = 0

            # Main processing loop: iterate per variable, per timestep
            for var in variables:
                if var not in ds_s:
                    raise ValueError(f"Variable '{var}' not found in source dataset")
                if var not in ds_t:
                    raise ValueError(f"Variable '{var}' not found in test dataset")

                s_var = ds_s[var]
                t_var = ds_t[var]
                n_dims = len(s_var.shape)

                for out_t, (s_t_idx, t_t_idx) in enumerate(
                    zip(self._source_time_indices, self._test_time_indices)
                ):
                    # Read a single timestep and squeeze to 2D (y, x)
                    # cfdb preserves all dimensions on integer indexing,
                    # so we squeeze to remove scalar dims (time, height)
                    if n_dims == 4:
                        # (time, height, y, x) -> select time and height -> squeeze to (y, x)
                        s_data = s_var[(int(s_t_idx), height_idx, self.y_slice, self.x_slice)].data
                        t_data = t_var[(int(t_t_idx), height_idx, self.y_slice, self.x_slice)].data
                        s_data = s_data[0, 0]
                        t_data = t_data[0, 0]
                    elif n_dims == 3:
                        # (time, y, x) -> select time -> squeeze to (y, x)
                        s_data = s_var[(int(s_t_idx), self.y_slice, self.x_slice)].data
                        t_data = t_var[(int(t_t_idx), self.y_slice, self.x_slice)].data
                        s_data = s_data[0]
                        t_data = t_data[0]
                    else:
                        raise ValueError(
                            f"Variable '{var}' has {n_dims} dimensions, expected 3 or 4"
                        )

                    # Metrics expect (time, y, x) — add time axis back as size 1
                    s_3d = s_data[np.newaxis]  # (1, y, x)
                    t_3d = t_data[np.newaxis]  # (1, y, x)

                    if agg_type == 'cell':
                        for metric in metrics:
                            res = self._compute_cell_metric(
                                s_3d, t_3d, metric, threshold, epsilon
                            )
                            if self.use_mask:
                                res = np.where(self.spatial_mask, res, np.float32(np.nan))
                            out_vars[(var, metric)][(out_t, slice(None), slice(None))] = res[0]
                    else:
                        results = np.zeros(len(metrics), dtype=np.float32)
                        for m_idx, metric in enumerate(metrics):
                            res = self._compute_domain_metric(
                                s_3d, t_3d, metric, threshold, epsilon
                            )
                            if isinstance(res, (list, np.ndarray)):
                                results[m_idx] = float(res[0])
                            else:
                                results[m_idx] = float(res)
                        out_vars[var][(out_t, slice(None))] = results

        return output_path

    def _compute_cell_metric(self, s_chunk, t_chunk, metric, threshold, epsilon):
        if metric == 'ne':
            return compute_ne(s_chunk, t_chunk, epsilon)
        if metric == 'ane':
            return compute_ane(s_chunk, t_chunk, epsilon)
        if metric == 'rse':
            return compute_rse(s_chunk, t_chunk)
        if metric == 'bias':
            return compute_bias(s_chunk, t_chunk)
        if metric == 'mae':
            return compute_mae(s_chunk, t_chunk)

        # Categorical cell-by-cell
        if threshold is None:
            raise ValueError(f"Threshold required for {metric}")
        s_yes, t_yes = s_chunk >= threshold, t_chunk >= threshold
        if metric == 'pod':
            return np.where(s_yes, t_yes.astype(np.float32), np.nan)
        if metric == 'far':
            return np.where(t_yes, (~s_yes).astype(np.float32), np.nan)
        if metric == 'csi':
            return np.where(s_yes | t_yes, (s_yes & t_yes).astype(np.float32), np.nan)
        return (s_yes & t_yes).astype(np.float32)

    def _compute_domain_metric(self, s_chunk, t_chunk, metric, threshold, epsilon):
        if metric == 'ne':
            return compute_ne_domain(s_chunk, t_chunk, self.spatial_mask, epsilon)
        if metric == 'ane':
            return compute_ane_domain(s_chunk, t_chunk, self.spatial_mask, epsilon)
        if metric == 'rmse':
            return compute_rmse_domain(s_chunk, t_chunk, self.spatial_mask)
        if metric == 'bias':
            return compute_bias_domain(s_chunk, t_chunk, self.spatial_mask)
        if metric == 'pearson':
            return compute_pearson_domain(s_chunk, t_chunk, self.spatial_mask)

        # Categorical domain-aggregated
        if threshold is None:
            raise ValueError(f"Threshold required for {metric}")
        results = []
        for i in range(s_chunk.shape[0]):
            s_step, t_step = s_chunk[i], t_chunk[i]
            if self.use_mask:
                s_step, t_step = s_step[self.spatial_mask], t_step[self.spatial_mask]
            ct = ContingencyTable.from_data(s_step, t_step, threshold)
            if metric == 'pod':
                results.append(ct.pod())
            elif metric == 'far':
                results.append(ct.far())
            elif metric == 'csi':
                results.append(ct.csi())
            elif metric == 'gss':
                results.append(ct.gss())
            elif metric == 'fbias':
                results.append(ct.bias())
        return results

    def evaluate_fss(
        self,
        output_path: Union[str, pathlib.Path],
        variables: List[str],
        threshold: float,
        neighborhood_sizes: list = None,
    ) -> pathlib.Path:
        """
        Compute Fractions Skill Score across multiple spatial scales.

        Parameters
        ----------
        output_path : str or pathlib.Path
            Path for output cfdb dataset.
        variables : list[str]
            Variable names to evaluate.
        threshold : float
            Binary event threshold.
        neighborhood_sizes : list[int], optional
            Neighborhood sizes. Default: [1, 3, 5, 9, 17, 33, 65].

        Returns
        -------
        pathlib.Path
            Path to the output cfdb dataset.
        """
        output_path = pathlib.Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(variables, str):
            variables = [variables]
        if neighborhood_sizes is None:
            neighborhood_sizes = [1, 3, 5, 9, 17, 33, 65]

        height_idx = 0

        with (
            cfdb.open_dataset(self.source_path) as ds_s,
            cfdb.open_dataset(self.test_path) as ds_t,
            cfdb.open_dataset(output_path, 'n', dataset_type='grid') as ds_out,
        ):
            time_coord = ds_out.create.coord.time(data=self.time_values)
            scale_data = np.array(neighborhood_sizes, dtype='int32')
            scale_coord = ds_out.create.coord.generic('scale', data=scale_data, dtype='int32')

            ds_out.attrs['source_path'] = str(self.source_path)
            ds_out.attrs['test_path'] = str(self.test_path)
            ds_out.attrs['evaluation_type'] = 'fss'
            ds_out.attrs['threshold'] = str(threshold)

            out_vars = {}
            for var in variables:
                out_var = ds_out.create.data_var.generic(
                    var, ('time', 'scale'), dtype='float32',
                    chunk_shape=(1, len(neighborhood_sizes)),
                )
                out_var.attrs['long_name'] = f'FSS for {var}'
                out_var.attrs['units'] = '1'
                out_vars[var] = out_var

            for var in variables:
                if var not in ds_s:
                    raise ValueError(f"Variable '{var}' not found in source dataset")
                if var not in ds_t:
                    raise ValueError(f"Variable '{var}' not found in test dataset")

                s_var = ds_s[var]
                t_var = ds_t[var]
                n_dims = len(s_var.shape)

                for out_t, (s_t_idx, t_t_idx) in enumerate(
                    zip(self._source_time_indices, self._test_time_indices)
                ):
                    if n_dims == 4:
                        s_data = s_var[(int(s_t_idx), height_idx, self.y_slice, self.x_slice)].data[0, 0]
                        t_data = t_var[(int(t_t_idx), height_idx, self.y_slice, self.x_slice)].data[0, 0]
                    elif n_dims == 3:
                        s_data = s_var[(int(s_t_idx), self.y_slice, self.x_slice)].data[0]
                        t_data = t_var[(int(t_t_idx), self.y_slice, self.x_slice)].data[0]
                    else:
                        raise ValueError(f"Variable '{var}' has {n_dims} dimensions, expected 3 or 4")

                    fss_results = compute_fss_multi_scale(
                        s_data, t_data, threshold, neighborhood_sizes, self.spatial_mask
                    )
                    fss_arr = np.array([fss_results[n] for n in neighborhood_sizes], dtype=np.float32)
                    out_vars[var][(out_t, slice(None))] = fss_arr

        return output_path

    def evaluate_wind(
        self,
        output_path: Union[str, pathlib.Path],
        u_var: str = 'u_wind',
        v_var: str = 'v_wind',
        metrics: Union[str, List[str]] = 'vector_rmse',
    ) -> pathlib.Path:
        """
        Compute vector wind metrics from U/V components.

        Parameters
        ----------
        output_path : str or pathlib.Path
            Path for output cfdb dataset.
        u_var : str
            Name of U-component variable.
        v_var : str
            Name of V-component variable.
        metrics : str or list[str]
            Wind metric(s): 'vector_rmse', 'speed_bias', 'direction_bias'.

        Returns
        -------
        pathlib.Path
            Path to the output cfdb dataset.
        """
        output_path = pathlib.Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(metrics, str):
            metrics = [metrics]
        metrics = [m.lower() for m in metrics]
        for m in metrics:
            if m not in AVAILABLE_WIND_METRICS:
                raise ValueError(f"Unknown wind metric '{m}'. Available: {AVAILABLE_WIND_METRICS}")

        height_idx = 0

        with (
            cfdb.open_dataset(self.source_path) as ds_s,
            cfdb.open_dataset(self.test_path) as ds_t,
            cfdb.open_dataset(output_path, 'n', dataset_type='grid') as ds_out,
        ):
            time_coord = ds_out.create.coord.time(data=self.time_values)
            metric_indices = np.arange(len(metrics), dtype='int32')
            metric_coord = ds_out.create.coord.generic('metric', data=metric_indices, dtype='int32')
            metric_coord.attrs['flag_meanings'] = ' '.join(metrics)

            ds_out.attrs['source_path'] = str(self.source_path)
            ds_out.attrs['test_path'] = str(self.test_path)
            ds_out.attrs['evaluation_type'] = 'wind'

            out_var = ds_out.create.data_var.generic(
                'wind', ('time', 'metric'), dtype='float32',
                chunk_shape=(1, len(metrics)),
            )
            out_var.attrs['long_name'] = 'Vector wind metrics'

            for v in [u_var, v_var]:
                if v not in ds_s:
                    raise ValueError(f"Variable '{v}' not found in source dataset")
                if v not in ds_t:
                    raise ValueError(f"Variable '{v}' not found in test dataset")

            su_var, sv_var = ds_s[u_var], ds_s[v_var]
            tu_var, tv_var = ds_t[u_var], ds_t[v_var]
            n_dims = len(su_var.shape)

            for out_t, (s_t_idx, t_t_idx) in enumerate(
                zip(self._source_time_indices, self._test_time_indices)
            ):
                if n_dims == 4:
                    su = su_var[(int(s_t_idx), height_idx, self.y_slice, self.x_slice)].data[0, 0]
                    sv = sv_var[(int(s_t_idx), height_idx, self.y_slice, self.x_slice)].data[0, 0]
                    tu = tu_var[(int(t_t_idx), height_idx, self.y_slice, self.x_slice)].data[0, 0]
                    tv = tv_var[(int(t_t_idx), height_idx, self.y_slice, self.x_slice)].data[0, 0]
                elif n_dims == 3:
                    su = su_var[(int(s_t_idx), self.y_slice, self.x_slice)].data[0]
                    sv = sv_var[(int(s_t_idx), self.y_slice, self.x_slice)].data[0]
                    tu = tu_var[(int(t_t_idx), self.y_slice, self.x_slice)].data[0]
                    tv = tv_var[(int(t_t_idx), self.y_slice, self.x_slice)].data[0]
                else:
                    raise ValueError(f"Variable '{u_var}' has {n_dims} dimensions, expected 3 or 4")

                if self.use_mask:
                    su, sv = su[self.spatial_mask], sv[self.spatial_mask]
                    tu, tv = tu[self.spatial_mask], tv[self.spatial_mask]

                results = np.zeros(len(metrics), dtype=np.float32)
                for m_idx, metric in enumerate(metrics):
                    if metric == 'vector_rmse':
                        results[m_idx] = compute_vector_rmse(su, sv, tu, tv)
                    elif metric == 'speed_bias':
                        results[m_idx] = compute_wind_speed_bias(su, sv, tu, tv)
                    elif metric == 'direction_bias':
                        results[m_idx] = compute_wind_direction_bias(su, sv, tu, tv)
                out_var[(out_t, slice(None))] = results

        return output_path

    def evaluate_diurnal(
        self,
        output_path: Union[str, pathlib.Path],
        variables: List[str],
        metrics: Union[str, List[str]] = 'bias',
    ) -> pathlib.Path:
        """
        Compute diurnal cycle of domain-aggregated metrics.

        Parameters
        ----------
        output_path : str or pathlib.Path
            Path for output cfdb dataset.
        variables : list[str]
            Variable names to evaluate.
        metrics : str or list[str]
            Metric(s) to group by hour: 'bias', 'rmse', 'mae', 'pearson'.

        Returns
        -------
        pathlib.Path
            Path to the output cfdb dataset.
        """
        output_path = pathlib.Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(variables, str):
            variables = [variables]
        if isinstance(metrics, str):
            metrics = [metrics]

        height_idx = 0

        with (
            cfdb.open_dataset(self.source_path) as ds_s,
            cfdb.open_dataset(self.test_path) as ds_t,
        ):
            # Collect all data for diurnal binning
            var_source_data = {}
            var_test_data = {}

            for var in variables:
                if var not in ds_s:
                    raise ValueError(f"Variable '{var}' not found in source dataset")
                if var not in ds_t:
                    raise ValueError(f"Variable '{var}' not found in test dataset")

                s_var = ds_s[var]
                t_var = ds_t[var]
                n_dims = len(s_var.shape)

                s_means = np.zeros(self.n_times, dtype=np.float64)
                t_means = np.zeros(self.n_times, dtype=np.float64)

                for out_t, (s_t_idx, t_t_idx) in enumerate(
                    zip(self._source_time_indices, self._test_time_indices)
                ):
                    if n_dims == 4:
                        s_data = s_var[(int(s_t_idx), height_idx, self.y_slice, self.x_slice)].data[0, 0]
                        t_data = t_var[(int(t_t_idx), height_idx, self.y_slice, self.x_slice)].data[0, 0]
                    elif n_dims == 3:
                        s_data = s_var[(int(s_t_idx), self.y_slice, self.x_slice)].data[0]
                        t_data = t_var[(int(t_t_idx), self.y_slice, self.x_slice)].data[0]
                    else:
                        raise ValueError(f"Variable '{var}' has {n_dims} dimensions, expected 3 or 4")

                    if self.use_mask:
                        s_data = s_data[self.spatial_mask]
                        t_data = t_data[self.spatial_mask]
                    s_means[out_t] = np.mean(s_data)
                    t_means[out_t] = np.mean(t_data)

                var_source_data[var] = s_means
                var_test_data[var] = t_means

        # Write output
        with cfdb.open_dataset(output_path, 'n', dataset_type='grid') as ds_out:
            hour_data = np.arange(24, dtype='int32')
            hour_coord = ds_out.create.coord.generic('hour', data=hour_data, dtype='int32')
            metric_indices = np.arange(len(metrics), dtype='int32')
            metric_coord = ds_out.create.coord.generic('metric', data=metric_indices, dtype='int32')
            metric_coord.attrs['flag_meanings'] = ' '.join(metrics)

            ds_out.attrs['source_path'] = str(self.source_path)
            ds_out.attrs['test_path'] = str(self.test_path)
            ds_out.attrs['evaluation_type'] = 'diurnal'

            for var in variables:
                out_var = ds_out.create.data_var.generic(
                    var, ('hour', 'metric'), dtype='float64',
                    chunk_shape=(24, len(metrics)),
                )
                out_var.attrs['long_name'] = f'Diurnal cycle metrics for {var}'

                for m_idx, metric in enumerate(metrics):
                    _, values = compute_diurnal_stats(
                        self.time_values, var_test_data[var], var_source_data[var], metric
                    )
                    for h in range(24):
                        out_var[(h, m_idx)] = np.array([values[h]], dtype=np.float64)

        return output_path
