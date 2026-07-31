"""
Station evaluator for comparing gridded model output to point observations.

Equivalent to MET's Point-Stat tool. Compares cfdb grid datasets to
cfdb ts_ortho (time series orthogonal) station observation datasets.
"""
import pathlib
from typing import List, Union

import cfdb
import numpy as np
import shapely

from modverif.metrics import (
    AVAILABLE_STATION_METRICS,
    AVAILABLE_WIND_METRICS,
    compute_ane_1d,
    compute_diurnal_stats,
    compute_lagged_correlation,
    compute_mae_1d,
    compute_mean_bias,
    compute_ne_1d,
    compute_pearson_correlation,
    compute_rmse_1d,
    compute_vector_rmse,
    compute_wind_direction_bias,
    compute_wind_speed_bias,
)


class StationEvaluator:
    """
    Compares gridded model output to point observations at weather stations.

    Parameters
    ----------
    model : str or pathlib.Path
        Path to cfdb grid dataset (model output).
    observations : str or pathlib.Path
        Path to cfdb ts_ortho dataset (station observations).
    start_time : str or np.datetime64, optional
        Start of evaluation period (inclusive).
    end_time : str or np.datetime64, optional
        End of evaluation period (inclusive).
    interpolation_order : int
        Spatial interpolation order for grid-to-point extraction.
        0=nearest, 1=linear (default), 3=cubic.
    height : float or None, optional
        Default target height in meters for single-level observations.
        Used when observation variables have no height coordinate.
        If None, uses the first (surface) model height level.
    variable_heights : dict[str, float] or None, optional
        Per-variable target heights in meters, overriding ``height``.
        e.g., ``{'air_temperature': 2.0, 'wind_speed': 10.0}``.
    """

    def __init__(
        self,
        model: Union[str, pathlib.Path],
        observations: Union[str, pathlib.Path],
        start_time=None,
        end_time=None,
        interpolation_order: int = 1,
        height: float = None,
        variable_heights: dict = None,
    ):
        self.model_path = pathlib.Path(model)
        self.observations_path = pathlib.Path(observations)
        self.start_time = np.datetime64(start_time) if isinstance(start_time, str) else start_time
        self.end_time = np.datetime64(end_time) if isinstance(end_time, str) else end_time
        self.interpolation_order = interpolation_order
        self.height = height
        self.variable_heights = variable_heights or {}

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model dataset not found: {self.model_path}")
        if not self.observations_path.exists():
            raise FileNotFoundError(f"Observations dataset not found: {self.observations_path}")

        self._initialize_context()

    def _initialize_context(self):
        """Extract shared metadata, find common times, detect height coordinate."""
        with (
            cfdb.open_dataset(self.model_path) as ds_m,
            cfdb.open_dataset(self.observations_path) as ds_o,
        ):
            # Read time coordinates
            m_time = ds_m['time'].data
            o_time = ds_o['time'].data

            common_times = np.intersect1d(m_time, o_time)
            if len(common_times) == 0:
                raise ValueError("No common timesteps found between model and observations")

            if self.start_time is not None:
                common_times = common_times[common_times >= self.start_time]
            if self.end_time is not None:
                common_times = common_times[common_times <= self.end_time]
            if len(common_times) == 0:
                raise ValueError("No timesteps remain after time filtering")

            self._model_time_indices = np.searchsorted(m_time, common_times)
            self._obs_time_indices = np.searchsorted(o_time, common_times)
            self.time_values = common_times
            self.n_times = len(common_times)

            # Station geometry
            self._geo_coord_name = self._find_geo_coord(ds_o)
            self.station_points = ds_o[self._geo_coord_name].data
            self.n_stations = len(self.station_points)

            # Extract lon/lat from points
            coords = shapely.get_coordinates(self.station_points)
            self.station_lons = coords[:, 0]
            self.station_lats = coords[:, 1]

            # Detect observation height coordinate
            self.obs_has_height = 'height' in ds_o.coord_names
            if self.obs_has_height:
                self.obs_heights = ds_o['height'].data
                self.n_obs_heights = len(self.obs_heights)
            else:
                self.obs_heights = None
                self.n_obs_heights = 0

            # Model height levels
            if 'height' in ds_m.coord_names:
                self.model_heights = ds_m['height'].data
            else:
                self.model_heights = None

    @staticmethod
    def _find_geo_coord(ds):
        """Find the geometry coordinate name in a ts_ortho dataset."""
        for name in ds.coord_names:
            if name in ('point', 'station'):
                return name
        # Fall back to checking for xy axis
        for name in ds.coord_names:
            if name not in ('time', 'height'):
                return name
        raise ValueError("Could not find geometry coordinate in observations dataset")

    def _resolve_height_index(self, var_name):
        """
        Determine which model height index to use for a variable.

        Returns
        -------
        int
            Model height level index.
        """
        target_height = self.variable_heights.get(var_name, self.height)

        if self.model_heights is None:
            return 0

        if target_height is None:
            return 0

        # Find nearest model height
        diffs = np.abs(self.model_heights - target_height)
        return int(np.argmin(diffs))

    def evaluate(
        self,
        output_path: Union[str, pathlib.Path],
        variables: List[str],
        metrics: Union[str, List[str]] = 'bias',
        epsilon: float = 1e-10,
    ) -> pathlib.Path:
        """
        Compute per-station, per-timestep metrics.

        For each variable and timestep:
        1. Extract model values at station locations via GridInterp.to_points()
        2. Select the appropriate model height level
        3. Pair with observation values
        4. Compute scalar metric per station per timestep

        Parameters
        ----------
        output_path : str or pathlib.Path
            Path for output cfdb ts_ortho dataset.
        variables : list[str]
            Variable names to evaluate (must exist in both model and obs).
        metrics : str or list[str]
            Metric(s) to compute. Default is 'bias'.
        epsilon : float
            Small value to avoid division by zero.

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
        metrics = [m.lower() for m in metrics]

        for m in metrics:
            if m not in AVAILABLE_STATION_METRICS:
                raise ValueError(f"Unknown station metric '{m}'. Available: {AVAILABLE_STATION_METRICS}")

        with (
            cfdb.open_dataset(self.model_path) as ds_m,
            cfdb.open_dataset(self.observations_path) as ds_o,
            cfdb.open_dataset(output_path, 'n', dataset_type='ts_ortho') as ds_out,
        ):
            # Validate variables exist in both datasets
            for var in variables:
                if var not in ds_m:
                    raise ValueError(f"Variable '{var}' not found in model dataset")
                if var not in ds_o:
                    raise ValueError(f"Variable '{var}' not found in observations dataset")

            # Create output coordinates
            geo_coord = ds_out.create.coord.point()
            geo_coord.append(list(self.station_points))
            ds_out.create.coord.time(data=self.time_values)
            ds_out.create.crs.from_user_input(4326, xy_coord='point')

            ds_out.attrs['model_path'] = str(self.model_path)
            ds_out.attrs['observations_path'] = str(self.observations_path)
            ds_out.attrs['interpolation_order'] = str(self.interpolation_order)

            for var in variables:
                m_var = ds_m[var]
                o_var = ds_o[var]
                n_dims_m = len(m_var.shape)
                height_idx = self._resolve_height_index(var)

                # Collect model and obs data per timestep
                model_data = np.full((self.n_times, self.n_stations), np.nan, dtype=np.float64)
                obs_data = np.full((self.n_times, self.n_stations), np.nan, dtype=np.float64)

                # Extract model values at station locations using GridInterp
                interp = m_var.interp()
                interp_results = {}
                for time_val, values in interp.to_points(
                    o_var, order=self.interpolation_order
                ):
                    interp_results[time_val] = values

                for out_t, time_val in enumerate(self.time_values):
                    if time_val in interp_results:
                        vals = interp_results[time_val]
                        if vals.ndim > 1 and n_dims_m == 4:
                            # (n_heights, n_stations) -> select height
                            model_data[out_t] = vals[height_idx]
                        else:
                            model_data[out_t] = vals

                    # Read observation data
                    o_t_idx = int(self._obs_time_indices[out_t])
                    if self.obs_has_height:
                        obs_raw = o_var[(o_t_idx, 0, slice(None))].data
                        obs_data[out_t] = obs_raw[0, 0] if obs_raw.ndim > 1 else obs_raw
                    else:
                        obs_raw = o_var[(o_t_idx, slice(None))].data
                        obs_data[out_t] = obs_raw[0] if obs_raw.ndim > 1 else obs_raw

                # Compute metrics and write output
                for metric in metrics:
                    out_name = f"{var}_{metric}"
                    out_dv = ds_out.create.data_var.generic(
                        out_name, ('time', 'point'), dtype='float32',
                        chunk_shape=(self.n_times, self.n_stations),
                    )
                    out_dv.attrs['long_name'] = f'{metric.upper()} for {var}'

                    if metric in ('bias', 'mae', 'ne', 'ane'):
                        # Per-timestep, per-station scalar
                        for t in range(self.n_times):
                            vals = np.zeros(self.n_stations, dtype=np.float32)
                            for s in range(self.n_stations):
                                m_val = model_data[t, s]
                                o_val = obs_data[t, s]
                                if np.isnan(m_val) or np.isnan(o_val):
                                    vals[s] = np.nan
                                elif metric == 'bias':
                                    vals[s] = m_val - o_val
                                elif metric == 'mae':
                                    vals[s] = abs(m_val - o_val)
                                elif metric == 'ne':
                                    vals[s] = ((m_val - o_val) / o_val * 100) if abs(o_val) >= epsilon else 0.0
                                elif metric == 'ane':
                                    vals[s] = abs(
                                        (m_val - o_val) / o_val * 100
                                    ) if abs(o_val) >= epsilon else 0.0
                            out_dv[(t, slice(None))] = vals
                    elif metric in ('rmse', 'pearson'):
                        # Aggregated over time per station -> replicate value across time
                        station_vals = np.zeros(self.n_stations, dtype=np.float32)
                        for s in range(self.n_stations):
                            m_ts = model_data[:, s]
                            o_ts = obs_data[:, s]
                            valid = ~(np.isnan(m_ts) | np.isnan(o_ts))
                            if np.sum(valid) < 2:
                                station_vals[s] = np.nan
                            elif metric == 'rmse':
                                station_vals[s] = compute_rmse_1d(m_ts[valid], o_ts[valid])
                            elif metric == 'pearson':
                                station_vals[s] = compute_pearson_correlation(m_ts[valid], o_ts[valid])
                        # Write same value for all timesteps
                        for t in range(self.n_times):
                            out_dv[(t, slice(None))] = station_vals

        return output_path

    def evaluate_aggregate(
        self,
        output_path: Union[str, pathlib.Path],
        variables: List[str],
        metrics: Union[str, List[str]] = 'bias',
        epsilon: float = 1e-10,
    ) -> pathlib.Path:
        """
        Compute summary statistics aggregated over all stations per timestep.

        Parameters
        ----------
        output_path : str or pathlib.Path
            Path for output cfdb grid dataset.
        variables : list[str]
            Variable names to evaluate.
        metrics : str or list[str]
            Metric(s) to compute.
        epsilon : float
            Small value to avoid division by zero.

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
        metrics = [m.lower() for m in metrics]

        with (
            cfdb.open_dataset(self.model_path) as ds_m,
            cfdb.open_dataset(self.observations_path) as ds_o,
        ):
            # Collect model and obs data
            var_model = {}
            var_obs = {}

            for var in variables:
                if var not in ds_m:
                    raise ValueError(f"Variable '{var}' not found in model dataset")
                if var not in ds_o:
                    raise ValueError(f"Variable '{var}' not found in observations dataset")

                m_var = ds_m[var]
                o_var = ds_o[var]
                n_dims_m = len(m_var.shape)
                height_idx = self._resolve_height_index(var)

                model_data = np.full((self.n_times, self.n_stations), np.nan, dtype=np.float64)
                obs_data = np.full((self.n_times, self.n_stations), np.nan, dtype=np.float64)

                interp = m_var.interp()
                interp_results = {}
                for time_val, values in interp.to_points(
                    o_var, order=self.interpolation_order
                ):
                    interp_results[time_val] = values

                for out_t, time_val in enumerate(self.time_values):
                    if time_val in interp_results:
                        vals = interp_results[time_val]
                        if vals.ndim > 1 and n_dims_m == 4:
                            model_data[out_t] = vals[height_idx]
                        else:
                            model_data[out_t] = vals

                    o_t_idx = int(self._obs_time_indices[out_t])
                    if self.obs_has_height:
                        obs_raw = o_var[(o_t_idx, 0, slice(None))].data
                        obs_data[out_t] = obs_raw[0, 0] if obs_raw.ndim > 1 else obs_raw
                    else:
                        obs_raw = o_var[(o_t_idx, slice(None))].data
                        obs_data[out_t] = obs_raw[0] if obs_raw.ndim > 1 else obs_raw

                var_model[var] = model_data
                var_obs[var] = obs_data

        # Write aggregated output
        with cfdb.open_dataset(output_path, 'n', dataset_type='grid') as ds_out:
            ds_out.create.coord.time(data=self.time_values)
            metric_indices = np.arange(len(metrics), dtype='int32')
            metric_coord = ds_out.create.coord.generic('metric', data=metric_indices, dtype='int32')
            metric_coord.attrs['flag_meanings'] = ' '.join(metrics)

            ds_out.attrs['model_path'] = str(self.model_path)
            ds_out.attrs['observations_path'] = str(self.observations_path)
            ds_out.attrs['evaluation_type'] = 'station_aggregate'

            for var in variables:
                out_var = ds_out.create.data_var.generic(
                    var, ('time', 'metric'), dtype='float64',
                    chunk_shape=(self.n_times, len(metrics)),
                )
                out_var.attrs['long_name'] = f'Station-aggregated metrics for {var}'

                model_data = var_model[var]
                obs_data = var_obs[var]

                for t in range(self.n_times):
                    results = np.zeros(len(metrics), dtype=np.float64)
                    m_row = model_data[t]
                    o_row = obs_data[t]
                    valid = ~(np.isnan(m_row) | np.isnan(o_row))

                    for m_idx, metric in enumerate(metrics):
                        if np.sum(valid) < 2:
                            results[m_idx] = np.nan
                        elif metric == 'bias':
                            results[m_idx] = compute_mean_bias(m_row[valid], o_row[valid])
                        elif metric == 'mae':
                            results[m_idx] = compute_mae_1d(m_row[valid], o_row[valid])
                        elif metric == 'rmse':
                            results[m_idx] = compute_rmse_1d(m_row[valid], o_row[valid])
                        elif metric == 'ne':
                            results[m_idx] = compute_ne_1d(m_row[valid], o_row[valid], epsilon)
                        elif metric == 'ane':
                            results[m_idx] = compute_ane_1d(m_row[valid], o_row[valid], epsilon)
                        elif metric == 'pearson':
                            results[m_idx] = compute_pearson_correlation(m_row[valid], o_row[valid])
                    out_var[(t, slice(None))] = results

        return output_path

    def evaluate_wind(
        self,
        output_path: Union[str, pathlib.Path],
        u_var: str = 'u_wind',
        v_var: str = 'v_wind',
        metrics: Union[str, List[str]] = 'vector_rmse',
    ) -> pathlib.Path:
        """
        Compute vector wind metrics at station locations.

        Parameters
        ----------
        output_path : str or pathlib.Path
            Path for output cfdb dataset.
        u_var : str
            U-component variable name.
        v_var : str
            V-component variable name.
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

        with (
            cfdb.open_dataset(self.model_path) as ds_m,
            cfdb.open_dataset(self.observations_path) as ds_o,
        ):
            # Extract model u/v at station locations
            for v in [u_var, v_var]:
                if v not in ds_m:
                    raise ValueError(f"Variable '{v}' not found in model dataset")
                if v not in ds_o:
                    raise ValueError(f"Variable '{v}' not found in observations dataset")

            height_idx_u = self._resolve_height_index(u_var)
            height_idx_v = self._resolve_height_index(v_var)

            def _extract(ds_m_var, ds_o_var, h_idx):
                n_dims = len(ds_m_var.shape)
                model_data = np.full((self.n_times, self.n_stations), np.nan, dtype=np.float64)
                obs_data = np.full((self.n_times, self.n_stations), np.nan, dtype=np.float64)

                interp = ds_m_var.interp()
                interp_results = {}
                for time_val, values in interp.to_points(ds_o_var, order=self.interpolation_order):
                    interp_results[time_val] = values

                for out_t, time_val in enumerate(self.time_values):
                    if time_val in interp_results:
                        vals = interp_results[time_val]
                        if vals.ndim > 1 and n_dims == 4:
                            model_data[out_t] = vals[h_idx]
                        else:
                            model_data[out_t] = vals

                    o_t_idx = int(self._obs_time_indices[out_t])
                    if self.obs_has_height:
                        obs_raw = ds_o_var[(o_t_idx, 0, slice(None))].data
                        obs_data[out_t] = obs_raw[0, 0] if obs_raw.ndim > 1 else obs_raw
                    else:
                        obs_raw = ds_o_var[(o_t_idx, slice(None))].data
                        obs_data[out_t] = obs_raw[0] if obs_raw.ndim > 1 else obs_raw

                return model_data, obs_data

            mu, ou = _extract(ds_m[u_var], ds_o[u_var], height_idx_u)
            mv, ov = _extract(ds_m[v_var], ds_o[v_var], height_idx_v)

        # Write output
        with cfdb.open_dataset(output_path, 'n', dataset_type='grid') as ds_out:
            ds_out.create.coord.time(data=self.time_values)
            metric_indices = np.arange(len(metrics), dtype='int32')
            metric_coord = ds_out.create.coord.generic('metric', data=metric_indices, dtype='int32')
            metric_coord.attrs['flag_meanings'] = ' '.join(metrics)

            ds_out.attrs['model_path'] = str(self.model_path)
            ds_out.attrs['observations_path'] = str(self.observations_path)
            ds_out.attrs['evaluation_type'] = 'station_wind'

            out_var = ds_out.create.data_var.generic(
                'wind', ('time', 'metric'), dtype='float32',
                chunk_shape=(self.n_times, len(metrics)),
            )
            out_var.attrs['long_name'] = 'Vector wind metrics at stations'

            for t in range(self.n_times):
                valid = ~(
                    np.isnan(mu[t]) | np.isnan(mv[t]) | np.isnan(ou[t]) | np.isnan(ov[t])
                )
                results = np.zeros(len(metrics), dtype=np.float32)
                for m_idx, metric in enumerate(metrics):
                    if np.sum(valid) == 0:
                        results[m_idx] = np.nan
                    elif metric == 'vector_rmse':
                        results[m_idx] = compute_vector_rmse(ou[t, valid], ov[t, valid], mu[t, valid], mv[t, valid])
                    elif metric == 'speed_bias':
                        results[m_idx] = compute_wind_speed_bias(ou[t, valid], ov[t, valid], mu[t, valid], mv[t, valid])
                    elif metric == 'direction_bias':
                        results[m_idx] = compute_wind_direction_bias(
                            ou[t, valid], ov[t, valid], mu[t, valid], mv[t, valid]
                        )
                out_var[(t, slice(None))] = results

        return output_path

    def evaluate_diurnal(
        self,
        output_path: Union[str, pathlib.Path],
        variables: List[str],
        metrics: Union[str, List[str]] = 'bias',
        utc_offset: float = 0.0,
    ) -> pathlib.Path:
        """
        Compute diurnal cycle of metrics per station.

        Parameters
        ----------
        output_path : str or pathlib.Path
            Path for output cfdb ts_ortho dataset.
        variables : list[str]
            Variable names to evaluate.
        metrics : str or list[str]
            Metric(s) to compute by hour: 'bias', 'rmse', 'mae', 'pearson'.
        utc_offset : float
            Hours to add to UTC to get local time.

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
        metrics = [m.lower() for m in metrics]

        with (
            cfdb.open_dataset(self.model_path) as ds_m,
            cfdb.open_dataset(self.observations_path) as ds_o,
        ):
            var_model = {}
            var_obs = {}

            for var in variables:
                if var not in ds_m:
                    raise ValueError(f"Variable '{var}' not found in model dataset")
                if var not in ds_o:
                    raise ValueError(f"Variable '{var}' not found in observations dataset")

                m_var = ds_m[var]
                o_var = ds_o[var]
                n_dims_m = len(m_var.shape)
                height_idx = self._resolve_height_index(var)

                model_data = np.full((self.n_times, self.n_stations), np.nan, dtype=np.float64)
                obs_data = np.full((self.n_times, self.n_stations), np.nan, dtype=np.float64)

                interp = m_var.interp()
                interp_results = {}
                for time_val, values in interp.to_points(o_var, order=self.interpolation_order):
                    interp_results[time_val] = values

                for out_t, time_val in enumerate(self.time_values):
                    if time_val in interp_results:
                        vals = interp_results[time_val]
                        if vals.ndim > 1 and n_dims_m == 4:
                            model_data[out_t] = vals[height_idx]
                        else:
                            model_data[out_t] = vals

                    o_t_idx = int(self._obs_time_indices[out_t])
                    if self.obs_has_height:
                        obs_raw = o_var[(o_t_idx, 0, slice(None))].data
                        obs_data[out_t] = obs_raw[0, 0] if obs_raw.ndim > 1 else obs_raw
                    else:
                        obs_raw = o_var[(o_t_idx, slice(None))].data
                        obs_data[out_t] = obs_raw[0] if obs_raw.ndim > 1 else obs_raw

                var_model[var] = model_data
                var_obs[var] = obs_data

        # Write output as ts_ortho with (hour, point) dims
        with cfdb.open_dataset(output_path, 'n', dataset_type='ts_ortho') as ds_out:
            geo_coord = ds_out.create.coord.point()
            geo_coord.append(list(self.station_points))
            hour_data = np.arange(24, dtype='int32')
            ds_out.create.coord.generic('hour', data=hour_data, dtype='int32')
            ds_out.create.crs.from_user_input(4326, xy_coord='point')

            ds_out.attrs['model_path'] = str(self.model_path)
            ds_out.attrs['observations_path'] = str(self.observations_path)
            ds_out.attrs['evaluation_type'] = 'station_diurnal'
            ds_out.attrs['utc_offset'] = str(utc_offset)

            for var in variables:
                model_data = var_model[var]
                obs_data = var_obs[var]

                for metric in metrics:
                    out_name = f"{var}_{metric}"
                    out_dv = ds_out.create.data_var.generic(
                        out_name, ('hour', 'point'), dtype='float64',
                        chunk_shape=(24, self.n_stations),
                    )
                    out_dv.attrs['long_name'] = f'Diurnal {metric.upper()} for {var}'

                    for s in range(self.n_stations):
                        m_ts = model_data[:, s]
                        o_ts = obs_data[:, s]
                        valid = ~(np.isnan(m_ts) | np.isnan(o_ts))
                        if np.sum(valid) < 2:
                            for h in range(24):
                                out_dv[(h, s)] = np.array([np.nan], dtype=np.float64)
                            continue

                        _, values = compute_diurnal_stats(
                            self.time_values[valid], m_ts[valid], o_ts[valid],
                            metric, utc_offset
                        )
                        for h in range(24):
                            out_dv[(h, s)] = np.array([values[h]], dtype=np.float64)

        return output_path

    def evaluate_lagged_correlation(
        self,
        output_path: Union[str, pathlib.Path],
        variables: List[str],
        max_lag: int = None,
    ) -> pathlib.Path:
        """
        Compute lagged cross-correlation per station to detect timing offsets.

        A positive optimal lag means the model leads (event arrives early in the
        model relative to observations). A negative optimal lag means the model
        lags behind.

        Parameters
        ----------
        output_path : str or pathlib.Path
            Path for output cfdb ts_ortho dataset.
        variables : list[str]
            Variable names to evaluate.
        max_lag : int, optional
            Maximum lag in timesteps. Default is n_times // 4.

        Returns
        -------
        pathlib.Path
            Path to the output cfdb dataset with variables:
            ``{var}_lag_correlations`` (lag, station) and
            ``{var}_optimal_lag`` (station,) and
            ``{var}_peak_correlation`` (station,).
        """
        output_path = pathlib.Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(variables, str):
            variables = [variables]

        if max_lag is None:
            max_lag = self.n_times // 4
        max_lag = min(max_lag, self.n_times - 2)
        n_lags = 2 * max_lag + 1

        with (
            cfdb.open_dataset(self.model_path) as ds_m,
            cfdb.open_dataset(self.observations_path) as ds_o,
        ):
            var_model = {}
            var_obs = {}

            for var in variables:
                if var not in ds_m:
                    raise ValueError(f"Variable '{var}' not found in model dataset")
                if var not in ds_o:
                    raise ValueError(f"Variable '{var}' not found in observations dataset")

                m_var = ds_m[var]
                o_var = ds_o[var]
                n_dims_m = len(m_var.shape)
                height_idx = self._resolve_height_index(var)

                model_data = np.full((self.n_times, self.n_stations), np.nan, dtype=np.float64)
                obs_data = np.full((self.n_times, self.n_stations), np.nan, dtype=np.float64)

                interp = m_var.interp()
                interp_results = {}
                for time_val, values in interp.to_points(o_var, order=self.interpolation_order):
                    interp_results[time_val] = values

                for out_t, time_val in enumerate(self.time_values):
                    if time_val in interp_results:
                        vals = interp_results[time_val]
                        if vals.ndim > 1 and n_dims_m == 4:
                            model_data[out_t] = vals[height_idx]
                        else:
                            model_data[out_t] = vals

                    o_t_idx = int(self._obs_time_indices[out_t])
                    if self.obs_has_height:
                        obs_raw = o_var[(o_t_idx, 0, slice(None))].data
                        obs_data[out_t] = obs_raw[0, 0] if obs_raw.ndim > 1 else obs_raw
                    else:
                        obs_raw = o_var[(o_t_idx, slice(None))].data
                        obs_data[out_t] = obs_raw[0] if obs_raw.ndim > 1 else obs_raw

                var_model[var] = model_data
                var_obs[var] = obs_data

        # Write output
        with cfdb.open_dataset(output_path, 'n', dataset_type='ts_ortho') as ds_out:
            geo_coord = ds_out.create.coord.point()
            geo_coord.append(list(self.station_points))

            lag_data = np.arange(-max_lag, max_lag + 1, dtype='int32')
            ds_out.create.coord.generic('lag', data=lag_data, dtype='int32')
            ds_out.create.crs.from_user_input(4326, xy_coord='point')

            ds_out.attrs['model_path'] = str(self.model_path)
            ds_out.attrs['observations_path'] = str(self.observations_path)
            ds_out.attrs['evaluation_type'] = 'lagged_correlation'
            ds_out.attrs['max_lag'] = str(max_lag)

            for var in variables:
                model_data = var_model[var]
                obs_data = var_obs[var]

                # Full correlation curves: (lag, station)
                corr_var = ds_out.create.data_var.generic(
                    f'{var}_lag_correlations', ('lag', 'point'), dtype='float64',
                    chunk_shape=(n_lags, self.n_stations),
                )
                corr_var.attrs['long_name'] = f'Lagged correlation for {var}'

                # Optimal lag per station: (station,)
                opt_var = ds_out.create.data_var.generic(
                    f'{var}_optimal_lag', ('point',), dtype='int32',
                    chunk_shape=(self.n_stations,),
                )
                opt_var.attrs['long_name'] = f'Optimal lag (timesteps) for {var}'
                opt_var.attrs['description'] = 'Positive = model leads, negative = model lags'

                # Peak correlation per station: (station,)
                peak_var = ds_out.create.data_var.generic(
                    f'{var}_peak_correlation', ('point',), dtype='float64',
                    chunk_shape=(self.n_stations,),
                )
                peak_var.attrs['long_name'] = f'Peak correlation for {var}'

                optimal_lags = np.zeros(self.n_stations, dtype=np.int32)
                peak_corrs = np.full(self.n_stations, np.nan)

                for s in range(self.n_stations):
                    lags, corrs = compute_lagged_correlation(
                        model_data[:, s], obs_data[:, s], max_lag
                    )
                    # Pad/align to the output lag array
                    corr_out = np.full(n_lags, np.nan)
                    for li, lag_val in enumerate(lags):
                        out_idx = lag_val + max_lag
                        if 0 <= out_idx < n_lags:
                            corr_out[out_idx] = corrs[li]

                    for li in range(n_lags):
                        corr_var[(li, s)] = np.array([corr_out[li]], dtype=np.float64)

                    valid_corrs = ~np.isnan(corr_out)
                    if np.any(valid_corrs):
                        best_idx = np.nanargmax(corr_out)
                        optimal_lags[s] = lag_data[best_idx]
                        peak_corrs[s] = corr_out[best_idx]

                opt_var[(slice(None),)] = optimal_lags
                peak_var[(slice(None),)] = peak_corrs

        return output_path
