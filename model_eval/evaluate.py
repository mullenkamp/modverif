"""
Convenience functions for evaluating model outputs stored in cfdb datasets.
"""
import pathlib
from typing import Union, List, Tuple

import cfdb
import numpy as np

from model_eval.evaluator import Evaluator
from model_eval.station import StationEvaluator
from model_eval.cyclone import (
    CyclonePosition,
    _estimate_cyclone_radius,
    _find_pressure_minimum,
    _grid_distances_km,
    _haversine_distance,
    _read_latlon_2d,
    _read_slp_from_cfdb,
    _read_var_2d,
)


def evaluate_models_cell(
    source: Union[str, pathlib.Path],
    test: Union[str, pathlib.Path],
    output_path: Union[str, pathlib.Path],
    variables: List[str],
    metrics: Union[str, List[str]] = 'ne',
    region: Union[Tuple[float, float, float, float], np.ndarray, None] = None,
    start_time: Union[str, np.datetime64, None] = None,
    end_time: Union[str, np.datetime64, None] = None,
    threshold: float = None,
    epsilon: float = 1e-10,
) -> pathlib.Path:
    """
    Evaluate two model runs at cell level.

    Convenience wrapper around :class:`~model_eval.evaluator.Evaluator`.

    Parameters
    ----------
    source : str or pathlib.Path
        Path to source/reference cfdb dataset.
    test : str or pathlib.Path
        Path to test cfdb dataset.
    output_path : str or pathlib.Path
        Path for the output cfdb dataset.
    variables : list[str]
        cfdb variable names to evaluate (e.g. ``['air_temperature', 'u_wind']``).
    metrics : str or list[str]
        Metric(s) to compute. Default is ``'ne'``.
    region : tuple or np.ndarray, optional
        Bounding box ``(min_lon, min_lat, max_lon, max_lat)`` or 2D boolean mask.
    start_time : str or np.datetime64, optional
        Start of evaluation period (inclusive).
    end_time : str or np.datetime64, optional
        End of evaluation period (inclusive).
    threshold : float, optional
        Threshold for categorical metrics.
    epsilon : float
        Small value to avoid division by zero.

    Returns
    -------
    pathlib.Path
        Path to the output cfdb dataset.
    """
    evaluator = Evaluator(source, test, region, start_time, end_time)
    return evaluator.evaluate_cell(output_path, variables, metrics, threshold, epsilon)


def evaluate_models_domain(
    source: Union[str, pathlib.Path],
    test: Union[str, pathlib.Path],
    output_path: Union[str, pathlib.Path],
    variables: List[str],
    metrics: Union[str, List[str]] = 'ne',
    region: Union[Tuple[float, float, float, float], np.ndarray, None] = None,
    start_time: Union[str, np.datetime64, None] = None,
    end_time: Union[str, np.datetime64, None] = None,
    threshold: float = None,
    epsilon: float = 1e-10,
) -> pathlib.Path:
    """
    Evaluate two model runs at domain-aggregated level.

    Convenience wrapper around :class:`~model_eval.evaluator.Evaluator`.

    Parameters
    ----------
    source : str or pathlib.Path
        Path to source/reference cfdb dataset.
    test : str or pathlib.Path
        Path to test cfdb dataset.
    output_path : str or pathlib.Path
        Path for the output cfdb dataset.
    variables : list[str]
        cfdb variable names to evaluate.
    metrics : str or list[str]
        Metric(s) to compute. Default is ``'ne'``.
    region : tuple or np.ndarray, optional
        Bounding box ``(min_lon, min_lat, max_lon, max_lat)`` or 2D boolean mask.
    start_time : str or np.datetime64, optional
        Start of evaluation period (inclusive).
    end_time : str or np.datetime64, optional
        End of evaluation period (inclusive).
    threshold : float, optional
        Threshold for categorical metrics.
    epsilon : float
        Small value to avoid division by zero.

    Returns
    -------
    pathlib.Path
        Path to the output cfdb dataset.
    """
    evaluator = Evaluator(source, test, region, start_time, end_time)
    return evaluator.evaluate_domain(output_path, variables, metrics, threshold, epsilon)


def evaluate_cyclones(
    source: Union[str, pathlib.Path],
    test: Union[str, pathlib.Path],
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
    Evaluate two model runs containing the same cyclone.

    Tracks the cyclone independently in both source and test datasets,
    then computes domain-aggregated metrics over each model's own cyclone
    region at each timestep. Output includes cyclone track positions,
    track differences, and per-variable metric time series.

    Parameters
    ----------
    source : str or pathlib.Path
        Path to source/reference cfdb dataset.
    test : str or pathlib.Path
        Path to test cfdb dataset.
    output_path : str or pathlib.Path
        Path for the output cfdb dataset.
    variables : list[str]
        cfdb variable names to evaluate within the cyclone region.
    metrics : str or list[str]
        Domain-aggregated metric(s) to compute. Default is ``'ne'``.
    start_lat : float, optional
        Initial search latitude. If None, uses global pressure minimum at t=0.
    start_lon : float, optional
        Initial search longitude. If None, uses global pressure minimum at t=0.
    search_radius_km : float
        Radius in km to search for pressure minimum. Default is 500 km.
    pressure_threshold_pa : float
        Pressure threshold for cyclone edge detection. Default is 400 Pa.
    max_cyclone_radius_km : float
        Maximum cyclone radius. Default is 1000 km.
    smoothing_sigma : float, optional
        Gaussian smoothing sigma for SLP field. If None, no smoothing.
    epsilon : float
        Small value to avoid division by zero.

    Returns
    -------
    pathlib.Path
        Path to the output cfdb dataset.
    """
    from model_eval.metrics import AVAILABLE_DOMAIN_METRICS

    source = pathlib.Path(source)
    test = pathlib.Path(test)
    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(metrics, str):
        metrics = [metrics]
    metrics = [m.lower() for m in metrics]

    for m in metrics:
        if m not in AVAILABLE_DOMAIN_METRICS:
            raise ValueError(f"Unknown metric '{m}'. Available: {AVAILABLE_DOMAIN_METRICS}")

    with cfdb.open_dataset(source) as ds_s, cfdb.open_dataset(test) as ds_t:
        xlat_s, xlong_s = _read_latlon_2d(ds_s)
        xlat_t, xlong_t = _read_latlon_2d(ds_t)

        s_time = ds_s['time'].data
        t_time = ds_t['time'].data
        common_times = np.intersect1d(s_time, t_time)
        n_times = len(common_times)
        if n_times == 0:
            raise ValueError("No common timesteps found between source and test datasets")

        s_time_indices = np.searchsorted(s_time, common_times)
        t_time_indices = np.searchsorted(t_time, common_times)

        # Validate variables exist
        for var in variables:
            if var not in ds_s:
                raise ValueError(f"Variable '{var}' not found in source dataset")
            if var not in ds_t:
                raise ValueError(f"Variable '{var}' not found in test dataset")

        source_positions = []
        test_positions = []
        current_lat_s, current_lon_s = start_lat, start_lon
        current_lat_t, current_lon_t = start_lat, start_lon

        var_results = {var: np.zeros((n_times, len(metrics)), dtype=np.float32) for var in variables}

        for out_t, (s_t_idx, t_t_idx) in enumerate(zip(s_time_indices, t_time_indices)):
            # Track source cyclone
            slp_s = _read_slp_from_cfdb(ds_s, int(s_t_idx), smoothing_sigma=smoothing_sigma)
            if out_t == 0 and current_lat_s is None:
                y_s, x_s, p_s = _find_pressure_minimum(slp_s, xlat_s, xlong_s)
            else:
                y_s, x_s, p_s = _find_pressure_minimum(
                    slp_s, xlat_s, xlong_s, current_lat_s, current_lon_s, search_radius_km
                )
            lat_s = float(xlat_s[y_s, x_s])
            lon_s = float(xlong_s[y_s, x_s])
            rad_s = _estimate_cyclone_radius(
                slp_s, xlat_s, xlong_s, y_s, x_s, pressure_threshold_pa, max_cyclone_radius_km
            )
            source_positions.append(CyclonePosition(out_t, y_s, x_s, lat_s, lon_s, p_s, rad_s))
            current_lat_s, current_lon_s = lat_s, lon_s

            # Track test cyclone
            slp_t = _read_slp_from_cfdb(ds_t, int(t_t_idx), smoothing_sigma=smoothing_sigma)
            if out_t == 0 and current_lat_t is None:
                y_t, x_t, p_t = _find_pressure_minimum(slp_t, xlat_t, xlong_t)
            else:
                y_t, x_t, p_t = _find_pressure_minimum(
                    slp_t, xlat_t, xlong_t, current_lat_t, current_lon_t, search_radius_km
                )
            lat_t = float(xlat_t[y_t, x_t])
            lon_t = float(xlong_t[y_t, x_t])
            rad_t = _estimate_cyclone_radius(
                slp_t, xlat_t, xlong_t, y_t, x_t, pressure_threshold_pa, max_cyclone_radius_km
            )
            test_positions.append(CyclonePosition(out_t, y_t, x_t, lat_t, lon_t, p_t, rad_t))
            current_lat_t, current_lon_t = lat_t, lon_t

            # Build spatial masks for each cyclone
            mask_s = _grid_distances_km(xlat_s, xlong_s, lat_s, lon_s) <= rad_s
            mask_t = _grid_distances_km(xlat_t, xlong_t, lat_t, lon_t) <= rad_t

            # Compute metrics for each variable
            for var in variables:
                s_data = _read_var_2d(ds_s, var, int(s_t_idx))
                t_data = _read_var_2d(ds_t, var, int(t_t_idx))

                for m_idx, metric in enumerate(metrics):
                    if metric == 'ne':
                        s_sum = np.sum(np.where(mask_s, s_data, 0.0))
                        t_sum = np.sum(np.where(mask_t, t_data, 0.0))
                        val = ((t_sum - s_sum) / s_sum * 100) if np.abs(s_sum) >= epsilon else 0.0
                    elif metric == 'ane':
                        s_sum = np.sum(np.where(mask_s, s_data, 0.0))
                        t_sum = np.sum(np.where(mask_t, t_data, 0.0))
                        val = np.abs((t_sum - s_sum) / s_sum * 100) if np.abs(s_sum) >= epsilon else 0.0
                    elif metric == 'rmse':
                        s_mean = np.sum(np.where(mask_s, s_data, 0.0)) / max(np.sum(mask_s), 1)
                        t_mean = np.sum(np.where(mask_t, t_data, 0.0)) / max(np.sum(mask_t), 1)
                        val = np.abs(t_mean - s_mean)
                    elif metric == 'bias':
                        s_mean = np.sum(np.where(mask_s, s_data, 0.0)) / max(np.sum(mask_s), 1)
                        t_mean = np.sum(np.where(mask_t, t_data, 0.0)) / max(np.sum(mask_t), 1)
                        val = t_mean - s_mean
                    else:
                        val = 0.0
                    var_results[var][out_t, m_idx] = val

    # Write output to cfdb
    with cfdb.open_dataset(output_path, 'n', dataset_type='grid') as ds_out:
        time_coord = ds_out.create.coord.time(data=common_times)
        metric_indices = np.arange(len(metrics), dtype='int32')
        metric_coord = ds_out.create.coord.generic('metric', data=metric_indices, dtype='int32')
        metric_coord.attrs['flag_meanings'] = ' '.join(metrics)

        ds_out.attrs['source_path'] = str(source)
        ds_out.attrs['test_path'] = str(test)
        ds_out.attrs['evaluation_type'] = 'cyclone'

        # Source track variables
        for prefix, positions in [('source', source_positions), ('test', test_positions)]:
            lat_arr = np.array([p.latitude for p in positions], dtype='float32')
            lon_arr = np.array([p.longitude for p in positions], dtype='float32')
            pres_arr = np.array([p.central_pressure for p in positions], dtype='float32')
            rad_arr = np.array([p.radius_km for p in positions], dtype='float32')

            for name, data, units in [
                (f'{prefix}_latitude', lat_arr, 'degrees_north'),
                (f'{prefix}_longitude', lon_arr, 'degrees_east'),
                (f'{prefix}_pressure', pres_arr, 'Pa'),
                (f'{prefix}_radius', rad_arr, 'km'),
            ]:
                v = ds_out.create.data_var.generic(name, ('time',), dtype='float32')
                v.attrs['units'] = units
                for i, val in enumerate(data):
                    v[(i,)] = np.array([val], dtype='float32')

        # Track difference variables
        pos_diff = np.array([
            _haversine_distance(
                source_positions[t].latitude, source_positions[t].longitude,
                test_positions[t].latitude, test_positions[t].longitude
            ) for t in range(n_times)
        ], dtype='float32')
        pres_diff = np.array([
            test_positions[t].central_pressure - source_positions[t].central_pressure
            for t in range(n_times)
        ], dtype='float32')
        rad_diff = np.array([
            test_positions[t].radius_km - source_positions[t].radius_km
            for t in range(n_times)
        ], dtype='float32')

        for name, data, units in [
            ('position_difference_km', pos_diff, 'km'),
            ('pressure_difference', pres_diff, 'Pa'),
            ('radius_difference', rad_diff, 'km'),
        ]:
            v = ds_out.create.data_var.generic(name, ('time',), dtype='float32')
            v.attrs['units'] = units
            for i, val in enumerate(data):
                v[(i,)] = np.array([val], dtype='float32')

        # Per-variable metric results
        for var in variables:
            v = ds_out.create.data_var.generic(var, ('time', 'metric'), dtype='float32')
            v.attrs['long_name'] = f'Cyclone-region metrics for {var}'
            for t in range(n_times):
                v[(t, slice(None))] = var_results[var][t]

    return output_path


def evaluate_stations(
    model: Union[str, pathlib.Path],
    observations: Union[str, pathlib.Path],
    output_path: Union[str, pathlib.Path],
    variables: List[str],
    metrics: Union[str, List[str]] = 'bias',
    start_time: Union[str, np.datetime64, None] = None,
    end_time: Union[str, np.datetime64, None] = None,
    interpolation_order: int = 1,
    height: float = None,
    variable_heights: dict = None,
) -> pathlib.Path:
    """
    Evaluate model output against station observations.

    Convenience wrapper around :class:`~model_eval.station.StationEvaluator`.

    Parameters
    ----------
    model : str or pathlib.Path
        Path to cfdb grid dataset (model output).
    observations : str or pathlib.Path
        Path to cfdb ts_ortho dataset (station observations).
    output_path : str or pathlib.Path
        Path for the output cfdb dataset.
    variables : list[str]
        Variable names to evaluate.
    metrics : str or list[str]
        Metric(s) to compute. Default is ``'bias'``.
    start_time : str or np.datetime64, optional
        Start of evaluation period (inclusive).
    end_time : str or np.datetime64, optional
        End of evaluation period (inclusive).
    interpolation_order : int
        Spatial interpolation order (0=nearest, 1=linear, 3=cubic).
    height : float, optional
        Default target model height level in meters.
    variable_heights : dict, optional
        Per-variable target heights.

    Returns
    -------
    pathlib.Path
        Path to the output cfdb dataset.
    """
    evaluator = StationEvaluator(
        model, observations, start_time, end_time,
        interpolation_order, height, variable_heights,
    )
    return evaluator.evaluate(output_path, variables, metrics)


def evaluate_stations_aggregate(
    model: Union[str, pathlib.Path],
    observations: Union[str, pathlib.Path],
    output_path: Union[str, pathlib.Path],
    variables: List[str],
    metrics: Union[str, List[str]] = 'bias',
    start_time: Union[str, np.datetime64, None] = None,
    end_time: Union[str, np.datetime64, None] = None,
    interpolation_order: int = 1,
    height: float = None,
    variable_heights: dict = None,
) -> pathlib.Path:
    """
    Evaluate model vs station observations with aggregation over stations.

    Convenience wrapper around :class:`~model_eval.station.StationEvaluator`.

    Parameters
    ----------
    model : str or pathlib.Path
        Path to cfdb grid dataset (model output).
    observations : str or pathlib.Path
        Path to cfdb ts_ortho dataset (station observations).
    output_path : str or pathlib.Path
        Path for the output cfdb dataset.
    variables : list[str]
        Variable names to evaluate.
    metrics : str or list[str]
        Metric(s) to compute. Default is ``'bias'``.
    start_time : str or np.datetime64, optional
        Start of evaluation period (inclusive).
    end_time : str or np.datetime64, optional
        End of evaluation period (inclusive).
    interpolation_order : int
        Spatial interpolation order (0=nearest, 1=linear, 3=cubic).
    height : float, optional
        Default target model height level in meters.
    variable_heights : dict, optional
        Per-variable target heights.

    Returns
    -------
    pathlib.Path
        Path to the output cfdb dataset.
    """
    evaluator = StationEvaluator(
        model, observations, start_time, end_time,
        interpolation_order, height, variable_heights,
    )
    return evaluator.evaluate_aggregate(output_path, variables, metrics)


def evaluate_fss(
    source: Union[str, pathlib.Path],
    test: Union[str, pathlib.Path],
    output_path: Union[str, pathlib.Path],
    variables: List[str],
    threshold: float,
    neighborhood_sizes: list = None,
    region: Union[Tuple[float, float, float, float], np.ndarray, None] = None,
    start_time: Union[str, np.datetime64, None] = None,
    end_time: Union[str, np.datetime64, None] = None,
) -> pathlib.Path:
    """
    Compute Fractions Skill Score across multiple spatial scales.

    Convenience wrapper around :meth:`~model_eval.evaluator.Evaluator.evaluate_fss`.

    Parameters
    ----------
    source : str or pathlib.Path
        Path to source/reference cfdb dataset.
    test : str or pathlib.Path
        Path to test cfdb dataset.
    output_path : str or pathlib.Path
        Path for the output cfdb dataset.
    variables : list[str]
        Variable names to evaluate.
    threshold : float
        Binary event threshold.
    neighborhood_sizes : list[int], optional
        Neighborhood sizes. Default: [1, 3, 5, 9, 17, 33, 65].
    region : tuple or np.ndarray, optional
        Bounding box or 2D boolean mask.
    start_time : str or np.datetime64, optional
        Start of evaluation period (inclusive).
    end_time : str or np.datetime64, optional
        End of evaluation period (inclusive).

    Returns
    -------
    pathlib.Path
        Path to the output cfdb dataset.
    """
    evaluator = Evaluator(source, test, region, start_time, end_time)
    return evaluator.evaluate_fss(output_path, variables, threshold, neighborhood_sizes)


def evaluate_wind(
    source: Union[str, pathlib.Path],
    test: Union[str, pathlib.Path],
    output_path: Union[str, pathlib.Path],
    u_var: str = 'u_wind',
    v_var: str = 'v_wind',
    metrics: Union[str, List[str]] = 'vector_rmse',
    region: Union[Tuple[float, float, float, float], np.ndarray, None] = None,
    start_time: Union[str, np.datetime64, None] = None,
    end_time: Union[str, np.datetime64, None] = None,
) -> pathlib.Path:
    """
    Compute vector wind metrics from U/V components.

    Convenience wrapper around :meth:`~model_eval.evaluator.Evaluator.evaluate_wind`.

    Parameters
    ----------
    source : str or pathlib.Path
        Path to source/reference cfdb dataset.
    test : str or pathlib.Path
        Path to test cfdb dataset.
    output_path : str or pathlib.Path
        Path for the output cfdb dataset.
    u_var : str
        Name of U-component variable. Default: ``'u_wind'``.
    v_var : str
        Name of V-component variable. Default: ``'v_wind'``.
    metrics : str or list[str]
        Wind metric(s). Default: ``'vector_rmse'``.
    region : tuple or np.ndarray, optional
        Bounding box or 2D boolean mask.
    start_time : str or np.datetime64, optional
        Start of evaluation period (inclusive).
    end_time : str or np.datetime64, optional
        End of evaluation period (inclusive).

    Returns
    -------
    pathlib.Path
        Path to the output cfdb dataset.
    """
    evaluator = Evaluator(source, test, region, start_time, end_time)
    return evaluator.evaluate_wind(output_path, u_var, v_var, metrics)
