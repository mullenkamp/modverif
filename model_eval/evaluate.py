"""
Functions for evaluating WRF model outputs.
"""
import pathlib
from datetime import date
from typing import Union, List, Tuple

import numpy as np

from model_eval.evaluator import WRFEvaluator, find_wrfout_files, _get_wrf_proj4, _find_latlon_bounds
from model_eval.wrfio import WRFFile
from model_eval.cyclone import (
    CyclonePosition,
    _compute_sea_level_pressure,
    _estimate_cyclone_radius,
    _find_pressure_minimum,
    _grid_distances_km,
    _haversine_distance,
)

def evaluate_models_cell(
    source_folder: Union[str, pathlib.Path],
    test_folder: Union[str, pathlib.Path],
    output_path: Union[str, pathlib.Path],
    domain: int,
    variables: List[str],
    metrics: Union[str, List[str]] = 'ne',
    region: Union[Tuple[float, float, float, float], np.ndarray, None] = None,
    start_date: Union[str, date] = None,
    end_date: Union[str, date] = None,
    threshold: float = None,
    epsilon: float = 1e-10,
    max_memory_bytes: int = 2**29,
) -> pathlib.Path:
    """Wrapper for WRFEvaluator.evaluate_cell."""
    evaluator = WRFEvaluator(source_folder, test_folder, domain, region, start_date, end_date)
    return evaluator.evaluate_cell(output_path, variables, metrics, threshold, epsilon, max_memory_bytes)

def evaluate_models_domain(
    source_folder: Union[str, pathlib.Path],
    test_folder: Union[str, pathlib.Path],
    output_path: Union[str, pathlib.Path],
    domain: int,
    variables: List[str],
    metrics: Union[str, List[str]] = 'ne',
    region: Union[Tuple[float, float, float, float], np.ndarray, None] = None,
    start_date: Union[str, date] = None,
    end_date: Union[str, date] = None,
    threshold: float = None,
    epsilon: float = 1e-10,
    max_memory_bytes: int = 2**29,
) -> pathlib.Path:
    """Wrapper for WRFEvaluator.evaluate_domain."""
    evaluator = WRFEvaluator(source_folder, test_folder, domain, region, start_date, end_date)
    return evaluator.evaluate_domain(output_path, variables, metrics, threshold, epsilon, max_memory_bytes)

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
    domain-aggregated metrics over each model's own cyclone region.
    """
    from model_eval.metrics import AVAILABLE_DOMAIN_METRICS
    from model_eval.wrfio import NetCDF4Writer

    source_path = pathlib.Path(source_path)
    test_path = pathlib.Path(test_path)
    output_path = pathlib.Path(output_path)

    if isinstance(metrics, str):
        metrics = [metrics]
    metrics = [m.lower() for m in metrics]

    for m in metrics:
        if m not in AVAILABLE_DOMAIN_METRICS:
            raise ValueError(f"Unknown metric '{m}'. Available: {AVAILABLE_DOMAIN_METRICS}")

    with WRFFile(source_path) as wrf_s, WRFFile(test_path) as wrf_t:
        required_vars = ['PSFC', 'HGT', 'T2', 'XLAT', 'XLONG']
        for var in required_vars:
            if not wrf_s.has_variable(var): raise ValueError(f"Required variable '{var}' not found in {source_path}")
            if not wrf_t.has_variable(var): raise ValueError(f"Required variable '{var}' not found in {test_path}")

        for var in variables:
            if not wrf_s.has_variable(var): raise ValueError(f"Variable '{var}' not found in {source_path}")
            if not wrf_t.has_variable(var): raise ValueError(f"Variable '{var}' not found in {test_path}")

        n_times = min(wrf_s.n_times, wrf_t.n_times)
        xlat_s, xlong_s = wrf_s.xlat, wrf_s.xlong
        xlat_t, xlong_t = wrf_t.xlat, wrf_t.xlong
        time_values = wrf_s.time_values[:n_times] if wrf_s.time_values is not None else None

        source_positions, test_positions = [], []
        current_lat_s, current_lon_s = start_lat, start_lon
        current_lat_t, current_lon_t = start_lat, start_lon

        var_results = {var: np.zeros((n_times, len(metrics)), dtype=np.float32) for var in variables}

        for t in range(n_times):
            # Track Source
            slp_s = wrf_s.get_slp(t, smoothing_sigma=smoothing_sigma)
            y_idx_s, x_idx_s, min_p_s = _find_pressure_minimum(slp_s, xlat_s, xlong_s, current_lat_s, current_lon_s, search_radius_km)
            center_lat_s, center_lon_s = float(xlat_s[y_idx_s, x_idx_s]), float(xlong_s[y_idx_s, x_idx_s])
            radius_s = _estimate_cyclone_radius(slp_s, xlat_s, xlong_s, y_idx_s, x_idx_s, pressure_threshold_pa, max_cyclone_radius_km)
            source_positions.append(CyclonePosition(t, y_idx_s, x_idx_s, center_lat_s, center_lon_s, min_p_s, radius_s))
            current_lat_s, current_lon_s = center_lat_s, center_lon_s

            # Track Test
            slp_t = wrf_t.get_slp(t, smoothing_sigma=smoothing_sigma)
            y_idx_t, x_idx_t, min_p_t = _find_pressure_minimum(slp_t, xlat_t, xlong_t, current_lat_t, current_lon_t, search_radius_km)
            center_lat_t, center_lon_t = float(xlat_t[y_idx_t, x_idx_t]), float(xlong_t[y_idx_t, x_idx_t])
            radius_t = _estimate_cyclone_radius(slp_t, xlat_t, xlong_t, y_idx_t, x_idx_t, pressure_threshold_pa, max_cyclone_radius_km)
            test_positions.append(CyclonePosition(t, y_idx_t, x_idx_t, center_lat_t, center_lon_t, min_p_t, radius_t))
            current_lat_t, current_lon_t = center_lat_t, center_lon_t

            mask_s = _grid_distances_km(xlat_s, xlong_s, center_lat_s, center_lon_s) <= radius_s
            mask_t = _grid_distances_km(xlat_t, xlong_t, center_lat_t, center_lon_t) <= radius_t

            for var in variables:
                s_data, t_data = wrf_s.get_variable(var, t), wrf_t.get_variable(var, t)
                for m_idx, metric in enumerate(metrics):
                    if metric == 'ne':
                        s_sum, t_sum = np.sum(np.where(mask_s, s_data, 0.0)), np.sum(np.where(mask_t, t_data, 0.0))
                        var_results[var][t, m_idx] = ((t_sum - s_sum) / s_sum * 100) if np.abs(s_sum) >= epsilon else 0.0
                    elif metric == 'ane':
                        s_sum, t_sum = np.sum(np.where(mask_s, s_data, 0.0)), np.sum(np.where(mask_t, t_data, 0.0))
                        var_results[var][t, m_idx] = np.abs((t_sum - s_sum) / s_sum * 100) if np.abs(s_sum) >= epsilon else 0.0
                    elif metric == 'rmse':
                        s_mean = np.sum(np.where(mask_s, s_data, 0.0)) / max(np.sum(mask_s), 1)
                        t_mean = np.sum(np.where(mask_t, t_data, 0.0)) / max(np.sum(mask_t), 1)
                        var_results[var][t, m_idx] = np.abs(t_mean - s_mean)

    with NetCDF4Writer(output_path) as nc:
        nc.set_global_attrs(source_file=str(source_path), test_file=str(test_path), evaluation_type='cyclone')
        time_ds = nc.create_time_dimension(n_times, data=time_values)
        metric_ds = nc.create_metric_dimension(metrics)

        # Track Vars (Source)
        nc.attach_scales(nc.create_variable('source_latitude', (n_times,), data=np.array([p.latitude for p in source_positions])), [time_ds])
        nc.attach_scales(nc.create_variable('source_longitude', (n_times,), data=np.array([p.longitude for p in source_positions])), [time_ds])
        nc.attach_scales(nc.create_variable('source_pressure', (n_times,), data=np.array([p.central_pressure for p in source_positions])), [time_ds])
        nc.attach_scales(nc.create_variable('source_radius', (n_times,), data=np.array([p.radius_km for p in source_positions])), [time_ds])
        # Track Vars (Test)
        nc.attach_scales(nc.create_variable('test_latitude', (n_times,), data=np.array([p.latitude for p in test_positions])), [time_ds])
        nc.attach_scales(nc.create_variable('test_longitude', (n_times,), data=np.array([p.longitude for p in test_positions])), [time_ds])
        nc.attach_scales(nc.create_variable('test_pressure', (n_times,), data=np.array([p.central_pressure for p in test_positions])), [time_ds])
        nc.attach_scales(nc.create_variable('test_radius', (n_times,), data=np.array([p.radius_km for p in test_positions])), [time_ds])

        # Comparison
        pos_diff = np.array([_haversine_distance(source_positions[t].latitude, source_positions[t].longitude, test_positions[t].latitude, test_positions[t].longitude) for t in range(n_times)], dtype=np.float32)
        nc.attach_scales(nc.create_variable('position_difference_km', (n_times,), data=pos_diff), [time_ds])

        pres_diff = np.array([test_positions[t].central_pressure - source_positions[t].central_pressure for t in range(n_times)], dtype=np.float32)
        nc.attach_scales(nc.create_variable('pressure_difference', (n_times,), data=pres_diff), [time_ds])

        rad_diff = np.array([test_positions[t].radius_km - source_positions[t].radius_km for t in range(n_times)], dtype=np.float32)
        nc.attach_scales(nc.create_variable('radius_difference', (n_times,), data=rad_diff), [time_ds])

        for var in variables:
            out_ds = nc.create_variable(var, (n_times, len(metrics)), data=var_results[var])
            nc.attach_scales(out_ds, [time_ds, metric_ds])

    return output_path
