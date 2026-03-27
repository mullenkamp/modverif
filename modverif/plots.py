"""
Verification plots for model evaluation results.

All functions return (fig, ax) tuples and optionally save to file.
Cartopy is optional for geographic plots.
"""
import pathlib
from typing import Union

import matplotlib.pyplot as plt
import numpy as np

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False


def plot_scatter(
    model_values: np.ndarray,
    obs_values: np.ndarray,
    save_path: Union[str, pathlib.Path] = None,
    variable_name: str = None,
    units: str = None,
    figsize: tuple = (8, 8),
    dpi: int = 150,
    show_stats: bool = True,
    density: bool = False,
) -> tuple:
    """
    Scatter plot of model vs observed values with 1:1 reference line.

    Parameters
    ----------
    model_values : np.ndarray
        Model/forecast values.
    obs_values : np.ndarray
        Observed/reference values.
    save_path : str or pathlib.Path, optional
        Path to save figure.
    variable_name : str, optional
        Variable name for axis labels.
    units : str, optional
        Units string for axis labels.
    figsize : tuple
        Figure size.
    dpi : int
        Figure DPI.
    show_stats : bool
        Show statistics annotation box.
    density : bool
        Use hexbin for large datasets.

    Returns
    -------
    tuple
        (fig, ax) matplotlib objects.
    """
    model_flat = model_values.flatten()
    obs_flat = obs_values.flatten()
    valid = ~(np.isnan(model_flat) | np.isnan(obs_flat))
    model_flat, obs_flat = model_flat[valid], obs_flat[valid]

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    vmin = min(np.min(model_flat), np.min(obs_flat))
    vmax = max(np.max(model_flat), np.max(obs_flat))
    margin = (vmax - vmin) * 0.05
    lim = (vmin - margin, vmax + margin)

    if density and len(model_flat) > 1000:
        hb = ax.hexbin(obs_flat, model_flat, gridsize=50, cmap='YlOrRd', mincnt=1)
        fig.colorbar(hb, ax=ax, label='Count')
    else:
        ax.scatter(obs_flat, model_flat, alpha=0.4, s=10, c='steelblue', edgecolors='none')

    ax.plot(lim, lim, 'k--', linewidth=1, label='1:1')
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect('equal')

    label = variable_name or 'Value'
    unit_str = f' ({units})' if units else ''
    ax.set_xlabel(f'Observed {label}{unit_str}')
    ax.set_ylabel(f'Model {label}{unit_str}')
    ax.set_title(f'Model vs Observed: {label}')

    if show_stats and len(model_flat) > 1:
        r = np.corrcoef(model_flat, obs_flat)[0, 1]
        rmse = np.sqrt(np.mean((model_flat - obs_flat) ** 2))
        bias = np.mean(model_flat - obs_flat)
        mae = np.mean(np.abs(model_flat - obs_flat))
        stats_text = f'R = {r:.3f}\nRMSE = {rmse:.3f}\nBias = {bias:.3f}\nMAE = {mae:.3f}\nN = {len(model_flat)}'
        ax.text(
            0.05, 0.95, stats_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
        )

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    return fig, ax


def plot_station_map(
    station_lons: np.ndarray,
    station_lats: np.ndarray,
    values: np.ndarray,
    save_path: Union[str, pathlib.Path] = None,
    metric_name: str = 'Bias',
    units: str = None,
    cmap: str = 'RdBu_r',
    vmin: float = None,
    vmax: float = None,
    symmetric: bool = True,
    marker_size: float = 80,
    figsize: tuple = (12, 8),
    dpi: int = 150,
) -> tuple:
    """
    Map of station locations colored by metric value.

    Parameters
    ----------
    station_lons : np.ndarray
        Station longitudes.
    station_lats : np.ndarray
        Station latitudes.
    values : np.ndarray
        Metric values per station.
    save_path : str or pathlib.Path, optional
        Path to save figure.
    metric_name : str
        Metric name for colorbar label.
    units : str, optional
        Units string.
    cmap : str
        Colormap name.
    vmin, vmax : float, optional
        Color limits.
    symmetric : bool
        Center colormap on zero.
    marker_size : float
        Marker size.
    figsize : tuple
        Figure size.
    dpi : int
        Figure DPI.

    Returns
    -------
    tuple
        (fig, ax) matplotlib objects.
    """
    if symmetric and vmin is None and vmax is None:
        abs_max = np.nanmax(np.abs(values))
        vmin, vmax = -abs_max, abs_max

    if HAS_CARTOPY:
        fig, ax = plt.subplots(
            figsize=figsize, dpi=dpi,
            subplot_kw={'projection': ccrs.PlateCarree()},
        )
        ax.set_facecolor('#D6EAF8')
        ax.add_feature(cfeature.LAND, facecolor='#F5F5F5', edgecolor='none')
        ax.add_feature(cfeature.LAKES, facecolor='#D6EAF8', edgecolor='none')
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle=':')
        ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)
        sc = ax.scatter(
            station_lons, station_lats, c=values, s=marker_size,
            cmap=cmap, vmin=vmin, vmax=vmax,
            edgecolors='black', linewidth=0.5,
            transform=ccrs.PlateCarree(), zorder=5,
        )
    else:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        sc = ax.scatter(
            station_lons, station_lats, c=values, s=marker_size,
            cmap=cmap, vmin=vmin, vmax=vmax,
            edgecolors='black', linewidth=0.5,
        )
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')

    unit_str = f' ({units})' if units else ''
    fig.colorbar(sc, ax=ax, label=f'{metric_name}{unit_str}', shrink=0.7)
    ax.set_title(f'Station {metric_name}')

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    return fig, ax


def plot_timeseries(
    times: np.ndarray,
    model_values: np.ndarray,
    obs_values: np.ndarray = None,
    save_path: Union[str, pathlib.Path] = None,
    variable_name: str = None,
    units: str = None,
    station_name: str = None,
    model_label: str = 'Model',
    obs_label: str = 'Observed',
    figsize: tuple = (12, 5),
    dpi: int = 150,
) -> tuple:
    """
    Time series plot comparing model output to observations.

    Parameters
    ----------
    times : np.ndarray
        Datetime array.
    model_values : np.ndarray
        Model time series.
    obs_values : np.ndarray, optional
        Observation time series.
    save_path : str or pathlib.Path, optional
        Path to save figure.
    variable_name : str, optional
        Variable name for y-axis label.
    units : str, optional
        Units string.
    station_name : str, optional
        Station name for title.
    model_label : str
        Legend label for model.
    obs_label : str
        Legend label for observations.
    figsize : tuple
        Figure size.
    dpi : int
        Figure DPI.

    Returns
    -------
    tuple
        (fig, ax) matplotlib objects.
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    ax.plot(times, model_values, '-', color='steelblue', linewidth=1.5, label=model_label)
    if obs_values is not None:
        ax.plot(times, obs_values, 'o', color='black', markersize=4, label=obs_label)

    label = variable_name or 'Value'
    unit_str = f' ({units})' if units else ''
    ax.set_ylabel(f'{label}{unit_str}')
    ax.set_xlabel('Time')

    title = label
    if station_name:
        title = f'{label} at {station_name}'
    ax.set_title(title)

    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    return fig, ax


def plot_performance_diagram(
    pod_values,
    far_values,
    labels: list = None,
    save_path: Union[str, pathlib.Path] = None,
    figsize: tuple = (8, 8),
    dpi: int = 150,
) -> tuple:
    """
    Performance diagram (Roebber 2009).

    Parameters
    ----------
    pod_values : float or list[float]
        Probability of Detection values.
    far_values : float or list[float]
        False Alarm Ratio values.
    labels : list[str], optional
        Labels for each point.
    save_path : str or pathlib.Path, optional
        Path to save figure.
    figsize : tuple
        Figure size.
    dpi : int
        Figure DPI.

    Returns
    -------
    tuple
        (fig, ax) matplotlib objects.
    """
    if isinstance(pod_values, (int, float)):
        pod_values = [pod_values]
    if isinstance(far_values, (int, float)):
        far_values = [far_values]

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # CSI contours
    sr_range = np.linspace(0.01, 1.0, 200)
    pod_range = np.linspace(0.01, 1.0, 200)
    sr_grid, pod_grid = np.meshgrid(sr_range, pod_range)
    csi_grid = 1.0 / (1.0 / sr_grid + 1.0 / pod_grid - 1.0)
    csi_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    cs = ax.contour(sr_grid, pod_grid, csi_grid, levels=csi_levels, colors='gray', linewidths=0.5, linestyles='solid')
    ax.clabel(cs, fmt='%.1f', fontsize=7)

    # Frequency bias lines
    for fbias in [0.25, 0.5, 1.0, 2.0, 4.0]:
        sr_line = np.linspace(0.01, 1.0, 100)
        pod_line = np.minimum(fbias * sr_line, 1.0)
        ax.plot(sr_line, pod_line, 'k--', linewidth=0.5, alpha=0.5)
        # Label bias lines
        idx = min(int(0.7 / fbias * 100), 99) if fbias <= 1.0 else min(int(70), 99)
        if pod_line[idx] <= 1.0:
            ax.text(sr_line[idx], pod_line[idx], f'{fbias}', fontsize=7, alpha=0.6)

    # Plot experiment points
    colors = plt.cm.Set1(np.linspace(0, 1, max(len(pod_values), 2)))
    for i, (pod, far) in enumerate(zip(pod_values, far_values)):
        sr = 1.0 - far
        label = labels[i] if labels else f'Exp {i + 1}'
        ax.scatter(sr, pod, c=[colors[i]], s=100, zorder=5, edgecolors='black', linewidth=0.5, label=label)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Success Ratio (1 - FAR)')
    ax.set_ylabel('Probability of Detection (POD)')
    ax.set_title('Performance Diagram')
    ax.set_aspect('equal')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    return fig, ax


def plot_taylor_diagram(
    std_obs: float,
    std_model,
    correlations,
    labels: list = None,
    save_path: Union[str, pathlib.Path] = None,
    figsize: tuple = (8, 8),
    dpi: int = 150,
) -> tuple:
    """
    Taylor diagram (Taylor 2001).

    Parameters
    ----------
    std_obs : float
        Standard deviation of observations (reference point).
    std_model : float or list[float]
        Standard deviation(s) of model(s).
    correlations : float or list[float]
        Pearson correlation(s).
    labels : list[str], optional
        Labels for each model.
    save_path : str or pathlib.Path, optional
        Path to save figure.
    figsize : tuple
        Figure size.
    dpi : int
        Figure DPI.

    Returns
    -------
    tuple
        (fig, ax) matplotlib objects.
    """
    if isinstance(std_model, (int, float)):
        std_model = [std_model]
    if isinstance(correlations, (int, float)):
        correlations = [correlations]

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi, subplot_kw={'projection': 'polar'})
    ax.set_thetamin(0)
    ax.set_thetamax(90)

    # Correlation lines
    corr_ticks = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]
    theta_ticks = np.arccos(corr_ticks)
    ax.set_thetagrids(np.degrees(theta_ticks), labels=[str(c) for c in corr_ticks])

    max_std = max(std_obs * 1.5, max(std_model) * 1.2)
    ax.set_rlim(0, max_std)

    # Reference point (observations)
    ax.plot(0, std_obs, 'ko', markersize=10, label='Observed')

    # Reference std arc
    theta_arc = np.linspace(0, np.pi / 2, 100)
    ax.plot(theta_arc, [std_obs] * 100, 'k--', linewidth=0.5)

    # Centered RMSE arcs
    for rmse_val in np.arange(0.5, max_std, 0.5 * std_obs):
        theta_crmse = np.linspace(0, np.pi / 2, 200)
        r_crmse = []
        for th in theta_crmse:
            # r^2 + std_obs^2 - 2*r*std_obs*cos(th) = rmse_val^2
            # Solve quadratic in r
            a = 1
            b = -2 * std_obs * np.cos(th)
            c = std_obs ** 2 - rmse_val ** 2
            disc = b ** 2 - 4 * a * c
            if disc >= 0:
                r1 = (-b + np.sqrt(disc)) / 2
                if r1 >= 0:
                    r_crmse.append((th, r1))
        if r_crmse:
            ths, rs = zip(*r_crmse)
            ax.plot(ths, rs, 'g--', linewidth=0.3, alpha=0.5)

    # Plot model points
    colors = plt.cm.Set1(np.linspace(0, 1, max(len(std_model), 2)))
    for i, (std_m, corr) in enumerate(zip(std_model, correlations)):
        theta = np.arccos(np.clip(corr, -1, 1))
        label = labels[i] if labels else f'Model {i + 1}'
        ax.scatter(theta, std_m, c=[colors[i]], s=100, zorder=5, edgecolors='black', linewidth=0.5, label=label)

    ax.set_title('Taylor Diagram', pad=20)
    if len(std_model) <= 10:
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    return fig, ax


def plot_diurnal(
    hours: np.ndarray,
    model_values: np.ndarray,
    obs_values: np.ndarray = None,
    save_path: Union[str, pathlib.Path] = None,
    variable_name: str = None,
    units: str = None,
    model_label: str = 'Model',
    obs_label: str = 'Observed',
    show_spread: bool = True,
    model_std: np.ndarray = None,
    obs_std: np.ndarray = None,
    figsize: tuple = (10, 6),
    dpi: int = 150,
) -> tuple:
    """
    Diurnal cycle comparison plot.

    Parameters
    ----------
    hours : np.ndarray
        Hour of day (0-23).
    model_values : np.ndarray
        Model mean values per hour.
    obs_values : np.ndarray, optional
        Observation mean values per hour.
    save_path : str or pathlib.Path, optional
        Path to save figure.
    variable_name : str, optional
        Variable name for y-axis.
    units : str, optional
        Units string.
    model_label : str
        Legend label for model.
    obs_label : str
        Legend label for observations.
    show_spread : bool
        Show standard deviation shading.
    model_std : np.ndarray, optional
        Model standard deviation per hour (for shading).
    obs_std : np.ndarray, optional
        Observation standard deviation per hour (for shading).
    figsize : tuple
        Figure size.
    dpi : int
        Figure DPI.

    Returns
    -------
    tuple
        (fig, ax) matplotlib objects.
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    ax.plot(hours, model_values, '-o', color='steelblue', linewidth=1.5, markersize=4, label=model_label)
    if show_spread and model_std is not None:
        ax.fill_between(hours, model_values - model_std, model_values + model_std, alpha=0.2, color='steelblue')

    if obs_values is not None:
        ax.plot(hours, obs_values, '-s', color='black', linewidth=1.5, markersize=4, label=obs_label)
        if show_spread and obs_std is not None:
            ax.fill_between(hours, obs_values - obs_std, obs_values + obs_std, alpha=0.2, color='gray')

    label = variable_name or 'Value'
    unit_str = f' ({units})' if units else ''
    ax.set_xlabel('Hour of Day (UTC)')
    ax.set_ylabel(f'{label}{unit_str}')
    ax.set_title(f'Diurnal Cycle: {label}')
    ax.set_xticks(np.arange(0, 24, 3))
    ax.set_xlim(-0.5, 23.5)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    return fig, ax


def plot_lagged_correlation(
    lags: np.ndarray,
    correlations: np.ndarray,
    save_path: Union[str, pathlib.Path] = None,
    station_name: str = None,
    variable_name: str = None,
    figsize: tuple = (10, 5),
    dpi: int = 150,
) -> tuple:
    """
    Lagged cross-correlation plot.

    Parameters
    ----------
    lags : np.ndarray
        Lag values (in timesteps).
    correlations : np.ndarray
        Correlation at each lag. Can be 1D (single station) or 2D (lag, station).
    save_path : str or pathlib.Path, optional
        Path to save figure.
    station_name : str, optional
        Station name for title.
    variable_name : str, optional
        Variable name for title.
    figsize : tuple
        Figure size.
    dpi : int
        Figure DPI.

    Returns
    -------
    tuple
        (fig, ax) matplotlib objects.
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    if correlations.ndim == 2:
        # Multiple stations: plot mean with spread
        mean_corr = np.nanmean(correlations, axis=1)
        std_corr = np.nanstd(correlations, axis=1)
        ax.plot(lags, mean_corr, '-o', color='steelblue', linewidth=1.5, markersize=3, label='Mean')
        ax.fill_between(lags, mean_corr - std_corr, mean_corr + std_corr, alpha=0.2, color='steelblue')
        best_idx = np.nanargmax(mean_corr)
    else:
        ax.plot(lags, correlations, '-o', color='steelblue', linewidth=1.5, markersize=4)
        best_idx = np.nanargmax(correlations)

    # Mark optimal lag
    ax.axvline(x=lags[best_idx], color='red', linestyle='--', linewidth=1, alpha=0.7)
    ax.axvline(x=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)

    title_parts = ['Lagged Correlation']
    if variable_name:
        title_parts.append(f'({variable_name})')
    if station_name:
        title_parts.append(f'at {station_name}')
    ax.set_title(' '.join(title_parts))

    ax.set_xlabel('Lag (timesteps, positive = model leads)')
    ax.set_ylabel('Pearson Correlation')
    ax.set_ylim(-1.05, 1.05)
    ax.grid(True, alpha=0.3)

    peak_corr = np.nanmean(correlations, axis=1)[best_idx] if correlations.ndim == 2 else correlations[best_idx]
    ax.text(
        0.95, 0.05,
        f'Optimal lag: {lags[best_idx]}\nPeak r: {peak_corr:.3f}',
        transform=ax.transAxes, fontsize=9, ha='right',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
    )

    if correlations.ndim == 2:
        ax.legend()

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    return fig, ax


def plot_fss(
    neighborhood_sizes: np.ndarray,
    fss_values: np.ndarray,
    save_path: Union[str, pathlib.Path] = None,
    threshold: float = None,
    multi_times: dict = None,
    figsize: tuple = (10, 6),
    dpi: int = 150,
) -> tuple:
    """
    FSS vs spatial scale plot.

    Parameters
    ----------
    neighborhood_sizes : np.ndarray
        Neighborhood sizes (grid cells).
    fss_values : np.ndarray
        FSS values corresponding to neighborhood sizes.
    save_path : str or pathlib.Path, optional
        Path to save figure.
    threshold : float, optional
        Threshold value (for title).
    multi_times : dict, optional
        Dict mapping label -> fss_values array for multiple lines.
    figsize : tuple
        Figure size.
    dpi : int
        Figure DPI.

    Returns
    -------
    tuple
        (fig, ax) matplotlib objects.
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    if multi_times:
        colors = plt.cm.viridis(np.linspace(0, 1, len(multi_times)))
        for (label, vals), color in zip(multi_times.items(), colors):
            ax.plot(neighborhood_sizes, vals, '-o', color=color, label=label, markersize=4)
    else:
        ax.plot(neighborhood_sizes, fss_values, '-o', color='steelblue', linewidth=2, markersize=6)

    ax.axhline(y=0.5, color='red', linestyle='--', linewidth=1, label='Useful skill (0.5)')
    ax.set_xlabel('Neighborhood Size (grid cells)')
    ax.set_ylabel('Fractions Skill Score')
    title = 'FSS vs Spatial Scale'
    if threshold is not None:
        title += f' (threshold={threshold})'
    ax.set_title(title)
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    return fig, ax


def plot_wind_rose_comparison(
    model_speed: np.ndarray,
    model_direction: np.ndarray,
    obs_speed: np.ndarray = None,
    obs_direction: np.ndarray = None,
    save_path: Union[str, pathlib.Path] = None,
    speed_bins: list = None,
    n_direction_bins: int = 16,
    figsize: tuple = (14, 6),
    dpi: int = 150,
) -> tuple:
    """
    Side-by-side wind roses for model vs observations.

    Parameters
    ----------
    model_speed : np.ndarray
        Model wind speeds.
    model_direction : np.ndarray
        Model wind directions (degrees, meteorological convention).
    obs_speed : np.ndarray, optional
        Observed wind speeds.
    obs_direction : np.ndarray, optional
        Observed wind directions (degrees).
    save_path : str or pathlib.Path, optional
        Path to save figure.
    speed_bins : list[float], optional
        Wind speed bin edges. Default: [0, 2, 4, 6, 8, 10, 15].
    n_direction_bins : int
        Number of direction bins. Default: 16.
    figsize : tuple
        Figure size.
    dpi : int
        Figure DPI.

    Returns
    -------
    tuple
        (fig, axes) matplotlib objects.
    """
    if speed_bins is None:
        speed_bins = [0, 2, 4, 6, 8, 10, 15]

    dir_bins = np.linspace(0, 360, n_direction_bins + 1)
    dir_centers = (dir_bins[:-1] + dir_bins[1:]) / 2
    theta = np.radians(dir_centers)

    has_obs = obs_speed is not None and obs_direction is not None
    n_panels = 2 if has_obs else 1

    fig, axes = plt.subplots(
        1, n_panels, figsize=figsize, dpi=dpi,
        subplot_kw={'projection': 'polar'},
    )
    if n_panels == 1:
        axes = [axes]

    cmap = plt.cm.YlOrRd
    colors = cmap(np.linspace(0.2, 0.9, len(speed_bins) - 1))

    def _draw_rose(ax, speed, direction, title):
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)

        valid = ~(np.isnan(speed) | np.isnan(direction))
        speed, direction = speed[valid], direction[valid]
        total = len(speed)
        calm = np.sum(speed < speed_bins[0]) if speed_bins[0] > 0 else 0

        for si in range(len(speed_bins) - 1):
            mask = (speed >= speed_bins[si]) & (speed < speed_bins[si + 1])
            counts = np.zeros(n_direction_bins)
            for di in range(n_direction_bins):
                d_lo = dir_bins[di]
                d_hi = dir_bins[di + 1]
                counts[di] = np.sum(mask & (direction >= d_lo) & (direction < d_hi))

            widths = np.radians(360 / n_direction_bins)
            pct = counts / total * 100 if total > 0 else counts
            ax.bar(
                theta, pct, width=widths, bottom=0,
                color=colors[si], edgecolor='gray', linewidth=0.3,
                label=f'{speed_bins[si]}-{speed_bins[si + 1]}',
                alpha=0.8,
            )

        ax.set_title(title, pad=15, fontsize=11)
        if total > 0:
            ax.text(0, 0, f'Calm: {calm / total * 100:.1f}%', ha='center', va='center', fontsize=8)

    _draw_rose(axes[0], model_speed, model_direction, 'Model')
    if has_obs:
        _draw_rose(axes[1], obs_speed, obs_direction, 'Observed')

    # Shared legend
    handles, leg_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, leg_labels, loc='lower center', ncol=len(speed_bins) - 1, title='Wind Speed (m/s)')

    fig.tight_layout(rect=[0, 0.08, 1, 1])
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    return fig, axes


def plot_station_evaluation(
    eval_path: Union[str, pathlib.Path],
    variable: str,
    metric: str = 'bias',
    plot_type: str = 'map',
    save_path: Union[str, pathlib.Path] = None,
    **kwargs,
) -> tuple:
    """
    Read station evaluation output and produce the appropriate plot.

    Parameters
    ----------
    eval_path : str or pathlib.Path
        Path to cfdb station evaluation output.
    variable : str
        Variable name.
    metric : str
        Metric name (e.g., 'bias', 'rmse').
    plot_type : str
        One of 'map', 'scatter', 'timeseries'.
    save_path : str or pathlib.Path, optional
        Path to save figure.
    **kwargs
        Additional keyword arguments passed to the plot function.

    Returns
    -------
    tuple
        (fig, ax) matplotlib objects.
    """
    import cfdb
    import shapely

    var_name = f'{variable}_{metric}'

    with cfdb.open_dataset(eval_path) as ds:
        geo_coord_name = None
        for name in ds.coord_names:
            if name in ('point', 'station'):
                geo_coord_name = name
                break
        if geo_coord_name is None:
            for name in ds.coord_names:
                if name not in ('time', 'height', 'hour', 'metric'):
                    geo_coord_name = name
                    break

        points = ds[geo_coord_name].data
        coords = shapely.get_coordinates(points)
        lons, lats = coords[:, 0], coords[:, 1]
        times = ds['time'].data if 'time' in ds.coord_names else None
        data = ds[var_name].data

    if plot_type == 'map':
        # Average over time for map display
        if data.ndim == 2:
            values = np.nanmean(data, axis=0)
        else:
            values = data
        return plot_station_map(
            lons, lats, values, save_path=save_path,
            metric_name=f'{metric.upper()} ({variable})', **kwargs,
        )
    elif plot_type == 'scatter':
        # data is (time, station) - flatten model=data, obs=0 baseline
        return plot_scatter(
            data.flatten(), np.zeros_like(data.flatten()),
            save_path=save_path, variable_name=variable, **kwargs,
        )
    elif plot_type == 'timeseries' and times is not None:
        # Plot mean across stations
        if data.ndim == 2:
            mean_vals = np.nanmean(data, axis=1)
        else:
            mean_vals = data
        return plot_timeseries(
            times, mean_vals, save_path=save_path,
            variable_name=f'{variable} {metric}', **kwargs,
        )
    else:
        raise ValueError(f"Unknown plot_type '{plot_type}'. Use 'map', 'scatter', or 'timeseries'.")


def plot_domain_evaluation(
    eval_path: Union[str, pathlib.Path],
    variable: str,
    metric: str = 'bias',
    plot_type: str = 'timeseries',
    save_path: Union[str, pathlib.Path] = None,
    **kwargs,
) -> tuple:
    """
    Read domain evaluation output and produce the appropriate plot.

    Parameters
    ----------
    eval_path : str or pathlib.Path
        Path to cfdb domain evaluation output.
    variable : str
        Variable name.
    metric : str
        Metric name (e.g., 'bias', 'rmse').
    plot_type : str
        One of 'timeseries'.
    save_path : str or pathlib.Path, optional
        Path to save figure.
    **kwargs
        Additional keyword arguments passed to the plot function.

    Returns
    -------
    tuple
        (fig, ax) matplotlib objects.
    """
    import cfdb

    with cfdb.open_dataset(eval_path) as ds:
        times = ds['time'].data
        metric_coord = ds['metric']
        flag_meanings = metric_coord.attrs.get('flag_meanings', '').split()

        if metric in flag_meanings:
            m_idx = flag_meanings.index(metric)
        else:
            m_idx = 0

        data = ds[variable].data
        values = data[:, m_idx]

    if plot_type == 'timeseries':
        return plot_timeseries(
            times, values, save_path=save_path,
            variable_name=f'{variable} ({metric})', **kwargs,
        )
    else:
        raise ValueError(f"Unknown plot_type '{plot_type}'. Use 'timeseries'.")
