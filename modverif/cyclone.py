"""
Functions for tracking cyclones in model output using sea level pressure.

Reads data from cfdb datasets. If the dataset contains pre-computed ``mslp``,
it is used directly. Otherwise, SLP is estimated from ``surface_pressure``,
``terrain_height``, and ``air_temperature`` (with optional ``mixing_ratio``
for virtual temperature correction).
"""
import pathlib
from dataclasses import dataclass
from typing import Union

import cfdb
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False

# Physical constants
GRAVITY = 9.80665  # m/s^2
GAS_CONSTANT_DRY = 287.05  # J/(kg·K)
STANDARD_LAPSE_RATE = 0.0065  # K/m


def _compute_sea_level_pressure(
    psfc: np.ndarray,
    hgt: np.ndarray,
    t2: np.ndarray,
    q2: np.ndarray = None,
) -> np.ndarray:
    """
    Estimate sea level pressure from surface variables using the hypsometric equation.

    Parameters
    ----------
    psfc : np.ndarray
        Surface pressure in Pa.
    hgt : np.ndarray
        Terrain height in meters.
    t2 : np.ndarray
        2-meter temperature in Kelvin.
    q2 : np.ndarray, optional
        2-meter water vapor mixing ratio in kg/kg.
        If provided, virtual temperature correction is applied.

    Returns
    -------
    np.ndarray
        Estimated sea level pressure in Pa.
    """
    if q2 is not None:
        t_virtual = t2 * (1.0 + 0.61 * q2)
    else:
        t_virtual = t2

    t_sea_level = t_virtual + STANDARD_LAPSE_RATE * hgt
    t_avg = 0.5 * (t_virtual + t_sea_level)
    slp = psfc * np.exp(GRAVITY * hgt / (GAS_CONSTANT_DRY * t_avg))

    return slp


def _read_var_2d(ds, var_name, time_idx, height_idx=0):
    """
    Read a 2D (y, x) slice from a cfdb dataset variable.

    Handles 3D (time, y, x) and 4D (time, height, y, x) variables,
    squeezing scalar dimensions that cfdb preserves on integer indexing.
    """
    var = ds[var_name]
    n_dims = len(var.shape)
    if n_dims == 4:
        data = var[(time_idx, height_idx, slice(None), slice(None))].data
        return data[0, 0]
    elif n_dims == 3:
        data = var[(time_idx, slice(None), slice(None))].data
        return data[0]
    elif n_dims == 2:
        data = var[(slice(None), slice(None))].data
        return data
    else:
        raise ValueError(f"Variable '{var_name}' has {n_dims} dimensions, expected 2-4")


def _read_var_3d_time_slice(ds, var_name, t_slice, height_idx=0):
    """
    Read a (n_t, ny, nx) block from a cfdb variable across a time slice.

    Bulk equivalent of ``_read_var_2d`` covering a contiguous time range in
    one cfdb call. Used to avoid re-decompressing the same multi-timestep
    chunk for every frame in a plot loop.
    """
    var = ds[var_name]
    n_dims = len(var.shape)
    if n_dims == 4:
        data = var[(t_slice, height_idx, slice(None), slice(None))].data
        return data[:, 0]
    elif n_dims == 3:
        return var[(t_slice, slice(None), slice(None))].data
    else:
        raise ValueError(f"Variable '{var_name}' has {n_dims} dimensions, expected 3-4 for time-slice reads")


def _read_latlon_2d(ds):
    """
    Read 2D latitude and longitude arrays from a cfdb dataset.

    Handles two cases:
    - 1D coordinates (``latitude``/``longitude``): creates 2D meshgrid
    - 2D data variables (``latitude``/``longitude``): reads directly

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (xlat, xlong) each with shape (n_y, n_x).
    """
    coord_names = set(ds.coord_names)
    var_names = set(ds.data_var_names)

    if 'latitude' in coord_names and 'longitude' in coord_names:
        lat_1d = ds['latitude'].data
        lon_1d = ds['longitude'].data
        xlong, xlat = np.meshgrid(lon_1d, lat_1d)
        return xlat, xlong
    elif 'latitude' in var_names and 'longitude' in var_names:
        lat_var = ds['latitude']
        lon_var = ds['longitude']
        n_dims = len(lat_var.shape)
        if n_dims == 2:
            xlat = lat_var[(slice(None), slice(None))].data
            xlong = lon_var[(slice(None), slice(None))].data
        elif n_dims == 3:
            # (time, y, x) — take first timestep
            xlat = lat_var[(0, slice(None), slice(None))].data[0]
            xlong = lon_var[(0, slice(None), slice(None))].data[0]
        elif n_dims == 4:
            # (time, height, y, x) — take first timestep and height
            xlat = lat_var[(0, 0, slice(None), slice(None))].data[0, 0]
            xlong = lon_var[(0, 0, slice(None), slice(None))].data[0, 0]
        else:
            raise ValueError(f"latitude variable has {n_dims} dimensions, expected 2-4")
        return xlat, xlong
    else:
        raise ValueError(
            "Dataset must contain 'latitude' and 'longitude' as either "
            f"coordinates or data variables. Found coords={ds.coord_names}, "
            f"data_vars={ds.data_var_names}"
        )


def _read_slp_from_cfdb(ds, time_idx, smoothing_sigma=None):
    """
    Read or compute sea level pressure from a cfdb dataset for one timestep.

    Uses ``mslp`` if available. Otherwise computes from ``surface_pressure``,
    ``terrain_height``, and ``air_temperature`` (with optional ``mixing_ratio``).

    Returns
    -------
    np.ndarray
        2D SLP field (y, x) in Pa.
    """
    all_vars = set(ds.coord_names) | set(ds.data_var_names)

    if 'mslp' in all_vars:
        slp = _read_var_2d(ds, 'mslp', time_idx)
    else:
        required = {'surface_pressure', 'terrain_height', 'air_temperature'}
        missing = required - all_vars
        if missing:
            raise ValueError(
                f"Dataset does not contain 'mslp' and is missing variables "
                f"needed to compute SLP: {missing}"
            )
        psfc = _read_var_2d(ds, 'surface_pressure', time_idx)
        hgt = _read_var_2d(ds, 'terrain_height', time_idx)
        t2 = _read_var_2d(ds, 'air_temperature', time_idx)
        q2 = _read_var_2d(ds, 'mixing_ratio', time_idx) if 'mixing_ratio' in all_vars else None
        slp = _compute_sea_level_pressure(psfc, hgt, t2, q2)

    if smoothing_sigma is not None:
        slp = gaussian_filter(slp.astype(np.float64), sigma=smoothing_sigma).astype(slp.dtype)

    return slp


def _read_slp_block_from_cfdb(ds, t_slice):
    """
    Bulk-read or compute SLP for a time slice. Returns a (n_t, ny, nx) block in Pa.

    Mirrors ``_read_slp_from_cfdb`` but reads each underlying variable as a single
    slab to avoid per-timestep chunk decompression.
    """
    all_vars = set(ds.coord_names) | set(ds.data_var_names)

    if 'mslp' in all_vars:
        return _read_var_3d_time_slice(ds, 'mslp', t_slice)

    required = {'surface_pressure', 'terrain_height', 'air_temperature'}
    missing = required - all_vars
    if missing:
        raise ValueError(
            f"Dataset does not contain 'mslp' and is missing variables "
            f"needed to compute SLP: {missing}"
        )
    psfc = _read_var_3d_time_slice(ds, 'surface_pressure', t_slice)
    hgt = _read_var_3d_time_slice(ds, 'terrain_height', t_slice)
    t2 = _read_var_3d_time_slice(ds, 'air_temperature', t_slice)
    q2 = _read_var_3d_time_slice(ds, 'mixing_ratio', t_slice) if 'mixing_ratio' in all_vars else None
    return _compute_sea_level_pressure(psfc, hgt, t2, q2)


@dataclass
class CyclonePosition:
    """Position and characteristics of a cyclone at a single timestep."""

    time_index: int
    y_index: int
    x_index: int
    latitude: float
    longitude: float
    central_pressure: float
    radius_km: float
    time_str: str = None


def _haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great-circle distance between two points in kilometers.

    Parameters
    ----------
    lat1, lon1 : float
        Latitude and longitude of first point in degrees.
    lat2, lon2 : float
        Latitude and longitude of second point in degrees.

    Returns
    -------
    float
        Distance in kilometers.
    """
    R = 6371.0  # Earth's radius in km

    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return R * c


def _grid_distances_km(
    xlat: np.ndarray,
    xlong: np.ndarray,
    center_lat: float,
    center_lon: float,
) -> np.ndarray:
    """
    Calculate distance in km from each grid point to a center point.

    Parameters
    ----------
    xlat : np.ndarray
        2D array of latitudes (y, x).
    xlong : np.ndarray
        2D array of longitudes (y, x).
    center_lat : float
        Center point latitude.
    center_lon : float
        Center point longitude.

    Returns
    -------
    np.ndarray
        2D array of distances in km.
    """
    R = 6371.0

    lat1_rad = np.radians(center_lat)
    lat2_rad = np.radians(xlat)
    dlat = np.radians(xlat - center_lat)
    dlon = np.radians(xlong - center_lon)

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return R * c


def _find_pressure_minimum(
    pressure: np.ndarray,
    xlat: np.ndarray,
    xlong: np.ndarray,
    search_lat: float = None,
    search_lon: float = None,
    search_radius_km: float = None,
) -> tuple[int, int, float]:
    """
    Find the location of minimum pressure, optionally within a search radius.

    Parameters
    ----------
    pressure : np.ndarray
        2D array of surface pressure (y, x).
    xlat : np.ndarray
        2D array of latitudes (y, x).
    xlong : np.ndarray
        2D array of longitudes (y, x).
    search_lat : float, optional
        Center latitude for search region.
    search_lon : float, optional
        Center longitude for search region.
    search_radius_km : float, optional
        Search radius in kilometers.

    Returns
    -------
    tuple[int, int, float]
        (y_index, x_index, min_pressure)
    """
    if search_lat is not None and search_lon is not None and search_radius_km is not None:
        # Create mask for search region
        distances = _grid_distances_km(xlat, xlong, search_lat, search_lon)
        mask = distances <= search_radius_km

        if not np.any(mask):
            raise ValueError(
                f"No grid points found within {search_radius_km} km of ({search_lat}, {search_lon})"
            )

        # Set pressure outside search region to infinity
        pressure_masked = np.where(mask, pressure, np.inf)
    else:
        pressure_masked = pressure

    # Find minimum
    min_idx = np.argmin(pressure_masked)
    y_idx, x_idx = np.unravel_index(min_idx, pressure.shape)

    return int(y_idx), int(x_idx), float(pressure[y_idx, x_idx])


def _estimate_cyclone_radius(
    pressure: np.ndarray,
    xlat: np.ndarray,
    xlong: np.ndarray,
    center_y: int,
    center_x: int,
    pressure_threshold_pa: float = 400.0,
    max_radius_km: float = 1000.0,
) -> float:
    """
    Estimate cyclone radius based on pressure gradient from center.

    The radius is defined as the distance where pressure increases by
    a threshold amount from the central minimum, or where the outermost
    closed isobar would approximately be.

    Parameters
    ----------
    pressure : np.ndarray
        2D array of surface pressure (y, x).
    xlat : np.ndarray
        2D array of latitudes (y, x).
    xlong : np.ndarray
        2D array of longitudes (y, x).
    center_y : int
        Y index of cyclone center.
    center_x : int
        X index of cyclone center.
    pressure_threshold_pa : float
        Pressure increase from center that defines the edge (default 400 Pa = 4 hPa).
    max_radius_km : float
        Maximum radius to consider (default 1000 km).

    Returns
    -------
    float
        Estimated radius in kilometers.
    """
    center_lat = xlat[center_y, center_x]
    center_lon = xlong[center_y, center_x]
    center_pressure = pressure[center_y, center_x]

    # Calculate distances from center
    distances = _grid_distances_km(xlat, xlong, center_lat, center_lon)

    # Find where pressure exceeds threshold above center
    pressure_diff = pressure - center_pressure
    edge_mask = (pressure_diff >= pressure_threshold_pa) & (distances <= max_radius_km)

    if np.any(edge_mask):
        # Find minimum distance where threshold is exceeded
        edge_distances = np.where(edge_mask, distances, np.inf)
        radius = np.min(edge_distances)
    else:
        # If threshold not reached, use max radius or domain edge
        within_max = distances <= max_radius_km
        if np.any(within_max & (pressure_diff > 0)):
            # Use distance to furthest point with positive pressure gradient
            positive_gradient = (pressure_diff > 0) & within_max
            radius = np.max(np.where(positive_gradient, distances, 0))
        else:
            radius = max_radius_km

    return float(radius)


def track_cyclone(
    cfdb_path: Union[str, pathlib.Path],
    start_lat: float = None,
    start_lon: float = None,
    search_radius_km: float = 500.0,
    pressure_threshold_pa: float = 400.0,
    max_cyclone_radius_km: float = 1000.0,
    smoothing_sigma: float = None,
) -> list[CyclonePosition]:
    """
    Track a cyclone through time using sea level pressure minima.

    If the dataset contains ``mslp``, it is used directly. Otherwise,
    SLP is estimated from ``surface_pressure``, ``terrain_height``, and
    ``air_temperature`` using the hypsometric equation. If ``mixing_ratio``
    is available, a virtual temperature correction is applied.

    Starting from an initial position (or the global minimum if not specified),
    tracks the cyclone by finding the SLP minimum within a search radius
    of the previous position at each timestep.

    Parameters
    ----------
    cfdb_path : str or pathlib.Path
        Path to cfdb dataset.
    start_lat : float, optional
        Initial search latitude. If None, uses global pressure minimum at t=0.
    start_lon : float, optional
        Initial search longitude. If None, uses global pressure minimum at t=0.
    search_radius_km : float
        Radius in km to search for pressure minimum at each timestep.
        The search is centered on the previous timestep's position.
        Default is 500 km.
    pressure_threshold_pa : float
        Pressure increase from center that defines cyclone edge (default 400 Pa = 4 hPa).
    max_cyclone_radius_km : float
        Maximum cyclone radius to consider (default 1000 km).
    smoothing_sigma : float, optional
        Standard deviation for Gaussian smoothing of the SLP field.
        Higher values produce more smoothing. If None (default), no smoothing
        is applied. Typical values range from 1 to 5 grid cells.

    Returns
    -------
    list[CyclonePosition]
        List of CyclonePosition objects, one per timestep.

    Raises
    ------
    FileNotFoundError
        If the cfdb dataset does not exist.
    ValueError
        If required variables are not found.

    Examples
    --------
    >>> positions = track_cyclone(
    ...     'model_output.cfdb',
    ...     start_lat=-45.0,
    ...     start_lon=170.0,
    ...     search_radius_km=300.0,
    ... )
    >>> for pos in positions:
    ...     print(f"t={pos.time_index}: ({pos.latitude:.2f}, {pos.longitude:.2f}) "
    ...           f"SLP={pos.central_pressure/100:.1f} hPa, R={pos.radius_km:.0f} km")
    """
    cfdb_path = pathlib.Path(cfdb_path)
    if not cfdb_path.exists():
        raise FileNotFoundError(f"Dataset not found: {cfdb_path}")

    positions = []

    with cfdb.open_dataset(cfdb_path) as ds:
        xlat, xlong = _read_latlon_2d(ds)
        time_values = ds['time'].data
        n_times = len(time_values)

        # Initialize search position
        current_lat = start_lat
        current_lon = start_lon

        for t in range(n_times):
            slp = _read_slp_from_cfdb(ds, t, smoothing_sigma=smoothing_sigma)

            # Find pressure minimum
            if t == 0 and current_lat is None:
                # First timestep without start position: global search
                y_idx, x_idx, min_pressure = _find_pressure_minimum(slp, xlat, xlong)
            else:
                # Search within radius of current position
                y_idx, x_idx, min_pressure = _find_pressure_minimum(
                    slp,
                    xlat,
                    xlong,
                    search_lat=current_lat,
                    search_lon=current_lon,
                    search_radius_km=search_radius_km,
                )

            # Get position
            center_lat = float(xlat[y_idx, x_idx])
            center_lon = float(xlong[y_idx, x_idx])

            # Estimate cyclone radius
            radius = _estimate_cyclone_radius(
                slp,
                xlat,
                xlong,
                y_idx,
                x_idx,
                pressure_threshold_pa=pressure_threshold_pa,
                max_radius_km=max_cyclone_radius_km,
            )

            # Format time string
            time_str = str(time_values[t]) if time_values is not None else None

            # Create position record
            pos = CyclonePosition(
                time_index=t,
                y_index=y_idx,
                x_index=x_idx,
                latitude=center_lat,
                longitude=center_lon,
                central_pressure=min_pressure,
                radius_km=radius,
                time_str=time_str,
            )
            positions.append(pos)

            # Update search position for next timestep
            current_lat = center_lat
            current_lon = center_lon

    return positions


def track_cyclone_multi_file(
    cfdb_paths: list[Union[str, pathlib.Path]],
    start_lat: float = None,
    start_lon: float = None,
    search_radius_km: float = 500.0,
    pressure_threshold_pa: float = 400.0,
    max_cyclone_radius_km: float = 1000.0,
    smoothing_sigma: float = None,
) -> list[CyclonePosition]:
    """
    Track a cyclone across multiple cfdb datasets.

    Files are processed in the order provided. The cyclone position from the
    last timestep of each file is used as the starting search position for
    the next file.

    Parameters
    ----------
    cfdb_paths : list of str or pathlib.Path
        List of paths to cfdb datasets, in chronological order.
    start_lat : float, optional
        Initial search latitude for first file.
    start_lon : float, optional
        Initial search longitude for first file.
    search_radius_km : float
        Radius in km to search for pressure minimum. Default is 500 km.
    pressure_threshold_pa : float
        Pressure threshold for cyclone edge detection. Default is 400 Pa.
    max_cyclone_radius_km : float
        Maximum cyclone radius. Default is 1000 km.
    smoothing_sigma : float, optional
        Standard deviation for Gaussian smoothing of the SLP field.
        If None (default), no smoothing is applied.

    Returns
    -------
    list[CyclonePosition]
        Combined list of CyclonePosition objects from all files.
    """
    all_positions = []
    current_lat = start_lat
    current_lon = start_lon

    for path in cfdb_paths:
        positions = track_cyclone(
            path,
            start_lat=current_lat,
            start_lon=current_lon,
            search_radius_km=search_radius_km,
            pressure_threshold_pa=pressure_threshold_pa,
            max_cyclone_radius_km=max_cyclone_radius_km,
            smoothing_sigma=smoothing_sigma,
        )

        # Adjust time indices for continuity
        time_offset = len(all_positions)
        for pos in positions:
            pos.time_index += time_offset

        all_positions.extend(positions)

        # Use last position as start for next file
        if positions:
            current_lat = positions[-1].latitude
            current_lon = positions[-1].longitude

    return all_positions


def positions_to_array(positions: list[CyclonePosition]) -> np.ndarray:
    """
    Convert list of CyclonePosition to a structured numpy array.

    Parameters
    ----------
    positions : list[CyclonePosition]
        List of cyclone positions.

    Returns
    -------
    np.ndarray
        Structured array with fields: time_index, y_index, x_index,
        latitude, longitude, central_pressure, radius_km.
    """
    dtype = np.dtype([
        ('time_index', np.int32),
        ('y_index', np.int32),
        ('x_index', np.int32),
        ('latitude', np.float64),
        ('longitude', np.float64),
        ('central_pressure', np.float64),
        ('radius_km', np.float64),
    ])

    arr = np.zeros(len(positions), dtype=dtype)
    for i, pos in enumerate(positions):
        arr[i] = (
            pos.time_index,
            pos.y_index,
            pos.x_index,
            pos.latitude,
            pos.longitude,
            pos.central_pressure,
            pos.radius_km,
        )

    return arr


def _radius_km_to_degrees(radius_km: float, latitude: float) -> tuple[float, float]:
    """
    Convert radius in km to approximate degrees for plotting.

    Parameters
    ----------
    radius_km : float
        Radius in kilometers.
    latitude : float
        Latitude at which to compute conversion (affects longitude scaling).

    Returns
    -------
    tuple[float, float]
        (radius_lat_deg, radius_lon_deg) - radius in degrees for lat and lon.
    """
    # Approximate degrees per km
    km_per_deg_lat = 111.0  # ~111 km per degree latitude
    km_per_deg_lon = 111.0 * np.cos(np.radians(latitude))

    radius_lat_deg = radius_km / km_per_deg_lat
    radius_lon_deg = radius_km / km_per_deg_lon if km_per_deg_lon > 0 else radius_km / km_per_deg_lat

    return radius_lat_deg, radius_lon_deg


def plot_cyclone_timestep(
    cfdb_path: Union[str, pathlib.Path],
    position: CyclonePosition,
    output_path: Union[str, pathlib.Path],
    slp_levels: list[float] = None,
    figsize: tuple[float, float] = (12, 10),
    dpi: int = 150,
    cmap: str = 'RdYlBu_r',
    title: str = None,
):
    """
    Plot SLP field with cyclone center and radius for a single timestep.

    Parameters
    ----------
    cfdb_path : str or pathlib.Path
        Path to cfdb dataset.
    position : CyclonePosition
        Cyclone position for this timestep.
    output_path : str or pathlib.Path
        Output path for PNG file.
    slp_levels : list[float], optional
        Contour levels for SLP in hPa. Default is 960-1040 hPa every 4 hPa.
    figsize : tuple
        Figure size in inches. Default is (12, 10).
    dpi : int
        Output resolution. Default is 150.
    cmap : str
        Colormap for filled contours. Default is 'RdYlBu_r'.
    title : str, optional
        Custom title. If None, auto-generated from position data.
    """
    output_path = pathlib.Path(output_path)

    if slp_levels is None:
        slp_levels = list(range(960, 1044, 4))

    with cfdb.open_dataset(cfdb_path) as ds:
        t = position.time_index
        xlat, xlong = _read_latlon_2d(ds)

        # Wrap longitudes to -180 to 180 range
        xlong = np.where(xlong < 0, xlong + 360, xlong)

        slp = _read_slp_from_cfdb(ds, t)
        slp_hpa = slp / 100.0  # Convert to hPa

    # Create figure with cartopy projection if available
    if HAS_CARTOPY:
        fig, ax = plt.subplots(figsize=figsize, subplot_kw={'projection': ccrs.PlateCarree()})
        transform = ccrs.PlateCarree()
    else:
        fig, ax = plt.subplots(figsize=figsize)
        transform = None

    # Plot filled contours of SLP
    if transform:
        cf = ax.contourf(
            xlong, xlat, slp_hpa,
            levels=slp_levels,
            cmap=cmap,
            extend='both',
            transform=transform,
        )
    else:
        cf = ax.contourf(
            xlong, xlat, slp_hpa,
            levels=slp_levels,
            cmap=cmap,
            extend='both',
        )

    # Plot contour lines
    if transform:
        cs = ax.contour(
            xlong, xlat, slp_hpa,
            levels=slp_levels,
            colors='black',
            linewidths=0.5,
            transform=transform,
        )
    else:
        cs = ax.contour(
            xlong, xlat, slp_hpa,
            levels=slp_levels,
            colors='black',
            linewidths=0.5,
        )
    ax.clabel(cs, inline=True, fontsize=8, fmt='%.0f')

    # Add land and coastlines if cartopy available
    if HAS_CARTOPY:
        ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.5)
        ax.add_feature(cfeature.COASTLINE, linewidth=1, edgecolor='black')
        ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle='--', edgecolor='gray')

    # Add colorbar
    cbar = plt.colorbar(cf, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('Sea Level Pressure (hPa)', fontsize=12)

    # Wrap cyclone center longitude to match
    center_lon = position.longitude
    if center_lon < 0:
        center_lon = center_lon + 360

    # Plot cyclone center
    if HAS_CARTOPY:
        ax.plot(
            center_lon, position.latitude,
            'ko', markersize=10, markerfacecolor='red',
            markeredgecolor='black', markeredgewidth=2,
            label=f'Center: {position.central_pressure/100:.1f} hPa',
            transform=ccrs.PlateCarree(),
        )
    else:
        ax.plot(
            center_lon, position.latitude,
            'ko', markersize=10, markerfacecolor='red',
            markeredgecolor='black', markeredgewidth=2,
            label=f'Center: {position.central_pressure/100:.1f} hPa',
        )

    # Plot cyclone radius as a circle
    radius_lat, radius_lon = _radius_km_to_degrees(position.radius_km, position.latitude)

    if HAS_CARTOPY:
        # Use theta array to draw circle in lat/lon coordinates
        theta = np.linspace(0, 2 * np.pi, 100)
        circle_lons = center_lon + radius_lon * np.cos(theta)
        circle_lats = position.latitude + radius_lat * np.sin(theta)
        ax.plot(
            circle_lons, circle_lats,
            color='red', linewidth=2, linestyle='--',
            label=f'Radius: {position.radius_km:.0f} km',
            transform=ccrs.PlateCarree(),
        )
    else:
        avg_radius_deg = (radius_lat + radius_lon) / 2
        circle = plt.Circle(
            (center_lon, position.latitude),
            avg_radius_deg,
            fill=False,
            color='red',
            linewidth=2,
            linestyle='--',
            label=f'Radius: {position.radius_km:.0f} km',
            clip_on=True,
        )
        ax.add_patch(circle)

    # Add gridlines
    if HAS_CARTOPY:
        gl = ax.gridlines(draw_labels=True, linestyle='--', alpha=0.6, color='gray')
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {'fontsize': 10}
        gl.ylabel_style = {'fontsize': 10}
    else:
        ax.grid(True, linestyle='--', alpha=0.6, color='gray')
        ax.set_xlabel('Longitude', fontsize=12)
        ax.set_ylabel('Latitude', fontsize=12)

    # Set title
    if title is None:
        time_str = position.time_str if position.time_str else f't={position.time_index}'
        title = (
            f'Cyclone Track - {time_str}\n'
            f'Center: ({position.latitude:.2f}\u00b0, {position.longitude:.2f}\u00b0) | '
            f'SLP: {position.central_pressure/100:.1f} hPa | '
            f'Radius: {position.radius_km:.0f} km'
        )
    ax.set_title(title, fontsize=14)

    # Add legend
    ax.legend(loc='upper right', fontsize=10)

    # Set map extent to match data
    if HAS_CARTOPY:
        x_max = xlong.max()
        if x_max > 180:
            x_max = 180

        ax.set_extent([xlong.min(), x_max, xlat.min(), xlat.max()], crs=ccrs.PlateCarree())

    # Save figure
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def plot_cyclone_track(
    cfdb_path: Union[str, pathlib.Path],
    positions: list[CyclonePosition],
    output_dir: Union[str, pathlib.Path],
    filename_prefix: str = 'cyclone',
    slp_levels: list[float] = None,
    figsize: tuple[float, float] = (12, 10),
    dpi: int = 150,
    cmap: str = 'RdYlBu_r',
):
    """
    Plot SLP field with cyclone center and radius for all timesteps.

    Creates one PNG file per timestep in the output directory.

    Parameters
    ----------
    cfdb_path : str or pathlib.Path
        Path to cfdb dataset.
    positions : list[CyclonePosition]
        List of cyclone positions from track_cyclone().
    output_dir : str or pathlib.Path
        Directory to save PNG files.
    filename_prefix : str
        Prefix for output filenames. Default is 'cyclone'.
        Files will be named: {prefix}_t{time_index:03d}.png
    slp_levels : list[float], optional
        Contour levels for SLP in hPa. Default is 960-1040 hPa every 4 hPa.
    figsize : tuple
        Figure size in inches. Default is (12, 10).
    dpi : int
        Output resolution. Default is 150.
    cmap : str
        Colormap for filled contours. Default is 'RdYlBu_r'.

    Returns
    -------
    list[pathlib.Path]
        List of paths to created PNG files.

    Examples
    --------
    >>> positions = track_cyclone('model_output.cfdb', start_lat=-45.0, start_lon=170.0)
    >>> png_files = plot_cyclone_track('model_output.cfdb', positions, './cyclone_plots/')
    """
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_files = []

    for pos in positions:
        out_path = output_dir / f'{filename_prefix}_t{pos.time_index:03d}.png'

        plot_cyclone_timestep(
            cfdb_path=cfdb_path,
            position=pos,
            output_path=out_path,
            slp_levels=slp_levels,
            figsize=figsize,
            dpi=dpi,
            cmap=cmap,
        )

        output_files.append(out_path)

    return output_files


def plot_cyclone_track_multi_file(
    cfdb_paths: list[Union[str, pathlib.Path]],
    positions: list[CyclonePosition],
    output_dir: Union[str, pathlib.Path],
    filename_prefix: str = 'cyclone',
    slp_levels: list[float] = None,
    figsize: tuple[float, float] = (12, 10),
    dpi: int = 150,
    cmap: str = 'RdYlBu_r',
):
    """
    Plot cyclone track across multiple cfdb datasets.

    Parameters
    ----------
    cfdb_paths : list of str or pathlib.Path
        List of cfdb dataset paths in chronological order.
    positions : list[CyclonePosition]
        List of cyclone positions from track_cyclone_multi_file().
    output_dir : str or pathlib.Path
        Directory to save PNG files.
    filename_prefix : str
        Prefix for output filenames. Default is 'cyclone'.
    slp_levels : list[float], optional
        Contour levels for SLP in hPa.
    figsize : tuple
        Figure size in inches.
    dpi : int
        Output resolution.
    cmap : str
        Colormap for filled contours.

    Returns
    -------
    list[pathlib.Path]
        List of paths to created PNG files.
    """
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build mapping of time indices to files
    file_time_ranges = []
    time_offset = 0

    for path in cfdb_paths:
        path = pathlib.Path(path)
        with cfdb.open_dataset(path) as ds:
            n_times = len(ds['time'].data)
        file_time_ranges.append((time_offset, time_offset + n_times, path))
        time_offset += n_times

    output_files = []

    for pos in positions:
        # Find which file contains this timestep
        cfdb_path = None
        local_time_index = pos.time_index

        for start_t, end_t, path in file_time_ranges:
            if start_t <= pos.time_index < end_t:
                cfdb_path = path
                local_time_index = pos.time_index - start_t
                break

        if cfdb_path is None:
            raise ValueError(f"Could not find file for time index {pos.time_index}")

        # Create a temporary position with local time index for plotting
        local_pos = CyclonePosition(
            time_index=local_time_index,
            y_index=pos.y_index,
            x_index=pos.x_index,
            latitude=pos.latitude,
            longitude=pos.longitude,
            central_pressure=pos.central_pressure,
            radius_km=pos.radius_km,
            time_str=pos.time_str,
        )

        out_path = output_dir / f'{filename_prefix}_t{pos.time_index:03d}.png'

        plot_cyclone_timestep(
            cfdb_path=cfdb_path,
            position=local_pos,
            output_path=out_path,
            slp_levels=slp_levels,
            figsize=figsize,
            dpi=dpi,
            cmap=cmap,
        )

        output_files.append(out_path)

    return output_files
