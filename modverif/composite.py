"""
Storm composite plots for assessing cyclone and major storm development.

Overlays Total Precipitable Water (PWAT) as color fill, Mean Sea Level
Pressure (MSLP) as contour lines, and Vertically Integrated Moisture Flux
(VIMF) as wind barbs or quiver vectors on a geographic map.
"""
import pathlib
from typing import Union

import cfdb
import matplotlib.pyplot as plt
import numpy as np
import pyproj

from modverif.cyclone import _read_latlon_2d, _read_var_2d, _read_slp_from_cfdb

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False


def _pyproj_to_cartopy(crs):
    """
    Convert a pyproj CRS to a cartopy CRS projection.

    Supports Lambert Conformal Conic, Polar Stereographic, Mercator,
    and geographic (lat/lon) projections.
    """
    if not HAS_CARTOPY:
        return None

    cf = crs.to_cf()
    proj_name = cf.get('grid_mapping_name', '')

    if proj_name == 'lambert_conformal_conic':
        central_lat = cf.get('latitude_of_projection_origin', 0)
        # Default cutoff=-30 clips SH domains at 30°S.  Set the cutoff
        # on the opposite side of the equator so the full domain is visible.
        cutoff = 30 if central_lat < 0 else -30
        return ccrs.LambertConformal(
            central_longitude=cf.get('longitude_of_central_meridian', 0),
            central_latitude=central_lat,
            standard_parallels=cf.get('standard_parallel', []),
            false_easting=cf.get('false_easting', 0),
            false_northing=cf.get('false_northing', 0),
            cutoff=cutoff,
        )
    elif proj_name == 'polar_stereographic':
        return ccrs.Stereographic(
            central_latitude=cf.get('latitude_of_projection_origin', 90),
            central_longitude=cf.get('straight_vertical_longitude_from_pole', 0),
            false_easting=cf.get('false_easting', 0),
            false_northing=cf.get('false_northing', 0),
        )
    elif proj_name == 'mercator':
        return ccrs.Mercator(
            central_longitude=cf.get('longitude_of_projection_origin', 0),
            false_easting=cf.get('false_easting', 0),
            false_northing=cf.get('false_northing', 0),
        )

    # Default: PlateCarree for geographic CRS
    return ccrs.PlateCarree()


def _read_grid_from_ds(ds):
    """
    Read 2D grid coordinate arrays and CRS info from a cfdb dataset.

    For datasets with ``latitude``/``longitude``: returns lat/lon arrays,
    data_crs=PlateCarree, is_projected=False.

    For projected datasets with ``y``/``x`` and a CRS: returns native
    projected coordinate arrays, the CRS, is_projected=True.

    Returns
    -------
    tuple
        (x2d, y2d, data_crs, is_projected) where x2d/y2d are 2D arrays
        in native coordinates and data_crs is a pyproj.CRS.
    """
    coord_names = set(ds.coord_names)
    var_names = set(ds.data_var_names)

    if 'latitude' in coord_names or 'latitude' in var_names:
        xlat, xlong = _read_latlon_2d(ds)
        # Discard columns past the antimeridian to avoid cartopy wrapping issues
        lon_mask = None
        if xlong.ndim == 2 and np.any(xlong > 180):
            lon_mask = xlong[0, :] <= 180
            xlong = xlong[:, lon_mask]
            xlat = xlat[:, lon_mask]
        return xlong, xlat, None, False, lon_mask

    if 'y' in coord_names and 'x' in coord_names and ds.crs is not None:
        y_1d = ds['y'].data
        x_1d = ds['x'].data
        x2d, y2d = np.meshgrid(x_1d, y_1d)
        return x2d, y2d, ds.crs, True, None

    raise ValueError(
        "Dataset must contain either 'latitude'/'longitude' coordinates "
        "or 'y'/'x' coordinates with a CRS. "
        f"Found coords={ds.coord_names}, crs={ds.crs}"
    )


def _map_aspect_ratio(x2d, y2d, is_projected):
    """
    Estimate the height/width aspect ratio of the map domain.

    For projected grids uses native coordinate extents directly.
    For geographic grids corrects longitude range for latitude.

    Returns
    -------
    float
        Aspect ratio (height / width). Values > 1 mean taller than wide.
    """
    if is_projected:
        x_range = float(x2d.max() - x2d.min())
        y_range = float(y2d.max() - y2d.min())
    else:
        y_range = float(y2d.max() - y2d.min())
        mid_lat = np.radians((float(y2d.max()) + float(y2d.min())) / 2)
        x_range = float(x2d.max() - x2d.min()) * abs(np.cos(mid_lat))

    if x_range <= 0:
        return 1.0
    return y_range / x_range


def _apply_lon_mask(data_2d, lon_mask):
    """Discard columns past the antimeridian if needed."""
    if lon_mask is not None:
        return data_2d[:, lon_mask]
    return data_2d


def _draw_composite_layers(
    ax,
    x2d: np.ndarray,
    y2d: np.ndarray,
    vimf_u: np.ndarray,
    vimf_v: np.ndarray,
    pwat: np.ndarray,
    mslp_hpa: np.ndarray,
    plot_kwargs: dict,
    vector_type: str = 'barbs',
    thin_factor: int = None,
    pwat_cmap: str = 'YlGnBu',
    pwat_levels: list = None,
    mslp_levels: list = None,
    mslp_color: str = 'black',
    vector_color: str = 'black',
    gridline_labels: dict = None,
):
    """
    Draw PWAT, MSLP, and VIMF layers onto an existing axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axis (may be a cartopy GeoAxes).
    x2d : np.ndarray
        2D x-coordinate array (ny, nx).
    y2d : np.ndarray
        2D y-coordinate array (ny, nx).
    vimf_u : np.ndarray
        Eastward VIMF component (ny, nx) in kg/m/s.
    vimf_v : np.ndarray
        Northward VIMF component (ny, nx) in kg/m/s.
    pwat : np.ndarray
        Total precipitable water (ny, nx) in kg/m2.
    mslp_hpa : np.ndarray
        Mean sea level pressure (ny, nx) in hPa.
    plot_kwargs : dict
        Keyword arguments for plotting calls (e.g. ``{'transform': crs}``).
    vector_type : str
        ``'barbs'`` or ``'quiver'``.
    thin_factor : int, optional
        Stride for vector thinning. Auto-computed if None.
    pwat_cmap : str
        Colormap for PWAT fill.
    pwat_levels : list, optional
        Contour levels for PWAT.
    mslp_levels : list, optional
        Contour levels for MSLP in hPa.
    mslp_color : str
        Color for MSLP contour lines.
    vector_color : str
        Color for barbs or quiver arrows.
    gridline_labels : dict, optional
        Controls which edges get gridline labels. Keys: ``'top'``,
        ``'right'``, ``'bottom'``, ``'left'``; values: bool.
        Default suppresses top and right labels.

    Returns
    -------
    matplotlib.contour.QuadContourSet
        The PWAT contourf mappable (for colorbar attachment).
    """
    if mslp_levels is None:
        mslp_levels = list(range(960, 1044, 4))

    if pwat_levels is None:
        pmin = max(0, np.floor(np.nanmin(pwat)))
        pmax = np.ceil(np.nanmax(pwat))
        if pmax > pmin:
            pwat_levels = np.linspace(pmin, pmax, 15)
        else:
            pwat_levels = np.linspace(0, 80, 17)

    ny, nx = x2d.shape
    if thin_factor is None:
        thin_factor = max(1, min(ny, nx) // 20)

    if gridline_labels is None:
        gridline_labels = {'top': False, 'right': False, 'bottom': True, 'left': True}

    # Layer 1: PWAT color fill
    cf = ax.contourf(x2d, y2d, pwat, levels=pwat_levels, cmap=pwat_cmap, extend='both', **plot_kwargs)

    # Layer 2: MSLP contour lines
    cs = ax.contour(
        x2d, y2d, mslp_hpa, levels=mslp_levels, colors=mslp_color, linewidths=0.8, **plot_kwargs
    )
    ax.clabel(cs, inline=True, fontsize=8, fmt='%.0f')

    # Layer 3: VIMF vectors (thinned)
    thin = (slice(None, None, thin_factor), slice(None, None, thin_factor))
    x_t = x2d[thin]
    y_t = y2d[thin]
    u_t = vimf_u[thin]
    v_t = vimf_v[thin]

    if vector_type == 'barbs':
        scale_factor = 20.0
        u_scaled = u_t / scale_factor
        v_scaled = v_t / scale_factor
        ax.barbs(
            x_t, y_t, u_scaled, v_scaled,
            length=5, linewidth=0.4, color=vector_color,
            barb_increments={'half': 5, 'full': 10, 'flag': 50},
            **plot_kwargs,
        )
    else:
        magnitude = np.sqrt(u_t**2 + v_t**2)
        ref_val = np.nanpercentile(magnitude, 90) if magnitude.size > 0 else 200.0
        q = ax.quiver(
            x_t, y_t, u_t, v_t,
            color=vector_color, scale_units='inches', scale=ref_val / 0.8,
            width=0.003, headwidth=4,
            **plot_kwargs,
        )
        ax.quiverkey(q, 0.9, 1.03, ref_val, f'{ref_val:.0f} kg/m/s', labelpos='E', fontproperties={'size': 9})

    # Map features (only on cartopy GeoAxes)
    if HAS_CARTOPY and hasattr(ax, 'add_feature'):
        ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.3, zorder=0)
        ax.add_feature(cfeature.COASTLINE, linewidth=1, edgecolor='black')
        ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle='--', edgecolor='gray')
        gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)
        gl.x_inline = False
        gl.y_inline = False
        gl.rotate_labels = False
        gl.top_labels = gridline_labels.get('top', False)
        gl.right_labels = gridline_labels.get('right', False)
        gl.bottom_labels = gridline_labels.get('bottom', True)
        gl.left_labels = gridline_labels.get('left', True)

    return cf


def _align_times(times_a, times_b, start_time=None, end_time=None):
    """
    Find matching timesteps between two datasets using exact matching.

    Parameters
    ----------
    times_a : np.ndarray
        Time values from dataset A.
    times_b : np.ndarray
        Time values from dataset B.
    start_time : np.datetime64, optional
        Start of time range to include.
    end_time : np.datetime64, optional
        End of time range to include.

    Returns
    -------
    list of tuple
        ``[(time_value, index_in_a, index_in_b), ...]`` sorted by time.
    """
    common = np.intersect1d(times_a, times_b)

    if start_time is not None:
        common = common[common >= start_time]
    if end_time is not None:
        common = common[common <= end_time]

    lookup_a = {t: i for i, t in enumerate(times_a)}
    lookup_b = {t: i for i, t in enumerate(times_b)}

    return [(t, lookup_a[t], lookup_b[t]) for t in common]


def _plot_storm_composite_frame(
    x2d: np.ndarray,
    y2d: np.ndarray,
    vimf_u: np.ndarray,
    vimf_v: np.ndarray,
    pwat: np.ndarray,
    mslp: np.ndarray,
    data_crs=None,
    is_projected: bool = False,
    time_str: str = None,
    output_path: Union[str, pathlib.Path] = None,
    vector_type: str = 'barbs',
    thin_factor: int = None,
    pwat_cmap: str = 'YlGnBu',
    pwat_levels: list = None,
    mslp_levels: list = None,
    mslp_color: str = 'black',
    vector_color: str = 'black',
    figsize: tuple = None,
    dpi: int = 150,
    title: str = None,
):
    """
    Plot a single storm composite frame.

    Parameters
    ----------
    x2d : np.ndarray
        2D x-coordinate array (ny, nx). Longitude if geographic, easting if projected.
    y2d : np.ndarray
        2D y-coordinate array (ny, nx). Latitude if geographic, northing if projected.
    vimf_u : np.ndarray
        Eastward VIMF component (ny, nx) in kg/m/s.
    vimf_v : np.ndarray
        Northward VIMF component (ny, nx) in kg/m/s.
    pwat : np.ndarray
        Total precipitable water (ny, nx) in kg/m2.
    mslp : np.ndarray
        Mean sea level pressure (ny, nx) in Pa or hPa.
    data_crs : pyproj.CRS, optional
        CRS of the data coordinates. If None, assumes geographic (lat/lon).
    is_projected : bool
        Whether coordinates are in a projected CRS.
    time_str : str, optional
        Timestamp string for title.
    output_path : str or pathlib.Path, optional
        Path to save figure. If None, returns (fig, ax).
    vector_type : str
        ``'barbs'`` or ``'quiver'``. Default is ``'barbs'``.
    thin_factor : int, optional
        Stride for vector thinning. Auto-computed if None.
    pwat_cmap : str
        Colormap for PWAT fill. Default is ``'YlGnBu'``.
    pwat_levels : list, optional
        Contour levels for PWAT. Auto-computed from data if None.
    mslp_levels : list, optional
        Contour levels for MSLP in hPa. Default is 960–1040 every 4 hPa.
    mslp_color : str
        Color for MSLP contour lines. Default is ``'black'``.
    vector_color : str
        Color for barbs or quiver arrows. Default is ``'black'``.
    figsize : tuple, optional
        Figure size in inches. Auto-computed from domain aspect ratio if None.
    dpi : int
        Output resolution.
    title : str, optional
        Custom title. Auto-generated if None.

    Returns
    -------
    tuple or None
        ``(fig, ax)`` if ``output_path`` is None, otherwise None.
    """
    if figsize is None:
        figsize = (14, 10)
    # Convert MSLP to hPa if in Pa
    if np.nanmean(mslp) > 10000:
        mslp_hpa = mslp / 100.0
    else:
        mslp_hpa = mslp.copy()

    # Set up cartopy projections
    if HAS_CARTOPY:
        if is_projected and data_crs is not None:
            map_projection = _pyproj_to_cartopy(data_crs)
            data_transform = map_projection
        else:
            map_projection = ccrs.PlateCarree()
            data_transform = ccrs.PlateCarree()

        fig, ax = plt.subplots(figsize=figsize, subplot_kw={'projection': map_projection})
        plot_kwargs = {'transform': data_transform}

    else:
        fig, ax = plt.subplots(figsize=figsize)
        plot_kwargs = {}

    # Draw layers
    cf = _draw_composite_layers(
        ax, x2d, y2d, vimf_u, vimf_v, pwat, mslp_hpa,
        plot_kwargs,
        vector_type=vector_type, thin_factor=thin_factor,
        pwat_cmap=pwat_cmap, pwat_levels=pwat_levels,
        mslp_levels=mslp_levels, mslp_color=mslp_color,
        vector_color=vector_color,
    )

    # Colorbar
    cbar = plt.colorbar(cf, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('Precipitable Water (kg/m\u00b2)', fontsize=11)

    # Title
    if title is None:
        parts = ['Storm Composite']
        if time_str is not None:
            parts.append(f'- {time_str}')
        title = ' '.join(parts)
    ax.set_title(title, fontsize=14)

    if output_path is not None:
        output_path = pathlib.Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        return None

    return fig, ax


def plot_storm_composite_timestep(
    cfdb_path: Union[str, pathlib.Path],
    time_index: int,
    output_path: Union[str, pathlib.Path] = None,
    vimf_u_var: str = 'vimf_u',
    vimf_v_var: str = 'vimf_v',
    pwat_var: str = 'pwat',
    mslp_var: str = 'mslp',
    **plot_kwargs,
):
    """
    Plot a storm composite for a single timestep from a cfdb dataset.

    Parameters
    ----------
    cfdb_path : str or pathlib.Path
        Path to cfdb dataset containing PWAT, MSLP, and VIMF variables.
    time_index : int
        Timestep index to plot.
    output_path : str or pathlib.Path, optional
        Path to save figure. If None, returns (fig, ax).
    vimf_u_var : str
        Name of the eastward VIMF variable. Default is ``'vimf_u'``.
    vimf_v_var : str
        Name of the northward VIMF variable. Default is ``'vimf_v'``.
    pwat_var : str
        Name of the PWAT variable. Default is ``'pwat'``.
    mslp_var : str
        Name of the MSLP variable. Default is ``'mslp'``.
    **plot_kwargs
        Additional keyword arguments passed to ``_plot_storm_composite_frame``.

    Returns
    -------
    tuple or None
        ``(fig, ax)`` if ``output_path`` is None, otherwise None.
    """
    cfdb_path = pathlib.Path(cfdb_path)
    if not cfdb_path.exists():
        raise FileNotFoundError(f"Dataset not found: {cfdb_path}")

    with cfdb.open_dataset(cfdb_path) as ds:
        x2d, y2d, data_crs, is_projected, lon_mask = _read_grid_from_ds(ds)
        time_values = ds['time'].data

        # Validate time index
        if time_index < 0 or time_index >= len(time_values):
            raise IndexError(f"time_index {time_index} out of range [0, {len(time_values) - 1}]")

        # Read fields
        vimf_u = _apply_lon_mask(_read_var_2d(ds, vimf_u_var, time_index), lon_mask)
        vimf_v = _apply_lon_mask(_read_var_2d(ds, vimf_v_var, time_index), lon_mask)
        pwat = _apply_lon_mask(_read_var_2d(ds, pwat_var, time_index), lon_mask)

        # MSLP with compute fallback
        if mslp_var == 'mslp':
            mslp = _apply_lon_mask(_read_slp_from_cfdb(ds, time_index), lon_mask)
        else:
            mslp = _apply_lon_mask(_read_var_2d(ds, mslp_var, time_index), lon_mask)

        time_str = str(time_values[time_index])

    return _plot_storm_composite_frame(
        x2d, y2d, vimf_u, vimf_v, pwat, mslp,
        data_crs=data_crs, is_projected=is_projected,
        time_str=time_str, output_path=output_path,
        **plot_kwargs,
    )


def plot_storm_composite(
    cfdb_path: Union[str, pathlib.Path],
    output_dir: Union[str, pathlib.Path],
    filename_prefix: str = 'storm_composite',
    webp_path: Union[str, pathlib.Path] = None,
    webp_duration: int = 500,
    webp_loop: int = 0,
    webp_quality: int = 80,
    start_time: Union[str, np.datetime64, None] = None,
    end_time: Union[str, np.datetime64, None] = None,
    vimf_u_var: str = 'vimf_u',
    vimf_v_var: str = 'vimf_v',
    pwat_var: str = 'pwat',
    mslp_var: str = 'mslp',
    **plot_kwargs,
) -> tuple[list[pathlib.Path], pathlib.Path]:
    """
    Generate storm composite images for all timesteps and an animated WebP.

    Parameters
    ----------
    cfdb_path : str or pathlib.Path
        Path to cfdb dataset.
    output_dir : str or pathlib.Path
        Directory to save individual PNG frames.
    filename_prefix : str
        Prefix for frame filenames. Default is ``'storm_composite'``.
    webp_path : str or pathlib.Path, optional
        Path for the animated WebP file. If None, saved as
        ``{output_dir}/{filename_prefix}.webp``.
    webp_duration : int
        Milliseconds per frame in the animation. Default is 500.
    webp_loop : int
        Number of animation loops (0 = infinite). Default is 0.
    webp_quality : int
        WebP quality (1–100). Default is 80.
    start_time : str or np.datetime64, optional
        Start of time range to include.
    end_time : str or np.datetime64, optional
        End of time range to include.
    vimf_u_var : str
        Name of the eastward VIMF variable. Default is ``'vimf_u'``.
    vimf_v_var : str
        Name of the northward VIMF variable. Default is ``'vimf_v'``.
    pwat_var : str
        Name of the PWAT variable. Default is ``'pwat'``.
    mslp_var : str
        Name of the MSLP variable. Default is ``'mslp'``.
    **plot_kwargs
        Additional keyword arguments passed to ``_plot_storm_composite_frame``.

    Returns
    -------
    tuple[list[pathlib.Path], pathlib.Path]
        ``(png_files, webp_path)`` — list of individual frame paths and
        the path to the animated WebP.
    """
    cfdb_path = pathlib.Path(cfdb_path)
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if webp_path is None:
        webp_path = output_dir / f'{filename_prefix}.webp'
    else:
        webp_path = pathlib.Path(webp_path)

    if not cfdb_path.exists():
        raise FileNotFoundError(f"Dataset not found: {cfdb_path}")

    png_files = []

    with cfdb.open_dataset(cfdb_path) as ds:
        x2d, y2d, data_crs, is_projected, lon_mask = _read_grid_from_ds(ds)
        time_values = ds['time'].data
        n_times = len(time_values)

        # Determine time range
        if start_time is not None:
            start_time = np.datetime64(start_time)
        if end_time is not None:
            end_time = np.datetime64(end_time)

        # Compute consistent PWAT levels across all frames if not provided
        if 'pwat_levels' not in plot_kwargs or plot_kwargs.get('pwat_levels') is None:
            global_min = np.inf
            global_max = -np.inf
            for t in range(n_times):
                t_val = time_values[t]
                if start_time is not None and t_val < start_time:
                    continue
                if end_time is not None and t_val > end_time:
                    continue
                pwat_t = _read_var_2d(ds, pwat_var, t)
                global_min = min(global_min, np.nanmin(pwat_t))
                global_max = max(global_max, np.nanmax(pwat_t))
            pmin = max(0, np.floor(global_min))
            pmax = np.ceil(global_max)
            if pmax > pmin:
                plot_kwargs['pwat_levels'] = np.linspace(pmin, pmax, 15)
            else:
                plot_kwargs['pwat_levels'] = np.linspace(0, 80, 17)

        for t in range(n_times):
            t_val = time_values[t]

            if start_time is not None and t_val < start_time:
                continue
            if end_time is not None and t_val > end_time:
                continue

            # Read fields
            vimf_u = _apply_lon_mask(_read_var_2d(ds, vimf_u_var, t), lon_mask)
            vimf_v = _apply_lon_mask(_read_var_2d(ds, vimf_v_var, t), lon_mask)
            pwat = _apply_lon_mask(_read_var_2d(ds, pwat_var, t), lon_mask)

            if mslp_var == 'mslp':
                mslp = _apply_lon_mask(_read_slp_from_cfdb(ds, t), lon_mask)
            else:
                mslp = _apply_lon_mask(_read_var_2d(ds, mslp_var, t), lon_mask)

            time_str = str(t_val)
            frame_path = output_dir / f'{filename_prefix}_{np.datetime_as_string(t_val, unit="h")}.png'

            _plot_storm_composite_frame(
                x2d, y2d, vimf_u, vimf_v, pwat, mslp,
                data_crs=data_crs, is_projected=is_projected,
                time_str=time_str, output_path=frame_path,
                **plot_kwargs,
            )
            png_files.append(frame_path)

    # Assemble animated WebP
    _assemble_webp(png_files, webp_path, webp_duration, webp_loop, webp_quality)

    return png_files, webp_path


def _plot_storm_composite_comparison_frame(
    x2d_a: np.ndarray,
    y2d_a: np.ndarray,
    vimf_u_a: np.ndarray,
    vimf_v_a: np.ndarray,
    pwat_a: np.ndarray,
    mslp_a: np.ndarray,
    data_crs_a=None,
    is_projected_a: bool = False,
    label_a: str = 'Model A',
    x2d_b: np.ndarray = None,
    y2d_b: np.ndarray = None,
    vimf_u_b: np.ndarray = None,
    vimf_v_b: np.ndarray = None,
    pwat_b: np.ndarray = None,
    mslp_b: np.ndarray = None,
    data_crs_b=None,
    is_projected_b: bool = False,
    label_b: str = 'Model B',
    time_str: str = None,
    output_path: Union[str, pathlib.Path] = None,
    vector_type: str = 'barbs',
    thin_factor: int = None,
    pwat_cmap: str = 'YlGnBu',
    pwat_levels: list = None,
    mslp_levels: list = None,
    mslp_color: str = 'black',
    vector_color: str = 'black',
    figsize: tuple = None,
    dpi: int = 150,
    title: str = None,
):
    """
    Plot a side-by-side storm composite comparison frame.

    Parameters
    ----------
    x2d_a, y2d_a : np.ndarray
        2D coordinate arrays for dataset A.
    vimf_u_a, vimf_v_a, pwat_a, mslp_a : np.ndarray
        Field arrays for dataset A (ny, nx).
    data_crs_a : pyproj.CRS, optional
        CRS of dataset A. If None, assumes geographic (lat/lon).
    is_projected_a : bool
        Whether dataset A coordinates are projected.
    label_a : str
        Title label for the left panel.
    x2d_b, y2d_b : np.ndarray
        2D coordinate arrays for dataset B.
    vimf_u_b, vimf_v_b, pwat_b, mslp_b : np.ndarray
        Field arrays for dataset B (ny, nx).
    data_crs_b : pyproj.CRS, optional
        CRS of dataset B. If None, assumes geographic (lat/lon).
    is_projected_b : bool
        Whether dataset B coordinates are projected.
    label_b : str
        Title label for the right panel.
    time_str : str, optional
        Timestamp string for the suptitle.
    output_path : str or pathlib.Path, optional
        Path to save figure. If None, returns ``(fig, (ax_a, ax_b))``.
    vector_type : str
        ``'barbs'`` or ``'quiver'``.
    thin_factor : int, optional
        Stride for vector thinning. Auto-computed per panel if None.
    pwat_cmap : str
        Colormap for PWAT fill.
    pwat_levels : list, optional
        Shared contour levels for PWAT. Auto-computed from both datasets if None.
    mslp_levels : list, optional
        Shared contour levels for MSLP in hPa.
    mslp_color : str
        Color for MSLP contour lines.
    vector_color : str
        Color for barbs or quiver arrows.
    figsize : tuple, optional
        Figure size in inches. Auto-computed from domain aspect ratio if None.
    dpi : int
        Output resolution.
    title : str, optional
        Custom suptitle. Auto-generated if None.

    Returns
    -------
    tuple or None
        ``(fig, (ax_a, ax_b))`` if ``output_path`` is None, otherwise None.
    """
    # Convert MSLP to hPa
    mslp_hpa_a = mslp_a / 100.0 if np.nanmean(mslp_a) > 10000 else mslp_a.copy()
    mslp_hpa_b = mslp_b / 100.0 if np.nanmean(mslp_b) > 10000 else mslp_b.copy()

    # Compute shared PWAT levels from both datasets if not provided
    if pwat_levels is None:
        pmin = max(0, np.floor(min(np.nanmin(pwat_a), np.nanmin(pwat_b))))
        pmax = np.ceil(max(np.nanmax(pwat_a), np.nanmax(pwat_b)))
        if pmax > pmin:
            pwat_levels = np.linspace(pmin, pmax, 15)
        else:
            pwat_levels = np.linspace(0, 80, 17)

    # Auto-compute figsize and wspace from domain aspect ratio
    map_aspect = _map_aspect_ratio(x2d_a, y2d_a, is_projected_a)
    fig_height = 10
    if figsize is None:
        panel_width = fig_height / max(map_aspect, 0.3)
        figsize = (2 * panel_width + 3, fig_height)

    wspace = 0.02

    # Set up projections for each panel
    if HAS_CARTOPY:
        if is_projected_a and data_crs_a is not None:
            proj_a = _pyproj_to_cartopy(data_crs_a)
            transform_a = proj_a
        else:
            proj_a = ccrs.PlateCarree()
            transform_a = ccrs.PlateCarree()

        if is_projected_b and data_crs_b is not None:
            proj_b = _pyproj_to_cartopy(data_crs_b)
            transform_b = proj_b
        else:
            proj_b = ccrs.PlateCarree()
            transform_b = ccrs.PlateCarree()

        from matplotlib.gridspec import GridSpec
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(1, 2, figure=fig, wspace=wspace)
        ax_a = fig.add_subplot(gs[0, 0], projection=proj_a)
        ax_b = fig.add_subplot(gs[0, 1], projection=proj_b)
        plot_kwargs_a = {'transform': transform_a}
        plot_kwargs_b = {'transform': transform_b}

    else:
        fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=figsize)
        plot_kwargs_a = {}
        plot_kwargs_b = {}

    # Draw layers on each panel
    cf_a = _draw_composite_layers(
        ax_a, x2d_a, y2d_a, vimf_u_a, vimf_v_a, pwat_a, mslp_hpa_a,
        plot_kwargs_a,
        vector_type=vector_type, thin_factor=thin_factor,
        pwat_cmap=pwat_cmap, pwat_levels=pwat_levels,
        mslp_levels=mslp_levels, mslp_color=mslp_color,
        vector_color=vector_color,
        gridline_labels={'top': False, 'right': False, 'bottom': True, 'left': True},
    )
    _draw_composite_layers(
        ax_b, x2d_b, y2d_b, vimf_u_b, vimf_v_b, pwat_b, mslp_hpa_b,
        plot_kwargs_b,
        vector_type=vector_type, thin_factor=thin_factor,
        pwat_cmap=pwat_cmap, pwat_levels=pwat_levels,
        mslp_levels=mslp_levels, mslp_color=mslp_color,
        vector_color=vector_color,
        gridline_labels={'top': False, 'right': False, 'bottom': True, 'left': False},
    )

    # Per-panel titles
    ax_a.set_title(label_a, fontsize=13)
    ax_b.set_title(label_b, fontsize=13)

    # Shared colorbar
    cbar = fig.colorbar(cf_a, ax=[ax_a, ax_b], shrink=0.8, pad=0.02)
    cbar.set_label('Precipitable Water (kg/m\u00b2)', fontsize=11)

    # Suptitle
    if title is None:
        parts = ['Storm Composite Comparison']
        if time_str is not None:
            parts.append(f'- {time_str}')
        title = ' '.join(parts)
    fig.suptitle(title, fontsize=14, y=1.02)

    if output_path is not None:
        output_path = pathlib.Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        return None

    return fig, (ax_a, ax_b)


def plot_storm_composite_comparison_timestep(
    cfdb_path_a: Union[str, pathlib.Path],
    cfdb_path_b: Union[str, pathlib.Path],
    time_index: int,
    output_path: Union[str, pathlib.Path] = None,
    label_a: str = 'Model A',
    label_b: str = 'Model B',
    vimf_u_var_a: str = 'vimf_u',
    vimf_v_var_a: str = 'vimf_v',
    pwat_var_a: str = 'pwat',
    mslp_var_a: str = 'mslp',
    vimf_u_var_b: str = 'vimf_u',
    vimf_v_var_b: str = 'vimf_v',
    pwat_var_b: str = 'pwat',
    mslp_var_b: str = 'mslp',
    **plot_kwargs,
):
    """
    Plot a side-by-side storm composite comparison for a single timestep.

    Opens both datasets, finds the intersection of their time arrays,
    and plots the frame at the given index into the matched times.

    Parameters
    ----------
    cfdb_path_a : str or pathlib.Path
        Path to the first cfdb dataset.
    cfdb_path_b : str or pathlib.Path
        Path to the second cfdb dataset.
    time_index : int
        Index into the matched (intersected) timesteps.
    output_path : str or pathlib.Path, optional
        Path to save figure. If None, returns ``(fig, (ax_a, ax_b))``.
    label_a : str
        Title label for the left panel. Default is ``'Model A'``.
    label_b : str
        Title label for the right panel. Default is ``'Model B'``.
    vimf_u_var_a, vimf_v_var_a, pwat_var_a, mslp_var_a : str
        Variable names in dataset A.
    vimf_u_var_b, vimf_v_var_b, pwat_var_b, mslp_var_b : str
        Variable names in dataset B.
    **plot_kwargs
        Additional keyword arguments passed to
        ``_plot_storm_composite_comparison_frame``.

    Returns
    -------
    tuple or None
        ``(fig, (ax_a, ax_b))`` if ``output_path`` is None, otherwise None.
    """
    cfdb_path_a = pathlib.Path(cfdb_path_a)
    cfdb_path_b = pathlib.Path(cfdb_path_b)
    if not cfdb_path_a.exists():
        raise FileNotFoundError(f"Dataset not found: {cfdb_path_a}")
    if not cfdb_path_b.exists():
        raise FileNotFoundError(f"Dataset not found: {cfdb_path_b}")

    with cfdb.open_dataset(cfdb_path_a) as ds_a, cfdb.open_dataset(cfdb_path_b) as ds_b:
        x2d_a, y2d_a, data_crs_a, is_projected_a, lon_mask_a = _read_grid_from_ds(ds_a)
        x2d_b, y2d_b, data_crs_b, is_projected_b, lon_mask_b = _read_grid_from_ds(ds_b)

        times_a = ds_a['time'].data
        times_b = ds_b['time'].data
        matched = _align_times(times_a, times_b)

        if not matched:
            raise ValueError("No common timesteps between the two datasets.")
        if time_index < 0 or time_index >= len(matched):
            raise IndexError(f"time_index {time_index} out of range [0, {len(matched) - 1}]")

        t_val, idx_a, idx_b = matched[time_index]

        # Read fields from dataset A
        vimf_u_a = _apply_lon_mask(_read_var_2d(ds_a, vimf_u_var_a, idx_a), lon_mask_a)
        vimf_v_a = _apply_lon_mask(_read_var_2d(ds_a, vimf_v_var_a, idx_a), lon_mask_a)
        pwat_a = _apply_lon_mask(_read_var_2d(ds_a, pwat_var_a, idx_a), lon_mask_a)
        if mslp_var_a == 'mslp':
            mslp_a = _apply_lon_mask(_read_slp_from_cfdb(ds_a, idx_a), lon_mask_a)
        else:
            mslp_a = _apply_lon_mask(_read_var_2d(ds_a, mslp_var_a, idx_a), lon_mask_a)

        # Read fields from dataset B
        vimf_u_b = _apply_lon_mask(_read_var_2d(ds_b, vimf_u_var_b, idx_b), lon_mask_b)
        vimf_v_b = _apply_lon_mask(_read_var_2d(ds_b, vimf_v_var_b, idx_b), lon_mask_b)
        pwat_b = _apply_lon_mask(_read_var_2d(ds_b, pwat_var_b, idx_b), lon_mask_b)
        if mslp_var_b == 'mslp':
            mslp_b = _apply_lon_mask(_read_slp_from_cfdb(ds_b, idx_b), lon_mask_b)
        else:
            mslp_b = _apply_lon_mask(_read_var_2d(ds_b, mslp_var_b, idx_b), lon_mask_b)

        time_str = str(t_val)

    return _plot_storm_composite_comparison_frame(
        x2d_a, y2d_a, vimf_u_a, vimf_v_a, pwat_a, mslp_a,
        data_crs_a=data_crs_a, is_projected_a=is_projected_a, label_a=label_a,
        x2d_b=x2d_b, y2d_b=y2d_b,
        vimf_u_b=vimf_u_b, vimf_v_b=vimf_v_b, pwat_b=pwat_b, mslp_b=mslp_b,
        data_crs_b=data_crs_b, is_projected_b=is_projected_b, label_b=label_b,
        time_str=time_str, output_path=output_path,
        **plot_kwargs,
    )


def plot_storm_composite_comparison(
    cfdb_path_a: Union[str, pathlib.Path],
    cfdb_path_b: Union[str, pathlib.Path],
    output_dir: Union[str, pathlib.Path],
    filename_prefix: str = 'storm_composite_comparison',
    label_a: str = 'Model A',
    label_b: str = 'Model B',
    webp_path: Union[str, pathlib.Path] = None,
    webp_duration: int = 500,
    webp_loop: int = 0,
    webp_quality: int = 80,
    start_time: Union[str, np.datetime64, None] = None,
    end_time: Union[str, np.datetime64, None] = None,
    vimf_u_var_a: str = 'vimf_u',
    vimf_v_var_a: str = 'vimf_v',
    pwat_var_a: str = 'pwat',
    mslp_var_a: str = 'mslp',
    vimf_u_var_b: str = 'vimf_u',
    vimf_v_var_b: str = 'vimf_v',
    pwat_var_b: str = 'pwat',
    mslp_var_b: str = 'mslp',
    **plot_kwargs,
) -> tuple[list[pathlib.Path], pathlib.Path]:
    """
    Generate side-by-side storm composite comparison images and animated WebP.

    Opens both datasets, finds the intersection of their time arrays,
    and generates a comparison frame for each matched timestep.

    Parameters
    ----------
    cfdb_path_a : str or pathlib.Path
        Path to the first cfdb dataset.
    cfdb_path_b : str or pathlib.Path
        Path to the second cfdb dataset.
    output_dir : str or pathlib.Path
        Directory to save individual PNG frames.
    filename_prefix : str
        Prefix for frame filenames. Default is ``'storm_composite_comparison'``.
    label_a : str
        Title label for the left panel. Default is ``'Model A'``.
    label_b : str
        Title label for the right panel. Default is ``'Model B'``.
    webp_path : str or pathlib.Path, optional
        Path for the animated WebP file. If None, saved as
        ``{output_dir}/{filename_prefix}.webp``.
    webp_duration : int
        Milliseconds per frame in the animation. Default is 500.
    webp_loop : int
        Number of animation loops (0 = infinite). Default is 0.
    webp_quality : int
        WebP quality (1–100). Default is 80.
    start_time : str or np.datetime64, optional
        Start of time range to include.
    end_time : str or np.datetime64, optional
        End of time range to include.
    vimf_u_var_a, vimf_v_var_a, pwat_var_a, mslp_var_a : str
        Variable names in dataset A.
    vimf_u_var_b, vimf_v_var_b, pwat_var_b, mslp_var_b : str
        Variable names in dataset B.
    **plot_kwargs
        Additional keyword arguments passed to
        ``_plot_storm_composite_comparison_frame``.

    Returns
    -------
    tuple[list[pathlib.Path], pathlib.Path]
        ``(png_files, webp_path)`` — list of individual frame paths and
        the path to the animated WebP.
    """
    cfdb_path_a = pathlib.Path(cfdb_path_a)
    cfdb_path_b = pathlib.Path(cfdb_path_b)
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if webp_path is None:
        webp_path = output_dir / f'{filename_prefix}.webp'
    else:
        webp_path = pathlib.Path(webp_path)

    if not cfdb_path_a.exists():
        raise FileNotFoundError(f"Dataset not found: {cfdb_path_a}")
    if not cfdb_path_b.exists():
        raise FileNotFoundError(f"Dataset not found: {cfdb_path_b}")

    png_files = []

    with cfdb.open_dataset(cfdb_path_a) as ds_a, cfdb.open_dataset(cfdb_path_b) as ds_b:
        x2d_a, y2d_a, data_crs_a, is_projected_a, lon_mask_a = _read_grid_from_ds(ds_a)
        x2d_b, y2d_b, data_crs_b, is_projected_b, lon_mask_b = _read_grid_from_ds(ds_b)

        times_a = ds_a['time'].data
        times_b = ds_b['time'].data

        # Parse time range
        st = np.datetime64(start_time) if start_time is not None else None
        et = np.datetime64(end_time) if end_time is not None else None

        matched = _align_times(times_a, times_b, start_time=st, end_time=et)
        if not matched:
            raise ValueError("No common timesteps between the two datasets in the specified range.")

        # Compute consistent PWAT levels across both datasets and all frames
        if 'pwat_levels' not in plot_kwargs or plot_kwargs.get('pwat_levels') is None:
            global_min = np.inf
            global_max = -np.inf
            for t_val, idx_a, idx_b in matched:
                pwat_t_a = _read_var_2d(ds_a, pwat_var_a, idx_a)
                pwat_t_b = _read_var_2d(ds_b, pwat_var_b, idx_b)
                global_min = min(global_min, np.nanmin(pwat_t_a), np.nanmin(pwat_t_b))
                global_max = max(global_max, np.nanmax(pwat_t_a), np.nanmax(pwat_t_b))
            pmin = max(0, np.floor(global_min))
            pmax = np.ceil(global_max)
            if pmax > pmin:
                plot_kwargs['pwat_levels'] = np.linspace(pmin, pmax, 15)
            else:
                plot_kwargs['pwat_levels'] = np.linspace(0, 80, 17)

        for i, (t_val, idx_a, idx_b) in enumerate(matched):
            # Read fields from dataset A
            vimf_u_a = _apply_lon_mask(_read_var_2d(ds_a, vimf_u_var_a, idx_a), lon_mask_a)
            vimf_v_a = _apply_lon_mask(_read_var_2d(ds_a, vimf_v_var_a, idx_a), lon_mask_a)
            pwat_a = _apply_lon_mask(_read_var_2d(ds_a, pwat_var_a, idx_a), lon_mask_a)
            if mslp_var_a == 'mslp':
                mslp_a = _apply_lon_mask(_read_slp_from_cfdb(ds_a, idx_a), lon_mask_a)
            else:
                mslp_a = _apply_lon_mask(_read_var_2d(ds_a, mslp_var_a, idx_a), lon_mask_a)

            # Read fields from dataset B
            vimf_u_b = _apply_lon_mask(_read_var_2d(ds_b, vimf_u_var_b, idx_b), lon_mask_b)
            vimf_v_b = _apply_lon_mask(_read_var_2d(ds_b, vimf_v_var_b, idx_b), lon_mask_b)
            pwat_b = _apply_lon_mask(_read_var_2d(ds_b, pwat_var_b, idx_b), lon_mask_b)
            if mslp_var_b == 'mslp':
                mslp_b = _apply_lon_mask(_read_slp_from_cfdb(ds_b, idx_b), lon_mask_b)
            else:
                mslp_b = _apply_lon_mask(_read_var_2d(ds_b, mslp_var_b, idx_b), lon_mask_b)

            time_str = str(t_val)
            frame_path = output_dir / f'{filename_prefix}_{np.datetime_as_string(t_val, unit="h")}.png'

            _plot_storm_composite_comparison_frame(
                x2d_a, y2d_a, vimf_u_a, vimf_v_a, pwat_a, mslp_a,
                data_crs_a=data_crs_a, is_projected_a=is_projected_a, label_a=label_a,
                x2d_b=x2d_b, y2d_b=y2d_b,
                vimf_u_b=vimf_u_b, vimf_v_b=vimf_v_b, pwat_b=pwat_b, mslp_b=mslp_b,
                data_crs_b=data_crs_b, is_projected_b=is_projected_b, label_b=label_b,
                time_str=time_str, output_path=frame_path,
                **plot_kwargs,
            )
            png_files.append(frame_path)

    # Assemble animated WebP
    _assemble_webp(png_files, webp_path, webp_duration, webp_loop, webp_quality)

    return png_files, webp_path


def _assemble_webp(
    png_files: list[pathlib.Path],
    webp_path: pathlib.Path,
    duration: int = 500,
    loop: int = 0,
    quality: int = 80,
):
    """
    Assemble PNG frames into an animated WebP file using Pillow.

    Parameters
    ----------
    png_files : list of pathlib.Path
        Ordered list of frame image paths.
    webp_path : pathlib.Path
        Output path for the animated WebP.
    duration : int
        Milliseconds per frame.
    loop : int
        Number of loops (0 = infinite).
    quality : int
        WebP quality (1–100).
    """
    if not png_files:
        return

    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError("Pillow is required for WebP animation. Install with: pip install Pillow") from exc

    images = []
    for png_path in png_files:
        img = Image.open(png_path).convert('RGB')
        images.append(img)

    # Ensure all frames have the same dimensions (required by WebP encoder)
    target_size = images[0].size
    for i in range(1, len(images)):
        if images[i].size != target_size:
            images[i] = images[i].resize(target_size, Image.LANCZOS)

    webp_path.parent.mkdir(parents=True, exist_ok=True)

    images[0].save(
        str(webp_path),
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=loop,
        quality=quality,
    )

    for img in images:
        img.close()
