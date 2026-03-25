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

from modverif.cyclone import _read_latlon_2d, _read_var_2d, _read_slp_from_cfdb

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False


def _plot_storm_composite_frame(
    xlat: np.ndarray,
    xlong: np.ndarray,
    vimf_u: np.ndarray,
    vimf_v: np.ndarray,
    pwat: np.ndarray,
    mslp: np.ndarray,
    time_str: str = None,
    output_path: Union[str, pathlib.Path] = None,
    vector_type: str = 'barbs',
    thin_factor: int = None,
    pwat_cmap: str = 'YlGnBu',
    pwat_levels: list = None,
    mslp_levels: list = None,
    mslp_color: str = 'black',
    vector_color: str = 'black',
    figsize: tuple = (14, 10),
    dpi: int = 150,
    title: str = None,
):
    """
    Plot a single storm composite frame.

    Parameters
    ----------
    xlat : np.ndarray
        2D latitude array (ny, nx).
    xlong : np.ndarray
        2D longitude array (ny, nx).
    vimf_u : np.ndarray
        Eastward VIMF component (ny, nx) in kg/m/s.
    vimf_v : np.ndarray
        Northward VIMF component (ny, nx) in kg/m/s.
    pwat : np.ndarray
        Total precipitable water (ny, nx) in kg/m2.
    mslp : np.ndarray
        Mean sea level pressure (ny, nx) in Pa or hPa.
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
    figsize : tuple
        Figure size in inches.
    dpi : int
        Output resolution.
    title : str, optional
        Custom title. Auto-generated if None.

    Returns
    -------
    tuple or None
        ``(fig, ax)`` if ``output_path`` is None, otherwise None.
    """
    # Convert MSLP to hPa if in Pa
    if np.nanmean(mslp) > 10000:
        mslp_hpa = mslp / 100.0
    else:
        mslp_hpa = mslp.copy()

    if mslp_levels is None:
        mslp_levels = list(range(960, 1044, 4))

    if pwat_levels is None:
        pmin = max(0, np.floor(np.nanmin(pwat)))
        pmax = np.ceil(np.nanmax(pwat))
        if pmax > pmin:
            pwat_levels = np.linspace(pmin, pmax, 15)
        else:
            pwat_levels = np.linspace(0, 80, 17)

    ny, nx = xlat.shape
    if thin_factor is None:
        thin_factor = max(1, min(ny, nx) // 25)

    # Create figure
    if HAS_CARTOPY:
        fig, ax = plt.subplots(figsize=figsize, subplot_kw={'projection': ccrs.PlateCarree()})
        transform = ccrs.PlateCarree()
    else:
        fig, ax = plt.subplots(figsize=figsize)
        transform = None

    plot_kwargs = {'transform': transform} if transform else {}

    # Layer 1: PWAT color fill
    cf = ax.contourf(xlong, xlat, pwat, levels=pwat_levels, cmap=pwat_cmap, extend='both', **plot_kwargs)

    # Layer 2: MSLP contour lines
    cs = ax.contour(
        xlong, xlat, mslp_hpa, levels=mslp_levels, colors=mslp_color, linewidths=0.8, **plot_kwargs
    )
    ax.clabel(cs, inline=True, fontsize=8, fmt='%.0f')

    # Layer 3: VIMF vectors (thinned)
    thin = (slice(None, None, thin_factor), slice(None, None, thin_factor))
    lat_t = xlat[thin]
    lon_t = xlong[thin]
    u_t = vimf_u[thin]
    v_t = vimf_v[thin]

    if vector_type == 'barbs':
        # Scale VIMF to knot-like display: divide by 10 to bring typical values
        # (~100-500 kg/m/s) into wind-barb-friendly range (~10-50 knots)
        scale_factor = 10.0
        u_scaled = u_t / scale_factor
        v_scaled = v_t / scale_factor
        ax.barbs(
            lon_t, lat_t, u_scaled, v_scaled,
            length=6, linewidth=0.5, color=vector_color,
            barb_increments={'half': 5, 'full': 10, 'flag': 50},
            **plot_kwargs,
        )
    else:
        magnitude = np.sqrt(u_t**2 + v_t**2)
        ref_val = np.nanpercentile(magnitude, 90) if magnitude.size > 0 else 200.0
        q = ax.quiver(
            lon_t, lat_t, u_t, v_t,
            color=vector_color, scale_units='inches', scale=ref_val / 0.8,
            width=0.003, headwidth=4,
            **plot_kwargs,
        )
        ax.quiverkey(q, 0.9, 1.03, ref_val, f'{ref_val:.0f} kg/m/s', labelpos='E', fontproperties={'size': 9})

    # Map features
    if HAS_CARTOPY:
        ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.3, zorder=0)
        ax.add_feature(cfeature.COASTLINE, linewidth=1, edgecolor='black')
        ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle='--', edgecolor='gray')
        gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)
        gl.top_labels = False
        gl.right_labels = False
        ax.set_extent(
            [xlong.min(), xlong.max(), xlat.min(), xlat.max()],
            crs=ccrs.PlateCarree(),
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
        xlat, xlong = _read_latlon_2d(ds)
        time_values = ds['time'].data

        # Validate time index
        if time_index < 0 or time_index >= len(time_values):
            raise IndexError(f"time_index {time_index} out of range [0, {len(time_values) - 1}]")

        # Read fields
        vimf_u = _read_var_2d(ds, vimf_u_var, time_index)
        vimf_v = _read_var_2d(ds, vimf_v_var, time_index)
        pwat = _read_var_2d(ds, pwat_var, time_index)

        # MSLP with compute fallback
        if mslp_var == 'mslp':
            mslp = _read_slp_from_cfdb(ds, time_index)
        else:
            mslp = _read_var_2d(ds, mslp_var, time_index)

        time_str = str(time_values[time_index])

    return _plot_storm_composite_frame(
        xlat, xlong, vimf_u, vimf_v, pwat, mslp,
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
        xlat, xlong = _read_latlon_2d(ds)
        time_values = ds['time'].data
        n_times = len(time_values)

        # Determine time range
        if start_time is not None:
            start_time = np.datetime64(start_time)
        if end_time is not None:
            end_time = np.datetime64(end_time)

        for t in range(n_times):
            t_val = time_values[t]

            if start_time is not None and t_val < start_time:
                continue
            if end_time is not None and t_val > end_time:
                continue

            # Read fields
            vimf_u = _read_var_2d(ds, vimf_u_var, t)
            vimf_v = _read_var_2d(ds, vimf_v_var, t)
            pwat = _read_var_2d(ds, pwat_var, t)

            if mslp_var == 'mslp':
                mslp = _read_slp_from_cfdb(ds, t)
            else:
                mslp = _read_var_2d(ds, mslp_var, t)

            time_str = str(t_val)
            frame_path = output_dir / f'{filename_prefix}_t{t:03d}.png'

            _plot_storm_composite_frame(
                xlat, xlong, vimf_u, vimf_v, pwat, mslp,
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
