# Storm Composite Plots

Storm composite plots overlay three meteorological fields on a geographic map:

- **PWAT** (Total Precipitable Water) as color-filled contours
- **MSLP** (Mean Sea Level Pressure) as contour lines with labels
- **VIMF** (Vertically Integrated Moisture Flux) as wind barbs or quiver arrows

These plots are useful for assessing cyclone and major storm development.

## Single Timestep

```python
from modverif.composite import plot_storm_composite_timestep

fig, ax = plot_storm_composite_timestep(
    'model.cfdb',
    time_index=0,
)

# Or save directly
plot_storm_composite_timestep(
    'model.cfdb',
    time_index=0,
    output_path='composite_t0.png',
)
```

### Custom Variable Names

If your dataset uses different variable names:

```python
plot_storm_composite_timestep(
    'model.cfdb',
    time_index=0,
    vimf_u_var='vimf_u',
    vimf_v_var='vimf_v',
    pwat_var='pwat',
    mslp_var='mslp',
)
```

## Animated Sequence

Generate PNG frames for all timesteps plus an animated WebP:

```python
from modverif.composite import plot_storm_composite

png_files, webp_path = plot_storm_composite(
    'model.cfdb',
    output_dir='composite_frames/',
)
```

### Time Filtering

```python
png_files, webp_path = plot_storm_composite(
    'model.cfdb',
    output_dir='composite_frames/',
    start_time='2023-02-12T06:00',
    end_time='2023-02-13T00:00',
)
```

### Animation Options

```python
png_files, webp_path = plot_storm_composite(
    'model.cfdb',
    output_dir='composite_frames/',
    webp_duration=500,   # ms per frame
    webp_quality=80,     # 1-100
    webp_loop=0,         # 0 = infinite
)
```

## Side-by-Side Model Comparison

Compare two models (e.g., WRF vs ERA5) with synchronized timesteps:

```python
from modverif.composite import plot_storm_composite_comparison

png_files, webp_path = plot_storm_composite_comparison(
    'wrf.cfdb',
    'era5.cfdb',
    output_dir='comparison_frames/',
    label_a='WRF',
    label_b='ERA5',
)
```

Only timesteps that exist in both datasets are plotted. The color scale is computed from both datasets for consistent comparison.

### Single Comparison Timestep

```python
from modverif.composite import plot_storm_composite_comparison_timestep

fig, (ax_a, ax_b) = plot_storm_composite_comparison_timestep(
    'wrf.cfdb',
    'era5.cfdb',
    time_index=0,           # index into matched timesteps
    label_a='WRF',
    label_b='ERA5',
)
```

The `time_index` refers to the index into the intersection of both datasets' time arrays, not either dataset's raw time array.

### Different Variable Names Per Dataset

When datasets use different naming conventions:

```python
png_files, webp_path = plot_storm_composite_comparison(
    'wrf.cfdb',
    'era5.cfdb',
    output_dir='comparison_frames/',
    label_a='WRF',
    label_b='ERA5',
    # WRF variable names
    vimf_u_var_a='vimf_u',
    vimf_v_var_a='vimf_v',
    pwat_var_a='pwat',
    mslp_var_a='mslp',
    # ERA5 variable names
    vimf_u_var_b='vimf_u',
    vimf_v_var_b='vimf_v',
    pwat_var_b='tcwv',
    mslp_var_b='sp',
)
```

### Different Projections

Each panel automatically uses its own map projection based on the dataset's CRS. For example, a WRF dataset in Lambert Conformal and an ERA5 dataset in PlateCarree will each render correctly in their respective panels.

## Plot Customization

All composite functions accept these keyword arguments:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `vector_type` | `'barbs'` | `'barbs'` or `'quiver'` for VIMF display |
| `thin_factor` | auto | Stride for vector thinning (auto-computed from grid size) |
| `pwat_cmap` | `'YlGnBu'` | Colormap for PWAT fill |
| `pwat_levels` | auto | Contour levels for PWAT (auto-computed from data) |
| `mslp_levels` | 960--1040 by 4 | Contour levels for MSLP in hPa |
| `mslp_color` | `'black'` | Color for MSLP contour lines |
| `vector_color` | `'black'` | Color for barbs or quiver arrows |
| `figsize` | auto | Figure size (auto-computed from domain aspect ratio) |
| `dpi` | 150 | Output resolution |

## Cartopy Projection Notes

modverif uses [cartopy](https://scitools.org.uk/cartopy/) for geographic map projections. Cartopy is optional -- plots fall back to plain matplotlib axes if it is not installed -- but there are several non-obvious behaviours to be aware of when working with projected WRF domains.

### Lambert Conformal `cutoff` and Southern Hemisphere domains

Cartopy's `LambertConformal` projection has a `cutoff` parameter (default `-30`) that silently limits how far the map extends from the projection centre. The default boundary sits at 30°S, so any data north of 30°S is simply not displayed -- with no error or warning.

This is a common issue for Southern Hemisphere WRF domains. For example, a domain centred on New Zealand at 34°S with an outer domain extending to 15°S will have its northern portion clipped.

modverif handles this automatically by setting `cutoff=30` for SH projections in `_pyproj_to_cartopy`. If you create your own cartopy axes for a SH Lambert Conformal domain, remember to pass an appropriate cutoff:

```python
import cartopy.crs as ccrs

proj = ccrs.LambertConformal(
    central_longitude=178, central_latitude=-34,
    standard_parallels=[-41, -41],
    cutoff=30,  # default -30 clips SH domains at 30°S
)
```

### Antimeridian (180° longitude)

Domains that cross the 180° meridian (e.g., New Zealand) require careful handling of longitude conventions:

- **ERA5 / NCAR archive data** typically uses 0--360° longitude convention, where values near the antimeridian are stored as 180--190° rather than -180° to -170°.
- **pyproj** returns longitudes in -180/180° convention when transforming projected coordinates (e.g., Lambert Conformal) to geographic coordinates.
- This mismatch means that a naive regridding interpolation will produce NaN for grid points near the antimeridian, because -175° falls outside an ERA5 source range of [144, 190].

The fix is to convert target longitudes to 0--360° before interpolation:

```python
from scipy.interpolate import RegularGridInterpolator

# Transform WRF projected coords to lat/lon
wrf_lons, wrf_lats = transformer.transform(xx, yy)

# Convert to 0-360 to match ERA5 convention
wrf_lons_360 = wrf_lons % 360

# Interpolate with explicit convention match
interp = RegularGridInterpolator(
    (era5_lats, era5_lons), data,
    method='linear', bounds_error=False, fill_value=np.nan,
)
target_points = np.column_stack([wrf_lats.ravel(), wrf_lons_360.ravel()])
regridded = interp(target_points).reshape(ny, nx)
```

Note that cfdb's `GridInterp.to_grid()` transforms target coordinates internally via pyproj (returning -180/180°), so it cannot resolve the convention mismatch on its own. Use `scipy.interpolate.RegularGridInterpolator` directly for antimeridian-crossing domains.

Longitude values > 180° in WRF `latitude`/`longitude` auxiliary variables are also masked out in modverif's composite plots to prevent cartopy from wrapping the map around the full globe. This only applies to datasets with geographic (lat/lon) coordinates, not projected (y/x) grids.

### Comparison panel spacing

Side-by-side comparison plots auto-compute the figure size and subplot spacing from the domain's aspect ratio. Cartopy GeoAxes maintain the map's native aspect ratio regardless of how much space the subplot allocates, so different domain shapes need different spacing to avoid gaps or overlaps. If you need manual control, pass an explicit `figsize` to the comparison functions.
