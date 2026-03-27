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
| `figsize` | `(14, 10)` | Figure size (comparison default: `(24, 10)`) |
| `dpi` | 150 | Output resolution |
