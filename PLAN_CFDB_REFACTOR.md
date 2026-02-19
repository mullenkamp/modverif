# Plan: Refactor model_eval to use cfdb as primary input data class

## Context

model_eval is currently tightly coupled to WRF output files (NetCDF4/HDF5). The `WRFEvaluator` discovers wrfout files by filename pattern, reads them directly via h5py, and uses WRF-native variable names (T2, Q2, U10, V10, PSFC, etc.). This refactor makes the package model-agnostic by using `cfdb.Dataset` as the standard input format with CF-compliant variable names. Output will also use cfdb format. Users convert source data (e.g. WRF) to cfdb separately using `cfdb-ingest` before evaluation (clean separation of concerns).

---

## Variables model_eval requires that cfdb DOES NOT have predefined templates for

These need to be added to cfdb (`../cfdb/cfdb/utils.py` and `../cfdb/cfdb/creation.py`):

1. **`mixing_ratio`** — WRF's Q2 is water vapor mixing ratio (kg/kg), distinct from cfdb's `specific_humidity`. CF standard name: `humidity_mixing_ratio`. Needs new template with dtype float32, uint16 encoded, precision 6.
2. **`terrain_height`** — WRF's HGT is a 2D spatial data variable (y, x), not a z-axis coordinate. cfdb has `altitude` as a z-coordinate, but terrain height is a field. CF standard name: `surface_altitude`. Needs new template with dtype float32, uint16 encoded, precision 1.
3. **`potential_temperature`** — Derived from T2 and PSFC. CF standard name: `air_potential_temperature`. Needs new template, dtype float32, uint16 encoded, precision 2.
4. **`equivalent_potential_temperature`** — Derived from T2, Q2, PSFC (Bolton 1980). CF standard name: `equivalent_potential_temperature`. Needs new template, dtype float32, uint16 encoded, precision 2.

### cfdb-ingest status

The `cfdb-ingest` package (`../cfdb-ingest`) handles WRF-to-cfdb conversion. Q2 (and QVAPOR) are already correctly mapped to `mixing_ratio`. cfdb-ingest does not currently have mappings for `terrain_height`, `relative_humidity`, `dew_point_temperature`, `potential_temperature`, or `equivalent_potential_temperature` — these can be added to cfdb-ingest as needed.

---

## Variable Mapping: WRF -> cfdb standard names

### Direct data variable mappings

| WRF Variable | cfdb shortcut | cfdb stored name | Units |
|---|---|---|---|
| `T2` | `air_temp` | `air_temperature` | K |
| `Q2` | `mixing_ratio` | `mixing_ratio` | kg/kg |
| `U10` | `u_wind` | `u_wind` | m/s |
| `V10` | `v_wind` | `v_wind` | m/s |
| `PSFC` | `surface_pressure` | `surface_pressure` | Pa |
| `RAINNC` / `PREC_ACC_NC` | `precip` | `precipitation` | mm |
| `HGT` | `terrain_height` | `terrain_height` | m |

### Derived variable mappings

| WRFFile method | cfdb shortcut | cfdb stored name |
|---|---|---|
| `get_wind_speed()` | `wind_speed` | `wind_speed` |
| `get_wind_direction()` | `wind_direction` | `wind_direction` |
| `get_rh()` | `relative_humidity` | `relative_humidity` |
| `get_dewpoint()` | `dew_temp` | `dew_point_temperature` |
| `get_slp()` | `mslp` | `mslp` |
| `get_theta()` | `potential_temperature` | `potential_temperature` |
| `get_theta_e()` | `equivalent_potential_temperature` | `equivalent_potential_temperature` |

### Coordinate mappings

| WRF | cfdb shortcut | cfdb stored name |
|---|---|---|
| `XLAT` | `lat` | `latitude` |
| `XLONG` | `lon` | `longitude` |
| `Times` | `time` | `time` |
| projected x | `x` | `x` |
| projected y | `y` | `y` |

---

## Implementation Steps

### Step 1: Add missing variables to cfdb

**Files:** `../cfdb/cfdb/utils.py`, `../cfdb/cfdb/creation.py`

Add to `default_dtype_params`, `default_var_params`, and `default_attrs` in `utils.py`:

```python
# default_dtype_params additions:
'mixing_ratio': {'precision': 6, 'name': 'float32', 'offset': -0.000001, 'dtype_encoded': 'uint16', 'fillvalue': 0},
'terrain_height': {'precision': 1, 'name': 'float32', 'offset': -1, 'dtype_encoded': 'uint16', 'fillvalue': 0},
'potential_temperature': {'precision': 2, 'name': 'float32', 'offset': -61, 'dtype_encoded': 'uint16', 'fillvalue': 0},
'equivalent_potential_temperature': {'precision': 2, 'name': 'float32', 'offset': -61, 'dtype_encoded': 'uint16', 'fillvalue': 0},

# default_var_params additions:
'mixing_ratio': {'name': 'mixing_ratio'},
'terrain_height': {'name': 'terrain_height'},
'potential_temperature': {'name': 'potential_temperature'},
'equivalent_potential_temperature': {'name': 'equivalent_potential_temperature'},

# default_attrs additions:
mixing_ratio={
    'long_name': 'humidity mixing ratio',
    'units': 'kg kg-1',
    'standard_name': 'humidity_mixing_ratio',
},
terrain_height={
    'long_name': 'terrain height above sea level',
    'units': 'm',
    'standard_name': 'surface_altitude',
},
potential_temperature={
    'long_name': 'air potential temperature',
    'units': 'K',
    'standard_name': 'air_potential_temperature',
},
equivalent_potential_temperature={
    'long_name': 'equivalent potential temperature',
    'units': 'K',
    'standard_name': 'equivalent_potential_temperature',
},
```

Add these four names to the `@create_data_var_methods` decorator var_names tuple in `creation.py` (line 288).

### Step 2: Refactor Evaluator (`model_eval/evaluator.py`)

**Rename `WRFEvaluator` -> `Evaluator`**

New interface:

```python
class Evaluator:
    def __init__(
        self,
        source: str | Path,          # path to cfdb file
        test: str | Path,            # path to cfdb file
        region: tuple | np.ndarray | None = None,   # lat/lon bounds or mask
        start_time: str | np.datetime64 | None = None,
        end_time: str | np.datetime64 | None = None,
    ):
        ...

    def evaluate_cell(
        self,
        output_path: str | Path,
        variables: list[str],         # cfdb standard names
        metrics: str | list[str] = 'ne',
        threshold: float = None,
        epsilon: float = 1e-10,
    ) -> Path

    def evaluate_domain(
        self,
        output_path: str | Path,
        variables: list[str],
        metrics: str | list[str] = 'ne',
        threshold: float = None,
        epsilon: float = 1e-10,
    ) -> Path
```

Key changes from current `WRFEvaluator`:
- Opens cfdb datasets with `cfdb.open_dataset()` instead of h5py
- Reads data via `ds[var_name][sel]` or `ds[var_name].iter_chunks()` instead of rechunkit on h5py
- Grid info (dx, dy) from coordinate step values or differences
- CRS from `ds.crs`
- Spatial subsetting via cfdb `select_loc()` or coordinate indexing
- Time subsetting via time coordinate selection
- Remove `find_wrfout_files()`, `_get_wrf_proj4()`, `_find_latlon_bounds()` (WRF-specific, handled by cfdb-ingest)
- **Output** written to cfdb dataset using cfdb API (not NetCDF4Writer)

### Step 3: Refactor output to cfdb format

Replace `NetCDF4Writer` usage in the evaluator with cfdb output:
- Create output cfdb grid dataset
- Time coordinate from input
- For cell evaluation: y, x coordinates + data variables per `{var}_{metric}`
- For domain evaluation: time + metric dimension + data variables per `{var}`
- Set CRS and attributes from source dataset

Keep `NetCDF4Writer` in `wrfio.py` for other uses, but the evaluator writes cfdb.

### Step 4: Refactor cyclone module (`model_eval/cyclone.py`)

- `track_cyclone()` accepts cfdb file path instead of WRF file path
- Reads `mslp` directly from cfdb dataset (pre-computed during ingestion) instead of computing SLP on the fly
- Uses cfdb variable names (`latitude`/`longitude`, etc.)
- Falls back to computing SLP from `surface_pressure`, `terrain_height`, `air_temperature` if `mslp` not present
- Keep SLP computation as a static utility for fallback

### Step 5: Refactor functional wrappers (`model_eval/evaluate.py`)

- `evaluate_models_cell()` and `evaluate_models_domain()` take cfdb paths
- `evaluate_cyclones()` takes cfdb paths
- Remove WRF-specific parameters (domain)
- Parameters use cfdb variable names

### Step 6: Update exports (`model_eval/__init__.py`)

```python
from model_eval.evaluator import Evaluator
```

### Step 7: Update tests

- Create helper to build mock cfdb datasets (replacing `create_mock_wrfout()`)
- Rewrite `test_evaluate.py` to use cfdb mock data with cfdb variable names
- Keep `test_wrfio.py` (WRFFile still exists, used by cfdb-ingest)
- Keep `test_metrics.py` (pure numpy, no changes)
- Update `conftest.py` fixtures

### Step 8: Update pyproject.toml

- Add `cfdb` as a dependency
- Update any relevant metadata

---

## Files Summary

| File | Action |
|---|---|
| `../cfdb/cfdb/utils.py` | Add 4 new variable templates |
| `../cfdb/cfdb/creation.py` | Add 4 new names to `@create_data_var_methods` |
| `model_eval/__init__.py` | Update exports: `Evaluator` |
| `model_eval/evaluator.py` | Major refactor: rename class, cfdb input/output |
| `model_eval/evaluate.py` | Update wrappers for cfdb inputs |
| `model_eval/wrfio.py` | Keep as-is (used by cfdb-ingest) |
| `model_eval/cyclone.py` | Refactor to accept cfdb datasets |
| `model_eval/metrics.py` | No changes (pure numpy) |
| `model_eval/tests/test_evaluate.py` | Rewrite with cfdb mock data |
| `model_eval/tests/conftest.py` | Update fixtures |
| `pyproject.toml` | Add cfdb dependency |

### Separate packages to update (outside model_eval)

| File | Action |
|---|---|
| `../cfdb-ingest/cfdb_ingest/wrf.py` | Q2/QVAPOR -> `mixing_ratio` already done; optionally add `terrain_height`, `relative_humidity`, `dew_point_temperature`, `potential_temperature`, `equivalent_potential_temperature` mappings |

---

## Verification

1. **cfdb additions**: Run cfdb's test suite to confirm new variable templates work
2. **cfdb-ingest**: Convert a sample WRF file with updated mappings, inspect cfdb output
3. **Evaluator**: Run evaluation on converted cfdb data, compare numerical results to old WRF-based path
4. **Full test suite**: `pytest model_eval/tests/`
5. **Round-trip check**: WRF -> cfdb (via cfdb-ingest) -> evaluate -> verify output cfdb has correct structure
