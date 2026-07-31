"""
Tests for the modverif.io module.
"""

import h5py
import numpy as np
import pytest

from modverif.wrfio import NetCDF4Writer, WRFFile


class TestWRFFile:
    """Tests for the WRFFile reader class."""

    @pytest.fixture
    def sample_wrf_file(self, tmp_path):
        """Create a minimal WRF-like HDF5 file for testing."""
        filepath = tmp_path / "wrfout_test.nc"
        n_times = 3
        n_y = 10
        n_x = 15

        with h5py.File(filepath, 'w') as f:
            # Global attributes
            f.attrs['DX'] = 1000.0
            f.attrs['DY'] = 1000.0
            f.attrs['MAP_PROJ'] = 1  # Lambert Conformal
            f.attrs['TRUELAT1'] = -30.0
            f.attrs['TRUELAT2'] = -60.0
            f.attrs['STAND_LON'] = 175.0
            f.attrs['MOAD_CEN_LAT'] = -45.0

            # Create lat/lon grids (3D with time dimension)
            lats = np.linspace(-46, -44, n_y)
            lons = np.linspace(168, 172, n_x)
            lon_grid, lat_grid = np.meshgrid(lons, lats)
            xlat_data = np.broadcast_to(lat_grid, (n_times, n_y, n_x)).copy()
            xlong_data = np.broadcast_to(lon_grid, (n_times, n_y, n_x)).copy()
            f.create_dataset('XLAT', data=xlat_data)
            f.create_dataset('XLONG', data=xlong_data)

            # Terrain height (3D)
            hgt_data = np.random.uniform(0, 500, (n_times, n_y, n_x)).astype(np.float32)
            f.create_dataset('HGT', data=hgt_data)

            # Surface pressure
            psfc_data = np.random.uniform(95000, 102000, (n_times, n_y, n_x)).astype(np.float32)
            f.create_dataset('PSFC', data=psfc_data)

            # 2-meter temperature
            t2_data = np.random.uniform(280, 300, (n_times, n_y, n_x)).astype(np.float32)
            f.create_dataset('T2', data=t2_data)

            # 2-meter mixing ratio (optional)
            q2_data = np.random.uniform(0.005, 0.015, (n_times, n_y, n_x)).astype(np.float32)
            f.create_dataset('Q2', data=q2_data)

            # 10-meter wind components
            u10_data = np.random.uniform(-10, 10, (n_times, n_y, n_x)).astype(np.float32)
            v10_data = np.random.uniform(-10, 10, (n_times, n_y, n_x)).astype(np.float32)
            f.create_dataset('U10', data=u10_data)
            f.create_dataset('V10', data=v10_data)

            # Times variable - WRF stores as 2D char array (n_times, str_len)
            times = ['2020-01-01_00:00:00', '2020-01-01_01:00:00', '2020-01-01_02:00:00']
            max_len = max(len(t) for t in times)
            times_data = np.array([[c.encode() for c in t.ljust(max_len)] for t in times], dtype='S1')
            f.create_dataset('Times', data=times_data)

        return filepath

    def test_context_manager(self, sample_wrf_file):
        """Test that WRFFile works as a context manager."""
        with WRFFile(sample_wrf_file) as wrf:
            assert wrf.n_times == 3

    def test_file_not_found(self, tmp_path):
        """Test that FileNotFoundError is raised for missing files."""
        with pytest.raises(FileNotFoundError):
            with WRFFile(tmp_path / "nonexistent.nc"):
                pass

    def test_n_times(self, sample_wrf_file):
        """Test n_times property."""
        with WRFFile(sample_wrf_file) as wrf:
            assert wrf.n_times == 3

    def test_n_y_n_x(self, sample_wrf_file):
        """Test n_y and n_x properties."""
        with WRFFile(sample_wrf_file) as wrf:
            assert wrf.n_y == 10
            assert wrf.n_x == 15

    def test_shape(self, sample_wrf_file):
        """Test shape property."""
        with WRFFile(sample_wrf_file) as wrf:
            assert wrf.shape == (3, 10, 15)

    def test_xlat_xlong(self, sample_wrf_file):
        """Test xlat and xlong properties return 2D arrays."""
        with WRFFile(sample_wrf_file) as wrf:
            xlat = wrf.xlat
            xlong = wrf.xlong
            assert xlat.ndim == 2
            assert xlong.ndim == 2
            assert xlat.shape == (10, 15)
            assert xlong.shape == (10, 15)

    def test_hgt(self, sample_wrf_file):
        """Test hgt property returns 2D array."""
        with WRFFile(sample_wrf_file) as wrf:
            hgt = wrf.hgt
            assert hgt.ndim == 2
            assert hgt.shape == (10, 15)

    def test_times(self, sample_wrf_file):
        """Test times property returns list of strings."""
        with WRFFile(sample_wrf_file) as wrf:
            times = wrf.times
            assert len(times) == 3
            assert times[0] == '2020-01-01_00:00:00'

    def test_time_values(self, sample_wrf_file):
        """Test time_values property returns hours since epoch."""
        with WRFFile(sample_wrf_file) as wrf:
            time_values = wrf.time_values
            assert len(time_values) == 3
            # Check values are hours since 1970-01-01
            assert time_values[1] - time_values[0] == pytest.approx(1.0)  # 1 hour apart

    def test_dx_dy(self, sample_wrf_file):
        """Test dx and dy properties."""
        with WRFFile(sample_wrf_file) as wrf:
            assert wrf.dx == 1000.0
            assert wrf.dy == 1000.0

    def test_proj4(self, sample_wrf_file):
        """Test proj4 property for Lambert Conformal."""
        with WRFFile(sample_wrf_file) as wrf:
            proj4 = wrf.proj4
            assert '+proj=lcc' in proj4
            assert '+lat_1=-30.0' in proj4
            assert '+lat_2=-60.0' in proj4

    def test_has_variable(self, sample_wrf_file):
        """Test has_variable method."""
        with WRFFile(sample_wrf_file) as wrf:
            assert wrf.has_variable('T2') is True
            assert wrf.has_variable('NONEXISTENT') is False

    def test_get_variable_all_times(self, sample_wrf_file):
        """Test get_variable without time index returns all timesteps."""
        with WRFFile(sample_wrf_file) as wrf:
            t2 = wrf.get_variable('T2')
            assert t2.shape == (3, 10, 15)

    def test_get_variable_single_time(self, sample_wrf_file):
        """Test get_variable with time index returns single timestep."""
        with WRFFile(sample_wrf_file) as wrf:
            t2 = wrf.get_variable('T2', time_index=1)
            assert t2.shape == (10, 15)

    def test_get_variable_not_found(self, sample_wrf_file):
        """Test get_variable raises ValueError for missing variable."""
        with WRFFile(sample_wrf_file) as wrf:
            with pytest.raises(ValueError, match="not found"):
                wrf.get_variable('NONEXISTENT')

    def test_get_slp(self, sample_wrf_file):
        """Test get_slp computes sea level pressure."""
        with WRFFile(sample_wrf_file) as wrf:
            slp = wrf.get_slp(0)
            assert slp.shape == (10, 15)
            # SLP should be higher than surface pressure due to elevation
            psfc = wrf.get_variable('PSFC', 0)
            assert np.all(slp >= psfc)

    def test_get_slp_with_smoothing(self, sample_wrf_file):
        """Test get_slp with Gaussian smoothing."""
        with WRFFile(sample_wrf_file) as wrf:
            slp_raw = wrf.get_slp(0)
            slp_smooth = wrf.get_slp(0, smoothing_sigma=2.0)
            assert slp_smooth.shape == slp_raw.shape
            # Smoothed data should have smaller variance
            assert np.std(slp_smooth) < np.std(slp_raw)

    def test_caching(self, sample_wrf_file):
        """Test that xlat, xlong, hgt are cached."""
        with WRFFile(sample_wrf_file) as wrf:
            xlat1 = wrf.xlat
            xlat2 = wrf.xlat
            # Should be the same object (cached)
            assert xlat1 is xlat2

    # --- Diagnostic variable tests ---

    def test_get_wind_speed_single_time(self, sample_wrf_file):
        """Test get_wind_speed for a single timestep."""
        with WRFFile(sample_wrf_file) as wrf:
            ws = wrf.get_wind_speed(0)
            assert ws.shape == (10, 15)
            # Wind speed should be non-negative
            assert np.all(ws >= 0)

    def test_get_wind_speed_all_times(self, sample_wrf_file):
        """Test get_wind_speed for all timesteps."""
        with WRFFile(sample_wrf_file) as wrf:
            ws = wrf.get_wind_speed()
            assert ws.shape == (3, 10, 15)
            assert np.all(ws >= 0)

    def test_get_wind_speed_calculation(self, tmp_path):
        """Test wind speed calculation with known values."""
        filepath = tmp_path / "wind_test.nc"
        with h5py.File(filepath, 'w') as f:
            # U=3, V=4 should give speed=5
            f.create_dataset('U10', data=np.full((1, 2, 2), 3.0, dtype=np.float32))
            f.create_dataset('V10', data=np.full((1, 2, 2), 4.0, dtype=np.float32))
            f.create_dataset('PSFC', data=np.full((1, 2, 2), 101325.0, dtype=np.float32))

        with WRFFile(filepath) as wrf:
            ws = wrf.get_wind_speed(0)
            np.testing.assert_array_almost_equal(ws, 5.0)

    def test_get_wind_direction_single_time(self, sample_wrf_file):
        """Test get_wind_direction for a single timestep."""
        with WRFFile(sample_wrf_file) as wrf:
            wd = wrf.get_wind_direction(0)
            assert wd.shape == (10, 15)
            # Wind direction should be in [0, 360)
            assert np.all(wd >= 0)
            assert np.all(wd < 360)

    def test_get_wind_direction_all_times(self, sample_wrf_file):
        """Test get_wind_direction for all timesteps."""
        with WRFFile(sample_wrf_file) as wrf:
            wd = wrf.get_wind_direction()
            assert wd.shape == (3, 10, 15)
            assert np.all(wd >= 0)
            assert np.all(wd < 360)

    def test_get_wind_direction_calculation(self, tmp_path):
        """Test wind direction calculation with known values."""
        filepath = tmp_path / "wind_dir_test.nc"
        with h5py.File(filepath, 'w') as f:
            # Wind FROM north (V negative = wind going south = from north)
            f.create_dataset('U10', data=np.array([[[0.0]]], dtype=np.float32))
            f.create_dataset('V10', data=np.array([[[-5.0]]], dtype=np.float32))
            f.create_dataset('PSFC', data=np.full((1, 1, 1), 101325.0, dtype=np.float32))

        with WRFFile(filepath) as wrf:
            wd = wrf.get_wind_direction(0)
            # Wind from north should be ~0 or 360 degrees
            assert wd[0, 0] == pytest.approx(0.0, abs=1.0) or wd[0, 0] == pytest.approx(360.0, abs=1.0)

    def test_get_wind_direction_from_east(self, tmp_path):
        """Test wind direction for wind from east."""
        filepath = tmp_path / "wind_dir_east.nc"
        with h5py.File(filepath, 'w') as f:
            # Wind FROM east (U negative = wind going west = from east)
            f.create_dataset('U10', data=np.array([[[-5.0]]], dtype=np.float32))
            f.create_dataset('V10', data=np.array([[[0.0]]], dtype=np.float32))
            f.create_dataset('PSFC', data=np.full((1, 1, 1), 101325.0, dtype=np.float32))

        with WRFFile(filepath) as wrf:
            wd = wrf.get_wind_direction(0)
            # Wind from east should be ~90 degrees
            assert wd[0, 0] == pytest.approx(90.0, abs=1.0)

    def test_get_rh_single_time(self, sample_wrf_file):
        """Test get_rh for a single timestep."""
        with WRFFile(sample_wrf_file) as wrf:
            rh = wrf.get_rh(0)
            assert rh.shape == (10, 15)
            # RH should be clipped to [0, 100]
            assert np.all(rh >= 0)
            assert np.all(rh <= 100)

    def test_get_rh_all_times(self, sample_wrf_file):
        """Test get_rh for all timesteps."""
        with WRFFile(sample_wrf_file) as wrf:
            rh = wrf.get_rh()
            assert rh.shape == (3, 10, 15)
            assert np.all(rh >= 0)
            assert np.all(rh <= 100)

    def test_get_rh_calculation(self, tmp_path):
        """Test RH calculation with known values."""
        filepath = tmp_path / "rh_test.nc"
        with h5py.File(filepath, 'w') as f:
            # At 20C (293.15K), saturation vapor pressure is ~2339 Pa
            # With mixing ratio giving vapor pressure ~1170 Pa, RH should be ~50%
            t2 = np.array([[[293.15]]], dtype=np.float32)
            psfc = np.array([[[101325.0]]], dtype=np.float32)
            # q = 0.622 * e / (p - e), solving for q when e = 1170 Pa
            # q = 0.622 * 1170 / (101325 - 1170) ≈ 0.00726
            q2 = np.array([[[0.00726]]], dtype=np.float32)
            f.create_dataset('T2', data=t2)
            f.create_dataset('Q2', data=q2)
            f.create_dataset('PSFC', data=psfc)

        with WRFFile(filepath) as wrf:
            rh = wrf.get_rh(0)
            # Should be approximately 50%
            assert rh[0, 0] == pytest.approx(50.0, rel=0.1)

    def test_get_dewpoint_single_time(self, sample_wrf_file):
        """Test get_dewpoint for a single timestep."""
        with WRFFile(sample_wrf_file) as wrf:
            td = wrf.get_dewpoint(0)
            assert td.shape == (10, 15)
            # Dewpoint should be in a reasonable range (200K to 320K)
            assert np.all(td > 200)
            assert np.all(td < 320)

    def test_get_dewpoint_all_times(self, sample_wrf_file):
        """Test get_dewpoint for all timesteps."""
        with WRFFile(sample_wrf_file) as wrf:
            td = wrf.get_dewpoint()
            assert td.shape == (3, 10, 15)

    def test_get_dewpoint_calculation(self, tmp_path):
        """Test dewpoint calculation - at 100% RH, dewpoint equals temperature."""
        filepath = tmp_path / "td_test.nc"
        with h5py.File(filepath, 'w') as f:
            t2 = np.array([[[293.15]]], dtype=np.float32)  # 20C
            psfc = np.array([[[101325.0]]], dtype=np.float32)
            # Saturation mixing ratio at 20C, ~0.0147 kg/kg
            # At saturation, e_s ≈ 2339 Pa, q_s = 0.622 * 2339 / (101325 - 2339) ≈ 0.0147
            q2 = np.array([[[0.0147]]], dtype=np.float32)
            f.create_dataset('T2', data=t2)
            f.create_dataset('Q2', data=q2)
            f.create_dataset('PSFC', data=psfc)

        with WRFFile(filepath) as wrf:
            td = wrf.get_dewpoint(0)
            # At saturation, dewpoint should equal temperature
            assert td[0, 0] == pytest.approx(293.15, rel=0.02)

    def test_get_theta_single_time(self, sample_wrf_file):
        """Test get_theta for a single timestep."""
        with WRFFile(sample_wrf_file) as wrf:
            theta = wrf.get_theta(0)
            assert theta.shape == (10, 15)
            # Potential temperature should generally be >= actual temperature
            # (for surface pressure <= 1000 hPa)
            t2 = wrf.get_variable('T2', 0)
            psfc = wrf.get_variable('PSFC', 0)
            # Where psfc < P0 (100000 Pa), theta > T
            mask = psfc < 100000.0
            if np.any(mask):
                assert np.all(theta[mask] >= t2[mask] - 0.1)

    def test_get_theta_all_times(self, sample_wrf_file):
        """Test get_theta for all timesteps."""
        with WRFFile(sample_wrf_file) as wrf:
            theta = wrf.get_theta()
            assert theta.shape == (3, 10, 15)

    def test_get_theta_calculation(self, tmp_path):
        """Test potential temperature calculation with known values."""
        filepath = tmp_path / "theta_test.nc"
        with h5py.File(filepath, 'w') as f:
            # At P0 (1000 hPa), theta = T
            t2 = np.array([[[300.0]]], dtype=np.float32)
            psfc = np.array([[[100000.0]]], dtype=np.float32)  # P0
            f.create_dataset('T2', data=t2)
            f.create_dataset('PSFC', data=psfc)

        with WRFFile(filepath) as wrf:
            theta = wrf.get_theta(0)
            # At reference pressure, theta = T
            assert theta[0, 0] == pytest.approx(300.0, rel=0.001)

    def test_get_theta_e_single_time(self, sample_wrf_file):
        """Test get_theta_e for a single timestep."""
        with WRFFile(sample_wrf_file) as wrf:
            theta_e = wrf.get_theta_e(0)
            assert theta_e.shape == (10, 15)
            # Equivalent potential temperature should be >= potential temperature
            theta = wrf.get_theta(0)
            assert np.all(theta_e >= theta - 1.0)  # Small tolerance

    def test_get_theta_e_all_times(self, sample_wrf_file):
        """Test get_theta_e for all timesteps."""
        with WRFFile(sample_wrf_file) as wrf:
            theta_e = wrf.get_theta_e()
            assert theta_e.shape == (3, 10, 15)

    def test_get_theta_e_dry_air(self, tmp_path):
        """Test theta_e approaches theta for very dry air."""
        filepath = tmp_path / "theta_e_dry.nc"
        with h5py.File(filepath, 'w') as f:
            t2 = np.array([[[300.0]]], dtype=np.float32)
            psfc = np.array([[[100000.0]]], dtype=np.float32)
            # Very small mixing ratio (nearly dry)
            q2 = np.array([[[0.0001]]], dtype=np.float32)
            f.create_dataset('T2', data=t2)
            f.create_dataset('PSFC', data=psfc)
            f.create_dataset('Q2', data=q2)

        with WRFFile(filepath) as wrf:
            theta = wrf.get_theta(0)
            theta_e = wrf.get_theta_e(0)
            # For very dry air, theta_e should be close to theta
            assert theta_e[0, 0] == pytest.approx(theta[0, 0], rel=0.05)


class TestNetCDF4Writer:
    """Tests for the NetCDF4Writer class."""

    def test_context_manager(self, tmp_path):
        """Test that NetCDF4Writer works as a context manager."""
        filepath = tmp_path / "test.nc"
        with NetCDF4Writer(filepath) as nc:
            nc.set_global_attrs()
        assert filepath.exists()

    def test_creates_parent_directory(self, tmp_path):
        """Test that parent directories are created."""
        filepath = tmp_path / "subdir" / "nested" / "test.nc"
        with NetCDF4Writer(filepath) as nc:
            nc.set_global_attrs()
        assert filepath.exists()

    def test_set_global_attrs(self, tmp_path):
        """Test setting global attributes."""
        filepath = tmp_path / "test.nc"
        with NetCDF4Writer(filepath) as nc:
            nc.set_global_attrs(source='test', custom_attr='value')

        with h5py.File(filepath, 'r') as f:
            assert f.attrs['Conventions'] == b'CF-1.8'
            assert b'modverif' in f.attrs['history']
            assert f.attrs['source'] == b'test'
            assert f.attrs['custom_attr'] == b'value'

    def test_create_dimension(self, tmp_path):
        """Test creating a dimension scale."""
        filepath = tmp_path / "test.nc"
        with NetCDF4Writer(filepath) as nc:
            dim_ds = nc.create_dimension('x', 10, units='m', long_name='X coordinate')
            assert dim_ds.shape == (10,)

        with h5py.File(filepath, 'r') as f:
            assert 'x' in f
            assert f['x'].attrs['CLASS'] == b'DIMENSION_SCALE'
            assert f['x'].attrs['units'] == b'm'
            assert f['x'].attrs['long_name'] == b'X coordinate'

    def test_create_dimension_with_data(self, tmp_path):
        """Test creating a dimension scale with explicit data."""
        filepath = tmp_path / "test.nc"
        data = np.array([0.5, 1.5, 2.5, 3.5])
        with NetCDF4Writer(filepath) as nc:
            nc.create_dimension('x', 4, data=data)

        with h5py.File(filepath, 'r') as f:
            np.testing.assert_array_equal(f['x'][:], data)

    def test_create_time_dimension(self, tmp_path):
        """Test creating a CF-compliant time dimension."""
        filepath = tmp_path / "test.nc"
        time_data = np.array([0.0, 1.0, 2.0])
        with NetCDF4Writer(filepath) as nc:
            nc.create_time_dimension(3, data=time_data)

        with h5py.File(filepath, 'r') as f:
            assert f['time'].attrs['units'] == b'hours since 1970-01-01'
            assert f['time'].attrs['calendar'] == b'proleptic_gregorian'
            assert f['time'].attrs['standard_name'] == b'time'

    def test_create_metric_dimension(self, tmp_path):
        """Test creating a metric dimension with flag_meanings."""
        filepath = tmp_path / "test.nc"
        metrics = ['ne', 'ane', 'rmse']
        with NetCDF4Writer(filepath) as nc:
            nc.create_metric_dimension(metrics)

        with h5py.File(filepath, 'r') as f:
            assert f['metric'].attrs['flag_meanings'] == b'ne ane rmse'
            np.testing.assert_array_equal(f['metric'].attrs['flag_values'], [0, 1, 2])

    def test_create_spatial_dimensions(self, tmp_path):
        """Test creating y and x spatial dimensions."""
        filepath = tmp_path / "test.nc"
        with NetCDF4Writer(filepath) as nc:
            y_ds, x_ds = nc.create_spatial_dimensions(20, 30)

        with h5py.File(filepath, 'r') as f:
            assert f['y'].shape == (20,)
            assert f['x'].shape == (30,)
            assert f['y'].attrs['standard_name'] == b'projection_y_coordinate'
            assert f['x'].attrs['standard_name'] == b'projection_x_coordinate'

    def test_create_variable(self, tmp_path):
        """Test creating a variable with attributes."""
        filepath = tmp_path / "test.nc"
        with NetCDF4Writer(filepath) as nc:
            nc.create_variable(
                'temperature',
                shape=(10, 20, 30),
                dtype='f4',
                units='K',
                long_name='Air Temperature',
                standard_name='air_temperature',
            )

        with h5py.File(filepath, 'r') as f:
            assert f['temperature'].shape == (10, 20, 30)
            assert f['temperature'].dtype == np.float32
            assert f['temperature'].attrs['units'] == b'K'
            assert f['temperature'].attrs['long_name'] == b'Air Temperature'

    def test_create_variable_with_data(self, tmp_path):
        """Test creating a variable with initial data."""
        filepath = tmp_path / "test.nc"
        data = np.random.rand(5, 10).astype(np.float32)
        with NetCDF4Writer(filepath) as nc:
            nc.create_variable('data', shape=data.shape, data=data)

        with h5py.File(filepath, 'r') as f:
            np.testing.assert_array_equal(f['data'][:], data)

    def test_create_variable_with_compression(self, tmp_path):
        """Test that compression is applied by default."""
        filepath = tmp_path / "test.nc"
        with NetCDF4Writer(filepath) as nc:
            nc.create_variable('data', shape=(10, 20, 30), dtype='f4')

        with h5py.File(filepath, 'r') as f:
            assert f['data'].compression == 'gzip'

    def test_create_variable_without_compression(self, tmp_path):
        """Test creating a variable without compression."""
        filepath = tmp_path / "test.nc"
        with NetCDF4Writer(filepath) as nc:
            nc.create_variable('data', shape=(10,), dtype='f4', compress=False)

        with h5py.File(filepath, 'r') as f:
            assert f['data'].compression is None

    def test_create_variable_with_fill_value(self, tmp_path):
        """Test creating a variable with fill value."""
        filepath = tmp_path / "test.nc"
        with NetCDF4Writer(filepath) as nc:
            nc.create_variable(
                'data', shape=(10,), dtype='f4', fill_value=np.float32(-999.0)
            )

        with h5py.File(filepath, 'r') as f:
            assert f['data'].attrs['_FillValue'] == -999.0

    def test_attach_scales(self, tmp_path):
        """Test attaching dimension scales to a variable."""
        filepath = tmp_path / "test.nc"
        with NetCDF4Writer(filepath) as nc:
            time_ds = nc.create_dimension('time', 5)
            y_ds = nc.create_dimension('y', 10)
            x_ds = nc.create_dimension('x', 15)
            var_ds = nc.create_variable('data', shape=(5, 10, 15), dtype='f4')
            nc.attach_scales(var_ds, [time_ds, y_ds, x_ds])

        # Verify scales are attached by checking with h5py
        with h5py.File(filepath, 'r') as f:
            # The DIMENSION_LIST attribute should exist
            assert len(f['data'].dims) == 3

    def test_get_dimension(self, tmp_path):
        """Test retrieving a dimension by name."""
        filepath = tmp_path / "test.nc"
        with NetCDF4Writer(filepath) as nc:
            time_ds = nc.create_dimension('time', 5)
            retrieved = nc.get_dimension('time')
            assert retrieved is time_ds

    def test_get_dimension_not_found(self, tmp_path):
        """Test that KeyError is raised for missing dimension."""
        filepath = tmp_path / "test.nc"
        with NetCDF4Writer(filepath) as nc:
            with pytest.raises(KeyError):
                nc.get_dimension('nonexistent')

    def test_h5_property(self, tmp_path):
        """Test accessing the underlying h5py.File."""
        filepath = tmp_path / "test.nc"
        with NetCDF4Writer(filepath) as nc:
            assert isinstance(nc.h5, h5py.File)
            # Should be able to use h5py directly
            nc.h5.create_dataset('direct', data=[1, 2, 3])

        with h5py.File(filepath, 'r') as f:
            assert 'direct' in f


class TestWRFFileWithoutOptionalVariables:
    """Test WRFFile behavior with minimal WRF files."""

    @pytest.fixture
    def minimal_wrf_file(self, tmp_path):
        """Create a WRF file without Q2 or Times."""
        filepath = tmp_path / "wrfout_minimal.nc"
        n_times = 2
        n_y = 5
        n_x = 5

        with h5py.File(filepath, 'w') as f:
            # Minimal required variables
            f.create_dataset('XLAT', data=np.zeros((n_times, n_y, n_x)))
            f.create_dataset('XLONG', data=np.zeros((n_times, n_y, n_x)))
            f.create_dataset('HGT', data=np.zeros((n_y, n_x)))  # 2D version
            f.create_dataset('PSFC', data=np.full((n_times, n_y, n_x), 101325.0))
            f.create_dataset('T2', data=np.full((n_times, n_y, n_x), 288.0))

        return filepath

    def test_hgt_2d_handling(self, minimal_wrf_file):
        """Test that 2D HGT is handled correctly."""
        with WRFFile(minimal_wrf_file) as wrf:
            hgt = wrf.hgt
            assert hgt.shape == (5, 5)

    def test_slp_without_q2(self, minimal_wrf_file):
        """Test SLP calculation without Q2 variable."""
        with WRFFile(minimal_wrf_file) as wrf:
            slp = wrf.get_slp(0)
            assert slp.shape == (5, 5)

    def test_times_empty_when_missing(self, minimal_wrf_file):
        """Test that times returns empty list when Times variable is missing."""
        with WRFFile(minimal_wrf_file) as wrf:
            times = wrf.times
            assert times == []

    def test_time_values_none_when_missing(self, minimal_wrf_file):
        """Test that time_values returns None when Times is missing."""
        with WRFFile(minimal_wrf_file) as wrf:
            time_values = wrf.time_values
            assert time_values is None
