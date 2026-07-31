"""
Tests for modverif.composite module.

Unit tests for time alignment and smoke tests for composite plotting functions.
"""
import matplotlib as mpl
import numpy as np
import pytest

mpl.use('Agg')
import cfdb
import matplotlib.pyplot as plt

from modverif.composite import (
    _align_times,
    _draw_composite_layers,
    _plot_storm_composite_comparison_frame,
    plot_storm_composite_comparison,
    plot_storm_composite_comparison_timestep,
)

# --- Synthetic data helpers ---

def _make_grid(ny=10, nx=12):
    """Create synthetic lat/lon 2D arrays."""
    lats = np.linspace(-45, -40, ny)
    lons = np.linspace(170, 175, nx)
    x2d, y2d = np.meshgrid(lons, lats)
    return x2d, y2d


def _make_fields(ny=10, nx=12):
    """Create synthetic PWAT, MSLP, VIMF fields."""
    pwat = np.random.rand(ny, nx).astype(np.float32) * 60 + 10
    mslp_hpa = np.random.rand(ny, nx).astype(np.float32) * 40 + 980
    vimf_u = np.random.rand(ny, nx).astype(np.float32) * 200 - 100
    vimf_v = np.random.rand(ny, nx).astype(np.float32) * 200 - 100
    return vimf_u, vimf_v, pwat, mslp_hpa


def _create_composite_cfdb(path, n_times=4, ny=10, nx=12, start_time=None):
    """Create a mock cfdb dataset with composite-compatible variables."""
    if start_time is None:
        start_time = np.datetime64('2020-01-01T00:00')
    times = np.array([start_time + np.timedelta64(i, 'h') for i in range(n_times)])
    lats = np.linspace(-45, -40, ny, dtype='float32')
    lons = np.linspace(170, 175, nx, dtype='float32')

    with cfdb.open_dataset(path, 'n', dataset_type='grid') as ds:
        ds.create.coord.time(data=times)
        ds.create.coord.lat(data=lats)
        ds.create.coord.lon(data=lons)

        for var_name in ('vimf_u', 'vimf_v', 'pwat', 'mslp'):
            var = ds.create.data_var.generic(var_name, ('time', 'latitude', 'longitude'), dtype='float32')
            for t in range(n_times):
                if var_name == 'mslp':
                    data = np.random.rand(ny, nx).astype(np.float32) * 4000 + 98000  # Pa
                elif var_name == 'pwat':
                    data = np.random.rand(ny, nx).astype(np.float32) * 60 + 10
                else:
                    data = np.random.rand(ny, nx).astype(np.float32) * 200 - 100
                var[(t, slice(None), slice(None))] = data

    return path


# --- _align_times tests ---

class TestAlignTimes:
    def test_exact_match(self):
        times_a = np.array([np.datetime64('2020-01-01') + np.timedelta64(i, 'h') for i in range(6)])
        times_b = times_a.copy()
        matched = _align_times(times_a, times_b)
        assert len(matched) == 6
        assert all(t == times_a[i] for t, i, _ in matched)

    def test_partial_overlap(self):
        times_a = np.array([np.datetime64('2020-01-01') + np.timedelta64(i, 'h') for i in range(6)])
        times_b = np.array([np.datetime64('2020-01-01') + np.timedelta64(i, 'h') for i in range(3, 9)])
        matched = _align_times(times_a, times_b)
        assert len(matched) == 3
        # Common times are hours 3, 4, 5
        assert matched[0][1] == 3  # index in A
        assert matched[0][2] == 0  # index in B

    def test_no_overlap(self):
        times_a = np.array([np.datetime64('2020-01-01') + np.timedelta64(i, 'h') for i in range(3)])
        times_b = np.array([np.datetime64('2020-01-02') + np.timedelta64(i, 'h') for i in range(3)])
        matched = _align_times(times_a, times_b)
        assert len(matched) == 0

    def test_different_frequency(self):
        # A is hourly, B is 3-hourly
        times_a = np.array([np.datetime64('2020-01-01') + np.timedelta64(i, 'h') for i in range(12)])
        times_b = np.array([np.datetime64('2020-01-01') + np.timedelta64(i * 3, 'h') for i in range(4)])
        matched = _align_times(times_a, times_b)
        assert len(matched) == 4
        assert matched[0][1] == 0  # hour 0 in A
        assert matched[1][1] == 3  # hour 3 in A
        assert matched[2][1] == 6  # hour 6 in A
        assert matched[3][1] == 9  # hour 9 in A

    def test_start_time_filter(self):
        times_a = np.array([np.datetime64('2020-01-01') + np.timedelta64(i, 'h') for i in range(6)])
        times_b = times_a.copy()
        matched = _align_times(times_a, times_b, start_time=np.datetime64('2020-01-01T03:00'))
        assert len(matched) == 3

    def test_end_time_filter(self):
        times_a = np.array([np.datetime64('2020-01-01') + np.timedelta64(i, 'h') for i in range(6)])
        times_b = times_a.copy()
        matched = _align_times(times_a, times_b, end_time=np.datetime64('2020-01-01T02:00'))
        assert len(matched) == 3

    def test_start_and_end_time_filter(self):
        times_a = np.array([np.datetime64('2020-01-01') + np.timedelta64(i, 'h') for i in range(12)])
        times_b = times_a.copy()
        matched = _align_times(
            times_a, times_b,
            start_time=np.datetime64('2020-01-01T03:00'),
            end_time=np.datetime64('2020-01-01T06:00'),
        )
        assert len(matched) == 4

    def test_sorted_output(self):
        times_a = np.array([np.datetime64('2020-01-01') + np.timedelta64(i, 'h') for i in range(6)])
        times_b = times_a[::-1].copy()  # reversed
        matched = _align_times(times_a, times_b)
        time_vals = [t for t, _, _ in matched]
        assert time_vals == sorted(time_vals)


# --- _draw_composite_layers tests ---

class TestDrawCompositeLayers:
    def test_returns_contourf_mappable(self):
        x2d, y2d = _make_grid()
        vimf_u, vimf_v, pwat, mslp_hpa = _make_fields()
        fig, ax = plt.subplots()
        cf = _draw_composite_layers(ax, x2d, y2d, vimf_u, vimf_v, pwat, mslp_hpa, {})
        assert cf is not None
        plt.close(fig)

    def test_with_quiver(self):
        x2d, y2d = _make_grid()
        vimf_u, vimf_v, pwat, mslp_hpa = _make_fields()
        fig, ax = plt.subplots()
        cf = _draw_composite_layers(ax, x2d, y2d, vimf_u, vimf_v, pwat, mslp_hpa, {}, vector_type='quiver')
        assert cf is not None
        plt.close(fig)

    def test_with_explicit_levels(self):
        x2d, y2d = _make_grid()
        vimf_u, vimf_v, pwat, mslp_hpa = _make_fields()
        fig, ax = plt.subplots()
        cf = _draw_composite_layers(
            ax, x2d, y2d, vimf_u, vimf_v, pwat, mslp_hpa, {},
            pwat_levels=np.linspace(0, 70, 15),
            mslp_levels=list(range(970, 1030, 4)),
        )
        assert cf is not None
        plt.close(fig)

    def test_custom_gridline_labels(self):
        x2d, y2d = _make_grid()
        vimf_u, vimf_v, pwat, mslp_hpa = _make_fields()
        fig, ax = plt.subplots()
        cf = _draw_composite_layers(
            ax, x2d, y2d, vimf_u, vimf_v, pwat, mslp_hpa, {},
            gridline_labels={'top': False, 'right': False, 'bottom': True, 'left': False},
        )
        assert cf is not None
        plt.close(fig)


# --- _plot_storm_composite_comparison_frame tests ---

class TestComparisonFrame:
    def test_creates_two_axes(self):
        x2d_a, y2d_a = _make_grid()
        x2d_b, y2d_b = _make_grid(ny=8, nx=10)
        vimf_u_a, vimf_v_a, pwat_a, mslp_a = _make_fields()
        vimf_u_b, vimf_v_b, pwat_b, mslp_b = _make_fields(ny=8, nx=10)

        fig, (ax_a, ax_b) = _plot_storm_composite_comparison_frame(
            x2d_a, y2d_a, vimf_u_a, vimf_v_a, pwat_a, mslp_a,
            label_a='WRF',
            x2d_b=x2d_b, y2d_b=y2d_b,
            vimf_u_b=vimf_u_b, vimf_v_b=vimf_v_b, pwat_b=pwat_b, mslp_b=mslp_b,
            label_b='ERA5',
        )
        assert fig is not None
        assert ax_a is not None
        assert ax_b is not None
        assert ax_a.get_title() == 'WRF'
        assert ax_b.get_title() == 'ERA5'
        plt.close(fig)

    def test_with_timestamp(self):
        x2d, y2d = _make_grid()
        fields = _make_fields()
        fig, (ax_a, ax_b) = _plot_storm_composite_comparison_frame(
            x2d, y2d, *fields,
            x2d_b=x2d, y2d_b=y2d,
            vimf_u_b=fields[0], vimf_v_b=fields[1], pwat_b=fields[2], mslp_b=fields[3],
            time_str='2020-01-01T00:00',
        )
        assert '2020-01-01' in fig._suptitle.get_text()
        plt.close(fig)

    def test_saves_file(self, tmp_path):
        x2d, y2d = _make_grid()
        fields = _make_fields()
        save_path = tmp_path / 'comparison.png'
        result = _plot_storm_composite_comparison_frame(
            x2d, y2d, *fields,
            x2d_b=x2d, y2d_b=y2d,
            vimf_u_b=fields[0], vimf_v_b=fields[1], pwat_b=fields[2], mslp_b=fields[3],
            output_path=save_path,
        )
        assert result is None
        assert save_path.exists()

    def test_mslp_pa_conversion(self):
        """Both panels should handle Pa-to-hPa conversion independently."""
        x2d, y2d = _make_grid()
        vimf_u, vimf_v, pwat, _ = _make_fields()
        mslp_pa = np.random.rand(10, 12).astype(np.float32) * 4000 + 98000  # Pa
        mslp_hpa = np.random.rand(10, 12).astype(np.float32) * 40 + 980  # hPa

        fig, (ax_a, ax_b) = _plot_storm_composite_comparison_frame(
            x2d, y2d, vimf_u, vimf_v, pwat, mslp_pa,
            x2d_b=x2d, y2d_b=y2d,
            vimf_u_b=vimf_u, vimf_v_b=vimf_v, pwat_b=pwat, mslp_b=mslp_hpa,
        )
        assert fig is not None
        plt.close(fig)


# --- Integration tests with mock cfdb datasets ---

class TestComparisonTimestep:
    def test_creates_comparison(self, tmp_path):
        path_a = _create_composite_cfdb(tmp_path / 'a.cfdb')
        path_b = _create_composite_cfdb(tmp_path / 'b.cfdb')

        fig, (ax_a, ax_b) = plot_storm_composite_comparison_timestep(
            path_a, path_b, time_index=0,
            label_a='Model A', label_b='Model B',
        )
        assert fig is not None
        assert ax_a.get_title() == 'Model A'
        assert ax_b.get_title() == 'Model B'
        plt.close(fig)

    def test_saves_file(self, tmp_path):
        path_a = _create_composite_cfdb(tmp_path / 'a.cfdb')
        path_b = _create_composite_cfdb(tmp_path / 'b.cfdb')
        save_path = tmp_path / 'out.png'

        result = plot_storm_composite_comparison_timestep(
            path_a, path_b, time_index=1, output_path=save_path,
        )
        assert result is None
        assert save_path.exists()

    def test_time_index_out_of_range(self, tmp_path):
        path_a = _create_composite_cfdb(tmp_path / 'a.cfdb', n_times=4)
        path_b = _create_composite_cfdb(tmp_path / 'b.cfdb', n_times=4)

        with pytest.raises(IndexError):
            plot_storm_composite_comparison_timestep(path_a, path_b, time_index=10)

    def test_no_common_timesteps(self, tmp_path):
        path_a = _create_composite_cfdb(
            tmp_path / 'a.cfdb', n_times=3,
            start_time=np.datetime64('2020-01-01'),
        )
        path_b = _create_composite_cfdb(
            tmp_path / 'b.cfdb', n_times=3,
            start_time=np.datetime64('2020-06-01'),
        )
        with pytest.raises(ValueError, match="No common timesteps"):
            plot_storm_composite_comparison_timestep(path_a, path_b, time_index=0)

    def test_partial_overlap(self, tmp_path):
        # A: hours 0-3, B: hours 2-5 -> overlap at hours 2, 3
        path_a = _create_composite_cfdb(
            tmp_path / 'a.cfdb', n_times=4,
            start_time=np.datetime64('2020-01-01T00:00'),
        )
        path_b = _create_composite_cfdb(
            tmp_path / 'b.cfdb', n_times=4,
            start_time=np.datetime64('2020-01-01T02:00'),
        )
        fig, _ = plot_storm_composite_comparison_timestep(path_a, path_b, time_index=0)
        assert fig is not None
        plt.close(fig)

        # Only 2 matched timesteps, so index 2 should fail
        with pytest.raises(IndexError):
            plot_storm_composite_comparison_timestep(path_a, path_b, time_index=2)

    def test_missing_dataset(self, tmp_path):
        path_a = _create_composite_cfdb(tmp_path / 'a.cfdb')
        with pytest.raises(FileNotFoundError):
            plot_storm_composite_comparison_timestep(path_a, tmp_path / 'nonexistent.cfdb', time_index=0)


class TestComparisonBatch:
    def test_generates_frames_and_webp(self, tmp_path):
        path_a = _create_composite_cfdb(tmp_path / 'a.cfdb', n_times=3)
        path_b = _create_composite_cfdb(tmp_path / 'b.cfdb', n_times=3)
        output_dir = tmp_path / 'frames'

        png_files, webp_path = plot_storm_composite_comparison(
            path_a, path_b, output_dir,
            label_a='WRF', label_b='ERA5',
        )
        assert len(png_files) == 3
        assert all(p.exists() for p in png_files)
        assert webp_path.exists()

    def test_time_range_filter(self, tmp_path):
        path_a = _create_composite_cfdb(tmp_path / 'a.cfdb', n_times=6)
        path_b = _create_composite_cfdb(tmp_path / 'b.cfdb', n_times=6)
        output_dir = tmp_path / 'frames'

        png_files, _ = plot_storm_composite_comparison(
            path_a, path_b, output_dir,
            start_time='2020-01-01T02:00',
            end_time='2020-01-01T04:00',
        )
        assert len(png_files) == 3  # hours 2, 3, 4

    def test_custom_webp_path(self, tmp_path):
        path_a = _create_composite_cfdb(tmp_path / 'a.cfdb', n_times=2)
        path_b = _create_composite_cfdb(tmp_path / 'b.cfdb', n_times=2)
        output_dir = tmp_path / 'frames'
        webp_out = tmp_path / 'custom.webp'

        _, webp_path = plot_storm_composite_comparison(
            path_a, path_b, output_dir, webp_path=webp_out,
        )
        assert webp_path == webp_out
        assert webp_path.exists()

    def test_no_common_timesteps_raises(self, tmp_path):
        path_a = _create_composite_cfdb(
            tmp_path / 'a.cfdb', n_times=2,
            start_time=np.datetime64('2020-01-01'),
        )
        path_b = _create_composite_cfdb(
            tmp_path / 'b.cfdb', n_times=2,
            start_time=np.datetime64('2020-06-01'),
        )
        with pytest.raises(ValueError, match="No common timesteps"):
            plot_storm_composite_comparison(path_a, path_b, tmp_path / 'out')
