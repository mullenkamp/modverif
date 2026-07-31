"""
Tests for modverif.cyclone.

The module's first coverage. Written alongside the change that added a tracking time window
and projected-grid support, so several of these are regression tests for behaviour that did
not exist -- or actively raised -- before.

Synthetic data throughout: a Gaussian pressure low walked along a prescribed track, so the
correct answer is known by construction rather than by comparison with a previous run.
"""
import matplotlib as mpl
import numpy as np
import pytest

mpl.use('Agg')

import cfdb
import pyproj

from modverif.cyclone import (
    CyclonePosition,
    _find_pressure_minimum,
    compare_cyclone_tracks,
    haversine_distance,
    match_cyclone_positions,
    plot_cyclone_comparison,
    plot_cyclone_timestep,
    read_latlon_2d,
    track_cyclone,
    track_cyclone_multi_file,
)

BASE_PRESSURE = 101000.0
LOW_DEPTH_PA = 4000.0
START_TIME = np.datetime64('2023-02-10T00:00')

# A track that moves one cell diagonally per step -- unambiguous, and slow enough that a
# radius-limited search cannot lose it.
TRACK = ((5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10))


# --- synthetic data helpers -------------------------------------------------------------

def _gaussian_low(ny, nx, cy, cx, sigma=2.0):
    """A 2D pressure field (Pa) whose unique minimum sits exactly at (cy, cx)."""
    yy, xx = np.mgrid[0:ny, 0:nx]
    r2 = (yy - cy) ** 2 + (xx - cx) ** 2
    return (BASE_PRESSURE - LOW_DEPTH_PA * np.exp(-r2 / (2 * sigma**2))).astype('float32')


def _times(n):
    return np.array([START_TIME + np.timedelta64(i, 'h') for i in range(n)])


def _lcc_crs(lon_0=170.0):
    return pyproj.CRS.from_proj4(
        f'+proj=lcc +lat_0=-41 +lon_0={lon_0} +lat_1=-30 +lat_2=-50 +a=6370000 +b=6370000'
    )


def _create_latlon_cfdb(path, track=TRACK, ny=21, nx=21):
    """Grid with 1D latitude/longitude coordinates -- the reanalysis layout."""
    lats = np.linspace(-50, -30, ny).astype('float32')
    lons = np.linspace(160, 180, nx).astype('float32')
    with cfdb.open_dataset(path, 'n', dataset_type='grid') as ds:
        ds.create.coord.time(data=_times(len(track)))
        ds.create.coord.lat(data=lats)
        ds.create.coord.lon(data=lons)
        var = ds.create.data_var.generic('mslp', ('time', 'latitude', 'longitude'), dtype='float32')
        for t, (cy, cx) in enumerate(track):
            var[(t, slice(None), slice(None))] = _gaussian_low(ny, nx, cy, cx)
    return path


def _create_projected_cfdb(path, track=TRACK, ny=21, nx=21, lon_0=170.0, half_span_m=5e5):
    """Grid with y/x coordinates and a CRS -- the WRF layout, which used to raise."""
    y = np.linspace(-half_span_m, half_span_m, ny)
    x = np.linspace(-half_span_m, half_span_m, nx)
    crs = _lcc_crs(lon_0)
    with cfdb.open_dataset(path, 'n', dataset_type='grid') as ds:
        ds.create.coord.time(data=_times(len(track)))
        ds.create.coord.generic('y', data=y, axis='Y')
        ds.create.coord.generic('x', data=x, axis='X')
        var = ds.create.data_var.generic('mslp', ('time', 'y', 'x'), dtype='float32')
        for t, (cy, cx) in enumerate(track):
            var[(t, slice(None), slice(None))] = _gaussian_low(ny, nx, cy, cx)
        ds.create.crs.from_user_input(crs, x_coord='x', y_coord='y')
    return path


def _create_latlon_datavar_cfdb(path, track=TRACK, ny=21, nx=21, with_crs=False):
    """Grid carrying latitude/longitude as 2D *data variables* over y/x coordinates.

    With ``with_crs=True`` the dataset satisfies both the data-variable branch and the
    projected branch at once, which pins the documented branch ordering.
    """
    y = np.arange(ny, dtype='float64') * 5e4
    x = np.arange(nx, dtype='float64') * 5e4
    lat2d, lon2d = np.meshgrid(np.linspace(-50, -30, ny), np.linspace(160, 180, nx), indexing='ij')
    with cfdb.open_dataset(path, 'n', dataset_type='grid') as ds:
        ds.create.coord.time(data=_times(len(track)))
        ds.create.coord.generic('y', data=y, axis='Y')
        ds.create.coord.generic('x', data=x, axis='X')
        for name, data in (('latitude', lat2d), ('longitude', lon2d)):
            v = ds.create.data_var.generic(name, ('y', 'x'), dtype='float32')
            v[(slice(None), slice(None))] = data.astype('float32')
        var = ds.create.data_var.generic('mslp', ('time', 'y', 'x'), dtype='float32')
        for t, (cy, cx) in enumerate(track):
            var[(t, slice(None), slice(None))] = _gaussian_low(ny, nx, cy, cx)
        if with_crs:
            ds.create.crs.from_user_input(_lcc_crs(), x_coord='x', y_coord='y')
    return path


def _cells(positions):
    return [(p.y_index, p.x_index) for p in positions]


# --- read_latlon_2d ---------------------------------------------------------------------

class TestReadLatLon2d:
    def test_1d_coordinates(self, tmp_path):
        path = _create_latlon_cfdb(tmp_path / 'latlon.cfdb')
        with cfdb.open_dataset(path) as ds:
            xlat, xlong = read_latlon_2d(ds)
        assert xlat.shape == xlong.shape == (21, 21)
        # meshgrid orientation: latitude varies down rows, longitude across columns
        assert xlat[0, 0] == pytest.approx(-50)
        assert xlat[-1, 0] == pytest.approx(-30)
        assert xlong[0, 0] == pytest.approx(160)
        assert xlong[0, -1] == pytest.approx(180)

    def test_2d_data_variables(self, tmp_path):
        path = _create_latlon_datavar_cfdb(tmp_path / 'datavar.cfdb')
        with cfdb.open_dataset(path) as ds:
            xlat, xlong = read_latlon_2d(ds)
        assert xlat.shape == (21, 21)
        assert xlat[0, 0] == pytest.approx(-50)
        assert xlong[0, -1] == pytest.approx(180)

    def test_projected_grid_round_trips(self, tmp_path):
        """The regression test for the bug this was written to fix: on 0.2.11 this raised."""
        path = _create_projected_cfdb(tmp_path / 'proj.cfdb')
        with cfdb.open_dataset(path) as ds:
            xlat, xlong = read_latlon_2d(ds)
            y = np.asarray(ds['y'].data)
            x = np.asarray(ds['x'].data)
            crs = ds.crs

        assert xlat.shape == xlong.shape == (21, 21)
        # Transform back and confirm we land on the native coordinates
        inverse = pyproj.Transformer.from_crs('EPSG:4326', crs, always_xy=True)
        xx_back, yy_back = inverse.transform(xlong, xlat)
        expected_x, expected_y = np.meshgrid(x, y)
        assert np.allclose(xx_back, expected_x, atol=1e-3)
        assert np.allclose(yy_back, expected_y, atol=1e-3)

        # Sanity: the domain sits where the projection says it should
        assert -50 < xlat.min() < xlat.max() < -30
        assert 160 < xlong.min() < xlong.max() < 180

    def test_data_variables_take_precedence_over_projection(self, tmp_path):
        """Documented branch order: stored lat/lon wins over deriving it from the CRS."""
        path = _create_latlon_datavar_cfdb(tmp_path / 'both.cfdb', with_crs=True)
        with cfdb.open_dataset(path) as ds:
            xlat, xlong = read_latlon_2d(ds)
        # The stored values span exactly 160..180; a CRS-derived grid would not.
        assert xlong.min() == pytest.approx(160)
        assert xlong.max() == pytest.approx(180)

    def test_no_usable_coordinates_raises(self, tmp_path):
        path = tmp_path / 'bare.cfdb'
        with cfdb.open_dataset(path, 'n', dataset_type='grid') as ds:
            ds.create.coord.time(data=_times(2))
            ds.create.coord.generic('y', data=np.arange(4, dtype='float64'), axis='Y')
            ds.create.coord.generic('x', data=np.arange(5, dtype='float64'), axis='X')
        with cfdb.open_dataset(path) as ds, pytest.raises(ValueError, match="'y'/'x' coordinates with a CRS"):
            read_latlon_2d(ds)


# --- haversine_distance -----------------------------------------------------------------

class TestHaversineDistance:
    def test_one_degree_of_latitude(self):
        # One degree of latitude on a 6371 km sphere: 6371 * pi/180
        assert haversine_distance(-41.0, 174.0, -40.0, 174.0) == pytest.approx(111.19, abs=0.01)

    def test_zero_for_identical_points(self):
        assert haversine_distance(-41.0, 174.0, -41.0, 174.0) == pytest.approx(0.0)

    def test_symmetric(self):
        a = haversine_distance(-41.0, 174.0, -35.0, 178.0)
        b = haversine_distance(-35.0, 178.0, -41.0, 174.0)
        assert a == pytest.approx(b)

    def test_crosses_the_antimeridian(self):
        """179E to 179W is 2 degrees apart, not 358."""
        near = haversine_distance(-41.0, 179.0, -41.0, -179.0)
        assert near == pytest.approx(haversine_distance(-41.0, 170.0, -41.0, 172.0), rel=1e-9)


# --- track_cyclone: the prescribed track ------------------------------------------------

class TestTrackCyclone:
    def test_recovers_track_on_latlon_grid(self, tmp_path):
        path = _create_latlon_cfdb(tmp_path / 'latlon.cfdb')
        positions = track_cyclone(path, search_radius_km=600.0)
        assert len(positions) == len(TRACK)
        assert _cells(positions) == list(TRACK)
        assert [p.time_index for p in positions] == list(range(len(TRACK)))

    def test_recovers_track_on_projected_grid(self, tmp_path):
        """The headline regression test: this call raises ValueError on modverif 0.2.11."""
        path = _create_projected_cfdb(tmp_path / 'proj.cfdb')
        positions = track_cyclone(path, search_radius_km=600.0)
        assert _cells(positions) == list(TRACK)

    def test_central_pressure_matches_the_synthetic_low(self, tmp_path):
        path = _create_latlon_cfdb(tmp_path / 'latlon.cfdb')
        positions = track_cyclone(path, search_radius_km=600.0)
        for p in positions:
            assert p.central_pressure == pytest.approx(BASE_PRESSURE - LOW_DEPTH_PA, abs=1.0)

    def test_smoothing_preserves_the_track(self, tmp_path):
        """The production path uses sigma=2 on float32 fields; the filter casts via float64."""
        path = _create_latlon_cfdb(tmp_path / 'latlon.cfdb')
        positions = track_cyclone(path, search_radius_km=600.0, smoothing_sigma=2.0)
        assert _cells(positions) == list(TRACK)

    def test_antimeridian_straddling_projected_grid(self, tmp_path):
        """A domain centred at 178E spans both sides of 180; distance masks must cope."""
        path = _create_projected_cfdb(tmp_path / 'straddle.cfdb', lon_0=178.0, half_span_m=8e5)
        with cfdb.open_dataset(path) as ds:
            _, xlong = read_latlon_2d(ds)
        assert xlong.min() < -170 and xlong.max() > 170, 'test grid should straddle the antimeridian'
        positions = track_cyclone(path, search_radius_km=600.0)
        assert _cells(positions) == list(TRACK)


# --- track_cyclone: the time window -----------------------------------------------------

class TestTrackCycloneWindow:
    def test_window_selects_the_expected_subset(self, tmp_path):
        path = _create_latlon_cfdb(tmp_path / 'latlon.cfdb')
        positions = track_cyclone(
            path, search_radius_km=600.0,
            start_time=START_TIME + np.timedelta64(2, 'h'),
            end_time=START_TIME + np.timedelta64(4, 'h'),
        )
        assert len(positions) == 3
        assert _cells(positions) == list(TRACK[2:5])

    def test_time_index_stays_absolute(self, tmp_path):
        """plot_cyclone_timestep re-reads the field by time_index, so it must index the
        dataset's own axis -- not the position within the window."""
        path = _create_latlon_cfdb(tmp_path / 'latlon.cfdb')
        positions = track_cyclone(
            path, search_radius_km=600.0,
            start_time=START_TIME + np.timedelta64(3, 'h'),
        )
        assert [p.time_index for p in positions] == [3, 4, 5]
        assert positions[0].time_str == str(START_TIME + np.timedelta64(3, 'h'))

    def test_windowed_equals_unwindowed_over_the_overlap(self, tmp_path):
        """The sharpest net for an off-by-one in the window selection.

        Seeded explicitly and anchored at the first timestep so the two runs share an
        identical search history -- otherwise the comparison would confound window
        selection with a different starting position.
        """
        path = _create_latlon_cfdb(tmp_path / 'latlon.cfdb')
        kwargs = {'start_lat': -40.0, 'start_lon': 165.0, 'search_radius_km': 600.0}
        full = track_cyclone(path, **kwargs)
        windowed = track_cyclone(path, end_time=START_TIME + np.timedelta64(3, 'h'), **kwargs)
        assert len(windowed) == 4
        # strict=False is deliberate here, unlike everywhere else in the package: the windowed
        # track is compared against the *prefix* of the longer full track, so truncation is the
        # intent. The length assertion above is what stops that hiding a short result.
        for w, f in zip(windowed, full, strict=False):
            assert (w.time_index, w.y_index, w.x_index) == (f.time_index, f.y_index, f.x_index)
            assert w.latitude == f.latitude and w.longitude == f.longitude
            assert w.central_pressure == f.central_pressure
            assert w.radius_km == f.radius_km

    def test_open_ended_bounds(self, tmp_path):
        path = _create_latlon_cfdb(tmp_path / 'latlon.cfdb')
        assert len(track_cyclone(path, start_time=START_TIME + np.timedelta64(4, 'h'))) == 2
        assert len(track_cyclone(path, end_time=START_TIME + np.timedelta64(1, 'h'))) == 2
        assert len(track_cyclone(path)) == len(TRACK)

    def test_empty_window_raises_with_coverage(self, tmp_path):
        path = _create_latlon_cfdb(tmp_path / 'latlon.cfdb')
        with pytest.raises(ValueError, match='No timesteps in'):
            track_cyclone(path, start_time=np.datetime64('2024-01-01'))
        try:
            track_cyclone(path, start_time=np.datetime64('2024-01-01'))
        except ValueError as exc:
            assert '2023-02-10' in str(exc), 'the error should quote the actual coverage'

    def test_global_search_happens_on_the_first_in_window_step(self, tmp_path):
        """Without a start position the search is global -- and with a window that must mean
        the first step *inside* the window, not index 0."""
        path = _create_latlon_cfdb(tmp_path / 'latlon.cfdb')
        positions = track_cyclone(
            path, search_radius_km=1.0,  # so small that a wrong seed cannot find the low
            start_time=START_TIME + np.timedelta64(4, 'h'),
        )
        assert positions[0].time_index == 4
        assert (positions[0].y_index, positions[0].x_index) == TRACK[4]


# --- the invariant that justifies deleting the consumer's resolve_start_position ---------

class TestSeedingEquivalence:
    @pytest.mark.parametrize('factory', [_create_latlon_cfdb, _create_projected_cfdb])
    def test_global_search_equals_radius_search_centred_on_its_own_result(self, tmp_path, factory):
        """A radius-limited search centred on a step's global minimum returns that minimum.

        This is the whole justification for a consumer dropping a separate
        "resolve the start position first" step: tracking from t0 with no seed does the
        same thing. Pinned as a library property rather than left as an observation.
        """
        path = factory(tmp_path / 'grid.cfdb')
        from modverif.cyclone import _read_slp_from_cfdb

        with cfdb.open_dataset(path) as ds:
            xlat, xlong = read_latlon_2d(ds)
            slp = _read_slp_from_cfdb(ds, 0, smoothing_sigma=2.0)

        gy, gx, gp = _find_pressure_minimum(slp, xlat, xlong)
        sy, sx, sp = _find_pressure_minimum(
            slp, xlat, xlong,
            search_lat=float(xlat[gy, gx]), search_lon=float(xlong[gy, gx]),
            search_radius_km=400.0,
        )
        assert (sy, sx) == (gy, gx)
        assert sp == gp


# --- documented exclusions and guards ---------------------------------------------------

class TestMultiFileWindowExclusion:
    def test_multi_file_does_not_accept_a_window(self, tmp_path):
        """Pins a deliberate omission.

        track_cyclone_multi_file offsets indices by len(all_positions) while
        plot_cyclone_track_multi_file maps them back by each file's n_times. Those agree
        only while a track yields one position per timestep, so a window must not be
        plumbed through until that pair is redesigned.
        """
        path_a = _create_latlon_cfdb(tmp_path / 'a.cfdb')
        with pytest.raises(TypeError):
            track_cyclone_multi_file([path_a], start_time=START_TIME)

    def test_multi_file_indices_stay_contiguous(self, tmp_path):
        path_a = _create_latlon_cfdb(tmp_path / 'a.cfdb')
        path_b = _create_latlon_cfdb(tmp_path / 'b.cfdb')
        positions = track_cyclone_multi_file([path_a, path_b], search_radius_km=600.0)
        assert [p.time_index for p in positions] == list(range(2 * len(TRACK)))


# --- comparing two tracks ---------------------------------------------------------------

def _synthetic_track(n=6, *, start=START_TIME, step_h=1, lat0=-40.0, lon0=170.0,
                     dlat=0.5, dlon=0.5, min_pa=96000.0, min_at=3):
    """A track whose deepest point, timing and geometry are all known by construction."""
    positions = []
    for i in range(n):
        # V-shaped pressure profile with its unique minimum at index `min_at`
        pressure = min_pa + 500.0 * abs(i - min_at)
        t = start + np.timedelta64(i * step_h, 'h')
        positions.append(CyclonePosition(
            time_index=i, y_index=i, x_index=i,
            latitude=lat0 + dlat * i, longitude=lon0 + dlon * i,
            central_pressure=pressure, radius_km=300.0, time_str=str(t),
        ))
    return positions


class TestMatchCyclonePositions:
    def test_identical_timestamps_match_one_to_one(self):
        a = _synthetic_track()
        b = _synthetic_track()
        pairs = match_cyclone_positions(a, b)
        assert len(pairs) == len(a)
        assert all(sep == pytest.approx(0.0) for _, _, sep in pairs)

    def test_separation_is_the_great_circle_distance(self):
        a = _synthetic_track(lat0=-40.0, lon0=170.0, dlat=0.0, dlon=0.0)
        b = _synthetic_track(lat0=-41.0, lon0=170.0, dlat=0.0, dlon=0.0)
        pairs = match_cyclone_positions(a, b)
        expected = haversine_distance(-40.0, 170.0, -41.0, 170.0)
        assert all(sep == pytest.approx(expected) for _, _, sep in pairs)

    def test_tolerance_boundary(self):
        """A 90-minute offset is inside the default tolerance; 91 minutes is outside."""
        a = _synthetic_track(n=1)
        inside = _synthetic_track(n=1, start=START_TIME + np.timedelta64(90, 'm'))
        outside = _synthetic_track(n=1, start=START_TIME + np.timedelta64(91, 'm'))
        assert len(match_cyclone_positions(a, inside)) == 1
        assert len(match_cyclone_positions(a, outside)) == 0

    def test_offset_cadence_still_pairs(self):
        """3-hourly vs hourly output shares no exact timestamps, which is the point."""
        a = _synthetic_track(n=3, step_h=3)
        b = _synthetic_track(n=9, step_h=1, start=START_TIME + np.timedelta64(20, 'm'))
        pairs = match_cyclone_positions(a, b)
        assert len(pairs) == 3

    def test_empty_input_returns_no_pairs(self):
        assert match_cyclone_positions([], _synthetic_track()) == []
        assert match_cyclone_positions(_synthetic_track(), []) == []


class TestCompareCycloneTracks:
    def test_recovers_an_injected_pressure_bias(self):
        a = _synthetic_track(min_pa=96000.0)
        b = _synthetic_track(min_pa=95000.0)
        _, metrics = compare_cyclone_tracks(a, b)
        assert metrics['a_min_hpa'] == pytest.approx(960.0)
        assert metrics['b_min_hpa'] == pytest.approx(950.0)
        assert metrics['min_slp_bias_hpa'] == pytest.approx(10.0)

    def test_recovers_an_injected_timing_offset(self):
        a = _synthetic_track(min_at=4)
        b = _synthetic_track(min_at=2)
        _, metrics = compare_cyclone_tracks(a, b)
        assert metrics['timing_offset_h'] == pytest.approx(2.0)

    def test_recovers_an_injected_track_separation(self):
        a = _synthetic_track(lat0=-40.0, dlat=0.0, dlon=0.0)
        b = _synthetic_track(lat0=-41.0, dlat=0.0, dlon=0.0)
        _, metrics = compare_cyclone_tracks(a, b)
        expected = haversine_distance(-40.0, 170.0, -41.0, 170.0)
        assert metrics['mean_track_sep_km'] == pytest.approx(expected)
        assert metrics['max_track_sep_km'] == pytest.approx(expected)
        assert metrics['n_matched_timesteps'] == 6

    def test_sign_conventions_are_a_minus_b(self):
        """Positive bias = a is shallower; positive offset = a lags. Both directions."""
        shallow_late = _synthetic_track(min_pa=97000.0, min_at=4)
        deep_early = _synthetic_track(min_pa=95000.0, min_at=1)
        _, forward = compare_cyclone_tracks(shallow_late, deep_early)
        _, reverse = compare_cyclone_tracks(deep_early, shallow_late)
        assert forward['min_slp_bias_hpa'] > 0 and forward['timing_offset_h'] > 0
        assert reverse['min_slp_bias_hpa'] == pytest.approx(-forward['min_slp_bias_hpa'])
        assert reverse['timing_offset_h'] == pytest.approx(-forward['timing_offset_h'])

    def test_no_matched_timesteps_leaves_separation_undefined(self):
        a = _synthetic_track()
        b = _synthetic_track(start=START_TIME + np.timedelta64(30, 'D'))
        _, metrics = compare_cyclone_tracks(a, b)
        assert metrics['n_matched_timesteps'] == 0
        assert metrics['mean_track_sep_km'] is None
        assert metrics['max_track_sep_km'] is None
        # depth and timing are still comparable -- they do not depend on pairing
        assert metrics['min_slp_bias_hpa'] == pytest.approx(0.0)

    def test_step_counts_are_reported_separately(self):
        a = _synthetic_track(n=4)
        b = _synthetic_track(n=9)
        _, metrics = compare_cyclone_tracks(a, b)
        assert (metrics['n_a_steps'], metrics['n_b_steps']) == (4, 9)

    def test_empty_track_raises(self):
        with pytest.raises(ValueError, match='non-empty'):
            compare_cyclone_tracks([], _synthetic_track())

    def test_end_to_end_on_tracked_data(self, tmp_path):
        """Two grids offset by one cell: the comparison should see a real separation."""
        a_path = _create_latlon_cfdb(tmp_path / 'a.cfdb')
        shifted = tuple((cy, cx + 1) for cy, cx in TRACK)
        b_path = _create_latlon_cfdb(tmp_path / 'b.cfdb', track=shifted)
        pos_a = track_cyclone(a_path, search_radius_km=600.0)
        pos_b = track_cyclone(b_path, search_radius_km=600.0)
        pairs, metrics = compare_cyclone_tracks(pos_a, pos_b)
        assert metrics['n_matched_timesteps'] == len(TRACK)
        assert metrics['mean_track_sep_km'] > 0
        assert metrics['min_slp_bias_hpa'] == pytest.approx(0.0, abs=0.1)


class TestPlotCycloneComparison:
    def test_writes_a_figure(self, tmp_path):
        a = _synthetic_track()
        b = _synthetic_track(lat0=-41.0)
        pairs, metrics = compare_cyclone_tracks(a, b)
        out = tmp_path / 'compare.png'
        plot_cyclone_comparison(out, a, b, pairs, metrics, start_position=(-40.0, 170.0),
                                label_a='WRF', label_b='ERA5', title='test')
        assert out.is_file() and out.stat().st_size > 0

    def test_handles_no_matched_timesteps(self, tmp_path):
        a = _synthetic_track()
        b = _synthetic_track(start=START_TIME + np.timedelta64(30, 'D'))
        pairs, metrics = compare_cyclone_tracks(a, b)
        out = tmp_path / 'compare.png'
        plot_cyclone_comparison(out, a, b, pairs, metrics)
        assert out.is_file()


class TestPlotCycloneTimestepGuard:
    def test_projected_grid_raises_not_implemented(self, tmp_path):
        """read_latlon_2d can now derive lat/lon here, which would otherwise let this
        function emit a silently wrong map."""
        path = _create_projected_cfdb(tmp_path / 'proj.cfdb')
        position = CyclonePosition(
            time_index=0, y_index=5, x_index=5, latitude=-41.0, longitude=170.0,
            central_pressure=97000.0, radius_km=300.0,
        )
        with pytest.raises(NotImplementedError, match='projected grids'):
            plot_cyclone_timestep(path, position, tmp_path / 'out.png')

    def test_latlon_grid_still_plots(self, tmp_path):
        path = _create_latlon_cfdb(tmp_path / 'latlon.cfdb')
        positions = track_cyclone(path, search_radius_km=600.0)
        out = tmp_path / 'out.png'
        plot_cyclone_timestep(path, positions[0], out)
        assert out.is_file() and out.stat().st_size > 0
