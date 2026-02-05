"""
Tests for coordinate and projection data in model_eval.evaluate.
"""
import pathlib
from datetime import date

import h5py
import numpy as np
import pytest

from model_eval.evaluate import evaluate_models


def create_mock_wrfout_with_coords(path, variables, shape, dx=1000.0, dy=1000.0, map_proj=1):
    """Create a mock WRF output file with coordinate attributes and Times variable."""
    with h5py.File(path, 'w') as f:
        f.attrs['DX'] = dx
        f.attrs['DY'] = dy
        f.attrs['MAP_PROJ'] = map_proj
        f.attrs['TRUELAT1'] = 30.0
        f.attrs['TRUELAT2'] = 60.0
        f.attrs['STAND_LON'] = -100.0
        f.attrs['MOAD_CEN_LAT'] = 45.0
        
        # Times dataset: (Time, 19)
        times = [b"2020-09-30_00:00:00", b"2020-09-30_01:00:00"]
        times_data = []
        for t in times:
            # Pad with null bytes or spaces, usually spaces for WRF
            t_padded = t.ljust(19)
            # Create list of single-char bytes: [b'2', b'0', b'2', b'0', ...]
            row = [bytes([c]) for c in t_padded]
            times_data.append(row)
        
        times_arr = np.array(times_data, dtype='S1')
        f.create_dataset('Times', data=times_arr)
        
        for var in variables:
            data = np.ones(shape, dtype=np.float32) * 100
            f.create_dataset(var, data=data)


class TestCoordinates:
    def test_coordinates_and_proj4_saved(self, tmp_path):
        source_dir = tmp_path / 'source'
        test_dir = tmp_path / 'test'
        source_dir.mkdir()
        test_dir.mkdir()

        variables = ['T2']
        shape = (2, 5, 5)
        dx, dy = 2000.0, 2000.0

        source_file = source_dir / "wrfout_d04_2020-09-30_00:00:00"
        test_file = test_dir / "wrfout_d04_2020-09-30_00:00:00"
        
        create_mock_wrfout_with_coords(source_file, variables, shape, dx=dx, dy=dy)
        create_mock_wrfout_with_coords(test_file, variables, shape, dx=dx, dy=dy)

        output_path = tmp_path / 'output.nc'
        evaluate_models(source_dir, test_dir, output_path, domain=4, variables=variables)

        with h5py.File(output_path, 'r') as f:
            # Check proj4 attribute
            assert 'proj4' in f.attrs
            proj4_str = f.attrs['proj4'].decode('utf-8')
            assert '+proj=lcc' in proj4_str
            assert f'lat_1=30.0' in proj4_str
            assert f'lon_0=-100.0' in proj4_str

            # Check x and y values
            np.testing.assert_array_equal(f['x'][:], np.arange(5) * dx)
            np.testing.assert_array_equal(f['y'][:], np.arange(5) * dy)

            # Check time values (hours since 1970-01-01)
            assert f['time'].attrs['units'] == b'hours since 1970-01-01'
            
            dt = np.datetime64('2020-09-30T00:00:00')
            expected_hour = (dt - np.datetime64('1970-01-01')) / np.timedelta64(1, 'h')
            assert f['time'][0] == expected_hour
            assert f['time'][1] == expected_hour + 1.0

            # Check Times variable
            assert 'Times' in f
            times_out = f['Times'][:]
            t0 = b"".join(times_out[0]).decode('utf-8')
            assert t0.startswith("2020-09-30_00:00:00")
