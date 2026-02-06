"""
I/O classes for reading WRF files and writing NetCDF4 output.
"""
import pathlib
from datetime import datetime
from typing import Union

import h5py
import numpy as np


# NetCDF4 dimension scale attribute names
CLASS = 'CLASS'
NAME = 'NAME'


class WRFFile:
    """
    Reader for WRF output files (NetCDF4/HDF5 format).

    Provides convenient access to common WRF variables and handles
    time-invariant vs time-varying dimensions automatically.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to WRF output file.

    Examples
    --------
    >>> with WRFFile('wrfout_d01_2020-01-01_00:00:00') as wrf:
    ...     print(wrf.n_times)
    ...     slp = wrf.get_slp(0)
    ...     temp = wrf.get_variable('T2', 0)
    """

    def __init__(self, path: Union[str, pathlib.Path]):
        self.path = pathlib.Path(path)
        self._h5file = None
        self._xlat = None
        self._xlong = None
        self._hgt = None
        self._times = None
        self._time_values = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def open(self):
        """Open the WRF file."""
        if not self.path.exists():
            raise FileNotFoundError(f"WRF file not found: {self.path}")
        self._h5file = h5py.File(self.path, 'r')

    def close(self):
        """Close the WRF file."""
        if self._h5file is not None:
            self._h5file.close()
            self._h5file = None

    @property
    def h5(self) -> h5py.File:
        """Access the underlying h5py.File object."""
        if self._h5file is None:
            raise RuntimeError("File not open. Use 'with WRFFile(path) as wrf:' or call open()")
        return self._h5file

    @property
    def n_times(self) -> int:
        """Number of timesteps in the file."""
        return self.h5['PSFC'].shape[0] if 'PSFC' in self.h5 else self.h5['T2'].shape[0]

    @property
    def n_y(self) -> int:
        """Number of grid points in y direction."""
        return self.xlat.shape[0]

    @property
    def n_x(self) -> int:
        """Number of grid points in x direction."""
        return self.xlat.shape[1]

    @property
    def shape(self) -> tuple[int, int, int]:
        """Shape as (n_times, n_y, n_x)."""
        return (self.n_times, self.n_y, self.n_x)

    @property
    def xlat(self) -> np.ndarray:
        """Latitude grid (2D array)."""
        if self._xlat is None:
            xlat_ds = self.h5['XLAT']
            self._xlat = xlat_ds[0, :, :] if xlat_ds.ndim == 3 else xlat_ds[:, :]
        return self._xlat

    @property
    def xlong(self) -> np.ndarray:
        """Longitude grid (2D array)."""
        if self._xlong is None:
            xlong_ds = self.h5['XLONG']
            self._xlong = xlong_ds[0, :, :] if xlong_ds.ndim == 3 else xlong_ds[:, :]
        return self._xlong

    @property
    def hgt(self) -> np.ndarray:
        """Terrain height (2D array)."""
        if self._hgt is None:
            hgt_ds = self.h5['HGT']
            self._hgt = hgt_ds[0, :, :] if hgt_ds.ndim == 3 else hgt_ds[:, :]
        return self._hgt

    @property
    def times(self) -> list[str]:
        """List of time strings from the Times variable."""
        if self._times is None:
            self._times = []
            if 'Times' in self.h5:
                times_data = self.h5['Times'][:]
                for t_row in times_data:
                    if isinstance(t_row, (bytes, str)):
                        t_str = t_row.decode('utf-8') if isinstance(t_row, bytes) else t_row
                    else:
                        t_str = b"".join(t_row).decode('utf-8')
                    self._times.append(t_str)
        return self._times

    @property
    def time_values(self) -> np.ndarray:
        """Time values as hours since 1970-01-01."""
        if self._time_values is None:
            values = []
            for t_str in self.times:
                t_str = t_str.replace('_', 'T')
                try:
                    dt = np.datetime64(t_str)
                    hours = (dt - np.datetime64('1970-01-01')) / np.timedelta64(1, 'h')
                    values.append(hours)
                except ValueError:
                    values.append(np.nan)
            self._time_values = np.array(values, dtype='f8') if values else None
        return self._time_values

    @property
    def dx(self) -> float:
        """Grid spacing in x direction (meters)."""
        return self.h5.attrs.get('DX')

    @property
    def dy(self) -> float:
        """Grid spacing in y direction (meters)."""
        return self.h5.attrs.get('DY')

    @property
    def proj4(self) -> str:
        """PROJ4 string for the WRF projection, or None if unknown."""
        attrs = self.h5.attrs
        map_proj = attrs.get('MAP_PROJ')
        if map_proj is None:
            return None

        r = attrs.get('EARTH_RADIUS', 6370000.0)
        proj_base = f"+a={r} +b={r} +no_defs"

        if map_proj == 1:  # Lambert Conformal
            truelat1 = attrs.get('TRUELAT1')
            truelat2 = attrs.get('TRUELAT2')
            stand_lon = attrs.get('STAND_LON')
            moad_cen_lat = attrs.get('MOAD_CEN_LAT')
            return f"+proj=lcc +lat_1={truelat1} +lat_2={truelat2} +lat_0={moad_cen_lat} +lon_0={stand_lon} {proj_base}"
        elif map_proj == 2:  # Polar Stereographic
            truelat1 = attrs.get('TRUELAT1')
            stand_lon = attrs.get('STAND_LON')
            return f"+proj=stere +lat_ts={truelat1} +lat_0=90 +lon_0={stand_lon} +k=1 +x_0=0 +y_0=0 {proj_base}"
        elif map_proj == 3:  # Mercator
            truelat1 = attrs.get('TRUELAT1')
            stand_lon = attrs.get('STAND_LON')
            return f"+proj=merc +lat_ts={truelat1} +lon_0={stand_lon} +x_0=0 +y_0=0 {proj_base}"
        elif map_proj == 6:  # Cylindrical Equidistant
            stand_lon = attrs.get('STAND_LON')
            moad_cen_lat = attrs.get('MOAD_CEN_LAT')
            return f"+proj=longlat +lon_0={stand_lon} +lat_0={moad_cen_lat} {proj_base}"

        return None

    def has_variable(self, name: str) -> bool:
        """Check if a variable exists in the file."""
        return name in self.h5

    def get_variable(self, name: str, time_index: int = None) -> np.ndarray:
        """
        Get a variable from the file.

        Parameters
        ----------
        name : str
            Variable name.
        time_index : int, optional
            If provided, return data for this timestep only.
            If None, return all timesteps.

        Returns
        -------
        np.ndarray
            Variable data.
        """
        if name not in self.h5:
            raise ValueError(f"Variable '{name}' not found in {self.path}")

        ds = self.h5[name]
        if time_index is not None:
            return ds[time_index, ...]
        return ds[:]

    def get_slp(self, time_index: int, smoothing_sigma: float = None) -> np.ndarray:
        """
        Compute sea level pressure for a timestep.

        Parameters
        ----------
        time_index : int
            Timestep index.
        smoothing_sigma : float, optional
            Gaussian smoothing sigma. If None, no smoothing.

        Returns
        -------
        np.ndarray
            Sea level pressure in Pa (2D array).
        """
        psfc = self.get_variable('PSFC', time_index)
        t2 = self.get_variable('T2', time_index)
        q2 = self.get_variable('Q2', time_index) if self.has_variable('Q2') else None

        slp = self._compute_slp(psfc, self.hgt, t2, q2)

        if smoothing_sigma is not None:
            from scipy.ndimage import gaussian_filter
            slp = gaussian_filter(slp, sigma=smoothing_sigma)

        return slp

    @staticmethod
    def _compute_slp(psfc: np.ndarray, hgt: np.ndarray, t2: np.ndarray, q2: np.ndarray = None) -> np.ndarray:
        """Compute sea level pressure using hypsometric equation."""
        GRAVITY = 9.80665
        GAS_CONSTANT_DRY = 287.05
        STANDARD_LAPSE_RATE = 0.0065

        if q2 is not None:
            t_virtual = t2 * (1.0 + 0.61 * q2)
        else:
            t_virtual = t2

        t_sea_level = t_virtual + STANDARD_LAPSE_RATE * hgt
        t_avg = 0.5 * (t_virtual + t_sea_level)
        slp = psfc * np.exp(GRAVITY * hgt / (GAS_CONSTANT_DRY * t_avg))

        return slp


class NetCDF4Writer:
    """
    Writer for CF-compliant NetCDF4 files using h5py.

    Simplifies creation of NetCDF4 files with proper dimension scales,
    CF-compliant attributes, and consistent compression settings.

    Parameters
    ----------
    path : str or pathlib.Path
        Output file path.
    compression : str
        Compression algorithm ('gzip', 'lzf', or None).
    compression_opts : int
        Compression level (1-9 for gzip).

    Examples
    --------
    >>> with NetCDF4Writer('output.nc') as nc:
    ...     nc.set_global_attrs(source='model_eval')
    ...     time_ds = nc.create_dimension('time', 10)
    ...     var = nc.create_variable('temperature', (10, 50, 50), 'f4')
    ...     nc.attach_scale(var, 0, time_ds)
    """

    def __init__(
        self,
        path: Union[str, pathlib.Path],
        compression: str = 'gzip',
        compression_opts: int = 4,
    ):
        self.path = pathlib.Path(path)
        self.compression = compression
        self.compression_opts = compression_opts
        self._h5file = None
        self._dimensions = {}

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def open(self):
        """Open the file for writing."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._h5file = h5py.File(self.path, 'w')

    def close(self):
        """Close the file."""
        if self._h5file is not None:
            self._h5file.close()
            self._h5file = None

    @property
    def h5(self) -> h5py.File:
        """Access the underlying h5py.File object."""
        if self._h5file is None:
            raise RuntimeError("File not open")
        return self._h5file

    def set_global_attrs(
        self,
        conventions: str = 'CF-1.8',
        history: str = None,
        **kwargs,
    ):
        """
        Set global attributes.

        Parameters
        ----------
        conventions : str
            CF conventions version.
        history : str, optional
            History attribute. If None, auto-generated.
        **kwargs
            Additional attributes.
        """
        self.h5.attrs['Conventions'] = np.bytes_(conventions)
        if history is None:
            history = f'Created {datetime.now().isoformat()} by model_eval'
        self.h5.attrs['history'] = np.bytes_(history)

        for key, value in kwargs.items():
            if isinstance(value, str):
                self.h5.attrs[key] = np.bytes_(value)
            else:
                self.h5.attrs[key] = value

    def create_dimension(
        self,
        name: str,
        size: int,
        data: np.ndarray = None,
        dtype: str = 'f4',
        units: str = None,
        long_name: str = None,
        standard_name: str = None,
        **attrs,
    ) -> h5py.Dataset:
        """
        Create a NetCDF4-compliant dimension scale.

        Parameters
        ----------
        name : str
            Dimension name.
        size : int
            Dimension size.
        data : np.ndarray, optional
            Coordinate data. If None, uses np.arange(size).
        dtype : str
            Data type for coordinate values.
        units : str, optional
            Units attribute.
        long_name : str, optional
            Long name attribute.
        standard_name : str, optional
            CF standard name.
        **attrs
            Additional attributes.

        Returns
        -------
        h5py.Dataset
            The dimension scale dataset.
        """
        if data is not None:
            dim_ds = self.h5.create_dataset(name, data=data)
        else:
            dim_ds = self.h5.create_dataset(name, data=np.arange(size, dtype=dtype))

        # Mark as dimension scale
        dim_ds.attrs[CLASS] = np.bytes_('DIMENSION_SCALE')
        dim_ds.attrs[NAME] = np.bytes_(name)

        if units:
            dim_ds.attrs['units'] = np.bytes_(units)
        if long_name:
            dim_ds.attrs['long_name'] = np.bytes_(long_name)
        if standard_name:
            dim_ds.attrs['standard_name'] = np.bytes_(standard_name)

        for key, value in attrs.items():
            if isinstance(value, str):
                dim_ds.attrs[key] = np.bytes_(value)
            else:
                dim_ds.attrs[key] = value

        self._dimensions[name] = dim_ds
        return dim_ds

    def create_time_dimension(
        self,
        size: int,
        data: np.ndarray = None,
        units: str = 'hours since 1970-01-01',
        calendar: str = 'proleptic_gregorian',
    ) -> h5py.Dataset:
        """
        Create a time dimension with CF-compliant attributes.

        Parameters
        ----------
        size : int
            Number of timesteps.
        data : np.ndarray, optional
            Time coordinate values.
        units : str
            CF time units.
        calendar : str
            CF calendar type.

        Returns
        -------
        h5py.Dataset
            The time dimension dataset.
        """
        return self.create_dimension(
            'time',
            size,
            data=data,
            units=units,
            standard_name='time',
            calendar=calendar,
        )

    def create_metric_dimension(
        self,
        metrics: list[str],
    ) -> h5py.Dataset:
        """
        Create a metric dimension with flag_meanings attribute.

        Parameters
        ----------
        metrics : list[str]
            List of metric names.

        Returns
        -------
        h5py.Dataset
            The metric dimension dataset.
        """
        n_metrics = len(metrics)
        metric_ds = self.create_dimension(
            'metric',
            n_metrics,
            data=np.arange(n_metrics, dtype=np.int32),
            long_name='Evaluation metric index',
            flag_values=np.arange(n_metrics, dtype=np.int32),
            flag_meanings=' '.join(metrics),
        )
        return metric_ds

    def create_spatial_dimensions(
        self,
        n_y: int,
        n_x: int,
        y_data: np.ndarray = None,
        x_data: np.ndarray = None,
    ) -> tuple[h5py.Dataset, h5py.Dataset]:
        """
        Create y and x dimensions for spatial data.

        Parameters
        ----------
        n_y : int
            Number of y grid points.
        n_x : int
            Number of x grid points.
        y_data : np.ndarray, optional
            Y coordinate values.
        x_data : np.ndarray, optional
            X coordinate values.

        Returns
        -------
        tuple[h5py.Dataset, h5py.Dataset]
            (y_ds, x_ds) dimension datasets.
        """
        y_ds = self.create_dimension(
            'y', n_y, data=y_data,
            standard_name='projection_y_coordinate',
            units='m',
        )
        x_ds = self.create_dimension(
            'x', n_x, data=x_data,
            standard_name='projection_x_coordinate',
            units='m',
        )
        return y_ds, x_ds

    def create_variable(
        self,
        name: str,
        shape: tuple,
        dtype: str = 'f4',
        data: np.ndarray = None,
        units: str = None,
        long_name: str = None,
        standard_name: str = None,
        fill_value=None,
        chunks: tuple = None,
        compress: bool = True,
        **attrs,
    ) -> h5py.Dataset:
        """
        Create a variable with optional compression.

        Parameters
        ----------
        name : str
            Variable name.
        shape : tuple
            Variable shape.
        dtype : str
            Data type.
        data : np.ndarray, optional
            Initial data. If provided, shape is inferred.
        units : str, optional
            Units attribute.
        long_name : str, optional
            Long name attribute.
        standard_name : str, optional
            CF standard name.
        fill_value : optional
            Fill value for missing data.
        chunks : tuple, optional
            Chunk shape. If None, auto-determined.
        compress : bool
            Whether to apply compression.
        **attrs
            Additional attributes.

        Returns
        -------
        h5py.Dataset
            The created dataset.
        """
        kwargs = {'dtype': dtype}

        if data is not None:
            kwargs['data'] = data
            shape = data.shape
        else:
            kwargs['shape'] = shape

        if compress and self.compression:
            kwargs['compression'] = self.compression
            kwargs['compression_opts'] = self.compression_opts

        if chunks is not None:
            kwargs['chunks'] = chunks
        elif compress and len(shape) >= 3:
            # Default chunking: 1 timestep at a time
            kwargs['chunks'] = (1,) + shape[1:]

        if fill_value is not None:
            kwargs['fillvalue'] = fill_value

        ds = self.h5.create_dataset(name, **kwargs)

        if units:
            ds.attrs['units'] = np.bytes_(units)
        if long_name:
            ds.attrs['long_name'] = np.bytes_(long_name)
        if standard_name:
            ds.attrs['standard_name'] = np.bytes_(standard_name)
        if fill_value is not None:
            ds.attrs['_FillValue'] = fill_value

        for key, value in attrs.items():
            if isinstance(value, str):
                ds.attrs[key] = np.bytes_(value)
            else:
                ds.attrs[key] = value

        return ds

    def attach_scales(
        self,
        dataset: h5py.Dataset,
        dimensions: list[h5py.Dataset],
    ):
        """
        Attach dimension scales to a dataset.

        Parameters
        ----------
        dataset : h5py.Dataset
            The dataset to attach scales to.
        dimensions : list[h5py.Dataset]
            List of dimension scale datasets, one per dimension.
        """
        for i, dim_ds in enumerate(dimensions):
            dim_name = dim_ds.name.split('/')[-1]
            dim_ds.make_scale(dim_name)
            dataset.dims[i].attach_scale(dim_ds)

    def get_dimension(self, name: str) -> h5py.Dataset:
        """Get a dimension by name."""
        if name in self._dimensions:
            return self._dimensions[name]
        if name in self.h5:
            return self.h5[name]
        raise KeyError(f"Dimension '{name}' not found")
