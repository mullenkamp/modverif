#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  4 13:43:27 2026

@author: mike
"""
import hdf5tools
import h5py
import pathlib
import cfdb
import rechunkit
import salem
from datetime import date


##################################################
#### Parameters

source_path = pathlib.Path('/home/mike/data/wrf/tests/physics_tests/2020-09-30/d04_SMS-3DTKE')
test_path = pathlib.Path('/home/mike/data/wrf/tests/physics_tests/2020-09-30/d04_SMS-3DTKE_ndown')

export_path = pathlib.Path('/home/mike/data/wrf/model_eval/test1')
source_cfdb_path = export_path.joinpath('source_data.cfdb')
test_cfdb_path = export_path.joinpath('test_data.cfdb')
output_path = export_path.joinpath('results1.nc')

domain = 4

variables = ['Q2', 'T2', 'U10', 'V10', 'PREC_ACC_NC']
region = (-46.5, -45.0, 166.5, 169.5)
start_lat = -36.824549
start_lon = 176.240509

cyc_path = '/home/mike/data/wrf/tests/physics_tests/2023-02-10/d03_SMS-3DTKE/wrfout_d03_2023-02-13_00:00:00.nc'

plot_path = export_path.joinpath('plots')

#################################################
### Tests


source_files = {}
for file in source_path.iterdir():
    if file.is_file():
        file_name = file.name
        if file_name.startswith('wrfout_'):
            _, domain_str, date_str, _ = file_name.split('_')
            file_domain = int(domain_str[1:])
            date1 = date.fromisoformat(date_str)
            if file_domain == domain:
                source_files[date1] = file


test_files = {}
for file in test_path.iterdir():
    if file.is_file():
        file_name = file.name
        if file_name.startswith('wrfout_'):
            _, domain_str, date_str, _ = file_name.split('_')
            file_domain = int(domain_str[1:])
            date1 = date.fromisoformat(date_str)
            if file_domain == domain:
                test_files[date1] = file


dates = list(source_files.keys())
dates.sort()

cfdb.netcdf4_to_cfdb(sfile, source_cfdb_path, include_data_vars=variables)


for var in variables:
    for run_date in dates:
        sfile = source_files[run_date]
        tfile = test_files[run_date]
        h5s = h5py.File(sfile)
        h5t = h5py.File(tfile)
        sdata_var = h5s[var]
        tdata_var = h5t[var]

        target_chunk_shape = (1, tdata_var.shape[1], tdata_var.shape[2])

        schunker = rechunkit.rechunker(sdata_var.__getitem__, sdata_var.shape, sdata_var.dtype, sdata_var.dtype.itemsize, sdata_var.chunks, target_chunk_shape, 2**29)
        tchunker = rechunkit.rechunker(tdata_var.__getitem__, tdata_var.shape, tdata_var.dtype, tdata_var.dtype.itemsize, tdata_var.chunks, target_chunk_shape, 2**29)
        for source_chunk, source_data in schunker:
            for test_chunk, test_data in tchunker:
                ne = (((test_data - source_data)/source_data) * 100).round().astype('int8')



results_path = evaluate_models(source_path, test_path, output_path, domain, variables, region=region)


positions = track_cyclone(
    cyc_path,
    start_lat=start_lat,
    start_lon=start_lon,
    smoothing_sigma=5,
)

png_files = plot_cyclone_track(cyc_path, positions, plot_path)


results_path = evaluate_models_domain(source_path, test_path, output_path, domain, variables, region=region)


















































































