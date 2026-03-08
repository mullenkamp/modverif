#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  4 13:43:27 2026

@author: mike
"""
import sys
# import h5py
import pathlib
# import cfdb
# import rechunkit
# import salem
from datetime import date

module_path = '/home/mike/git/modverif'
if module_path not in sys.path:
    sys.path.append(module_path)

import modverif
from modverif.evaluator import Evaluator

##################################################
#### Parameters

source_dir = pathlib.Path('/home/mike/data/wrf/tests/nudge_tests/test_d03_no_nudge')
test_dir = pathlib.Path('/home/mike/data/wrf/tests/nudge_tests/test_d03_yes_nudge')

export_path = pathlib.Path('/home/mike/data/wrf/tests/nudge_tests/test_d03_eval')
source_cfdb_path = export_path.joinpath('source_data.cfdb')
test_cfdb_path = export_path.joinpath('test_data.cfdb')
output_path = export_path.joinpath('results1.nc')

domain = 1
region = None
start_date = None
end_date = None

variables = ['Q2', 'T2', 'U10', 'V10', 'PREC_ACC_NC']
region = (-46.5, -45.0, 166.5, 169.5)
start_lat = -36.824549
start_lon = 176.240509

cyc_path = '/home/mike/data/wrf/tests/physics_tests/2023-02-10/d03_SMS-3DTKE/wrfout_d03_2023-02-13_00:00:00.nc'

plot_path = export_path.joinpath('plots')

#################################################
### Tests



# results_path = evaluate_models(source_path, test_path, output_path, domain, variables, region=region)


# positions = track_cyclone(
#     cyc_path,
#     start_lat=start_lat,
#     start_lon=start_lon,
#     smoothing_sigma=5,
# )

# png_files = plot_cyclone_track(cyc_path, positions, plot_path)


# results_path = evaluate_models_domain(source_path, test_path, output_path, domain, variables, region=region)


self = Evaluator(source_cfdb_path, test_cfdb_path)

res = self.evaluate_domain(output_path, variables, 'ne')













































































