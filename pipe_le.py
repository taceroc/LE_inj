import numpy as np
import pandas as pd
import os
import pickle
import sys
import subprocess
import yaml
import multiprocessing
import argparse
from argparse import FileType, ArgumentParser

import create_fits

prog_name = 'LESimulations plus FITS creation'
description = '''Simulate some LE images and generate fits file'''
prog = 'pipe_le.py'


parser = ArgumentParser(description=description, prog=prog)

# Collect input file arguments into a group.
pred_file_info = '''-file_to_parameters, '''

print(pred_file_info)
pred_files = parser.add_argument_group('Mandatory Inputs', pred_file_info)

# Input file arguments.
pred_files.add_argument(
    'funcsim',
    choices=['SimulateLEInfPlane', 'SimulateLECenterSphericalShell', 'SimulateLEFixPlane'],
    metavar='TYPESIMLE',
    help='Define type of simulation: SimulateLEInfPlane, ',
    type=str)
pred_files.add_argument(
    '-file_to_parameters',
    metavar='PARAMETERS',
    dest='txtparameters',
    required=True,
    help='txt containing a dict with the mandatory parameters for each type simulation',
    type=str)
pred_files.add_argument(
    '--bool_save',
    dest='bool_save',
    help='Save results',
    action=argparse.BooleanOptionalAction)
pred_files.add_argument(
    '--bool_show_plots',
    dest='bool_show_plots',
    help='Show Plots',
    action=argparse.BooleanOptionalAction)
pred_files.add_argument(
    '--bool_show_initial_object',
    dest='bool_show_initial_object',
    help='Show Initial position Plots',
    action=argparse.BooleanOptionalAction)




# Input file arguments.
pred_files.add_argument(
    '-file_to_parameters_surface',
    metavar='PARAMETERS_surface',
    dest='txtparameters_surface',
    required=True,
    help='file to yaml file with path to surface values',
    type=str)
# pred_files.add_argument(
#     '-loc_to_values',
#     dest='loc_to_values',
#     help='dir to simulations results',
#     type=str)
pred_files.add_argument(
    '-loc_to_fits',
    dest='loc_to_fits',
    help='dir to fits files',
    type=str)



if __name__ == '__main__':
    ins = parser.parse_args()

    if ins.bool_show_initial_object:
        bool_show_initial_object = '--bool_show_initial_object'
    else:
        bool_show_initial_object = '--no-bool_show_initial_object'
    
    if ins.bool_show_plots:
        bool_show_plots = '--bool_show_plots'
    else:
        bool_show_plots = '--no-bool_show_plots'
    
    if ins.bool_save:
        bool_save = '--bool_save'
    else:
        bool_save = '--no-bool_save'

    with open(ins.txtparameters, 'r') as f:
        valuesYaml = yaml.safe_load(f)

    if ~os.path.exists(ins.loc_to_fits):
        os.mkdir(ins.loc_to_fits)

    # worker_count = 2
    list_id = np.arange(len(valuesYaml))
    tasks = [(i, ins) for i in list_id]
    
    def command_to_run(ids, ins):
        run_le = ['python', 'lightecho_modeling_oop/OOP/main.py', f'{ins.funcsim}', '-file_to_parameters', f'{ins.txtparameters}', '-id', f"{ids}", f'{bool_save}', f'{bool_show_plots}', f'{bool_show_initial_object}']

        subprocess.call(run_le)
        create_fits.main(ins)

    def worker(args):
        """Unpacks arguments for multiprocessing."""
        return command_to_run(*args)
        
    with multiprocessing.Pool(processes=len(valuesYaml)) as pool:
        results = pool.map(worker, tasks)

        

    
        
#### python pipe_le.py SimulateLEInfPlane -file_to_parameters /pscratch/sd/t/taceroc/LE_inj/params_le.yml --bool_save --bool_show_plots --no-bool_show_initial_object -file_to_parameters_surface /pscratch/sd/t/taceroc/LE_inj/name_surface.yml -loc_to_fits /pscratch/sd/t/taceroc/LE_inj/fits

#### python pipe_le.py SimulateLEInfPlane -file_to_parameters /pscratch/sd/t/taceroc/LE_inj/params_le_for_4026_15_2022_y.yml --bool_save --no-bool_show_plots --no-bool_show_initial_object -file_to_parameters_surface /pscratch/sd/t/taceroc/LE_inj/name_surface.yml -loc_to_fits /pscratch/sd/t/taceroc/LE_inj/fits/4026_15


    

    

    