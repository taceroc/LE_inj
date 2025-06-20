from astropy.io import fits
from astropy import wcs
import os
import yaml
import numpy as np

prog_name = 'Convert FITS'
description = '''Create FITS files from LE simulations'''
prog = 'create_fits.py'

import argparse
from argparse import FileType, ArgumentParser

parser = ArgumentParser(description=description, prog=prog)

pred_files = parser.add_argument_group('Mandatory Inputs')

# Input file arguments.
pred_files.add_argument(
    '-file_to_parameters_surface',
    metavar='PARAMETERS',
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



def main(ins):
    # with open(ins.txtparameters, 'rb') as handle:
    #     parameters = json.load(handle)

    with open(ins.txtparameters_surface, 'r') as f:
        name_surface = yaml.safe_load(f)
        
    loc_to_values = name_surface[0]['path_to_surface_values']
    loc_to_fits = ins.loc_to_fits #parameters['loc_to_fits']
    # 'InfPlane_dt0_loop_ct165_loc[0.3, -1.3, 1, -32.6156]_dz0.06_w473_angle[0, -30]'
    # type_sim = parameters['type_sim']
    # ct = parameters['ct']
    # loc_plane = parameters['loc_plane']
    # depth_dust = parameters['dz0']
    # wavelength = parameters['w']
    # angles = parameters['angles']
    
    # name = f'surface_values{type_sim}_dt0_loop_ct{ct}_loc[{loc_plane}]_dz0{depth_dust}_w{wavelength}_angle[{angles}]'
    # path = os.path.join(loc_to_values, name+'.npy')
    path = os.path.join(loc_to_values)


    xx = path.replace("surface_values", "ximg_arcsec")
    yy = path.replace("surface_values", "yimg_arcsec")
    
    x = np.load(xx)
    y = np.load(yy)
    
    # pscratch/sd/t/taceroc/LE_inj/lightecho_modeling_oop/OOP/results/LC_infplane_test_multi/arrays/surface_valuesInfPlane_dt0_loop_ct165_loc[0.3, -1.3, 1, -32.6156]_dz0.06_w473_angle[0, -30].npy
    sv_ct_no0 = np.load(path)
    sv_ct_no0 = sv_ct_no0*(sv_ct_no0>0)
    # hdu = fits.PrimaryHDU(data=sv_ct_no0)
    # hdul = fits.HDUList([hdu])


    w = wcs.WCS(naxis=2)
    
    # what is the center pixel of the XY grid.
    w.wcs.crpix = [x.shape[0]/2, y.shape[1]/2]
    
    # what is the galactic coordinate of that pixel.
    w.wcs.crval = [x.min()+np.abs(x.max() - x.min())/2, y.min()+np.abs(y.max() - y.min())/2]
    
    # what is the pixel scale in lon, lat.
    w.wcs.cdelt = np.array([0.2, 0.2])


    # write the HDU object WITH THE HEADER
    header = w.to_header()
    hdul = fits.PrimaryHDU(data=sv_ct_no0, header=header)
    # hdu.writeto(filename)
    # hdul = fits.HDUList([
    #     hdu,
    #     fits.ImageHDU(x, name='X'),
    #     fits.ImageHDU(y, name='Y'),
    #     ])

    
    name = loc_to_values.split('/')[-1].replace('.npy', '.fits')
    print(name)
    path_fits = os.path.join(loc_to_fits, name)
    print(path_fits)
    hdul.writeto(path_fits, overwrite=True)

if __name__ == '__main__':
    ins = parser.parse_args()
    main(ins)

    