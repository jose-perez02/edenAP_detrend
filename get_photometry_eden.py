# -*- coding: utf-8 -*-
import argparse
import glob
import os
import pickle
from configparser import ConfigParser

import astropy.io.fits as pyfits
import numpy as np

import PhotUtils
from constants import get_telescopes, find_val, LOOKDATE, log

# define constants from config.ini
config = ConfigParser()
config.read('config.ini')
server_destination = config['FOLDER OPTIONS']['server_destination']

###################################################################

#  Get user input:
parser = argparse.ArgumentParser()
parser.add_argument('-telescope', default=None)
parser.add_argument('-datafolder', default=None)
parser.add_argument('-minap', default=5)
parser.add_argument('-maxap', default=50)
parser.add_argument('-apstep', default=1)
# If you are doing photometric monitoring of an object, sometimes you want to join 
# all the images from one object taken on the same band in the same output. Set this 
# to True if you want to do this:

# Run astrometry on the images?
parser.add_argument('--get_astrometry', dest='get_astrometry', action='store_true')
parser.set_defaults(get_astrometry=False)

# Refine the centroids of each target?
parser.add_argument('--ref_centers', dest='ref_centers', action='store_true')
parser.set_defaults(ref_centers=False)

args = parser.parse_args()

# Get the telescope name:
telescope = args.telescope

# Get datafolder/date of the observations that the user wants to reduce:
datafolder = args.datafolder

# Check if given telescope is in server, if it is use it, else exit program:
inServer = any([telescope.upper() == tel.upper() for tel in get_telescopes()])
data_folder = os.path.join(server_destination, telescope.upper()) if inServer else None

if data_folder is None:
    print("Telescope doesn't exist in server, attempting to retrieve from config.ini")
    if telescope in config['Manual Data Folders']:
        data_folder = config['Manual Data Folders'][telescope]
    else:
        print("No existing folder... Exiting...")
        exit(-1)

out_cal_folder = os.path.join(data_folder, 'cal')
out_red_folder = os.path.join(data_folder, 'red')

# Define apertures for aperture photometry:
min_aperture = int(args.minap)
max_aperture = int(args.maxap)
aperture_step = int(args.apstep)

get_astrometry = args.get_astrometry
ref_centers = args.ref_centers

###################################################################
print('\t ###################################')
print('\t Pre-processing....')
print('\t ###################################')

if not os.path.exists(os.path.join(out_red_folder, datafolder)):
    os.makedirs(os.path.join(out_red_folder, datafolder))

files_path = os.path.join(out_cal_folder, datafolder, '*.fits')
# Now, organize all observed objects in the given observing night:
files = sorted(glob.glob(files_path))
files_fz = sorted(glob.glob(os.path.join(os.path.dirname(files_path), '*.fits.fz')))
for i in range(len(files_fz)):
    fits_name = files_fz[i].split('.fz')[0]
    if fits_name not in files:
        files.append(files_fz[i])
all_objects = []  # This saves all the objects
all_ras = []  # This saves the RA of each object
all_decs = []  # This saves the DEC of each object
object_in_files = len(files) * ['']  # This saves what object is in each file
for file in files:
    if file.endswith('.wcs.fits') or file.endswith('_gf.fits'):
        files.remove(file)
good_objects = []
for i in range(len(files)):
    f = files[i]
    try:
        h0 = pyfits.getheader(f)
    except:
        print('File ', f, ' is corrupted. Skipping it')
        raise
    else:
        target = h0['OBJECT']
        filter = find_val(h0, 'FILTER', typ=str)
        obj_name = '{:}%%{:}'.format(target, filter)
        object_in_files[i] = obj_name
        if ('bias' in obj_name) or ('flat' in obj_name) or ('dark' in obj_name):
            continue
        if obj_name not in all_objects:
            all_objects.append(obj_name)
            date = LOOKDATE(h0)
            RA, DEC = PhotUtils.get_general_coords(target, date)
            if RA == 'NoneFound':
                RA = find_val(h0, 'RA')
                DEC = find_val(h0, 'DEC')
            all_ras.append(RA)
            all_decs.append(DEC)
            out_folder = os.path.join(out_red_folder, datafolder)
            if not os.path.exists(out_folder):
                os.mkdir(out_folder)
        good_objects.append(i)
files = [files[i] for i in good_objects]
object_in_files = [object_in_files[i] for i in good_objects]

print('\t Found ', len(all_objects), ' object(s) for the observations under ' + datafolder)
print('\t They are:', all_objects)

print('\t ###################################')
print('\t Going to photometric extraction....')
print('\t ###################################')

# Create apertures:
R = np.arange(min_aperture, max_aperture + 1, aperture_step)
# Get photometry for the objects:
for i in range(len(all_objects)):
    obj_name = all_objects[i]
    target, filter = obj_name.split('%%')
    print('\t Working on ' + obj_name)
    # out_data_folder = out_red_folder+datafolder+'/'+obj_name+'/'
    out_data_folder = os.path.join(out_red_folder, datafolder)
    all_files = []
    for j in range(len(files)):
        if obj_name == object_in_files[j]:
            all_files.append(files[j])
    # Convert RA and DECs of object to decimal degrees:
    ra_obj, dec_obj = PhotUtils.CoordsToDecimal([[all_ras[i], all_decs[i]]])
    if not os.path.exists(os.path.join(out_data_folder, 'photometry.pkl')):
        master_dict = None
    else:
        print('\t Found photometry.pkl')
        master_dict = pickle.load(open(os.path.join(out_data_folder, 'photometry.pkl'), 'rb'))
    # Get master dictionary for photometry, saving progress every 10 files:
    n_chunks = np.max([1, int(len(all_files) / 10)])
    chunked_files = np.array_split(all_files, n_chunks)
    for i, chunk in enumerate(chunked_files):
        log("Looping through chunked files #%d" % (i+1))
        master_dict = PhotUtils.getPhotometry(chunk, target, telescope, R, ra_obj, dec_obj, out_data_folder, filter,
                                              get_astrometry=get_astrometry, refine_cen=ref_centers,
                                              master_dict=master_dict)
        # Save dictionary:
        print('\t Saving photometry at ' + out_data_folder + '...')
        OUT_FILE = open(os.path.join(out_data_folder, 'photometry.pkl'), 'wb')
        pickle.dump(master_dict, OUT_FILE)
        OUT_FILE.close()
