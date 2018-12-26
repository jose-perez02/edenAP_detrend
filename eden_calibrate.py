import argparse
import astropy.config as astropy_config
from configparser import ConfigParser
from datetime import datetime
from dateutil import parser
import glob
import numpy as np
import os
import shutil
import sys
import traceback

from astropy import units as u
from astropy.time import Time
from astropy.io import fits
import time

import cal_data
from cal_data import filter_objects, find_val, prepare_cal, last_processing2, get_gain_rdnoise
from DirSearch_Functions import search_all_fits, set_mulifits

from constants import log, ModHDUList, check_and_rename
from multiprocessing import Pool

#
# Script to calibrate eden data using functions in cal_data.py
#

# define constants from config.ini
config = ConfigParser()
config.read('config.ini')
server_destination = config['FOLDER OPTIONS']['server_destination']

def quantity_check(calib_folder):
    # function to see if there are at least 5 calib files in a folder
    c_num = len(glob.glob(calib_folder+'/*'))
    check = True if c_num > 5 else False
    return check

def find_closest(telescope, datafolder, calib_type):
    # 
    # Look for the closest day with calibration files to use for creation of master calibs
    # Currently limits search to 20 days before and after for flats, as others are probably not valid
    # 365 days for bias and darks
    #
    date = Time(datafolder.strip('/').split('/')[-1])
    target = datafolder.strip('/').split('/')[-2]
    dcounter = 0
    check = False

    limit = 20 if calib_type == 'FLAT' else 365
    
    # Begin looking for nearest day
    while dcounter < limit and not check: 
        
        calib_date = date + dcounter * u.day # days ahead
        calib_folder = datafolder.replace('RAW', 'Calibrations').replace('/'+target,'').replace(date.iso.split()[0], calib_date.iso.split()[0]).replace(telescope, telescope+'/'+calib_type)
        if os.path.isdir(calib_folder):
            check = quantity_check(calib_folder)
        
        if not check: # days behind
            calib_date = date - dcounter * u.day
            calib_folder = datafolder.replace('RAW', 'Calibrations').replace('/'+target,'').replace(date.iso.split()[0], calib_date.iso.split()[0]).replace(telescope, telescope+'/'+calib_type)
            if os.path.isdir(calib_folder):
                check = quantity_check(calib_folder)
            
        if check: # after finding closest calibs, check for calibrated
            found_date = calib_date.iso.split()[0]
            print('\t Nearest '+calib_type+' folder with more than 5 files found on '+found_date)
        else:   
            dcounter += 1
    try:
        found_folder = datafolder.replace('RAW', 'Calibrations').replace('/'+target,'').replace(date.iso.split()[0], found_date).replace(telescope, telescope+'/'+calib_type)
        return found_folder
    except:
        print('\t NO ' + calib_type + ' folder found with more than 5 files within '+str(limit)+' days... ')
        return None

def eden_calibrate(telescope, datafolder, files):
    #
    # Function to produce calibrated images for photometry using cal_data functions
    # Writes in CALIBRATED directory
    #
    print('\n\t BEGINNING CALIBRATION PROCEDURE')
    # check if calibrated files exist, if so return
    # TO DO

    # Get calibration folders from closest day
    flat_folder = find_closest(telescope, datafolder, 'FLAT')
    bias_folder = find_closest(telescope, datafolder, 'BIAS')
    dark_folder = find_closest(telescope, datafolder, 'DARK')

    calibration = []
    if flat_folder != None:
        calibration.append(flat_folder)
    if bias_folder != None:
        calibration.append(bias_folder)
    if dark_folder != None:
        calibration.append(dark_folder)

    # get compatibility factors
    comp = cal_data.get_comp_info(datafolder)
    filters, bins, exptime_obj, rdnoise_gain = comp
    
    #calculate median/mean calibration files using search_median from cal_data
    # assumes calibrations is a string (directory path)
    bias_median, darks_median, flats_median, exptime_dark = cal_data.search_median(calibration,
                                                                                   comp,
                                                                                   twi_flat=False,
                                                                                   recycle=True,
                                                                                   median_opt=True)

    # If read noise doesn't exist in at least one header, calculate and put in header files.
    if not rdnoise_gain:
        print("Information about ReadNoise or Gain couldn't be found... Assigning New Values")
        log("Information about ReadNoise or Gain couldn't be found... Assigning New Values")
        # parse calibrations folder/file path
        
        log("Applying get_gain_rdnoise function...")
        gains, read_noises = get_gain_rdnoise(calibration, bins=bins, filters=filters)
        log("get_gain_rdnoise sucessful")
        #telescop = '/'.join(cal_folder.split('/')[:-1])
        print("Assigning following gain values:\n{}\n...and readnoise:\n{}".format(gains, read_noises))
        for i in range(len(gains)):
            value_set = 'GAIN{}'.format(i + 1), gains[i]
            comm = 'EDEN Corrected Gain of AMP{} in units of e-/ADU'.format(i + 1)
            set_mulifits(datafolder, '*.fits', value_set, comment=comm, keep_originals=False)
            for j in range(len(calibration)):
                set_mulifits(calibration[j], '*.fits', value_set, comment=comm, keep_originals=False)
            value_set = 'RDNOISE{}'.format(i + 1), read_noises[i]
            comm = 'EDEN Corrected Read Noise of AMP{} in units of e-'.format(i + 1)
            set_mulifits(datafolder, '*.fits', value_set, comment=comm, keep_originals=False)
            for j in range(len(calibration)):
                set_mulifits(calibration[j], '*.fits', value_set, comment=comm, keep_originals=False)
        log("Values have been assigned!... Continuing Calibration.")
        print("Values have been assigned!... Continuing Calibration.")
        
    # setup object files for calibration
    final_dir = datafolder.replace('RAW', 'CALIBRATED')
    if not os.path.isdir(final_dir):
        os.makedirs(final_dir)
    list_objects = list(search_all_fits(datafolder))
    filtered_objects = [obj_path for obj_path in list_objects if filter_objects(obj_path, bins)]
    # beta variable is to be multiplied by the corrected_darks to normalize it in respect to obj files
    betas = [find_val(objs, 'exptime') / exptime_dark for objs in filtered_objects]
    assert len(betas) == len(filtered_objects), "For some reason betas and objects aren't the same size"
    # set up calibration files to pickle through multiprocessing
    t0 = time.time()
    normflats = ModHDUList(flats_median)
    medbias = ModHDUList(bias_median)
    # try to get rid of infs/nans/zeroes
    normflats.interpolate()
    if darks_median:
        meddark = ModHDUList(darks_median)
        _list = [normflats, meddark, medbias]
        normflats, meddark, medbias = prepare_cal(filtered_objects[0], *_list)
    else:
        _list = [normflats, medbias]
        normflats, medbias = prepare_cal(filtered_objects[0], *_list)
        try: # sometimes normflats contains None for some reason
            meddark = [np.zeros(normflats[i].shape) for i in range(len(normflats))]
        except:
            for i in range(len(normflats)):
                try:
                    if normflats[i] != None:
                        meddark = [np.zeros(normflats[i].shape)]
                except:
                    meddark = []
                    for j  in range(len(normflats)):
                        meddark.append(np.zeros(normflats[i].shape))
                    #[np.zeros(normflats[i].shape),np.zeros(normflats[i].shape),np.zeros(normflats[i].shape)] # this is not going to work robustly
    

    lapse = time.time() - t0
    log("Preparation right before calibration took %.4f " % lapse)

    # create arguments list/iterator
    arguments = []
    for obj, beta in zip(filtered_objects, betas):
        # each argument will have an object frame, normalization constant(Beta), and directory names of
        # super calibration files, and final directory for object frames.
        arguments.append((obj, beta, normflats, meddark, medbias, final_dir))
        #arguments = [obj,beta,normflats,meddark,medbias,final_dir]
    # initialize multiprocessing pool in try/except block in order to avoid problems
    split = 3 # number of subprocesses
    pool = Pool(processes=split)
    try:
        t0 = time.time()
        pool.starmap(last_processing2, arguments)
        #map(last_processing2, arguments)
        lapse = time.time() - t0
        log("WHOLE CALIBRATION PROCESS IN ALL FILES TOOK %.4f" % lapse)
    except Exception:
        log("An error occurred during the multiprocessing, closing pool...")
        raise
    finally:
        # the finally block will ensure the pool is closed no matter what
        pool.close()
        pool.join()
        del arguments[:]
    log("FULL DATA CALIBRATION COMPLETED")
    print("CALIBRATION COMPLETED!")
    
    return
    
    
