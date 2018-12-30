import glob
import os
import time
from configparser import ConfigParser
from multiprocessing import Pool

import numpy as np
from astropy import units as u
from astropy.io import fits
from astropy.time import Time

from DirSearch_Functions import search_all_fits
from cal_data import filter_objects, find_val, last_processing
from cal_data import update_calibrations, get_best_comb, get_comp_info
from constants import log
from dirs_mgmt import validate_dirs

#
# Script to calibrate eden data using functions in cal_data.py
#

# define constants from config.ini
config = ConfigParser()
config.read('config.ini')
server_destination = config['FOLDER OPTIONS']['server_destination']


def quantity_check(calib_folder):
    # function to see if there are at least 5 calib files in a folder
    c_num = len(glob.glob(calib_folder + '/*'))
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

        calib_date = date + dcounter * u.day  # days ahead
        calib_folder = datafolder.replace('RAW', 'Calibrations').replace('/' + target, '').replace(date.iso.split()[0],
                                                                                                   calib_date.iso.split()[
                                                                                                       0]).replace(
            telescope, telescope + '/' + calib_type)
        if os.path.isdir(calib_folder):
            check = quantity_check(calib_folder)

        if not check:  # days behind
            calib_date = date - dcounter * u.day
            calib_folder = datafolder.replace('RAW', 'Calibrations').replace('/' + target, '').replace(
                date.iso.split()[0], calib_date.iso.split()[0]).replace(telescope, telescope + '/' + calib_type)
            if os.path.isdir(calib_folder):
                check = quantity_check(calib_folder)

        if check:  # after finding closest calibs, check for calibrated
            found_date = calib_date.iso.split()[0]
            print('\t Nearest ' + calib_type + ' folder with more than 5 files found on ' + found_date)
        else:
            dcounter += 1
    try:
        found_folder = datafolder.replace('RAW', 'Calibrations').replace('/' + target, '').replace(date.iso.split()[0],
                                                                                                   found_date).replace(
            telescope, telescope + '/' + calib_type)
        return found_folder
    except:
        print('\t NO ' + calib_type + ' folder found with more than 5 files within ' + str(limit) + ' days... ')
        return None


def eden_calibrate(telescope, datafolder, comp_info=None, starter=True):
    #
    # Function to produce calibrated images for photometry using cal_data functions
    # Writes in CALIBRATED directory
    #
    print('\n\t BEGINNING CALIBRATION PROCEDURE')

    if starter:
        # Update Combined Calibrations server
        update_calibrations(telescope)

    # Get compatibility info;  Override if given
    if comp_info is None:
        filters, bins, exptime_obj, rdnoise_gain = get_comp_info(datafolder)
    else:
        filters, bins, exptime_obj, rdnoise_gain = comp_info

    # Extract date
    date = datafolder.split(os.sep)[-1]

    # Get calibration folders from closest day
    bias_median, flats_median, darks_median = get_best_comb(telescope, date, 'BIAS', 'FLAT', 'DARK',
                                                            filt=filters, bins=bins, ret_none=True)

    assert bias_median, 'No compatible combined bias calibration file was found!'
    assert flats_median, 'No compatible combined flats calibration file was found!'

    # Get exptime for dark frame
    EXP = 'EXPOSURE' if 'EXPOSURE' in fits.getheader(bias_median) else 'EXPTIME'
    exptime_dark = None if not darks_median else find_val(darks_median, EXP)

    # Setup object files for calibration
    final_dir = datafolder.replace('RAW', 'CALIBRATED')
    validate_dirs(final_dir)

    # Search ALL fits.
    list_objects = search_all_fits(datafolder)
    filter_aid = [(filters, bins)]

    # We now must filter our target frames while also making sure the filtered frames get calibrated.
    filtered = []
    for frame in list_objects:
        if filter_objects(frame, bins, filters):
            filtered.append(frame)
            continue
        if starter:
            comp_info2 = get_comp_info(frame)
            if (comp_info2[0], comp_info2[1]) not in filter_aid:
                filter_aid.append((comp_info2[0], comp_info2[1]))
                log('Found target frames that must be calibrated separately.')
                log('Executing new calibration for Filter=%s and Binning=%r' % (comp_info2[0], comp_info2[1]))
                eden_calibrate(telescope, datafolder, comp_info=comp_info2, starter=False)
                log('Continuing with calibration for Filter=%s and Binning=%r' % (filters, bins))
                continue

    # beta variable is to be multiplied by the corrected_darks to normalize it in respect to obj files
    if exptime_dark:
        betas = [find_val(objs, EXP) / exptime_dark for objs in filtered]
    else:
        betas = np.zeros(len(filtered))

    # Create arguments list/iterator for last_processing function!
    arguments = []
    for obj, beta in zip(filtered, betas):
        # each 'argument_list' will have an object frame, beta, path files to combined calibration files,
        # and final directory for calibrated frames.
        arguments.append((obj, beta, flats_median, darks_median, bias_median, final_dir))

    # initialize multiprocessing pool in try/except block in order to avoid problems
    split = 4  # number of subprocesses
    pool = Pool(processes=split)
    try:
        t0 = time.time()
        pool.starmap(last_processing, arguments)
        lapse = time.time() - t0
        log("WHOLE CALIBRATION PROCESS IN ALL FILES TOOK %.4f" % lapse)
    except Exception:
        log("An error occurred during the multiprocessing, closing pool...")
        raise
    finally:
        # the finally block will ensure the pool is closed no matter what
        pool.close()
        pool.join()
    log("FULL DATA CALIBRATION COMPLETED")
    print("CALIBRATION COMPLETED!")
    return


if __name__ == '__main__':
    eden_calibrate('VATT', "/home/rayzote/EDEN/TestServer/RAW/VATT/2MUCD20263/2018-12-18")
