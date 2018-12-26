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

# Wrapper script for the photometry, post-processing, and detrending scripts.

# Clear the astroquery cache to ensure up-to-date values are grabbed from SIMBAD
if os.path.exists(astropy_config.get_cache_dir()+'/astroquery/'):
    # First delete the astropy cache for astroquery; otherwise the data might be out of date!
    shutil.rmtree(astropy_config.get_cache_dir()+'/astroquery/')
    
# Then import astropy, astroquery, and other modules in this code
from astropy import units as u
from astropy.time import Time
from astroquery import simbad

from constants import log
from get_photometry_eden import get_photometry
import PhotUtils
from transit_photometry import post_processing
from eden_GPDetrend import eden_GPDetrend


# Read config.ini
config = ConfigParser()
config.read('config.ini')
server_destination = config['FOLDER OPTIONS']['server_destination']
ASTROMETRY = config['PHOTOMETRY OPTIONS'].getboolean('ASTROMETRY')
REF_CENTERS = config['PHOTOMETRY OPTIONS'].getboolean('REF_CENTERS')

# Parse arguments
parserIO = argparse.ArgumentParser(description='Performs photometry and produces optimal light curves for all data from TELESCOPE within the past NDAYS.')
parserIO.add_argument('-telescope',default=None,help='name(s) of telescopes (e.g., VATT)')
parserIO.add_argument('-ndays',default=7,help='number of days to look back')
parserIO.add_argument('-target',default=None,help='specify the target')
parserIO.add_argument('--calibrated',action='store_true',help='look for data in the CALIBRATED directory instead of RAW')
parserIO.add_argument('--no-calibration',action='store_true',help='do not perform calibration, just use raw images')
parserIO.add_argument('--overwrite',action='store_true',help='overwrite existing photometry (photometry.pkl)')
parserIO.add_argument('--photometry',action='store_true',help='ONLY do the photometry')
parserIO.add_argument('--post-processing',action='store_true',help='ONLY do the post-processing (must do photometry first!)')
parserIO.add_argument('--detrending',action='store_true',help='ONLY do the detrending with GPDetrend (must do photometry and post-processing first!)')
args = parserIO.parse_args()
tele,ndays = args.telescope,int(args.ndays)

# Find the selected telescope under RAW/, otherwise exit
telescopes_list = [d.strip('/').split('/')[-1] for d in glob.glob(server_destination+'/RAW/*/')]
if tele not in telescopes_list:
    print("Telescope {:s} not supported.")
    print("Supported telescopes: "+', '.join(telescopes_list))
    exit()

# Are we using RAW or CALIBRATED data?
dtype = '/CALIBRATED/' if args.calibrated else '/RAW/'
if not args.no_calibration or args.calibrated:
    pkl_name = 'calib_photometry.pkl'
    post_name = 'calib_post_processing/'
else:
    pkl_name = 'photometry.pkl'
    post_name = 'post_processing/'

# Is the target specified?
if args.target is not None:
    # Do a SIMBAD lookup for this target name
    target_names = simbad.Simbad.query_objectids(args.target)
    
    # If the lookup fails, just use the argument target name
    if target_names is None:
        print("\nSIMBAD lookup failed for target {:s}".format(args.target))
        target_names = [args.target]
    else:
        # Convert the astropy Table into a string array
        target_names = target_names.as_array().astype(str)
        
        # Replace double spaces (why are these in SIMBAD?)
        target_names = [name.replace('  ',' ') for name in target_names]
        
        print("\nTarget {:s} identified by SIMBAD under the following names:".format(args.target))
        print(target_names)
        
        # Append the argument target name as well, in case it isn't in this list
        target_names.append(args.target)
else:
    target_names = None

# Find all of the dates under RAW or CALIBRATED for this telescope
date_dirs = np.sort(glob.glob(os.path.join(server_destination,dtype.replace('/',''),tele,'*','*-*-*/')))
dates = np.array([Time(d.strip('/').split('/')[-1]) for d in date_dirs])

# Filter these to dates which lie within ndays of today (ignore the time)
today = Time(datetime.today().strftime('%Y-%m-%d'))
mask = dates>(today-ndays*u.day)
date_dirs,dates = date_dirs[mask],dates[mask]

# Filter to the selected target
targets = np.array([d.strip('/').split('/')[-2] for d in date_dirs])
if target_names is not None:
    mask = np.in1d(targets,target_names)

    # Also check for names with underscores
    mask = mask|np.in1d(targets,[name.replace(' ','_') for name in target_names])
    date_dirs,dates,targets = date_dirs[mask],dates[mask],targets[mask]

# Print some info about the reduction to be performed
print("\nFound {:d} data sets from within the past {:d} days under {:s}.".format(mask.sum(),ndays,dtype))
print("Targets: "+", ".join(np.unique(targets)))

# Overwrite bypass (will double-check once)
bypass = False

# Loop through the directories; save the data sets which fail with exceptions
failed_phot = []
failed_post = []
failed_det = []
for i in range(len(date_dirs)):
    print("\nTarget: {:s} | Date: {:s}".format(targets[i],dates[i].iso.split()[0]))
    log("Now working on {:s}".format(date_dirs[i]))
    reduced_dir = date_dirs[i].replace(dtype,'/REDUCED/')
    lightcurves_dir = date_dirs[i].replace(dtype,'/LIGHTCURVES/')

    # Run the astrometry & photometry routine (unless --post-processing or --detrending is passed as an argument)
    # This produces photometry.pkl (under the REDUCED directory tree), which contains the absolute
    # flux of every star across several aperture sizes, as well as the x/y positions and FWHM
    if not args.post_processing and not args.detrending:
        print('\n\t###################################')
        print('\tDoing photometry....')
        print('\t###################################')
        
        # Delete photometry.pkl if --overwrite is passed as an argument, but check first
        if args.overwrite and os.path.exists(reduced_dir+pkl_name):            
            if bypass or input("Overwriting photometry.pkl! Press return to confirm, or any other key to exit: ") == '':
                os.remove(reduced_dir+pkl_name)
                log("Removing {:s}".format(reduced_dir+pkl_name))
                bypass = True
            else:
                exit()
                
        # Try to run photometry, but skip this data set if it fails
        try:
            # Run the photometry routine
            get_photometry(tele,date_dirs[i], calibrate= not args.no_calibration, use_calibrated=args.calibrated)
        except (KeyboardInterrupt,SystemExit):
            raise
        except:
            e = sys.exc_info()
            print("\t Photometry FAILED with error type: {0}".format(e[0]))
            print("\t See the log for more information.")
            log("\t Photometry failed for {:s} with the following error:".format(date_dirs[i]))
            log(traceback.print_exception(*e))
            failed_phot.append(i)
    else:
        print('\n\t###################################')
        print('\tSkipping photometry....')
        print('\t###################################')

    # Run the post-processing routine (unless --photometry is passed as an argument)
    # Chooses the optimal aperture size & set of reference stars then creates a light curve
    if not args.photometry and not args.detrending:
        print('\n\t###################################')
        print('\tDoing post-processing....')
        print('\t###################################')
        # Get the target coordinates first (legacy)
        RA, DEC = PhotUtils.get_general_coords(targets[i],parser.parse(dates[i].isot))
        target_coords = [[RA,DEC]]
        
        # Run the post-processing routine (if photometry was successful)
        print(reduced_dir+pkl_name)
        if os.path.exists(reduced_dir+pkl_name):
            # Try to do the post-processing, but except errors and continue
            try:
                # Run the post-processing routine
                post_processing(tele,reduced_dir,targets[i],target_coords,overwrite=True,ncomp=6, filename=pkl_name, outname=post_name)
                
                # Copy all of the .epdlc files into the LIGHTCURVES directory
                print('\t Copying lightcurves into {:s}'.format(lightcurves_dir))
                if not os.path.isdir(lightcurves_dir):
                    os.makedirs(lightcurves_dir)
                for filename in glob.glob(reduced_dir+post_name+'/LC/*.epdlc'):
                    shutil.copyfile(filename,lightcurves_dir+'/'+filename.split('/')[-1])
            except (KeyboardInterrupt,SystemExit):
                raise
            except:
                e = sys.exc_info()
                print("\t Post-processing FAILED with error type: {0}".format(e[0]))
                print("\t See the log for more information.")
                log("\t Post-processing failed for {:s} with the following error:".format(date_dirs[i]))
                log(traceback.print_exception(*e))
                failed_post.append(i)
            

        else:
            print('\t No photometry.pkl found - run the photometry routine first!')
        
    else:
        print('\n\t###################################')
        print('\tSkipping post-processing....')
        print('\t###################################')

    if not args.photometry and not args.post_processing:
        print('\n\t###################################')
        print('\tDoing detrending....')
        print('\t###################################')
        # call eden_GPDetrend
        if os.path.exists(reduced_dir+post_name):
            # Try to run detrending, but skip this data set if it fails
            try:
                # Run the detrending routine
                eden_GPDetrend(tele,reduced_dir,targets, calibrated=not args.no_calibration)
            except (KeyboardInterrupt,SystemExit):
                raise
            except:
                e = sys.exc_info()
                print("\t Detrending FAILED with error type: {0}".format(e[0]))
                print("\t See the log for more information.")
                log("\t Detrending failed for {:s} with the following error:".format(date_dirs[i]))
                log(traceback.print_exception(*e))
                failed_det.append(i)
    else:
        print('\n\t###################################')
        print('\tSkipping detrending....')
        print('\t###################################')

        

# At the end, print which data sets failed
if len(failed_phot)>0 or len(failed_post)>0:
    print("The following data sets FAILED to reduce:")
    log("The following data sets FAILED to reduce:")
if len(failed_phot)>0:
    string = ["({:s}/{:s}/{:s})".format(tele,targets[i],dates[i].iso.split()[0]) for i in failed_phot]
    print("Photometry:  {:s}".format('  '.join(string)))
    log("Photometry:  {:s}".format('  '.join(string)))
if len(failed_post)>0:
    string = ["({:s}/{:s}/{:s})".format(tele,targets[i],dates[i].iso.split()[0]) for i in failed_post]
    print("Post-processing:  {:s}".format('  '.join(string)))
    log("Post-processing:  {:s}".format('  '.join(string)))
if len(failed_det)>0:
    string = ["({:s}/{:s}/{:s})".format(tele,targets[i],dates[i].iso.split()[0]) for i in failed_det]
    print("Detrending:  {:s}".format('  '.join(string)))
    log("Detrending:  {:s}".format('  '.join(string)))




