import argparse
from astropy import units as u
import astropy.config as astropy_config
from astropy.io import fits
from astropy.time import Time
from configparser import ConfigParser
from datetime import datetime
import glob
import numpy as np
import os
import pickle
import shutil
from tqdm import tqdm as bar
import time
from constants import get_telescopes, find_val, LOOKDATE, log
import PhotUtils

def get_photometry(telescope,datafolder,minap=5,maxap=50,apstep=1,get_astrometry=True,ref_centers=True):
    # define constants from config.ini
    config = ConfigParser()
    config.read('config.ini')
    server_destination = config['FOLDER OPTIONS']['server_destination']
    astrometry_timeout = float(config['ASTROMETRY OPTIONS']['timeout'])
    
    #Get all of the (good) files in datafolder and determine the filter
    files,filters=[],[]
    target = None
    for path in np.sort(glob.glob(datafolder+'/*')):
        # Check that this is a FITS file
        if path.strip('/').split('.')[-1] not in ['fits','fts','fit','fits.gz','fts.gz','fit.gz']:
            continue
        
        # Try to open the header
        try:
            h = fits.getheader(path)
        except:
            continue
        
        # Get the filter name
        filt = find_val(h,'FILTER',typ=str)
        
        # First image only
        if target is None:
            # Get the target, filter, date from the header
            target = h['OBJECT']
            date = LOOKDATE(h)
            
            # Get RA, Dec either by target lookup (preferred) or through the header    
            RA, Dec = PhotUtils.get_general_coords(target, date)
            if RA == 'NoneFound':
                RA = find_val(h0,'RA')
                Dec = find_val(h0,'DEC')
                
            # Convert RA and DECs of object to decimal degrees:
            RA_d,Dec_d = PhotUtils.CoordsToDecimal([[RA,Dec]])
            
        # Add these values to the lists
        files.append(path)
        filters.append(filt)
    files,filters = np.array(files),np.array(filters)
    
    for filt in np.unique(filters):
        print("\t Found {:d} images with filter: {:s}".format((filters==filt).sum(),filt))
    
    # If it doesn't already exist, create the output directory for this data set
    outdir = datafolder.replace('/RAW/','/REDUCED/').replace('/CALIBRATED/','/REDUCED/')
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    
    # Find photometry.pkl if it already exists
    if os.path.exists(outdir+'/photometry.pkl'):
        print("\t Found photometry.pkl")
        master_dict = pickle.load(open(outdir+'/photometry.pkl','rb'))
        # If all of the images have been reduced then we can skip this one
        frame_name = [filename.replace(server_destination,'') for filename in files]
        if np.in1d(frame_name,master_dict['frame_name']).all():
            print("\t Photometry complete! Skipping...")
            return
    else:
        master_dict = None
    
    # Aperture sizes
    R = np.arange(minap,maxap + 1,apstep)
        
    # Get master dictionary for photometry, saving progress every 10 files:
    n_chunks = np.max([1,int(len(files)/10)])
    chunked_files = np.array_split(files,n_chunks)
    for i in range(n_chunks):
        # Perform the photometry for this chunk
        master_dict = PhotUtils.getPhotometry(chunked_files[i],target,telescope,filters,R,RA_d,Dec_d,outdir,None,
                                              get_astrometry=get_astrometry, refine_cen=ref_centers,
                                              astrometry_timeout=astrometry_timeout,master_dict=master_dict)

        # Save dictionary:
        print('\t Saving photometry at ' + outdir + '...')
        OUT_FILE = open(os.path.join(outdir, 'photometry.pkl'), 'wb')
        pickle.dump(master_dict, OUT_FILE)
        OUT_FILE.close()
    


if __name__=="__main__":
    #  Get user input:
    parser = argparse.ArgumentParser()
    
    # Name of telescope
    parser.add_argument('-telescope', default=None)
    
    # Directory containing the raw OR calibrated image files
    parser.add_argument('-datafolder', default=None)
    
    # Defines the range of aperture radii to use (px)
    parser.add_argument('-minap', default=5)
    parser.add_argument('-maxap', default=50)
    parser.add_argument('-apstep', default=1)

    # Run astrometry on the images?
    parser.add_argument('--get_astrometry', dest='get_astrometry', action='store_true')
    parser.set_defaults(get_astrometry=True)

    # Refine the centroids of each target?
    parser.add_argument('--ref_centers', dest='ref_centers', action='store_true')
    parser.set_defaults(ref_centers=True)

    args = parser.parse_args()

    # Run the photometry routine
    get_photometry(args.telescope,args.datafolder,args.minap,args.maxap,args.apstep,args.get_astrometry,args.ref_centers)
