import numpy as np
import argparse
import glob
import os
from astropy import units as u
from astroquery import simbad
from astropy.time import Time
from astropy.stats import sigma_clip
from datetime import datetime
import astropy.config as astropy_config
from configparser import ConfigParser
import pickle
import time

def eden_GPDetrend(telescope, datafolder, targets, calibrated=True):
    # define constants from config.ini
    config = ConfigParser()
    config.read('config.ini')
    server_destination = config['FOLDER OPTIONS']['server_destination']

    # create GPDetrend inputs, and run GPDetrend
    
    # create folder in post_processing for GPLC files
    if calibrated:
        out_dir = datafolder + 'calib_post_processing/' + 'GPLC/'
    else:
        out_dir = datafolder +'post_processing/'+'GPLC/'
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    # Load necessary files to create GPDetrend inputs
    nflux = np.genfromtxt(datafolder+'post_processing/'+targets[0]+'.dat')
    LC = np.genfromtxt(datafolder+'post_processing/LC/'+targets[0]+'.epdlc')
    comps = glob.glob(datafolder+'post_processing/comp_light_curves/*')
      
    # Store information
    # lc file
    times = nflux[:,0]
    flux = nflux[:,1]

    # sigma clip flux
    filt = sigma_clip(flux, sigma=5)
    filt = np.invert(filt.mask)
        
    # eparams file
    etime = LC[:,1]
    airmass = LC[:,22]
    FWHM = LC[:,19]
    cen_X = LC[:,15]
    cen_Y = LC[:,16]
    bg = LC[:,17]
    # for some reason the eparam times do not always match the flux times, this (usually) finds and removes the extras
    if len(times) != len(airmass):
        print('LC length does not match comps and eparams length!')
        print('Time length: ', len(times), 'eparams length: ', len(airmass))

        # rounding to 5 gets rid of small differences
        times = np.round(times, decimals=5)
        etime = np.round(etime, decimals=5)
        mask = np.in1d(etime, times) # find values truly not in lc

        airmass = airmass[mask]
        FWHM = FWHM[mask]
        cen_X = cen_X[mask]
        cen_Y = cen_Y[mask]
        bg = bg[mask]

    # comps file
    cflux = np.zeros((len(times),int(len(comps)/4)))
    count = 0
    for j in range(len(comps)):
        if 'pdf' or 'sigma' or 'norm' not in comps[j]:
            try: # does not always work
                comp = np.genfromtxt(comps[j])
                if len(times) != len(airmass):
                    comp = comp[mask]
                cflux[:,count] = comp[:,1]
                count =  count + 1
            except:
                pass
            else:
                pass
            
    # sigma mask
    times, flux = times[filt], flux[filt]
    airmass, FWHM, cen_X, cen_Y, bg = airmass[filt], FWHM[filt], cen_X[filt], cen_Y[filt], bg[filt]
    cflux = cflux[filt, :]
                
    # Write the GPDetrend input files
    # array format
    light = np.array([times, flux, np.zeros(len(times))])
    eparams = np.array([times, airmass, FWHM, bg, cen_X, cen_Y], dtype='float')

    # the FWHM often contains nans, this removes those times from all files.
    rem = np.where(np.isnan(FWHM))
    
    # Remove times with FWHM Nans and transpose
    eparams = np.delete(np.transpose(eparams), rem, axis=0)
    light = np.delete(np.transpose(light), rem, axis=0)
    cflux = np.delete(cflux, rem, axis=0)

    # write
    lfile = out_dir+'lc.dat'
    efile = out_dir+'eparams.dat'
    cfile = out_dir+'comps.dat'
    ofolder = 'detrend_'
        
    np.savetxt(lfile,light, fmt='%1.6f', delimiter='       ')
    np.savetxt(cfile,cflux, fmt='%1.6f', delimiter='       ')
    np.savetxt(efile,eparams, fmt='%1.6f', delimiter='       ', header='times, airmass, fwhm, background, x_cen, y_cen')

    # RUN DETREND
    # changing directories seems necessary, otherwise the out folder is too long a name for multinest
    mycwd = os.getcwd()
    os.chdir(out_dir)
    os.system('python '+mycwd+'/GPDetrend.py -ofolder '+ofolder+' -lcfile '+lfile+' -eparamfile '+efile+' -compfile '+cfile+' -eparamtouse  all' )
    os.chdir(mycwd)



    
