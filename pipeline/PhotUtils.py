import astropy.io.fits as pyfits
import time as clocking_time
import dateutil
import glob
import os
import subprocess
import sys
import numpy as np
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

# For airmass calculation:
import ephem as E
import jdcal
from math import modf

# Use for UTC -> BJD conversion:
import dateutil
import matplotlib.dates as mdates
from astropy.coordinates import SkyCoord
from astropy import units as u

# For aperture photometry:
import multiprocessing as mp
from scipy.ndimage.filters import gaussian_filter

# Ignore computation errors:
np.seterr(divide='ignore', invalid='ignore')

# Define style of plotting (ggplot is nicer):
plt.style.use('ggplot')

def read_setupfile():
    fin = open('../setup.dat','r')
    fpack_folder = ''
    astrometry_folder = ''
    SEND_EMAIL = False
    emailsender = ''
    emailsender_pwd = ''
    emailreceiver = ['']
    ASTROMETRY = False
    GF_ASTROMETRY = False
    done = False
    while True:
        line = fin.readline()
        if 'FOLDER OPTIONS' in line:
            while True:
                line = fin.readline()
                if 'funpack' in line:
                    fpack_folder = line.split('=')[-1].split('\n')[0].strip()
                if 'astrometry' in line:
                    astrometry_folder = line.split('=')[-1].split('\n')[0].strip()
                if 'USER OPTIONS' in line:
                    break
        if 'USER OPTIONS' in line:
            while True:
                line = fin.readline()
                line_splitted = line.split('=')
                if len(line_splitted) == 2:
                    opt,res = line_splitted
                    opt = opt.strip()
                    res = res.split('\n')[0].strip()
                    if 'SENDEMAIL' == opt:
                        if res.lower() == 'true':
                            SEND_EMAIL = True
                    if 'EMAILSENDER' == opt:
                            emailsender = res
                    if 'EMAILSENDER_PASSWORD' == opt:
                            emailsender_pwd = res
                    if 'EMAILRECEIVER' == opt:
                            emailreceiver = res.split(',')
                if 'PHOTOMETRY OPTIONS' in line:
                    break
        if 'PHOTOMETRY OPTIONS' in line:
            while True:
                line = fin.readline()
                line_splitted = line.split('=')
                if len(line_splitted) == 2:
                    opt,res = line_splitted
                    opt = opt.strip()
                    res = res.split('\n')[0].strip()
                    if opt == 'ASTROMETRY':
                        if res.lower() == 'true':
                            ASTROMETRY = True
                    if opt == 'GFASTROMETRY':
                        if res.lower() == 'true':
                            GF_ASTROMETRY = True
                if line == '':
                    done = True
                    break
        if done:
            break
    return fpack_folder,astrometry_folder,SEND_EMAIL,emailsender,emailsender_pwd,\
           emailreceiver,ASTROMETRY,GF_ASTROMETRY


fpack_folder,astrometry_folder,SEND_EMAIL,emailsender,emailsender_pwd,\
             emailreceiver,ASTROMETRY,GF_ASTROMETRY = read_setupfile()

# Define astrometry source directory:
astrometry_directory = astrometry_folder # '/data/astrometry/bin/'

def date_format(year,month,day,hour,minute,second):
        def fn(number):
                if number > 9:
                        return str(number)
                else:
                        return '0'+str(number)
        return fn(year)+'-'+fn(month)+'-'+fn(day)+'T'+\
                        fn(hour)+':'+fn(minute)+':'+fn(second)

def TrimLim(s):
    cols,rows = s[1:-1].split(',')
    min_col,max_col = cols.split(':')
    min_row,max_row = rows.split(':')
    return int(min_col),int(max_col),int(min_row),int(max_row)

def site_data_2_string(sitelong,sitelat,sitealt):
    try:
        basestring
    except NameError:
        basestring = str
    if not isinstance(sitelong,basestring):
        longitude = str(int(modf(360.-sitelong)[1]))+':'+str(modf(360.-sitelong)[0]*60.)
    else:
        longitude = sitelong
    if not isinstance(sitelat,basestring):
        latitude = str(int(modf(sitelat)[1]))+':'+str(-modf(sitelat)[0]*60.)
    else:
        latitude = sitelat
    if not isinstance(sitealt,basestring):
        return longitude,latitude,str(sitealt)
    else:
        return longitude,latitude,sitealt

def getCalDay(JD):
    year, month, day, hour= jdcal.jd2gcal(JD,0.0)
    hour = hour*24
    minutes = modf(hour)[0]*60.0
    seconds = modf(minutes)[0]*60.0
    hh = int(modf(hour)[1])
    mm = int(modf(minutes)[1])
    ss = seconds
    if(hh<10):
       hh = '0'+str(hh)
    else:
       hh = str(hh)
    if(mm<10):
       mm = '0'+str(mm)
    else:
       mm = str(mm)
    if(ss<10):
       ss = '0'+str(np.round(ss,1))
    else:
       ss = str(np.round(ss,1))
    return year,month,day,hh,mm,ss

def getTime(year,month,day,hh,mm,ss):
    return str(year)+'/'+str(month)+'/'+str(day)+' '+hh+':'+mm+':'+ss

def getAirmass(ra,dec,day,longitude,latitude,elevation):
    star = E.FixedBody()
    star._ra = ra
    star._dec = dec

    observer = E.Observer()
    observer.date = day
    observer.long = -1*E.degrees(longitude)
    observer.lat = E.degrees(latitude)
    observer.elevation = elevation
    observer.temp = 7.0
    observer.pressure = 760.0
    observer.epoch = 2000.0
    observer.horizon = -1*np.sqrt(2*observer.elevation/E.earth_radius)

    star.compute(observer)
    airmass = secz(star.alt)
#     h = star.alt * np.pi/180.
#     airmass = 1./np.sin(h + 244./(165.+47.*h**1.1))
    return airmass

def get_planet_data(planet_data,target_object_name):
    f = open(planet_data,'r')
    while True:
        line = f.readline()
        if line == '':
            break
        elif line[0] != '#':
            name,ra,dec = line.split()
            print( name,ra,dec )
            if target_object_name == name:
                f.close()
                return [[ra,dec]]
    f.close()
    print( 'Planet '+target_object_name+' not found in file '+planet_data )
    print( 'Add it to the list and try running the code again.' )
    sys.exit()

from astroquery.irsa import Irsa
# Default limit of rows is 500. Go for infinity and beyond!
Irsa.ROW_LIMIT = np.inf
import astropy.units as u
def get_dict(target,central_ra,central_dec,central_radius, ra_obj, dec_obj, hdulist, exts, R,\
        catalog = u'fp_psc',date='20180101'):

    print( '\t > Generating master dictionary for coordinates',central_ra,central_dec,'...' )
    # Make query to 2MASS:
    result = Irsa.query_region(coord.SkyCoord(central_ra,central_dec,unit=(u.deg,u.deg)),spatial = 'Cone',\
                               radius=central_radius*3600.*u.arcsec,catalog=catalog)

    # Query to PPMXL to get proper motions:
    resultppm = Irsa.query_region(coord.SkyCoord(central_ra,central_dec,unit=(u.deg,u.deg)),spatial = 'Cone',\
                               radius=central_radius*3600.*u.arcsec,catalog='ppmxl')

    # Get RAs, DECs, and PMs from this last catalog:
    rappmxl = np.array(resultppm['ra'].data.data.tolist())
    decppmxl = np.array(resultppm['dec'].data.data.tolist())
    rappm = np.array(resultppm['pmra'].data.data.tolist())
    decppm = np.array(resultppm['pmde'].data.data.tolist())

    # Save coordinates, magnitudes and errors on the magnitudes:
    all_ids = np.array(result['designation'].data.data.tolist()).astype(str)
    all_ra = np.array(result['ra'].data.data.tolist())
    all_dec = np.array(result['dec'].data.data.tolist())
    all_j = np.array(result['j_m'].data.data.tolist())
    all_j_err = np.array(result['j_msigcom'].data.data.tolist())
    all_k = np.array(result['k_m'].data.data.tolist())
    all_k_err = np.array(result['k_msigcom'].data.data.tolist())
    all_h = np.array(result['h_m'].data.data.tolist())
    all_h_err = np.array(result['h_msigcom'].data.data.tolist())

    # Correct RA and DECs for PPM. First, get delta T:
    year = int(date[:4])
    month = int(date[4:6])
    day = int(date[6:8])
    s = str(year)+'.'+str(month)+'.'+str(day)
    dt = dateutil.parser.parse(s)
    data_jd = sum(jdcal.gcal2jd(dt.year, dt.month, dt.day))
    deltat = (data_jd-2451544.5)/365.25
    print( '\t Correcting PPM for date '+date+', deltat: ',deltat,'...' )
    for i in range(len(all_ra)):
        c_ra = all_ra[i]
        c_dec = all_dec[i]
        #dist_hats = np.sqrt((59.410455-c_ra)**2 + (-25.189964-c_dec)**2)
        #if dist_hats < 3./3600.:
         #   print 'Found it!',c_ra,c_dec
        dist = (c_ra-rappmxl)**2 + (c_dec-decppmxl)**2
        min_idx = np.where(dist == np.min(dist))[0]
        # 3 arcsec tolerance:
        if dist[min_idx] < 3./3600.:
            all_ra[i] = all_ra[i] + deltat*rappm[min_idx]
            all_dec[i] = all_dec[i] + deltat*decppm[min_idx]
            #if dist_hats < 3./3600.:
            #    print 'New RA DEC:',all_ra[i],all_dec[i]
    print( '\t Done.' )

    # Check which ras and decs have valid coordinates inside the first image...
    # ...considering all image extensions.
    # Save only those as valid objects for photometry:
    idx = []
    all_extensions = np.array([])
    for ext in exts:
        h = hdulist[ext].header
        x_max = hdulist[ext].data.shape[1]
        y_max = hdulist[ext].data.shape[0]
        x,y = SkyToPix(h,all_ra,all_dec)
        for i in range(len(x)):
            if x[i]>0 and x[i]<x_max and y[i]>0 and y[i]<y_max:
                idx.append(i)
                all_extensions = np.append(all_extensions, ext)

    # Create dictionary that will save all the data:
    master_dict = {}
    master_dict['frame_name'] = np.array([])
    master_dict['UTC_times'] = np.array([])
    master_dict['BJD_times'] = np.array([])
    master_dict['JD_times'] = np.array([])
    master_dict['LST'] = np.array([])
    master_dict['exptimes'] = np.array([])
    master_dict['airmasses'] = np.array([])

    # Generate a flux dictionary for each target.
    master_dict['data'] = {}
    master_dict['data']['RA_degs'] = all_ra[idx]
    master_dict['data']['IDs'] = all_ids[idx]
    master_dict['data']['DEC_degs'] = all_dec[idx]
    master_dict['data']['RA_coords'],master_dict['data']['DEC_coords'] = DecimalToCoords(all_ra[idx],all_dec[idx])
    master_dict['data']['ext'] = all_extensions
    master_dict['data']['Jmag'] = all_j[idx]
    master_dict['data']['Jmag_err'] = all_j_err[idx]
    master_dict['data']['Kmag'] = all_k[idx]
    master_dict['data']['Kmag_err'] = all_k_err[idx]
    master_dict['data']['Hmag'] = all_h[idx]
    master_dict['data']['Hmag_err'] = all_h_err[idx]

    all_names = len(idx)*[[]]

    # Get index of target star:
    distances = np.sqrt((all_ra[idx]-ra_obj)**2 + (all_dec[idx]-dec_obj)**2)
    target_idx = np.where(distances == np.min(distances))[0]

    for i in range(len(idx)):
        if i != target_idx:
            all_names[i] = 'star_'+str(i)
        else:
            all_names[i] = 'target_star_'+str(i)
            # Replace RA and DEC with the ones given by the user:
            ra_str, dec_str = get_general_coords(target, date)
            ra_deg, dec_deg = CoordsToDecimal([[ra_str, dec_str]])
            master_dict['data']['RA_degs'][i] = ra_deg
            master_dict['data']['DEC_degs'][i] = dec_deg
            master_dict['data']['RA_coords'][i] = ra_str
            master_dict['data']['DEC_coords'][i] = dec_str
            #try:
            #    master_dict['data']['RA_degs'][i],master_dict['data']['DEC_degs'][i] = ra_obj,dec_obj
            #except:
            #    master_dict['data']['RA_degs'][i],master_dict['data']['DEC_degs'][i] = ra_obj[0],dec_obj[0]
            #master_dict['data']['RA_coords'][i],master_dict['data']['DEC_coords'][i] = DecimalToCoords(ra_obj,dec_obj)
            #master_dict['data']['RA_degs'][i],master_dict['data']['DEC_degs'][i] = CoordsToDecimal([['00:34:47.32','04:39:27.93']])
        master_dict['data'][all_names[i]] = {}
        master_dict['data'][all_names[i]]['centroids_x'] = np.array([])
        master_dict['data'][all_names[i]]['centroids_y'] = np.array([])
        master_dict['data'][all_names[i]]['background'] = np.array([])
        master_dict['data'][all_names[i]]['background_err'] = np.array([])
        master_dict['data'][all_names[i]]['fwhm'] = np.array([])
        for r in R:
            master_dict['data'][all_names[i]]['fluxes_'+str(r)+'_pix_ap'] = np.array([])
            master_dict['data'][all_names[i]]['fluxes_'+str(r)+'_pix_ap_err'] = np.array([])

    master_dict['data']['names'] = np.array(all_names)
    print ('\t > Extracting data for '+str(len(all_names))+' sources')
    return master_dict

from astropy.io import fits
def getPhotometry(filenames,target,telescope,R,ra_obj,dec_obj,out_data_folder,use_filter,\
                  get_astrometry=True,sitelong=None,sitelat=None,sitealt=None,refine_cen=False,\
                  master_dict=None, gf_opt = False):

    # Define radius in which to search for targets (in degrees):
    search_radius = 0.25

    # Initiallize empty dictionary if not saved and different variables:
    longitude,latitude,elevation = None, None, None
    if master_dict is None:
        master_dict = {}
        updating_dict = False
    else:
        updating_dict = True

    if telescope == 'SWOPE':
        filter_h_name = 'FILTER'
        long_h_name = 'SITELONG'
        lat_h_name = 'SITELAT'
        alt_h_name = 'SITEALT'
        exptime_h_name = 'EXPTIME'
        airmass_h_name = 'AIRMASS'
        lst_h_name = None
        t_scale_low = 0.4
        t_scale_high = 0.45
        egain = 2.3
        times_method = 1

    elif telescope == 'CHAT':
        filter_h_name = 'FILTERS'
        long_h_name = 'SITELONG'
        lat_h_name = 'SITELAT'
        alt_h_name = 'SITEALT'
        exptime_h_name = 'EXPTIME'
        airmass_h_name = 'Z'
        lst_h_name = 'LST'
        t_scale_low = 0.6
        t_scale_high = 0.65
        egain = 1.0
        times_method = 3

    elif telescope == 'LCOGT':
        # This data is good for LCOGT 1m.
        filter_h_name = 'FILTER'
        long_h_name = 'LONGITUD'
        lat_h_name = 'LATITUDE'
        alt_h_name = 'HEIGHT'
        exptime_h_name = 'EXPTIME'
        airmass_h_name = 'AIRMASS'
        lst_h_name = 'LST'
        t_scale_low = 0.3
        t_scale_high = 0.5
        egain = 'GAIN'
        times_method = 2

    elif telescope == 'SMARTS':
        filter_h_name = 'CCDFLTID'
        long_h_name = 'LONGITUD'
        lat_h_name = 'LATITUDE'
        alt_h_name = 'ALTITIDE'
        exptime_h_name = 'EXPTIME'
        airmass_h_name = 'SECZ'
        lst_h_name = 'ST'
        t_scale_low = 0.1
        t_scale_high = 0.4
        egain = 3.2
        times_method = 2

    elif telescope == 'OBSUC':
        filter_h_name = 'FILTER'
        long_h_name = None
        sitelong = 289.4656
        sitelat = -33.2692
        sitealt = 1450.
        lat_h_name = None
        alt_h_name = None
        exptime_h_name = 'EXPTIME'
        airmass_h_name = None
        lst_h_name = None
        t_scale_low = 0.2
        t_scale_high = 0.8
        egain = 'EGAIN'
        times_method = 2

    elif telescope == 'NTT':
        filter_h_name = 'HIERARCH ESO INS FILT1 NAME'
        long_h_name = 'HIERARCH ESO TEL GEOLON'
        lat_h_name = 'HIERARCH ESO TEL GEOLAT'
        alt_h_name = 'HIERARCH ESO TEL GEOELEV'
        exptime_h_name = 'HIERARCH ESO DET WIN1 DIT1'
        airmass_h_name = 'HIERARCH ESO TEL AIRM START'
        lst_h_name = None
        t_scale_low = 0.2
        t_scale_high = 0.8
        egain = 'HIERARCH ESO DET OUT1 GAIN'
        times_method = 2

    elif telescope == 'KUIPER':
        filter_h_name = 'FILTER'
        long_h_name = None
        lat_h_name = None
        alt_h_name = None
        sitelong = -110.73453
        sitelat = 32.41647
        sitealt = 2510.
        exptime_h_name = 'EXPTIME'
        airmass_h_name = 'AIRMASS'
        lst_h_name = 'LST-OBS'
        t_scale_low = 0.145
        t_scale_high = 0.145*4
        egain = 3.173 # 'GAIN1'
        times_method = 3

    elif telescope == 'SCHULMAN':
        filter_h_name = 'FILTER'
        long_h_name = 'LONG-OBS'
        lat_h_name = 'LAT-OBS'
        alt_h_name = 'ALT-OBS'
        exptime_h_name = 'EXPTIME'
        airmass_h_name = 'AIRMASS'
        lst_h_name = 'ST'
        t_scale_low = 0.32
        t_scale_high = 0.32*3
        egain = 1.28 # 'EGAIN'
        times_method = 2

    elif telescope == 'VATT':
        filter_h_name = 'FILTER'
        long_h_name = None
        lat_h_name = None
        alt_h_name = None
        sitelong = -109.892014
        sitelat = 32.701303
        sitealt = 3191.
        exptime_h_name = 'EXPTIME'
        airmass_h_name = 'AIRMASS'
        lst_h_name = 'ST'
        t_scale_low = 0.188
        t_scale_high = 0.188*4
        egain = 1.9 # 'DETGAIN'
        times_method = 3
        
    elif telescope == 'BOK': 
    	filter_h_name = 'FILTER'
    	long_h_name = None
    	lat_h_name = None
    	alt_h_name = None
    	sitelong = -111.6
    	sitelat = 31.98
    	sitealt = 2120.
    	exptime_h_name = 'EXPTIME'
    	airmass_h_name = 'AIRMASS'
    	lst_h_name = 'ST'
    	t_scale_low = 0.1
    	t_scale_high = 0.4*4
    	egain = 1.5
    	times_method = 3
    	
    else:
        print( 'ERROR: the selected telescope '+telescope+' is not supported.' )
        sys.exit()

    # Iterate through the files:
    first_time = True
    for f in filenames:
        # Decompress file. Necessary because Astrometry cant handle this:
        if f[-7:] == 'fits.fz':
            p = subprocess.Popen(fpack_folder+'funpack '+f, stdout = subprocess.PIPE, \
                                 stderr = subprocess.PIPE, shell = True)
            p.wait()
            if(p.returncode != 0 and p.returncode != None):
                print ('\t Funpack command failed. The error was:')
                out, err = p.communicate()
                print (err)
                print ('\n\t Exiting...\n')
                sys.exit()
            f = f[:-3]
        # Try opening the fits file (might be corrupt):
        try:
            hdulist = fits.open(f)
            fitsok = True
        except:
            print('\t Encountered error opening {:}'.format(f))
            fitsok = False

        # If frame already reduced, skip it:
        if updating_dict:
            if f in master_dict['frame_name']:
                fitsok = False
        if fitsok:
            h0 = hdulist[0].header  # primary fits header
            exts = get_exts(f)
            # Check filter:
            filter_ok = False
            if use_filter is None:
                filter_ok = True
            else:
                if use_filter == h0[filter_h_name]:
                    filter_ok = True

            if filter_ok:
                print ('\t Working on frame '+f)

                ########## OBTAINING THE ASTROMETRY ###############
                # First, run astrometry on the current frame if not ran already:
                filename = f.split('.fits')[0]
                if not os.path.exists(filename+'.wcs.fits') and get_astrometry:
                    print ('\t Calculating astrometry...')
                    run_astrometry(f, ra=ra_obj[0], dec=dec_obj[0], radius=0.5,\
                                   scale_low=t_scale_low, scale_high=t_scale_high, \
                                   apply_gaussian_filter=gf_opt)
                    print ('\t ...done!')

                # Now get data if astrometry worked...
                if os.path.exists(filename+'.wcs.fits') and get_astrometry:
                    # If get_astrometry flag is on, prefer the generated file instead of the original:
                    print( '\t Detected file '+filename+'.wcs.fits'+'. Using it...' )
                    hdulist = fits.open(filename+'.wcs.fits')
                # ...or if no astrometry was needed on the frame:
                elif not get_astrometry:
                    hdulist = fits.open(filename+'.fits')

                # Create master dictionary and define data to be used in the code:    
                if first_time:
                    if times_method == 2:
                        date = ''.join(h0['DATE-OBS'].split('T')[0].split('-'))
                    else:
                        date = ''.join(h0['DATE-OBS'].split('-'))
                    central_ra, central_dec = CoordsToDecimal([[h0['RA'], h0['DEC']]])
#                         print (central_ra, central_dec)
#                         print (ra_obj, dec_obj)
                    if not updating_dict:
                        master_dict = get_dict(target,central_ra[0],central_dec[0],search_radius,ra_obj,dec_obj,\
                                                         hdulist, exts, R,date=date)
                    else:
                        all_names = master_dict['data']['names']
                        all_data = master_dict['data'].keys()
                        all_names_d = []
                        all_idx = []
                        for dataname in all_data:
                            if 'star' in dataname:
                                all_names_d.append(dataname)
                                all_idx.append(int(dataname.split('_')[-1]))
                        idxs = np.argsort(all_idx)
                        all_names = len(all_names_d)*[[]]
                        for i in range(len(idxs)):
                            all_names[i] = all_names_d[idxs[i]]

                    if sitelong is None:
                        sitelong = h0[long_h_name]
                    if sitelat is None:
                        sitelat = h0[lat_h_name]
                    if sitealt is None:
                        sitealt = h0[alt_h_name]
                    if longitude is None:
                       #print sitelong,sitelat,sitealt
                       longitude,latitude,elevation = site_data_2_string(sitelong,sitelat,sitealt)
                    first_time = False

                # Save filename to master dictionary:
                master_dict['frame_name'] = np.append(master_dict['frame_name'],f)

                ########## OBTAINING THE TIMES OF OBSERVATION ####################
                # Get the BJD time. First, add exposure time:
                if times_method == 1:
                    utc_time = h0['DATE-OBS']+'T-'+h0['UT-TIME']
                elif times_method == 2:
                    utc_time = h0['DATE-OBS']
                elif times_method == 3:
                    utc_time = h0['DATE-OBS']+'T-'+h0['TIME-OBS']

                # Get time at the center of the observations (initial + exptime/2):
                t_center = mdates.date2num(dateutil.parser.parse(\
                                           utc_time)) + (h0[exptime_h_name]/(2.))*(1./(24.*3600.))

                # Convert back to string:
                date = (mdates.num2date(t_center))
                string_date = date_format(date.year,date.month,date.day,\
                                          date.hour,date.minute,date.second)

                # Prepare object in order to convert this UTC time to BJD time:
                t = Time(string_date, format='isot', scale='utc', \
                         location=(str(sitelong)+'d',str(sitelat)+'d',sitealt))

                coords = SkyCoord(h0['RA']+' '+h0['DEC'], unit=(u.hourangle, u.deg))

                # Save UTC, exposure, JD and BJD and LS times. Save also the airmass:
                master_dict['UTC_times'] = np.append(master_dict['UTC_times'],utc_time)
                master_dict['exptimes'] = np.append(master_dict['exptimes'],h0[exptime_h_name])
                master_dict['JD_times'] = np.append(master_dict['JD_times'],t.jd)
                master_dict['BJD_times'] = np.append(master_dict['BJD_times'],((t.bcor(coords)).jd))
                if lst_h_name is not None:
                    master_dict['LST'] = np.append(master_dict['LST'],h0[lst_h_name])
                else:
                    t.delta_ut1_utc = 0.
                    c_lst = str(t.sidereal_time('mean', 'greenwich'))
                    c_lst = c_lst.split('h')
                    hh = c_lst[0]
                    c_lst = c_lst[1].split('m')
                    mm = c_lst[0]
                    ss = (c_lst[1])[:-1]
                    master_dict['LST'] = np.append(master_dict['LST'],hh+':'+mm+':'+ss)
                if airmass_h_name is not None:
                    master_dict['airmasses'] = np.append(master_dict['airmasses'],h0[airmass_h_name])
                else:
                    year,month,day,hh,mm,ss = getCalDay((t.bcor(coords)).jd)
                    day = getTime(year,month,day,hh,mm,ss)
                    master_dict['airmasses'] = np.append(master_dict['airmasses'],getAirmass(central_ra[0],central_dec[0],day,longitude,latitude,float(elevation)))

                ########## OBTAINING THE FLUXES ###################
                #master_dict['data']['RA_degs'][223],master_dict['data']['DEC_degs'][223] = 19.4620208,0.3419944
                for ext in exts:
                    # Load the data:
                    h = hdulist[ext]
                    data = hdulist[ext].data
                    # Get the indices of stars on this extension
                    idx = np.where(master_dict['data']['ext']==ext)
                    # Get the names of stars on this extension
                    names_ext = master_dict['data']['names'][idx]
                    x,y = SkyToPix(h,master_dict['data']['RA_degs'][idx],master_dict['data']['DEC_degs'][idx])
                    # Get fluxes of all the targets in this extension for different apertures:
                    print ('\t Performing aperture photometry on objects...')
                    tic = clocking_time.time()
                    if type(egain) == type('str'):
                        fluxes,errors,x_ref,y_ref,bkg,bkg_err,fwhm = getAperturePhotometry(data,h,x,y,R,\
                               names_ext, frame_name = filename.split('/')[-1], \
                               out_dir = out_data_folder, GAIN = h[egain], saveplot = False, refine_centroids = refine_cen)
                    else:
                        fluxes,errors,x_ref,y_ref,bkg,bkg_err,fwhm = getAperturePhotometry(data,h,x,y,R,\
                               names_ext, frame_name = filename.split('/')[-1], \
                               out_dir = out_data_folder, GAIN = egain, saveplot = False, refine_centroids = refine_cen)
                    #print all_names[71]
                    #print 'Centroids, before:',x[71],y[71]
                    #print 'Centroids, after :',x_ref[71],y_ref[71]
                    toc = clocking_time.time()
                    print ('\t Took {:1.2f} seconds.'.format(toc-tic))
                    # Save everything in the dictionary:
                    for i in range(len(names_ext)):
                        master_dict['data'][names_ext[i]]['centroids_x'] = np.append(master_dict['data'][names_ext[i]]['centroids_x'],x_ref[i])
                        master_dict['data'][names_ext[i]]['centroids_y'] = np.append(master_dict['data'][names_ext[i]]['centroids_y'],y_ref[i])
                        master_dict['data'][names_ext[i]]['background'] = np.append(master_dict['data'][names_ext[i]]['background'],bkg[i])
                        master_dict['data'][names_ext[i]]['background_err'] = np.append(master_dict['data'][names_ext[i]]['background_err'],bkg_err[i])
                        master_dict['data'][names_ext[i]]['fwhm'] = np.append(master_dict['data'][names_ext[i]]['fwhm'],fwhm[i])
                        for j in range(len(R)):
                            master_dict['data'][names_ext[i]]['fluxes_'+str(R[j])+'_pix_ap'] = \
                                np.append(master_dict['data'][names_ext[i]]['fluxes_'+str(R[j])+'_pix_ap'],fluxes[i,j])
                            master_dict['data'][names_ext[i]]['fluxes_'+str(R[j])+'_pix_ap_err'] = \
                                np.append(master_dict['data'][names_ext[i]]['fluxes_'+str(R[j])+'_pix_ap_err'],errors[i,j])
    return master_dict

def organize_files(files,obj_name,filt,leaveout=''):
    dome_flats = []
    sky_flats = []
    bias = []
    objects = []
    all_objects = len(files)*[[]]
    unique_objects = []
    for i in range(len(files)):
        try:
            with pyfits.open(files[i]) as hdulist:
                d, h = hdulist[1].data, hdulist[0].header
#             d,h = pyfits.getdata(files[i],header=True)
        except:
            print( 'File '+files[i]+' probably corrupt. Skipping it' )
            if i+1 == len(files):
                break
            i = i + 1

        if h['EXPTYPE'] == 'Bias':
            all_objects[i] = 'Bias'
        else:
            all_objects[i] = h['OBJECT']

        try:
                c_filter = h['FILTER']
        except:
                c_filter = filt

        if h['OBJECT'] not in unique_objects and c_filter == filt and h['EXPTYPE'] != 'Bias':
                unique_objects.append(h['OBJECT'])
        elif h['EXPTYPE'] == 'Bias':
                unique_objects.append('Bias')#h['OBJECT'])

    print( '\t We found the following frames:' )
    for i in range(len(unique_objects)):
        counter = 0
        for obj in all_objects:
                if obj == unique_objects[i]:
                        counter = counter + 1
        print( '\t   ('+str(i)+') '+unique_objects[i]+' ('+str(counter)+' frames)' )

    print( '\t Which ones are the (separate your selection with commas, e.g., 0,3,4)...' )
    idx_biases = [int(i) for i in raw_input('\t ...biases?').split(',')]
    idx_dome_flats = [int(i) for i in raw_input('\t ...dome flats?').split(',')]
    idx_sky_flats = [int(i) for i in raw_input('\t ...sky flats?').split(',')]
    idx_science = [int(i) for i in raw_input('\t ...science frames?').split(',')]

    for i in range(len(files)):
        for j in range(len(unique_objects)):
            if unique_objects[j] == all_objects[i]:
                if leaveout != '':
                    im_name = files[i].split(leaveout)[0]
                else:
                    im_name = files[i]
                if j in idx_biases:
                    bias.append(im_name)
                elif j in idx_dome_flats:
                    dome_flats.append(im_name)
                elif j in idx_sky_flats:
                    sky_flats.append(im_name)
                elif j in idx_science:
                    objects.append(im_name)

    return bias,dome_flats,sky_flats,objects

def NormalizeFlat(MasterFlat):
        original_shape = MasterFlat.shape
        flattened_Flat = MasterFlat.flatten()
        median_f = np.median(flattened_Flat)
        idx = np.where(flattened_Flat==0)
        flattened_Flat = flattened_Flat/median_f
        flattened_Flat[idx] = np.ones(len(idx))
        return flattened_Flat.reshape(original_shape)

def MedianCombine(ImgList,MB=None,flatten_counts = False):
    n = len(ImgList)
    if n==0:
        raise ValueError("empty list provided!")

    with pyfits.open(ImgList[0]) as hdulist:
        d, h = hdulist[1].data, hdulist[1].header
#     data,h = pyfits.getdata(ImgList[0],header=True)
    datasec1,datasec2 = h['DATASEC'][1:-1].split(',')
    ri,rf = datasec2.split(':')
    ci,cf = datasec1.split(':')
    data = data[int(ri)-1:int(rf),int(ci)-1:int(cf)]

    factor = 1.25
    if (n < 3):
        factor = 1

    ronoise = factor * h['ENOISE'] / np.sqrt(n)
    gain    = h['EGAIN']

    if (n == 1):
        if h['EXPTIME']>0:
                texp = h['EXPTIME']
        else:
                texp = 1.
        if flatten_counts:
            return ((data-MB)/texp)/np.median((data-MB)/texp),ronoise, gain
        else:
            return data/texp, ronoise, gain
    else:
        for i in range(n-1):
            with pyfits.open(ImgList[i+1]) as hdulist:
                d, h = hdulist[1].data, hdulist[1].header
#             d,h = pyfits.getdata(ImgList[i+1],header=True)
            datasec1,datasec2 = h['DATASEC'][1:-1].split(',')
            ri,rf = datasec2.split(':')
            ci,cf = datasec1.split(':')
            d = d[int(ri)-1:int(rf),int(ci)-1:int(cf)]
            if h['EXPTIME'] > 0:
                    texp = h['EXPTIME']
            else:
                    texp = 1.
            if flatten_counts:
               data = np.dstack((data,((d-MB)/texp)/np.median((d-MB)/texp)))
            else:
               data = np.dstack((data,d/texp))
        return np.median(data,axis=2), ronoise, gain
 
def run_astrometry(filename, ra=None, dec=None, radius=None, scale_low= 0.1, scale_high=1., apply_gaussian_filter=False):
    """
    This code runs Astrometry.net on a frame.

    * ra and dec:     are guesses of the ra and the dec of the center of the field (in degrees).

    * radius:     radius (in degrees) around the input guess ra,dec that astrometry should look for.

    * scale_[low, high]:     are scale limits (arcsec/pix) for the image.
    """
    
    exts = get_exts(filename)
    print('\t\t Found {:} extensions'.format(len(exts)))

    true_filename = filename
    if apply_gaussian_filter:
        print('\t\t Applying gaussian filter...')
        filename = filename.replace('.fits', '_gf.fits')
        with pyfits.open(true_filename) as hdulist:
            for ext in exts:
                hdulist[ext].data = gaussian_filter(hdulist[ext].data, 5)
            # Overwrite argument is helpful if pipeline failed previously
            hdulist.writeto(filename, overwrite=True)
    
    for ext in exts:
        print('\t\t Working on extension {:}...'.format(ext))           
        ext_fname = filename.replace('.fits', '_'+str(ext)+'.wcs.fits')
        if (ra is not None) and (dec is not None) and (radius is not None):
            p = subprocess.Popen(astrometry_directory+'solve-field --overwrite --no-plots --downsample 2 --cpulimit 60 --extension '+ str(ext)+\
                    ' --scale-units arcsecperpix --scale-low '+str(scale_low)+' --scale-high '+str(scale_high)+\
                    ' --ra '+str(ra)+' --dec '+str(dec)+' --radius '+str(radius)+\
                    ' --new-fits '+ext_fname+\
                    ' '+filename, stdout = subprocess.PIPE, \
                    stderr = subprocess.PIPE, shell = True)
        else:
            p = subprocess.Popen(astrometry_directory+'solve-field --overwrite --no-plots --downsample 2 --cpulimit 60 --extension '+ str(ext)+\
                    ' --scale-units arcsecperpix --scale-low '+str(scale_low)+' --scale-high '+str(scale_high)+\
                    ' --new-fits '+ext_fname+\
                    ' '+filename, stdout = subprocess.PIPE, \
                    stderr = subprocess.PIPE, shell = True)
        p.wait()
        if(p.returncode != 0 and p.returncode != None):
            print ('\t ASTROMETRY FAILED. The error was:')
            out, err = p.communicate()
            print (err)
            print ('\n\t Exiting...\n')
            sys.exit()
    
    # Astrometry.net is run on individual extensions, which are saved above.
    # Combining them back into a single file.
    with pyfits.open(true_filename) as hdulist:
        for ext in exts:
            ext_fname = filename.replace('.fits', '_'+str(ext)+'.wcs.fits')
            # Try to save new WCS info in original file
            try:
                with pyfits.open(ext_fname) as hdulist_new:
                    if ext==0:
                        hdulist[ext].header = hdulist_new[ext].header
                    else:
                        hdulist[ext].header = hdulist_new[1].header
            # If it doesn't work, copy the WCS info from the last file
            except FileNotFoundError:
                data_dir = '/'.join(true_filename.split('/')[:-1])
                this_file = true_filename.split('/')[-1]
                suffix = get_suffix(this_file)
                this_frame = get_trailing_number(this_file.replace(suffix, ''))
                last_frame = this_frame - 1
                last_file = str(last_frame).join(this_file.split(str(this_frame)))
                last_filename = '/'.join([data_dir, last_file])
                last_wcs_filename = last_filename.replace('.fits', '.wcs.fits')
                print('\t\t Astrometry failed for extension {:}'.format(ext))
                print('\t\t Using WCS info from previous frame {:}'.format(last_wcs_filename))
                with pyfits.open(last_wcs_filename) as hdulist_new:
                    hdulist[ext].header = hdulist_new[ext].header
            # If it works, remove the single-extension astrometry file
            else:
                os.remove(ext_fname)
        # Save the original file with the WCS info
        hdulist.writeto(true_filename.replace('.fits', '.wcs.fits'))
    
    # Save space by removing the gaussian filtered image, if any
    # Second condition is not necessary--just a sanity check
    if apply_gaussian_filter and '_gf.fits' in filename:
        os.remove(filename)
                        
from astropy import wcs
def SkyToPix(h,ras,decs):
    """
    This code converts input ra and decs to pixel coordinates given the
    header information (h).
    """
    # Load WCS information:
#     h['EPOCH'] = float(h['EPOCH'])
#     h['EQUINOX'] = float(h['EQUINOX'])
    w = wcs.WCS(h)
    # Generate matrix that will contain sky coordinates:
    sky_coords = np.zeros([len(ras),2])
    # Fill it:
    for i in range(len(ras)):
        sky_coords[i,0] = ras[i]
        sky_coords[i,1] = decs[i]
    # Get pixel coordinates:
    pix_coords = w.wcs_world2pix(sky_coords, 1)
#     for i in range(len(pix_coords)):
#         print(sky_coords[i], pix_coords[i])
    # Return x,y pixel coordinates:
    return pix_coords[:,0],pix_coords[:,1]

from photutils import CircularAperture,CircularAnnulus,aperture_photometry
from photutils import DAOStarFinder as daofind
from astropy.stats import median_absolute_deviation as mad

global_d = 0
global_h = 0
global_x = 0
global_y = 0
global_R = 0
global_target_names = 0
global_frame_name = None
global_out_dir = None
global_saveplot = False
global_refine_centroids = False
global_half_size = 30
global_GAIN = 2.3
fwhm_factor = 2.
sigma_gf = 5.*fwhm_factor # 5.

def getAperturePhotometry(d,h,x,y,R,target_names, frame_name = None, out_dir = None, saveplot = False, \
        refine_centroids = False, half_size = 50, GAIN = 1.0, ncores = None):

    global global_d,global_h,global_x,global_y,global_R,global_target_names,global_frame_name,\
              global_out_dir,global_saveplot,global_refine_centroids,global_half_size,global_GAIN
    global_d = d
    global_h = h
    global_x = x
    global_y = y
    global_R = R
    global_target_names = target_names
    global_frame_name = frame_name
    global_out_dir = out_dir
    global_saveplot = saveplot
    global_refine_centroids = refine_centroids
    global_half_size = half_size
    global_GAIN = GAIN

    fluxes = np.zeros([len(x),len(R)])
    fluxes_err = np.zeros([len(x),len(R)])
    x_ref = np.zeros(len(x))
    y_ref = np.zeros(len(y))
    bkg = np.zeros(len(x))
    bkg_err = np.zeros(len(x))
    fwhm = np.zeros(len(x))

    if ncores is None:
        pool = mp.Pool(processes=4)
    else:
        pool = mp.Pool(processes=ncores)
    results = pool.map(getCentroidsAndFluxes, range(len(x)))
    pool.terminate()

    for i in range(len(x)):
        fluxes[i,:],fluxes_err[i,:],x_ref[i],y_ref[i],bkg[i],bkg_err[i],fwhm[i] = results[i]
    return fluxes,fluxes_err,x_ref,y_ref,bkg,bkg_err,fwhm

def getCentroidsAndFluxes(i):
    fluxes_R = np.ones(len(global_R))*(-1)
    fluxes_err_R = np.ones(len(global_R))*(-1)
    # Generate a sub-image around the centroid, if centroid is inside the image:
    if global_x[i]>0 and global_x[i]<global_d.shape[1] and \
        global_y[i]>0 and global_y[i]<global_d.shape[0]:
        x0 = np.max([0,int(global_x[i])-global_half_size])
        x1 = np.min([int(global_x[i])+global_half_size,global_d.shape[1]])
        y0 = np.max([0,int(global_y[i])-global_half_size])
        y1 = np.min([int(global_y[i])+global_half_size,global_d.shape[0]])
        subimg = np.float64(np.copy(global_d[y0:y1,x0:x1]))

        # Substract the (background) median counts, get estimate 
        # of the sky std dev:
        background = np.median(subimg)
        background_sigma = 1.48 * mad(subimg)
        subimg -= background
        sky_sigma = np.ones(subimg.shape)*background_sigma
        x_cen = global_x[i] - x0
        y_cen = global_y[i] - y0
        if global_refine_centroids:
            # Refine the centroids
            x_new, y_new = get_refined_centroids(subimg, x_cen, y_cen)
            if x_new>0 and x_new<global_d.shape[1] and y_new>0 and y_new<global_d.shape[0]:
                x_cen, y_cen = x_new, y_new
        x_ref = x0 + x_cen
        y_ref = y0 + y_cen
        # If saveplot is True, save image and the centroid:
        if global_saveplot and ('target' in global_target_names[i]):
            if not os.path.exists(global_out_dir+global_target_names[i]):
                os.mkdir(global_out_dir+global_target_names[i])
            im = plt.imshow(subimg)
            im.set_clim(0,1000)
            plt.plot(x_cen,y_cen,'wx',markersize=15,alpha=0.5)
            circle = plt.Circle((x_cen,y_cen),np.min(global_R),color='black',fill=False)
            circle2 = plt.Circle((x_cen,y_cen),np.max(global_R),color='black',fill=False)
            plt.gca().add_artist(circle)
            plt.gca().add_artist(circle2)
            if not os.path.exists(global_out_dir+global_target_names[i]+'/'+global_frame_name+'.png'):
                plt.savefig(global_out_dir+global_target_names[i]+'/'+global_frame_name+'.png')
            plt.close()
        # With the calculated centroids, get aperture photometry:
        for j in range(len(global_R)):
            fluxes_R[j],fluxes_err_R[j] = getApertureFluxes(subimg,x_cen,y_cen,global_R[j],sky_sigma,global_GAIN)
        try:
            fwhm = estimate_fwhm(subimg,x_cen,y_cen)
        except:
            fwhm = -1
        return fluxes_R, fluxes_err_R, x_ref, y_ref,background,background_sigma,fwhm
    else:
        return fluxes_R, fluxes_err_R, global_x[i], global_y[i],0.,0.,0.

import warnings
def get_refined_centroids(data, x_init, y_init, half_size=15):
    """
    Refines the centroids by fitting a centroid to the central portion of an image
    Method assumes initial astrometry is accurate within the half_size
    """
    # Take the central portion of the data (i.e., the subimg)
    x0 = np.max([0, int(x_init)-half_size])
    x1 = np.min([int(x_init)+half_size, data.shape[1]])
    y0 = np.max([0, int(y_init)-half_size])
    y1 = np.min([int(y_init)+half_size, data.shape[0]])
    x_refined = x_init - x0
    y_refined = y_init - y0
    x_guess = x_init - x0
    y_guess = y_init - y0
    cen_data = np.float64(np.copy(data[y0:y1, x0:x1]))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        sources = daofind(threshold=0, fwhm=2.35*5).find_stars(gaussian_filter(cen_data, 5))
    xcents = sources['xcentroid']
    ycents = sources['ycentroid']
    dists = np.sqrt( (x_refined-xcents)**2 + (y_refined-ycents)**2 )
    try:
        idx_min = np.where(dists == np.min(dists))[0]
        if(len(idx_min)>1):
            idx_min = idx_min[0]
        x_guess = xcents[idx_min][0]
        y_guess = ycents[idx_min][0]
    except:
        print('\t\t Daofind failed. Refining pointing with a gaussian...')
        try:
            # Robustly fit a gaussian
            p = fit_gaussian(cen_data)
            x_guess = p[1]
            y_guess = p[2]
        except:
            print('\t\t No luck. Resorting to astrometric coordinates.')
    # Don't let the new coordinates stray outside of the sub-image
    if x_guess < data.shape[0] and x_guess>0:
        x_refined = x_guess
    if y_guess < data.shape[1] and y_guess>0:
        y_refined = y_guess
    return x_refined+x0, y_refined+y0

def gaussian(height, center_x, center_y, width_x, width_y):
    """Returns a gaussian function with the given parameters"""
    width_x = float(width_x)
    width_y = float(width_y)
    return lambda x,y: height*np.exp(
                -(((center_x-x)/width_x)**2+((center_y-y)/width_y)**2)/2)

def moments(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution by calculating its
    moments """
    total = data.sum()
    X, Y = np.indices(data.shape)
    x = (X*data).sum()/total
    y = (Y*data).sum()/total
    col = data[:, int(y)]
    width_x = np.sqrt(np.abs((np.arange(col.size)-y)**2*col).sum()/col.sum())
    row = data[int(x), :]
    width_y = np.sqrt(np.abs((np.arange(row.size)-x)**2*row).sum()/row.sum())
    height = data.max()
    return height, x, y, width_x, width_y

from scipy import optimize
def fit_gaussian(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution found by a fit"""
    params = moments(data)
    errorfunction = lambda p: np.ravel(gaussian(*p)(*np.indices(data.shape)) -
                                 data)
    p, success = optimize.leastsq(errorfunction, params)
    return p

def estimate_fwhm(data,x0,y0):
    """
    This function estimates the FWHM of an image containing only one source
    (possibly gaussian) and estimates the FWHM of it by performing two cuts on 
    X and Y and finding the stddev of both cuts. The resulting FWHM is obtained from 
    the mean of those stddev.  
    """

    def get_second_moment(x,y,mu):
        moment = np.sqrt(np.sum(y*(x-mu)**2)/np.sum(y))
        if not np.isnan(moment):
            return np.sqrt(np.sum(y*(x-mu)**2)/np.sum(y))
        else:
            return 0
         
    sigma_y = get_second_moment(np.arange(data.shape[1]),data[np.int(y0),:],x0)
    sigma_x = get_second_moment(np.arange(data.shape[0]),data[:,np.int(x0)],y0)
    if sigma_y == 0.0:
       sigma = sigma_x
    if sigma_x == 0.0:
       sigma = 0.0
    if sigma_x != 0. and sigma_y != 0.:
       sigma = (sigma_x + sigma_y)/2.
    return 2.*np.sqrt(2.*np.log(2.0))*sigma

def getApertureFluxes(subimg,x_cen,y_cen,Radius,sky_sigma,GAIN):
    apertures = CircularAperture([(x_cen,y_cen)],r=Radius)
    rawflux_table = aperture_photometry(subimg, apertures, \
            error=sky_sigma)#, effective_gain = GAIN)
    return rawflux_table['aperture_sum'][0],rawflux_table['aperture_sum_err'][0]

def CoordsToDecimal(coords):
    ras = np.array([])
    decs = np.array([])
    for i in range(len(coords)):
            ra_string,dec_string = coords[i]
            # Get hour, minutes and secs from RA string:
            hh,mm,ss = ra_string.replace(' ', ':').split(':')
            # Convert to decimal:
            ra_decimal = np.float(hh) + (np.float(mm)/60.) + \
                            (np.float(ss)/3600.0)
            # Convert to degrees:
            ras = np.append(ras,ra_decimal * (360./24.))
            # Now same thing for DEC:
            dd,mm,ss = dec_string.replace(' ', ':').split(':')
            dec_decimal = np.abs(np.float(dd)) + (np.float(mm)/60.) + \
                            (np.float(ss)/3600.0)
            if dd[0] == '-':
                    decs = np.append(decs,-1*dec_decimal)
            else:
                    decs = np.append(decs,dec_decimal)
    return ras,decs

def DecimalToCoords(ra_degs,dec_degs):
    ra_coords = len(ra_degs)*[[]]
    dec_coords = len(dec_degs)*[[]]
    for i in range(len(ra_degs)):
        c_ra = (ra_degs[i]/360.)*24.
        c_dec = dec_degs[i]
        ra_hr = int(c_ra)
        ra_min = int((c_ra - ra_hr)*60.)
        ra_sec = (c_ra - ra_hr - ra_min/60.0)*3600.
        dec_deg = int(c_dec)
        dec_min = int(np.abs(c_dec-dec_deg)*60.)
        dec_sec = (np.abs(c_dec-dec_deg)-dec_min/60.)*3600.
        ra_coords[i] = NumToStr(ra_hr)+':'+NumToStr(ra_min)+':'+NumToStr(ra_sec,roundto=3)
        dec_coords[i] = NumToStr(dec_deg)+':'+NumToStr(dec_min)+':'+NumToStr(dec_sec,roundto=3)
    return ra_coords,dec_coords

import decimal
def NumToStr(number,roundto=None):
    if roundto is not None:
        number = round(decimal.Decimal(str(number)),roundto)
    abs_number = np.abs(number)
    if abs_number < 10:
        str_number = '0'+str(abs_number)
    else:
        str_number = str(abs_number)
    if number < 0:
        return '-'+str_number
    else:
        return str_number

def SuperComparison(fluxes,errors):
        flux = np.sum(fluxes/errors**2)/np.sum(1./errors**2)
        err_flux = np.sqrt(np.sum(errors**2)/np.double(len(fluxes)))
        return flux,err_flux

from astropy import time
from astropy import constants as const
from astropy import units as u
from astropy.utils.iers import IERS
from astropy.utils.iers import IERS_A, IERS_A_URL
from astropy.utils.data import download_file
from astropy.io import ascii
from astropy import coordinates as coord
from astropy import _erfa as erfa
import numpy as np
import warnings

# mean sidereal rate (at J2000) in radians per (UT1) second
SR = 7.292115855306589e-5

''' time class: inherits astropy time object, and adds heliocentric, barycentric
    correction utilities.'''
class Time(time.Time):

    def __init__(self,*args,**kwargs):
        super(Time, self).__init__(*args, **kwargs)
        self.height = kwargs.get('height',0.0)

    def _pvobs(self):
        '''calculates position and velocity of the telescope
           returns position/velocity in AU and AU/d in GCRS reference frame
        '''

        # convert obs position from WGS84 (lat long) to ITRF geocentric coords in AU
        xyz = self.location.to(u.AU).value

        # now we need to convert this position to Celestial Coords
        # specifically, the GCRS coords.
        # conversion from celestial to terrestrial coords given by
        # [TRS] = RPOM * R_3(ERA) * RC2I * [CRS]
        # where:
        # [CRS] is vector in GCRS (geocentric celestial system)
        # [TRS] is vector in ITRS (International Terrestrial Ref System)
        # ERA is earth rotation angle
        # RPOM = polar motion matrix

        tt = self.tt
        mjd = self.utc.mjd

        # we need the IERS values to correct for the precession/nutation of the Earth
        iers_tab = IERS.open()

        # Find UT1, which is needed to calculate ERA
        # uses IERS_B by default , for more recent times use IERS_A download
        try:      
            ut1 = self.ut1 
        except:
            try:
                iers_a_file = download_file(IERS_A_URL, cache=True)
                iers_a = IERS_A.open(iers_a_file)
                self.delta_ut1_utc = self.get_delta_ut1_utc(iers_a)
                ut1 = self.ut1
            except:
                # fall back to UTC with degraded accuracy
                warnings.warn('Cannot calculate UT1: using UTC with degraded accuracy') 
                ut1 = self.utc

        # Gets x,y coords of Celestial Intermediate Pole (CIP) and CIO locator s
        # CIO = Celestial Intermediate Origin
        # Both in GCRS
        X,Y,S = erfa.xys00a(tt.jd1,tt.jd2)

        # Get dX and dY from IERS B
        dX = np.interp(mjd, iers_tab['MJD'], iers_tab['dX_2000A']) * u.arcsec 
        dY = np.interp(mjd, iers_tab['MJD'], iers_tab['dY_2000A']) * u.arcsec

        # Get GCRS to CIRS matrix
        # can be used to convert to Celestial Intermediate Ref Sys
        # from GCRS.
        rc2i = erfa.c2ixys(X+dX.to(u.rad).value, Y+dY.to(u.rad).value, S)

        # Gets the Terrestrial Intermediate Origin (TIO) locator s'
        # Terrestrial Intermediate Ref Sys (TIRS) defined by TIO and CIP.
        # TIRS related to to CIRS by Earth Rotation Angle
        sp = erfa.sp00(tt.jd1,tt.jd2)

        # Get X and Y from IERS B
        # X and Y are
        xp = np.interp(mjd, iers_tab['MJD'], iers_tab['PM_x']) * u.arcsec
        yp = np.interp(mjd, iers_tab['MJD'], iers_tab['PM_y']) * u.arcsec 

        # Get the polar motion matrix. Relates ITRF to TIRS.
        rpm = erfa.pom00(xp.to(u.rad).value, yp.to(u.rad).value, sp)

        # multiply ITRF position of obs by transpose of polar motion matrix
        # Gives Intermediate Ref Frame position of obs
        x,y,z = np.array([rpmMat.T.dot(xyz) for rpmMat in rpm]).T

        # Functions of Earth Rotation Angle, theta
        # Theta is angle bewtween TIO and CIO (along CIP)
        # USE UT1 here.
        theta = erfa.era00(ut1.jd1,ut1.jd2)
        S,C = np.sin(theta),np.cos(theta)

        # Position #GOT HERE
        pos = np.asarray([C*x - S*y, S*x + C*y, z]).T

        # multiply by inverse of GCRS to CIRS matrix
        # different methods for scalar times vs arrays
        if pos.ndim > 1:
            pos = np.array([np.dot(rc2i[j].T,pos[j]) for j in range(len(pos))])
        else:   
            pos = np.dot(rc2i.T,pos)

        # Velocity
        vel = np.asarray([SR*(-S*x - C*y), SR*(C*x-S*y), np.zeros_like(x)]).T
        # multiply by inverse of GCRS to CIRS matrix
        if vel.ndim > 1:
            vel = np.array([np.dot(rc2i[j].T,vel[j]) for j in range(len(pos))])
        else:        
            vel = np.dot(rc2i.T,vel)

        #return position and velocity
        return pos,vel

    def _obs_pos(self):
        '''calculates heliocentric and barycentric position of the earth in AU and AU/d'''
        tdb = self.tdb

        # get heliocentric and barycentric position and velocity of Earth
        # BCRS reference frame
        h_pv,b_pv = erfa.epv00(tdb.jd1,tdb.jd2)

        # h_pv etc can be shape (ntimes,2,3) or (2,3) if given a scalar time
        if h_pv.ndim == 2:
            h_pv = h_pv[np.newaxis,:]
        if b_pv.ndim == 2:
            b_pv = b_pv[np.newaxis,:]

        # unpack into position and velocity arrays
        h_pos = h_pv[:,0,:]
        h_vel = h_pv[:,1,:]

        # unpack into position and velocity arrays
        b_pos = b_pv[:,0,:]
        b_vel = b_pv[:,1,:]

        #now need position and velocity of observing station
        pos_obs, vel_obs = self._pvobs()

        #add this to heliocentric and barycentric position of center of Earth
        h_pos += pos_obs
        b_pos += pos_obs
        h_vel += vel_obs        
        b_vel += vel_obs        
        return (h_pos,h_vel,b_pos,b_vel)

    def _vect(self,coord):
        '''get unit vector pointing to star, and modulus of vector, in AU
           coordinate of star supplied as astropy.coordinate object

           assume zero proper motion, parallax and radial velocity'''
        pmra = pmdec = px = rv = 0.0

        rar  = coord.ra.radian
        decr = coord.dec.radian
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # ignore warnings about 0 parallax
            pos,vel = erfa.starpv(rar,decr,pmra,pmdec,px,rv)

        modulus = np.sqrt(pos.dot(pos))
        unit = pos/modulus
        modulus /= const.au.value
        return modulus, unit

    def hcor(self,coord):
        mod, spos = self._vect(coord)
        # get helio/bary-centric position and velocity of telescope, in AU, AU/d
        h_pos,h_vel,b_pos,b_vel = self._obs_pos()

        # heliocentric light travel time, s
        tcor_hel = const.au.value * np.array([np.dot(spos,hpos) for hpos in h_pos]) / const.c.value
        #print 'Correction to add to get time at heliocentre = %.7f s' % tcor_hel
        dt = time.TimeDelta(tcor_hel, format='sec', scale='tdb')
        return self.utc + dt

    def bcor(self,coord):
        mod, spos = self._vect(coord)
        # get helio/bary-centric position and velocity of telescope, in AU, AU/d
        h_pos,h_vel,b_pos,b_vel  = self._obs_pos()

        # barycentric light travel time, s
        tcor_bar = const.au.value *  np.array([np.dot(spos,bpos) for bpos in b_pos]) / const.c.value
        #print 'Correction to add to get time at barycentre  = %.7f s' % tcor_bar
        dt = time.TimeDelta(tcor_bar, format='sec', scale='tdb')
        return self.tdb + dt

def get_exts(filename):
    """
    Returns a list of the fits extensions containing data
    """
    h = pyfits.getheader(filename)
    try:
        EXTEND = h['EXTEND']
    except KeyError:
        EXTEND = False
    if EXTEND:
        exts = range(1, h['NEXTEND']+1)
    else:
        exts = [0]
    return exts

from astroquery.simbad import Simbad
Simbad.add_votable_fields('propermotions')
def get_general_coords(target,date):
    """
    Given a target name, returns RA and DEC from simbad.
    """
    date = date.replace('-', '')
    try:
        # Try to get info from Simbad
#         print('\t Getting coordinates for target: {:}'.format(target))
        result = Simbad.query_object(target)
    except:
        # Manually load values
        coords_file = open('../manual_object_coords.dat','r')
        while True:
            line = coords_file.readline()
            if line != '':
                name,ra,dec = line.split()
                if name.lower() == target.lower():
                    coords_file.close()
                    return ra,dec
            else:
                break
        coords_file.close()
        # As a last resort:
        return 'NoneFound','NoneFound'
    else:
        # Assuming the Simbad query worked, load the coordinates:
        # Load positions as strings
        if result is None:
            print("WARNING: Simbad query for {0} failed!".format(target))
        rahh, ramm, rass = result['RA'][0].split()
        decdd, decmm, decss = result['DEC'][0].split()
        # Load proper motions as arcsec / year
        pmra = result['PMRA'].to(u.arcsec/u.year).value[0]
        pmdec = result['PMDEC'].to(u.arcsec/u.year).value[0]
#         print('\t\t proper motion: {:0.3f}, {:0.3f} arcsec/yr'.format(pmra, pmdec))
        # Convert RA and DEC to whole numbers:
        ra = np.double(rahh)+(np.double(ramm)/60.)+(np.double(rass)/3600.)
        if np.double(decdd)<0:
            dec = np.double(decdd)-(np.double(decmm)/60.)-(np.double(decss)/3600.)
        else:
            dec = np.double(decdd)+(np.double(decmm)/60.)+(np.double(decss)/3600.) 
        # Calculate time difference from J2000:
        year = int(date[:4])
        month = int(date[4:6])
        day = int(date[6:8])
        s = str(year)+'.'+str(month)+'.'+str(day)
        dt = dateutil.parser.parse(s)
        data_jd = sum(jdcal.gcal2jd(dt.year, dt.month, dt.day))
        deltat = (data_jd-2451544.5)/365.25
        # Calculate total PM:
#         pmra = np.double(pmra)*deltat # <- This works for AZ Cnc, LSPM J1403+3007
        pmra = np.double(pmra)*deltat/15. # Conversion from arcsec to sec <- This works for GJ 1214, TRAPPIST-1
        pmdec = np.double(pmdec)*deltat
        # Correct proper motion:
        c_ra = ra + ((pmra)/3600.)
        c_dec = dec + ((pmdec)/3600.)
        # Return RA and DEC:
        ra_hr = int(c_ra)
        ra_min = int((c_ra - ra_hr)*60.)
        ra_sec = (c_ra - ra_hr - ra_min/60.0)*3600.
        dec_deg = int(c_dec)
        dec_min = int(np.abs(c_dec-dec_deg)*60.)
        dec_sec = (np.abs(c_dec-dec_deg)-dec_min/60.)*3600.
        return NumToStr(ra_hr)+':'+NumToStr(ra_min)+':'+NumToStr(ra_sec,roundto=3),\
               NumToStr(dec_deg)+':'+NumToStr(dec_min)+':'+NumToStr(dec_sec,roundto=3)

import re
def get_suffix(s):
    m = re.match('.+([0-9])[^0-9]*$', s)
    idx = m.start(1)+1
    return s[idx:]

def get_trailing_number(s):
    m = re.search(r'\d+$', s)
    return int(m.group()) if m else None