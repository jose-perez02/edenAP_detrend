import multiprocessing as mp
import os
import re
import signal
import subprocess
import sys
import time as clocking_time
import warnings
from configparser import ConfigParser
from math import modf

# Use for UTC -> BJD conversion:
import dateutil
import ephem as E
import jdcal
import matplotlib
import numpy as np
from astropy import _erfa as erfa
from astropy import constants as const
from astropy import coordinates as coord
from astropy import time
from astropy import units as u
from astropy import wcs
from astropy.coordinates import SkyCoord, Angle
from astropy.io import fits
from astropy.stats import gaussian_sigma_to_fwhm
from astropy.stats import sigma_clipped_stats
from astropy.utils.data import download_file
from astropy.utils.iers import IERS
from astropy.utils.iers import IERS_A, IERS_A_URL
from astroquery.irsa import Irsa
from astroquery.simbad import Simbad
from photutils import CircularAperture, aperture_photometry
from photutils import DAOStarFinder
from photutils import make_source_mask
from photutils.centroids import centroid_com
from scipy import optimize
from scipy.ndimage.filters import gaussian_filter

# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from constants import log, natural_keys, LOOKDATE, find_val

# define constants from config.ini
config = ConfigParser()
config.read('config.ini')
server_destination = config['FOLDER OPTIONS']['server_destination']
fpack_folder = config['FOLDER OPTIONS']['funpack']
astrometry_directory = config['FOLDER OPTIONS']['astrometry']
manual_object_coords = config['Manual Coords']

# Ignore computation errors:
np.seterr(divide='ignore', invalid='ignore')

# Define style of plotting (ggplot is nicer):
plt.style.use('ggplot')

# Default limit of rows is 500. Go for infinity and beyond!
Irsa.ROW_LIMIT = np.inf

# mean sidereal rate (at J2000) in radians per (UT1) second
SR = 7.292115855306589e-5

# configure simbad query
Simbad.add_votable_fields('propermotions')

# define globals
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
sigma_gf = 5. * fwhm_factor  # 5.


def date_format(year, month, day, hour, minute, second):
    def fn(number):
        if number > 9:
            return str(number)
        else:
            return '0' + str(number)

    return fn(year) + '-' + fn(month) + '-' + fn(day) + 'T' + \
           fn(hour) + ':' + fn(minute) + ':' + fn(second)


def TrimLim(s):
    cols, rows = s[1:-1].split(',')
    min_col, max_col = cols.split(':')
    min_row, max_row = rows.split(':')
    return int(min_col), int(max_col), int(min_row), int(max_row)


def site_data_2_string(sitelong, sitelat, sitealt):
    basestring = str
    if not isinstance(sitelong, basestring):
        longitude = str(int(modf(360. - sitelong)[1])) + ':' + str(modf(360. - sitelong)[0] * 60.)
    else:
        longitude = sitelong
    if not isinstance(sitelat, basestring):
        latitude = str(int(modf(sitelat)[1])) + ':' + str(-modf(sitelat)[0] * 60.)
    else:
        latitude = sitelat
    if not isinstance(sitealt, basestring):
        return longitude, latitude, str(sitealt)
    else:
        return longitude, latitude, sitealt


def getCalDay(JD):
    year, month, day, hour = jdcal.jd2gcal(JD, 0.0)
    hour = hour * 24
    minutes = modf(hour)[0] * 60.0
    seconds = modf(minutes)[0] * 60.0
    hh = int(modf(hour)[1])
    mm = int(modf(minutes)[1])
    ss = seconds
    if (hh < 10):
        hh = '0' + str(hh)
    else:
        hh = str(hh)
    if (mm < 10):
        mm = '0' + str(mm)
    else:
        mm = str(mm)
    if (ss < 10):
        ss = '0' + str(np.round(ss, 1))
    else:
        ss = str(np.round(ss, 1))
    return year, month, day, hh, mm, ss


def getTime(year, month, day, hh, mm, ss):
    return str(year) + '/' + str(month) + '/' + str(day) + ' ' + hh + ':' + mm + ':' + ss


def getAirmass(ra, dec, day, longitude, latitude, elevation):
    """
    Get airmass given the RA, DEC of object in sky
    longitude, latitude and elevation of site.
    :param ra: RA of object in decimal degrees or string in format HMS
    :param dec: Dec of object in decimal degrees or string in format DMS
    :param day: date time string or object; e.g. '2018-05-29 05:20:30.5'
    :param longitude: longitude in degrees east of Greenwich ( East +, West - )
    :param latitude: longitude in degrees north of the Equator ( North +, South -)
    :param elevation: Elevation above sea level in meters
    :return: airmass of celestial object


    Exampple: Kuiper Telescope    Target: 2MASSI J183579+325954   Airmass: 1.01 (header)
    >> sitelong = -110.73453
    >> sitelat = 32.41647
    >> sitealt = 2510.
    >> Day = '2018-07-19 05:43:20.411'
    >> Dec = '+32:59:54.5'
    >> RA = '18:35:37.92'
    >> getAirmass(RA, Dec, Day, sitelong, sitelat, sitealt)
    Out[1]: 1.005368864146983

    """
    star = E.FixedBody()
    star._ra = ra if isinstance(ra, str) else ra * E.degree
    star._dec = dec if isinstance(dec, str) else dec * E.degree

    observer = E.Observer()
    observer.date = day
    # convert longitude and latitude to radians with E.degree
    observer.long = E.degree * longitude
    observer.lat = E.degree * latitude

    observer.elevation = elevation
    observer.temp = 7.0
    observer.pressure = 760.0

    star.compute(observer)
    # star.alt: altitude of object in radians
    # h = altitude of object in degrees
    # factor is a corrected altitude for more accurate airmass (in radians)
    h = star.alt * 180 / np.pi
    factor = h + E.degree * 244. / (165. + 47. * (h ** 1.1))
    airmass = 1. / np.sin(factor * np.pi / 180.)
    return airmass


def get_planet_data(planet_data, target_object_name):
    f = open(planet_data, 'r')
    while True:
        line = f.readline()
        if line == '':
            break
        elif line[0] != '#':
            name, ra, dec = line.split()
            print(name, ra, dec)
            if target_object_name == name:
                f.close()
                return [[ra, dec]]
    f.close()
    print('Planet ' + target_object_name + ' not found in file ' + planet_data)
    print('Add it to the list and try running the code again.')
    sys.exit()


def get_dict(target, central_ra, central_dec, central_radius, ra_obj, dec_obj,
             hdulist, exts, R, catalog=u'fp_psc', date=dateutil.parser.parse('20180101')):
    print('\t > Generating master dictionary for coordinates', central_ra, central_dec, '...')
    # Make query to 2MASS:
    result = Irsa.query_region(coord.SkyCoord(central_ra, central_dec, unit=(u.deg, u.deg)),
                               spatial='Cone', radius=central_radius * 5400. * u.arcsec, catalog=catalog)
    # Query to PPMXL to get proper motions:
    resultppm = Irsa.query_region(coord.SkyCoord(central_ra, central_dec, unit=(u.deg, u.deg)),
                                  spatial='Cone', radius=central_radius * 5400. * u.arcsec, catalog='ppmxl')

    # Get RAs, DECs, and PMs from this last catalog:
    rappmxl = resultppm['ra'].data.data
    decppmxl = resultppm['dec'].data.data
    rappm = resultppm['pmra'].data.data
    decppm = resultppm['pmde'].data.data

    # Save coordinates, magnitudes and errors on the magnitudes:
    all_ids = result['designation'].data.data.astype(str)
    all_ra = result['ra'].data.data
    all_dec = result['dec'].data.data
    all_j = result['j_m'].data.data
    all_j_err = result['j_msigcom'].data.data
    all_k = result['k_m'].data.data
    all_k_err = result['k_msigcom'].data.data
    all_h = result['h_m'].data.data
    all_h_err = result['h_msigcom'].data.data
    # Correct RA and DECs for PPM. First, get delta T:
    data_jd = sum(jdcal.gcal2jd(date.year, date.month, date.day))
    deltat = (data_jd - 2451544.5) / 365.25
    print('\t Correcting PPM for date ', date, ', deltat: ', deltat, '...')
    for i in range(len(all_ra)):
        c_ra = all_ra[i]
        c_dec = all_dec[i]
        dist = (c_ra - rappmxl) ** 2 + (c_dec - decppmxl) ** 2
        min_idx = np.argmin(dist)
        # 3 arcsec tolerance:
        if dist[min_idx] < 3. / 3600.:
            all_ra[i] = all_ra[i] + deltat * rappm[min_idx]
            all_dec[i] = all_dec[i] + deltat * decppm[min_idx]
    print('\t Done.')

    # Check which ras and decs have valid coordinates inside the first image...
    # ...considering all image extensions.
    # Save only those as valid objects for photometry:
    idx = []
    all_extensions = np.array([])
    for ext in exts:
        h = hdulist[ext].header
        x_max = hdulist[ext].data.shape[1]
        y_max = hdulist[ext].data.shape[0]
        x, y = SkyToPix(h, all_ra, all_dec)
        for i in range(len(x)):
            if 0 < x[i] < x_max and 0 < y[i] < y_max:
                idx.append(i)
                all_extensions = np.append(all_extensions, ext)
    assert len(idx) > 0, "Indeces list for reference stars could not be generated while creating MasterDict"
    # Create dictionary that will save all the data:
    log('CREATING DICTIONARY FOR THE FIRST TIME. CREATING KEYS')
    master_dict = {}
    master_dict['frame_name'] = np.array([])
    master_dict['UTC_times'] = np.array([])
    master_dict['BJD_times'] = np.array([])
    master_dict['JD_times'] = np.array([])
    master_dict['LST'] = np.array([])
    master_dict['exptimes'] = np.array([])
    master_dict['airmasses'] = np.array([])
    master_dict['filters'] = np.array([])
    master_dict['source'] = np.array([])

    # Generate a flux dictionary for each target.
    master_dict['data'] = {}
    master_dict['data']['RA_degs'] = all_ra[idx]
    master_dict['data']['IDs'] = all_ids[idx]
    master_dict['data']['DEC_degs'] = all_dec[idx]
    master_dict['data']['RA_coords'], master_dict['data']['DEC_coords'] = DecimalToCoords(all_ra[idx], all_dec[idx])
    master_dict['data']['ext'] = all_extensions
    master_dict['data']['Jmag'] = all_j[idx]
    master_dict['data']['Jmag_err'] = all_j_err[idx]
    master_dict['data']['Kmag'] = all_k[idx]
    master_dict['data']['Kmag_err'] = all_k_err[idx]
    master_dict['data']['Hmag'] = all_h[idx]
    master_dict['data']['Hmag_err'] = all_h_err[idx]

    all_names = len(idx) * ['']

    # Get index of target star:
    distances = (all_ra[idx] - ra_obj) ** 2 + (all_dec[idx] - dec_obj) ** 2
    target_idx = np.argmin(distances)

    # Dictionaries per reference star: centroids_x, centroids_y, background, background_err, fwhm
    # A fluxes_{aperture}_pix_ap keyword per aperture as well
    for i in range(len(idx)):
        if i != target_idx:
            all_names[i] = 'star_' + str(i)
        else:
            all_names[i] = 'target_star_' + str(i)
            # Replace RA and DEC with the ones given by the user:
            ra_str, dec_str = get_general_coords(target, date)
            ra_deg, dec_deg = CoordsToDecimal([[ra_str, dec_str]])
            master_dict['data']['RA_degs'][i] = ra_deg
            master_dict['data']['DEC_degs'][i] = dec_deg
            master_dict['data']['RA_coords'][i] = ra_str
            master_dict['data']['DEC_coords'][i] = dec_str
        master_dict['data'][all_names[i]] = {}
        master_dict['data'][all_names[i]]['centroids_x'] = np.array([])
        master_dict['data'][all_names[i]]['centroids_y'] = np.array([])
        master_dict['data'][all_names[i]]['background'] = np.array([])
        master_dict['data'][all_names[i]]['background_err'] = np.array([])
        master_dict['data'][all_names[i]]['fwhm'] = np.array([])
        for r in R:
            master_dict['data'][all_names[i]]['fluxes_' + str(r) + '_pix_ap'] = np.array([])
            master_dict['data'][all_names[i]]['fluxes_' + str(r) + '_pix_ap_err'] = np.array([])

    master_dict['data']['names'] = np.array(all_names)
    print('\t > Extracting data for ' + str(len(all_names)) + ' sources')
    return master_dict


def getPhotometry(filenames, target: str, telescope: str, filters, R, ra_obj, dec_obj, out_data_folder, use_filter: str,
                  get_astrometry=True, sitelong=None, sitelat=None, sitealt=None, refine_cen=False,astrometry_timeout=30,master_dict=None):
                      
    # Define radius in which to search for targets (in degrees):
    search_radius = 0.25

    # Initiallize empty dictionary if not saved and different variables:
    longitude, latitude, elevation = None, None, None
    if master_dict is None:
        master_dict = {}
        updating_dict = False
    else:
        updating_dict = True

    if telescope == 'SWOPE':
        long_h_name = 'SITELONG'
        lat_h_name = 'SITELAT'
        alt_h_name = 'SITEALT'
        exptime_h_name = 'EXPTIME'
        lst_h_name = None
        t_scale_low = 0.4
        t_scale_high = 0.45
        egain = 2.3

    elif telescope == 'CHAT':
        long_h_name = 'SITELONG'
        lat_h_name = 'SITELAT'
        alt_h_name = 'SITEALT'
        exptime_h_name = 'EXPTIME'
        lst_h_name = 'LST'
        t_scale_low = 0.6
        t_scale_high = 0.65
        egain = 1.0

    elif telescope == 'LCOGT':
        # This data is good for LCOGT 1m.
        long_h_name = 'LONGITUD'
        lat_h_name = 'LATITUDE'
        alt_h_name = 'HEIGHT'
        exptime_h_name = 'EXPTIME'
        lst_h_name = 'LST'
        t_scale_low = 0.3
        t_scale_high = 0.5
        egain = 'GAIN'

    elif telescope == 'SMARTS':
        long_h_name = 'LONGITUD'
        lat_h_name = 'LATITUDE'
        alt_h_name = 'ALTITIDE'
        exptime_h_name = 'EXPTIME'
        lst_h_name = 'ST'
        t_scale_low = 0.1
        t_scale_high = 0.4
        egain = 3.2

    elif telescope == 'OBSUC':
        long_h_name = None
        sitelong = 289.4656
        sitelat = -33.2692
        sitealt = 1450.
        lat_h_name = None
        alt_h_name = None
        exptime_h_name = 'EXPTIME'
        lst_h_name = None
        t_scale_low = 0.2
        t_scale_high = 0.8
        egain = 'EGAIN'

    elif telescope == 'NTT':
        long_h_name = 'HIERARCH ESO TEL GEOLON'
        lat_h_name = 'HIERARCH ESO TEL GEOLAT'
        alt_h_name = 'HIERARCH ESO TEL GEOELEV'
        exptime_h_name = 'HIERARCH ESO DET WIN1 DIT1'
        lst_h_name = None
        t_scale_low = 0.2
        t_scale_high = 0.8
        egain = 'HIERARCH ESO DET OUT1 GAIN'

    elif telescope == 'KUIPER':
        long_h_name = None
        lat_h_name = None
        alt_h_name = None
        sitelong = -110.73453
        sitelat = 32.41647
        sitealt = 2510.
        exptime_h_name = 'EXPTIME'
        lst_h_name = 'LST-OBS'
        t_scale_low = 0.145
        t_scale_high = 0.145 * 4
        egain = 3.173

    elif telescope == 'SCHULMAN':
        long_h_name = 'LONG-OBS'
        lat_h_name = 'LAT-OBS'
        alt_h_name = 'ALT-OBS'
        exptime_h_name = 'EXPTIME'
        lst_h_name = 'ST'
        t_scale_low = 0.25
        t_scale_high = 0.35
        egain = 1.28

    elif telescope == 'VATT':
        long_h_name = None
        lat_h_name = None
        alt_h_name = None
        sitelong = -109.892014
        sitelat = 32.701303
        sitealt = 3191.
        exptime_h_name = 'EXPTIME'
        lst_h_name = 'ST'
        t_scale_low = 0.185
        t_scale_high = 0.188 * 4
        egain = 1.9  # 'DETGAIN'

    elif telescope == 'BOK':
        long_h_name = None
        lat_h_name = None
        alt_h_name = None
        sitelong = -111.6
        sitelat = 31.98
        sitealt = 2120.
        exptime_h_name = 'EXPTIME'
        lst_h_name = 'ST'
        t_scale_low = 0.1
        t_scale_high = 0.4 * 4
        egain = 1.5
    elif telescope == 'CASSINI':
        long_h_name = None
        lat_h_name = None
        alt_h_name = None
        sitelong = 11.3
        sitelat = 44.3
        sitealt = 785.
        exptime_h_name = 'EXPTIME'
        lst_h_name = 'ST'
        t_scale_low = 0.55
        t_scale_high = 0.58 * 3
        egain = 2.22
    elif telescope == 'CAHA':
        long_h_name = None
        lat_h_name = None
        alt_h_name = None
        sitelong = 2.55
        sitelat = 37.2
        sitealt = 2168.
        exptime_h_name = 'EXPTIME'
        lst_h_name = 'LST'
        t_scale_low = 0.31
        t_scale_high = 0.3132 * 3
        egain = 'GAIN'
    elif telescope == 'GUFI':
        long_h_name = None
        lat_h_name = None
        alt_h_name = None
        sitelong = -109.892014
        sitelat = 32.701303
        sitealt = 3191.
        exptime_h_name = 'EXPTIME'
        lst_h_name = None
        t_scale_low = 0.05  # wrong
        t_scale_high = 0.35 * 3  # wrong
        egain = 'GAIN'
    elif telescope == 'LOT':
        long_h_name = None
        lat_h_name = None
        alt_h_name = None
        sitelong = 120.873611
        sitelat = 23.468611
        sitealt = 2862
        exptime_h_name = 'EXPTIME'
        lst_h_name = None
        t_scale_low = 0.375  # This number wasn't taken from specs, it was a matching value from astrometry
        t_scale_high = 0.375 * 3
        egain = 'GAIN'
    else:
        print('ERROR: the selected telescope %s is not supported.' % telescope)
        sys.exit()
    
    # Iterate through the files:
    first_time = True
    # print('reach iteration through the files')
    for f in filenames:
        # print('iterating through files')
        # Decompress file. Necessary because Astrometry cant handle this:
        if f[-7:] == 'fits.fz':
            p = subprocess.Popen(fpack_folder + 'funpack ' + f, stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE, shell=True)
            p.wait()
            if p.returncode != 0 and p.returncode != None:
                print('\t Funpack command failed. The error was:')
                out, err = p.communicate()
                print(err)
                print('\n\t Exiting...\n')
                sys.exit()
            f = f[:-3]
        # Try opening the fits file (might be corrupt):
        try:
            hdulist = fits.open(f)
            fitsok = True
        except Exception as e:
            print('\t Encountered error opening {:}'.format(f))
            log(str(e))
            fitsok = False
        # If frame already reduced, skip it:
        if updating_dict:
            if 'frame_name' in master_dict.keys() and f.replace(server_destination,'') in master_dict['frame_name']:
                fitsok = False
        if fitsok:
            h0 = hdulist[0].header  # primary fits header
            exts = get_exts(hdulist)
            # Check filter:
            filter_ok = False
            if use_filter is None:
                filter_ok = True
            else:
                temp_filter: str = find_val(h0, 'FILTER', typ=str)
                if use_filter.lower() == temp_filter.lower():
                    filter_ok = True
            log("FILTER_OK: %r\t%s" % (filter_ok, f))
            if filter_ok:
                print('\t Working on frame ' + f)
                ########## OBTAINING THE ASTROMETRY ###############
                # First, run astrometry on the current frame if not ran already:
                filename = f.split('.fits')[0]
                # astrometry will save the file in the /REDUCED/ folder
                if f.startswith(server_destination):
                    wcs_filepath = f.replace('/CALIBRATED/', '/REDUCED/').replace('/RAW/','/REDUCED/')
                    wcs_filepath = os.path.join(os.path.dirname(wcs_filepath),
                                                'wcs_fits/',
                                                os.path.basename(wcs_filepath))
                    wcs_filepath = wcs_filepath.replace('.fits', '.wcs.fits')
                    
                    if not os.path.exists(os.path.dirname(wcs_filepath)):
                        os.makedirs(os.path.dirname(wcs_filepath))
                else:
                    wcs_filepath = f.replace('.fits', '.wcs.fits')
                if not os.path.exists(wcs_filepath) and get_astrometry:
                    print('\t Calculating astrometry...')
                    run_astrometry(f, exts,ra=ra_obj[0], dec=dec_obj[0], radius=0.5,
                                   scale_low=t_scale_low, scale_high=t_scale_high, astrometry_timeout=astrometry_timeout)
                    print('\t ...done!')
                # Now get data if astrometry worked...
                if os.path.exists(wcs_filepath) and get_astrometry:
                    # If get_astrometry flag is on, load the header from the WCS-solved image onto the input data
                    # (Don't just use the WCS solved image; this way we can change between RAW and CALIBRATED input types
                    # without having to redo the astrometry)
                    print('\t Detected file ' + wcs_filepath + '. Using it...')
                    #hdulist = fits.open(f)
                    hdulist_wcs = fits.open(wcs_filepath)
                    for ext in exts:
                        hdulist[ext].header = hdulist_wcs[ext].header
                    hdulist_wcs.close()
                    
                # ...or if no astrometry was needed on the frame:
                elif not get_astrometry:
                    continue # hdulist = fits.open(f)
                # or if the astrometry ultimately failed... skip file and enter None values in the dict
                else:
                    print("\t Skipping file " + f)
                    continue
                # Create master dictionary and define data to be used in the code:    
                if first_time:
                    date_time = LOOKDATE(h0)  # datetime object
                    try:
                        central_ra, central_dec = CoordsToDecimal([[find_val(h0, 'RA'),
                                                                    find_val(h0, 'DEC')]])
                    except ValueError:
                        # If there's no RA, Dec in the header, just use the target RA, Dec
                        central_ra, central_dec = ra_obj, dec_obj
                    
                    if not updating_dict:
                        master_dict = get_dict(target, central_ra[0], central_dec[0], search_radius,
                                               ra_obj, dec_obj, hdulist, exts, R, date=date_time.date())
                    else:
                        all_data = master_dict['data'].keys()
                        all_names_d = []
                        all_idx = []
                        for dataname in all_data:
                            if 'star' in dataname:
                                all_names_d.append(dataname)
                                all_idx.append(int(dataname.split('_')[-1]))
                        idxs = np.argsort(all_idx)
                        all_names = len(all_names_d) * [[]]
                        for i in range(len(idxs)):
                            all_names[i] = all_names_d[idxs[i]]

                    if sitelong is None:
                        sitelong = h0[long_h_name]
                    if sitelat is None:
                        sitelat = h0[lat_h_name]
                    if sitealt is None:
                        sitealt = h0[alt_h_name]
                    first_time = False

                # Save filename to master dictionary
                # Remove the server root; this way it works on multiple computers with different mount points
                log('SETTING A FILENAME TO frame_name key in master_dict\t %s' % f.replace(server_destination,''))
                master_dict['frame_name'] = np.append(master_dict['frame_name'], f.replace(server_destination,''))

                ########## OBTAINING THE TIMES OF OBSERVATION ####################
                # Get the BJD time. First, add exposure time:
                utc_time = LOOKDATE(h0)
                t_center = utc_time + dateutil.relativedelta.relativedelta(seconds=h0[exptime_h_name] / 2.)
                t = Time(t_center, scale='utc', location=(str(sitelong) + 'd', str(sitelat) + 'd', sitealt))
                RA = find_val(h0, 'RA')
                DEC = find_val(h0, 'DEC')
                try:
                    # the purpose of this is to see if values are floats...therefore degrees
                    float(RA)
                    coords = SkyCoord(str(RA) + ' ' + str(DEC), unit=(u.deg, u.deg))
                except ValueError:
                    # if there is a colon in the values, assume sexagesimal
                    if ':' in RA and ':' in DEC:
                        coords = SkyCoord(RA + ' ' + DEC, unit=(u.hourangle, u.deg))
                    # Otherwise use the target coordinates
                    else:
                        coords = SkyCoord(str(ra_obj[0]) + ' ' + str(dec_obj[0]), unit=(u.deg,u.deg))
                # Save UTC, exposure, JD and BJD and LS times. Also airmass and filter used.
                master_dict['UTC_times'] = np.append(master_dict['UTC_times'], str(utc_time).replace(' ', 'T'))
                master_dict['exptimes'] = np.append(master_dict['exptimes'], h0[exptime_h_name])
                master_dict['JD_times'] = np.append(master_dict['JD_times'], t.jd)
                master_dict['BJD_times'] = np.append(master_dict['BJD_times'], ((t.bcor(coords)).jd))
                if lst_h_name is not None:
                    master_dict['LST'] = np.append(master_dict['LST'], h0[lst_h_name])
                else:
                    t.delta_ut1_utc = 0.
                    c_lst = str(t.sidereal_time('mean', 'greenwich'))
                    c_lst = c_lst.split('h')
                    hh = c_lst[0]
                    c_lst = c_lst[1].split('m')
                    mm = c_lst[0]
                    ss = (c_lst[1])[:-1]
                    master_dict['LST'] = np.append(master_dict['LST'], hh + ':' + mm + ':' + ss)
                # Calculate Accurate Airmass
                year, month, day, hh, mm, ss = getCalDay((t.bcor(coords)).jd)
                day = getTime(year, month, day, hh, mm, ss)
                master_dict['airmasses'] = np.append(master_dict['airmasses'],
                                                     getAirmass(central_ra[0], central_dec[0], day, sitelong,
                                                                sitelat, sitealt))
                
                # Save the filters
                master_dict['filters'] = np.append(master_dict['filters'],filters)

                # Save the data source (RAW or CALIBRATED)
                source = 'RAW' if '/RAW/' in f[0] else 'CALIBRATED' if '/CALIBRATED/' in f[0] else 'unknown'
                master_dict['source'] = np.append(master_dict['source'],source)
                
                ########## OBTAINING THE FLUXES ###################
                for ext in exts:
                    # Load the data:
                    h = hdulist[ext].header
                    data = hdulist[ext].data
                    # Get the indices of stars on this extension
                    idx = np.where(master_dict['data']['ext'] == ext)
                    if not np.any(idx):
                        continue
                    # Get the names of stars on this extension
                    names_ext = master_dict['data']['names'][idx]
                    x, y = SkyToPix(h, master_dict['data']['RA_degs'][idx], master_dict['data']['DEC_degs'][idx])
                    # Get fluxes of all the targets in this extension for different apertures:
                    print('\t Performing aperture photometry on objects...')
                    tic = clocking_time.time()
                    if isinstance(egain, str):
                        fluxes, errors, x_ref, y_ref, bkg, bkg_err, fwhm = getAperturePhotometry(data, h, x, y, R,
                                                                                                 names_ext,
                                                                                                 frame_name=
                                                                                                 filename.split('/')[
                                                                                                     -1],
                                                                                                 out_dir=out_data_folder,
                                                                                                 GAIN=h[egain],
                                                                                                 saveplot=False,
                                                                                                 refine_centroids=refine_cen)
                    else:
                        fluxes, errors, x_ref, y_ref, bkg, bkg_err, fwhm = getAperturePhotometry(data, h, x, y, R,
                                                                                                 names_ext,
                                                                                                 frame_name=
                                                                                                 filename.split('/')[
                                                                                                     -1],
                                                                                                 out_dir=out_data_folder,
                                                                                                 GAIN=egain,
                                                                                                 saveplot=False,
                                                                                                 refine_centroids=refine_cen)
                    toc = clocking_time.time()
                    print('\t Took {:1.2f} seconds.'.format(toc - tic))
                    # Save everything in the dictionary:
                    for i in range(len(names_ext)):
                        extended_centroidsx = np.append(master_dict['data'][names_ext[i]]['centroids_x'], x_ref[i])
                        extended_centroidsy = np.append(master_dict['data'][names_ext[i]]['centroids_y'], y_ref[i])
                        extended_background = np.append(master_dict['data'][names_ext[i]]['background'], bkg[i])
                        extended_background_err = np.append(master_dict['data'][names_ext[i]]['background_err'],
                                                            bkg_err[i])
                        extended_fwhm = np.append(master_dict['data'][names_ext[i]]['fwhm'], fwhm[i])

                        master_dict['data'][names_ext[i]]['centroids_x'] = extended_centroidsx
                        master_dict['data'][names_ext[i]]['centroids_y'] = extended_centroidsy
                        master_dict['data'][names_ext[i]]['background'] = extended_background
                        master_dict['data'][names_ext[i]]['background_err'] = extended_background_err
                        master_dict['data'][names_ext[i]]['fwhm'] = extended_fwhm
                        
                        for j in range(len(R)):
                            idx_fluxes = 'fluxes_%d_pix_ap' % R[j]
                            idx_fluxes_err = 'fluxes_%d_pix_ap_err' % R[j]
                            this_flux = fluxes[i, j]
                            this_flux_err = errors[i, j]

                            # quick test for nans
                            test = np.append(this_flux, this_flux_err)
                            if np.isnan(np.sum(test)):
                                log("ALERT: During Saving Aperture Photometry Properties:"
                                    " Fluxes and Fluxes Err contain NaNs. Details:")
                                log("names_ext: %s\nidx_fluxes: %s" % (names_ext[i], idx_fluxes))

                            extended_idx_fluxes = np.append(master_dict['data'][names_ext[i]][idx_fluxes], this_flux)
                            extended_idx_fluxes_err = np.append(master_dict['data'][names_ext[i]][idx_fluxes_err],
                                                                this_flux_err)

                            master_dict['data'][names_ext[i]][idx_fluxes] = extended_idx_fluxes
                            master_dict['data'][names_ext[i]][idx_fluxes_err] = extended_idx_fluxes_err

    return master_dict


def organize_files(files, obj_name, filt, leaveout=''):
    dome_flats = []
    sky_flats = []
    bias = []
    objects = []
    all_objects = len(files) * [[]]
    unique_objects = []
    for i in range(len(files)):
        try:
            with fits.open(files[i]) as hdulist:
                d, h = hdulist[1].data, hdulist[0].header
        #             d,h = fits.getdata(files[i],header=True)
        except:
            print('File ' + files[i] + ' probably corrupt. Skipping it')
            if i + 1 == len(files):
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
            unique_objects.append('Bias')  # h['OBJECT'])

    print('\t We found the following frames:')
    for i in range(len(unique_objects)):
        counter = 0
        for obj in all_objects:
            if obj == unique_objects[i]:
                counter = counter + 1
        print('\t   (' + str(i) + ') ' + unique_objects[i] + ' (' + str(counter) + ' frames)')

    print('\t Which ones are the (separate your selection with commas, e.g., 0,3,4)...')
    idx_biases = [int(i) for i in input('\t ...biases?').split(',')]
    idx_dome_flats = [int(i) for i in input('\t ...dome flats?').split(',')]
    idx_sky_flats = [int(i) for i in input('\t ...sky flats?').split(',')]
    idx_science = [int(i) for i in input('\t ...science frames?').split(',')]

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

    return bias, dome_flats, sky_flats, objects


def NormalizeFlat(MasterFlat):
    original_shape = MasterFlat.shape
    flattened_Flat = MasterFlat.flatten()
    median_f = np.median(flattened_Flat)
    idx = np.where(flattened_Flat == 0)
    flattened_Flat = flattened_Flat / median_f
    flattened_Flat[idx] = np.ones(len(idx))
    return flattened_Flat.reshape(original_shape)


def MedianCombine(ImgList, MB=None, flatten_counts=False):
    n = len(ImgList)
    if n == 0:
        raise ValueError("empty list provided!")

    with fits.open(ImgList[0]) as hdulist:
        d, h = hdulist[1].data, hdulist[1].header
    #     data,h = fits.getdata(ImgList[0],header=True)
    datasec1, datasec2 = h['DATASEC'][1:-1].split(',')
    ri, rf = datasec2.split(':')
    ci, cf = datasec1.split(':')
    data = d[int(ri) - 1:int(rf), int(ci) - 1:int(cf)]

    factor = 1.25
    if (n < 3):
        factor = 1

    ronoise = factor * h['ENOISE'] / np.sqrt(n)
    gain = h['EGAIN']

    if (n == 1):
        if h['EXPTIME'] > 0:
            texp = h['EXPTIME']
        else:
            texp = 1.
        if flatten_counts:
            return ((data - MB) / texp) / np.median((data - MB) / texp), ronoise, gain
        else:
            return data / texp, ronoise, gain
    else:
        for i in range(n - 1):
            with fits.open(ImgList[i + 1]) as hdulist:
                d, h = hdulist[1].data, hdulist[1].header
            #             d,h = fits.getdata(ImgList[i+1],header=True)
            datasec1, datasec2 = h['DATASEC'][1:-1].split(',')
            ri, rf = datasec2.split(':')
            ci, cf = datasec1.split(':')
            d = d[int(ri) - 1:int(rf), int(ci) - 1:int(cf)]
            if h['EXPTIME'] > 0:
                texp = h['EXPTIME']
            else:
                texp = 1.
            if flatten_counts:
                data = np.dstack((data, ((d - MB) / texp) / np.median((d - MB) / texp)))
            else:
                data = np.dstack((data, d / texp))
        return np.median(data, axis=2), ronoise, gain


def run_astrometry(filename,exts, ra=None, dec=None, radius=0.5, scale_low=0.1, scale_high=1., astrometry_timeout = 30):
    """
    This code runs Astrometry.net on a frame.

    * ra and dec:     are guesses of the ra and the dec of the center of the field (in degrees).

    * radius:     radius (in degrees) around the input guess ra,dec that astrometry should look for.

    * scale_[low, high]:     are scale limits (arcsec/pix) for the image.
    
    * astrometry_timeout:   maximum number of seconds to run astrometry.net (per attempt)
    """
    # flags
    success = False
    # server work is a flag to work on Project's directory structure
    server_work = True if filename.startswith(server_destination) else False

    true_filename = filename
    print('\t\t Found {:} extensions'.format(len(exts)))

    # setup gf_filepath for gaussian filtered file and final WCS filepath
    # MOD: JOSE. Save gf,wcs,etc files to /red/*/*/*/wcs_fits folder
    if server_work:
        filename = filename.replace('/CALIBRATED/','/REDUCED/').replace('/RAW/','/REDUCED/')
        filename = os.path.join(os.path.dirname(filename), 'wcs_fits/', os.path.basename(filename))
        gf_filepath = filename.replace('.fits', '_gf.fits')
        wcs_filepath = filename.replace('.fits', '.wcs.fits')
    else:
        gf_filepath = filename.replace('.fits', '_gf.fits')
        wcs_filepath = filename.replace('.fits', '.wcs.fits')

    # check if wcs_filepath exists, if so has it been ran by astrometry?
    isCorrect = False
    if os.path.isfile(wcs_filepath):
        with fits.open(wcs_filepath) as hdulist:
            for comm in hdulist[0].header['COMMENT']:
                if 'solved_' in comm.lower():
                    isCorrect = True
                    break

    if not isCorrect:
        # Create file to save gaussian filtered fits
        with fits.open(true_filename) as hdulist:
            # Overwrite argument is helpful if pipeline failed previously
            hdulist.writeto(gf_filepath, overwrite=True)

        # variables to check if current file is
        cal_dir = os.path.dirname(true_filename)
        current_file = os.path.basename(true_filename)
        files = sorted(os.listdir(cal_dir), key=natural_keys)

        # sigma of gaussian filter, and astrometry's radius search are increased per loop
        nsigmas = 2
        nradii = 2
        sigmas = np.linspace(0, 5, nsigmas)
        radii = np.linspace(0.6, 1.8, nradii)
        log("Starting Astrometry on %s" % true_filename)
        astrometry_path = os.path.join(astrometry_directory, 'solve-field')
        for ext in exts:
            print('\t\t Working on extension {:}...'.format(ext))
            ext_fname = gf_filepath.replace('.fits', '_' + str(ext) + '.wcs.fits')
            if (ra is not None) and (dec is not None) and (radius is not None):
                CODE = '{} --continue --no-plots --downsample 2 --cpulimit 60 --extension "{}" ' \
                       '--scale-units arcsecperpix --scale-low {} --scale-high {} --ra {} ' \
                       '--dec {} --radius {} --new-fits "{}" "{}"'.format(astrometry_path, ext, scale_low, scale_high,
                                                                          ra, dec, 'FIXME', ext_fname, gf_filepath)
            else:
                CODE = '{} --continue --no-plots --downsample 2 --cpulimit 60 --extension "{}" ' \
                       '--scale-units arcsecperpix --scale-low ' \
                       '{} --scale-high {} --new-fits "{}" "{}"'.format(astrometry_path, ext, scale_low, scale_high,
                                                                        ext_fname, gf_filepath)
            # attempt multiple sigmas/radii;
            count = 0
            for i in range(nradii):
                temp_CODE = CODE.replace('FIXME', '%.3f' % radii[i])
                for j in range(nsigmas):
                    with fits.open(true_filename) as hdulist:
                        data = gaussian_filter(hdulist[ext].data, sigmas[j])
                        fits.update(gf_filepath, data, hdulist[ext].header, ext)
                    log("Executing Astrometry Code:")
                    log(temp_CODE)
                    p = subprocess.Popen(temp_CODE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, preexec_fn=os.setsid)
                    #p.wait() # Commented out - we *don't* want to wait if it's going to take forever
                    
                    # Log the start time
                    t_proc_start = clocking_time.time()
                    while True:
                        # Check if the process has returned an error or created the file
                        done = (p.returncode != 0 and p.returncode is not None) or (os.path.isfile(ext_fname))
                        
                        if not done:
                            # How many seconds have elapsed?
                            elapsed_time = clocking_time.time()-t_proc_start
                            
                            # Sleep for 1s
                            clocking_time.sleep(1)
                            
                            # Kill the entire process group if the timeout is reached
                            if elapsed_time>astrometry_timeout:
                                print("\t\t Astrometry.net timed out after {:.0f} seconds! ({:d}/{:d} attempts)"\
                                      .format(astrometry_timeout,count+1,nsigmas*nradii))
                                try:
                                    os.killpg(os.getpgid(p.pid),signal.SIGINT)
                                except ProcessLookupError:
                                    # On OS X the above line fails for some reason; this will just skip killing the process
                                    pass
                                break
                        else:
                            p.wait() # Wait for astrometry.net to finish running, just in case
                            break
                    
                    if p.returncode != 0 and p.returncode is not None:
                        print('\t\t ASTROMETRY FAILED. The error was:')
                        out, err = p.communicate()
                        print(err)
                        print('\n\t Exiting...\n')
                        sys.exit()
                    count += 1
                    success = os.path.isfile(ext_fname)
                    if success:
                        # if success, save successful parameters to first attempt
                        sigmas[0] = sigmas[j]
                        break
                if success:
                    # if success, save successful parameters to first attempt in following exts
                    print("\t\t Astrometry was successful for ext " + str(ext))
                    log("Astrometry was successful for ext %d" % ext)
                    radii[0] = radii[i]
                    break 
            log("Total Number of Attempts: % d" % count)
            print("\t\t Total Number of Attempts: " + str(count))
            if not success:
                log("Astrometry failed for ext %d" % ext)
                print("\t\t Astronomy failed for ext %d" % ext)
                print("\t\t Program will attempt to get WCS from latest processed file later.")

        # Astrometry.net is run on individual extensions, which are saved above.
        # Combining them back into a single file.
        with fits.open(true_filename) as hdulist:
            for ext in exts:
                ext_fname = gf_filepath.replace('.fits', '_' + str(ext) + '.wcs.fits')
                # Try to save new WCS info in original file
                try:
                    with fits.open(ext_fname) as hdulist_new:
                        if ext == 0:
                            hdulist[0].header = hdulist_new[0].header
                        else:
                            hdulist[ext].header = hdulist_new[1].header
                # If it doesn't work, copy the WCS info from the last file
                except FileNotFoundError as err:
                    # print out err
                    print("\t\t " + str(err))

                    # search adjecent files to see if they have been proccessed...
                    # this will look for the files before and after the current one
                    for i in range(len(files)):
                        if files[i] == current_file and i != 0:
                            last_filename = files[i - 1]
                            break
                    else:
                        # if there is no previous file, get next file
                        for i in range(len(files)):
                            if files[i] == current_file and i != 0:
                                last_filename = files[i + 1]
                                break
                        else:
                            # if there are no such files... then skip file
                            log('Quitting astrometry and skipping file')
                            print("\t\t Quitting astrometry and skipping file")
                            return 0

                    # fix up file name of latest WCS file
                    if server_work:
                        last_wcs_filename = os.path.join(cal_dir, last_filename).replace('.fits', '.wcs.fits')
                        last_wcs_filename = last_wcs_filename.replace('/RAW/', '/REDUCED/').replace('/CALIBRATED/','/REDUCED/')
                        last_wcs_filename = os.path.join(os.path.dirname(last_wcs_filename),
                                                         'wcs_fits/',
                                                         os.path.basename(last_wcs_filename))
                    else:
                        last_wcs_filename = os.path.join(cal_dir, last_filename).replace('.fits', '.wcs.fits')

                    # does it exist?
                    isFile = os.path.isfile(last_wcs_filename)
                    if not isFile:
                        print("\t\t Could not find previous frame {:}".format(last_wcs_filename))
                        # if reference file doesn't exit, skip this one
                        # doing this will avoid raising an exception and quitting photometry proccess
                        return 0

                    # if it exists, then keep going
                    print('\t\t Using WCS info from previous frame {:}'.format(last_wcs_filename))
                    with fits.open(last_wcs_filename) as hdulist_new:
                        if ext == 0:
                            hdulist[ext].header = hdulist_new[ext].header
                        else:
                            hdulist[ext].header = hdulist_new[1].header
                # If it works, remove the single-extension astrometry file
                else:
                    os.remove(ext_fname)
            
            # Strip the data from the WCS solution to save space
            for ext in exts:
                hdulist[ext].data = None
            
            # Save the original headers with the WCS info
            hdulist.writeto(wcs_filepath)

        # Save space by removing the gaussian filtered image, if any
        os.remove(gf_filepath)


def SkyToPix(h, ras, decs):
    """
    This code converts input ra and decs to pixel coordinates given the
    header information (h).
    """
    # Load WCS information:
    if 'EPOCH' in h:
        h['EPOCH'] = float(h['EPOCH'])
    if 'EQUINOX' in h:
        h['EQUINOX'] = float(h['EQUINOX'])
    w = wcs.WCS(h)
    # Generate matrix that will contain sky coordinates:
    sky_coords = np.zeros([len(ras), 2])
    # Fill it:
    for i in range(len(ras)):
        sky_coords[i, 0] = ras[i]
        sky_coords[i, 1] = decs[i]
    # Get pixel coordinates
    pix_coords = w.wcs_world2pix(sky_coords, 1)
    # Return x,y pixel coordinates:
    return pix_coords[:, 0], pix_coords[:, 1]


def _moments_central(data, center=None, order=1):
    """
    Calculate the central image moments up to the specified order.
    Parameters
    ----------
    data : 2D array-like
        The input 2D array.
    center : tuple of two floats or `None`, optional
        The ``(x, y)`` center position.  If `None` it will calculated as
        the "center of mass" of the input ``data``.
    order : int, optional
        The maximum order of the moments to calculate.
    Returns
    -------
    moments : 2D `~numpy.ndarray`
        The central image moments.
    """

    data = np.asarray(data).astype(float)

    if data.ndim != 2:
        raise ValueError('data must be a 2D array.')

    if center is None:
        center = centroid_com(data)

    indices = np.ogrid[[slice(0, i) for i in data.shape]]
    ypowers = (indices[0] - center[1]) ** np.arange(order + 1)
    xpowers = np.transpose(indices[1] - center[0]) ** np.arange(order + 1)

    return np.dot(np.dot(np.transpose(ypowers), data), xpowers)



def getAperturePhotometry(d, h, x, y, R, target_names, frame_name=None, out_dir=None, saveplot=False,
                          refine_centroids=False, half_size=50, GAIN=1.0, ncores=None):
    """
    Define/Set global variables for next aperture photometry
    :param d: image data
    :param h: image header
    :param x: x-Coordinate of object in pixels
    :param y: y-Coordinate of object in pixels
    :param R: list/array of aperture radii
    :param target_names:
    :param frame_name:
    :param out_dir: output directory for
    :param saveplot:
    :param refine_centroids:
    :param half_size:
    :param GAIN:
    :param ncores:
    :return:
    """
    global global_d, global_h, global_x, global_y, global_R, global_target_names, global_frame_name, \
        global_out_dir, global_saveplot, global_refine_centroids, global_half_size, global_GAIN
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

    fluxes = np.zeros([len(x), len(R)])
    fluxes_err = np.zeros([len(x), len(R)])
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
        fluxes[i, :], fluxes_err[i, :], x_ref[i], y_ref[i], bkg[i], bkg_err[i], fwhm[i] = results[i]
    return fluxes, fluxes_err, x_ref, y_ref, bkg, bkg_err, fwhm


def getCentroidsAndFluxes(i):
    fluxes_R = np.ones(len(global_R)) * (-1)
    fluxes_err_R = np.ones(len(global_R)) * (-1)
    # Generate a sub-image around the centroid, if centroid is inside the image:
    if 0 < global_x[i] < global_d.shape[1] and 0 < global_y[i] < global_d.shape[0]:
        # Reminder (I might need it in case of confusion later) :
        # x1 and y1 aren't indeces, they're exclusive boundaries
        # while x0 and y0 are indeces and inclusive boundaries
        x0 = max(0, int(global_x[i]) - global_half_size)
        x1 = min(int(global_x[i]) + global_half_size, global_d.shape[1])
        y0 = max(0, int(global_y[i]) - global_half_size)
        y1 = min(int(global_y[i]) + global_half_size, global_d.shape[0])
        x_cen = global_x[i] - x0
        y_cen = global_y[i] - y0
        subimg = global_d[y0:y1, x0:x1].astype(float)
        x_ref = x0 + x_cen
        y_ref = y0 + y_cen
        if global_refine_centroids:
            # Refine the centroids, if falls on full image, then redefine subimage to center object
            x_new, y_new = get_refined_centroids(subimg, x_cen, y_cen)
            if 0 < x_new < global_d.shape[1] and 0 < y_new < global_d.shape[0]:
                x_cen, y_cen = int(x_new), int(y_new)
                x_ref = x0 + x_cen
                y_ref = y0 + y_cen
                x0 = max(0, x_ref - global_half_size)
                x1 = min(x_ref + global_half_size, global_d.shape[1])
                y0 = max(0, y_ref - global_half_size)
                y1 = min(y_ref + global_half_size, global_d.shape[0])
                subimg = global_d[y0:y1, x0:x1].astype(float)

        # Estimate background level: mask out sources, get median of masked image, and noise
        mask = make_source_mask(subimg, snr=2, npixels=5, dilate_size=11)
        mean, median, std = sigma_clipped_stats(subimg, sigma=3.0, mask=mask)
        background = median
        background_sigma = std
        subimg -= background
        sky_sigma = np.ones(subimg.shape) * background_sigma
        # If saveplot is True, save image and the centroid:
        if global_saveplot and ('target' in global_target_names[i]):
            if not os.path.exists(global_out_dir + global_target_names[i]):
                os.mkdir(global_out_dir + global_target_names[i])
            im = plt.imshow(subimg)
            im.set_clim(0, 1000)
            plt.plot(x_cen, y_cen, 'wx', markersize=15, alpha=0.5)
            circle = plt.Circle((x_cen, y_cen), np.min(global_R), color='black', fill=False)
            circle2 = plt.Circle((x_cen, y_cen), np.max(global_R), color='black', fill=False)
            plt.gca().add_artist(circle)
            plt.gca().add_artist(circle2)
            if not os.path.exists(global_out_dir + global_target_names[i] + '/' + global_frame_name + '.png'):
                plt.savefig(global_out_dir + global_target_names[i] + '/' + global_frame_name + '.png')
            plt.close()
        # With the calculated centroids, get aperture photometry:
        for j in range(len(global_R)):
            fluxes_R[j], fluxes_err_R[j] = getApertureFluxes(subimg, x_cen, y_cen, global_R[j], sky_sigma)
        fwhm = estimate_fwhm(subimg, x_cen, y_cen)
        return fluxes_R, fluxes_err_R, x_ref, y_ref, background, background_sigma, fwhm
    else:
        return fluxes_R, fluxes_err_R, global_x[i], global_y[i], 0., 0., 0.


def get_refined_centroids(data, x_init, y_init, half_size=25):
    """
    Refines the centroids by fitting a centroid to the central portion of an image
    Method assumes initial astrometry is accurate within the half_size
    """
    # Take the central portion of the data (i.e., the subimg)
    x0 = max(0, int(x_init) - half_size)
    x1 = min(int(x_init) + half_size, data.shape[1])
    y0 = max(0, int(y_init) - half_size)
    y1 = min(int(y_init) + half_size, data.shape[0])
    x_guess = x_refined = x_init - x0
    y_guess = y_refined = y_init - y0
    cen_data = data[y0:y1, x0:x1].astype(float)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        sources = DAOStarFinder(threshold=0, fwhm=2.35 * 5).find_stars(gaussian_filter(cen_data, 5))
    xcents = sources['xcentroid']
    ycents = sources['ycentroid']
    dists = (x_refined - xcents) ** 2 + (y_refined - ycents) ** 2
    try:
        idx_min = np.argmin(dists)
        x_guess = xcents[idx_min]
        y_guess = ycents[idx_min]
    except Exception as e:
        print('\t\t DAOStarFinder failed. Refining pointing with a gaussian...')
        try:
            # Robustly fit a gaussian
            p = fit_gaussian(cen_data)
            x_guess = p[1]
            y_guess = p[2]
        except Exception:
            print('\t\t No luck. Resorting to astrometric coordinates.')
    # Don't let the new coordinates stray outside of the sub-image
    if data.shape[0] > x_guess > 0:
        x_refined = x_guess
    if data.shape[1] > y_guess > 0:
        y_refined = y_guess
    return x_refined + x0, y_refined + y0


def gaussian(height, center_x, center_y, width_x, width_y):
    """Returns a gaussian function with the given parameters"""
    width_x = float(width_x)
    width_y = float(width_y)
    return lambda x, y: height * np.exp(
        -(((center_x - x) / width_x) ** 2 + ((center_y - y) / width_y) ** 2) / 2)


def moments(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution by calculating its
    moments """
    total = data.sum()
    X, Y = np.indices(data.shape)
    x = (X * data).sum() / total
    y = (Y * data).sum() / total
    col = data[:, int(y)]
    width_x = np.sqrt(np.abs((np.arange(col.size) - y) ** 2 * col).sum() / col.sum())
    row = data[int(x), :]
    width_y = np.sqrt(np.abs((np.arange(row.size) - x) ** 2 * row).sum() / row.sum())
    height = data.max()
    return height, x, y, width_x, width_y


def fit_gaussian(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution found by a fit"""
    params = moments(data)
    errorfunction = lambda p: np.ravel(gaussian(*p)(*np.indices(data.shape)) -
                                       data)
    p, success = optimize.leastsq(errorfunction, params)
    return p


def estimate_fwhm(data, x0, y0):
    """
    This function estimates the FWHM of an image containing only one source
    (possibly gaussian) and estimates the FWHM of it by performing two cuts on 
    X and Y and finding the stddev of both cuts. The resulting FWHM is obtained from 
    the mean of those stddev.  
    """

    def get_second_moment(x, y, mu):
        moment = np.sqrt(np.sum(y * (x - mu) ** 2) / np.sum(y))
        return moment
    # the following two lines might be unnecessary; since data 'should' be centered anyway
    y0_idx = int(y0) if y0 < data.shape[0] else data.shape[0] - 1
    x0_idx = int(x0) if x0 < data.shape[1] else data.shape[1] - 1
    sigma_y = get_second_moment(np.arange(data.shape[1]), data[y0_idx, :], x0)
    sigma_x = get_second_moment(np.arange(data.shape[0]), data[:, x0_idx], y0)
    # Context: if sigma_x and sigma_y are not zero or NaN, then average them, else return the one that isn't zero/NaN
    sigma = (sigma_x + sigma_y) / 2. if sigma_x and sigma_y else sigma_y or sigma_x
    return gaussian_sigma_to_fwhm * sigma


def getApertureFluxes(subimg, x_cen, y_cen, Radius, sky_sigma):
    apertures = CircularAperture([(x_cen, y_cen)], r=Radius)
    rawflux_table = aperture_photometry(subimg, apertures,
                                        error=sky_sigma)
    return rawflux_table['aperture_sum'][0], rawflux_table['aperture_sum_err'][0]


def angle2degree(raw_angle, unit):
    """
    Convert given angle with known unit (astropy.unit) to degrees.
    :param raw_angle: numberic or string value
    :param unit: unit of the angle; a astropy.unit object
    :return: angle in degrees (decimal)
    """
    return Angle(raw_angle, unit=unit).deg


def CoordsToDecimal(coords):
    """
    Function to convert given angles to degree decimals. This function makes big assumptions given the wide variety
    of formats that EDEN has come across.
    ASSUMPTION:
    - if given coordinates are numeric values, then both RA/DEC are given in degrees
    - if given coordinates are strings/non-numeric values, then RA is given in hour angle and DEC in degrees.
    :param coords:
    :return: ras, decs in decimal degrees
    """
    ras = np.array([])
    decs = np.array([])
    for i in range(len(coords)):
        # JOSE's mod ----- function used to assume consistent formatting of RA/DEC in Header.
        raw_RA, raw_DEC = coords[i]
        # the following if-else statement only works for current telescope usage (Aug. 2018)
        try:
            # Both are in degrees format
            ras = np.append(ras, angle2degree(float(raw_RA), u.deg))
            decs = np.append(decs, angle2degree(float(raw_DEC), u.deg))
        except ValueError:
            # it must be a string, and with the following formats
            ras = np.append(ras, angle2degree(raw_RA, u.hourangle))
            decs = np.append(decs, angle2degree(raw_DEC, u.deg))
    return ras, decs


def DecimalToCoords(ra_degs, dec_degs):
    ra_coords = len(ra_degs) * [[]]
    dec_coords = len(dec_degs) * [[]]
    for i in range(len(ra_degs)):
        c_ra = (ra_degs[i] / 360.) * 24.
        c_dec = dec_degs[i]
        ra_hr = int(c_ra)
        ra_min = int((c_ra - ra_hr) * 60.)
        ra_sec = (c_ra - ra_hr - ra_min / 60.0) * 3600.
        dec_deg = int(c_dec)
        dec_min = int(np.abs(c_dec - dec_deg) * 60.)
        dec_sec = (np.abs(c_dec - dec_deg) - dec_min / 60.) * 3600.
        ra_coords[i] = NumToStr(ra_hr) + ':' + NumToStr(ra_min) + ':' + NumToStr(ra_sec, roundto=3)
        dec_coords[i] = NumToStr(dec_deg) + ':' + NumToStr(dec_min) + ':' + NumToStr(dec_sec, roundto=3)
    return ra_coords, dec_coords


def NumToStr(number, roundto=None):
    """
    Convert number to string using string formatting.
    :param number: integer or floating point
    :param roundto: round to decimal points
    :return: string formatted value
    """
    formatString = '%02d'
    if isinstance(number, float):
        if roundto is not None and roundto != 0:
            # the + 3 is the decimal point and the 2 minimum digit before decimal pt
            totalDigits = roundto + 4 if number < 0 else roundto + 3
            formatString = '%0{:d}.{:d}f'.format(totalDigits, roundto)
        else:
            number = int(number)
    if isinstance(number, int):
        totalDigits = 3 if number < 0 else 2
        formatString = '%0{:d}d'.format(totalDigits)
    return formatString % number


def SuperComparison(fluxes, errors):
    flux = np.sum(fluxes / errors ** 2) / np.sum(1. / errors ** 2)
    err_flux = np.sqrt(np.sum(errors ** 2) / np.double(len(fluxes)))
    return flux, err_flux


class Time(time.Time):
    """
    time class: inherits astropy time object, and adds heliocentric, barycentric
        correction utilities.
    """

    def __init__(self, *args, **kwargs):
        super(Time, self).__init__(*args, **kwargs)
        self.height = kwargs.get('height', 0.0)

    def _pvobs(self):
        """
        calculates position and velocity of the telescope
        :return: position/velocity in AU and AU/d in GCRS reference frame
        """
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
        X, Y, S = erfa.xys00a(tt.jd1, tt.jd2)

        # Get dX and dY from IERS B
        dX = np.interp(mjd, iers_tab['MJD'], iers_tab['dX_2000A']) * u.arcsec
        dY = np.interp(mjd, iers_tab['MJD'], iers_tab['dY_2000A']) * u.arcsec

        # Get GCRS to CIRS matrix
        # can be used to convert to Celestial Intermediate Ref Sys
        # from GCRS.
        rc2i = erfa.c2ixys(X + dX.to(u.rad).value, Y + dY.to(u.rad).value, S)

        # Gets the Terrestrial Intermediate Origin (TIO) locator s'
        # Terrestrial Intermediate Ref Sys (TIRS) defined by TIO and CIP.
        # TIRS related to to CIRS by Earth Rotation Angle
        sp = erfa.sp00(tt.jd1, tt.jd2)

        # Get X and Y from IERS B
        # X and Y are
        xp = np.interp(mjd, iers_tab['MJD'], iers_tab['PM_x']) * u.arcsec
        yp = np.interp(mjd, iers_tab['MJD'], iers_tab['PM_y']) * u.arcsec

        # Get the polar motion matrix. Relates ITRF to TIRS.
        rpm = erfa.pom00(xp.to(u.rad).value, yp.to(u.rad).value, sp)

        # multiply ITRF position of obs by transpose of polar motion matrix
        # Gives Intermediate Ref Frame position of obs
        x, y, z = np.array([rpmMat.T.dot(xyz) for rpmMat in rpm]).T

        # Functions of Earth Rotation Angle, theta
        # Theta is angle bewtween TIO and CIO (along CIP)
        # USE UT1 here.
        theta = erfa.era00(ut1.jd1, ut1.jd2)
        S, C = np.sin(theta), np.cos(theta)

        # Position #GOT HERE
        pos = np.asarray([C * x - S * y, S * x + C * y, z]).T

        # multiply by inverse of GCRS to CIRS matrix
        # different methods for scalar times vs arrays
        if pos.ndim > 1:
            pos = np.array([np.dot(rc2i[j].T, pos[j]) for j in range(len(pos))])
        else:
            pos = np.dot(rc2i.T, pos)

        # Velocity
        vel = np.asarray([SR * (-S * x - C * y), SR * (C * x - S * y), np.zeros_like(x)]).T
        # multiply by inverse of GCRS to CIRS matrix
        if vel.ndim > 1:
            vel = np.array([np.dot(rc2i[j].T, vel[j]) for j in range(len(pos))])
        else:
            vel = np.dot(rc2i.T, vel)

        # return position and velocity
        return pos, vel

    def _obs_pos(self):
        """
        calculates heliocentric and barycentric position of the earth in AU and AU/d
        """
        tdb = self.tdb

        # get heliocentric and barycentric position and velocity of Earth
        # BCRS reference frame
        h_pv, b_pv = erfa.epv00(tdb.jd1, tdb.jd2)

        # h_pv etc can be shape (ntimes,2,3) or (2,3) if given a scalar time
        if h_pv.ndim == 2:
            h_pv = h_pv[np.newaxis, :]
        if b_pv.ndim == 2:
            b_pv = b_pv[np.newaxis, :]

        # unpack into position and velocity arrays
        h_pos = h_pv[:, 0, :]
        h_vel = h_pv[:, 1, :]

        # unpack into position and velocity arrays
        b_pos = b_pv[:, 0, :]
        b_vel = b_pv[:, 1, :]

        # now need position and velocity of observing station
        pos_obs, vel_obs = self._pvobs()

        # add this to heliocentric and barycentric position of center of Earth
        h_pos += pos_obs
        b_pos += pos_obs
        h_vel += vel_obs
        b_vel += vel_obs
        return h_pos, h_vel, b_pos, b_vel

    def _vect(self, coord):
        '''get unit vector pointing to star, and modulus of vector, in AU
           coordinate of star supplied as astropy.coordinate object

           assume zero proper motion, parallax and radial velocity'''
        pmra = pmdec = px = rv = 0.0

        rar = coord.ra.radian
        decr = coord.dec.radian
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # ignore warnings about 0 parallax
            pos, vel = erfa.starpv(rar, decr, pmra, pmdec, px, rv)

        modulus = np.sqrt(pos.dot(pos))
        unit = pos / modulus
        modulus /= const.au.value
        return modulus, unit

    def hcor(self, coord):
        mod, spos = self._vect(coord)
        # get helio/bary-centric position and velocity of telescope, in AU, AU/d
        h_pos, h_vel, b_pos, b_vel = self._obs_pos()

        # heliocentric light travel time, s
        tcor_hel = const.au.value * np.array([np.dot(spos, hpos) for hpos in h_pos]) / const.c.value
        # print 'Correction to add to get time at heliocentre = %.7f s' % tcor_hel
        dt = time.TimeDelta(tcor_hel, format='sec', scale='tdb')
        return self.utc + dt

    def bcor(self, coord):
        mod, spos = self._vect(coord)
        # get helio/bary-centric position and velocity of telescope, in AU, AU/d
        h_pos, h_vel, b_pos, b_vel = self._obs_pos()

        # barycentric light travel time, s
        tcor_bar = const.au.value * np.array([np.dot(spos, bpos) for bpos in b_pos]) / const.c.value
        # print 'Correction to add to get time at barycentre  = %.7f s' % tcor_bar
        dt = time.TimeDelta(tcor_bar, format='sec', scale='tdb')
        return self.tdb + dt


def get_exts(hdulist):
    """
    Returns a list of the fits extensions containing data
    """
    
    # If the input isn't an HDUList then assume it is a filename and open the file
    if type(hdulist) is not fits.HDUList:
        hdulist = fits.open(hdulist)
        close = True
    else:
        close = False
    
    exts = []
    for i in range(len(hdulist)):
        if hdulist[i].data is not None:
            exts.append(i)
    
    # If the input wasn't an HDUList, then close the image
    if close:
        im.close()
    
    return exts
    
    # Old method (not compatible with CAHA data; also assumes extension 0 is empty
    """
    h = fits.getheader(filename)
    try:
        EXTEND = h['EXTEND']
    except KeyError:
        EXTEND = False
    if EXTEND:
        exts = range(1, h['NEXTEND'] + 1)
    else:
        exts = [0]
    return exts
    """


def get_general_coords(target, date):
    """
    Given a target name, returns RA and DEC from simbad.
    :param target: string name of target
    :param date: date string or datetime object
    :return:
    """
    if isinstance(date, str):
        date = date.replace('-', '')
    try:
        # Try to get info from Simbad
        target_fixed = target.replace('_', ' ')
        log("Querying Simbad Target: %s" % target_fixed)
        result = Simbad.query_object(target_fixed)
        # If none, try again with a dash
        if result is None:
            result = Simbad.query_object(target_fixed.replace(' ','-'))
        if result is None:
            # result is None when query fails
            raise KeyError('Invalid target name in Simbad Query: %s' %target_fixed)
        else:
            # print("\t Simbad lookup successful for {:s}!".format(target_fixed))
            log("Simbad lookup successful for {:s}!".format(target_fixed))
    except KeyError as e:
        # Manually load values
        log(str(e))
        if target in manual_object_coords:
            ra, dec = manual_object_coords[target].split(' ')
            return ra, dec
        else:
            # no other option but to raise err
            raise
    else:
        # Assuming the Simbad query worked, load the coordinates:
        # Load positions as strings
        rahh, ramm, rass = result['RA'][0].split()
        decdd, decmm, decss = result['DEC'][0].split()
        # Load proper motions as arcsec / year
        pmra = result['PMRA'].to(u.arcsec / u.year).value[0]
        pmdec = result['PMDEC'].to(u.arcsec / u.year).value[0]
        # Convert RA and DEC to whole numbers:
        ra = np.double(rahh) + (np.double(ramm) / 60.) + (np.double(rass) / 3600.)
        if np.double(decdd) < 0:
            dec = np.double(decdd) - (np.double(decmm) / 60.) - (np.double(decss) / 3600.)
        else:
            dec = np.double(decdd) + (np.double(decmm) / 60.) + (np.double(decss) / 3600.)
            # Calculate time difference from J2000:
        if isinstance(date, str):
            year = int(date[:4])
            month = int(date[4:6])
            day = int(date[6:8])
            s = str(year) + '.' + str(month) + '.' + str(day)
            dt = dateutil.parser.parse(s)
        else:
            dt = date
        data_jd = sum(jdcal.gcal2jd(dt.year, dt.month, dt.day))
        deltat = (data_jd - 2451544.5) / 365.25
        # Calculate total PM:
        pmra = np.double(pmra) * deltat / 15.  # Conversion from arcsec to sec <- This works for GJ 1214, TRAPPIST-1
        pmdec = np.double(pmdec) * deltat
        # Correct proper motion:
        c_ra = ra + ((pmra) / 3600.)
        c_dec = dec + ((pmdec) / 3600.)
        # Return RA and DEC:
        ra_hr = int(c_ra) 
        ra_min = int((c_ra - ra_hr) * 60.)
        ra_sec = (c_ra - ra_hr - ra_min / 60.0) * 3600.
        dec_deg = int(c_dec)
        dec_min = int(np.abs(c_dec - dec_deg) * 60.)
        dec_sec = (np.abs(c_dec - dec_deg) - dec_min / 60.) * 3600.
        return NumToStr(ra_hr) + ':' + NumToStr(ra_min) + ':' + NumToStr(ra_sec, roundto=3), \
               NumToStr(dec_deg) + ':' + NumToStr(dec_min) + ':' + NumToStr(dec_sec, roundto=3)


def get_suffix(s):
    m = re.match('.+([0-9])[^0-9]*$', s)
    idx = m.start(1) + 1
    return s[idx:]


def get_trailing_number(s):
    m = re.search(r'\d+$', s)
    return int(m.group()) if m else None
