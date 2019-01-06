import fnmatch
import random
import sys
import time
from functools import partial
from multiprocessing import Pool
from os.path import join, isdir, isfile, dirname, basename

import numpy as np
from astropy.io import fits
from astropy.modeling import models, fitting
from ccdproc import cosmicray_lacosmic
from dateutil.parser import parse

from DirSearch_Functions import search_all_fits
from constants import log, ModHDUList, natural_keys, find_dates, find_val, get_calibrations
from constants import server_destination, find_dimensions, filter_fits
from dirs_mgmt import validate_dirs

# Fitter/Models
_of = fitting.LinearLSQFitter()
poly_model = models.Polynomial1D(3)


def find_imgtype(filepath_header, filename=None, ext=0, median_opt=False):
    """
    Only for calibration files!!. Find the image type (flats,darks,bias)
    :param filepath_header: this must be the full address to file, or the header of the file
    :param filename: the file name. Only actually used when filepath_header is a header object
    :param ext: Optional, extension header to which check info
    :param median_opt: if True, this function will allow super calibration files to pass the test
    :return: the image type, returns None if image is not a calibration file
    """
    mylist = ["FLAT", "ZERO", "BIAS", "DARK", "BLANK"]
    bias_labels = ["ZERO", "BLANK"]
    # Check object header
    option1 = find_val(filepath_header, "object", ext, raise_err=False).upper()
    # Check image type header
    option2 = find_val(filepath_header, "imagety", ext, raise_err=False).upper()
    # Check filename
    if isinstance(filepath_header, str):
        option3 = filename = basename(filepath_header).upper()
    else:
        if filename is None:
            option3 = filename = ''
        else:
            option3 = filename.upper()
    # a fourth one... make sure you're not getting an already median file
    option4 = True if median_opt else ("MEDIAN" not in filename.upper()) or ("MEAN" not in filename.upper())
    mylist2 = [option1, option2, option3]
    for key in mylist:
        # this loop checks for all options and see if there is a match
        truth_func = any([key in key2 for key2 in mylist2])
        if truth_func and option4:
            if key in bias_labels:
                return "BIAS"
            return key
    else:
        return None


def find_gain(filepath_header, namps=1):
    """
    Assumes gain keys are in primary header are in order for amplifier
    :param filepath_header: filepath or header of FITS file
    :param namps: number of amplifiers
    :return: list with gains per amplifier in the order of amplifier
    """
    hrd = get_header(filepath_header)
    comms = hrd.comments
    # HERE GAIN KEYS WILL BE IN ORDER, SO WE ASSUME GAIN KEYWORDS
    # HAVE SOME SORT OF ORDERING
    # Create filtering function for invalid gain values

    def filter_function(x):
        return 'GAIN' in x.upper() and ('VID' not in x.upper() and 'VIDEO' not in comms[x].upper())

    # filter and sort gain keywords
    gain_keys = sorted(filter(filter_function, hrd.keys()), key=natural_keys)
    # now with this order, get gains
    gains = [float(hrd[key]) for key in gain_keys if hrd[key]]
    if len(gains) == namps:
        return gains
    else:
        return None


def find_rdnoise(filepath_header, namps=1):
    """
    Assumes read noise keys are in primary header are in order for amplifier
    :param filepath_header: filepath or header of FITS file
    :param namps: number of amplifiers
    :return: list with read noises per amplifier in the order of amplifier
    """
    hrd = get_header(filepath_header)
    # filter and sort gain keywords
    rdnois_keys = sorted(filter(lambda x: 'RDNOIS' in x.upper(), hrd.keys()), key=natural_keys)
    # now with this order, get read noises
    rdnoises = [float(hrd[key]) for key in rdnois_keys if hrd[key]]
    if len(rdnoises) == namps:
        return rdnoises
    else:
        return None


def get_data(image, ext=0) -> np.ndarray:
    if isinstance(image, str):
        return fits.getdata(image, ext=ext)
    elif isinstance(image, fits.HDUList):
        return image[ext].data
    else:
        return image[ext]


def get_header(filename_hdu, ext=0):
    """
    Get header from filepath of FITS file or HDUList object.
    :param filename_hdu: filepath to file (string) or HDUList object
    :param ext: extension; default [0]
    :return: return the header
    """
    if isinstance(filename_hdu, fits.Header):
        header = filename_hdu
    else:
        with fits.open(filename_hdu) as hdul:
            header = hdul[ext].header
    return header


def get_secs(hdu: fits.ImageHDU, as_ref=False):
    # Return BIASSEC & DATASEC as numpy index.
    # Return DATASEC ONLY if BIASSEC doesn't exist.
    # Return None if neither keyword is found
    hdr = hdu.header
    naxis1 = find_val(hdr, 'NAXIS1')
    naxis2 = find_val(hdr, 'NAXIS2')
    if 'BIASSEC' in hdr:
        biassec = hdr['BIASSEC'].replace(':', ',').strip('][').split(',')
    else:
        biassec = None

    if 'DATASEC' in hdr:
        datasec = hdr['DATASEC'].replace(':', ',').strip('][').split(',')
    else:
        datasec = None

    # If neither exist, return None
    if not datasec and not biassec:
        return None
    elif datasec and not biassec:
        # FITS slice format -> Numpy slice format
        x1, x2, y1, y2 = [int(c) for c in datasec]
        x1, y1 = x1 - 1, y1 - 1

        # This indicates CAHA/CASSINI-LIKE FORMAT
        if y2 < y1 or x2 < x1 or x1 == x2:
            x1, y1, x2, y2 = [int(c) for c in datasec]

        # This indicates the slice is actually the whole image!
        whole_im = y2 - y1 == naxis2 and x2 - x1 == naxis1

        if not as_ref and whole_im:
            return np.s_[:, :]
        else:
            return np.s_[y1:y2, x1:x2]
    else:
        # Both exist!
        x1, x2, y1, y2 = [int(c) for c in datasec]
        x1, y1 = x1 - 1, y1 - 1

        # Indicates CAHA/CASSINI-LIKE FORMAT
        if y2 < y1 or x2 < x1:
            x1, y1, x2, y2 = [int(c) for c in datasec]

        # This indicates the slice is actually the whole image!
        whole_im = y2 - y1 == naxis2 and x2 - x1 == naxis1

        # If dataslice is the whole image then there can't be bias, return!
        if not as_ref and whole_im:
            return np.s_[:, :]

        # At this point I assume both dataslice and biaslice will be non-zero slices.
        dataslice = np.s_[y1:y2, x1:x2]

        x1, x2, y1, y2 = [int(c) for c in biassec]
        x1, y1 = x1 - 1, y1 - 1
        if y2 < y1 or x2 < x1:
            # Indicates CAHA/CASSINI-LIKE FORMAT
            x1, y1, x2, y2 = [int(c) for c in biassec]

        biasslice = np.s_[y1:y2, x1:x2]
        return dataslice, biasslice


def overscan_sub(hdul) -> ModHDUList:
    """
    Subtract overscan region of frame
    """
    hdul = ModHDUList(hdul)
    exts = len(hdul)
    trimmed_hdul = hdul.copy()
    for i in range(exts):
        if hdul[i].data is None:
            continue
        secs = get_secs(hdul[i])
        if isinstance(secs[0], tuple):
            dataslice, biasslice = secs
            bias = hdul[i].data[biasslice]
            axis = 0 if bias.shape[1] > bias.shape[0] else 1
            oscan = np.median(bias, axis=axis, overwrite_input=True)
            yarr = np.arange(len(oscan))
            # noinspection PyTypeChecker
            oscan = _of(poly_model, yarr, oscan)
            oscan = oscan(yarr)
            if axis == 1:
                oscan = np.reshape(oscan, (oscan.size, 1))
            else:
                oscan = np.reshape(oscan, (1, oscan.size))
            trimmed_hdul[i].data = hdul[i].data[dataslice] - oscan
        elif secs and secs != np.s_[:, :]:
            dataslice = secs
            trimmed_hdul[i].data = hdul[i].data[dataslice]
    return trimmed_hdul


def get_best_comb(telescope: str, date: str, *kinds, bins=None,
                  filt=None, twilight=True, ndims=None, ret_none=False):
    """
    Function to get best calibration file for a dataset of given types.
    This will return a file per each kind, in the same order as kinds.
    These files are the combined calibrations images.

    :param telescope: telescope for which to get calibration files
    :param date: date for which to find best cal
    :param kinds: any of types; FLAT, DARK, BIAS
    :param bins: bins to match calibration file with
    :param filt: filter to match calibration file with, only needed for flats
    :param twilight: whether twilight images are allowed.
    :param ndims: dimensions of dataset; tuple-like: (X, Y) OR (NAXIS1, NAXIS2)
    :param ret_none: if true, function will return if no such calibration file is found; otherwise it will raise Err.
    """
    assert bins is not None, 'The parameter bins must be passed to match files'
    if 'FLAT' in kinds and filt is None:
        log('Warning. Using get_best_cal for FLAT file without passing filt arg.')
        log('Results may vary.')

    telescope = telescope.upper()
    date = parse(date)
    cals = []
    comb_tree = get_calibrations(telescope, combined=True)
    for kind in kinds:
        kind = kind.upper()
        root_dir = join(server_destination, 'COMBINED', telescope, kind)
        root_join = lambda *paths: join(root_dir, *paths)

        # Get all dates
        str_dates = list(comb_tree[kind])

        # Get all images!
        images = []
        for dt in str_dates:
            # Get full path to images
            imgs = map(lambda img: root_join(dt, img), comb_tree[kind][dt])
            images.append(list(imgs))

        # Decide filtering function for compatibility
        args = (twilight, bins, filt) if filt is not None and kind == 'FLAT' else (bins,)
        args = (args + (ndims,)) if ndims is not None else args
        check_func = lambda img: check_comp(img, *args)

        # filter out incompatible dates
        comps = [list(map(check_func, imgs)) for imgs in images]
        comp_dates = [str_dates[i] for i in range(len(comps)) if any(comps[i])]

        # get absolute differences from given date
        dates = [parse(dt) for dt in comp_dates]
        diff_dates = [abs(dt - date) for dt in dates]

        # if diff_dates is empty ==> No such calibration files, then return None
        if not diff_dates:
            if ret_none:
                log('No %s calibration files were found for %s, skipping...' % (kind, telescope))
                cals.append(None)
                continue
            else:
                raise ValueError('No compatible calibration files were found.')

        # find date closest to working date
        indx = diff_dates.index(min(diff_dates))
        best_date = comp_dates[indx]

        # We now must grab the compatible file in the best_date
        indx_orig = str_dates.index(best_date)
        indx_comp = comps[indx_orig].index(True)
        best_cal = images[indx_orig][indx_comp]

        cals.append(best_cal)

    # Return one object if only one kind was given.
    if len(cals) == 1:
        return cals[0]
    else:
        return cals


def _imcombine_help(im_list):
    """
    Helper function that aids the imcombine functions for the calibration images.
    :param im_list: list of FITS path files
    """
    # Find date for which to we're getting data
    date = find_dates(im_list[0])[0]

    log('Stacking data!')
    start = time.time()
    # Trim image and apply overscan subtraction if supported.
    images = [overscan_sub(ModHDUList(im, in_mmem=True)) for im in im_list]
    end = time.time() - start
    if end > 5.:
        log('Data stacking took %.3f secs' % end)

    # We assume binning and dimensions are the same across images.
    bins = 1 if 'CASSINI' in im_list[0] else find_val(images[0], "bin")
    ndims = find_dimensions(images[0])
    return images, date, bins, ndims


def _normalize_exptimes(images, all_exptime=None) -> list:
    """
    Function that will normalize all given images using a randomly chosen exposure time from the list.
    This will only be done if images do vary in time exposure.
    :param images: list of ModHDULists
    :return: list of normalized ModHDULists
    """
    if not all_exptime:
        all_exptime = [find_val(im, 'exptime') for im in images]
    if len(set(all_exptime)) > 1:
        log("Exposure time of darks varies, applying normalizing method")
        exptime = random.choice(all_exptime)

        def normalize(im, exptimed):
            im[0].header['EXPTIME'] = exptime
            return im * exptime / exptimed

        images = [normalize(im, exptimed) for im, exptimed in zip(images, all_exptime)]
    return images


def imcombine_bias(*args):
    log('\nStarting imcombine_darks')
    im_list = args[0]
    assert isinstance(im_list, list), 'Unexpected first positional argument: %r' % im_list
    log('SAMPLE IMAGE INPUT: %s' % im_list[0])

    log('Stacking data!')
    start = time.time()
    # Trim image and apply overscan subtraction if supported.
    images = [overscan_sub(ModHDUList(im, in_mmem=True)) for im in im_list]
    end = time.time() - start
    if end > 5.:
        log('Data stacking took %.3f secs' % end)

    # Modify args accordingly
    args = (images,) + args[1:]

    return imcombine(*args)


def imcombine_darks(*args, telescop=None):
    log('\nStarting imcombine_darks')
    im_list = args[0]
    assert isinstance(im_list, list), 'Unexpected first positional argument: %r' % im_list
    log('SAMPLE IMAGE INPUT: %s' % im_list[0])

    images, date, bins, ndims = _imcombine_help(im_list)

    log('Subtracting bias frame')
    bias_im = get_best_comb(telescop, date, 'BIAS', bins=bins, ndims=ndims)
    log('Best BIAS frame found %s' % bias_im)
    bias_hdul = ModHDUList(bias_im)
    images = [hdul - bias_hdul for hdul in images]

    # As last step, normalize (if needed) all images to a exposure time; Making sure we combine equivalent darks
    images = _normalize_exptimes(images)

    # Modify args accordingly
    args = (images,) + args[1:]

    return imcombine(*args)


def imcombine_flats(*args, telescop=None, darks=False):
    log('\nStarting imcombine_darks')
    im_list = args[0]
    assert isinstance(im_list, list), 'Unexpected first positional argument: %r' % im_list
    log('SAMPLE IMAGE INPUT: %s' % im_list[0])

    images, date, bins, ndims = _imcombine_help(im_list)
    all_exptime = [find_val(im, 'exptime') for im in images]
    multi_exptime = len(set(all_exptime)) > 1

    if darks:
        log('Subtracting bias/dark frame')
        bias_im, dark_im = get_best_comb(telescop, date, 'BIAS', 'DARK', bins=bins, ndims=ndims)
        log('Best BIAS frame found %s' % bias_im)
        log('Best DARK frame found %s' % dark_im)
        bias_hdul = ModHDUList(bias_im)
        dark_hdul = ModHDUList(dark_im)

        exptime_dark = float(find_val(dark_hdul, 'exptime'))
        # Now we must make sure that dark_hdul has same exposure time as all flat images
        if multi_exptime:
            images = [(im - bias_hdul - dark_hdul * exptime / exptime_dark) for im, exptime in zip(images, all_exptime)]
        else:
            exptime = random.choice(all_exptime)
            super_bias = bias_hdul - dark_hdul * exptime / exptime_dark
            images = [im - super_bias for im in images]
        print(f'Reference count for dark_hdul: %r ' % sys.getrefcount(dark_hdul))

    else:
        log('Subtracting bias frame')
        bias_im = get_best_comb(telescop, date, 'BIAS', bins=bins, ndims=ndims)
        log('Best BIAS frame found %s' % bias_im)
        bias_hdul = ModHDUList(bias_im)
        images = [hdul - bias_hdul for hdul in images]

    print(f'Reference count for bias_hdul: %r' % sys.getrefcount(bias_hdul))

    # Now we normalize so that we are combining images of same exposure time
    images = _normalize_exptimes(images)

    # And then we normalize them by their mean
    images = [hdul.flatten() for hdul in images]

    # Modify args accordingly
    args = (images,) + args[1:]

    return imcombine(*args)


def imcombine(images_list: list, root_dir=None, this_type="",
              save_dir=None, combine='median', overwrite=True) -> str:
    """
    Warning! This function is quite inefficient with memory handling. Use with caution!
    Function will create a median FITS image retaining all header information and save it, then return
    the filepath to median file.
    :param images_list: list of filepaths to FITS files, or a list of HDULists, or a list of lists
    with data (numpy array) per extension
    :param root_dir: Only Required when given image_list is a list of HDULists or numpy data AND save_dir is not given.
    :param this_type: this isn't required, it simply gives context to the file; stirng is appended to save_path
    :param save_dir: directory path to which save final image
    :param combine: combination method; 'mean' or 'median'
    :param overwrite: whether the final image should be overwritten
    :return: filepath to final image
    """
    log('\nStarting imcombine. Sample:\n%s' % images_list[0])
    log('Saving to %s' % save_dir)

    combine = combine.lower()
    # make sure combine is either 'median' or 'mean'
    assert combine == 'median' or combine == 'mean', 'Incompatible parameter given for "combine" argument.'

    # find root dir for the given files (it will be used to save the mean_file)
    is_str = isinstance(images_list[-1], str)
    root_dir = dirname(images_list[-1]) if root_dir is None and is_str else root_dir
    save_dir = root_dir if save_dir is None else save_dir

    # Make sure these are actual paths
    assert isdir(save_dir) or isdir(root_dir), 'Given root_dir or save_dir are not actual paths!'

    # get save path for combined file
    combine_path = join(save_dir, '{}_{}.fits'.format(combine.upper(), this_type))

    # check if this file already exist
    exists = isfile(combine_path)
    if exists and not overwrite:
        log('File already exists! Skipping! %s' % combine_path)
        return combine_path
    elif exists and overwrite:
        log('Overwriting combined image: %s' % combine_path)

    # Use the last image as template, assumed to be random one, or newest frame
    template_hdul = ModHDUList(images_list[-1], in_mem=True)

    # add history/comment that it is a mean file
    template_hdul[0].header.add_history("Combined calibration image: " + combine.upper())

    # number of extensions
    exts = len(template_hdul)
    log("About to loop over calibration files in imcombine")
    for i in range(exts):
        if template_hdul[i].data is None:
            continue

        log("Getting %s for ext #%d" % (combine, i))

        log('Stacking data!')
        start = time.time()
        # Stack data to get median across images for extension-i
        data_list = [get_data(image, i) for image in images_list]
        end = time.time() - start
        if end > 5.:
            log('Data stacking took %.3f secs' % end)

        try:
            if combine == 'median':
                template_hdul[i].data = np.median(data_list, axis=0, overwrite_input=True)
            else:
                template_hdul[i].data = np.mean(data_list, axis=0, dtype=np.float32)
        except ValueError as e:
            # This error can be thrown if shape mismatch; make sure this is the reason...
            if 'broadcast' not in e.args[0].lower():
                raise

            # get shapes info and update image list
            shapes = [im.shape for im in data_list]
            # the_shape will be the largest shape in shapes; assumed correct one.
            the_shape = sorted(shapes, reverse=True)[0]

            # log info
            log('WARNING! Shapes of images are different. Implementing mask for filtering....')
            log('Detection happened for extension #%d' % i)
            log('Chosen shape as correct CCD shape: {}'.format(the_shape))

            # modify images_list accordingly
            images_list = [images_list[k] for k in len(images_list) if the_shape == shapes[k]]

            # attempt again with modified images_list
            data_list = [get_data(image, i) for image in images_list]
            if combine == 'median':
                template_hdul[i].data = np.median(data_list, axis=0, overwrite_input=True)
            else:
                template_hdul[i].data = np.mean(data_list, axis=0, dtype=np.float32)
        data_list = None

    # if 'FLAT' in this_type.upper():
    #     # flat tends to have negative values that (may) impact badly...
    #     # results may vary!!!
    #     template_hdul.interpolate()

    template_hdul.writeto(combine_path, overwrite=True)
    template_hdul.close()
    return combine_path


def check_comp(filepath, *args):
    """
    Recursive solution to find compatibility of a FITS image against multiple given parameters.
    This function will return True only when attributes of given file match the ones given.

    Attributes rules:
        - Binning: A number/string digit must be given
        - Filter: A non-digit string must be given
        - Twilight: A boolean  must be given indicating whether you allow twilight flats or not.
        - Dimensions: A tuple/list must be given (X, Y) or (NAXIS1, NAXIS2)
    :param filepath: path to FITS image/file
    :param args: arguments that follow the attribute rules listed above
    :return:
    """
    if not args:
        return True
    comp = args[0]
    if comp is True:
        return True and check_comp(filepath, *args[1:])
    elif comp is False:
        # assume we're getting asked about twilight_flats
        # check if it is a twilight_flats
        a = "twilight" in filepath.lower()
        b = "twilight" in find_val(filepath, "object").lower()
        z = a or b
        # this covers whether what happens if we are looking for twilight_flats
        return not z
    elif isinstance(comp, (list, tuple)):
        # assume we're getting asked about dimensions of data
        return find_dimensions(filepath) == comp and check_comp(filepath, *args[1:])
    elif isinstance(comp, (int, float)) or isinstance(comp, str) and all([c.isdigit() for c in comp.split()]):
        # assume we're getting asked about binning of data
        # hardcoded CASSINI exception; no binning keyword in headers
        actual_bin = 1 if 'CASSINI' in filepath else find_val(filepath, "bin")
        return str(comp) == str(actual_bin) and check_comp(filepath, *args[1:])
    else:
        # assume we're getting asked about filter
        actual_filter = find_val(filepath, "filter", is_str=True)
        return str(comp.upper()) == str(actual_filter.upper()) and check_comp(filepath, *args[1:])


def filter_objects(file_path, *args):
    """
    This is only used for target frames!
    Filter out object files using the following attributes:
    
    'FINAL' keyword in filename (an already calibrated file)
    Object fits has different binning/filter as given
    :param file_path: file path to frame
    :param args: arguments passed to check_comp
    :return: boolean determining whether this is an acceptable target frame
    """
    filename = basename(file_path).lower()
    if 'final' in filename or 'calibrated' in filename:
        return False
    elif not check_comp(file_path, *args):
        return False
    elif any([key in filename for key in ['flat', 'zero', 'bias', 'dark']]):
        return False
    else:
        return True


def get_comp_info(obj_address):
    """
    This function gets the compatibility parameters for the files from just one (random) file in the
    object file list. This function will also test if data doesn't have rdnoise/gain info.
    :param obj_address: this is the parsed address of the object files.
    :return: a list containing the BIN #, and the filter of the object.
    """
    if isfile(obj_address):
        file_name = obj_address
    else:
        file_name = next(search_all_fits(obj_address))
    hdul = ModHDUList(file_name)
    num_amps = len(hdul)
    if num_amps > 1 and hdul[0].data is None:
        num_amps -= 1
    gains = find_gain(hdul[0].header, num_amps)
    rdnoises = find_rdnoise(hdul[0].header, num_amps)
    m = False if gains is None or rdnoises is None else True
    bins = 1 if 'CASSINI' in obj_address else find_val(file_name, "bin")
    y = [i for i in bins if i.isdigit()][0] if isinstance(bins, str) else bins
    x = find_val(file_name, "filter", is_str=True)
    z = find_val(file_name, "exptime")
    comp = [x, y, z, m]
    # log(comp)
    for i in comp[:-1]:
        if not i:
            raise Exception("The program could not find the necessary info, this file is not compatible")
    return comp


def arguments_bd(kind, cals_tree, combs_tree, calpath):
    """
    Helper function generator for update_calibrations function.

    Arguments function that generates the positional arguments needed
    to compute combination of bias/flats/darks.
    Arguments to yield for imcombine, see imcombine for args overview

    This is a helper function for the multiprocessing capability
    of update_calibrations.

    :param kind: type target to compute combination: bias/flat/darks
    :param cals_tree: dictionary representing the directory structure of Calibrations folder
    :param combs_tree: dictionary representing the directory structure of Calibrations folder
    :param calpath: calibrations directory folder up to telescope's choice.
                    e.g '/home/Data/Calibrations/GUFI'
    :return: yields arguments
    """
    kind = kind.upper()
    log('Executed arugments_bd for %s' % calpath)
    log('Looking for %s' % kind)

    # Get dates!
    cal_dates = {*cals_tree[kind]}
    comb_dates = {*combs_tree[kind]}

    # false positives; date folders that are empty from COMBINED
    false_positives = {date for date in comb_dates if combs_tree[kind][date] == {}}
    # extras; date folders from COMBINED that don't exist in Calibrations folder.
    extras = comb_dates - cal_dates
    # date folders that have more than one image from Calibrated
    true_cal_dates = {date for date in cal_dates if len(cals_tree[kind][date]) > 1}

    # missing; dates that need to be combined
    missing = (true_cal_dates - comb_dates | false_positives) - extras

    # Let there no mistake that missing are all dates
    missing = fnmatch.filter(missing, '*-*-*')

    log('Missing data sets:\n%r' % (missing,))
    for date in missing:
        # Get rootdir and save dir, then validate
        rootdir = join(calpath, kind, date)
        save_dir = join(calpath.replace('Calibrations', 'COMBINED'), kind, date)
        validate_dirs(save_dir)

        # get pathfiles for all FITS in given date
        im_list = [join(rootdir, f) for f in filter_fits(cals_tree[kind][date])]
        imrange = range(len(im_list))

        # Get parallel list of bin numbers, and another of image dimensions
        if 'CASSINI' in calpath:
            bin_list = len(im_list) * [1]
        else:
            bin_list = [find_val(im, 'BIN') for im in im_list]

        dim_list = [find_dimensions(im) for im in im_list]

        # iterate over unique bin numbers!
        for bins in set(bin_list):
            # Check dimensions of this dataset, and make sure they match
            dims = [dim_list[i] for i in imrange if bin_list[i] == bins]
            max_dim = max(dims)

            imgs = [im_list[i] for i in imrange if bin_list[i] == bins and max_dim == dim_list[i]]
            imgsrange = range(len(imgs))

            if kind == 'FLAT':
                # get all filters of data
                filts = [find_val(im, "filter", is_str=True) for im in imgs]
                # iterate over unique filters
                for filt in set(filts):
                    flats = [imgs[i] for i in imgsrange if filts[i] == filt]
                    yield flats, 'Flats.BIN%d.%s.%s' % (bins, filt, date), save_dir
            else:
                yield imgs, kind + '.BIN%d.%s' % (bins, date), save_dir


def update_calibrations(telescope, method='median', processes=5):
    """
    Perform combination of calibration files; bias, flats, darks
    A combination of calibration files is performed per date.
    :param telescope: telescope for which date to perform
    :param method: method to use for image combinations
    :param processes:
    """
    global arg_wrapper
    assert method == 'median' or method == 'mean', 'Attribute method must be "mean" or "median"'
    log('\nUpdating calibration data for %s' % telescope)

    # get calibrations folder for selected telescope
    cals = join(server_destination, 'Calibrations', telescope)
    comb = join(server_destination, 'COMBINED', telescope)

    # create directories for calibration files,
    validate_dirs(join(comb, 'FLAT'), join(comb, 'BIAS'), join(comb, 'DARK'))

    # get directory tree of telescope's calibrations
    cals_tree, combs_tree = get_calibrations(telescope, both=True)

    # define wrapper for argument generator to complete all args:
    def arg_wrapper(*args):
        args_gen = arguments_bd(*args)
        for images, this_type, save_dir in args_gen:
            yield images, None, this_type, save_dir, method

    # check if there are bias, if so perform combinations for bias per date
    if 'BIAS' in cals_tree:
        log('Performing combinations for bias files per date')
        # initialize multiprocessing pool in context manager
        with Pool(processes=processes) as pool:
            args_gen = arg_wrapper('Bias', cals_tree, combs_tree, cals)
            pool.starmap(imcombine_bias, args_gen)

    # check if there are darks, if so perform combinations for darks per date
    dark_flag = 'DARK' in cals_tree and len(cals_tree['DARK']) > 0
    if dark_flag:
        log('Performing combinations for darks files per date')
        partial_imcombine = partial(imcombine_darks, telescop=telescope)
        # initialize multiprocessing pool in context manager
        with Pool(processes=processes) as pool:
            args_gen = arg_wrapper('Dark', cals_tree, combs_tree, cals)
            pool.starmap(partial_imcombine, args_gen)

    # check if there are flats, if so perform combinations for flats per date
    if 'FLAT' in cals_tree:
        log('Performing combinations for flats files per date')
        partial_imcombine = partial(imcombine_flats, darks=dark_flag, telescop=telescope)
        # initialize multiprocessing pool in context manager
        with Pool(processes=processes) as pool:
            args_gen = arg_wrapper('Flat', cals_tree, combs_tree, cals)
            pool.starmap(partial_imcombine, args_gen)


def find_calibrations(calibrations, filters, bins, central_flag, recycle=False, twi_flat=False, debug=False):
    flats = []
    darks = []
    bias = []
    bias_median = darks_median = flats_median = super_cal_found = False

    if debug:
        log("Searching and filtering calibration files")

    for filepath in calibrations:
        filename = basename(filepath)

        # check binning compatibility
        if not check_comp(filepath, bins):
            if debug:
                log("%s Not Compatible" % filename)
            continue
        if debug:
            log("%s Compatible" % filename)

        # check for type; FLAT/DARK/BIAS
        this_imagetype = find_imgtype(filepath)

        # check if is recyclable file
        if recycle and central_flag in filename:
            if debug:
                log("Super Calibration file found while filtering: %s" % filename)
            if "FLAT" == this_imagetype:
                if check_comp(filepath, filters, twi_flat):
                    flats_median = filepath
            elif "BIAS" == this_imagetype or "ZERO" == this_imagetype:
                bias_median = filepath
            elif "DARK" == this_imagetype:
                darks_median = filepath
            super_cal_found = flats_median and bias_median and darks_median

            # if we have all median/mean files, then no need to keep looking
            if super_cal_found:
                break
            else:
                continue

        # check type and append to corresponding list
        if "DARK" == this_imagetype:
            darks.append(filepath)
            continue
        elif "BIAS" == this_imagetype or "ZERO" == this_imagetype:
            bias.append(filepath)
            continue
        elif "FLAT" == this_imagetype:
            if check_comp(filepath, filters, twi_flat):
                flats.append(filepath)
            continue

    # if no dark frames, then set flag to False
    darks_flag = super_cal_found or len(darks) > 0
    return bias, flats, darks, bias_median, darks_median, flats_median, darks_flag


def trim_reference_image(image, image2) -> ModHDUList:
    """
    Trim HDUList (image) according to reference window of other HDUList (image2)
    :param image: Image to be trimmed: HDUList
    :param image2: Image to use for reference trim section; HDUList
    :returns new_image: New HDUList of trimmed image(s)
    """
    # Make sure these are ModHDULists
    image = ModHDUList(image)
    image2 = ModHDUList(image2)

    # copy the HDUList
    new_image = image.copy()
    for i in range(len(image)):
        if image[i].data is None:
            continue
        secs = get_secs(image2[i], as_ref=True)

        if secs is None:
            continue
        else:
            slc = secs if isinstance(secs[0], slice) else secs[0]
            new_image[i].data = image[i].data[slc]
    return new_image


def last_processing(obj, beta, flats_median, darks_median, bias_median, final_dir):
    """
    New processing for images. Now instead of trying to reopen the FITS files in every 'process'
    :param obj: file path to object frame
    :param beta: normalizing value for dark frames ==>  OBJ_EXPTIME / DARK_EXPTIME
    :param flats_median:
    :param darks_median:
    :param bias_median:
    :param final_dir:
    :return:
    """
    filename = obj.split('/')[-1]
    this_filename = "Calibrated_" + filename
    log("Calibrating object frame: %s" % filename)
    save_to_path = join(final_dir, this_filename)
    # Proceed with calibration!
    obj_image = overscan_sub(ModHDUList(obj))

    # Trim images as needed; if dimensios of obj_image are the same as calibrations, then nothing changes
    bias_im = trim_reference_image(bias_median, obj_image)
    if darks_median:
        super_bias = bias_im + trim_reference_image(darks_median, obj_image) * beta
    else:
        super_bias = bias_im

    final_image = (obj_image - super_bias) / trim_reference_image(flats_median, obj_image)
    # Perform cosmic ray removal...
    final_image = cosmic_ray(final_image)
    # try to get rid of infs/nans/zeroes
    final_image.interpolate()
    final_image[0].header.add_history("Calibrated Image: Bias subtracted, Flattened, and Cosmic Ray Cleansed")
    final_image.writeto(save_to_path, overwrite=True)
    final_image.close()


def cosmic_ray(hdul):
    """
    Cosmic Ray Removal function. Assumes  gain/rdnoise info in headers.
    :param hdul: HDUList
    :return: new HDUList
    """
    num_amps = len(hdul)
    if num_amps > 1 and hdul[0].data is None:
        num_amps -= 1
    gains = find_gain(hdul[0].header, num_amps)
    rdnoises = find_rdnoise(hdul[0].header, num_amps)
    kwargs = {'sigclip': 4.5, 'objlim': 6, 'gain': 2., 'readnoise': 6.,
              'sepmed': True, 'cleantype': "idw", "verbose": False}
    if gains is None:
        gains = np.ones(num_amps) * 2.
    if rdnoises is None:
        rdnoises = np.ones(num_amps) * 6.

    amp_count = 0
    for i in range(len(hdul)):
        if hdul[i].data is None:
            continue
        # update keywords
        kwargs['gain'] = gains[amp_count]
        kwargs['readnoise'] = rdnoises[amp_count]

        data = hdul[i].data
        newdata, mask = cosmicray_lacosmic(data, **kwargs)
        hdul[i].data = newdata / gains[amp_count]
        amp_count += 1
    hdul[0].header.add_history('LaPlacian Cosmic Ray removal algorithm applied')
    hdul[0].header.add_history(f'Read Noise used: {rdnoises}')
    hdul[0].header.add_history(f'Gain used: {gains}')
    return hdul


if __name__ == '__main__':
    sample = join(server_destination, 'RAW/CAHA/LP_412-31/2018-09-22/MPIA_0220.fits')
    bias = join(server_destination, 'Calibrations/CAHA/BIAS/2018-09-22/MPIA_0013.fits')

    hdul = ModHDUList(sample)
    bias_im = ModHDUList(bias)

    slcs = get_secs(hdul[0])
