import resource
import os
import random
import re
from datetime import datetime
from gc import collect
from multiprocessing import Pool
import time
import objgraph
import psutil


import numpy as np
from astropy import units as u
from astropy.io import fits
from astropy.modeling import models
from astropy.nddata import CCDData
from ccdproc import trim_image, subtract_overscan, cosmicray_lacosmic

from DirSearch_Functions import search_all_fits, set_mulifits
from constants1 import log, ModHDUList, check_and_rename

todays_day = lambda: datetime.today()

global num_flats, num_bias, num_darks


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [atoi(c) for c in re.split('(\d+)', text)]


def create_generator(iterator):
    for obj in iterator:
        yield obj


def memory():
    # return the memory usage in MB
    import psutil
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / float(2 ** 20)
    return mem


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    chunks_list = []
    for i in range(0, len(l), n):
        chunks_list.append(l[i:i + n])
    return chunks_list


def replace_or_inf(hdul, constant, zero=False, infi=False):
    for hdu in hdul:
        if hdu.data is None:
            continue
        zero_indx = hdu.data == 0
        inf_indx = hdu.data == np.inf
        if zero:
            hdu.data[zero_indx] = constant
        if infi:
            hdu.data[inf_indx] = constant


def get_gain_rdnoise(calibration, bins=2, filters=None, twi_flat=False):
    basename = os.path.basename
    bias = []
    append_bias = bias.append
    flats = []
    append_flats = flats.append
    medbias_path = None
    if not filters:
        return None
    # edit for calibrations in different folders
    all_calibs = []
    for i in range(len(calibration)):
        for j in search_all_fits(calibration[i]):
            all_calibs.append(j)
    for filepath in all_calibs:
        filename = basename(filepath)
        this_imagetype = find_imgtype(filepath)
        if not check_comp(filepath, bins):
            continue
        # avoid all median/mean files, except bias
        if 'MEDIAN' in filename or 'MEAN' in filename.upper():
            if "BIAS" in filename.upper():
                medbias_path = filepath
            continue
        if "BIAS" == this_imagetype or "ZERO" == this_imagetype:
            append_bias(filepath)
            continue
        elif "FLAT" == this_imagetype:
            if check_comp(filepath, filters, twilight_flats=twi_flat):
                append_flats(filepath)
            continue
    # limit search
    assert bias and flats, "Either bias or flats files were not detected."
    if len(flats) > 100:
        flats = random.sample(flats, 100)
    if len(bias) > 100:
        bias = random.sample(bias, 100)
        # files with lowest sigma
    least_sigma = 1e6
    bias1 = bias2 = None
    log('looking for bias files with lowest sigma')
    for bias_file in bias:
        bias_ = ModHDUList(bias_file)
        bias_ = overscan_sub(bias_)
        sigma = bias_.std()
        if sigma < least_sigma:
            log("{:1.2f}\t{:s}".format(sigma, bias_file))
            # log(str(sigma) + bias_file)
            bias2 = bias1
            bias1 = bias_
            least_sigma = sigma
    log('looking for flats file with lowest sigma')
    flat1 = flat2 = None
    least_sigma = 1e6
    # we need to account for bias noise
    if not medbias_path:
        medbias_path = cross_median(bias, this_type='Bias')
    medbias = ModHDUList(medbias_path)
    for flat_file in flats:
        flat = ModHDUList(flat_file) - medbias
        flat = overscan_sub(flat)
        sigma = flat.std()
        if sigma < least_sigma:
            log("{:1.2f}\t{:s}".format(sigma, flat_file))
            flat2 = flat1
            flat1 = flat
            least_sigma = sigma
    collect()
    # MAKE SURE THERE ARE TWO FILES:
    if flat1 is None or flat2 is None:
        flat1 = random.sample(flats, 1)[0]
        diff = set(flats) - {flat1}
        flat2 = random.sample(list(diff), 1)[0]
        flat1 = ModHDUList(flat1)
        flat2 = ModHDUList(flat2)
    if bias1 is None or bias2 is None:
        bias1 = random.sample(bias, 1)[0]
        diff = set(bias) - {bias1}
        bias2 = random.sample(list(diff), 1)[0]
        bias1 = ModHDUList(bias1)
        bias2 = ModHDUList(bias2)
    # We must collect up all readnoise/gain per amplifier, store them in 'gains' and 'read_noises' in order
    # read_noises are read noises in units of ADU
    bias_diff = bias1 - bias2
    read_noise_calc = lambda i: bias_diff[i].data.std() / np.sqrt(2)
    read_noises = np.array([read_noise_calc(i) for i in range(len(bias1)) if bias1[i].data is not None])
    log("READ_NOISES:\t{} ADU".format(read_noises))
    # Now we get the gain using the flats
    normflat = flat2.flatten()
    flatcorr = flat1 / normflat
    if len(read_noises) == len(flat1) - 1:
        gain_calc = lambda i: flatcorr[i].data.mean() / (0.5 * flatcorr[i].data.std() ** 2 - read_noises[i - 1] ** 2)
    else:
        gain_calc = lambda i: flatcorr[i].data.mean() / (0.5 * flatcorr[i].data.std() ** 2 - read_noises[i] ** 2)
    gains = np.array([gain_calc(i) for i in range(len(flat1)) if flat1[i].data is not None])
    read_noises_e = gains * read_noises
    return gains, read_noises_e


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
        option3 = filename = os.path.basename(filepath_header).upper()
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


def find_val(filepath_header, keyword, ext=0, comment=False, regex=False, typ=None, raise_err=True):
    """
    This function takes a keyword and finds the FIRST matching key in the header and returns the its value.
    :param filepath_header: filepath for the file, filepath can also be a header
    :param keyword: keyword for the key header
    :param ext: extension to look for header. Default 0
    :param comment: Look for match in keyword comments. Default False
    :param regex: Look for match using regular expression; re.search function.
    :param typ:Type of object that you want returned. If keyword match, and value type is wrong, its comment is returned
    :return: value corresponding the key header. String or Float. Returns None if no match
    """
    hrd = get_header(filepath_header, ext=ext)
    return_val = None

    # Before attempting brute search. Try getting the value directly
    try:
        if not regex:
            return_val = hrd[keyword]
        else:
            raise KeyError
    except KeyError:
        for key, val in hrd.items():
            if regex:
                if re.search(keyword, key):
                    return_val = val
                elif re.search(keyword, hrd.comments[key]):
                    return_val = val
            else:
                inKeyword = keyword.upper() in key.upper()
                inComment = keyword.upper() in hrd.comments[key].upper()
                if inKeyword:
                    return_val = val
                if comment and inComment:
                    return_val = val
            if return_val is not None:
                if (typ is not None) and (typ is not type(return_val)):
                    comment = hrd.comments[key].strip('/').strip()
                    return_val = comment
                break
        else:
            if raise_err:
                raise
            else:
                return_val = None
    return return_val


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


# input FITS filename path
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


def get_scan(hdu):
    x_range, y_range = hdu.header['BIASSEC'].strip('][').split(',')
    xstart, xend = [int(j) for j in x_range.split(':')]
    xstart -= 1;
    xend -= 1
    ystart, yend = [int(j) for j in y_range.split(':')]
    ystart -= 1;
    yend -= 1
    return hdu.data[ystart:yend, xstart:xend]


def overscan_sub(hdul):
    exts = len(hdul)
    trimmed_hdul = ModHDUList([hdu.copy() for hdu in hdul])
    for i in range(exts):
        if hdul[i].data is not None:
            if 'BIASSEC' in hdul[i].header and 'TRIMSEC' in hdul[i].header:
                data = hdul[i].data
                ccdData = CCDData(data, unit=u.adu)
                poly_model = models.Polynomial1D(3)
                oscan_subtracted = subtract_overscan(ccdData, fits_section=hdul[i].header['BIASSEC'], model=poly_model)
                trimmed_hdul[i].data = trim_image(oscan_subtracted, fits_section=hdul[i].header['TRIMSEC']).data
    return trimmed_hdul


# input list of filepaths to FITS files
def cross_median(images_list, root_dir=None, this_type="", twilight=False):
    """
    Function will create a median FITS image retaining all header information and save it, then return
    the filepath to median file.
    :param images_list: list of filepaths to FITS files, or a list of HDULists, or a list of lists 
    with data (numpy array) per extension
    :param root_dir: Only Required when given image_list is a list of HDULists or numpy data
    :param this_type:
    :return: filepath to median file
    """
    if root_dir is None:
        # find root dir for the given files (it will be used to save the mean_file)
        root_dir = os.path.dirname(images_list[0])
    if twilight:
        this_type = "Twilight-" + this_type
    median_dir = os.path.join(root_dir, "MEDIAN:" + this_type + ".fits")
    # the first image as template (any would work)
    template_hdul = ModHDUList(images_list[0])
    # add history/comment that it is a median file
    if twilight:
        template_hdul[0].header.add_history("TWILIGHT FLAT")
    template_hdul[0].header.add_history("Pre-processed MEDIAN. {}".format(todays_day()))
    # number of extensions
    exts = len(template_hdul)
    for i in range(exts):
        if template_hdul[i].data is None:
            continue
        data_list = [ModHDUList(image)[i].data.astype(float) for image in images_list]
        template_hdul[i].data = np.median(data_list, axis=0)
        del data_list
    if 'FLAT' in this_type.upper():
        # flat tends to have negative values that impact badly...
        template_hdul.interpolate()
    template_hdul.writeto(median_dir, overwrite=True)
    template_hdul.close()
    return median_dir


def cross_mean(images_list, root_dir=None, this_type="", twilight=False):
    """
    Function will create a mean FITS image retaining all header information and save it, then return
    the filepath to mean file.
    :param images_list: list of filepaths to FITS files
    :param root_dir: Only give this parameter if the given image_list is a list of HDULists
    :param this_type:
    :return: filepath to mean file
    """
    if root_dir is None:
        # find root dir for the given files (it will be used to save the mean_file)
        root_dir = os.path.dirname(images_list[0])
    if twilight:
        this_type = "Twilight-" + this_type
    mean_dir = os.path.join(root_dir, "MEAN:" + this_type + ".fits")
    # use first image as template, any would work
    template_hdul = ModHDUList(images_list[0])
    # add history/comment that it is a mean file
    if twilight:
        template_hdul[0].header.add_history("TWILIGHT FLAT")
    if "DARK" in this_type.upper():
        template_hdul[0].header.add_history("BIAS SUBTRACTED")
    if "FLAT" in this_type.upper():
        template_hdul[0].header.add_history("BIAS/DARK SUBTRACTED")
    template_hdul[0].header.add_history("Pre-processed MEAN. {}".format(todays_day()))
    # number of extensions
    exts = len(template_hdul)
    for i in range(exts):
        if template_hdul[i].data is None:
            continue
        data_list = [ModHDUList(image)[i].data.astype(float) for image in images_list]
        template_hdul[i].data = np.mean(data_list, axis=0)
        del data_list
    if 'FLAT' in this_type.upper():
        # flat tends to have negative values that impact badly...
        template_hdul.interpolate()
    template_hdul.writeto(mean_dir, overwrite=True)
    template_hdul.close()
    return mean_dir


##function takes path of FITS, bin to test. and outputs the respective boolean if the binning is correct
def check_comp(filepath, comp, twilight_flats=False):
    """
    function takes a path of a fits, bin, filter or both and outputs the respective boolean if parameters
    of the given file matches the given parameters.
    :param filepath: filepath to fits
    :param comp: binning and filter (either/both) of filepath to be tested against.
    :param twilight_flats: bool indictating whether we are looking for twilight flats
    :return: boolean corresponding whether the parameters match
    """
    bool_list = []
    # check if it is a twilight_flats
    a = "twilight" in filepath.lower()
    b = "twilight" in find_val(filepath, "object").lower()
    z = a or b
    # this covers whether what happens if we are looking for twilight_flats
    if not twilight_flats and not z:
        z = True
    elif not twilight_flats and z:
        z = False
    bool_list.append(z)
    # specific behavior if comp is a list
    if type(comp) is list or type(comp) is tuple:
        # print('CHECKING COMP BIN/Filter: {}\t{}'.format(os.path.basename(filepath), comp))
        filters, bins = comp
        actual_bin = 1 if 'CASSINI' in filepath else find_val(filepath, "bin")
        actual_filter = find_val(filepath, "filter", typ=str)
        x = str(bins) in str(actual_bin)
        y = str(filters.upper()) in str(actual_filter.upper())
        bool_list.append(x)
        bool_list.append(y)
        # return x and y and z
    else:
        # else it must be one value, find which one (bin/filter)
        try:
            # test if comp is an #, therefore testing BIN
            float(comp)
            actual_bin = 1 if 'CASSINI' in filepath else find_val(filepath, "bin")
            mybool = str(comp) in str(actual_bin)
            bool_list.append(mybool)
        except ValueError:
            actual_filter = find_val(filepath, "filter", typ=str)
            x = str(comp.upper()) in str(actual_filter.upper())
            bool_list.append(x)
            # return x and z
    return all(bool_list)


def filter_objects(obj_str, bins):
    """
    filter out object files using the following attributes:
    
    'FINAL' keyword in filename (an already calibrated file)
    object file is a directory
    object fits has different binning
    """
    filename = os.path.basename(obj_str)
    cal_keys = ['flat', 'zero', 'bias', 'dark']
    isCal = any([key in filename.lower() for key in cal_keys])
    if 'FINAL' in obj_str:
        return False
    elif not os.path.isfile(obj_str):
        return False
    elif not check_comp(obj_str, bins):
        return False
    elif isCal:
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
    if os.path.isfile(obj_address):
        file_name = obj_address
    else:
        file_name = list(search_all_fits(obj_address))[0]
    hdul = ModHDUList(file_name)
    num_amps = len(hdul)
    if num_amps > 1 and hdul[0].data is None:
        num_amps -= 1
    gains = find_gain(hdul[0].header, num_amps)
    rdnoises = find_rdnoise(hdul[0].header, num_amps)
    m = True
    if gains is None or rdnoises is None:
        m = False
    bins = 1 if 'CASSINI' in obj_address else find_val(file_name, "bin")
    y = [i for i in bins if i.isdigit()][0] if isinstance(bins, str) else bins
    x = find_val(file_name, "filter", typ=str)
    z = find_val(file_name, "exptime")
    comp = [x, y, z, m]
    # log(comp)
    for i in comp[:-1]:
        if not i:
            raise Exception("The program could not find the necessary info, this file is not compatible")
    return comp


def search_median(calibration, comp, twi_flat=False, recycle=False, median_opt=True, print_out=None):
    """
    This function searches the calibration files in the given calibrations folder. It searches for specific
    binning and filter calibration files.
    :param calibration: address directory of the calibration files necessary for the objects
    :param comp: compatibility information list, binning and filter
    :param twi_flat: True if you want to apply twilight flats or False to use regular flats (or anything is found)
    :param recycle: True if  you want to try to find already calculated super calibration files, False otherwise
    :param median_opt: True if you want to apply MEDIAN to all central tendency applications, False to use MEAN
    :return: A list of the following [ superbias_path, superdark_path, superflats_path, exposureTime_Darks]
    """
    filters, bins = comp[:2]
    global num_flats, num_bias, num_darks, printout
    if print_out is None:
        printout = print
    else:
        printout = print_out.emit
    if median_opt:
        method = 'median'
        central_method = cross_median
    else:
        method = 'mean'
        central_method = cross_mean
    basename = os.path.basename
    # setup flags to use in calculation process
    darks_flag = True
    darks_norm = False
    flats_calculation = True
    bias_calculation = True
    darks_calculation = True
    # setup lists to use in search for fits
    bias = []
    darks = []
    flats = []
    bias_median = False
    darks_median = False
    flats_median = False
    super_cal_found = False
    central_flag = "MEDIAN" if median_opt else "MEAN"
    log("Searching and filtering calibration files")
    # edit for calibrations in different folders
    all_calibs = []
    for i in range(len(calibration)):
        for j in search_all_fits(calibration[i]):
            all_calibs.append(j)
    for filepath in all_calibs:
        filename = basename(filepath)
        # Compatibility test is overriden if calibration is done on CASSINI Files. Due to the fact
        # that there is no binning keyword in their calibrations headers.
        compatibility = '/CASSINI/' in calibration or check_comp(filepath, bins)
        if not compatibility:
            log("%s Not Compatible" % filename)
            continue
        log("%s Compatible" % filename)
        # capture all median/mean files
        if recycle and central_flag in filename:
            log("Super Calibration file found while filtering: %s" % filename)
            if "FLAT" in filename.upper():
                if check_comp(filepath, filters, twilight_flats=twi_flat):
                    flats_median = filepath
            elif "BIAS" in filename.upper():
                bias_median = filepath
            elif "DARK" in filename.upper():
                darks_median = filepath
            super_cal_found = flats_median and bias_median and darks_median
            continue
        this_imagetype = find_imgtype(filepath)
        if "DARK" == this_imagetype:
            darks.append(filepath)
            continue
        elif "BIAS" == this_imagetype or "ZERO" == this_imagetype:
            bias.append(filepath)
            continue
        elif "FLAT" == this_imagetype:
            if check_comp(filepath, filters, twilight_flats=twi_flat):
                flats.append(filepath)
            continue
        if super_cal_found:
            break
    num_bias = len(bias)
    num_darks = len(darks)
    num_flats = len(flats)
    if num_bias == 0 or num_flats == 0:
        printout("Either no bias or flat files found."
                 " If you want to find the 'superFlat' or 'superBias' sperately,"
                 "use cross_median or cross_mean instead of this function. Exiting...")
        return None
    # no existing darks, then set flag to False
    darks_flag = True if super_cal_found else len(darks) > 0
    ################ Evaluate found median files for re-using#######################
    # initialize variables
    if recycle:
        if darks_flag:
            exptime_dark = find_val(darks_median, "exptime") if darks_median else 1
        else:
            darks_median = None
            exptime_dark = 1
        # if all super calibration files are available, then return them
        if bias_median and flats_median and darks_median is not False:
            log('------MEDIANS FILES WERE FOUND------')
            log('{}\t{}\t{}'.format(bias_median, darks_median, flats_median))
            printout('Found MEDIAN Files. Skipping MEDIAN calibration files calculations...'
                     " CONTINUING WITH OBJECT FILES")
            return bias_median, darks_median, flats_median, exptime_dark
        else:
            # if at least one is available then re-use them
            if bias_median:
                log("----------BIAS ALREADY CALCULATED FOUND----------")
                bias_calculation = False
            if darks_median is not False:
                log("----------DARKS ALREADY CALCULATED FOUND----------")
                darks_calculation = False
            if flats_median:
                log("----------FLATS ALREADY CALCULATED FOUND----------")
                flats_calculation = False
    log("----------CALIBRATIONS FILES ARE NOW REDUCING--------")
    printout("Found: {} bias files\n\t{} darks files\n\t{} "
             "flats files\nCalibration files are now reducing".format(num_bias, num_darks, num_flats))
    # this is done for every flat to make in case exptime was changed at some point
    if darks_flag:
        # exptime_dark will be the exptime at all times, normalizing is applied if necessary
        exptime_flats = [find_val(x, 'exptime') for x in flats]
        exptime_darks = {find_val(x, 'exptime') for x in darks}

        exptime_dark = find_val(darks[0], "exptime")
        if len(exptime_darks) > 1:
            printout("Exposure time of darks varies, applying normalizing method")
            darks_norm = True
        # alphas is a list of normalization constants for the darks during subtraction from super flat
        alphas = list(map(lambda x: x / exptime_dark, exptime_flats))
    else:
        darks_median = None
        exptime_dark = 1
        alphas = np.zeros(len(flats))

    ###############################################################################
    ##########################----BIAS CALCULATIONS ----##########################
    if bias_calculation:
        bias_median = central_method(bias, this_type="Bias.BIN{}".format(bins))
        log("----------BIAS CALCULATION COMPLETED----------")
        printout("----------BIAS CALCULATION COMPLETED----------")
    ##########################----END BIAS CALCULATIONS ----##########################
    ##################################################################################

    ###############################################################################
    ##########################----DARKS CALCULATIONS ----##########################
    # open file in memory
    bias_med = fits.open(bias_median, memmap=False)
    if darks_calculation and darks_flag:
        unbiased_darks = []
        for dark in darks:
            raw_dark = ModHDUList(dark)
            if darks_norm:
                # applying darks normalization
                this_exposure = find_val(dark, "exptime")
                if exptime_dark != this_exposure:
                    # subtract bias
                    raw_dark = raw_dark - bias_med
                    # normalize to exptime_dark
                    constant = exptime_dark / this_exposure
                    raw_dark = raw_dark * constant
                    # change EXPTIME to display normalized exptime
                    raw_dark[0].header['EXPTIME'] = exptime_dark
                    unbiased_darks.append(raw_dark)
                    del raw_dark
                    continue
            unbiased_darks.append(raw_dark - bias_med)
            # collect() is python's garbage collector (memory release)
            collect()
        root_darks = os.path.dirname(darks[0])
        darks_median = central_method(unbiased_darks, root_dir=root_darks, this_type="Darks.BIN{}".format(bins))
        # unload the memory space/ a lot of memory must be used for unbiased_darks
        del unbiased_darks
        log("----------DARKS CALCULATION COMPLETED----------")
        printout("----------DARKS CALCULATION COMPLETED----------")
    ##########################----END DARKS CALCULATIONS ----##########################
    ###################################################################################

    ###############################################################################
    ##########################----FLATS CALCULATIONS ----##########################
    if flats_calculation:
        # THIS IS DONE THIS WAY, TO AVOID MISMATCHES OF NORMALIZATION WHEN SUBTRACTING
        root_flats = os.path.dirname(flats[0])
        fixed_flats_list = []
        flat_append = fixed_flats_list.append
        fits.conf.use_memmap = False
        dark_med = ModHDUList(darks_median) if darks_flag else 1
        for i, flat in enumerate(flats):
            raw_flat = ModHDUList(flat)
            super_bias = bias_med - (dark_med * alphas[i]) if darks_flag else bias_med
            # super_bias is either just the bias or the bias minus the dark frame
            calibrate_flat = raw_flat - super_bias
            calibrate_flat = calibrate_flat.flatten(method=method)
            flat_append(calibrate_flat)
            del raw_flat
            collect()
        del bias_med
        if darks_flag:
            del dark_med
        flats_median = central_method(fixed_flats_list, root_dir=root_flats,
                                      this_type="Flats.{}.BIN{}".format(filters, bins), twilight=twi_flat)
        fits.conf.use_memmap = True
        del fixed_flats_list
        log("----------FLATS CALCUATION COMPLETED----------")
        printout("----------FLATS CALCUATION COMPLETED----------")
    ##########################----END FLATS CALCULATIONS ----##########################
    ###################################################################################

    printout("CALIBRATION FILES WERE REDUCED.. CONTINUING WITH OBJECT FILES")
    return bias_median, darks_median, flats_median, exptime_dark


def trim_reference_image(image, image2) -> ModHDUList:
    """
    Trim HDUList (image) according to reference window of other HDUList (image2)
    Uses following range format to trim image:
    DATASEC = [X1, Y1, X2, Y2]
    :param image: Image to be trimmed
    :param image2: Image to use for reference trim section
    :returns new_image: New HDUList of trimmed image(s)
    """
    # copy the HDUList
    new_image = image.copy()
    for i in range(len(image)):
        if image[i].data is None:
            continue
        # copy the hdu
        new_hdu = image[i].copy()
        x1, y1, x2, y2 = eval(image2[i].header['DATASEC'])
        new_hdu.data = new_hdu.data[y1:y2, x1:x2]
        new_image[i] = new_hdu
    return new_image


def prepare_cal(filepath: str, *args) -> (list, tuple):
    """
    Prepare super calibration files by trimming if necessary and returning a list of their data attributes instead of
    the HDUList objects
    :param filepath: filepath to object file to use as reference for trimming
    :param args: calibration files
    :return: calibration data, one list of image data per given calibration file. returned in the same order.
    """
    trimmed = verify_window(filepath, *args)
    new_args = []
    for i in range(len(trimmed)):
        data = []
        for j in range(len(trimmed[i])):
            data.append(trimmed[i][j].data)
        new_args.append(data)
    return new_args


def verify_window(filepath, *args) -> (list, tuple):
    """
    verify existence of windowed file and trim accordingly
    :param filepath: filepath/HDUList to object image which you want to use as reference windowed frame
    :param args: ModHDUList/HDUList objects that you want to trim
    :return: list of given objects trimmed, same order
    """
    window = False
    isWindowed = False
    if '/CAHA/' in filepath:
        # all CAHA images have the DATASEC keyword. We verify if image is windowed by seeing if section
        # is not equal to maximum detector window [0, 0, 4096, 4112]
        window = find_val(filepath, 'DATASEC')
        isWindowed = eval(window) != [0, 0, 4096, 4112]
    elif 'CASSINI' in filepath:
        # Cassini's window verification is different due to the fact that windows are manually set
        # therefore, not every cassini data set will have a 'DATASEC' keyword in the header
        window = find_val(filepath, 'DATASEC') if 'DATASEC' in fits.getheader(filepath) else False
        isWindowed = True if window else False
    if window and isWindowed:
        log("FOUND CAHA WINDOW:\n\tFile:{}\n\tWINDOW:{}".format(os.path.basename(filepath), window))
        ref_window = ModHDUList(filepath)
        new_args = []
        for i in range(len(args)):
            new_args.append(trim_reference_image(args[i], ref_window))
        return new_args
    else:
        log("IMAGE DOES NOT CONTAIN CAHA-LIKE WINDOW FRAME")
        return args


def last_processing2(obj, beta, flats_median, darks_median, bias_median, final_dir):
    """
    New processing for images. Now instead of trying to reopen the FITS files in every 'process'
    :param obj: file path to object frame
    :param beta: normalizing value for dark frames ==>  OBJ_EXPTIME / DARK_EXPTIME
    :param flats_median: list of data attributes of super flats image ==> [data0, data1, data2]
    :param darks_median: list of data attributes of super darks image ==> [data0, data1, data2]
    :param bias_median: list of data attributes of super bias image ==> [data0, data1, data2]
    :param final_dir:
    :return:
    """
    filename = obj.split('/')[-1]
    this_filename = "Calibrated_" + filename
    log("Calibrating object frame: %s" % filename)
    mem = psutil.virtual_memory().percent
    #if mem > 90.:
    #    log("Memory usage spiked to {}%. Skipping frame {}".format(mem, filename))
    #    print('Mem problem')
    #    return False
    save_to_path = os.path.join(final_dir, this_filename)
    obj_image = ModHDUList(obj)
    exts = len(bias_median)
    super_bias = [None if bias_median[i] is None else bias_median[i] + darks_median[i]*beta for i in range(exts)]
    final_image = (obj_image - super_bias) / flats_median
    # RIGHT BEFORE SAVING, MUST DO COSMIC RAY REMOVAL...
    final_image = cosmic_ray(final_image)
    # INTERPOLATE RIGHT BEFORE SAVING
    # try to get rid of infs/nans/zeroes
    final_image.interpolate()
    final_image[0].header.add_history(
        "Calibrated Image: Bias subtracted, Flattened, and Cosmic Ray Cleansed"
    )
    final_image.writeto(save_to_path, overwrite=True)
    final_image.close()
    collect()


def cosmic_ray(hdul):
    num_amps = len(hdul)
    if num_amps > 1 and hdul[0].data is None:
        num_amps -= 1
    gains = find_gain(hdul[0].header, num_amps)
    rdnoises = find_rdnoise(hdul[0].header, num_amps)
    kwargs = {'sigclip': 4.5, 'objlim': 6, 'gain': 2., 'readnoise': 6.,
              'sepmed': True, 'cleantype': "idw", "verbose": True}
    amp_count = 0
    for i in range(len(hdul)):
        if hdul[i].data is None:
            continue
        data = hdul[i].data
        kwargs['gain'] = gains[amp_count]
        kwargs['readnoise'] = rdnoises[amp_count]
        newdata, mask = cosmicray_lacosmic(data, **kwargs)
        hdul[i].data = newdata / gains[amp_count]
        amp_count += 1
    hdul[0].header.add_history('LaPlacian Cosmic Ray removal algorithm applied')
    return hdul


def full_reduction(objects, calibrations, twilight_flats=False, split=3, recycle=False, median_opt=True,
                   print_out=None):
    """
    The function fully reduces the object files given the biases/flats/darks.
    This processing function works for almost all kinds of scenarios, must keep testing.
    :param objects: This can be the address of the object files or a list of selected object files.
    :param calibrations: This can be the address of all calibration files or a list of the following:
    FLATS MEDIAN FILEPATH, DARKS MEDIAN FILEPATH, BIAS MEDIAN FILEPATH, exposure time of darks value.
    :param twilight_flats: Set True, and function will only use twilight flats, else it will use regular ones.
    :param split: Default=3, This parameter will split the reduction into a number of subprocess,
     possibly speeding up the reduction time
    :param recycle: Default=False. If set to True, the program will use a compatible MEDIAN calibration file if found.
    :return: The program saves the final images to a new folder in the given objects-address.
    """
    global printout
    # printout = print_out
    log("\n{0}FULL REDUCTION STARTED{0}\n".format("-" * 7))
    if print_out is None:
        printout = print
    else:
        printout = print_out.emit
    printout("FULL REDUCTION STARTED")
    print(type(objects))
    ############################################################
    # parse directory folders
    flag_cal_list = False
    flag_objs_list = False
    if type(calibrations) is str:
        print("Calibrations variable is a string... assuming it is path to calibrations folder")
        calibrations = calibrations.rstrip(os.sep)
        x = os.path.isdir(calibrations)
    else:
        print("Calibrations variable is a list... assuming it is a list of calibration filepaths")
        mybool_list = []
        for i in range(len(calibrations)):
            calibrations[i] = calibrations[i].rstrip(os.sep)
            mybool_list.append(os.path.isfile(calibrations[i]))
        x = all(mybool_list)
        flag_cal_list = True
    if type(objects) is str:
        print("Objects variable is a string... assuming it is path to object frames folder")
        objects = objects.rstrip(os.sep)
        y = os.path.isdir(objects)
    else:
        print("Objects variable is a list... assuming it is a list of object frames filepaths")
        mybool_list = []
        for i in range(len(objects)):
            objects[i].rstrip(os.sep)
            mybool_list.append(os.path.isfile(objects[i]) or os.path.isdir(objects[i]))
        y = all(mybool_list)
        flag_objs_list = True
    if x and y:
        log("All Input files exist... continuing processing")
    else:
        log("At least one of the input files don't exist, quitting.")
        raise(ValueError('At least one of the input files don\'t exist, quitting.'))

    #############################################################
    # get compatibility factors (filter/binning/exposure time obj), extra one to see if read noise exists in header
    if flag_objs_list:
        comp = get_comp_info(objects[0])
    else:
        comp = get_comp_info(objects)
    filters, bins, exptime_obj, rdnoise_gain = comp
    # calculate median/mean calibration fles using search_median
    if not flag_cal_list:
        # assumes calibrations is a string (directory path)
        bias_median, darks_median, flats_median, exptime_dark = search_median(calibrations,
                                                                              comp,
                                                                              twi_flat=twilight_flats,
                                                                              recycle=recycle,
                                                                              median_opt=median_opt,
                                                                              print_out=print_out)
    elif flag_cal_list:
        # else assumes calibrations files are given
        bias_median, darks_median, flats_median, exptime_dark = calibrations
    else:
        log("Calibrations variable is neither a sequence or a string, quitting...")
        raise ValueError("Calibrations variable is neither a sequence or a string, quitting...")
    # If read noise doesn't exist in at least one header, calculate and put in header files.
    if not rdnoise_gain:
        printout("Information about ReadNoise or Gain couldn't be found... Assigning New Values")
        log("Information about ReadNoise or Gain couldn't be found... Assigning New Values")
        # parse calibrations folder/file path
        cal_folder = calibrations if type(calibrations) is str else os.path.dirname(calibrations)
        mybool_list = [key in cal_folder for key in ['BIAS', 'ZERO', 'FLAT', 'DARK']]
        if any(mybool_list):
            cal_folder = os.path.dirname(cal_folder)
        # setup search for available filters to use
        all_filters = []
        all_filters_append = all_filters.append
        for filepath in search_all_fits(cal_folder):
            image_type = find_imgtype(filepath)
            if image_type == 'FLAT':
                this_filter = find_val(filepath, 'filter')
                all_filters_append(this_filter)
        avfilters = list(set(all_filters))
        log("Found set of filters: {}".format(avfilters))
        # choosing flat frame filter to use, preference given to clear/blank/open filter labels
        filtr = random.choice(list(avfilters))
        for filter_type in ['clear', 'blank', 'open']:
            isFound = False
            for avfilter in avfilters:
                if filter_type.upper() in avfilter.upper():
                    filtr = avfilter
                    break
            if isFound:
                break
        log("Applying get_gain_rdnoise function...")
        gains, read_noises = get_gain_rdnoise(cal_folder, bins=bins, filters=filtr)
        log("get_gain_rdnoise sucessful")
        telescop = '/'.join(cal_folder.split('/')[:-1])
        printout("Assigninig following gain values:\n{}\n...and readnoise:\n{}".format(gains, read_noises))
        for i in range(len(gains)):
            value_set = 'GAIN{}'.format(i + 1), gains[i]
            comm = 'EDEN Corrected Gain of AMP{} in units of e-/ADU'.format(i + 1)
            set_mulifits(telescop, '*.fits', value_set, comment=comm, keep_originals=False)
            value_set = 'RDNOISE{}'.format(i + 1), read_noises[i]
            comm = 'EDEN Corrected Read Noise of AMP{} in units of e-'.format(i + 1)
            set_mulifits(telescop, '*.fits', value_set, comment=comm, keep_originals=False)
        log("Values have been assigned!... Continuing Calibration.")
        printout("Values have been assigned!... Continuing Calibration.")
    # set up object files for calibration
    if flag_objs_list:
        final_dir = os.path.dirname(objects[0]).replace("raw", 'cal')
        if final_dir == os.path.dirname(objects[0]):
            final_dir = os.path.join(os.path.dirname(objects[0]), 'calibrated')
        list_objects = objects
    else:
        final_dir = objects.replace("raw", 'cal')
        list_objects = list(search_all_fits(objects))
        if final_dir == objects:
            final_dir = os.path.join(objects, 'calibrated')
    if not os.path.isdir(final_dir):
        os.makedirs(final_dir)
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
        meddark = [np.zeros(normflats[i].shape) for i in range(len(normflats))]
    lapse = time.time() - t0
    log("Preparation right before calibration took %.4f " % lapse)

    # create arguments list/iterator
    arguments = []
    for obj, beta in zip(filtered_objects, betas):
        # each argument will have an object frame, normalization constant(Beta), and directory names of
        # super calibration files, and final directory for object frames.
        arguments.append((obj, beta, normflats, meddark, medbias, final_dir))
    # initialize multiprocessing pool in try/except block in order to avoid problems
    pool = Pool(processes=split)
    try:
        t0 = time.time()
        pool.starmap(last_processing2, arguments)
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
    log("FULL DATA REDUCTION COMPLETED")
    printout("REDUCTION COMPLETED!")


if __name__ == '__main__':
    # from DirSearch_Functions import walklevel
    # import fnmatch
    import glob

    dirs = glob.glob('/Volumes/home/Data/KUIPER/raw/*/*/*')
    dirs.pop(3)
    cal = '/Volumes/home/Data/KUIPER/Calibrations'
    print(dirs)
    for one_dir in dirs:
        print(" {} STARTED".format(one_dir))
        full_reduction(one_dir, cal, recycle=True)
