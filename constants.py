import logging
import os
import re
import warnings
from configparser import ConfigParser
from datetime import datetime
from glob import iglob

import numpy as np
from astropy.convolution import Gaussian2DKernel, interpolate_replace_nans, convolve_fft
from astropy.io import fits
from astropy.utils.exceptions import AstropyWarning
from dateutil import parser

from dirs_mgmt import dir_tree, validate_dirs

# useful variables
todays_date = datetime.today()
_prog_dir = os.path.dirname(os.path.abspath(__file__))
filterRegex = re.compile(r'.*\.fi?ts?')
subRegex = re.compile(r'\.fi?ts?')
dtpatt = re.compile(r'(\d{4}-\d{2}-\d{2})')


# STANDARD LIST OF TELESCOPES, UPDATE WHEN NEEDED
telescopes_list = ["VATT", "BOK", "KUIPER", "SCHULMAN", "CHAT", "CASSINI", "CAHA", "LOT", "GUFI"]

# Dictionary of EDEN Available Telescopes. The values are lists with respective labels found in header files
telescopes = {"GUFI": ['gufi', 'vatt_gufi'],
              "BOK": ["bok"],
              "KUIPER": ["kuiper", "bigelow-1.55m"],
              "SCHULMAN": ["schulman", "STX-16803"],
              "CASSINI": ["cassini", "Mt. Orzale 152 cm"],
              "CAHA": ["caha", "CA 1.23m"],
              "LOT": ["lot", "Driver for Princeton Instruments cameras"],
              "VATT": ["!vatt_gufi", "vatt"]}

bad_flags = ['BAD', 'TEST', 'RENAME', 'FOCUS', 'USELESS', 'RANDOM', 'PROVO', 'PROVA']

# String, float, int types
str_types = [str,np.str,np.str_]
float_types = [float,np.float,np.float64,np.float_]
int_types = [int,np.int,np.int64,np.int_]

# Suppress astropy warnings
warnings.simplefilter('ignore',category=AstropyWarning)

# Formatting/functions for logging
FORMAT1 = "%(message)s"
edenAP_path = os.path.abspath(os.path.dirname(__file__))
log_folder = os.path.join(edenAP_path, 'EDEN Logging')
validate_dirs(log_folder)
logging.basicConfig(filename=os.path.join(log_folder, 'edenAP.log'), format=FORMAT1, level=logging.INFO)
log = logging.info

# This is the server destination in the current computer
config = ConfigParser()
config.read(edenAP_path+'/config.ini')
server_destination = config['FOLDER OPTIONS']['server_destination']

# Open filters.dat to determine the filter sets
filter_sets = {}
for line in open(edenAP_path+'/filters.dat','r').readlines():
    if line.strip() == '' or line.strip()[0] == '#': continue
    keys = line.strip().split()
    filter_sets[keys[0]] = keys


# Server localizers

def get_calibrations(telescope, combined=False, both=False):
    """
    Get directory tree of given telescope's calibration files. e.g. /Server/Calibrations/Telescope
    :param telescope: telescope for which to get combined directory tree
    :param combined: If true, then retrieve COMBINED calibration directory tree instead of `Calibrations`
    :param both: If true, then retrieve both directory trees; Calibrations and COMBINED in that order.
    :return: python dictionary representing the directory tree
    """
    comb_dir = os.path.join(server_destination, 'COMBINED', telescope.upper())
    cal_dir = os.path.join(server_destination, 'Calibrations', telescope.upper())

    if both:
        assert os.path.isdir(comb_dir), "COMBINED folder for %s telescope doesn't exist." % telescope
        assert os.path.isdir(cal_dir), "Calibrations folder for %s telescope doesn't exist." % telescope
        return dir_tree(cal_dir), dir_tree(comb_dir)
    elif combined:
        assert os.path.isdir(comb_dir), "COMBINED folder for %s telescope doesn't exist." % telescope
        return dir_tree(comb_dir)
    else:
        assert os.path.isdir(cal_dir), "Calibrations folder for %s telescope doesn't exist." % telescope
        return dir_tree(cal_dir)


def get_telescopes():
    """
    Get the telescope names of the current ones in the server
    :return: list of telescope names
    """
    telescope_dirs = iglob(os.path.join(server_destination, 'RAW/*'))
    telescopes = [telescope_dir.split('/')[-1] for telescope_dir in telescope_dirs]
    return telescopes


# Convenience functions

# advanced function to get values from headers

def shorten_path(path):
    """
    function to shorten a path to a file. For displaying purposes.
    :param path: path to file
    :return:
    """
    path_list = path.split(os.sep)
    if len(path_list) < 5:
        short_path = os.path.join("~", *path_list[:-1])
    else:
        short_path = os.path.join("~", *path_list[-5:-1])
    # path_list = path_list[((len(path_list)+1)// 2):]
    # half_path = os.path.join(path_list[0], *path_list[1:])
    return short_path


def find_dimensions(file_hdul, keyword=None):
    """
    This function will find the dimensions of an image given its address.
    The default keywords for the dimensions are: NAXIS1 and NAXIS2.
    You may enter an optional argument keywords to use use instead of NAXIS.
    If this is a MEF, then we assume all extensions share the same dimensions!
    We therefore only need to check dimensions of ONE extension!

    :param file_hdul: filepath for the file, filepath can also be a HDUList
    :param keyword: keyword for the key header to use instead of NAXIS.
    """
    hdul = ModHDUList(file_hdul)
    nexts = hdul.len()

    keyword = 'NAXIS' if keyword is None else keyword
    xdim, ydim = keyword + '1', keyword + '2'

    if nexts > 1:
        # find dimensions of first one, then make sure its equal for all extensions, then return
        ndims = find_val(hdul[1].header, xdim), find_val(hdul[1].header, ydim)
        all_ndims = [(find_val(hdul, xdim, ext=i), find_val(hdul, ydim, ext=i)) for i in range(2, nexts)]
        if len(all_ndims) > 1:
            assert all([ndim == ndims for ndim in all_ndims]), 'Image has different dimensions along extensions'
    else:
        ndims = find_val(hdul[0].header, xdim), find_val(hdul[0].header, ydim)
    return ndims


def find_val(filepath_header, keyword, ext=0, comment=False, regex=False,
             is_str=False, raise_err=True):
    """
    This function takes a keyword and finds the FIRST matching key in the
    header and returns the its value.

    :param filepath_header: filepath for the file, filepath can also be a header
    :param keyword: keyword for the key header
    :param ext: extension to look for header. Default 0
    :param comment: Look for match in keyword comments. Default False
    :param regex: Look for match using regular expression; re.search(keyword, key)
    :param is_str: If value found isn't a string, its comment is returned.
    :param raise_err: If True, raise error when key header isn't found; else return None
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
        # Initiate intense search in headers.
        for key, val in hrd.items():
            if regex:
                # do regex search on key and comment
                if re.search(keyword, key):
                    return_val = val
                elif comment and re.search(keyword, hrd.comments[key]):
                    return_val = val
            else:
                inkeyword = keyword.upper() in key.upper()
                incomment = keyword.upper() in hrd.comments[key].upper()
                if inkeyword:
                    return_val = val
                if comment and incomment:
                    return_val = val
            if return_val is not None:
                if is_str and not isinstance(return_val, str):
                    return_val = hrd.comments[key].strip('/ =')
                break
        else:
            if raise_err:
                raise
            else:
                return_val = None
    return return_val


def remove_chars(string, chars):
    """
    Function to remove all characters in a string.
    :params string: string to replace chars in
    :params chars: characters to remove in string
    :return: filtered string.
    """
    return ''.join([c for c in string if c not in chars])


def find_dates(string) -> list:
    """
    useful simple function to find all matches to date format: YYYY-MM-DD
    :param string: string where to look for date match
    :return: all found dates
    """
    return dtpatt.findall(string)


def filter_fits(list_files: list) -> list:
    """
    Function to filter FITS files from given list of files.
    """
    fits_files = filter(filterRegex.search, list_files)
    return list(fits_files)


# function to validate directory paths. It creates a path if it doesn't already exist.
def validateDirs(*paths):
    """
    Validate directories. Create directory tree if it doesn't exist. Any number of arguments (paths) are valid.
    """
    for folder_path in paths:
        if not os.path.isdir(folder_path):
            os.makedirs(folder_path)


# natural sorting technique

def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [atoi(c) for c in re.split('(\d+)', text)]


# Identifies filters by their preferred name using the filters.dat file
# If the filter is not identified in filters.dat, just return the input
def id_filters(filters):
    if type(filters) in str_types:
        out = np.copy([filters])
    else:
        out = np.copy(filters)
    for key in filter_sets:
        out[np.in1d(out,filter_sets[key])] = key
    if type(filters) in str_types:
        return out[0]
    else:
        return out


def is_date(key, hdr):
    """
    This function by itself is useless. It will check the given keyword
    to see if its value corresponds to a time/date stamp.
    """
    val = hdr[key]
    is_str = isinstance(val, str)
    # Make sure value is a string and not a digit
    if not is_str or is_str and remove_chars(val, '. ').isdigit():
        return False

    # keys to reject as date/time
    not_time = [rej in key.lower() for rej in ['ra', 'dec', 'ha', 'lst', 'st', 'loctime', 'local']]

    # values to accept as date/time
    yes_time = [acc in val for acc in ['/', ':', '-']]

    # check if any value in not_time is True and check that at least one in yes_time is True
    if any(not_time) or not any(yes_time):
        return False

    try:
        # get potential date,
        date = parser.parse(val)

        # test if date is not actually possible; from last 10 years; SAFE-PLAY
        diff = todays_date - date
        return diff.days < 3650
    except (ValueError, TypeError):
        return False


def find_date(hdr):
    """
    Most general function to find date/time in header
    Implement is_date function to find all values in header that are time
    stamps. The code will return the earliest one. Which should correspond to
    the start exposure time.
    :params hdr: fits header
    :return: datetime object of start of exp time
    """
    # filter keywords in header to get all possible  dates
    dates = filter(lambda k: is_date(k, hdr), hdr.keys())

    # get unique date_time objects from filtered dates
    dates_obj = {parser.parse(hdr[date]) for date in dates}

    # get only times; those keys that only had time will have today's date.
    times = {obj for obj in dates_obj if obj.date() == todays_date.date()}
    # now separate dates and times, sort in descending order
    dates = sorted(dates_obj - times)
    times = sorted(times)

    # check if date-time exists, if it does return!
    for date in dates:
        # only date&time objects will have nonzero .hour/.minute
        if date.hour != 0:
            return date
    # since there is no date-time, create date-time with date/time objs
    return datetime(dates[0].year, dates[0].month, dates[0].day, times[0].hour,
                    times[0].minute, times[0].second, times[0].microsecond)


def get_date(hdr, ext=0):
    """
    Simple function to retrieve the date/time at which the exposure of the current header's image started.
    This function starts by trying to put together values from header into a date/time object.
    If simple procedure doesn't work, then proceed to brute force search.
    :params hdr: header object, filepath to fits file, or HDUList object; Default extension=0.
    :
    """
    hdr = get_header(hdr, ext=ext)
    try:
        if "T" in hdr['DATE-OBS']:
            date = parser.parse(hdr['DATE-OBS'])
        elif 'TIME-OBS' in hdr.keys() and '-' not in hdr['TIME-OBS']:
            date = parser.parse(hdr['DATE-OBS'] + 'T' + hdr['TIME-OBS'])
        elif 'UT' in hdr.keys() and '-' not in hdr['UT']:
            date = parser.parse(hdr['DATE-OBS'] + 'T' + hdr['UT'])
        else:
            raise KeyError
    except (KeyError, TypeError, ValueError):
        date = find_date(hdr)
    return date


def check_arraylist(array_list):
    """
    Helper function for the ModHDUList class. This function will check if the argument is a list of arrays
    :param array_list: any input will be check to match the above
    :return: True only if array_list is a list of numpy arrays
    """
    if isinstance(array_list, list):
        for i in range(len(array_list)):
            if not isinstance(array_list[i], (np.ndarray, type(None))):
                return False
    else:
        return False
    return True


def get_header(filename_hdu, ext=0):
    """
    Get header from filepath of FITS file or HDUList object.
    :param filename_hdu: filepath to file (string) or HDUList object
    :param ext: extension; default [0]
    :return: return the header
    """
    if isinstance(filename_hdu, fits.Header):
        return filename_hdu
    elif isinstance(filename_hdu, list):
        return filename_hdu[ext].header
    else:
        # assume is string giving location of HDUList
        return fits.getheader(filename_hdu, ext=ext)


class ModHDUList(fits.HDUList):
    # class attribute is the kernel
    _kernel = Gaussian2DKernel(5)

    def __init__(self, hdus=[], file=None, interpolate=False, **kwargs):
        """
        This is wrapper around fits.HDUList class
        This class will take all methods and properties of fits.HDUList
        while also being able to perform algebraic operations with the data of each HDU image, and
        contains a interpolate and copy methods. Interpolate will detect any infs/nans or values less than zero and
        interpolate them. Copy method will return a new (clone) instance of the object.

        (New) This class allows algebraic operations to lists of numpy arrays as long as they are the same length as
        the HDUList; len(hdul) == len(array_list) must be true

        :param hdus: filepath to fits file or a fits.HDUList object
        :param interpolate: if True, instance will interpolate negative/zero values in data at construction
        """
        if isinstance(hdus, str):
            hdus = fits.open(hdus, **kwargs)

        # validate dimensions of data; only needed a few times
        for i in range(len(hdus)):
            data: np.ndarray = hdus[i].data
            if data is None:
                continue
            # if data is three-dimensional and first axis is equals 1 (first axis is 3rd coordinate)
            if len(data.shape) > 2 and data.shape[0] == 1:
                # then reshape to 2 dimensions
                shaped_data = data.reshape((data.shape[1:]))
                hdus[i].data = shaped_data.astype(np.float32)
        super(ModHDUList, self).__init__(hdus, file=file)
        if interpolate:
            self.interpolate()

    def interpolate(self):
        """
        interpolate zeros and negative values in data using FFT convolve function
        """
        for i in range(self.len()):
            if self[i].data is None:
                continue
            with np.errstate(invalid='ignore'):
                non_finite = ~np.isfinite(self[i].data)
                less_zero = self[i].data <= 0
                if np.any(less_zero) or np.any(non_finite):
                    data = self[i].data.astype(np.float32)
                    data[less_zero] = np.nan
                    data[non_finite] = np.nan
                    data = interpolate_replace_nans(data, self._kernel, convolve=convolve_fft, allow_huge=True)
                    self[i].data = data

    def len(self):
        """
        :return: the current length of file (number of extensions)
        """
        return len(self)

    def MEF(self):
        """
        :return: flag tells whether self is a multiExtension fits
        """
        return self.len() > 1

    def copy(self):
        """
        create a copy of the HDUList
        :return: the copy will be a new ModHDUList object
        """
        return ModHDUList([hdu.copy() for hdu in self])

    def sub(self, hdul):
        return self.__sub__(hdul)

    def mul(self, hdul):
        return self.__mul__(hdul)

    def truediv(self, hdul):
        return self.__truediv__(hdul)

    def get_data(self, i):
        return self[i].data

    def check_data(self, hdul):
        """
        Check data before operations are applied to it. We allow None's to be in this list because the None is usually
        used instead for an empty data attribute.
        :param hdul: HDUList or list of arrays representing image date
        :return: (hdul_flag, arrays_flat) flags to decide calculation method
        """
        # use flags to tell whether given input is a HDUList or a list of numpy arrays
        hdul_flag = "HDUList" in str(type(hdul))
        arrays_flag = check_arraylist(hdul)
        if hdul_flag or arrays_flag:
            assert len(hdul) == self.len(), "HDULists don't have the same number of extensions"
        return hdul_flag, arrays_flag

    def __sub__(self, hdul):
        hdul_flag, arrays_flag = self.check_data(hdul)
        new_obj = self.copy()
        for i in range(self.len()):
            if self[i].data is None:
                continue
            if hdul_flag:
                # assuming hdul is another hdul
                hdu_data = hdul[i].data.astype(np.float32)
                data = self[i].data.astype(np.float32) - hdu_data
            elif arrays_flag:
                # assuming hdul is a list of ndarrays
                data = self[i].data.astype(np.float32) - hdul[i].astype(np.float32)
            else:
                # assuming hdul is a constant
                data = self[i].data.astype(np.float32) - hdul
            new_obj[i].data = data
        return new_obj

    def __truediv__(self, hdul):
        hdul_flag, arrays_flag = self.check_data(hdul)
        new_obj = self.copy()
        for i in range(self.len()):
            if self[i].data is None:
                continue
            if hdul_flag:
                # assuming hdul is another hdul
                hdu_data = hdul[i].data.astype(np.float32)
                data = self[i].data.astype(np.float32) / hdu_data
            elif arrays_flag:
                # assuming hdul is a list of ndarrays
                data = self[i].data.astype(np.float32) / hdul[i].astype(np.float32)
            else:
                # assuming hdul is a constant
                data = self[i].data.astype(np.float32) / hdul
            new_obj[i].data = data
        return new_obj

    def __add__(self, hdul):
        hdul_flag, arrays_flag = self.check_data(hdul)
        new_obj = self.copy()
        for i in range(self.len()):
            if self[i].data is None:
                continue
            if hdul_flag:
                # assuming hdul is another hdul
                hdu_data = hdul[i].data.astype(np.float32)
                data = self[i].data.astype(np.float32) + hdu_data
            elif arrays_flag:
                # assuming hdul is a list of ndarrays
                data = self[i].data.astype(np.float32) + hdul[i].astype(np.float32)
            else:
                # assuming hdul is a constant
                data = self[i].data.astype(np.float32) + hdul
            new_obj[i].data = data
        return new_obj

    def __mul__(self, hdul):
        hdul_flag, arrays_flag = self.check_data(hdul)
        new_obj = self.copy()
        for i in range(self.len()):
            if self[i].data is None:
                continue
            if hdul_flag:
                # assuming hdul is another hdul
                hdu_data = hdul[i].data.astype(np.float32)
                data = self[i].data.astype(np.float32) * hdu_data
            elif arrays_flag:
                # assuming hdul is a list of ndarrays
                data = self[i].data.astype(np.float32) * hdul[i].astype(np.float32)
            else:
                # assuming hdul is a constant
                data = self[i].data.astype(np.float32) * hdul
            new_obj[i].data = data
        return new_obj

    def __radd__(self, hdul):
        return self.__add__(hdul)

    def __rsub__(self, hdul):
        return self.__sub__(hdul)

    def __rmul__(self, hdul):
        return self.__mul__(hdul)

    def __rtruediv__(self, hdul):
        return self.__truediv__(hdul)

    def flatten(self, method='median'):
        """
        :param method: method to normalize the file; 'median' or 'mean'
        :return: normalized HDUList extension-wise.
        """
        if method != 'median' and method != 'mean':
            raise ValueError('Method "{}" doesn\'t exist please enter "median" or "mean"'.format(method))
        method = getattr(np, method)
        flatten_hdul = ModHDUList([hdu.copy() for hdu in self])
        for i in range(self.len()):
            if self[i].data is None:
                continue
            data = flatten_hdul[i].data
            flatten_hdul[i].data = data / method(self[i].data)
        flatten_hdul[0].header.add_history('FITS has been flattened by its {}'.format(method))
        return flatten_hdul

    def median(self, extension_wise=False):
        """
        Get median of all pixels in all extensions
        """
        if extension_wise:
            return np.array([np.nanmedian(hdu.data) for hdu in self if hdu.data is not None])
        return np.nanmedian([hdu.data for hdu in self if hdu.data is not None])

    def mean(self, extension_wise=False):
        """
        Get mean of all pixels in all extensions
        """
        if extension_wise:
            return np.array([np.nanmean(hdu.data) for hdu in self if hdu.data is not None])
        return np.nanmean([hdu.data for hdu in self if hdu.data is not None])

    def std(self, extension_wise=False):
        """
        Get standard deviation of all pixels in all extensions
        """
        if extension_wise:
            return np.array([np.nanstd(hdu.data) for hdu in self if hdu.data is not None])
        return np.nanstd([hdu.data for hdu in self if hdu.data is not None])
