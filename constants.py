import logging
import os
import re
from configparser import ConfigParser
from glob import iglob
import numpy as np

import jdcal
from astropy.io import fits
from dateutil import parser

from astropy.convolution import Gaussian2DKernel, interpolate_replace_nans, convolve_fft
from string import Formatter

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
import warnings
from astropy.utils.exceptions import AstropyWarning
warnings.simplefilter('ignore',category=AstropyWarning)

# Formatting/functions for logging
FORMAT1 = "%(message)s"
edenAP_path = os.path.abspath(os.path.dirname(__file__))
log_folder = os.path.join(edenAP_path, 'EDEN Logging')
if not os.path.isdir(log_folder):
    os.mkdir(log_folder)
logging.basicConfig(filename=os.path.join(log_folder, 'edenAP.log'), format=FORMAT1, level=logging.INFO)
log = logging.info

# This is the server destination in the current computer
config = ConfigParser()
config.read(edenAP_path+'/config.ini')
server_destination = config['FOLDER OPTIONS']['server_destination']

# EDEN Database shortcuts for mysql interactions
EDEN_data_cols = "`Filename`,`Directory`,`DATE-OBS`,`MJD`,`Telescope`,`Observer`," \
                 "`Target`,`Filter`,`Integration time [s]`,`Airmass`,`Instrument`," \
                 "`RA`,`Dec`,`Data Quality`,`RA hh`,`RA mm`,`RA ss.ss`,`Dec dd`,`Dec mm`,`Dec ss.ss`"

EDEN_data_vals = "'{}',\"{}\",'{}',{:1.3f},\"{}\",\"{}\",'{}',\"{}\",{:1.3f},{:1.3f},\"{}\","" \
                    ""{:1.3f},{:1.3f},{:1.3f},{:t},{:t},{:1.3f},{:t},{:t},{:1.3f}"

# Open filters.dat to determine the filter sets
filter_sets = {}
for line in open(edenAP_path+'/filters.dat','r').readlines():
    if line.strip() == '' or line.strip()[0] == '#': continue
    keys = line.strip().split()
    filter_sets[keys[0]] = keys

# Server localizers
def get_telescopes():
    """
    Get the telescope names of the current ones in the server
    :return: list of telescope names
    """
    telescope_dirs = iglob(os.path.join(server_destination, 'RAW/*'))
    telescopes = [telescope_dir.split('/')[-1] for telescope_dir in telescope_dirs]
    return telescopes


def get_target_dates(calibrated=True, telescope=None):
    """
    get target dates for calibrated/raw targets in our server
    :param calibrated: if True, only dates in the directory of calibrated objects returned, if False, only raw ones
    :param telescope: telescope for which to find the targets; if None, then find for all telescopes
    :return: list of dates
    """
    add2server = ['*', 'cal', '*', '*', '*']
    if telescope is not None:
        add2server[0] = telescope.upper()
    if not calibrated:
        add2server[1] = 'raw'
    date_dirs = iglob(os.path.join(server_destination, *add2server))
    dates = {date_dir.split('/')[-1] for date_dir in date_dirs}
    return sorted(list(dates))


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

def find_val(filepath_header, keyword, ext=0, comment=False, regex=False, typ=None):
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
    if isinstance(filepath_header, fits.Header):
        hdr = filepath_header
    else:
        with fits.open(filepath_header) as hdul:
            hdr = hdul[ext].header
    return_val = None

    # Before attempting brute search. Try getting the value
    try:
        if not regex:
            return_val = hdr[keyword]
        else:
            raise KeyError
    except KeyError:
        for key, val in hdr.items():
            if regex:
                if re.search(keyword, key):
                    return_val = val
                elif re.search(keyword, hdr.comments[key]):
                    return_val = val
            else:
                inKeyword = keyword.upper() in key.upper()
                inComment = keyword.upper() in hdr.comments[key].upper()
                if inKeyword:
                    return_val = val
                if comment and inComment:
                    return_val = val
            if return_val is not None:
                if (typ is not None) and (typ is not type(return_val)):
                    comment = hdr.comments[key].strip('/').strip()
                    return_val = comment
                break
        else:
            raise
    return return_val


def getjd(date):
    """
    get Julian Date given Gregorian Date as string or datetime object
    :param date: date string or datetime object
    :return: julian date
    """
    if isinstance(date, str):
        date = parser.parse(date)
    return sum(jdcal.gcal2jd(int(date.year), int(date.month), int(date.day)))


# function to find correct date in header
def LOOKDATE(header):
    """
    Persistent function that will look for the date of the observation recorded in the header.

    Procedure:
    1. Looks for 'DATE', 'DATE-OBS' or anything including 'DATE' in header.
    2. Tests format 'YYYY-MM-DDTHH:MM:SS.ss', or simply 'YYYY-MM-DD' or 'YYYY/MM/DD'
    3. If format doesn't include the time, it looks for 'UT' keyword to find time and appends it to the date string

    :param header: header of current file
    :return: datetime object
    """
    try:
        # find_val will first try to get 'DATE' header. If it doesn't work, it will find header keywords that
        # include the word 'DATE' which includes 'DATE-OBS'
        date_key = 'DATE-OBS' if 'DATE-OBS' in header else 'DATE'
        date = find_val(header, date_key)
        if "T" in date:
            temp_date = parser.parse(date)
        else:
            try:
                time = find_val(header, 'UT')
                if '/' in time or '-' in time or ':' not in time:
                    # if 'UT' value suggests date string, then raise err
                    raise KeyError
                temp_date = parser.parse(date + 'T' + time)
            except KeyError:
                time_key = 'TIME-OBS' if 'TIME-OBS' in header else 'TIME'
                time = find_val(header, time_key)
                temp_date = parser.parse(date + 'T' + time)
    except (KeyError, TypeError):
        date = find_val(header, 'DATE')
        temp_date = parser.parse(date)
    return temp_date


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

def copy(source, dest):
    """
    wrapper function that uses unix system's copy function: `cp -n`
    :param source: source file
    :param dest: destination file/folder
    :return:
    """
    dest = new_destination(source, dest)
    proc = subprocess.Popen(['cp', '-n', source, dest], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return_code = proc.wait()
    if return_code == 1:
        log("Copy Function encountered Error. File somehow exists already.")
    return return_code

def mv(source, dest):
    """
    wrapper function that uses unix system's move function: `mv -n`
    :param source: source file
    :param dest: destination file/folder
    :return:
    """
    dest = new_destination(source, dest)
    proc = subprocess.Popen(['mv', '-n', source, dest], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return_code = proc.wait()
    if return_code == 1:
        log("Move Function encountered Error. File somehow exists already.")
    return return_code

def check_and_rename(file, add=0):
    """
    Quick function to
    :param file:
    :param add:
    :return:
    """
    original_file = file
    if add != 0:
        split = file.split(".")
        ext = split[-1]
        before_ext = '.'.join(split[:-1])
        part_1 = before_ext + "_" + str(add)
        file = ".".join([part_1, ext])
    if not os.path.isfile(file):
        return file
    else:
        add += 1
        check_and_rename(original_file, add)

class ModHDUList(fits.HDUList):
    # class attribute is the kernel
    kernel = Gaussian2DKernel(5)

    def __init__(self, hdus=[], interpolate=False, **kwargs):
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
        if type(hdus) is str:
            hdus = fits.open(hdus)
        # validate dimensions of data
        for i in range(len(hdus)):
            data: np.ndarray = hdus[i].data
            if data is None:
                continue
            # if data is three-dimensional and first axis is equals 1 (first axis is 3rd coordinate)
            if len(data.shape) > 2 and data.shape[0] == 1:
                # then reshape to 2 dimensions
                shaped_data = data.reshape((data.shape[1:]))
                hdus[i].data = shaped_data.astype(float)
        super(ModHDUList, self).__init__(hdus, **kwargs)
        if interpolate:
            self.interpolate()

    def interpolate(self):
        """
        interpolate zeros and negative values in data using FFT convolve function
        """
        for hdu in self:
            if hdu.data is None:
                continue
            with np.errstate(invalid='ignore'):
                non_finite = ~np.isfinite(hdu.data)
                less_zero = hdu.data <= 0
                if np.any(less_zero) or np.any(non_finite):
                    data = hdu.data.astype(float)
                    mask_data = np.ma.masked_less_equal(data, 0)
                    # mask_data = np.ma.masked_inside(data, -1e5, 0)
                    mask_data.fill_value = np.nan
                    data = mask_data.filled()
                    data = interpolate_replace_nans(data, self.kernel, convolve=convolve_fft, allow_huge=True)
                    hdu.data = data

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
        :param hdul:
        :return:
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
                hdu_data = hdul[i].data.astype(float)
                data = self[i].data.astype(float) - hdu_data
            elif arrays_flag:
                # assuming hdul is a list of ndarrays
                data = self[i].data.astype(float) - hdul[i].astype(float)
            else:
                # assuming hdul is a constant
                data = self[i].data.astype(float) - hdul
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
                hdu_data = hdul[i].data.astype(float)
                data = self[i].data.astype(float) / hdu_data
            elif arrays_flag:
                # assuming hdul is a list of ndarrays
                data = self[i].data.astype(float) / hdul[i].astype(float)
            else:
                # assuming hdul is a constant
                data = self[i].data.astype(float) / hdul
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
                hdu_data = hdul[i].data.astype(float)
                data = self[i].data.astype(float) + hdu_data
            elif arrays_flag:
                # assuming hdul is a list of ndarrays
                data = self[i].data.astype(float) + hdul[i].astype(float)
            else:
                # assuming hdul is a constant
                data = self[i].data.astype(float) + hdul
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
                hdu_data = hdul[i].data.astype(float)
                data = self[i].data.astype(float) * hdu_data
            elif arrays_flag:
                # assuming hdul is a list of ndarrays
                data = self[i].data.astype(float) * hdul[i].astype(float)
            else:
                # assuming hdul is a constant
                data = self[i].data.astype(float) * hdul
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
            raise ValueError('Method {} doesn\'t exist please enter "median" or "mean"'.format(method))
        method = getattr(np, method)
        flatten_hdul = ModHDUList([hdu.copy() for hdu in self])
        for i in range(self.len()):
            if self[i].data is None:
                continue
            data = flatten_hdul[i].data
            flatten_hdul[i].data = data / method(self[i].data)
        flatten_hdul[0].header.add_history('FITS has been flattened by its {}'.format(method))
        return flatten_hdul

    def median(self):
        """
        Get median of all pixels in all extensions
        """
        return np.nanmedian([hdu.data.astype(float) for hdu in self if hdu.data is not None])

    def mean(self):
        """
        Get mean of all pixels in all extensions
        """
        return np.nanmean([hdu.data.astype(float) for hdu in self if hdu.data is not None])

    def std(self):
        """
        Get standard deviation of all pixels in all extensions
        """
        return np.nanstd([hdu.data.astype(float) for hdu in self if hdu.data is not None])# STANDARD LIST OF TELESCOPES AND TYPES OF CALIBRATIONS IMAGES, UPDATE WHEN NEEDED

# Simple class to avoid invalid integer to float implicit conversion when formatting a string
# Use... MyFormatter().format("{0} {1:t}", "Hello", 4.567)  # returns "Hello 4"
class MyFormatter(Formatter):
    """
    Simple class to avoid invalid integer to float implicit conversion when formatting a string.
    Usage:
    MyFormatter().format("{0} {1:t}", "Hello", 4.567)  # returns "Hello 4"
    """
    def format_field(self, value, format_spec):
        if format_spec == 't':  # Truncate and render as int
            return str(int(value))
        return super(MyFormatter, self).format_field(value, format_spec)


