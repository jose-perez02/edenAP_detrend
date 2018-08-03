import logging
import os
import re
from configparser import ConfigParser
from glob import iglob

import jdcal
from astropy.io import fits
from dateutil import parser

# This is the server destination in the current computer
config = ConfigParser()
config.read('config.ini')
server_destination = config['FOLDER OPTIONS']['server_destination']

# STANDARD LIST OF TELESCOPES, UPDATE WHEN NEEDED
telescopes_list = ["VATT", "BOK", "KUIPER", "SCHULMAN", "CHAT", "CASSINI", "CAHA"]

# Formatting/functions for logging
FORMAT1 = "%(message)s"
edenAP_path = os.path.abspath(os.path.dirname(__file__))
log_folder = os.path.join(edenAP_path, 'EDEN Logging')
if not os.path.isdir(log_folder):
    os.mkdir(log_folder)
logging.basicConfig(filename=os.path.join(log_folder, 'edenAP.log'), format=FORMAT1, level=logging.INFO)
log = logging.info


# Server localizers
def get_telescopes():
    """
    Get the telescope names of the current ones in the server
    :return: list of telescope names
    """
    telescope_dirs = iglob(os.path.join(server_destination, '*'))
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
            time = find_val(header, 'UT')
            if '/' in time or '-' in time:
                # if 'UT' value suggests date string, then raise err
                raise KeyError
            temp_date = parser.parse(date + 'T' + time)
    except (KeyError, TypeError):
        date = find_val(header, 'DATE')
        temp_date = parser.parse(date)
    return temp_date


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
