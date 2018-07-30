import logging
import os
from glob import iglob

import jdcal
from dateutil import parser

# photometry configuration | First two variables are `computer` dependent
# FOLDER OPTIONS
fpack_folder = "/Users/brackham/utilities/cfitsio/"
astrometry_folder = "/usr/local/Cellar/astrometry-net/0.73/bin/"
# PHOTOMETRY OPTIONS
ASTROMETRY = True
GF_ASTROMETRY = True
REF_CENTERS = True
SEND_EMAIL = False
# USER OPTIONS
emailsender = "mymail@gmail.com"
emailsender_pwd = "MyPassword"
emailreceiver = ["colleage1@gmail.com", "colleage2@mymail.de"]


# This is the server destination in the current computer
server_destination = "/Volumes/home/Data"


# STANDARD LIST OF TELESCOPES, UPDATE WHEN NEEDED
telescopes_list = ["VATT", "BOK", "KUIPER", "SCHULMAN", "CHAT", "CASSINI", "CAHA"]

# Formatting/functions for logging
FORMAT1 = "%(message)s"
if not os.path.isdir('/Users/eden/edenAP/EDEN Logging'):
    os.mkdir('/Users/eden/edenAP/EDEN Logging')
logging.basicConfig(filename='/Users/eden/edenAP/EDEN Logging/EDEN_Util.log', format=FORMAT1, level=logging.INFO)
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

def getjd(date):
    """
    get Julian Date given Gregorian Date as string or datetime object
    :param date: date string or datetime object
    :return: julian date
    """
    if isinstance(date, str):
        date = parser.parse(date)
    return sum(jdcal.gcal2jd(int(date.year), int(date.month), int(date.day)))
