import os

from astropy import units as u
from astropy.coordinates import Angle
from astropy.io import fits
from astroquery.simbad import Simbad
from dateutil.parser import parse
from dateutil.relativedelta import relativedelta
from jdcal import gcal2jd

from constants import telescopes, bad_flags, log

headers = ['INSTRUME', 'TELESCOP', 'OBJECT']


def hms2dec(hour, min, sec):
    return 15 * (hour + min / 60 + sec / 3600)


def dms2dec(degree, arcmin, arcsec):
    if degree > 0:
        return degree + arcmin / 60 + arcsec / 3600
    elif degree < 0:
        return degree - arcmin / 60 - arcsec / 3600
    else:
        return arcmin / 60 + arcsec / 3600


def remove_chars_iter(subj, chars):
    sc = set(chars)
    return ''.join([c for c in subj if c not in sc])


# function to find correct date in header
def LOOKDATE(header, time_obj=False):
    """
    Persistent function that will look for the date of the observation recorded in the header.
    It will correct for the timezone in order to get the local date/time when the observation started (locally).

    Procedure:
    1. Looks for 'DATE', 'DATE-OBS' or anything including 'DATE' in header.
    2. Tests format 'YYYY-MM-DDTHH:MM:SS.ss', or simply 'YYYY-MM-DD' or 'YYYY/MM/DD'
    3. If format doesn't include the time, it looks for 'UT' keyword to find time and appends it to the date string
    4. Looks for the 'TIMEZONE' keyword in order to correct for it, if not found uses default `7`


    :param header: header of current file
    :param time_obj: if True, it will return a tuple of ( date string, datetime object)
    :return: date string or (date string, datetime object)
    """
    from cal_data import find_val
    try:
        # find_val will first try to get 'DATE' header. If it doesn't work, it will find header keywords that
        # include the word 'DATE' which includes 'DATE-OBS'
        date_key = 'DATE-OBS' if 'DATE-OBS' in header else 'DATE'
        date = find_val(header, date_key)
        if "T" in date:
            temp_date = parse(date)
        else:
            try:
                time = find_val(header, 'UT')
                if '/' in time or '-' in time:
                    # if 'UT' value suggests date string, then raise err
                    raise KeyError
                temp_date = parse(date + 'T' + time)
            except KeyError:
                time_key = 'TIME-OBS' if 'TIME-OBS' in header else 'TIME'
                time = find_val(header, time_key)
                temp_date = parse(date + 'T' + time)
    except (KeyError, TypeError):
        date = find_val(header, 'DATE')
        temp_date = parse(date)
    try:
        time_diffUT = find_val(header, 'timezone')
    except (KeyError, TypeError):
        # try default
        time_diffUT = -7
    # FIRST TO CONVERT TO LOCAL TIME, THEN -7 AS LIMIT IN THE MORNING
    correct_date = temp_date + relativedelta(hours=time_diffUT - 7)
    if time_obj:
        return str(correct_date.date()), correct_date
    return str(correct_date.date())


def get_mjd(header):
    """
    Due to nonexistent standard julian day header... and the inaccuracy in some of them...
    I get the julian header from the starting date/time of the corresponding observation.
    :param header: header of fits file
    :return: Modified Julian Date
    """
    nouse, time_obj = LOOKDATE(header, time_obj=True)
    MJD = gcal2jd(time_obj.year, time_obj.month, time_obj.day)[-1]
    day_fraction = (time_obj.hour + time_obj.minute / 60.0 + time_obj.second / 3600) / 24.0
    return float(MJD + day_fraction)


def get_RADEC(header):
    """
    Get RA / DEC from FITS header no matter what the format is. Epoch. J2000
    Attempt to get from SIMBAD QUERY, if it fails then get RA/DEC of the center of the image (there no other way)
    :param header: fits header
    :return: two list, first one of RA values and other of DEC Values of the following format:
    [ degrees, hour|degree, minute|arcminute, second|arcsecond]
    Where RA is in HMS and DEC is in DMS format.
    """
    from cal_data import find_val
    result = Simbad.query_object(find_val(header, 'object'))
    if result is None:
        raw_RA = find_val(header, 'RA')
        raw_DEC = find_val(header, 'DEC')
    else:
        raw_RA = result['RA'][0]
        raw_DEC = result['DEC'][0]
    # assumes if raw_DEC is float then raw_RA is also a float
    if isinstance(raw_DEC, float):
        # Both are in degrees format
        temp_RA = Angle(raw_RA, unit=u.deg)
        temp_DEC = Angle(raw_DEC, unit=u.deg)
    else:
        # it must be a string, and with the following formats
        temp_RA = Angle(raw_RA, unit=u.hourangle)
        temp_DEC = Angle(raw_DEC, unit=u.deg)
    RA = [temp_RA.deg, *temp_RA.hms]
    DEC = [temp_DEC.deg, *temp_DEC.dms]
    return RA, DEC


# EVERY STRING INFO USED WILL BE UPPERCASE
# CLASS TO CREATE FITFILE
class FITFILE(object):
    def __init__(self, path):
        """
        Lightweight FITFILE object that saves the most relevant information of the image/object as attributes.
        Including filename, observers, integration time, instrument, telescope, RA, DEC, etc...
        :param path: path to fits file
        """
        from cal_data import find_val
        log("Creating FITFILE Object for %s" % path)
        # THIS IS A TAG TO BE CHANGED IF INCOMPATIBILITY IS CATCh
        self.EXIST = True
        self.flag_type = "There is no flag on this file."

        self.path = path
        self.filename = os.path.basename(path)
        path_list = path.split(os.sep)
        if len(path_list) < 4:
            self.short_path = os.path.join("~", *path_list[:-1])
        else:
            self.short_path = os.path.join("~", *path_list[-4:-1])

        self.hdr = fits.getheader(path)

        def find_key(x, **kwargs):
            return find_val(self.hdr, x, **kwargs)

        # find_key = lambda x, **kwargs: find_val(self.hdr, x, **kwargs)

        obs = find_key('OBSERVER')
        self.observer = remove_chars_iter(obs, ['"', '='])

        integration = find_key('EXPTIME', raise_err=False)
        self.integration = float(integration) if integration is not None else float(find_key('EXPOSURE'))
        # Make sure there is no flag regarding it as a 'bad' file
        obj = find_key("OBJECT")
        for bad in bad_flags:
            mybool1 = bad in self.filename.upper()
            mybool2 = bad in obj.upper()
            if mybool1 or mybool2:
                # if flag exists, set flag
                self.EXIST = False
                self.flag_type = "{} contains one of the following flags:" \
                                 "Bad,Test,Rename,Focus,Useless, Provo or Random".format(self.filename)

        # try to find information in headers
        try:
            from cal_data import find_imgtype
            img_type = find_imgtype(self.hdr, self.filename)
            # USE FUNCTION TO LOOK-UP DATE
            self.date = LOOKDATE(self.hdr)
            # USE DICTIONARY OF TELESCOPES TO LOOK FOR MATCH IN HEADER
            instrument = find_key("INSTRUME", raise_err=False)
            self.instrument = remove_chars_iter(instrument, ['"', '=']) if instrument is not None else ''
            telescop = find_key("TELESCOP").upper()
            telFound = False
            for name, label_list in telescopes.items():
                for label in label_list:
                    telFound = label.upper() in telescop
                    telFound = telFound if telFound else label.upper() in self.instrument.upper()
                    counter = label[0] == '!'
                    if telFound and not counter:
                        self.telescop = name
                        if not self.instrument:
                            self.instrument = name
                        break
                if telFound:
                    break
            # now info specific to type of data
            if img_type is not None:
                # If FITS is calibration fits....
                self.type = img_type
                self.name = ""
                self.airmass = 'N/A'
                self.RA = ['N/A', 'N/A', 'N/A', 'N/A']
                if img_type.upper() == 'FLAT':
                    self.filter = find_key('FILTER', typ=str).upper()
            else:
                # else FITS is an OBJECT FITS
                self.type = "OBJECT"
                self.name = obj.upper()  # use name-object in header
                self.MJD = get_mjd(self.hdr)
                self.filter = find_key('FILTER', typ=str).upper()
                self.RA, self.DEC = get_RADEC(self.hdr)
                if not self.name:
                    self.EXIST = False
                    self.flag_type = "%s was tagged as incomplete, OBJECT keyword " \
                                     "did not contain object name" % self.filename
                try:
                    self.airmass = float(find_key("AIRMASS"))
                except KeyError:
                    self.airmass = 0
        # failed attempts must be flagged
        except KeyError as e:
            self.EXIST = False
            keyword = e.args[0].strip('.')
            self.flag_type = "{} in header for {}.".format(keyword, self.filename)
