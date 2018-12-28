# -*- coding: utf-8 -*-
import fnmatch
import os

from astropy.io import fits

from constants import log
from constants import todays_date, filter_fits
from dirs_mgmt import copy, mv, shorten_path, walklevel

todays_day = todays_date.date()


def set_mulifits(src_path, pattern, value_setting, comment=None, level=100, destination="",
                 keep_originals=True, print_out=None):
    """
    The function will set the given keyword value to all FITS files with the matching root name in
    the directory sublevels. This function can also create a new keyword/value from the value_setting parameters, which would
    be appended at the end of the file.
    :param src_path: Root directory of the files
    :param pattern: Unique root name for finding the files
    :param value_setting: a tuple; (keyword,value) for which the value will be set
    :param comment:
    :param level: sublevels to walk on directory tree, default to 100. 0 means only the given folder
    :param destination: If it is desired to copy files to another directory then, input destination directory
    :param keep_originals:
    :param print_out:
    """
    # prepare for modifications
    keyword, newvalue = value_setting
    keyword = keyword.upper()
    # validate new value
    try:
        newvalue = float(newvalue)
    except ValueError:
        pass
    log('\nModifying FITS keyword values started\n'
        'Modification information:\n'
        '---FITSpath:---\t----Keyword:---\t'
        '---Before-Value---:\t---After-Value---:\n'.format(todays_day))
    for matched_fits in search_all_fits(src_path, level=level):
        filename = os.path.basename(matched_fits)
        # We use try/except block in order to avoid 'missing end card' errors. Good workaround?
        try:
            with fits.open(matched_fits, ignore_missing_end=True) as hdul:
                before_value = hdul[0].header[keyword]
        except (KeyError, OSError):
            before_value = None
        final_path = matched_fits
        if not fnmatch.fnmatchcase(filename, pattern):
            continue
        if destination and keep_originals:
            copy(matched_fits, destination)
            final_path = os.path.join(destination, filename)
        elif destination and not keep_originals:
            mv(matched_fits, destination)
            final_path = os.path.join(destination, filename)
        elif not destination and keep_originals:
            dir = os.path.dirname(matched_fits)
            mod_dest = os.path.join(dir, "Modified")
            final_path = os.path.join(mod_dest, filename)
            if not os.path.isdir(mod_dest):
                os.mkdir(mod_dest)
            copy(matched_fits, final_path)
        # We use try/except block in order to avoid 'missing end card' errors. Good workaround?
        try:
            fits.setval(final_path, keyword=keyword, value=newvalue, comment=comment)
        except (TypeError, OSError):
            # ignore missing end parameter is included so such files can be modified.
            with fits.open(final_path, ignore_missing_end=True) as hdul:
                if comment:
                    hdul[0].header[keyword] = newvalue
                else:
                    hdul[0].header[keyword] = newvalue, comment

                hdul.writeto(final_path, overwrite=True)
        log_string = "{}\t{}\t{}\t{}".format(shorten_path(matched_fits),
                                             keyword,
                                             before_value,
                                             newvalue)
        log(log_string)
    finish_msg = "Modifying FITS keyword values completed"
    log("\n{}".format(finish_msg))


# search all fits in given address, avoiding hidden folders/files
def search_all_fits(source, filter_files=False, filter_dirs=False, level=100, hidden=False):
    """
    Search all fits files in directory tree up to a 'level'.
    :param source: source (top) of directory tree
    :param filter_files: function callable that allows file filtering
    :param filter_dirs: function callable that allows folder filtering
    :param level: level up to which recurse. Default: 100 (recurse throughout)
    :param hidden:
    :returns: fits files generator
    """
    # strip last '/'
    source = source.rstrip(os.path.sep)
    assert os.path.isdir(source), "Invalid directory string, try again"
    for dirname, dirnames, filenames in walklevel(source, level, hidden=hidden, dirs_filter=filter_dirs):
        # save path to all file names
        if filter_files:
            filenames = [f for f in filenames if filter_files(f)]
        for filename in filter_fits(filenames):
            yield os.path.join(dirname, filename)
