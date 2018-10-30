# -*- coding: utf-8 -*-
import argparse
import os
import pickle
from configparser import ConfigParser
import pdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import Angle
from astropy.io import fits
from astropy.visualization import ZScaleInterval, ImageNormalize, LinearStretch
from matplotlib import ticker
from matplotlib.ticker import AutoMinorLocator
from mpl_toolkits.mplot3d import art3d
from scipy.signal import medfilt
from scipy.stats import linregress

from constants import get_telescopes, log, validateDirs

config = ConfigParser()
config.read('config.ini')
server_destination = config['FOLDER OPTIONS']['server_destination']

plt.style.use('ggplot')
minorLocator = AutoMinorLocator()


def angle2degree(raw_angle, unit):
    """
    Convert given angle with known unit (astropy.unit) to degrees in decimals.
    :param raw_angle: numberic or string value
    :param unit: unit of the angle; a astropy.unit object
    :return: angle in degrees (decimal)
    """
    return Angle(raw_angle, unit=unit).deg


def CoordsToDecimal(coords, hours=False):
    """
    Function to convert given angles to degree decimals. This function makes big assumptions given the wide variety
    of formats that EDEN has come across.
    ASSUMPTION:
    - if given coordinates are numeric values, then both RA/DEC are given in degrees
    - if given coordinates are strings/non-numeric values, then RA is given in hour angle and DEC in degrees.
    :return: numeric degree decimals
    """
    if hours:
        try:
            # check if object is iterable; list, array, etc
            isList = iter(coords)
            if isinstance(coords[0], str):
                # it must be a string, and with the following formats
                return angle2degree(coords, u.hourangle)
            else:
                return angle2degree(coords, u.deg)
        except TypeError:
            # if coords is a numeric value... else is a string
            if isinstance(coords, str):
                # it must be a string, and with the following formats
                return angle2degree(coords, u.hourangle)
            else:
                return angle2degree(coords, u.deg)

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


def get_super_comp(all_comp_fluxes, all_comp_fluxes_err):
    super_comp = np.zeros(all_comp_fluxes.shape[1])
    super_comp_err = np.zeros(all_comp_fluxes.shape[1])
    for i in range(all_comp_fluxes.shape[1]):
        data = all_comp_fluxes[:, i]
        data_err = all_comp_fluxes_err[:, i]
        med_data = np.nanmedian(data)
        sigma = get_sigma(data)

        idx = np.where((data <= med_data + 5 * sigma) & (data >= med_data - 5 * sigma) & \
                       (~np.isnan(data)) & (~np.isnan(data_err)))[0]
        super_comp[i] = np.nanmedian(data[idx])
        super_comp_err[i] = np.sqrt(np.sum(data_err[idx] ** 2) / np.double(len(data_err[idx])))
    return super_comp, super_comp_err


def get_sigma(data):
    """
    Get median absolute deviation from data
    :param data: numberical data
    :return: median absolute deviation
    """
    mad = np.nanmedian(np.abs(data - np.nanmedian(data)))
    return mad * 1.4826


def check_star(data, idx: int, min_ap: int, max_ap: int, force_aperture: bool, forced_aperture: int):
    """
    :param data: photometry data pickle saved from get_photometry
    :param idx: index of object
    :param min_ap:
    :param max_ap:
    :param force_aperture:
    :param forced_aperture:
    :return:
    """
    apertures = [forced_aperture] if force_aperture else (min_ap, max_ap)
    for chosen_aperture in apertures:
        try:
            target_flux = data['data']['star_' + str(idx)]['fluxes_' + str(chosen_aperture) + '_pix_ap']
        except KeyError:
            target_flux = data['data']['target_star_' + str(idx)]['fluxes_' + str(chosen_aperture) + '_pix_ap']
        if np.any(target_flux < 0):
            return False
    return True


def super_comparison_detrend(data, idx, idx_comparison, chosen_aperture,
                             comp_apertures=None, plot_comps=False, all_idx=None,
                             supercomp=False):
    try:
        n_comps = len(idx_comparison)
    except TypeError:
        idx_comparison = [idx_comparison]
    if comp_apertures is None:
        comp_apertures = [chosen_aperture] * len(idx_comparison)
    try:
        target_flux = data['data'][f'star_{idx}'][f'fluxes_{chosen_aperture}_pix_ap'][all_idx]
        target_flux_err = data['data']['star_' + str(idx)][f'fluxes_{chosen_aperture}_pix_ap_err'][all_idx]
    except KeyError:
        target_flux = data['data'][f'target_star_{idx}'][f'fluxes_{chosen_aperture}_pix_ap'][all_idx]
        target_flux_err = data['data'][f'target_star_{idx}'][f'fluxes_{chosen_aperture}_pix_ap_err'][all_idx]
    if plot_comps:
        plt.plot(target_flux / np.nanmedian(target_flux), 'b-')
    all_comp_fluxes = np.zeros((len(idx_comparison), target_flux.size))
    all_comp_fluxes_err = np.zeros(all_comp_fluxes.shape)
    for i in range(len(idx_comparison)):
        idx_c = idx_comparison[i]
        comp_aperture = comp_apertures[i]
        try:
            comp_flux = data['data'][f'star_{idx_c}'][f'fluxes_{comp_aperture}_pix_ap'][all_idx]
            comp_flux_err = data['data']['star_' + str(idx_c)][f'fluxes_{comp_aperture}_pix_ap_err'][all_idx]
        except KeyError:
            comp_flux = data['data'][f'target_star_{idx_c}'][f'fluxes_{comp_aperture}_pix_ap'][all_idx]
            comp_flux_err = data['data'][f'target_star_{idx_c}'][f'fluxes_{comp_aperture}_pix_ap_err'][all_idx]
        comp_med = np.nanmedian(comp_flux)
        all_comp_fluxes[i] = comp_flux / comp_med
        all_comp_fluxes_err[i] = comp_flux_err / comp_med
        if plot_comps:
            plt.plot(comp_flux / comp_med, 'r-', alpha=0.1)
    
    super_comp, super_comp_err = get_super_comp(all_comp_fluxes, all_comp_fluxes_err)
    if plot_comps:
        plt.plot(super_comp, 'r-')
        plt.show()
    
    relative_flux = target_flux / super_comp
    relative_flux_err = relative_flux * np.sqrt((target_flux_err / target_flux) ** 2 + \
                                                (super_comp_err / super_comp) ** 2)
    med_rel_flux = np.nanmedian(relative_flux)
    detrend_flux = relative_flux / med_rel_flux
    detrend_flux_err = relative_flux_err / med_rel_flux
    if supercomp:
        return detrend_flux, detrend_flux_err, super_comp, super_comp_err
    return detrend_flux, detrend_flux_err


def save_photometry(t, rf, rf_err, output_folder, target_name,
                    plot_data=False, title='', units='Relative Flux'):
    """
    Save given relative photometry into plots/files.
    :param t: times
    :param rf: relative fluxes
    :param rf_err: errors of relative fluxes
    :param output_folder: folder to save all files
    :param target_name: name of the target object
    :param plot_data:
    :param title: Title of the plots
    :param units: units of the Y axis on plots
    """
    log("Saving photometry for target {:s} on {:s} using the following:".format(target_name, output_folder))
    mag_fact = (100. ** .2)
    rf_mag = -mag_fact * np.log10(rf)
    rf_mag_err = rf_err * 2.5 / (np.log(10) * rf)
    f = open(output_folder + target_name + '.dat', 'w')
    f2 = open(output_folder + target_name + '_norm_flux.dat', 'w')
    f.write('# Times (BJD) \t Diff. Mag. \t Diff. Mag. Err.\n')
    f2.write('# Times (BJD) \t Norm. Flux. \t Norm. Flux Err.\n')
    for i in range(len(t)):
        if not np.isnan(rf_mag[i]) and not np.isnan(rf_mag_err[i]):
            f.write(str(t[i]) + '\t' + str(rf_mag[i]) + '\t' + str(rf_mag_err[i]) + '\n')
            f2.write(str(t[i]) + '\t' + str(rf[i]) + '\t' + str(rf_err[i]) + '\n')
    f.close()
    if plot_data:
        # Bin on a 10-min window:
        t_min = np.min(t)
        t_hours = (t - t_min) * 24.
        n_bin = 10  # minutes
        bin_width = n_bin / 60.  # hr
        bins = (t_hours / bin_width).astype('int')
        times_bins = []
        fluxes_bins = []
        errors_bins = []
        for i_bin in np.unique(bins):
            idx = np.where(bins == i_bin)
            times_bins.append(np.median(t_hours[idx]))
            fluxes_bins.append(np.nanmedian(rf[idx]))
            errors_bins.append(np.sqrt(np.sum(rf_err[idx] ** 2)) / np.double(len(idx[0])))

        # Calculate standard deviation of median filtered data
        mfilt = median_filter(rf)
        sigma = get_sigma(rf - mfilt) / np.nanmedian(rf)
        sigma_mag = -2.5 * np.log10((1. - sigma) / 1.)
        mfilt_bin = median_filter(fluxes_bins)
        sigma_bin = get_sigma(fluxes_bins - mfilt_bin) / np.median(fluxes_bins)
        sigma_mag_bin = -2.5 * np.log10((1. - sigma_bin) / 1.)
        sigma_top = '$\sigma_{{m}}$ = {:.0f} ppm = {:.1f} mmag'.format(sigma * 1e6, sigma_mag * 1e3)
        sigma_bott = '$\sigma_{{m,bin}}$ = {:.0f} ppm = {:.1f} mmag'.format(sigma_bin * 1e6, sigma_mag_bin * 1e3)
        sigma_file = open(output_folder + target_name + '.sigma.dat', 'w')
        sigma_file.write(sigma_top + '\n')
        sigma_file.write(sigma_bott)
        sigma_file.close()
        # Make plot
        plt.errorbar(t_hours, rf, rf_err, fmt='o', alpha=0.3, label='Data')
        plt.errorbar(np.array(times_bins), np.array(fluxes_bins), np.array(errors_bins), fmt='o', label='10-min bins')
        plt.annotate(sigma_top, xy=(0.5, 0.10), xycoords='axes fraction', va='bottom', ha='center')
        plt.annotate(sigma_bott, xy=(0.5, 0.05), xycoords='axes fraction', va='bottom', ha='center')
        plt.xlabel('Time from start (hr)')
        plt.ylabel(units)
        plt.title(title, fontsize='12')
        plt.xlim(-0.05 * np.ptp(t_hours), 1.05 * np.ptp(t_hours))
        nom_ymin = 0.95
        data_min = np.max([np.min(rf - 2 * rf_err), np.nanmedian(rf - rf_err) - 15 * get_sigma(rf - median_filter(rf))])
        nom_ymax = 1.05
        data_max = np.min([np.max(rf + 2 * rf_err), np.nanmedian(rf + rf_err) + 15 * get_sigma(rf - median_filter(rf))])
        try:
            plt.ylim(data_min, data_max)
        except:
            plt.ylim(nom_ymin, nom_ymax)
        x_formatter = ticker.ScalarFormatter(useOffset=False)
        plt.gca().xaxis.set_major_formatter(x_formatter)
        plt.legend()
        plt.gcf().savefig(output_folder + target_name + '.pdf', dpi=150, bbox_inches='tight')
        plt.close()


def save_trendStats(epdlc_path, output_folder: str, rmag_delta=0.5, mag_delta=0.8, starID=None):
    # prepare data and some vars for plots
    if isinstance(epdlc_path, pd.DataFrame):
        epdlc = epdlc_path
    else:
        # else it is assumed is a string (path)
        epdlc = pd.read_csv(epdlc_path)
        starID: str = os.path.basename(epdlc_path).split('.')[-2]
    # setup limits for magnitude and relative magnitude plots
    mean_mag = epdlc['mag1'].mean()
    rmag_lim = max(-rmag_delta, epdlc['rmag1'].min()), min(rmag_delta, epdlc['rmag1'].max())
    mag_lim = max(mean_mag - mag_delta, epdlc['mag1'].min()), min(mean_mag + mag_delta, epdlc['mag1'].max())
    airmass_savepath = os.path.join(output_folder, 'airmass_trends', starID + '_Airmass trend.png')
    fwhm_savepath = os.path.join(output_folder, 'seeing_trends', starID + '_Seeing trend.png')
    distance_savepath = os.path.join(output_folder, 'distance_trends', starID + '_Distance trend.png')
    validateDirs(output_folder, os.path.dirname(airmass_savepath),
                 os.path.dirname(fwhm_savepath), os.path.dirname(distance_savepath))

    # calculate distances from median:
    xdist = abs(epdlc['cen_x'] - epdlc['cen_x'].median())
    ydist = abs(epdlc['cen_y'] - epdlc['cen_y'].median())
    epdlc['cen_dist'] = (xdist ** 2 + ydist ** 2) ** 0.5
    corr = epdlc.corr()

    # start airmass plots
    fig, axes = plt.subplots(2, 2, sharex='col')
    epdlc.plot(x='BJD', y='Z', title='Airmass over time', ax=axes[0, 0])
    axes[0, 0].set_ylabel('Airmass')
    epdlc.plot.scatter(x='BJD', y='mag1', yerr='mag1_err', title='Magnitude over time', ax=axes[1, 0])
    axes[1, 0].set_ylabel('Magnitude')
    ax2 = epdlc.plot.scatter(x='Z', y='mag1', yerr='mag1_err', title='Magnitude vs Airmass', ax=axes[1, 1])
    axes[1, 1].set_xlabel('Airmass')
    ax1 = epdlc.plot.scatter(x='Z', y='rmag1', yerr='rmag1_err', title='Relative Magnitude vs Airmass', ax=axes[0, 1])
    axes[0, 1].set_ylabel('Relative Magnitude')
    rmag_corr = corr['rmag1'].loc['Z']
    mag_corr = corr['mag1'].loc['Z']
    ann = ax1.annotate('Correlation Coeff: %.3f' % rmag_corr, (30, 5), None, 'axes points')
    ann.set_size(8)
    ann = ax2.annotate('Correlation Coeff: %.3f' % mag_corr, (30, 5), None, 'axes points')
    ann.set_size(8)
    for ax in axes.reshape(axes.size):
        ax.autoscale(True, axis='both', tight=True)
    fig.suptitle(starID, y=1.05)
    plt.tight_layout()
    axes[0, 1].set_ylim(rmag_lim)
    axes[1, 0].set_ylim(mag_lim)
    axes[1, 1].set_ylim(mag_lim)
    fig.savefig(airmass_savepath, dpi=200, bbox_inches='tight')
    # end airmass plots
    plt.close()
    # start seeing plots
    fig, axes = plt.subplots(2, 2, sharex='col')
    epdlc.plot.scatter(x='BJD', y='FWHM', title='Seeing over time', ax=axes[0, 0])
    epdlc.plot.scatter(x='BJD', y='mag1', yerr='mag1_err', title='Magnitude over time', ax=axes[1, 0])
    axes[1, 0].set_ylabel('Magnitude')
    ax2 = epdlc.plot.scatter(x='FWHM', y='mag1', yerr='mag1_err', title='Magnitude vs Seeing', ax=axes[1, 1])
    ax1 = epdlc.plot.scatter(x='FWHM', y='rmag1', yerr='rmag1_err', title='Relative Magnitude vs Seeing', ax=axes[0, 1])
    axes[0, 1].set_ylabel('Relative Magnitude')
    rmag_corr = corr['rmag1'].loc['FWHM']
    mag_corr = corr['mag1'].loc['FWHM']
    ann = ax1.annotate('Correlation Coeff: %.3f' % rmag_corr, (30, 5), None, 'axes points')
    ann.set_size(8)
    ann = ax2.annotate('Correlation Coeff: %.3f' % mag_corr, (30, 5), None, 'axes points')
    ann.set_size(8)
    for ax in axes.reshape(axes.size):
        ax.autoscale(True, axis='both', tight=True)
    fig.suptitle(starID, y=1.05)
    plt.tight_layout()
    axes[0, 1].set_ylim(rmag_lim)
    axes[1, 0].set_ylim(mag_lim)
    axes[1, 1].set_ylim(mag_lim)
    fig.savefig(fwhm_savepath, dpi=200, bbox_inches='tight')
    # end seeing plots
    plt.close()
    # start distance plots
    fig, axes = plt.subplots(2, 2, sharex='col')
    epdlc.plot(x='BJD', y='cen_dist', title='Distance over time', ax=axes[0, 0])
    axes[0, 0].set_ylabel('Pixel distance')
    epdlc.plot.scatter(x='BJD', y='mag1', yerr='mag1_err',
                       title='Magnitude over time', ax=axes[1, 0])
    axes[1, 0].set_ylabel('Magnitude')
    ax2 = epdlc.plot.scatter(x='cen_dist', y='mag1', yerr='mag1_err',
                             title='Magnitude vs Distance', ax=axes[1, 1])
    axes[1, 1].set_xlabel('Pixel Distance')
    ax1 = epdlc.plot.scatter(x='cen_dist', y='rmag1', yerr='rmag1_err',
                             title='Relative Magnitude vs Distance', ax=axes[0, 1])
    axes[0, 1].set_ylabel('Relative Magnitude')
    rmag_corr = corr['rmag1'].loc['cen_dist']
    mag_corr = corr['mag1'].loc['cen_dist']
    ax1.annotate('Correlation Coeff: %.3f' % rmag_corr, (30, 5), None, 'axes points', size=8)
    ax2.annotate('Correlation Coeff: %.3f' % mag_corr, (30, 5), None, 'axes points', size=8)
    for ax in axes.reshape(axes.size):
        ax.autoscale(True, axis='both', tight=True)
    fig.suptitle(starID + '\nDistance: Distance from median center', y=1.1)
    plt.tight_layout()
    axes[0, 1].set_ylim(rmag_lim)
    axes[1, 0].set_ylim(mag_lim)
    axes[1, 1].set_ylim(mag_lim)
    fig.savefig(distance_savepath, dpi=200, bbox_inches='tight')
    # end distance plots
    plt.close()


def dataframe2epdlc(table: pd.DataFrame, out_path: str):
    """
    Convert dataframe to epdlc file
    :param table: table that must follow the format in the code
    :param out_path: complete file path to save epdlc file
    """
    # Define literal string with formatting:
    header = f'#{"frame":<31} {"BJD":<16} {"JD":<16} {"mag1":<8} {"mag1_err":<8} ' \
             f'{"mag2":<8} {"mag2_err":<8} {"mag3":<8} {"mag3_err":<8} ' \
             f'{"rmag1":<8} {"rmag1_err":<8} {"rmag2":<8} {"rmag2_err":<8}' \
             f' {"rmag3":<8} {"rmag3_err":<8} {"cen_x":<8} ' \
             f'{"cen_y":<8} {"bg":<8} {"bg_err":<8} {"FWHM":<8} ' \
             f'{"HA":<8} {"ZA":<8} {"Z":<8}\n'
    row_fmt = '{frame:<32} {BJD:<16.8f} {JD:<16.8f} {mag1:< 8.4f} {mag1_err:<8.4f} ' \
              '{mag2:< 8.4f} {mag2_err:<8.4f} {mag3:< 8.4f} {mag3_err:<8.4f} ' \
              '{rmag1:< 8.4f} {rmag1_err:< 8.4f} {rmag2:< 8.4f} {rmag2_err:< 8.4f} ' \
              '{rmag3:< 8.4f} {rmag3_err:< 8.4f} {cen_x:<8.3f} ' \
              '{cen_y:<8.3f} {bg:<8.3f} {bg_err:<8.3f} {FWHM:< 8.3f} ' \
              '{HA:< 8.2f} {ZA:<8.2f} {Z:<8.3f}\n'
    formatted_rows = [row_fmt.format(**row) for i,row in table.iterrows()]
    with open(out_path, 'w') as epdlc:
        epdlc.write(header)
        epdlc.writelines(formatted_rows)


def save_photometry_hs(data, idx, idx_comparison,idx_all_comps_sorted,
                       chosen_aperture: int, min_aperture: int, max_aperture: int,
                       comp_apertures, idx_sort_times, output_folder: str,
                       target_name: str, band='i', all_idx=None):
    """
    Save epdlc files
    :param data: picke data
    :param idx: index of main target
    :param idx_comparison: indeces of comparison stars to detrend
    :param chosen_aperture: choosen aperture to save
    :param min_aperture: minimum aperture to save
    :param max_aperture: maximum aperture to save
    :param comp_apertures:
    :param idx_sort_times: indeces of sorted times to use
    :param output_folder: folder to save all epdlc files
    :param target_name: name of the main target
    :param band:
    :param all_idx:
    :return:
    """
    header = ['frame', 'BJD', 'JD', 'mag1', 'mag1_err', 'mag2', 'mag2_err', 'mag3', 'mag3_err',
              'rmag1', 'rmag1_err', 'rmag2', 'rmag2_err', 'rmag3', 'rmag3_err', 'cen_x', 'cen_y',
              'bg', 'bg_err', 'FWHM', 'HA', 'ZA', 'Z']
    hs_folder = os.path.join(output_folder, 'LC')
    trends_folder = os.path.join(output_folder, 'trends')
    other_formats = os.path.join(hs_folder, 'csv_html')
    # Create folder for the outputs in HS format:
    if not os.path.exists(hs_folder):
        os.mkdir(hs_folder)
    # Create folder for the outputs in html format:
    if not os.path.exists(other_formats):
        os.mkdir(other_formats)
    # Create folder for trendStats
    if not os.path.exists(trends_folder):
        os.mkdir(trends_folder)
    # First, write lightcurve in the HS format for each star. First the comparisons:
    print('\t Saving data for target and', len(idx_comparison), 'comparison stars')
    for i in np.append(idx_comparison, idx):
        try:
            d = data['data']['star_%d' % i]
        except KeyError:
            d = data['data']['target_star_%d' % i]
        if i == idx:
            star_name = target_name
        else:
            star_name = str(data['data']['IDs'][i])
        fluxes_ap = 'fluxes_%d_pix_ap' % chosen_aperture
        fluxes_min_ap = 'fluxes_%d_pix_ap' % min_aperture
        fluxes_max_ap = 'fluxes_%d_pix_ap' % max_aperture
        fluxes_ap_err = 'fluxes_%d_pix_ap_err' % chosen_aperture
        fluxes_min_ap_err = 'fluxes_%d_pix_ap_err' % min_aperture
        fluxes_max_ap_err = 'fluxes_%d_pix_ap_err' % max_aperture

        flux = d[fluxes_ap][all_idx][idx_sort_times]
        flux_min = d[fluxes_min_ap][all_idx][idx_sort_times]
        flux_max = d[fluxes_max_ap][all_idx][idx_sort_times]
        flux_err = d[fluxes_ap_err][all_idx][idx_sort_times]
        flux_min_err = d[fluxes_min_ap_err][all_idx][idx_sort_times]
        flux_max_err = d[fluxes_max_ap_err][all_idx][idx_sort_times]

        FWHMs = d['fwhm'][all_idx][idx_sort_times]
        epdlc_path = os.path.join(hs_folder, star_name + '.epdlc')
        csv_path = os.path.join(other_formats, star_name + '.csv')
        html_path = os.path.join(other_formats, star_name + '.html')
        ra = data['data']['RA_degs'][i]

        #  Get super-comparison detrend for the current star:
        current_comps = [ii for ii in idx_comparison if ii != i]
        if len(current_comps) == 0:
            current_comps = idx_all_comps_sorted[0:10]
            comp_apertures = comp_apertures * len(current_comps)

        r_flux1, r_flux_err1 = super_comparison_detrend(data, i, current_comps, chosen_aperture,
                                                        comp_apertures=comp_apertures, plot_comps=False,
                                                        all_idx=all_idx)
        r_flux1, r_flux_err1 = r_flux1[idx_sort_times], r_flux_err1
        r_flux2, r_flux_err2 = super_comparison_detrend(data, i, current_comps, min_aperture,
                                                        comp_apertures=comp_apertures, plot_comps=False,
                                                        all_idx=all_idx)
        r_flux2, r_flux_err2 = r_flux2[idx_sort_times], r_flux_err2[idx_sort_times]
        r_flux3, r_flux_err3 = super_comparison_detrend(data, i, current_comps, max_aperture,
                                                        comp_apertures=comp_apertures, plot_comps=False,
                                                        all_idx=all_idx)
        r_flux3, r_flux_err3 = r_flux3[idx_sort_times], r_flux_err3[idx_sort_times]

        # (100.**.2) ~ 2.512
        mag_fact = (100. ** .2)
        # Get Relative Mags Data
        rmag1 = (-mag_fact * np.log10(r_flux1))
        rmag1_err = (r_flux_err1 * mag_fact / (np.log(10) * r_flux1))
        rmag2 = (-mag_fact * np.log10(r_flux2))
        rmag2_err = (r_flux_err2 * mag_fact / (np.log(10) * r_flux2))
        rmag3 = (-mag_fact * np.log10(r_flux3))
        rmag3_err = (r_flux_err3 * mag_fact / (np.log(10) * r_flux3))

        # Get Mags Data
        mag1 = (-mag_fact * np.log10(flux))
        mag1_err = (mag_fact * flux_err / (np.log(10.) * flux))
        mag2 = (-mag_fact * np.log10(flux_min))
        mag2_err = (mag_fact * flux_min_err / (np.log(10.) * flux_min))
        mag3 = (-mag_fact * np.log10(flux_max))
        mag3_err = (mag_fact * flux_max_err / (np.log(10.) * flux_max))

        # Set all fwhms that are 0 to -1
        FWHMs[FWHMs == 0] = -1

        lst_deg = CoordsToDecimal(data['LST'][all_idx][idx_sort_times], hours=True)
    
        # following line context: keep lst_deg - ra between 0 and 360 degrees
        HA = lst_deg - ra + 360
        HA[HA<0] += 360
        HA[HA>=360] -= 360
        #HA = lst_deg - ra + 360 if lst_deg - ra < 0 else lst_deg - ra - 360 if lst_deg - ra > 360 else lst_deg - ra
        Z = data['airmasses'][all_idx][idx_sort_times].astype(float)
        ZA = np.arccos(1. / Z) * (180. / np.pi)

        # frame names in order
        frame_names = [frame_name.split('/')[-1] for frame_name in data['frame_name'][all_idx][idx_sort_times]]
        bjd_times = data['BJD_times'][all_idx][idx_sort_times]
        jd_times = data['JD_times'][all_idx][idx_sort_times]
        centroids_x = d['centroids_x'][all_idx][idx_sort_times]
        centroids_y = d['centroids_y'][all_idx][idx_sort_times]
        background = d['background'][all_idx][idx_sort_times]
        background_err = d['background_err'][all_idx][idx_sort_times]

        tableData = {hrd: col for hrd, col in zip(header, [frame_names, bjd_times, jd_times, mag1, mag1_err,
                                                      mag2, mag2_err, mag3, mag3_err, rmag1, rmag1_err,
                                                      rmag2, rmag2_err, rmag3, rmag3_err, centroids_x,
                                                      centroids_y, background, background_err, FWHMs,
                                                      HA, ZA, Z])}
        table = pd.DataFrame(tableData)
        table.to_csv(csv_path, index=False)
        table.to_html(html_path, float_format=lambda double: '%.4f' % double)
        dataframe2epdlc(table, epdlc_path)
        save_trendStats(table, trends_folder, starID=star_name)


def radial_profile(data, center):
    """
    :param data: 2D array image
    :param center: x,y for the center
    :return: 1D array of average flux for each pixel distance from center
    """
    x, y = np.indices(data.shape)
    # get distances from center for each (x,y) pixel and round up to integer values
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2).astype(np.int)
    r = r.astype(np.int)
    #
    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / nr
    return radialprofile


def plot_images(data, idx, idx_comparison, aperture, min_ap, max_ap,
                comp_apertures, out_dir, frames, idx_frames, half_size=50, overwrite=False):
    def plot_im(d, cen_x, cen_y, obj_x, obj_y, ap: int, min_ap: int, max_ap: int,
                half_size, frame_name: str, object_name: str, overwrite, norm: ImageNormalize=None):
        if 'full_frame' in object_name:
            fullf_path = os.path.join(out_dir, 'sub_imgs', 'full_frame')
            if not os.path.exists(fullf_path):
                os.makedirs(fullf_path)
            # the following is an new python format... called Literal String Formatting or f'Strings
            fname = f'{out_dir:}/sub_imgs/full_frame/{object_name:}_{frame_name.split("/")[-1]:}.png'
            fname_3d = fname_r =None
        else:
            if not os.path.exists(os.path.join(out_dir, 'sub_imgs', object_name, 'surfaces')):
                os.makedirs(os.path.join(out_dir, 'sub_imgs', object_name, 'surfaces'))
            if not os.path.exists(os.path.join(out_dir, 'sub_imgs', object_name, 'radial_plot')):
                os.makedirs(os.path.join(out_dir, 'sub_imgs', object_name, 'radial_plot'))
            fname = f'{out_dir:}/sub_imgs/{object_name:}/{frame_name.split("/")[-1]:}_{object_name:}.png'
            fname_3d = f'{out_dir:}/sub_imgs/{object_name:}/surfaces/{frame_name.split("/")[-1]:}_{object_name:}.png'
            fname_r = f'{out_dir:}/sub_imgs/{object_name:}/radial_plot/{frame_name.split("/")[-1]:}_{object_name:}.png'
        if not os.path.exists(fname) or overwrite:
            # Plot image of the target:
            x0 = max(0, int(cen_x) - half_size)
            x1 = min(int(cen_x) + half_size, d.shape[1])
            y0 = max(0, int(cen_y) - half_size)
            y1 = min(int(cen_y) + half_size, d.shape[0])
            # plot 2D Subimage
            subimg = d[y0:y1, x0:x1].copy()
            subimg -= np.median(subimg)
            # create normalization for plotting
            if norm is not None:
                norm = ImageNormalize(subimg, interval=ZScaleInterval(), stretch=LinearStretch())
            extent = (x0, x1, y0, y1)
            plt.imshow(subimg, extent=extent, interpolation='none', origin='lower', norm=norm)
            plt.plot(obj_x, obj_y, 'wx', markersize=15, lw=2, alpha=0.5)
            circle = plt.Circle((obj_x, obj_y), min_ap, color='black', lw=2, alpha=0.5, fill=False)
            circle2 = plt.Circle((obj_x, obj_y), max_ap, color='black', lw=2, alpha=0.5, fill=False)
            circle3 = plt.Circle((obj_x, obj_y), ap, color='white', lw=2, alpha=0.5, fill=False)
            plt.gca().add_artist(circle)
            plt.gca().add_artist(circle2)
            plt.gca().add_artist(circle3)
            plt.savefig(fname, dpi=125, bbox_inches='tight')
            plt.close()

            if fname_3d is not None:
                # Plot Radial profile for target
                center = obj_x - x0, obj_y - y0
                rad_profile = radial_profile(subimg, center)[:max_ap + 10]
                fig, ax = plt.subplots()
                plt.plot(rad_profile, 'x-')
                plt.tick_params(which='both', width=2)
                plt.tick_params(which='major', length=7)
                plt.tick_params(which='minor', length=4, color='r')
                ax.xaxis.set_minor_locator(minorLocator)
                plt.grid()
                ax.set_ylabel("Average Count")
                ax.set_xlabel("Pixels")
                plt.grid(which="minor")
                fig.savefig(fname_r, bbox_inches='tight')
                plt.close()

                # plot 3D Surface subimage; we create an even smaller subimage for more detail
                half_size = max_ap + 1
                x0 = max(0, int(cen_x) - half_size)
                x1 = min(int(cen_x) + half_size, d.shape[1])
                y0 = max(0, int(cen_y) - half_size)
                y1 = min(int(cen_y) + half_size, d.shape[0])
                background_level = subimg.mean()
                subimg = d[y0:y1, x0:x1] - np.median(d[y0:y1, x0:x1])
                x = np.arange(x1 - x0) + x0
                y = np.arange(y1 - y0) + y0
                X, Y = np.meshgrid(x, y)
                fig = plt.figure()
                ax = fig.gca(projection='3d')
                ax.plot_surface(X, Y, subimg, rstride=1, cstride=1, alpha=0.9, cmap='summer', norm=norm)
                circle = plt.Circle((obj_x, obj_y), min_ap, color='black', lw=2, fill=False)
                circle2 = plt.Circle((obj_x, obj_y), max_ap, color='black', lw=2, fill=False)
                circle3 = plt.Circle((obj_x, obj_y), ap, color='#ad343a', lw=2, fill=False)
                ax.add_patch(circle)
                ax.add_patch(circle2)
                ax.add_patch(circle3)
                art3d.pathpatch_2d_to_3d(circle, z=background_level, zdir="z")
                art3d.pathpatch_2d_to_3d(circle2, z=background_level, zdir="z")
                art3d.pathpatch_2d_to_3d(circle3, z=background_level, zdir="z")
                ax.contour(X, Y, subimg, zdir='x', offset=X[0, 0], cmap='summer')
                ax.contour(X, Y, subimg, zdir='y', offset=Y[0, 0], cmap='summer')
                ax.view_init(25, 35)
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Counts')
                fig.savefig(fname_3d, dpi=125, bbox_inches='tight')
                plt.close()
    # Get the centroids of the target:
    target_cen_x, target_cen_y = get_cens(data, idx, idx_frames)
    # Same for the comparison stars:

    comp_cen_x, comp_cen_y = get_cens(data, idx_comparison[0], idx_frames)
    all_comp_cen_x = np.atleast_2d(comp_cen_x)
    all_comp_cen_y = np.atleast_2d(comp_cen_y)

    for i in range(1, len(idx_comparison)):
        idx_c = idx_comparison[i]
        comp_cen_x, comp_cen_y = get_cens(data, idx_c, idx_frames)
        all_comp_cen_x = np.vstack((all_comp_cen_x, comp_cen_x))
        all_comp_cen_y = np.vstack((all_comp_cen_y, comp_cen_y))

    # Now plot images around centroids plus annulus:
    exts = np.unique(data['data']['ext']).astype('int')
    nframes = len(frames)
    for i in range(nframes):
        for ext in exts:
            frame: str = frames[i]
            # temporary fix to underscore mislabeling
            if not os.path.isfile(frame):
                frame = frame.replace(' ', '_')
            d = fits.getdata(frame, ext=ext)
            idx_names = data['data']['ext'] == ext
            names_ext = data['data']['names'][idx_names]
            # Create normalization for plotting
            norm = ImageNormalize(d.copy() - np.median(d), interval=ZScaleInterval(), stretch=LinearStretch())
            for name in names_ext:
                if 'target' in name:
                    # Plot image of the target:
                    if i == 0:
                        plot_im(d, target_cen_x[0], target_cen_y[0], target_cen_x[i], target_cen_y[i],
                                aperture, min_ap, max_ap, 4 * half_size, frames[i], 'target_full_frame',
                                overwrite, norm)
                    plot_im(d, target_cen_x[0], target_cen_y[0], target_cen_x[i], target_cen_y[i],
                            aperture, min_ap, max_ap, half_size, frames[i], 'target', overwrite, norm)
            # Plot image of the comparisons:
            for j in range(len(idx_comparison)):
                idx_c = idx_comparison[j]
                name = 'star_' + str(idx_c)
                if name in names_ext:
                    if i == 0:
                        plot_im(d, np.median(all_comp_cen_x[j, :]), np.median(all_comp_cen_y[j, :]),
                                all_comp_cen_x[j, i], all_comp_cen_y[j, i], comp_apertures[j], min_ap, max_ap,
                                4 * half_size, frames[i], name + '_full_frame', overwrite, norm)
                    plot_im(d, np.median(all_comp_cen_x[j, :]), np.median(all_comp_cen_y[j, :]),
                            all_comp_cen_x[j, i], all_comp_cen_y[j, i], comp_apertures[j], min_ap, max_ap,
                            half_size, frames[i], name, overwrite, norm)


def plot_cmd(colors, data, idx, idx_comparison, post_dir):
    """
    Plot the color-magnitude diagram of all stars, 
    indicating the target and selected comparison stars.
    """
    ms = plt.rcParams['lines.markersize']
    plt.plot(colors, data['data']['Jmag'], 'b.', label='All stars')
    plt.plot(colors[idx], data['data']['Jmag'][idx], 'ro', ms=ms * 2, label='Target')
    plt.plot(colors[idx_comparison], data['data']['Jmag'][idx_comparison], 'r.', label='Selected comparisons')
    plt.title('Color-magnitude diagram of stars')
    plt.xlabel('J$-$H color')
    plt.ylabel('J (mag)')
    plt.legend(loc='best')
    plt.gca().invert_yaxis()
    plt.savefig(os.path.join(post_dir, 'CMD.pdf'))
    plt.close()


def median_filter(arr):
    median_window = int(np.sqrt(len(arr)))
    if median_window % 2 == 0:
        median_window += 1
    return medfilt(arr, median_window)


def get_cens(data, idx, idx_frames):
    try:
        cen_x = data['data']['star_' + str(idx)]['centroids_x'][idx_frames]
        cen_y = data['data']['star_' + str(idx)]['centroids_y'][idx_frames]
    except:
        cen_x = data['data']['target_star_' + str(idx)]['centroids_x'][idx_frames]
        cen_y = data['data']['target_star_' + str(idx)]['centroids_y'][idx_frames]
    return cen_x, cen_y

def post_processing(telescope,datafolder,target_name,target_coords,band='ip',ncomp=0,min_ap=5,max_ap=25,forced_aperture=15,filename='photometry.pkl',
                       force_aperture=False,optimize_apertures=False,plt_images=False,all_plots=False,overwrite=False):
    
    # Check if given telescope is in server, if it is use it, else exit program:
    inServer = any([telescope.upper() == tel.upper() for tel in get_telescopes()])
    if not inServer:
        print("Error, telescope not supported.")
        exit()
    
    red_path = datafolder

    # make directory for post processing files
    post_dir = os.path.join(red_path, 'post_processing/')
    if not os.path.exists(post_dir):
        os.makedirs(post_dir)

    if not os.path.exists(os.path.join(post_dir, 'post_processing_outputs')):
        os.makedirs(os.path.join(post_dir, 'post_processing_outputs'))

    #################################################
    
    # Convert target coordinates to degrees:
    target_ra, target_dec = CoordsToDecimal(target_coords)

    # Open dictionary, save times:
    try:
        data = pickle.load(open(os.path.join(red_path, filename), 'rb'))
    except FileNotFoundError:
        print("\t Pickle file not found. Please make sure it exists! Quitting...")
        raise
    all_sites = len(data['frame_name']) * [[]]
    all_cameras = len(data['frame_name']) * [[]]

    for i in range(len(data['frame_name'])):
        frames = data['frame_name'][i]
        if not os.path.exists(frames):
            frames = frames.replace(' ', '_')
        with fits.open(frames) as hdulist:
            h = hdulist[0].header
        try:
            all_sites[i] = telescope
            all_cameras[i] = h['INSTRUME']
        except (KeyError, IndexError):
            all_sites[i] = telescope
            all_cameras[i] = 'VATT4k'

    sites = []
    frames_from_site = {}
    for i in range(len(all_sites)):
        s = all_sites[i]
        c = all_cameras[i]
        if s + '+' + c not in sites:
            sites.append(s + '+' + c)
            frames_from_site[s + '+' + c] = [i]
        else:
            frames_from_site[s + '+' + c].append(i)

    print('Observations taken from: ', sites)

    # Get all the RAs and DECs of the objects:
    all_ras, all_decs = data['data']['RA_degs'], data['data']['DEC_degs']
    # Search for the target:
    distance = np.sqrt((all_ras - target_ra) ** 2 + (all_decs - target_dec) ** 2)
    # idx_target: index of main target
    idx_target: int = np.argmin(distance)
    # Search for closest stars in color to target star:
    target_hmag, target_jmag = data['data']['Hmag'][idx_target], data['data']['Jmag'][idx_target]
    colors = data['data']['Jmag'] - data['data']['Hmag']
    target_color = target_hmag - target_jmag
    color_distance = np.sqrt((colors - target_color) ** 2. + (target_jmag - data['data']['Jmag']) ** 2.)
    idx_distances = np.argsort(color_distance)
    idx_all_comps = []
    # Start with full set of comparison stars, provided they are good:
    for i in idx_distances:
        if i == idx_target:
            continue
        if check_star(data, i, min_ap, max_ap, force_aperture, forced_aperture):
            idx_all_comps.append(i)

    for site in sites:
        print('\t Photometry for site:', site)
        idx_frames = frames_from_site[site]
        times = data['BJD_times'][idx_frames]
        isWrong = np.all(times[0] == times)
        if isWrong:
            log("For some reason the BJD_times keyword in pickle contains an array repeating the same number.")
        idx_sort_times = np.argsort(times)
        step = len(data['frame_name'][idx_frames]) // 5
        step = step if step > 0 else 1
        print('\t Frames Summary: ~', str(data['frame_name'][idx_frames][0:-1:step]))
        # Check which is the aperture that gives a minimum rms:
        if force_aperture:
            print('\t Forced aperture to ', forced_aperture)
            chosen_aperture: int = forced_aperture
        else:
            print('\t Estimating optimal aperture...')
            apertures_to_check = range(min_ap, max_ap)
            precision = np.zeros(len(apertures_to_check))

            for i in range(len(apertures_to_check)):
                aperture = apertures_to_check[i]
                # Check the target
                relative_flux, relative_flux_err = super_comparison_detrend(data, idx_target, idx_all_comps, aperture,
                                                                            all_idx=idx_frames)

                save_photometry(times[idx_sort_times], relative_flux[idx_sort_times], relative_flux_err[idx_sort_times],
                                os.path.join(post_dir, 'post_processing_outputs/'),
                                target_name='target_photometry_ap' + str(aperture) + '_pix', plot_data=True)

                mfilt = median_filter(relative_flux[idx_sort_times])
                precision[i] = get_sigma((relative_flux[idx_sort_times] - mfilt) * 1e6)

            idx_max_prec = np.nanargmin(precision)
            chosen_aperture = apertures_to_check[idx_max_prec]
            print('\t >> Best precision achieved for target at an aperture of {:} pixels'.format(chosen_aperture))
            print('\t >> Precision achieved: {:.0f} ppm'.format(precision[idx_max_prec]))

        # Now determine the n best comparisons using the target aperture by ranking using correlation coefficient:
        idx_comparison = []
        comp_apertures = []
        comp_correlations = []
        target_flux = data['data']['target_star_' + str(idx_target)]['fluxes_' + str(chosen_aperture) + '_pix_ap'][idx_frames]
        target_flux_err = data['data']['target_star_' + str(idx_target)]['fluxes_' + str(chosen_aperture) + '_pix_ap_err'][
            idx_frames]
        exptimes = data['exptimes']
        for idx_c in idx_all_comps:
            star = 'star_%d' % idx_c
            flux_ap = 'fluxes_%d_pix_ap' % chosen_aperture
            flux_ap_err = 'fluxes_%d_pix_ap_err' % chosen_aperture
            comp_flux = (data['data'][star][flux_ap] / exptimes)[idx_frames]
            comp_flux_err = (data['data'][star][flux_ap_err] / exptimes)[idx_frames]

            # quick test for NaN's on all comparison stars
            isNan = np.isnan(np.sum(np.append(comp_flux, comp_flux_err)))
            if isNan:
                isFloat = isinstance(exptimes[0], float)
                isConsistent = np.all(exptimes == exptimes[0])
                log("ALERT: Reference Star's Flux or Flux Err contains NaNs. Details:")
                log("Star ID: %s\tAperture: %d\t Correct Exptime Format: %r" % (star, chosen_aperture,
                                                                                isFloat and isConsistent))
            # Check the correlation between the target and comparison flux
            result = linregress(target_flux, comp_flux)
            comp_correlations.append(result.rvalue ** 2)

        # set NaNs to 0
        comp_correlations = np.array(comp_correlations)
        comp_correlations[np.isnan(comp_correlations)] = 0

        # get comp_correlations in descending order
        comp_corr_idxsorted = np.argsort(comp_correlations)[::-1]
        comp_corr_sorted = np.array(comp_correlations)[comp_corr_idxsorted]
        log("Sorted Comparison correlations:\n{:}".format(comp_corr_sorted[:10]))
        idx_all_comps_sorted = np.array(idx_all_comps)[comp_corr_idxsorted]
        idx_comparison = idx_all_comps_sorted[0:ncomp]
        # Selecting optimal number of comparisons, if not pre-set with flag
        if ncomp == 0:
            print('\t Selecting optimal number of comparisons')
            closest_yet = np.inf
            idx_optimal_comparison = []
            for i in range(idx_all_comps_sorted.size):
                # Check the target
                relative_flux, relative_flux_err = super_comparison_detrend(data, idx_target, idx_all_comps_sorted[:i + 1],
                                                                            chosen_aperture, all_idx=idx_frames)
                mfilt = median_filter(relative_flux[idx_sort_times])
                prec = np.nanmedian(relative_flux_err) * 1e6
                rms_scatter = get_sigma(relative_flux[idx_sort_times] - mfilt) * 1e6
                rel_diff = np.abs(prec - rms_scatter) / prec
                if rel_diff < closest_yet:
                    ncomp += 1
                    closest_yet = rel_diff
                    idx_optimal_comparison.append(i)
            idx_comparison = idx_all_comps_sorted[idx_optimal_comparison]

        msg1 = '\t {:} comparison stars available'.format(len(idx_all_comps_sorted))
        msg2 = '\t Selected the {:} best: {:}'.format(ncomp, idx_comparison)
        print(msg1)
        print(msg2.replace('\n', '\n\t '))
        log(msg1.replace('\t', ''))
        log(msg2.replace('\t', ''))
        # pdb.set_trace()
        # Plot the color-magnitude diagram
        plot_cmd(colors, data, idx_target, idx_comparison, post_dir)

        comp_apertures = []
        # Check the comparisons, and optionally select their apertures
        if not os.path.exists(post_dir + 'raw_light_curves/'):
            os.mkdir(post_dir + 'raw_light_curves/')
        if not os.path.exists(post_dir + 'comp_light_curves/'):
            os.mkdir(post_dir + 'comp_light_curves/')

        for i_c in idx_comparison:
            # super detrend each comparison star
            idx_c = idx_comparison[idx_comparison != i_c]
            if len(idx_c) == 0:
                idx_c = idx_all_comps_sorted[0:10]
            if optimize_apertures:
                precision = np.zeros(len(apertures_to_check))
                for i_ap in range(len(apertures_to_check)):
                    aperture = apertures_to_check[i_ap]
                    rf_comp, rf_comp_err = super_comparison_detrend(data, i_c, idx_c, aperture, all_idx=idx_frames)
                    mfilt = median_filter(rf_comp[idx_sort_times])
                    precision[i_ap] = get_sigma((rf_comp[idx_sort_times] - mfilt) * 1e6)
                idx_max_prec = np.nanargmin(precision)
                the_aperture = apertures_to_check[idx_max_prec]
                print('\t >> Best precision for star_{:} achieved at an aperture of {:} pixels'.format(i_c, the_aperture))
                print('\t >> Precision achieved: {:.0f} ppm'.format(precision[idx_max_prec]))
            else:
                the_aperture = chosen_aperture

            # Save the raw and detrended light curves
            comp_star_id = 'star_%d' % i_c
            comp_fluxes_id = 'fluxes_%d_pix_ap' % the_aperture
            comp_fluxes_err_id = 'fluxes_%d_pix_ap_err' % the_aperture

            exptimes = data['exptimes']
            comp_flux = (data['data'][comp_star_id][comp_fluxes_id] / exptimes)[idx_frames]
            comp_flux_err = (data['data'][comp_star_id][comp_fluxes_err_id] / exptimes)[idx_frames]
            save_photometry(times[idx_sort_times], comp_flux[idx_sort_times], comp_flux_err[idx_sort_times],
                            post_dir + 'raw_light_curves/',
                            target_name='star_{:}_photometry_ap{:}_pix'.format(i_c, the_aperture),
                            plot_data=True, units='Counts')
            rf_comp, rf_comp_err = super_comparison_detrend(data, i_c, idx_c, the_aperture, all_idx=idx_frames)
            save_photometry(times[idx_sort_times], rf_comp[idx_sort_times], rf_comp_err[idx_sort_times],
                            post_dir + 'comp_light_curves/',
                            target_name='star_{:}_photometry_ap{:}_pix'.format(i_c, the_aperture),
                            plot_data=True, units='Counts')
            comp_apertures.append(the_aperture)

        # Save the raw light curves for the target as well
        target_flux = (data['data']['target_star_%d' % idx_target]['fluxes_%d_pix_ap' % chosen_aperture] / exptimes)[idx_frames]
        target_flux_err = \
            (data['data']['target_star_%d' % idx_target]['fluxes_%d_pix_ap_err' % chosen_aperture] / exptimes)[idx_frames]

        # detect if target_flux or target_flux_err contain NaNs
        # quick test for nans
        isNan = np.isnan(np.sum(np.append(target_flux, target_flux_err)))
        if isNan:
            log("ALERT: Target Star's Flux or Flux Err contains NaNs. Details:")
            log("Star ID: %d\tAperture: %d" % (idx_target, chosen_aperture))

        save_photometry(times[idx_sort_times], target_flux[idx_sort_times], target_flux_err[idx_sort_times],
                        post_dir + 'raw_light_curves/', target_name='target_photometry_ap{:}_pix'.format(chosen_aperture),
                        plot_data=True, units='Counts')
        # And save the super comparison
        _, _, super_comp, super_comp_err = super_comparison_detrend(data, idx_target, idx_all_comps_sorted[0:ncomp],
                                                                    chosen_aperture, comp_apertures=comp_apertures,
                                                                    all_idx=idx_frames, supercomp=True)
        save_photometry(times[idx_sort_times], super_comp[idx_sort_times], super_comp_err[idx_sort_times],
                        post_dir + 'raw_light_curves/',
                        target_name='super_comp_photometry_ap{:}_pix'.format(chosen_aperture),
                        plot_data=True)

        # Saving sub-images
        if plt_images:
            print('\t Plotting and saving sub-images...')
            log('Plotting and saving sub-images...')
            plot_images(data, idx_target, idx_comparison, chosen_aperture, min_ap, max_ap,
                        comp_apertures, post_dir, data['frame_name'][idx_frames],
                        idx_frames, overwrite=overwrite)
        # pdb.set_trace()
        # Save and plot final LCs:
        print('\t Getting final relative flux...')
        relative_flux, relative_flux_err = super_comparison_detrend(data, idx_target, idx_comparison, chosen_aperture,
                                                                     comp_apertures=comp_apertures, plot_comps=all_plots,
                                                                     all_idx=idx_frames)

        # pdb.set_trace()
        print('\t Saving...')
        save_photometry(times[idx_sort_times], relative_flux[idx_sort_times], relative_flux_err[idx_sort_times],
                        post_dir, target_name=target_name, plot_data=True,
                        title=target_name + ' on ' + red_path.split('/')[-1] + ' at ' + site)
        # pdb.set_trace()
        save_photometry_hs(data, idx_target, idx_comparison, idx_all_comps_sorted, chosen_aperture, min_ap, max_ap, comp_apertures,
                           idx_sort_times, post_dir, target_name, band=band, all_idx=idx_frames)

        print('\t Done!\n')
        plt.clf()

if __name__=="__main__":
    ################ INPUT DATA #####################

    parser = argparse.ArgumentParser()
    parser.add_argument('-telescope', default=None)
    parser.add_argument('-datafolder', default=None)
    parser.add_argument('-target_name', default=None)
    parser.add_argument('-ra', default=None)
    parser.add_argument('-dec', default=None)
    parser.add_argument('-band', default='ip')
    parser.add_argument('-dome', default='')
    parser.add_argument('-minap', default=5)
    parser.add_argument('-maxap', default=25)
    parser.add_argument('-apstep', default=1)
    parser.add_argument('-ncomp', default=0)
    parser.add_argument('-forced_aperture', default=15)
    parser.add_argument('--force_aperture', dest='force_aperture', action='store_true')
    parser.set_defaults(force_aperture=False)
    parser.add_argument('--optimize_apertures', dest='optimize_apertures', action='store_true')
    parser.set_defaults(optimize_apertures=False)
    parser.add_argument('--plt_images', dest='plt_images', action='store_true')
    parser.set_defaults(plt_images=False)
    parser.add_argument('--all_plots', dest='all_plots', action='store_true')
    parser.set_defaults(all_plots=False)
    parser.add_argument('--overwrite', dest='overwrite', action='store_true')
    parser.set_defaults(overwrite=False)

    args = parser.parse_args()

    force_aperture = args.force_aperture
    optimize_apertures = args.optimize_apertures
    plt_images = args.plt_images
    all_plots = args.all_plots
    overwrite = args.overwrite
    telescope = args.telescope
    target_name = args.target_name
    datafolder = args.datafolder
    band = args.band
    dome = args.dome
    target_coords = [[args.ra, args.dec.split()[0]]]
    min_ap = int(args.minap)
    max_ap = int(args.maxap)
    forced_aperture = int(args.forced_aperture)
    ncomp = int(args.ncomp)
    filename = 'photometry.pkl'

    post_processing(telescope,datafolder,target_name,target_coords,band,ncomp,min_ap,max_ap,forced_aperture,filename,
                    force_aperture,optimize_apertures,plt_images,all_plots,overwrite)

