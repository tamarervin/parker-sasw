#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 14:58:49 2024

@author: tamarervin

Model validation
"""
import glob
import sys, os
import shutil

import datetime
from datetime import timedelta

import numpy as np
import pandas as pd

from scipy import stats
from scipy.stats import linregress

import astrospice
import astropy.units as u
from astropy.coordinates import SkyCoord

import pyspedas
from pyspedas import time_string, time_double
from pytplot import tplot, get_data, cdf_to_tplot, store_data

import sunpy 
import sunpy.map
from sunpy.net import Fido
from sunpy.net import attrs as a
import sunpy.coordinates as scoords
import sunpy.visualization.colormaps as cm
from sunpy.map.header_helper import make_heliographic_header

from scipy.interpolate import interp1d
from astropy.coordinates import SkyCoord

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
from matplotlib.ticker import MaxNLocator

import plotting as plot
import calculations as calcs
import pfss_funcs as pfss_funcs
from categorize_wind import categorize_wind 

from sasw_sources.functions import read_df, gen_dt_arr, pfss, delta_long, ballistically_project, boxplot

kernels = astrospice.registry.get_kernels('psp','predict') 

### ------------- PLOT STYLING
from matplotlib import rc

# Set the font family to Times New Roman for text and equations
rc('font', **{'family': 'serif', 'serif': ['Times New Roman']})
rc('mathtext', fontset='custom', rm='Times New Roman', it='Times New Roman:italic', bf='Times New Roman:bold')
mpl.rcParams['font.size'] = 18
panel_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)', '(i)', '(j)', '(k)']

### ------------- PATHS
RES_DIR = os.path.realpath('sasw_sources/results')
FIG_DIR = os.path.realpath('sasw_sources/figures')
EPS_DIR = os.path.realpath('sasw_sources/eps_figures')
ADAPT_DIR = os.path.realpath('sasw_sources/adapt')

### ------------- ADAPT MAGNETOGRAMS
adapt_mag=['adapt40311_03k012_202001270000_i00025600n1.fts.gz', #### E4
           'adapt40311_03k012_202006060000_i00005600n1.fts.gz', ### E5
           'adapt40311_03k012_202009290000_i00005600n1.fts.gz', #### E6
           'adapt40311_03k012_202101170000_i00005500n1.fts.gz', #### E7
           'adapt40311_03k012_202105010000_i00005600n1.fts.gz', #### E8
           'adapt41311_03k012_202108020000_i00005600n1.fts.gz', #### E9
           'adapt40311_03k012_202111210000_i00010600n1.fts.gz', #### E10
           'adapt41311_03k012_202202240000_i00005600n1.fts.gz', #### E11
           'adapt40311_03k012_202205280000_i00000600n1.fts.gz', #### E12
           'adapt40311_03k012_202209060000_i00005600n1.fts.gz', #### E13
           '	adapt41311_03k012_202212140000_i00005600n1.fts.gz', #### E14
           # 'adapt40311_03k012_202303160000_i00053600n1.fts.gz', #### E15
           # 'adapt41311_03k012_202306210000_i00005600n1.fts.gz', #### E16
           # 'mrbqs230928t0004c2276_340.fits', #### E17
           ]

### ------------- SASW PERIODS
reg_sasw = [ ['2020-01-28 18:00:00', '2020-01-29 01:00:00', 4], ### E4
            ['2020-06-06 06:00:00', '2020-06-06 09:00:00', 5], ### E5 R1
                ['2020-06-09 10:00:00', '2020-06-09 22:00:00', 5], ### E5 R2
                ['2020-09-27 12:00:00', '2020-09-27 16:00:00', 6], ### E6 R1
                ['2020-09-29 02:00:00', '2020-09-29 05:00:00', 6], ### E6 R2
                ['2021-01-18 20:00:00', '2021-01-19 06:00:00', 7], ### E7
                ['2021-04-30 00:00:00', '2021-04-30 05:00:00', 8], ### E8 R1
                ['2021-04-30 12:00:00', '2021-04-30 19:00:00', 8], ### E8 R2
                ['2022-12-10 18:00:00', '2022-12-10 20:00:00', 14], ### E14 R1
                ['2022-12-10 20:30:00', '2022-12-10 22:00:00', 14], ### E14 R2
                ['2022-12-12 12:00:00', '2022-12-12 16:00:00', 14], ### E14 R3
                # ['2023-06-21 08:00:00', '2023-06-21 16:00:00', 16], ### E16 R1
                # ['2023-06-23 08:00:00', '2023-06-24 14:00:00', 16], ### E16 R2
                # ['2023-09-27 12:00:00', '2023-09-27 14:00:00', 17], ### E17
                ]
dates_sasw =['20200128_E4', '20200606_E5', '20200609_E5', '20200927_E6', '20200929_E6',
       '20210118_E7', '20210430_E8-1', '20210430_E8-2', '20221210_E14-1', '20221210_E14-2',
       '20221212_E14']

### ------------- FSW PERIODS
reg_fsw = [ 
        ['2020-01-27 04:55:00', '2020-01-27 05:15:00', 4], ### E4
        # ['2020-06-06 00:00:00', '2020-06-06 09:00:00', 5], ### E5
        ['2021-04-27 01:00:00', '2021-04-27 04:00:00', 8], ### E8
        ['2021-04-27 06:00:00', '2021-04-27 08:30:00', 8], ### E8
        ['2021-11-20 05:30:00', '2021-11-20 10:30:00', 10], ### E10
        # ['2022-05-29 21:30:00', '2022-05-30 23:30:00', 12], ### E12
        ['2022-12-15 21:15:00', '2022-12-15 22:15:00', 14], ### E14
        ]
dates_fsw = [
'2020-01-27', '2021-04-27',  '2021-04-27_2', '2021-11-20',  '2022-12-14'
          ]

### ------------- IDENTIFIED STREAMS
reg_streams = [
    ['2021-01-18 20:00:00', '2021-01-19 06:00:00', 7], ### E7
    ['2020-09-27 12:00:00', '2020-09-27 16:00:00', 6], ### E6 R1
    ['2021-11-20 05:30:00', '2021-11-20 10:30:00', 10], ### E10
    ]
dates_streams =['20210118_E7', '20200927_E6', '2021-11-20'
    ]
# =============================================================================
#     INDUCE NOISE IN VELOCITY
# =============================================================================
percent_noise, rss = 4, 2.5
fsw = False
if fsw:
    regs, dates = reg_fsw, dates_fsw
else:
    regs, dates = reg_sasw, dates_sasw
for i, [reg, date] in enumerate(zip(regs, dates)):
    if i < 9:
        pass
    else:
        enc =f'e{reg[2]}'
        ### read in dataframe
        df = read_df(filepath=os.path.join(RES_DIR, f"{enc}_modeling.csv"))
        df_reg = df[(df.Time >= datetime.datetime.strptime(reg[0], '%Y-%m-%d %H:%M:%S')) & (df.Time <= datetime.datetime.strptime(reg[1], '%Y-%m-%d %H:%M:%S'))]
        df_velnoise = df_reg.copy()
        # df_velnoise = pd.read_csv(os.path.join(f'validation_results', f'vel_noise_{date}.csv'))
        # df_velnoise['Time'] = df_reg.Time
        # df_velnoise.to_csv(os.path.join(f'validation_results', f'vel_noise_{date}.csv'))
        ### calculate noise
        vel_noise = (percent_noise / 100) * np.nanmean(df_reg.vr)
        
        ### get inertial PSP coordinates 
        psp_coords_inertial = astrospice.generate_coords('SOLAR PROBE PLUS', df_reg.Time)
        
        ### Transform to Heliographic Carrington, i.e. the frame that co-rotates with the Sun.
        psp_coords_carr = psp_coords_inertial.transform_to(
        sunpy.coordinates.HeliographicCarrington(observer="self"))    
        
        ### run PFSS model
        j = reg[2] - 4
        adapt_filepath = f"{ADAPT_DIR}/{adapt_mag[j]}"
        pfss_model = pfss_funcs.adapt2pfsspy(adapt_filepath,rss)
        
        ### ------------- INDUCE NOISE IN VELOCITY 
        velnoise_sslon, velnoise_sslat, velnoise_flon, velnoise_flat = [], [], [], []
        for bootstrap in np.arange(100):
            # calculate velocity noise
            noise = np.random.uniform(-1*vel_noise, 1*vel_noise, len(df_reg.vr))
            vel = df_reg.vr + noise
            sscoords = ballistically_project(psp_coords_carr,vr_arr=np.array(vel)*u.km/u.s, r_inner=rss*u.Rsun)
            velnoise_sslon.append(sscoords.lon.value)
            velnoise_sslat.append(sscoords.lat.value)
            
            # get footpoints
            fpsp = pfss_funcs.pfss2flines(pfss_model, skycoord_in=sscoords)
            use = np.where(fpsp.connectivities == 1)[0]
            flon = np.full(len(df_reg.Time), np.nan)
            flat = np.full(len(df_reg.Time), np.nan)
            flon[use] = fpsp.open_field_lines.solar_feet.lon.value
            flat[use] = fpsp.open_field_lines.solar_feet.lat.value
            velnoise_flon.append(flon)
            velnoise_flat.append(flat)
            
        ### calculate source surface averages
        sslon_avg =  np.nanmean(velnoise_sslon, axis=0)
        sslat_avg =  np.nanmean(velnoise_sslat, axis=0)
        sslon_std =  np.std(velnoise_sslon, axis=0)
        sslat_std  =  np.std(velnoise_sslat, axis=0)
        
        ### calculate footpoint averages
        flon_avg =  np.nanmean(velnoise_flon, axis=0)
        flat_avg =  np.nanmean(velnoise_flat, axis=0)
        flon_std =  np.std(velnoise_flon, axis=0)
        flat_std  =  np.std(velnoise_flat, axis=0)
        
        ### add to dataframe
        df_velnoise['Time'] = df_reg.Time
        df_velnoise['vel_noise_sslon_avg'] = sslon_avg
        df_velnoise['vel_noise_sslat_avg'] = sslat_avg
        df_velnoise['vel_noise_sslon_std'] = sslon_std
        df_velnoise['vel_noise_sslat_std'] = sslat_std
        df_velnoise['vel_noise_flon_avg'] = flon_avg
        df_velnoise['vel_noise_flat_avg'] = flat_avg
        df_velnoise['vel_noise_flon_std'] = flon_std
        df_velnoise['vel_noise_flat_std'] = flat_std
        
        if fsw:
            df_velnoise.to_csv(os.path.join(f'validation_results', f'fsw_vel_noise_{date}.csv'))
        else:
            df_velnoise.to_csv(os.path.join(f'validation_results', f'vel_noise_{date}.csv'))
        del df_velnoise, velnoise_sslon, velnoise_sslat, velnoise_flon, velnoise_flat
        print('Completed calculation for', date, 'and saved to CSV with noise level', vel_noise)
    
# =============================================================================
#     TESTING SOURCE SURFACE HEIGHT
# =============================================================================
RSS = [1.5, 2.0, 2.5, 3.0, 3.5]
fsw = False
if fsw:
    regs, dates = reg_fsw, dates_fsw
else:
    regs, dates = reg_sasw, dates_sasw
for i, [reg, date] in enumerate(zip(regs, dates)):
    if i < 5:
        pass
    else:
        df_velnoise = pd.DataFrame()
        enc =f'e{reg[2]}'
        j = reg[2] - 4
        adapt_filepath = f"{ADAPT_DIR}/{adapt_mag[j]}"
        ### read in dataframe
        df = read_df(filepath=os.path.join(RES_DIR, f"{enc}_modeling.csv"))
        df_reg = df[(df.Time >= datetime.datetime.strptime(reg[0], '%Y-%m-%d %H:%M:%S')) & (df.Time <= datetime.datetime.strptime(reg[1], '%Y-%m-%d %H:%M:%S'))]
    
        ### get inertial PSP coordinates 
        psp_coords_inertial = astrospice.generate_coords('SOLAR PROBE PLUS', df_reg.Time)
        
        ### Transform to Heliographic Carrington, i.e. the frame that co-rotates with the Sun.
        psp_coords_carr = psp_coords_inertial.transform_to(
        sunpy.coordinates.HeliographicCarrington(observer="self"))    
        
        df_Rss = df_reg.copy()
        for k, rss in enumerate(RSS):
            # run PFSS model
            pfss_model = pfss_funcs.adapt2pfsspy(adapt_filepath,rss)
    
            # get field lines
            psp_ss = ballistically_project(psp_coords_carr,vr_arr=np.array(df_reg.vr)*u.km/u.s, r_inner=rss*u.Rsun)
            fpsp = pfss_funcs.pfss2flines(pfss_model, skycoord_in=psp_ss)
    
            # add column to dataframe
            use = np.where(fpsp.connectivities == 1)[0]
            # df_Rss['sslon'+str(rss)] = np.nan* len(df_reg.Time)
            # df_Rss['sslat'+str(rss)] = np.nan* len(df_reg.Time)
            df_Rss['Time'] = df_reg.Time
            df_Rss['sslon'+str(rss)] = psp_ss.lon.value
            df_Rss['sslat'+str(rss)] = psp_ss.lat.value
            flon = np.full(len(df_reg.Time), np.nan)
            flat = np.full(len(df_reg.Time), np.nan)
            flon[use] = fpsp.open_field_lines.solar_feet.lon.value
            flat[use] = fpsp.open_field_lines.solar_feet.lat.value
            df_Rss['flon'+str(rss)] = flon
            df_Rss['flat'+str(rss)] =flat
            # pfss_lon.append(fpsp.open_field_lines.solar_feet.lon.value)
            # pfss_lat.append(fpsp.open_field_lines.solar_feet.lat.value)
    
            # print 
            print('Finished with source surface height:', rss)
        
        if fsw:
            df_Rss.to_csv(os.path.join(f'sasw_sources/validation_results', f'fsw_Rss_{date}.csv'))
        else:
            df_Rss.to_csv(os.path.join(f'sasw_sources/validation_results', f'Rss_{date}.csv'))
        del df_Rss
    print('Completed calculation for', date, 'and saved to CSV')
   
# =============================================================================
#     TESTING MAGNETOGRAM EFFECT
# =============================================================================
regs, dates = reg_streams, dates_streams
rss = 2.5
for i, [reg, date] in enumerate(zip(regs, dates)):
    enc =f'e{reg[2]}'
    adapt_filepath = f"{ADAPT_DIR}/{enc}"
    
    ### read in dataframe
    df = read_df(filepath=os.path.join(RES_DIR, f"{enc}_modeling.csv"))
    df_reg = df[(df.Time >= datetime.datetime.strptime(reg[0], '%Y-%m-%d %H:%M:%S')) & (df.Time <= datetime.datetime.strptime(reg[1], '%Y-%m-%d %H:%M:%S'))]

    ### get inertial PSP coordinates 
    psp_coords_inertial = astrospice.generate_coords('SOLAR PROBE PLUS', df_reg.Time)
    
    ### Transform to Heliographic Carrington, i.e. the frame that co-rotates with the Sun.
    psp_coords_carr = psp_coords_inertial.transform_to(
    sunpy.coordinates.HeliographicCarrington(observer="self"))    
    
    # ballistic propagation
    psp_ss = ballistically_project(psp_coords_carr,vr_arr=np.array(df_reg.vr)*u.km/u.s, r_inner=rss*u.Rsun)
    
    ### create dataframe
    df_magnetogram = df_reg.copy()
    ### get magnetigrams
    magnetograms = glob.glob(os.path.join(adapt_filepath, '*'))
    ### loop through and test magnetograms
    for ii, mag in enumerate(magnetograms):
        # run PFSS model
        pfss_model = pfss_funcs.adapt2pfsspy(mag, rss)

        # get field lines
        psp_ss = ballistically_project(psp_coords_carr,vr_arr=np.array(df_reg.vr)*u.km/u.s, r_inner=rss*u.Rsun)
        fpsp = pfss_funcs.pfss2flines(pfss_model, skycoord_in=psp_ss)
        
        # add column to dataframe
        use = np.where(fpsp.connectivities == 1)[0]
        df_magnetogram['sslon_mag' + str(ii)] = psp_ss.lon.value
        df_magnetogram['sslat_mag' + str(ii)] = psp_ss.lat.value
        flon = np.full(len(df_reg.Time), np.nan)
        flat = np.full(len(df_reg.Time), np.nan)
        flon[use] = fpsp.open_field_lines.solar_feet.lon.value
        flat[use] = fpsp.open_field_lines.solar_feet.lat.value
        df_magnetogram['flon_mag' + str(ii)] = flon
        df_magnetogram['flat_mag' + str(ii)] =flat
    # print 
        print('Finished with magnetogram:', mag)
    
    df_magnetogram.to_csv(os.path.join(f'sasw_sources/validation_results', f'mag_{date}.csv'))
    del df_magnetogram
print('Completed calculation for', date, 'and saved to CSV')

# =============================================================================
# =============================================================================
# =============================================================================
# # #     FIGURES
# =============================================================================
# =============================================================================
# =============================================================================
    
# =============================================================================
# ### BALLISITIC PROPAGATION FIGURE 
# choose three streams of varied speeds to show effect of vR error on source surface
# trajectory
# stream one (178 km/s): 20221210_E14-2
# stream two (306 km/s): 20200927_E6
# stream one (563 km/s): 20211120_E10 (FSW)
# =============================================================================
stream_one = read_df('/Users/tamarervin/development/sasw_sources/validation_results/vel_noise_20210118_E7.csv')
stream_two = read_df('/Users/tamarervin/development/sasw_sources/validation_results/vel_noise_20200927_E6.csv')
stream_three = read_df('/Users/tamarervin/development/sasw_sources/validation_results/fsw_vel_noise_2021-11-20.csv')
streams = [stream_one, stream_two, stream_three]
colors = ['tab:blue', 'tab:purple', 'tab:red']
titles = ['2021-01-18 18:00 to 01-19 06:00 (E7) \n$v_R$ = 231 km/s',
          '2020-09-27 12:00 to 16:00 (E6) \n$v_R$ = 306 km/s',
          '2021-11-20 05:30 to 10:30 (E10) \n$v_R$ = 563 km/s']

### ------------- AIA IMAGES
def get_aia(start_date, end_date, left, bot, width=50, height=40, cadence= a.Sample(24*u.hour)):
    aia_result = Fido.search(a.Time(start_date, end_date),
                         a.Instrument.aia, a.Wavelength(193 * u.angstrom), cadence)
    file_download = Fido.fetch(aia_result)
    aia_map = sunpy.map.Map(sorted(file_download))
    shape = (720, 1440)
    carr_header = make_heliographic_header(aia_map.date, aia_map.observer_coordinate, shape, frame='carrington')
    outmap = aia_map.reproject_to(carr_header)
    bottom_left = SkyCoord(left*u.degree, bot*u.degree, frame=outmap.coordinate_frame)
    submap = outmap.submap(bottom_left, width=width*u.degree, height=height*u.degree)
    return aia_map, submap

lefts = [90, 240, 330]
bots = [-50, -80, -40]
_, map_one = get_aia( start_date ='2021-01-18 00:00:00' , end_date = '2021-01-19 00:00:00', left=90, bot=-50)
_, map_two = get_aia( start_date ='2020-09-16 00:00:00' , end_date = '2020-09-17 00:00:00', left=240, bot=-80)
_, map_three = get_aia( start_date ='2021-11-20 00:00:00' , end_date = '2021-11-21 00:00:00', left=330, bot=-40)
maps = [map_one, map_two, map_three]

### ------------- FINAL FIGURE
out = False
nf = 16
fig = plt.figure(figsize=(40, 26))
gs = mpl.gridspec.GridSpec(3, 3, height_ratios=[0.3, 1, 1], wspace=0.15, hspace=0.16) 

### error bars
xlabel='Estimated Footpoint Error [deg]'
axs = [fig.add_subplot(gs[0, i]) for i in np.arange(3)]
for ax, stream, col, title, panel in zip(axs, streams, colors, titles, panel_labels):
    ax.set_title(title, fontsize=24)
    boxplot(stream['vel_noise_flon_std'], 'tab:red', 'lightpink', ax, 0.5, out)
    boxplot(stream['vel_noise_flat_std'], 'tab:purple', 'lavender', ax, 1, out)
    ax.set(yticks=[0.5, 1], yticklabels=['Longitude', 'Latitude'])
    ax.set_xlabel(xlabel, fontsize=20)
    ax.set_ylim(0.25, 1.25)
    ax.text(0.95, 0.95, panel, transform=ax.transAxes, fontsize=20, fontweight='bold', va='top', ha='left', zorder=10)
    
### source surface
axs = [fig.add_subplot(gs[1, i]) for i in np.arange(3)]
for ax, stream, col, title, panel in zip(axs, streams, colors, titles, panel_labels[3:]):
    sslon_avg = stream['vel_noise_sslon_avg'] 
    sslat_avg = stream['vel_noise_sslat_avg'] 
    sslon_std = stream['vel_noise_sslon_std']
    sslat_std = stream['vel_noise_sslat_std'] 
    index = np.linspace(1, len(sslon_avg), len(sslon_avg))
    ax.scatter(stream.Time[::nf], sslon_avg[::nf], marker='D', facecolor='None', edgecolor=col, lw=0.5, s=40)
    ax.errorbar(stream.Time[::nf], sslon_avg[::nf], yerr=sslon_std[::nf], fmt='None', ecolor=col, capsize=5, elinewidth=1, markeredgewidth=1)
    # ax.set_title(title, fontsize=20)
    ax.grid(True, linestyle='dashed', lw=0.2, zorder=-1)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.text(0.95, 0.95, panel, transform=ax.transAxes, fontsize=20, fontweight='bold', va='top', ha='left', zorder=10)
axs[0].set_ylabel('Source Surface Longitude [deg]', fontsize=20)

### footpoints
axs = [fig.add_subplot(gs[2, i]) for i in np.arange(3)]
for ax, stream, col, title, submap, left, bot, panel in zip(axs, streams, colors, titles, maps, lefts, bots, panel_labels[6:]):
    ### plot aia map
    data = submap.data
    x, y = np.meshgrid(np.arange(left, left+50, 50/data.shape[1]), np.arange(bot, bot+40, 40/data.shape[0]))
    lognorm = mpl.colors.LogNorm(vmin=np.nanpercentile(submap.data.flatten(),40), 
                            vmax=np.nanpercentile(submap.data.flatten(),99.9))
    ax.pcolormesh(x, y, data, norm=lognorm, cmap='sdoaia193')
    ### plot footpoints
    sslon_avg = stream['vel_noise_flon_avg'] 
    sslat_avg = stream['vel_noise_flat_avg'] 
    sslon_std = stream['vel_noise_flon_std']
    sslat_std = stream['vel_noise_flat_std'] 
    index = np.linspace(1, len(sslon_avg), len(sslon_avg))
    ax.scatter(sslon_avg[::nf], sslat_avg[::nf], marker='D', facecolor='None', edgecolor=col, lw=0.5, s=40)
    ax.errorbar(sslon_avg[::nf], sslat_avg[::nf], xerr=sslon_std[::nf], yerr=sslat_std[::nf], fmt='None', ecolor=col, capsize=5, elinewidth=1, markeredgewidth=1)
    ax.grid(True, linestyle='dashed', lw=0.2, zorder=-1)
    ax.set_xlabel('Carrington Longitude [deg]', fontsize=20)
    ax.set(xlim=(left, left+50), xticks=np.linspace(left, left+50, 6),
           ylim=(bot, bot+40), yticks=np.linspace(bot, bot+40, 5))
    ax.text(0.95, 0.95, panel, transform=ax.transAxes, fontsize=20, fontweight='bold', va='top', ha='left', zorder=10, c='white')
axs[0].set_ylabel('Carrington Latitude [deg]', fontsize=20)

plt.savefig(os.path.join(FIG_DIR, 'ballistic_noise.png'), bbox_inches='tight')
plt.savefig(os.path.join(EPS_DIR, 'ballistic_noise.eps'), bbox_inches='tight')

# =============================================================================
# ### SOURCE SURFACE FIGURE 
# choose three streams of varied speeds to show effect of vR error on source surface
# trajectory
# stream one (178 km/s): 20221210_E14-2
# stream two (306 km/s): 20200927_E6
# stream one (563 km/s): 20211120_E10 (FSW)
# =============================================================================
stream_one = pd.read_csv('/Users/tamarervin/development/sasw_sources/validation_results/Rss_20210118_E7.csv')
stream_two = pd.read_csv('/Users/tamarervin/development/sasw_sources/validation_results/Rss_20200927_E6.csv')
stream_three = pd.read_csv('/Users/tamarervin/development/sasw_sources/validation_results/fsw_Rss_2021-11-20.csv')
streams = [stream_one, stream_two, stream_three]
colors = ['tab:purple', 'tab:purple', 'tab:red']
RSS = [1.5, 2.0, 2.5, 3.0, 3.5]

### CALCULATE AVERAGE AND STD
for stream in streams:
    stream['flon_avg'] = np.nanmean([stream['flon'+str(rss)] for rss in RSS], axis=0)
    stream['flat_avg'] = np.nanmean([stream['flat'+str(rss)] for rss in RSS], axis=0)
    stream['flon_std'] = np.std([stream['flon'+str(rss)] for rss in RSS], axis=0)
    stream['flat_std'] = np.std([stream['flat'+str(rss)] for rss in RSS], axis=0)

### ------------- FINAL FIGURE
out = False
nf = 16
fig = plt.figure(figsize=(40, 16))
gs = mpl.gridspec.GridSpec(2, 3, height_ratios=[0.3, 1], wspace=0.15, hspace=0.16) 

### error bars
xlabel='Estimated Footpoint Error [deg]' 
axs = [fig.add_subplot(gs[0, i]) for i in np.arange(3)]
for ax, stream, col, title, panel in zip(axs, streams, colors, titles, panel_labels):
    ax.set_title(title, fontsize=24)
    boxplot(stream['flon_std'], 'tab:red', 'lightpink', ax, 0.5, out)
    boxplot(stream['flat_std'], 'tab:purple', 'lavender', ax, 1, out)
    ax.set(yticks=[0.5, 1], yticklabels=['Longitude', 'Latitude'])
    ax.set_xlabel(xlabel, fontsize=20)
    ax.set_ylim(0.25, 1.25)
    ax.text(0.95, 0.95, panel, transform=ax.transAxes, fontsize=20, fontweight='bold', va='top', ha='left', zorder=10)

### footpoints
axs = [fig.add_subplot(gs[1, i]) for i in np.arange(3)]
for ax, stream, col, title, submap, left, bot, panel in zip(axs, streams, colors, titles, maps, lefts, bots, panel_labels[3:]):
    ### plot aia map
    data = submap.data
    x, y = np.meshgrid(np.arange(left, left+50, 50/data.shape[1]), np.arange(bot, bot+40, 40/data.shape[0]))
    lognorm = mpl.colors.LogNorm(vmin=np.nanpercentile(submap.data.flatten(),40), 
                            vmax=np.nanpercentile(submap.data.flatten(),99.9))
    ax.pcolormesh(x, y, data, norm=lognorm, cmap='sdoaia193')
    ### plot footpoints
    for rss, col in zip(RSS, ['tab:red', 'tab:orange', 'tab:green', 'tab:blue', 'tab:purple']):
        flon = stream['flon'+str(rss)] 
        flat = stream['flat'+str(rss)]
        ax.scatter(flon[::nf], flat[::nf], marker='D', facecolor='None', edgecolor=col, lw=0.5, s=40, zorder=10)
    ax.grid(True, linestyle='dashed', lw=0.2, zorder=-1)
    ax.set_xlabel('Carrington Longitude [deg]', fontsize=20)
    ax.set(xlim=(left, left+50), xticks=np.linspace(left, left+50, 6),
            ylim=(bot, bot+40), yticks=np.linspace(bot, bot+40, 5))
    ax.text(0.95, 0.95, panel, transform=ax.transAxes, fontsize=20, fontweight='bold', va='top', ha='left', zorder=10, c='white')
axs[0].set_ylabel('Carrington Latitude [deg]', fontsize=20)

### legend
patches = [mpatches.Patch(edgecolor='k', facecolor=col, label=str(rss)+r'$R_{\odot}$') for rss, col in zip(RSS, 
                                                                                                        ['tab:red', 'tab:orange', 'tab:green', 'tab:blue', 'tab:purple'])]
leg0 = ax.legend(handles=patches, loc='upper left', ncol=2, fontsize=20) 
ax.add_artist(leg0)


plt.savefig(os.path.join(FIG_DIR, 'source_surface.png'), bbox_inches='tight')
plt.savefig(os.path.join(EPS_DIR, 'source_surface.eps'), bbox_inches='tight')

# =============================================================================
# ### NEUTRAL LINE VALIDATION
# validation of the neutral line crossing for different source surface heights for multiple streams
# stream one (178 km/s): 20221210_E14-2
# stream two (306 km/s): 20200927_E6
# stream one (563 km/s): 20211120_E10 (FSW)
# =============================================================================
stream_one = pd.read_csv('/Users/tamarervin/development/sasw_sources/validation_results/Rss_20210118_E7.csv')
stream_two = pd.read_csv('/Users/tamarervin/development/sasw_sources/validation_results/Rss_20200927_E6.csv')
stream_three = pd.read_csv('/Users/tamarervin/development/sasw_sources/validation_results/fsw_Rss_2021-11-20.csv')
streams = [stream_one, stream_two, stream_three]
colors = ['tab:purple', 'tab:purple', 'tab:red']
RSS = [1.5, 2.0, 2.5, 3.0, 3.5]

reg_streams = [
    ['2021-01-18 20:00:00', '2021-01-19 06:00:00', 7], ### E7
    ['2020-09-27 12:00:00', '2020-09-27 16:00:00', 6], ### E6 R1
    ['2021-11-20 05:30:00', '2021-11-20 10:30:00', 10], ### E10
    ]
xlimits = [[0, 200],
           [150, 360], 
           [0, 360]]

### PLOT ALL ON SAME SUBPLOT
# nf  = 60
# fig, axs = plt.subplots(3, 1, figsize=(12, 20), gridspec_kw={'hspace':0.25, 'wspace':0.25})
# axs[-1].set(xlabel='Carrington Longitude [deg]')
# for stream, ax, reg, xlim in zip(streams, axs, reg_streams, xlimits):
#         ### set axes bounds
#         ax.set(xlim=xlim, ylim=(-30, 30), ylabel='Carrington Latitude [deg]')
#         ### get dataframe associated with encounter
#         enc = f'e{reg[2]}'

#         ### read in dataframe
#         df = read_df(filepath=os.path.join(RES_DIR, f"{enc}_modeling.csv"))
    
#         ### get inertial PSP coordinates 
#         psp_coords_inertial = astrospice.generate_coords('SOLAR PROBE PLUS', df.Time[::nf])
        
#         ### Transform to Heliographic Carrington, i.e. the frame that co-rotates with the Sun.
#         psp_coords_carr = psp_coords_inertial.transform_to(
#         sunpy.coordinates.HeliographicCarrington(observer="self"))    
        

#         for rss, col, negcol in zip(RSS, ['tab:red', 'tab:orange', 'tab:green', 'tab:blue', 'tab:purple'],
#                                     ['darkred', 'darkgoldenrod','darkolivegreen', 'navy', 'indigo']):
#             ### calculate source surface trajectory
#             psp_ss = ballistically_project(psp_coords_carr,vr_arr=np.array(df.vr[::nf])*u.km/u.s, r_inner=rss*u.Rsun)

#             # run PFSS model
#             j = reg[2] - 4
#             adapt_filepath = f"{ADAPT_DIR}/{adapt_mag[j]}"
#             pfss_model = pfss_funcs.adapt2pfsspy(adapt_filepath,rss)

#             ### plot the HCS
#             hcs = pfss_model.source_surface_pils[0]
#             ax.plot(hcs.lon.value, hcs.lat.value, c=col, lw=0.5,  zorder=5)
            
#             ### plot the neutral line
#             polarity = np.sign(df.Br)
#             ax.scatter(psp_ss.lon.value[polarity[::nf]==-1], psp_ss.lat.value[polarity[::nf]==-1], marker='D', facecolor='None', edgecolor=col, lw=0.5, s=10, zorder=10)
#             ax.scatter(psp_ss.lon.value[polarity[::nf]==1], psp_ss.lat.value[polarity[::nf]==1], marker='o',  facecolor='None', edgecolor=negcol, lw=0.5, s=10, zorder=10)
            
# ### legend
# ax = axs[0]
# patches = [mpatches.Patch(facecolor=fcol, edgecolor='k', label=f'{rss}'+ r'$\rm R_{\odot}$') for rss, fcol in zip(RSS, 
#                                                                                                    ['tab:red', 'tab:orange', 'tab:green', 'tab:blue', 'tab:purple'])]
# leg0 = ax.legend(handles=patches, loc='upper right', fontsize=20)
# ax.add_artist(leg0)



### PLOT ON DIFFERENT SUBPLOTS
nf  = 60
fig, axes = plt.subplots(3, 5, figsize=(40, 20), gridspec_kw={'hspace':0.15, 'wspace':0.15}, sharey='all')
for stream, axs, reg, xlim in zip(streams, axes, reg_streams, xlimits):
       
        ### get dataframe associated with encounter
        enc = f'e{reg[2]}'

        ### read in dataframe
        df = read_df(filepath=os.path.join(RES_DIR, f"{enc}_modeling.csv"))
    
        ### get inertial PSP coordinates 
        psp_coords_inertial = astrospice.generate_coords('SOLAR PROBE PLUS', df.Time[::nf])
        
        ### Transform to Heliographic Carrington, i.e. the frame that co-rotates with the Sun.
        psp_coords_carr = psp_coords_inertial.transform_to(
        sunpy.coordinates.HeliographicCarrington(observer="self"))    
        

        for ax, rss, col, negcol in zip(axs, RSS, ['tab:red', 'tab:orange', 'tab:green', 'tab:blue', 'tab:purple'],
                                    ['darkred', 'darkgoldenrod','darkolivegreen', 'navy', 'indigo']):
            ### set axes bounds
            ax.set(xlim=xlim, ylim=(-30, 30))
            
            ### calculate source surface trajectory
            psp_ss = ballistically_project(psp_coords_carr,vr_arr=np.array(df.vr[::nf])*u.km/u.s, r_inner=rss*u.Rsun)

            # run PFSS model
            j = reg[2] - 4
            adapt_filepath = f"{ADAPT_DIR}/{adapt_mag[j]}"
            pfss_model = pfss_funcs.adapt2pfsspy(adapt_filepath,rss)
            flines_psp = pfss_funcs.pfss2flines(pfss_model, skycoord_in=psp_ss)

            ### plot the HCS
            hcs = pfss_model.source_surface_pils[0]
            ax.plot(hcs.lon.value, hcs.lat.value, c=col, lw=1.0,  zorder=5)
            
            ### plot the neutral line
            polarity = np.sign(df.Br)
            ax.scatter(psp_ss.lon.value[polarity[::nf]==-1], psp_ss.lat.value[polarity[::nf]==-1], marker='D', facecolor='None', edgecolor='red', lw=0.5, s=10, zorder=10)
            ax.scatter(psp_ss.lon.value[polarity[::nf]==1], psp_ss.lat.value[polarity[::nf]==1], marker='o',  facecolor='None', edgecolor='blue', lw=0.5, s=10, zorder=10)
            
            ### legend
            per_match = int(100 * len(np.where(flines_psp.polarities == polarity[::nf])[0]) / len(flines_psp.polarities))
            patch = mpatches.Patch(facecolor=col, edgecolor='k', label=f'{rss}' + r'$\rm R_\odot: $' + f'{per_match} %')
            leg0 = ax.legend(handles=[patch], loc='upper right', fontsize=20)
            ax.add_artist(leg0)
### axes labels
fig.text(0.5, 0.09, 'Carrington Longitude [deg]', ha='center', va='center')
fig.text(0.09, 0.5, 'Carrington Latitude [deg]', ha='center', va='center', rotation='vertical')
axes[0][0].set(ylabel=titles[0])
axes[1][0].set(ylabel=titles[1])
axes[2][0].set(ylabel=titles[2])

plt.savefig(os.path.join(FIG_DIR, 'polarity_val.png'), bbox_inches='tight')
plt.savefig(os.path.join(EPS_DIR, 'polarity_val.eps'), bbox_inches='tight')

# =============================================================================
# ### INOUT MAGNETOGRAM
# looking at the effect of changing the input magnetogram on the results
# stream one (178 km/s): 20221210_E14-2
# stream two (306 km/s): 20200927_E6
# stream one (563 km/s): 20211120_E10 (FSW)
# =============================================================================
stream_one = pd.read_csv('/Users/tamarervin/development/sasw_sources/validation_results/mag_20210118_E7.csv')
stream_two = pd.read_csv('/Users/tamarervin/development/sasw_sources/validation_results/mag_20200927_E6.csv')
stream_three = pd.read_csv('/Users/tamarervin/development/sasw_sources/validation_results/mag_2021-11-20.csv')
streams = [stream_one, stream_two, stream_three]
colors = ['tab:blue', 'tab:purple', 'tab:red']


### CALCULATE AVERAGE AND STD
for stream in streams:
    stream['flon_avg'] = np.nanmean([stream['flon_mag'+str(ii)] for ii in np.arange(5)], axis=0)
    stream['flat_avg'] = np.nanmean([stream['flat_mag'+str(ii)] for ii in np.arange(5)], axis=0)
    stream['flon_std'] = np.std([stream['flon_mag'+str(ii)] for ii in np.arange(5)], axis=0)
    stream['flat_std'] = np.std([stream['flat_mag'+str(ii)] for ii in np.arange(5)], axis=0)

### ------------- FINAL FIGURE
out = False
nf = 16
fig = plt.figure(figsize=(40, 18))
gs = mpl.gridspec.GridSpec(2, 3, height_ratios=[0.3, 1], wspace=0.15, hspace=0.16) 

### error bars
xlabel='Estimated Footpoint Error [deg]'
axs = [fig.add_subplot(gs[0, i]) for i in np.arange(3)]
for ax, stream, col, title, panel in zip(axs, streams, colors, titles, panel_labels):
    ax.set_title(title, fontsize=24)
    boxplot(stream['flon_std'], 'tab:red', 'lightpink', ax, 0.5, out)
    boxplot(stream['flat_std'], 'tab:purple', 'lavender', ax, 1, out)
    ax.set(yticks=[0.5, 1], yticklabels=['Longitude', 'Latitude'])
    ax.set_xlabel(xlabel, fontsize=20)
    ax.set_ylim(0.25, 1.25)
    ax.text(0.95, 0.95, panel, transform=ax.transAxes, fontsize=20, fontweight='bold', va='top', ha='left', zorder=10)
    
### footpoints
axs = [fig.add_subplot(gs[1, i]) for i in np.arange(3)]
for ax, stream, col, title, submap, left, bot, panel in zip(axs, streams, colors, titles, maps, lefts, bots, panel_labels[3:]):
    ### plot aia map
    data = submap.data
    x, y = np.meshgrid(np.arange(left, left+50, 50/data.shape[1]), np.arange(bot, bot+40, 40/data.shape[0]))
    lognorm = mpl.colors.LogNorm(vmin=np.nanpercentile(submap.data.flatten(),40), 
                            vmax=np.nanpercentile(submap.data.flatten(),99.9))
    ax.pcolormesh(x, y, data, norm=lognorm, cmap='sdoaia193')
    ### plot footpoints
    sslon_avg = stream['flon_avg'] 
    sslat_avg = stream['flat_avg'] 
    sslon_std = stream['flon_std']
    sslat_std = stream['flat_std'] 
    index = np.linspace(1, len(sslon_avg), len(sslon_avg))
    ax.scatter(sslon_avg[::nf], sslat_avg[::nf], marker='D', facecolor='None', edgecolor=col, lw=0.5, s=40)
    ax.errorbar(sslon_avg[::nf], sslat_avg[::nf], xerr=sslon_std[::nf], yerr=sslat_std[::nf], fmt='None', ecolor=col, capsize=5, elinewidth=1, markeredgewidth=1)
    ax.grid(True, linestyle='dashed', lw=0.2, zorder=-1)
    ax.set_xlabel('Carrington Longitude [deg]', fontsize=20)
    ax.set(xlim=(left, left+50), xticks=np.linspace(left, left+50, 6),
           ylim=(bot, bot+40), yticks=np.linspace(bot, bot+40, 5))
    ax.text(0.95, 0.95, panel, transform=ax.transAxes, fontsize=20, fontweight='bold', va='top', ha='left', zorder=10, c='white')
axs[0].set_ylabel('Carrington Latitude [deg]', fontsize=20)

plt.savefig(os.path.join(FIG_DIR, 'input_mag.png'), bbox_inches='tight')
plt.savefig(os.path.join(EPS_DIR, 'input_mag.eps'), bbox_inches='tight')


# =============================================================================
#     #### CODE 
# =============================================================================
plt.errorbar(sslon_avg, sslat_avg, xerr=sslon_std, yerr=sslat_std, fmt='D', ecolor='lavender', capsize=5, elinewidth=1, markeredgewidth=1)

### ------------ PFSS MODEL 
filepath = f"{ADAPT_DIR}/{adapt_mag[i]}"
print(f'Starting modeling for {enc} with file:', filepath)

### read in adapt magnetogram
adapt_magnetogram = pfss_funcs.adapt2pfsspy(filepath, return_magnetogram=True)
gong_map = sunpy.map.Map(adapt_magnetogram.data/1e5, adapt_magnetogram.meta)

### run PFSS model
pfss_model = pfss_funcs.adapt2pfsspy(filepath,rss=2.5)

### get Br at the source surface from the pfss model
pfss_br = pfss_model.source_surface_br

### get field lines
flines_psp = pfss_funcs.pfss2flines(pfss_model, skycoord_in=psp_projected)

### ------------- SOURCE SURFACE HEIGHT
RSS = [1.5, 2.0, 2.5, 3.0, 3.5]
pfss_lon, pfss_lat = [], []
ss_df = pd.DataFrame()
for i, rss in enumerate(RSS):
    # run PFSS model
    pfss_model = pfss_funcs.adapt2pfsspy(filepath,rss)

    # trace PFSS lines
    flines = pfss_funcs.pfss2flines(pfss_model)

    # get Br at the source surface from the pfss model
    pfss_br = pfss_model.source_surface_br

    # get HCS
    hcs = pfss_model.source_surface_pils[0]

    # get field lines
    psp_ss = psp_funcs.ballistically_project(psp_coords_carr,vr_arr=np.array(parker.vr)*u.km/u.s, r_inner=rss*u.Rsun)
    fpsp = pfss_funcs.pfss2flines(pfss_model, skycoord_in=psp_ss)

    # add column to dataframe
    use = np.where(fpsp.connectivities == 1)[0]
    df['lon'+str(rss)] = np.nans(len(df.Time))
    df['lat'+str(rss)] = np.nans(len(df.Time))
    ss_df['lon'+str(rss)] = fpsp.open_field_lines.solar_feet.lon.value
    ss_df['lat'+str(rss)] = fpsp.open_field_lines.solar_feet.lat.value
    # pfss_lon.append(fpsp.open_field_lines.solar_feet.lon.value)
    # pfss_lat.append(fpsp.open_field_lines.solar_feet.lat.value)

    # print 
    print('Finished with source surface height:', rss)

### ------------- INDUCE NOISE IN THE PROPAGATION RESULT
field_lines_ss, ss_new_coords = [], []
for i in np.arange(5):
    lon_noise = np.random.uniform(-5, 5, len(psp_at_source_surface.lat.value))
    lat_noise = np.random.uniform(-5, 5, len(psp_at_source_surface.lat.value))

    # create new coordinates
    new_coords = SkyCoord(lon=psp_at_source_surface.lon+lon_noise*u.deg, lat=psp_at_source_surface.lat+lat_noise*u.deg, 
                           radius=psp_at_source_surface.radius, representation_type="spherical",
                          frame = psp_at_source_surface.frame)
    ss_new_coords.append(new_coords)
    field_lines_ss.append(pfss_funcs.pfss2flines(pfss_model, skycoord_in=new_coords))
    
### ------------- INDUCE NOISE IN VELOCITY 
source_surface, field_lines, velocity = [], [], []
for i in np.arange(5):
    noise = np.random.uniform(-20, 20, len(parker.vr))
    vel = parker.vr + noise
    source_surface.append(psp_funcs.ballistically_project(psp_coords_carr,vr_arr=np.array(vel)*u.km/u.s, r_inner=2.5*u.Rsun))
    field_lines.append(pfss_funcs.pfss2flines(pfss_model, skycoord_in=source_surface[i]))
    velocity.append(vel)