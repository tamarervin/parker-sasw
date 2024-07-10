#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 20:00:49 2024

@author: tamarervin

Identifying source regions in AIA images
"""

### imports
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

from sunpy.coordinates import frames, get_horizons_coord

kernels = astrospice.registry.get_kernels('psp','predict') 

### ------------- PLOT STYLING
from matplotlib import rc

# Set the font family to Times New Roman for text and equations
rc('font', **{'family': 'serif', 'serif': ['Times New Roman']})
rc('mathtext', fontset='custom', rm='Times New Roman', it='Times New Roman:italic', bf='Times New Roman:bold')
mpl.rcParams['font.size'] = 18
panel_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)']

### ------------- PATHS
RES_DIR = os.path.realpath('results')
FIG_DIR = os.path.realpath('figures')
EPS_DIR = os.path.realpath('eps_figures')
ADAPT_DIR = os.path.realpath('adapt')

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
           '	adapt40311_03k012_202212140000_i00005600n1.fts.gz', #### E14
           'adapt40311_03k012_202303160000_i00053600n1.fts.gz', #### E15
           'adapt41311_03k012_202306210000_i00005600n1.fts.gz', #### E16
           'mrbqs230928t0004c2276_340.fits', #### E17
           ]

### ------------- DATAFRAMES
### perihelion dates
perihelion_dates = [ '2020-01-29 00:00:00', ### E4
                                        '2020-06-07 00:00:00', ### E5
                                        '2020-09-27 00:00:00', ### E6
                                        '2021-01-17 00:00:00', ### E7
                                        '2021-04-29 00:00:00', ### E8
                                        '2021-08-09 00:00:00', ### E9
                                        '2021-11-21 00:00:00', ### E10
                                        '2022-02-25 00:00:00', ### E11
                                        '2022-06-01 00:00:00', ### E12
                                        '2022-09-06 00:00:00', ### E13
                                        '2022-12-11 00:00:00', ### E14
                                        '2023-03-17 00:00:00', ### E15
                                        '2023-06-22 00:00:00', ### E16
                                        '2023-09-27 00:00:00' ### E17
                                        ]

### ------------- BALLSTIC PROPAGATION
def ballistically_project(skycoord,r_inner = 2.5*u.R_sun, vr_arr=None) :
    """
    Given a `SkyCoord` of a spacecraft trajectory in the Carrington frame,
    with `representation_type="spherical"`, and optionally an array of
    measured solar wind speeds at the same time intervals of the trajectory,
    return a SkyCoord for the trajectory ballistically projected down to 
    `r_inner` via a Parker spiral of the appropriate curvature. When `vr_arr`
    is not supplied, assumes wind speed is everywhere 360 km/s
    """
    if vr_arr is None : vr_arr = np.ones(len(skycoord))*360*u.km/u.s
    lons_shifted = skycoord.lon + delta_long(skycoord.radius,
                                              r_inner=r_inner,
                                              vsw=vr_arr
                                            )
    return SkyCoord(
        lon = lons_shifted, 
        lat = skycoord.lat,
        radius = r_inner * np.ones(len(skycoord)),
        representation_type="spherical",
        frame = skycoord.frame
    )

### ------------- BALLISTIC LONGITUDINAL SHIFT
@u.quantity_input
def delta_long(r:u.R_sun,
                r_inner=2.5*u.R_sun,
                vsw=360.*u.km/u.s,
                omega_sun=14.713*u.deg/u.d,
                ):
    """ 
    Ballistic longitudinal shift of a Parker spiral connecting two
    points at radius r and r_inner, for a solar wind speed vsw. Solar
    rotation rate is also tunable
    """
    return (omega_sun * (r - r_inner) / vsw).to("deg")

#     ### ------------   PFSS MODEL
#     ### find proper file
#     year = (pdate - timedelta(days=2)).year
#     month = (pdate - timedelta(days=2)).month
#     day = (pdate - timedelta(days=2)).day
    
#     ### download proper file
#     remote_path = f"ftp://gong.nso.edu//adapt/maps/gong/{year}"
#     filen = f"{year}{str(month).zfill(2)}{str(day).zfill(2)}0000"
#     filename =  input(f'Please input the filename associated with {filen}:' )
    
#     if not os.path.exists(f"{ADAPT_DIR}/{filename}") :
#         urllib.request.urlretrieve(f"{remote_path}/{filename}", 
#                                     f"{ADAPT_DIR}/{filename}"
#                                   )
        
#     ### run pfss model
#     flines_psp, topologies, hcs, gong_map = pfss(  f"{ADAPT_DIR}/{filename}", psp_projected)
    
#     ### add footpoints to df
#     use = np.where(flines_psp.connectivities == 1)[0]
#     df['flon'], df['flat'], df['pol_mod'], df['B0'],  df['I'], df['fss'] = [np.full(shape=len(df.lon), fill_value=None)] * 6
#     df['flon'][use] = flines_psp.open_field_lines.solar_feet.lon.value
#     df['flat'][use] = flines_psp.open_field_lines.solar_feet.lat
#     df['pol_mod'][use] = flines_psp.open_field_lines.polarities
#     df['fss']= flines_psp.expansion_factors 

#     ### ------------  FOOTPOINT BRIGHTNESS
    
#     ### ------------  FOOTPOINT FIELD STRENGTH
#     lats = np.array(flines_psp.open_field_lines.solar_feet.lat)
#     lons = np.array(flines_psp.open_field_lines.solar_feet.lon.value)
    
#     # Convert latitude and longitude arrays to SkyCoord
#     coords = SkyCoord(lon=lons*u.deg, lat=lats*u.deg, frame=gong_map.coordinate_frame)
    
#     # footpoint magnetic field
#     pixel_coords = coords.to_pixel(gong_map.wcs)
#     df['B0'] = np.full(shape=len(df.lon), fill_value=None)
#     df['B0'][use] = gong_map.data[pixel_coords[1].astype(int), pixel_coords[0].astype(int)]
        
#     ### ------------- MODELING FIGURES
#     ### AIA image as background
    
#     ### plot footpoints colored by expansion factor
    
#     ### ------------- POLARITY COMPARISON FIGURE
#     fig, [ax1, ax2, ax3] = plt.subplots(3, figsize=(30, 16), height_ratios=[3, 1, 1])

#     ### plot pfss
#     plot.plot_pfss(topologies, hcs, psp_projected, flines_psp, df.Time, ax1, df= df, nf=48)

#     ### plot modeling polarity
#     ax2.scatter(flines_psp.open_field_lines.source_surface_feet.lon.value, np.ones(len(flines_psp.open_field_lines.source_surface_feet.lon)), c=flines_psp.open_field_lines.polarities, cmap='RdBu', s=3)
#     ax2.set(xlabel='Source Surface Longitude', ylabel='Source Surface Latitude', xlim=(0, 140), ylim=(-.25, 1.25), yticks=(0, 1), yticklabels=['PFSS', 'PSP/FIELDS'])
#     ax2.set_ylabel('Polarity')

#     # plot measured polarity
#     polarity = np.sign(df.BrR2)
#     ax2.scatter(df.sslon, np.zeros(len(df.sslon)), c=polarity, cmap='RdBu', s=3)

#     # set title to percentage that matches
#     mm = len(np.where(flines_psp.open_field_lines.polarities == polarity[np.where(flines_psp.connectivities == 1)[0]])[0])
#     match = mm / len(flines_psp.open_field_lines.polarities) * 100
#     ax1.set_title('The percent of matching polarities is: ' + str(np.round(match, 4)))
     
#     # plot expansion factor
#     ax3.scatter(df.sslon[::100], df.fss[::100], c=df.vr[::100], cmap='RdPu', lw=0.5, s=30)
#     ax3.set_ylabel(r'$\rm f_{ss}$', fontsize=18)
    
#     # save fig 
#     plt.savefig(os.path.join(f'{FIG_DIR}/{year}{str(month).zfill(2)}{str(day).zfill(2)}_pfss.png'))
    
#     ### ------------ DELETE FILES
#     shutil.rmtree( f"{ADAPT_DIR}/{filename}")
    
#     ### ------------- DELETE VARIABLES
#     del flines_psp, topologies, hcs, gong_map
    
#     return df

### ------------- CREATE PFSS MODEL
def pfss(filepath, psp_at_source_surface, rss=2.5):
    ### read in adapt magnetogram
    adapt_magnetogram = pfss_funcs.adapt2pfsspy(filepath, return_magnetogram=True)
    gong_map = sunpy.map.Map(adapt_magnetogram.data/1e5, adapt_magnetogram.meta)

    ### run PFSS model
    pfss_model = pfss_funcs.adapt2pfsspy(filepath,rss)

    ### trace PFSS lines
    flines = pfss_funcs.pfss2flines(pfss_model)

    ### get Br at the source surface from the pfss model
    pfss_br = pfss_model.source_surface_br

    ### get HCS
    hcs = pfss_model.source_surface_pils[0]

    ### get field lines
    flines_psp = pfss_funcs.pfss2flines(pfss_model, skycoord_in=psp_at_source_surface)

    ### high res field lines
    flines_highres = pfss_funcs.pfss2flines(pfss_model,nth=181,nph=361)

    ### get field line topology defined by polarity
    topologies = flines_highres.polarities.reshape([181,361])

    return flines_psp, topologies, hcs, gong_map
### ------------------------------------------------------------------------------------------- ###
### ------------- SOURCE REGION IDENTIFICATION
### ------------------------------------------------------------------------------------------- ###
enc = 4
i = enc - 4
enc = f'e{i+4}'
pdate = perihelion_dates[i]
print('Encounter', enc)

### read in from file
df = pd.read_csv(os.path.join(RES_DIR, f"{enc}.csv"))

### turn date strings into datetime objects
min_avg = 20
df['Time'] = [datetime.datetime.strptime(d, '%Y-%m-%d %H:%M:%S.%f') for d in df.Time]
df.polarity= np.sign(df.Br)
df['sigmac'],df['sigmar'], df['vA'], df['Zp'],df['Zm'], df['deltav'], df['deltab'], df['rA'] = calcs.calc_sigma(df, num=int((min_avg*60)/7))

### ------------  CALCULATE TRAJECTORY
### create SkyCoord for PSP in the inertial (J2000) frame
psp_inertial = astrospice.generate_coords(
     'SOLAR PROBE PLUS', df.Time
 )
 
### transform to solar co-rotating frame 
psp_carrington = psp_inertial.transform_to(
    sunpy.coordinates.HeliographicCarrington(observer="self")
)

### velocity for ballistic propagation
ts_common = np.array([dt.timestamp() for dt in df.Time])
psp_vr_ts = [int(dt.timestamp()) for dt in df.Time]
psp_vr_common = interp1d(psp_vr_ts, df.vr, bounds_error=False)(ts_common)*u.km/u.s

### project onto source surface
psp_projected = calcs.ballistically_project(psp_carrington,vr_arr=psp_vr_common, r_inner=2.5*u.R_sun)

### ------------  CALCULATE WIND EMERGENCE TIME
vels = df.vr * u.km/u.s
rads = psp_carrington.radius.to(u.km).value

deltaT = np.array(rads / vels)
df['Tsun'] = [timestamp - timedelta(seconds=seconds) 
    if pd.notnull(seconds)  else np.nan for timestamp, seconds in zip(df.Time, deltaT)]   

# add to dataframe
df['lon'] = psp_carrington.lon.value
df['lat'] = psp_carrington.lat.value
df['rAU'] = psp_carrington.radius.to(u.AU).value
df['sslon'] = psp_projected.lon.value
df['sslat'] = psp_projected.lat.value
df['ssrAU'] = psp_projected.radius.to(u.AU).value
df['NpR2'] = df.Np * (df.rAU ** 2)
df['BrR2'] = df.Br * (df.rAU ** 2)

# categorize wind
df = categorize_wind(df)

# save to csv
df.to_csv(os.path.join(RES_DIR, f"{enc}.csv"))

### ------------ PFSS MODEL 
filename =  adapt_mag[i]
filepath = f"{ADAPT_DIR}/{filename}"
print(f'Starting modeling for {enc} at {pdate} with file:', filepath)

### read in adapt magnetogram
adapt_magnetogram = pfss_funcs.adapt2pfsspy(filepath, return_magnetogram=True)
gong_map = sunpy.map.Map(adapt_magnetogram.data/1e5, adapt_magnetogram.meta)

### run PFSS model
pfss_model = pfss_funcs.adapt2pfsspy(filepath,rss=2.5)

### get Br at the source surface from the pfss model
pfss_br = pfss_model.source_surface_br

### get field lines
flines_psp = pfss_funcs.pfss2flines(pfss_model, skycoord_in=psp_projected)

### ------------  COMPARISON
fig, axs = plt.subplots(2, height_ratios=[1, 4], figsize=(12, 8))

### MODEL POLARITY
ax = axs[0]
ax.scatter(flines_psp.open_field_lines.source_surface_feet.lon.value, np.ones(len(flines_psp.open_field_lines.source_surface_feet.lon)), c=flines_psp.open_field_lines.polarities, cmap='RdBu', s=3)
ax.set(xlabel='Source Surface Longitude', ylabel='Source Surface Latitude', xlim=(0, 360), ylim=(-.25, 1.25), yticks=(0, 1), yticklabels=['PFSS', 'PSP/FIELDS'])
ax.set_ylabel('Polarity')

### MEASURED POLARITY
polarity = np.sign(df.BrR2)
ax.scatter(df.sslon, np.zeros(len(df.sslon)), c=polarity, cmap='RdBu', s=3)

### SET TITLE
mm = len(np.where(flines_psp.open_field_lines.polarities == polarity[np.where(flines_psp.connectivities == 1)[0]])[0])
match = mm / len(flines_psp.open_field_lines.polarities) * 100
ax.set_title(f'The percent of matching polarities for {enc} is: ' + str(np.round(match, 4)))
    
### PLOT HCS
ax = axs[1]
for hcs in pfss_model.source_surface_pils:
    ax.scatter(hcs.lon, hcs.lat, c='k', s=1)
ax.scatter(df.sslon, df.sslat, c=polarity, cmap='RdBu', s=3)

### SAVE FIGURE
plt.savefig(os.path.join(FIG_DIR, f'{enc}_pfss_valid.png'))

### ------------  ADD FOOTPOINTS TO DF
use = np.where(flines_psp.connectivities == 1)[0]
df['flon'], df['flat'], df['pol_mod'], df['B0'],  df['I'], df['fss'] = [np.full(shape=len(df.lon), fill_value=None)] * 6
df['flon'][use] = flines_psp.open_field_lines.solar_feet.lon.value
df['flat'][use] = flines_psp.open_field_lines.solar_feet.lat
df['pol_mod'][use] = flines_psp.open_field_lines.polarities
df['fss']= flines_psp.expansion_factors 

### ------------  FOOTPOINT FIELD STRENGTH
lats = np.array(flines_psp.open_field_lines.solar_feet.lat)
lons = np.array(flines_psp.open_field_lines.solar_feet.lon.value)

# Convert latitude and longitude arrays to SkyCoord
coords = SkyCoord(lon=lons*u.deg, lat=lats*u.deg, frame=gong_map.coordinate_frame)

# footpoint magnetic field
pixel_coords = coords.to_pixel(gong_map.wcs)
df['B0'] = np.full(shape=len(df.lon), fill_value=None)
df['B0'][use] = gong_map.data[pixel_coords[1].astype(int), pixel_coords[0].astype(int)]

### ------------ SAVE TO CSV FILE
df.to_csv(os.path.join(RES_DIR, f"{enc}_modeling.csv"))

### ------------- CHECK SASW PERIOD
### check about 20 time range or something 
rolling_avg_vr = df['vr'].rolling(window=int((20*60)/7)).mean()
rolling_avg_sigma = df['sigmac'].rolling(window=int((20*60)/7)).mean()

# Apply conditions to categorize based on rolling averages
condition = (rolling_avg_vr <= 350) & (np.abs(rolling_avg_sigma) >= 0.8)
df['period'] = np.where(condition, 1, 0)

df_enc = df.loc[df['period'] == 1]

# plot to compare
sasw = np.where(df['period'] == 1)[0]
fig, axs = plt.subplots(2, figsize=(20, 8))
axs[0].scatter(df.Time, df.vr, c='lightpink', label='$v_R \; [km/s]$')
axs[0].scatter(df.Time[sasw], df.vr[sasw], c='lightblue', label='$v_R \; [km/s]$')
axs[0].set_ylabel('$v_R \; [km/s]$')
axs[1].scatter(df.Time, df.sigmac, c='lightpink', label='$\sigma_C$')
axs[1].scatter(df.Time[sasw], df.sigmac[sasw], c='lightblue', label='$\sigma_C$')
axs[1].set_ylabel('$\sigma_C$')

### ------------------------------------------------------------------------------------------- ###
### ------------- LOOKING AT SPECIFIC PERIODS
### ------------------------------------------------------------------------------------------- ###
enc = 6
i = enc - 4
enc = f'e{i+4}'
pdate = perihelion_dates[i]
print('Encounter', enc)
### read in from file
df = pd.read_csv(os.path.join(RES_DIR, f"{enc}_modeling.csv"))

df['Time'] = [datetime.datetime.strptime(d, '%Y-%m-%d %H:%M:%S.%f') for d in df.Time]
min_avg = 20
df.polarity= np.sign(df.Br)
df['sigmac'],df['sigmar'], df['vA'], df['Zp'],df['Zm'], df['deltav'], df['deltab'], df['rA'] = calcs.calc_sigma(df, num=int((min_avg*60)/7))


### turn date strings into datetime objects
df['Time'] = [datetime.datetime.strptime(d, '%Y-%m-%d %H:%M:%S.%f') for d in df.Time]
rolling_avg_vr = df['vr'].rolling(window=int((20*60)/7), min_periods=1).mean()
rolling_avg_sigma = df['sigmac'].rolling(window=int((20*60)/7), min_periods=1).mean()
plt.scatter(df.Time, df.sigmac, c='lightpink')
plt.plot(df.Time, rolling_avg_sigma, c='k')

### ------------  FOOTPOINT BRIGHTNESS
### ------------- SASW PERIODS
reg_sasw = [ ['2020-01-28 18:00:00', '2020-01-29 01:00:00', 4], ### E4
            ['2020-06-06 06:00:00', '2020-06-06 09:00:00', 5], ### E5 R1
                ['2020-06-09 10:00:00', '2020-06-09 22:00:00', 5], ### E5 R2
                ['2020-09-27 12:00:00', '2020-09-27 16:00:00', 6], ### E6 R1
                ['2020-09-29 02:00:00', '2020-09-29 05:00:00', 6], ### E6 R2
                ['2021-01-18 20:00:00', '2021-01-19 06:00:00', 7], ### E7
                ['2021-04-30 00:00:00', '2021-04-30 05:00:00', 8], ### E8 R1
                ['2021-04-30 12:00:00', '2021-04-30 19:00:00', 8], ### E8 R2
                ['2022-12-10 18:00:00', '2022-12-10 19:00:00', 14], ### E14 R1
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


# ## SSW
# reg = [ ['2020-01-30 06:00:00', '2020-01-30 12:00:00', 4 ], ### E4
#         ['2020-06-10 12:00:00', '2020-06-10 20:00:00', 5], ### E5
#         ['2020-09-26 00:00:00', '2020-09-27 00:00:00', 6], ### E6 -- to do // questionable
#         ['2021-01-16 06:00:00', '2021-01-16 10:00:00', 7], ### E7
#         ['2021-05-02 04:30:00', '2021-05-02 06:00:00', 8], ### E8
#         ['2021-08-07 23:00:00', '2021-08-08 05:00:00', 9], ### E9
#         ]

# dates = ['2020-01-30', '2020-06-10', '2020-09-26', '2021-01-16', '2021-05-02',
#           '2021-08-07']

j=3
reg, dates = reg_fsw, dates_fsw
df_sasw = df[(df.Time >= datetime.datetime.strptime(reg[j][0], '%Y-%m-%d %H:%M:%S')) & (df.Time <= datetime.datetime.strptime(reg[j][1], '%Y-%m-%d %H:%M:%S'))]
rolling_avg_vr = df_sasw['vr'].rolling(window=int((20*60)/7), min_periods=1).mean()
rolling_avg_sigma = df_sasw['sigmac'].rolling(window=int((20*60)/7), min_periods=1).mean()
plt.scatter(df_sasw.Time, df_sasw.sigmac, c='lightpink')


# plt.plot(df_sasw.Time, rolling_avg_sigma, c='k')
fig , axs = plt.subplots(3, figsize=(20, 6), sharex='all')
for ax, var, label in zip(axs, [df_sasw.BrR2, df_sasw.vr, df_sasw.sigmac], 
                          ['$B_R R^2$', '$v_R$', '$\sigma_C$']):
    ax.scatter(df_sasw.Time, var, facecolor='None', edgecolor='lightpink', marker='D')
    ax.set_ylabel(ylabel=label, fontsize=18)


# =============================================================================
# ### ------------ PFSS MODEL 
# =============================================================================
ADAPT_DIR = '/Users/tamarervin/development/sasw_sources/adapt'
filename =  adapt_mag[i]
filepath = f"{ADAPT_DIR}/{filename}"
print(f'Starting modeling for {enc} at {pdate} with file:', filepath)

### create SkyCoord for PSP in the inertial (J2000) frame
psp_inertial = astrospice.generate_coords(
     'SOLAR PROBE PLUS', df_sasw.Time
 )
 
### transform to solar co-rotating frame 
psp_carrington = psp_inertial.transform_to(
    sunpy.coordinates.HeliographicCarrington(observer="self")
)

### velocity for ballistic propagation
ts_common = np.array([dt.timestamp() for dt in df_sasw.Time])
psp_vr_ts = [int(dt.timestamp()) for dt in df_sasw.Time]
psp_vr_common = interp1d(psp_vr_ts, df_sasw.vr, bounds_error=False)(ts_common)*u.km/u.s

### project onto source surface
psp_projected = calcs.ballistically_project(psp_carrington,vr_arr=psp_vr_common, r_inner=2.5*u.R_sun)


### read in adapt magnetogram
adapt_magnetogram = pfss_funcs.adapt2pfsspy(filepath, return_magnetogram=True)
gong_map = sunpy.map.Map(adapt_magnetogram.data/1e5, adapt_magnetogram.meta)

### run PFSS model
pfss_model = pfss_funcs.adapt2pfsspy(filepath,rss=2.5)

### get field lines
flines_psp = pfss_funcs.pfss2flines(pfss_model, skycoord_in=psp_projected)
coords = flines_psp.open_field_lines.solar_feet #flines_psp.open_field_lines.solar_feet.observer)
left, bot = 225, -45


# =============================================================================
# ### ------------ GET AIA IMAGE AND PLOT
# =============================================================================
cadence = a.Sample(24*u.hour)  # querying cadence
start_date ='2020-09-13 00:00:00' # start date of query
end_date = '2020-09-17 00:00:00'# end date of query
print(start_date, end_date)

# query data
aia_result = Fido.search(a.Time(start_date, end_date),
                     a.Instrument.aia, a.Wavelength(193 * u.angstrom), cadence)
file_download = Fido.fetch(aia_result)
map_seq = sunpy.map.Map(sorted(file_download))

### ------------ CREATE FIGURE
plt.plot(df_sasw.flon, df_sasw.flat)
nf=4
left, bot = 115, 5
for i, mapp in enumerate(map_seq):
    aia_map = mapp
    
    ## carrington coordinates
    shape = (720, 1440)
    carr_header = make_heliographic_header(aia_map.date, aia_map.observer_coordinate, shape, frame='carrington')
    outmap = aia_map.reproject_to(carr_header)
    submap = outmap
    
    ## FIGURE
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(projection=submap)
    
    ## AIA MAP
    lognorm = mpl.colors.LogNorm(vmin=np.nanpercentile(submap.data.flatten(),40), 
                            vmax=np.nanpercentile(submap.data.flatten(),99.9))
    submap.plot(axes=ax, norm=lognorm)
    
    ### PLOT FOOTPOINTS
    pfss_coord = SkyCoord(lon=np.array(df_sasw.flon)*u.deg, lat=np.array(df_sasw.flat)*u.deg,
                        frame=submap.coordinate_frame) 
    # pfss_coord = flines_psp.open_field_lines.solar_feet
    # pfss_projected = pfss_coord.transform_to(submap.coordinate_frame)
    pixel_coords_x, pixel_coords_y = pfss_coord.to_pixel(outmap.wcs)
    ax.scatter(pixel_coords_x[::nf],pixel_coords_y[::nf], 
                    marker='D', zorder=3, s=60,
                    facecolor='none',  edgecolor='lightpink', linewidth=1)

    ### LABELS
    ax.tick_params(axis='both', which='major', labelsize=16) 
    ax.set_xlabel('Carrington Longitude [deg]', fontsize=18)
    ax.set_ylabel('Carrington Latitude [deg]', fontsize=18)

## TITLE                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                b
ax.set_title('SASW: ' + str(reg[j][0]) + ' to ' + str(reg[j][1]), fontsize=18)

### TESTING
plt.scatter(flines_psp.open_field_lines.solar_feet.lon, flines_psp.open_field_lines.solar_feet.lat)
plt.scatter(df.flon, df.flat)

### ------------ ADD FOOTPOINT BRIGHTNESS
crit = (df.Time >= datetime.datetime.strptime(reg[j][0], '%Y-%m-%d %H:%M:%S')) & (df.Time <= datetime.datetime.strptime(reg[j][1], '%Y-%m-%d %H:%M:%S'))
df['I'][crit] = sunpy.map.sample_at_coords(aia_map, coords) 

### ------------ SAVE TO CSV FILE
df.to_csv(os.path.join(RES_DIR, f"{enc}_modeling.csv"))

### ------------ DELETE DF
del df
    
## ------------ COMPLETED
print(f'Modeling complete for encounter {i}')


# =============================================================================
# ### ------ OVERVIEW + AIA PLOT
# =============================================================================
j=4
enc = 14
i = enc - 4
enc = f'e{i+4}'
pdate = perihelion_dates[i]
print('Encounter', enc)

### read in from file
df = pd.read_csv(os.path.join(RES_DIR, f"{enc}_modeling.csv"))
min_avg = 20
df.polarity= np.sign(df.Br)
df['Time'] = [datetime.datetime.strptime(d, '%Y-%m-%d %H:%M:%S.%f') for d in df.Time]
df['sigmac'],df['sigmar'], df['vA'], df['Zp'],df['Zm'], df['deltav'], df['deltab'], df['rA'] = calcs.calc_sigma(df, num=int((min_avg*60)/7))


### turn date strings into datetime objects
df_sasw = df[(df.Time >= datetime.datetime.strptime(reg[j][0], '%Y-%m-%d %H:%M:%S')) & (df.Time <= datetime.datetime.strptime(reg[j][1], '%Y-%m-%d %H:%M:%S'))]
df_sasw

### ------------ GET AIA IMAGE AND PLOT
cadence = a.Sample(24*u.hour)  # querying cadence
start_date ='2022-12-28 00:00:00' # start date of query
end_date = '2022-12-29 00:00:00'# end date of query
print(start_date, end_date)

# query data
aia_result = Fido.search(a.Time(start_date, end_date),
                     a.Instrument.aia, a.Wavelength(193 * u.angstrom), cadence)
file_download = Fido.fetch(aia_result)
map_seq = sunpy.map.Map(sorted(file_download))

# =============================================================================
# ### ------------ CREATE TWO PANEL FIGURE
# =============================================================================
def plot_data(ax, vv, ylabel, ylims, plabel, color='k'):
    if vv == 'sigmac':
        ax.plot(df_sasw.Time, np.abs(df_sasw[vv]), c=color, lw=2)
    else:
        ax.plot(df_sasw.Time, df_sasw[vv], c=color, lw=2)
    ax.set_ylabel(ylabel, fontsize=18, color=color)
    ax.grid(True, linestyle='--', linewidth=0.5)
    ### tick params
    ax.tick_params(axis='both', which='major', labelsize=16)
    ### ylimits
    ax.set(ylim=(ylims[0] - (ylims[2]/2), ylims[1] + (ylims[2]/2)), ### ylimits
               yticks=np.arange(ylims[0], ylims[1]+0.1, step=ylims[2]) ### yticks
               )
    ### panel label
    ax.text(0.95, 0.95, plabel, transform=ax.transAxes, fontsize=20, fontweight='bold', va='top', ha='left', zorder=10)

    
### LEFT PANEL: DATA OVERVIEW
labels = [r'$\rm v_R \; [km s^{-1}]$',  ### velocity
                  r'$\rm| \sigma_C|$', ### cross helicity
                  r'$\rm B_r \; R^2 \; [nT]$',  ### magnetic field
                  r'$\rm N_p \; R^2 \; [cm^{-3}]$',  ### proton density
                  r'$\rm T_p \; [eV]$',  ### proton temperature
                  r'$\rm \beta_p$'  ### plasma beta
          ]

 ### axes limits
ylimits = [[0, 800, 200],  ### velocity
                  [0, 1, 0.25], ### cross helicity
                  [-5, 5, 2.5],  ### magnetic field
                  [0, 40, 10],  ### proton density
                  [0, 200, 50],  ### proton temperature
                  [0, 10, 2.5]  ### plasma beta
         ]

### FIGURE SETUP
left, bot = 45, -5
fig = plt.figure(figsize=(40, 12))
gs = plt.GridSpec(4, 2, width_ratios=[2, 1], hspace=0, wspace=0.05)

### plot data
for i, vv in enumerate(['vr', 'sigmac', 'BrR2', 'NpR2']):
    ax = plt.subplot(gs[i, 0])
    plot_data(ax, vv, ylabel=labels[i], ylims=ylimits[i], plabel=panel_labels[i])

### radial distance
df_sasw['radius'] = (np.array(df_sasw.rAU)*u.AU).to(u.Rsun).value
ax  = plt.subplot(gs[2, 0])
ax.axhline(0, linestyle='dotted', color='k')

### RIGHT PANEL: AIA MAP
aia_map = map_seq
shape = (720, 1440)
carr_header = make_heliographic_header(aia_map.date, aia_map.observer_coordinate, shape, frame='carrington')
outmap = aia_map.reproject_to(carr_header)
bottom_left = SkyCoord(left*u.degree, bot*u.degree, frame=outmap.coordinate_frame)
submap = outmap.submap(bottom_left, width=50*u.degree, height=50*u.degree)
ax = plt.subplot(gs[:, 1], projection=submap)
## AIA MAP
lognorm = mpl.colors.LogNorm(vmin=np.nanpercentile(submap.data.flatten(),30), 
                        vmax=np.nanpercentile(submap.data.flatten(),99.9))
submap.plot(axes=ax, norm=lognorm)

### PLOT FOOTPOINTS
coords = SkyCoord(lon=np.array(df_sasw.flon)*u.deg, lat=np.array(df_sasw.flat)*u.deg,
                    frame=submap.coordinate_frame)
coords = coords.transform_to(submap.coordinate_frame)
pixel_coords_x, pixel_coords_y = coords.to_pixel(submap.wcs)
ax.scatter(pixel_coords_x[::nf],pixel_coords_y[::nf], 
                marker='D', zorder=3, s=60,
                facecolor='none',  edgecolor='lightpink', linewidth=1)

### LABELS
ax.tick_params(axis='both', which='major', labelsize=16) 
ax.set_xlabel('Carrington Longitude [deg]', fontsize=18)
ax.set_ylabel('Carrington Latitude [deg]', fontsize=18)
ax.set_title('')

### SAVE FIGURE
date = dates[j]
plt.savefig(os.path.join(FIG_DIR, f'{date}_FSW_AIA_Data.png'), bbox_inches='tight')
plt.savefig(os.path.join(EPS_DIR, f'{date}_FSW_AIA_Data.eps'), bbox_inches='tight')


### ------------ ADD FOOTPOINT BRIGHTNESS
crit = (df.Time >= datetime.datetime.strptime(reg[j][0], '%Y-%m-%d %H:%M:%S')) & (df.Time <= datetime.datetime.strptime(reg[j][1], '%Y-%m-%d %H:%M:%S'))
df['I'][crit] = sunpy.map.sample_at_coords(aia_map, coords) 

### ------------ SAVE TO CSV FILE
df.to_csv(os.path.join(RES_DIR, f"{enc}_modeling.csv"))
df_sasw['I'] = sunpy.map.sample_at_coords(aia_map, coords) 
df_sasw['I0'] = sunpy.map.sample_at_coords(aia_map, coords) / np.nanmax(aia_map.data)
df_sasw.to_csv(os.path.realpath(f'fsw_results/{date}.csv'))

### AVERAGES
column_averages = df_sasw.mean()
print(column_averages)

plt.hist(df.B0, density=True)
plt.hist(df_sasw.B0, color='red', density=True)

plt.hist(df.vr, density=True)
plt.hist(df_sasw.vr, color='red', density=True)

plt.hist(df.sigmac, density=True)
plt.hist(df_sasw.sigmac, color='red', density=True)

# =============================================================================
# ### ----- BRIGHTNESS COMPARISON WITH 1D HISTOGRAMS
# =============================================================================
### ------------ READ IN SASW FILES
sasw_files = glob.glob(os.path.join('sasw_sources/sasw_results', '*'))
sasw = pd.DataFrame()
for i, file in enumerate(sasw_files):
    ff = pd.read_csv(file)
    sasw = sasw.append(ff)

### ------------ GET A BUNCH OF AIA IMAGES
cadence = a.Sample(24*u.hour)  # querying cadence
start_date ='2023-06-15 00:00:00' # start date of query
end_date = '2023-06-16 00:00:00'# end date of query
print(start_date, end_date)

# query data
aia_result = Fido.search(a.Time(start_date, end_date),
                     a.Instrument.aia, a.Wavelength(193 * u.angstrom), cadence)
file_download = Fido.fetch(aia_result)
map_seq = sunpy.map.Map(sorted(file_download))


df = pd.DataFrame({})
for aia in map_seq:
    # create submap
    bottom_left = SkyCoord(-700 * u.arcsec, -700 * u.arcsec, frame=aia.coordinate_frame)
    top_right = SkyCoord(700 * u.arcsec, 700 * u.arcsec, frame=aia.coordinate_frame)
    aia_smap = aia.submap(bottom_left, top_right=top_right)
    # create histogram
    num_bins = 50
    bins = np.linspace(0, 10000, num_bins)
    hist, bin_edges = np.histogram(aia_smap.data, bins=bins)
    df[aia_smap.meta['date-obs']] = hist
# df['bins'] = bins[:-1]
df.to_csv(f'sasw_sources/aia_results/{start_date[:10]}.csv')

### TESTING CH STUFF
shape = (720, 1440)
carr_header = make_heliographic_header(aia_map.date, aia_map.observer_coordinate, shape, frame='carrington')
outmap = aia_map.reproject_to(carr_header)
lons = np.linspace(0, 361, 1441)
lats = np.linspace(-180, 181, 721)
fig= plt.figure()
ax = fig.add_subplot(projection=outmap)
lognorm = mpl.colors.LogNorm(vmin=np.nanpercentile(outmap.data.flatten(),30), 
                        vmax=np.nanpercentile(outmap.data.flatten(),99.9))
ax.pcolormesh(lons, lats, outmap.data,cmap='sdoaia193')
lons = np.linspace(0, 361, 1440)
lats = np.linspace(-180, 181, 720)
test=ax.contour(lons,lats,outmap.data,[np.nanpercentile(outmap.data.flatten(),f)  for  f in [4, 5]])

###  ---------- COMPARISON HISTOGRAM
### get aia histogram
aia_hist = []
for i in np.arange(0, 49):
    aia_hist.append(np.nansum(df.iloc[i], axis=0))
H_aia = np.insert(aia_hist, 0, aia_hist[0]) # / np.nanmax(aia_hist)

### get sasw histogram
num_bins = 50
bins = np.linspace(0, 10000, num_bins)
sasw_hist, bin_edges = np.histogram(sasw['I'], bins=bins)
H_sasw = np.insert(sasw_hist, 0, sasw_hist[0]) # / np.nanmax(sasw_hist)

fig, ax = plt.subplots(figsize=(12, 8))
ax.loglog(bins,  H_aia, '-', drawstyle='steps', c='k', lw=3)
ax.loglog(bins,  H_sasw, '-', drawstyle='steps', c='tab:purple', lw=3)
ax.set_xlabel(r'$\rm Intensity$', fontsize=18)
ax.tick_params(axis='both', which='major', labelsize=16)
ax.set_yscale('log')
ax.set(ylim=(1e0, 1e8))
locmaj = mpl.ticker.LogLocator(base=10,numticks=2) 
ax.xaxis.set_major_locator(locmaj)
locmin = mpl.ticker.LogLocator(base=10.0,subs=(0.2,0.4,0.6,0.8),numticks=10)
ax.xaxis.set_minor_locator(locmin)
ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
locmaj = mpl.ticker.LogLocator(base=10,numticks=10) 
ax.yaxis.set_major_locator(locmaj)
locmin = mpl.ticker.LogLocator(base=10.0,subs=(0.2,0.4,0.6,0.8),numticks=10)
ax.yaxis.set_minor_locator(locmin)
ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
ax.set_title(f'{start_date}', fontsize=22)

# =============================================================================
# ### FULL FIGURE
# =============================================================================
num_bins = 50
bins = np.linspace(0, 10000, num_bins)

### ------------ READ IN AR-SASW FILES
ar_files = glob.glob(os.path.join('sasw_active', '*'))
ar = pd.DataFrame()
for i, file in enumerate(ar_files):
    ff = pd.read_csv(file)
    ar = ar.append(ff)
### get sasw histogram
ar_hist, bin_edges = np.histogram(ar['I'], bins=bins)
H_ar = np.insert(ar_hist, 0, ar_hist[0])  / np.nanmax(ar_hist)

### ------------ READ IN CH-SASW FILES
ch_files =  glob.glob(os.path.join('sasw_results', '*'))
ch = pd.DataFrame()
for i, file in enumerate(ch_files):
    ff = pd.read_csv(file)
    ch = ch.append(ff)
### get sasw histogram
ch_hist, bin_edges = np.histogram(ch['I'], bins=bins)
H_ch = np.insert(ch_hist, 0, ch_hist[0])  / np.nanmax(ch_hist)

### ------------ READ IN SASW FILES
sasw_files = glob.glob(os.path.join('sasw_results', '*'))
sasw = ch.append(ar)
### get sasw histogram
sasw_hist, bin_edges = np.histogram(sasw['I'], bins=bins)
H_sasw = np.insert(sasw_hist, 0, sasw_hist[0])  / np.nanmax(sasw_hist)


## ------------ READ IN FSW FILES
fsw_files = glob.glob(os.path.join('fsw_results', '*'))
fsw = pd.DataFrame()
for i, file in enumerate(fsw_files):
    ff = pd.read_csv(file)
    fsw = fsw.append(ff)
### get fsw histogram
fsw_hist, bin_edges = np.histogram(fsw['I'], bins=bins)
H_fsw = np.insert(fsw_hist, 0, fsw_hist[0])  / np.nanmax(fsw_hist)

### ------------ READ IN AIA FILES
aia_files = glob.glob(os.path.join('aia_results', '*'))
aia = pd.DataFrame()
for i, file in enumerate(aia_files):
    ff = pd.read_csv(file)
    aia = aia.append(ff)
    
### get aia histogram
aia_hist = []
for i in np.arange(0, 49):
    aia_hist.append(np.nansum(aia.iloc[i], axis=0))
H_aia = np.insert(aia_hist, 0, aia_hist[0]) / np.nanmax(aia_hist)


### get quartiles
def boxplot(data, col, lcol, ax, position, outliers=False):
    data = data[~np.isnan(data)]
    quartiles = np.percentile(data, [25, 50, 75])
    q1, median, q3 = quartiles
    mean = np.mean(data)
    ax.boxplot(data, vert=False, patch_artist=True, boxprops=dict(facecolor=lcol, color=col),
               medianprops=dict(color=col), showfliers=outliers, positions=[position])
    ax.axvline(mean, color=col, linestyle='dashed', linewidth=1)
    return q1, median, q3


### ------------ FIGURE
fig = plt.figure(figsize=(16, 16))

# Create a GridSpec with 5 rows and 1 column
# gs = mpl.gridspec.GridSpec(4, 1, figure=fig) #, hspace=0.35)

### BOX PLOT
# ax = fig.add_subplot(gs[0, 0])
# ax.text(0.95, 0.95, panel_labels[0], transform=ax.transAxes, fontsize=20, fontweight='bold', va='top', ha='left')
# boxplot(fsw['I'], 'tab:red', 'lightpink', ax, 4)
# boxplot(sasw['I'], 'tab:purple', 'lavender', ax, 3)
# boxplot(ch['I'], 'tab:brown', 'sandybrown', ax, 2)
# boxplot(ar['I'], 'tab:orange', 'navajowhite', ax, 1)
# ax.set(yticks=[1, 2, 3, 4], yticklabels=['non-CH', 'CH-like', 'SASW', 'FSW'])

### HISTOGRAMS
# ax = fig.add_subplot(gs[1:, 0])
# ax.text(0.95, 0.98, panel_labels[1], transform=ax.transAxes, fontsize=20, fontweight='bold', va='top', ha='left', zorder=10)

fig, ax= plt.subplots(1, figsize=(16, 10))
ax.loglog(bins,  H_aia, '-', drawstyle='steps', c='k', lw=3)
ax.loglog(bins,  H_fsw, '-', drawstyle='steps', c='tab:red', lw=3)
# ax.loglog(bins,  H_ar, '-', drawstyle='steps', c='tab:orange', lw=3)
# ax.loglog(bins,  H_ch, '-', drawstyle='steps', c='tab:brown', lw=3)
ax.loglog(bins,  H_sasw, '-', drawstyle='steps', c='tab:purple', lw=3)
ax.set_xlabel(r'$\rm AIA \; 193{\AA} \; Intensity$', fontsize=20)
ax.set_ylabel(r'$\rm Normalized \; Frequency$', fontsize=20)
ax.tick_params(axis='both', which='both', labelsize=18)
ax.set_yscale('log')
locmaj = mpl.ticker.LogLocator(base=10,numticks=10) 
ax.xaxis.set_major_locator(locmaj)
locmin = mpl.ticker.LogLocator(base=10.0,subs=(0.2,0.4,0.6,0.8),numticks=10)
ax.xaxis.set_minor_locator(locmin)
ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
locmaj = mpl.ticker.LogLocator(base=10,numticks=10) 
ax.yaxis.set_major_locator(locmaj)
locmin = mpl.ticker.LogLocator(base=10.0,subs=(0.2,0.4,0.6,0.8),numticks=10)
ax.yaxis.set_minor_locator(locmin)
ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())

### legend
aia_patch = mpatches.Patch(edgecolor='k', facecolor='grey', label=r'$\rm AIA$')
fsw_patch = mpatches.Patch(edgecolor='tab:red', facecolor='lightpink', label=r'$\rm FSW$')
sasw_patch = mpatches.Patch(edgecolor='tab:purple', facecolor='lavender', label=r'$\rm SASW$')
ch_patch = mpatches.Patch(edgecolor='tab:brown', facecolor='sandybrown', label=r'$\rm SASW: CH-like$')
ar_patch = mpatches.Patch(edgecolor='tab:orange', facecolor='navajowhite', label=r'$\rm SASW: non-CH$')
leg0 = ax.legend(handles=[aia_patch, fsw_patch, sasw_patch], loc='upper right', fontsize=20) #, bbox_to_anchor=(1, 0.95))
ax.add_artist(leg0)

plt.savefig(os.path.join(FIG_DIR, 'intensity_hist.png'), bbox_inches='tight')
plt.savefig(os.path.join(EPS_DIR, 'intensity_hist.eps'), bbox_inches='tight')


### COMPARING EXPANSION AND B0
### ------------ FIGURE
fig, axs = plt.subplots(2, figsize=(20, 8), gridspec_kw={'hspace':0.35})
out = False

### B0
ax = axs[0]
boxplot(np.abs(fsw['B0']*1e5), 'tab:red', 'lightpink', ax, 4, outliers=out)
boxplot(np.abs(sasw['B0']*1e5), 'tab:purple', 'lavender', ax, 3, outliers=out)
boxplot(np.abs(ch['B0']*1e5), 'tab:brown', 'sandybrown', ax, 2, outliers=out)
boxplot(np.abs(ar['B0']*1e5), 'tab:orange', 'navajowhite', ax, 1, outliers=out)
ax.set(yticks=[1, 2, 3, 4], yticklabels=['non-CH', 'CH-like', 'SASW', 'FSW'])
ax.axvline(30, c='k', zorder=-1, linestyle='dashed', lw=1)
ax.set_xlabel(r'$\rm |B_0| \; [G]$', fontsize=20)
ax.text(0.97, 0.95, panel_labels[0], transform=ax.transAxes, fontsize=20, fontweight='bold', va='top', ha='left', zorder=10)


# ### intensity
# ax = axs[1]
# boxplot(fsw['I'], 'tab:red', 'lightpink', ax, 4)
# boxplot(sasw['I'], 'tab:purple', 'lavender', ax, 3)
# boxplot(ch['I'], 'tab:brown', 'sandybrown', ax, 2)
# boxplot(ar['I'], 'tab:orange', 'navajowhite', ax, 1)
# ax.set(yticks=[1, 2, 3, 4], yticklabels=['non-CH', 'CH-like', 'SASW', 'FSW'])
# ax.set_xlabel(r'$\rm  AIA \; 193{\AA} \; Intensity$', fontsize=20)
# ax.text(0.97, 0.95, panel_labels[1], transform=ax.transAxes, fontsize=20, fontweight='bold', va='top', ha='left', zorder=10)


### expansion
ax = axs[1]
boxplot(fsw['fss'], 'tab:red', 'lightpink', ax, 4,  outliers=out)
boxplot(sasw['fss'], 'tab:purple', 'lavender', ax, 3, outliers=out)
boxplot(ch['fss'], 'tab:brown', 'sandybrown', ax, 2, outliers=out)
boxplot(ar['fss'], 'tab:orange', 'navajowhite', ax, 1, outliers=out)
ax.set(yticks=[1, 2, 3, 4], yticklabels=['non-CH', 'CH-like', 'SASW', 'FSW'])
ax.set_xlabel(r'$\rm f_{ss}$', fontsize=20)
ax.set_xscale('log')
ax.text(0.97, 0.95, panel_labels[1], transform=ax.transAxes, fontsize=20, fontweight='bold', va='top', ha='left', zorder=10)

# ### density
# ax = axs[2]
# boxplot(fsw['NpR2'], 'tab:red', 'lightpink', ax, 4,  outliers=out)
# boxplot(sasw['NpR2'], 'tab:purple', 'lavender', ax, 3, outliers=out)
# boxplot(ch['NpR2'], 'tab:brown', 'sandybrown', ax, 2, outliers=out)
# boxplot(ar['NpR2'], 'tab:orange', 'navajowhite', ax, 1, outliers=out)
# ax.set(yticks=[1, 2, 3, 4], yticklabels=['non-CH', 'CH-like', 'SASW', 'FSW'])
# ax.set_xlabel(r'$\rm N_p R^2 \; [cm^{-3}]$', fontsize=20)

if out:
    plt.savefig(os.path.join(FIG_DIR, 'modeling_boxplot_outliers.png'), bbox_inches='tight')
    plt.savefig(os.path.join(EPS_DIR, 'modeling_boxplot_outliers.eps'), bbox_inches='tight')
else:
    plt.savefig(os.path.join(FIG_DIR, 'footpoint_expansion.png'), bbox_inches='tight')
    plt.savefig(os.path.join(EPS_DIR, 'footpoint_expansion.eps'), bbox_inches='tight')

###  ------------- HISTOGRAM OF EXPANSION vs. SPEED
### FIGURE
fig, axs = plt.subplots(1, 2, figsize=(16, 6), sharey='all', gridspec_kw={'hspace': 0.2, 'wspace': 0.2})

ax = axs[0]
ax.hist2d(sasw.vr, sasw.fss, cmap='Purples', cmin=10, vmin=40, bins=(20,20), range=[[0, 500], [0, 300]])
ax.hist2d(fsw.vr, fsw.fss, cmap='Reds', cmin=10, vmin=40, bins=(20,20), range=[[0, 800], [0, 300]])
ax.tick_params(axis='both', which='major', labelsize=18) 
ax.set_ylabel(r'$\rm f_{ss}$', fontsize=20)
ax.set_xlabel(r'$\rm v_{sw} \; [km/s]$', fontsize=20)

ax  = axs[1]
ax.hist2d(ar.vr, ar.fss, cmap='Oranges', cmin=10, vmin=40, bins=(20,20), range=[[0, 500], [0, 300]])
ax.hist2d(ch.vr, ch.fss, cmap='copper_r', cmin=10, vmin=40, bins=(20,20), range=[[0, 500], [0, 300]])
ax.set_xlabel(r'$\rm v_{sw} \; [km/s]$', fontsize=20)

### save figure
plt.savefig(os.path.join(FIG_DIR, 'expansion_factor.png'), bbox_inches='tight')
plt.savefig(os.path.join(EPS_DIR, 'expansion_factor.eps'), bbox_inches='tight')



# =============================================================================
# ### SLOW AND FAST STREAMS 
# =============================================================================

j = 0
## SSW
reg = [ ['2020-01-30 06:00:00', '2020-01-30 12:00:00', 4 ], ### E4
        ['2020-06-10 12:00:00', '2020-06-10 20:00:00', 5], ### E5
        ['2020-09-26 00:00:00', '2020-09-27 00:00:00', 6], ### E6 -- to do // questionable
        ['2021-01-16 06:00:00', '2021-01-16 10:00:00', 7], ### E7
        ['2021-05-02 04:30:00', '2021-05-02 06:00:00', 8], ### E8
        ['2021-08-07 23:00:00', '2021-08-08 05:00:00', 9], ### E9
        ]

dates = ['2020-01-30', '2020-06-10', '2020-09-26', '2021-01-16', '2021-05-02',
          '2021-08-07']
### FSW
j=5
reg = [ 
        ['2020-01-27 04:55:00', '2020-01-27 05:15:00', 4], ### E4
       # ['2020-06-06 00:00:00', '2020-06-06 09:00:00', 5], ### E5
        ['2021-04-27 01:00:00', '2021-04-27 04:00:00', 8], ### E8
        ['2021-04-27 06:00:00', '2021-04-27 08:30:00', 8], ### E8
        ['2021-11-20 05:30:00', '2021-11-20 10:30:00', 10], ### E10
        ['2022-05-29 21:30:00', '2022-05-30 23:30:00', 12], ### E12
        ['2022-12-15 21:15:00', '2022-12-15 22:15:00', 14], ### E14
        ]

dates = [
'2020-01-27',
  '2021-04-27',  '2021-04-27_2', '2021-11-20', '2022-05-29', '2022-12-14'
          ]

### read in from file
enc = f'e{reg[j][2]}'
df = pd.read_csv(os.path.join(RES_DIR, f"{enc}_modeling.csv"))

### turn date strings into datetime objects
df['Time'] = [datetime.datetime.strptime(d, '%Y-%m-%d %H:%M:%S.%f') for d in df.Time]

df_sasw = df[(df.Time >= datetime.datetime.strptime(reg[j][0], '%Y-%m-%d %H:%M:%S')) & (df.Time <= datetime.datetime.strptime(reg[j][1], '%Y-%m-%d %H:%M:%S'))]



df_sasw = df[(df.Time >= datetime.datetime.strptime(reg[j][0], '%Y-%m-%d %H:%M:%S')) & (df.Time <= datetime.datetime.strptime(reg[j][1], '%Y-%m-%d %H:%M:%S'))]
rolling_avg_vr = df_sasw['vr'].rolling(window=int((20*60)/7), min_periods=1).mean()
rolling_avg_sigma = df_sasw['sigmac'].rolling(window=int((20*60)/7), min_periods=1).mean()
# plt.scatter(df_sasw.Time, df_sasw.sigmac, c='lightpink')


# plt.plot(df_sasw.Time, rolling_avg_sigma, c='k')
fig , axs = plt.subplots(3, figsize=(20, 6), sharex='all')
for ax, var, label in zip(axs, [df_sasw.BrR2, df_sasw.vr, df_sasw.sigmac], 
                          ['$B_R R^2$', '$v_R$', '$\sigma_C$']):
    ax.scatter(df_sasw.Time, var, facecolor='None', edgecolor='lightpink', marker='D')
    ax.set_ylabel(ylabel=label, fontsize=18)

### AVERAGES
df_sasw['radius'] = (np.array(df_sasw.rAU)*u.AU).to(u.Rsun).value
column_averages = df_sasw.mean()
print(column_averages)

### read in from file
df = pd.read_csv(os.path.join(RES_DIR, f"{enc}_modeling.csv"))

### turn date strings into datetime objects
df['Time'] = [datetime.datetime.strptime(d, '%Y-%m-%d %H:%M:%S.%f') for d in df.Time]

df_reg = df[(df.Time >= datetime.datetime.strptime(reg[j][0], '%Y-%m-%d %H:%M:%S')) & (df.Time <= datetime.datetime.strptime(reg[j][1], '%Y-%m-%d %H:%M:%S'))]


### ------------ SAVE TO CSV FILE
date = dates[j]
df_reg.to_csv(os.path.realpath(f'fsw_results/{date}.csv'))

