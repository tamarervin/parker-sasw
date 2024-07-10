#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 21:23:51 2024

@author: tamarervin

Pipeline to study the source regions of the slow Alfvenic solar wind (SASW) observed at Parker Solar Probe (PSP)
"""

### imports
import glob
import sys, os
import shutil

import urllib.request

from datetime import timedelta
from datetime import datetime  

import numpy as np
import pandas as pd

from scipy import stats
from scipy.interpolate import interp1d

from plasmapy.formulary import beta

import astrospice
import astropy.units as u

import pyspedas
from pyspedas import time_string, time_double
from pytplot import tplot, get_data, cdf_to_tplot, store_data

import sunpy 
import sunpy.coordinates as scoords
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

kernels = astrospice.registry.get_kernels('psp','predict') 


### ------------- PATHS
RES_DIR = os.path.realpath('results_07102024')
FIG_DIR = os.path.realpath('figures')
EPS_DIR = os.path.realpath('eps_figures')
ADAPT_DIR  = os.path.realpath('sasw_sources/adapt')

### user inputted stuff
need_download = True

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
                                        '2023-09-27 00:00:00', ### E17
                                        '2023-12-29 00:00:00', ### E18
                                        ]

### turbulence averaging
min_avg = 60
dens_avg=10
### days plus // minus perihelion
date_range = 5


### ------------- CALCULATIONS
def do_calculations(trange, enc, min_avg=60, dens_avg=10):
    ### ------------  DATA READIN
    pyspedas.psp.fields(trange=trange, time_clip=True, datatype='mag_RTN_4_per_cycle')
    pyspedas.psp.spi(trange=trange, datatype='sf00_l3_mom', level='l3', time_clip=True)

    ### ------------  DATA OVERVIEW
    tplot(['psp_spi_VEL_RTN_SUN', 'psp_spi_EFLUX_VS_ENERGY',
     'psp_spi_EFLUX_VS_THETA',
     'psp_spi_EFLUX_VS_PHI','psp_spi_DENS', 'psp_fld_l2_mag_RTN_4_Sa_per_Cyc'])
    plt.savefig(os.path.join(f'{FIG_DIR}/{enc}.png'))
    
    ### ------------  PARTICLE CALCULATIONS
    dt = get_data('psp_spi_VEL_RTN_SUN')
    dt2 = get_data('psp_spi_DENS')
    dt3 = get_data('psp_spi_TEMP')
    date_obj = [datetime.strptime(time_string(d), '%Y-%m-%d %H:%M:%S.%f') for d in dt.times]

    rd = {'Time': date_obj, 'vr': np.abs(dt.y[:, 0]), 'vt': dt.y[:, 1], 'vn': dt.y[:, 2], 'Np': dt2.y, 'Tp': dt3.y}
    df_span = pd.DataFrame(data=rd)

    ### ADD ANGLE
    vx, vy, vz = [get_data('psp_spi_VEL_SC').y[:, i] for i in np.arange(0, 3)]
    mx, my, mz = [get_data('psp_spi_MAGF_SC').y[:, i] for i in np.arange(0, 3)]
    vdotb = vx*mx + vy*my + vz*mz
    v = np.sqrt(vx**2 + vy**2 + vz**2)
    b = np.sqrt(mx**2 + my**2 + mz**2)
    angle_vb = np.arccos(vdotb/(v*b))
    df_span['angle_vb'] = np.rad2deg(angle_vb)

    ### ------------ MAGNETIC FIELD CALCULATIONS
    dt = get_data('psp_fld_l2_mag_RTN_4_Sa_per_Cyc')
    date_obj = [datetime.strptime(time_string(d), '%Y-%m-%d %H:%M:%S.%f') for d in dt.times]

    rd = {'Time': date_obj, 'times': dt.times, 'Br': dt.y[:, 0], 'Bt': dt.y[:, 1], 'Bn': dt.y[:, 2]}
    dfmag = pd.DataFrame(data=rd)
    dfmag['B'] = np.sqrt(dfmag.Br**2 + dfmag.Bn**2 + dfmag.Bt**2)

    ### ------------ FULL CALCULATIONS
    ### MERGE DATAFRAME
    df = pd.merge_asof(df_span, dfmag, on='Time', direction='backward')

    ### ALFVEN SPEED
    df['VA'] = df.B * 1e-9 / np.sqrt(1.25e-6 * df.Np * 1e6 * 1.67e-27) / 1000.0
    df['mA'] = df.vr / df.VA

    ### CALCULATE DENSITY AVERAGE FOR TURBULENCE CALCULATIONS
    df['timestamp'] = pd.to_datetime(df['Time'])
    df['use_dens'] = df['Np'].rolling(window=int((dens_avg*60)/7), min_periods=1).mean()
    df['polarity'] = np.sign(df.Br)
    
    ### CALCULATE FLUCTUATIONS AND ELSASSER
    df['sigmac'],df['sigmar'], df['vA'], df['Zp'],df['Zm'], df['deltav'], df['deltab'], df['rA'] = calcs.calc_sigma(df, num=int((min_avg*60)/7))
    
    ### PLASMA BETA 
    condition = (df.Tp <= 0)
    df['Tp'][condition] = 0
    df['beta'] = beta(np.array(df.Tp)*u.eV, np.array(df.Np)/(u.cm*u.cm*u.cm), np.array(df.B)*u.nT).value
    
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

    # add to dataframe
    df['lon'] = psp_carrington.lon.value
    df['lat'] = psp_carrington.lat.value
    df['rAU'] = psp_carrington.radius.to(u.AU).value
    df['sslon'] = psp_projected.lon.value
    df['sslat'] = psp_projected.lat.value
    df['ssrAU'] = psp_projected.radius.to(u.AU).value
    df['NpR2'] = df.Np * (df.rAU ** 2)
    df['BrR2'] = df.Br * (df.rAU ** 2)
    
    ### ------------  CALCULATE WIND EMERGENCE TIME
    vels = df.vr * u.km/u.s
    rads = psp_carrington.radius.to(u.km).value
    
    deltaT = np.array(rads / vels)
    df['Tsun'] = [timestamp - timedelta(seconds=seconds) 
        if pd.notnull(seconds)  else np.nan for timestamp, seconds in zip(df.Time, deltaT)]   
    
    ### ------------ IDENTIFY WIND TYPES 
    ### check about 20 time range or something 
    rolling_avg_vr = df['vr'].rolling(window=int((20*60)/7)).mean()
    rolling_avg_sigma = df['sigmac'].rolling(window=int((20*60)/7)).mean()
    
    # Apply conditions to categorize based on rolling averages
    condition = (rolling_avg_vr <= 350) & (np.abs(rolling_avg_sigma) >= 0.8)
    df['period'] = np.where(condition, 1, 0)
    
    # ### PLOT SASW
    # sasw = np.where(df['period'] == 1)[0]
    
    # fig, axs = plt.subplots(2, figsize=(20, 8))
    # axs[0].scatter(df.Time, df.vr, c='lightpink', label=r'$v_R \; [km/s]$')
    # axs[0].scatter(df.Time[sasw], df.vr[sasw], c='lightblue', label=r'$v_R \; [km/s]$')
    # axs[0].set_ylabel('$v_R \; [km/s]$')
    # axs[1].scatter(df.Time, df.sigmac, c='lightpink', label=r'$\sigma_C$')
    # axs[1].scatter(df.Time[sasw], df.sigmac[sasw], c='lightblue', label=r'$\sigma_C$')
    # axs[1].set_ylabel(r'$\sigma_C$')
    
    ### ------------ COMPLETED CALCULATIONS
    print('completed calculations for time range:', trange)
    
    ### ------------ DELETE FILES
    shutil.rmtree('psp_data/')
    
    return df

### ------------ PIPELINE
for i, pdate in enumerate(perihelion_dates[7:-4]):
    i += 11
    print('Running Calculations for Encounter', i, 'Perihelion:', pdate)
    
    ## ------------   DETERMINE TIME RANGE
    # Convert back to string in the specified format
    format_string = '%Y-%m-%d %H:%M:%S'
    pdate = datetime.strptime(pdate,format_string)
    trange = [(pdate - timedelta(days = date_range)).strftime(format_string), 
              (pdate + timedelta(days = date_range)) .strftime(format_string)]

    ### ------------   CALCULATIONS
    df = do_calculations(trange, enc=f'e{i}')
    
    ### ------------ SAVE TO CSV FILE
    df.to_csv(os.path.join(RES_DIR, f"e{i}.csv"))
    
    ### ------------ DELETE DF
    del df
    
    ## ------------ COMPLETED
    print(f'complete for encounter {i} time range:', trange)
    
    
