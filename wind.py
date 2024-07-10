#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 16:57:49 2024

@author: tamarervin

Looking at Wind 3dp velocity distribution
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

### ------------- PLOT STYLING
from matplotlib import rc

# Set the font family to Times New Roman for text and equations
rc('font', **{'family': 'serif', 'serif': ['Times New Roman']})
rc('mathtext', fontset='custom', rm='Times New Roman', it='Times New Roman:italic', bf='Times New Roman:bold')
mpl.rcParams['font.size'] = 18

# =============================================================================
# ### FUNCTIONS 
# =============================================================================
def percentage_above_value(df, column_name, value):
    """
    Calculate the percentage of the distribution that is above a certain value.

    Parameters:
    df (pd.DataFrame): The dataframe containing the data.
    column_name (str): The name of the column to analyze.
    value (float): The value to find the percentile for.

    Returns:
    float: The percentage of the distribution above the specified value.
    """

    sorted_data = np.sort(df[column_name])
    rank = np.searchsorted(sorted_data, value, side='right')

    percentile = (rank / len(sorted_data)) * 100
    percentage_above = 100 - percentile

    return percentage_above

# =============================================================================
# ### DATA DOWNLOAD
# =============================================================================
years = [2020, 2021, 2022, 2023]

year = 2004
while year < 2020:
    time_range = [f'{year}-01-01/00:00', f'{year+1}-01-01/00:00']
    print('Starting calculation for', time_range)

    ### download data
    threedp_vars = pyspedas.wind.threedp(trange=time_range, datatype='3dp_pm')
    
    ### read in data
    # files = glob.glob(os.path.join(f'wind_data/3dp/3dp_pm/{year}', '*'))
    # vars = cdf_to_tplot(files)
    
    ### create dataframe
    dt = get_data('wi_3dp_pm_P_VELS')
    date_obj_vr = [datetime.strptime(time_string(d), '%Y-%m-%d %H:%M:%S.%f') for d in dt.times]
    rd = {'Time': date_obj_vr, 'vr': np.abs(dt.y[:, 0]), 'vt': dt.y[:, 1], 'vn': dt.y[:, 2]}
    df = pd.DataFrame(data=rd, index=None)
    
    ### create figure
    fig, ax = plt.subplots(1, figsize=(20, 4))
    ax.scatter(df.Time, df.vr, c='tab:purple', s=0.5)
    ax.set(ylabel=r'$\rm v_R \; [km/s]$', title=f'{year} Wind Data')
    ax.axhline(500, linestyle='dotted', color='k', zorder=-10)
    plt.savefig(f'wind/{year}.png', bbox_inches='tight')
    
    ### save dataframe
    df.to_csv(f'wind/{year}_wind_pm.csv')
    
    ### delete files
    shutil.rmtree('wind_data/')
    
    year += 1
    
# =============================================================================
# ### CREATE FIGURES
# =============================================================================
### read in dataframes
dataframes = []
csv_directory = 'wind'
for filename in os.listdir(csv_directory):
    if filename.endswith("_pm.csv"):
        file_path = os.path.join(csv_directory, filename)
        df = pd.read_csv(file_path)
        dataframes.append(df)

df_parker = pd.concat(dataframes, ignore_index=True)
df_full = pd.concat(dataframes, ignore_index=True)
df_full['Time'] = [datetime.strptime(d, '%Y-%m-%d %H:%M:%S.%f') for d in df_full.Time]

### figure
fig = plt.figure(figsize=(40, 12))
gs = mpl.gridspec.GridSpec(2, 3, height_ratios=[0.3, 1], wspace=0.15, hspace=0.16) 

### velocity overview
ax = fig.add_subplot(gs[0, :])
ax.scatter(df_full.Time, df_full.vr, c='tab:purple', s=0.5)
ax.set(ylabel=r'$\rm v_R \; [km/s]$', yticks=[100, 300, 500, 700, 900], yticklabels=[100, 300, 500, 700, 900])
ax.axhline(500, linestyle='dotted', color='k', zorder=-10)


### histograms
axs = [fig.add_subplot(gs[1, i]) for i in np.arange(3)]
# 2004 - 2024
ax = axs[0]

# solar cycle
ax = axs[1]

# 2020 - 2024
ax = axs[2]
per400 = percentage_above_value(df, 'vr', 400)
per500 = percentage_above_value(df, 'vr', 500)

ax.hist(df['vr'], bins=30, facecolor='lightgrey', edgecolor='black', density=True)
ax.set(xlabel=r'$\rm v_R \; [km/s]$', xlim=(150, 850), xticks=[200, 400, 600, 800], xticklabels=[200, 400, 600, 800])
ax.axvline(400, color='blue', linestyle='dotted', label=f'{round(per400, 1)}%')
ax.axvline(500, color='red', linestyle='dotted', label=f'{round(per500, 1)}%')
ax.set_title('2020 to 2024', fontsize=22)
ax.legend(loc='upper right')

plt.savefig('wind/wind_figure.png', bbox_inches='tight')
plt.savefig('wind/wind_figure.eps', bbox_inches='tight')