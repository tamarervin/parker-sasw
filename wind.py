#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 16:57:49 2024

@author: tamarervin

Looking at Wind 3dp velocity distribution
"""
### imports
import os

import numpy as np
import pandas as pd
from datetime import datetime  

import matplotlib as mpl
import matplotlib.pyplot as plt


### ------------- PLOT STYLING
from matplotlib import rc

# Set the font family to Times New Roman for text and equations
rc('font', **{'family': 'serif', 'serif': ['Times New Roman']})
rc('mathtext', fontset='custom', rm='Times New Roman', it='Times New Roman:italic', bf='Times New Roman:bold')
mpl.rcParams['font.size'] = 18

# =============================================================================
# ### FUNCTIONS 
# =============================================================================
def percentage_above_value(variable, value):
    """
    Calculate the percentage of the distribution that is above a certain value.

    Parameters:
    df (pd.DataFrame): The dataframe containing the data.
    column_name (str): The name of the column to analyze.
    value (float): The value to find the percentile for.

    Returns:
    float: The percentage of the distribution above the specified value.
    """

    sorted_data = np.sort(variable)
    rank = np.searchsorted(sorted_data, value, side='right')

    percentile = (rank / len(sorted_data)) * 100
    percentage_above = 100 - percentile

    return percentage_above

# =============================================================================
# ### DATA DOWNLOAD
# =============================================================================
# years = [2020, 2021, 2022, 2023]

# year = 2020
# while year < 2020:
#     time_range = [f'{year}-01-01/00:00', f'{year+1}-01-01/00:00']
#     print('Starting calculation for', time_range)

#     ### download data
#     threedp_vars = pyspedas.wind.threedp(trange=time_range, datatype='3dp_pm')
    
#     ### read in data
#     # files = glob.glob(os.path.join(f'wind_data/3dp/3dp_pm/{year}', '*'))
#     # vars = cdf_to_tplot(files)
    
#     ### create dataframe
#     dt = get_data('wi_3dp_pm_P_VELS')
#     date_obj_vr = [datetime.strptime(time_string(d), '%Y-%m-%d %H:%M:%S.%f') for d in dt.times]
#     rd = {'Time': date_obj_vr, 'vr': np.abs(dt.y[:, 0]), 'vt': dt.y[:, 1], 'vn': dt.y[:, 2]}
#     df = pd.DataFrame(data=rd, index=None)
    
#     ### create figure
#     fig, ax = plt.subplots(1, figsize=(20, 4))
#     ax.scatter(df.Time, df.vr, c='tab:purple', s=0.5)
#     ax.set(ylabel=r'$\rm v_R \; [km/s]$', title=f'{year} Wind Data')
#     ax.axhline(500, linestyle='dotted', color='k', zorder=-10)
#     plt.savefig(f'wind/{year}.png', bbox_inches='tight')
    
#     ### save dataframe
#     df.to_csv(f'wind/{year}_wind_pm.csv')
    
#     ### delete files
#     shutil.rmtree('wind_data/')
    
#     year += 1
    
# =============================================================================
# ### CREATE FIGURES
# =============================================================================
### read in year dataframes
# dataframes, vr = [], []
csv_directory = 'wind'
# for filename in sorted(os.listdir(csv_directory)):
#     if filename.endswith("_pm.csv"):
#         file_path = os.path.join(csv_directory, filename)
#         df = pd.read_csv(file_path)
#         condition = np.logical_or(df.vr <= 150, df.vr >= 1000)
#         df['vr'][condition] = np.nan
#         df = df.dropna()
#         dataframes.append(df)
#         # vr.append(df.vr)

# df_parker = pd.concat(dataframes[16:-1], ignore_index=True)
# df_cycle = pd.concat(dataframes[4:16], ignore_index=True)
# df_full = pd.concat(dataframes, ignore_index=True)
# df_jia = pd.concat(dataframes[:16], ignore_index=True)
# df_parker['Time'] = [datetime.strptime(d, '%Y-%m-%d %H:%M:%S.%f') for d in df_parker.Time]

### read in from full dataframe
# df_full.to_csv('full_df.csv')

df_full = pd.read_csv('full_df.csv')
df_full['Time'] = [datetime.strptime(d, '%Y-%m-%d %H:%M:%S.%f') for d in df_full.Time]

df_jia = df_full[(df_full.Time >= datetime.strptime('2004-01-01', '%Y-%m-%d')) & (df_full.Time <= datetime.strptime('2020-01-01', '%Y-%m-%d'))]
df_cycle = df_full[(df_full.Time >= datetime.strptime('2008-01-01', '%Y-%m-%d')) & (df_full.Time <= datetime.strptime('2020-01-01', '%Y-%m-%d'))]
df_parker = df_full[(df_full.Time >= datetime.strptime('2020-01-01', '%Y-%m-%d')) & (df_full.Time <= datetime.strptime('2023-01-01', '%Y-%m-%d'))]
print('Finished reading in files.')

# percentages
per400 = percentage_above_value(df_parker.vr, 400)
per500 = percentage_above_value(df_parker.vr, 500)
print('2020 - 2023, above 400 km/s', per400, 'above 500 km/s', per500)

# 2004 to 2020
per400 = percentage_above_value(df_jia.vr, 400)
per500 = percentage_above_value(df_jia.vr, 500)
print('2004 to 2020 above 400 km/s', per400, 'above 500 km/s', per500)

# flatten vr list
# flat_vr = [item for sublist in vr for item in sublist]

print('Creating figure.')

### figure
fig = plt.figure(figsize=(40, 12))
gs = mpl.gridspec.GridSpec(2, 3, height_ratios=[0.3, 1], wspace=0.15, hspace=0.16) 

### velocity overview
ax = fig.add_subplot(gs[0, :])
ax.scatter(df_parker.Time, df_parker.vr, c='k', s=0.5)
ax.set(ylabel=r'$\rm v_R \; [km/s]$', ylim=(0, 1000), yticks=[100, 300, 500, 700, 900], yticklabels=[100, 300, 500, 700, 900])
ax.axhline(500, linestyle='dotted', color='red', zorder=-10)
ax.axhline(400, linestyle='dotted', color='red', zorder=-10)
ax.text(0.95, 0.95, '(a)', transform=ax.transAxes, fontsize=20, fontweight='bold', va='top', ha='left', zorder=10)

### histograms
axs = [fig.add_subplot(gs[1, i]) for i in np.arange(3)]
# 2004 - 2024
ax = axs[0]
per400 = percentage_above_value(df_full.vr, 400)
per500 = percentage_above_value(df_full.vr, 500)

ax.hist(df_full.vr, bins=30, range=(0, 1000), facecolor='lightgrey', edgecolor='black', density=True)
ax.set(xlabel=r'$\rm v_R \; [km/s]$', xlim=(150, 850), xticks=[200, 400, 600, 800], xticklabels=[200, 400, 600, 800])
ax.axvline(400, color='blue', linestyle='dotted', label=f'{round(per400, 1)}%')
ax.axvline(500, color='red', linestyle='dotted', label=f'{round(per500, 1)}%')
ax.set_title('2004 to 2024', fontsize=22)
ax.legend(loc='upper right')
ax.text(0.95, 0.95, '(b)', transform=ax.transAxes, fontsize=20, fontweight='bold', va='top', ha='left', zorder=10)

# solar cycle
ax = axs[1]
per400 = percentage_above_value(df_cycle.vr, 400)
per500 = percentage_above_value(df_cycle.vr, 500)

ax.hist(df_cycle.vr, bins=30, range=(0, 1000), facecolor='lightgrey', edgecolor='black', density=True)
ax.set(xlabel=r'$\rm v_R \; [km/s]$', xlim=(150, 850), xticks=[200, 400, 600, 800], xticklabels=[200, 400, 600, 800])
ax.axvline(400, color='blue', linestyle='dotted', label=f'{round(per400, 1)}%')
ax.axvline(500, color='red', linestyle='dotted', label=f'{round(per500, 1)}%')
ax.set_title('Solar Cycle 24', fontsize=22)
ax.legend(loc='upper right')
ax.text(0.95, 0.95, '(c)', transform=ax.transAxes, fontsize=20, fontweight='bold', va='top', ha='left', zorder=10)

# 2020 - 2024
ax = axs[2]
per400 = percentage_above_value(df_parker.vr, 400)
per500 = percentage_above_value(df_parker.vr, 500)

ax.hist(df_parker.vr, bins=30, range=(0, 1000), facecolor='lightgrey', edgecolor='black', density=True)
ax.set(xlabel=r'$\rm v_R \; [km/s]$', xlim=(150, 850), xticks=[200, 400, 600, 800], xticklabels=[200, 400, 600, 800])
ax.axvline(400, color='blue', linestyle='dotted', label=f'{round(per400, 1)}%')
ax.axvline(500, color='red', linestyle='dotted', label=f'{round(per500, 1)}%')
ax.set_title('2020 to 2023', fontsize=22)
ax.legend(loc='upper right')
ax.text(0.95, 0.95, '(d)', transform=ax.transAxes, fontsize=20, fontweight='bold', va='top', ha='left', zorder=10)

plt.savefig('wind/wind_figure.png', bbox_inches='tight')
plt.savefig('wind/wind_figure.eps', bbox_inches='tight')

print('Saved figure.')
