#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 09:27:21 2024

@author: tamarervin

Looking at the helium abundance of the modeled streams.
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

kernels = astrospice.registry.get_kernels('psp','predict') 

from functions import read_df, gen_dt_arr, pfss, delta_long, ballistically_project, boxplot

### ------------- PLOT STYLING
from matplotlib import rc

# Set the font family to Times New Roman for text and equations
rc('font', **{'family': 'serif', 'serif': ['Times New Roman']})
rc('mathtext', fontset='custom', rm='Times New Roman', it='Times New Roman:italic', bf='Times New Roman:bold')
mpl.rcParams['font.size'] = 18
panel_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)']

### ------------- PATHS
sys.path.append(os.path.realpath('sasw_sources'))
RES_DIR = os.path.realpath('sasw_sources/results')
FIG_DIR = os.path.realpath('sasw_sources/figures')
EPS_DIR = os.path.realpath('sasw_sources/eps_figures')
ADAPT_DIR = os.path.realpath('adapt')

# =============================================================================
# ### PERIODS 
# =============================================================================
### ------------- SASW PERIODS
reg = [ ['2020-01-28 18:00:00', '2020-01-29 01:00:00', 4], ### E4
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
                ['2023-06-21 08:00:00', '2023-06-21 16:00:00', 16], ### E16 R1
                ['2023-06-23 08:00:00', '2023-06-24 14:00:00', 16], ### E16 R2
                ['2023-09-27 12:00:00', '2023-09-27 14:00:00', 17], ### E17
                ]
dates=['20200128_E4', '20200606_E5', '20200609_E5', '20200927_E6', '20200929_E6',
       '20210118_E7', '20210430_E8-1', '20210430_E8-2', '20221210_E14-1', '20221210_E14-2',
       '20221212_E14']

### ------------- FSW PERIODS
reg = [ 
        ['2020-01-27 04:55:00', '2020-01-27 05:15:00', 4], ### E4
        # ['2020-06-06 00:00:00', '2020-06-06 09:00:00', 5], ### E5
        ['2021-04-27 01:00:00', '2021-04-27 04:00:00', 8], ### E8
        ['2021-04-27 06:00:00', '2021-04-27 08:30:00', 8], ### E8
        ['2021-11-20 05:30:00', '2021-11-20 10:30:00', 10], ### E10
        # ['2022-05-29 21:30:00', '2022-05-30 23:30:00', 12], ### E12
        ['2022-12-15 21:15:00', '2022-12-15 22:15:00', 14], ### E14
        ]
dates = [
'2020-01-27', '2021-04-27',  '2021-04-27_2', '2021-11-20',  '2022-12-14'
          ]

# =============================================================================
#### CALCULATION
# =============================================================================
for j in np.arange(3, 11):
    enc = reg[j][2]
    i = enc - 4
    enc = f'e{i+4}'
    
    print('Encounter', enc)
    ### read in from file
    df = pd.read_csv(os.path.join(RES_DIR, f"{enc}_modeling.csv"))
    
    ### turn date strings into datetime objects
    df['Time'] = [datetime.datetime.strptime(d, '%Y-%m-%d %H:%M:%S.%f') for d in df.Time]
    df_reg = df[(df.Time >= datetime.datetime.strptime(reg[j][0], '%Y-%m-%d %H:%M:%S')) & (df.Time <= datetime.datetime.strptime(reg[j][1], '%Y-%m-%d %H:%M:%S'))]
    
    ### ------------- ALPHA DATA
    span_vars = pyspedas.psp.spi(trange=(reg[j][0], reg[j][1]), datatype='sf0a_l3_mom', level='l3', time_clip=True)
    
    ### ------------  PARTICLE CALCULATIONS
    dt = get_data('psp_spi_VEL_RTN_SUN')
    dt2 = get_data('psp_spi_DENS')
    dt3 = get_data('psp_spi_TEMP')
    date_obj = [datetime.datetime.strptime(time_string(d), '%Y-%m-%d %H:%M:%S.%f') for d in dt.times]
    
    rd = {'Time': date_obj, 'vra': np.abs(dt.y[:, 0]), 'vta': dt.y[:, 1], 'vna': dt.y[:, 2], 'Na': dt2.y, 'Ta': dt3.y}
    df_span = pd.DataFrame(data=rd)
    ff = pd.merge_asof(df_reg, df_span, on='Time', direction='backward')
    
    ### CALCULATIONS
    ff['B'] = np.sqrt(ff.Br**2 + ff.Bt**2 + ff.Bn**2)
    cost = np.abs(ff.Br/ff.B)
    ff['vap'] = (ff.vra - ff.vr)/cost
    ff['diff'] = np.abs(ff.vap)/ff.vA
    ff['Sp'] = ff.Tp / (ff.Np**(2/3))
    
    ### Ahe
    ff['Ahe'] = ff.Na/ff.Np
    
    ### ------------  SAVE DATAFRAME
    ff.to_csv(os.path.realpath(f'sasw_results/{dates[j]}.csv'))
    print('Saved file to:', f'sasw_results/{dates[j]}.csv')
    shutil.rmtree('psp_data/')

enc = reg[j][2]
i = enc - 4
enc = f'e{i+4}'

print('Encounter', enc)
### read in from file
df = pd.read_csv(os.path.join(RES_DIR, f"{enc}_modeling.csv"))

### turn date strings into datetime objects
df['Time'] = [datetime.datetime.strptime(d, '%Y-%m-%d %H:%M:%S.%f') for d in df.Time]
df_reg = df[(df.Time >= datetime.datetime.strptime(reg[j][0], '%Y-%m-%d %H:%M:%S')) & (df.Time <= datetime.datetime.strptime(reg[j][1], '%Y-%m-%d %H:%M:%S'))]

df_reg['Sp'] = df_reg.Tp / (df_reg.Np**(2/3))
df_reg.to_csv(os.path.realpath(f'fsw_results/{dates[j]}.csv'))
print('Saved file to:', f'fsw_results/{dates[j]}.csv')

# =============================================================================
# ### LOOK AT FITS 
# =============================================================================
fsw_fits = pd.read_csv('sasw_sources/FSW_Nap_fits.csv')
fsw_fits['Time'] = [datetime.datetime.strptime(d, '%Y-%m-%d/%H:%M:%S') for d in fsw_fits['Time FSW']]
fsw_fits['Nap'] = fsw_fits['Nap FSW']
fsw_fits['Nap'][fsw_fits['Nap'] > 0.2] = np.nan
sasw_fits = pd.read_csv('sasw_sources/SASW_Nap_fits.csv')
sasw_fits['Time'] = [datetime.datetime.strptime(d, '%Y-%m-%d/%H:%M:%S') for d in sasw_fits['Time ASSW']]
sasw_fits['Nap'] = sasw_fits['Nap ASSW']


ar_fits = pd.DataFrame()
ch_fits = pd.DataFrame()
js = [6, 8, 9]
for j in js:
    aa =  sasw_fits[(sasw_fits.Time >= datetime.datetime.strptime(reg[j][0], '%Y-%m-%d %H:%M:%S')) & (sasw_fits.Time <= datetime.datetime.strptime(reg[j][1], '%Y-%m-%d %H:%M:%S'))] 
    ar_fits = ar_fits.append(aa)

js = [0, 1, 2, 3, 4, 5, 7, 10]
for j in js:
    aa =  sasw_fits[(sasw_fits.Time >= datetime.datetime.strptime(reg[j][0], '%Y-%m-%d %H:%M:%S')) & (sasw_fits.Time <= datetime.datetime.strptime(reg[j][1], '%Y-%m-%d %H:%M:%S'))] 
    print(j, aa)
    ch_fits = ch_fits.append(aa)
    
fig, axs = plt.subplots(2, 1, figsize=(40, 7))
ax = axs[0]
ax.scatter(fsw_fits.Time, fsw_fits.Nap, c='lightpink')
ax = axs[1]
ax.scatter(sasw_fits.Time, sasw_fits.Nap, c='lavender')

fig, axs = plt.subplots(2, figsize=(20, 16), gridspec_kw={'hspace':0.35}, sharex='all')
out = False

ax = axs[0]
ax.set_title('Fits')
for var, xlabel, panel in zip(['Nap'], # r'$\rm F_{w0}$', 
                                  [r'$\rm A_{He}$'], panel_labels):
    boxplot(fsw_fits[var], 'tab:red', 'lightpink', ax, 4, out)
    boxplot(sasw_fits[var], 'tab:purple', 'lavender', ax, 3, out)
    boxplot(ch_fits[var], 'tab:brown', 'sandybrown', ax, 2, out)
    boxplot(ar_fits[var], 'tab:orange', 'navajowhite', ax, 1, out)
    ax.set(yticks=[1, 2, 3, 4], yticklabels=['non-CH', 'CH-like', 'SASW', 'FSW'])
    ax.set_xlabel(xlabel, fontsize=20)
    ax.text(0.97, 0.95, panel, transform=ax.transAxes, fontsize=20, fontweight='bold', va='top', ha='left', zorder=10)
ax = axs[1]
ax.set_title('Partial Moments')
for var, xlabel, panel in zip(['Ahe'], # r'$\rm F_{w0}$', 
                                  [r'$\rm A_{He}$'], panel_labels[1:]):
    boxplot(fsw[var], 'tab:red', 'lightpink', ax, 4, out)
    boxplot(sasw[var], 'tab:purple', 'lavender', ax, 3, out)
    boxplot(ch[var], 'tab:brown', 'sandybrown', ax, 2, out)
    boxplot(ar[var], 'tab:orange', 'navajowhite', ax, 1, out)
    ax.set(yticks=[1, 2, 3, 4], yticklabels=['non-CH', 'CH-like', 'SASW', 'FSW'])
    ax.set_xlabel(xlabel, fontsize=20)
    ax.text(0.97, 0.95, panel, transform=ax.transAxes, fontsize=20, fontweight='bold', va='top', ha='left', zorder=10)


# =============================================================================
# ### CREATE FIGURE
# =============================================================================
### ------------ READ IN AR-SASW FILES
ar_files = glob.glob(os.path.join('sasw_sources', 'sasw_active', '*'))
ar = pd.DataFrame()
for i, file in enumerate(ar_files):
    ff = pd.read_csv(file)
    ar = ar.append(ff)

### ------------ READ IN CH-SASW FILES
ch_files =  glob.glob(os.path.join('sasw_sources', 'sasw_results', '*'))
ch = pd.DataFrame()
for i, file in enumerate(ch_files):
    ff = pd.read_csv(file)
    ch = ch.append(ff)

### ------------ READ IN SASW FILES
sasw_files = glob.glob(os.path.join('sasw_sources', 'sasw_results', '*'))
sasw = ch.append(ar)

## ------------ READ IN FSW FILES
fsw_files = glob.glob(os.path.join('sasw_sources', 'fsw_results', '*'))
fsw = pd.DataFrame()
for i, file in enumerate(fsw_files):
    ff = pd.read_csv(file)
    fsw = fsw.append(ff)

# =============================================================================
# ### MASS FLUX CALCULATION
# =============================================================================
def flux(df):
    B0 = np.array(df.B0 * 1e5) * u.G
    Br = np.array(df.Br) * u.nT
    Np = np.array(df.Np) / (u.cm**3)
    vp = np.array(df.vr) * u.km / u.s
    flux = (B0 / Br) * Np * vp
    flux = flux.to(1/(u.cm**2 * u.s))
    df['flux'] = np.abs(flux.value) / 1e13

    condition = (np.abs(df.flux > 1000))
    df['flux'][condition] = np.nan

    # df['Eflux'] = (df.B0 / df.Br) * df.Np * (df.vr**3) * 1.67e-27 / 2
    # df['Eflux'][condition] = np.nan
    return df
    

ch, ar, sasw, fsw = [flux(dd) for dd in [ch, ar, sasw, fsw]]

# =============================================================================
# ### ------------ FIGURE COMPOSITION 
# =============================================================================
fig, axs = plt.subplots(4, figsize=(20, 16), gridspec_kw={'hspace':0.35})
out = False

for ax, var, xlabel, panel in zip(axs, ['NpR2', 'fluxR2','Ahe', 'diff'], # r'$\rm F_{w0}$', 
                                  [r'$\rm N_p R^2 \; [cm^{-3}]$', r'$\rm n_0 v_0 \; [10^{13} cm^{-2} s^{-1}]$', r'$\rm  A_{He}$', r'$\rm| v_{\alpha, p} |/ v_A$'], panel_labels):
    # if var == 'Sp':
    #     fsw[var] = fsw.Tp / (fsw.NpR2 ** 2/3)
    #     sasw[var] = sasw.Tp / (sasw.NpR2 ** 2/3)
    #     ch[var] = ch.Tp / (ch.NpR2 ** 2/3)
    #     ar[var] = ar.Tp / (ar.NpR2 ** 2/3)
    boxplot(fsw[var], 'tab:red', 'lightpink', ax, 4, out)
    boxplot(sasw[var], 'tab:purple', 'lavender', ax, 3, out)
    boxplot(ch[var], 'tab:brown', 'sandybrown', ax, 2, out)
    boxplot(ar[var], 'tab:orange', 'navajowhite', ax, 1, out)
    ax.set(yticks=[1, 2, 3, 4], yticklabels=['non-CH', 'CH-like', 'SASW', 'FSW'])
    ax.set_xlabel(xlabel, fontsize=20)
    ax.text(0.97, 0.95, panel, transform=ax.transAxes, fontsize=20, fontweight='bold', va='top', ha='left', zorder=10)
ax = axs[0]
# ax.set_xscale('log')
ax.axvline(4, c='k', linestyle='dashed')

ax = axs[3]
ax.axvline(0.015, c='k', linestyle='dashed')
ax.axvline(0.045, c='k', linestyle='dashed')

if out:
    plt.savefig(os.path.join(FIG_DIR, 'composition.png'), bbox_inches='tight')
    plt.savefig(os.path.join(EPS_DIR, 'composition.eps'), bbox_inches='tight')
else:
    plt.savefig(os.path.join(FIG_DIR, 'composition.png'), bbox_inches='tight')
    plt.savefig(os.path.join(EPS_DIR, 'composition.eps'), bbox_inches='tight')

# =============================================================================
# FIGURE HELIUM 
# =============================================================================
fig, ax = plt.subplots(1, figsize=(16, 12))
for df, c, lcol in zip([fsw, sasw, ch, ar],  ['tab:red', 'tab:purple', 'tab:brown', 'tab:orange'],
                       ['lightpink', 'lavender', 'sandybrown', 'navajowhite']):
    df['NaR2'] = df.Na * (df.rAU **2)
    ax.scatter(df.NpR2, df.NaR2, edgecolor=lcol, marker='D', facecolor='None', s=40, lw=0.5)
# ax.set_xscale('log')
# ax.set_yscale('log')
ax.set(xlim=(-0.5, 35), xticks=[0, 10, 20, 30], xticklabels=[0, 10, 20, 30],
        ylim=(-0.01, 1.25), yticks=[0, 0.5, 1.0], yticklabels=[0, 0.5, 1.0])
### legend
aia_patch = mpatches.Patch(edgecolor='k', facecolor='grey', label=r'$\rm AIA$')
fsw_patch = mpatches.Patch(edgecolor='tab:red', facecolor='lightpink', label=r'$\rm FSW$')
sasw_patch = mpatches.Patch(edgecolor='tab:purple', facecolor='lavender', label=r'$\rm SASW$')
ch_patch = mpatches.Patch(edgecolor='tab:brown', facecolor='sandybrown', label=r'$\rm SASW: CH-like$')
ar_patch = mpatches.Patch(edgecolor='tab:orange', facecolor='navajowhite', label=r'$\rm SASW: non-CH$')
leg0 = ax.legend(handles=[fsw_patch, ch_patch, ar_patch], loc='upper right', fontsize=20) #, bbox_to_anchor=(1, 0.95))
ax.add_artist(leg0)
### labels
ax.set_xlabel(r'$\rm N_p \; [cm^{-3}]$', fontsize=20)
ax.set_ylabel(r'$\rm N_\alpha \; [cm^{-3}]$', fontsize=20)
### lines
x = np.linspace(0, 30)
ax.plot(x, x*0.015, c='k', linestyle='dashed', zorder=-1)
ax.plot(x, x*0.045, c='k', linestyle='dotted', zorder=-1)
# ax.plot(x, 15/x)
plt.savefig(os.path.join(FIG_DIR, 'entropy.png'), bbox_inches='tight')
plt.savefig(os.path.join(EPS_DIR, 'entropy.eps'), bbox_inches='tight')

# =============================================================================
# ### ------------ FIGURE ENTROPY 
# =============================================================================
fig, ax = plt.subplots(1, figsize=(10, 8))
for df, c, lcol in zip([fsw, sasw, ch, ar],  ['tab:red', 'tab:purple', 'tab:brown', 'tab:orange'],
                       ['lightpink', 'lavender', 'sandybrown', 'navajowhite']):
    ax.scatter(df.Sp, df.vA, edgecolor=c, marker='D', facecolor=lcol, s=40, lw=0.5)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set(xlim=(0, 120), xticks=[0.1, 1.0, 10, 100], xticklabels=[0.1, 1.0, 10, 100],
       ylim=(60, 550), yticks=[80, 100, 200, 300, 400, 500], yticklabels=[80, 100, 200, 300, 400, 500])
### legend
aia_patch = mpatches.Patch(edgecolor='k', facecolor='grey', label=r'$\rm AIA$')
fsw_patch = mpatches.Patch(edgecolor='tab:red', facecolor='lightpink', label=r'$\rm FSW$')
sasw_patch = mpatches.Patch(edgecolor='tab:purple', facecolor='lavender', label=r'$\rm SASW$')
ch_patch = mpatches.Patch(edgecolor='tab:brown', facecolor='sandybrown', label=r'$\rm SASW: CH-like$')
ar_patch = mpatches.Patch(edgecolor='tab:orange', facecolor='navajowhite', label=r'$\rm SASW: non-CH$')
leg0 = ax.legend(handles=[fsw_patch, ch_patch, ar_patch], loc='lower right', fontsize=20) #, bbox_to_anchor=(1, 0.95))
ax.add_artist(leg0)
### labels
ax.axvline(1.5, c='k', linestyle='dashed', zorder=-1)
ax.set_xlabel(r'$\rm S_p \; [eV cm^2]$', fontsize=20)
ax.set_ylabel(r'$\rm v_A \; [km/s]$', fontsize=20)
### lines
x = np.linspace(0.1, 100)
ax.axvline(1.5, c='k', linestyle='dashed', zorder=-1)
# ax.plot(x, 15/x)
plt.savefig(os.path.join(FIG_DIR, 'entropy.png'), bbox_inches='tight')
plt.savefig(os.path.join(EPS_DIR, 'entropy.eps'), bbox_inches='tight')

# fig, ax = plt.subplots(1, figsize=(10, 8))
# for df, c, lcol in zip([fsw, sasw, ch, ar],  ['tab:red', 'tab:purple', 'tab:brown', 'tab:orange'],
#                        ['lightpink', 'lavender', 'sandybrown', 'navajowhite']):
#     ax.scatter(df.Tp / (df.NpR2**(2/3)), df.vA, edgecolor=c, marker='D', facecolor=lcol, s=40, lw=0.5)
# ax.set_xscale('log')
# ax.set_yscale('log')
# ax.set(xlim=(0, 120), xticks=[0.1, 1.0, 10, 100], xticklabels=[0.1, 1.0, 10, 100],
#        ylim=(60, 550), yticks=[80, 100, 200, 300, 400, 500], yticklabels=[80, 100, 200, 300, 400, 500])
# ### legend
# aia_patch = mpatches.Patch(edgecolor='k', facecolor='grey', label=r'$\rm AIA$')
# fsw_patch = mpatches.Patch(edgecolor='tab:red', facecolor='lightpink', label=r'$\rm FSW$')
# sasw_patch = mpatches.Patch(edgecolor='tab:purple', facecolor='lavender', label=r'$\rm SASW$')
# ch_patch = mpatches.Patch(edgecolor='tab:brown', facecolor='sandybrown', label=r'$\rm SASW: CH-like$')
# ar_patch = mpatches.Patch(edgecolor='tab:orange', facecolor='navajowhite', label=r'$\rm SASW: non-CH$')
# leg0 = ax.legend(handles=[fsw_patch, ch_patch, ar_patch], loc='lower right', fontsize=20) #, bbox_to_anchor=(1, 0.95))
# ax.add_artist(leg0)
# ### labels
# x = np.linspace(0.1, 100)
# ax.axvline(1.5, c='k', linestyle='dashed', zorder=-1)
# ax.plot(x, 40/x)
# ax.set_xlabel(r'$\rm S_p \; [eV cm^2]$', fontsize=20)
# ax.set_ylabel(r'$\rm v_A \; [km/s]$', fontsize=20)
# plt.savefig(os.path.join(FIG_DIR, 'entropy.png'), bbox_inches='tight')
# plt.savefig(os.path.join(EPS_DIR, 'entropy.eps'), bbox_inches='tight')
# =============================================================================
# ### ------------ RANDOM FIGURE
# =============================================================================
fig, ax = plt.subplots(1)
im = ax.scatter(sasw.Ahe, sasw['diff'], c=np.abs(sasw.B0*1e5), cmap='RdPu')
ax.set(ylim=(0, 2))
plt.colorbar(im)

fig, ax = plt.subplots(1)
im = ax.scatter(sasw.Ahe, sasw['diff'], c=np.abs(sasw.BrR2), cmap='RdPu')
ax.set(ylim=(0, 2))
plt.colorbar(im)

fig, ax = plt.subplots(1)
im = ax.scatter(sasw.mA, np.abs(sasw['B0']*1e5), c=np.abs(sasw.sigmac), cmap='RdPu')
# ax.set(ylim=(0, 2))
plt.colorbar(im)

plt.scatter(ar.fss, ar['vap'], c='r')
plt.scatter(ch.fss, ch['vap'], c='b')

plt.xlim(0, 3000)
plt.ylim(0, 100)

fig, axs = plt.subplots(1, 3, figsize=(24, 6))
varss = [['vr', 'Sp'], ['Sp', 'vA'], ['fss', 'Sp']]
for ax, var in zip(axs, varss):
    for df, c in zip([fsw, sasw, ch, ar], ['tab:red', 'tab:purple', 'tab:brown', 'tab:orange']):
        ax.scatter(df[var[0]], df[var[1]], c = c)
        ax.set_xscale('log')
        ax.set_yscale('log')
        
        
        
        
        