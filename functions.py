#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 15:00:19 2024

@author: tamarervin

Functions for SASW project.
"""

### imports
import pandas as pd
import datetime

from datetime import timedelta

import numpy as np

import astrospice
import astropy.units as u
from astropy.coordinates import SkyCoord


import sunpy 
import sunpy.map

import pfss_funcs as pfss_funcs

### ------------- BALLSTIC PROPAGATION
def read_df(filepath):
     df = pd.read_csv(filepath, low_memory=False)
     df['Time'] = [datetime.datetime.strptime(d, '%Y-%m-%d %H:%M:%S.%f') for d in df.Time]
     
     return df
 
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

    return flines_psp, topologies, hcs, pfss_model

### ------------ generate datetime array
def gen_dt_arr(dt_init,dt_final,cadence_days=1) :
    """
    Get array of datetime.datetime from {dt_init} to {dt_final} every 
    {cadence_days} days
    """
    dt_list = []
    while dt_init < dt_final :
        dt_list.append(dt_init)
        dt_init += timedelta(days=cadence_days)
    return np.array(dt_list)



### ------------ BOXPLOT
def boxplot(data, col, lcol, ax, position, outliers=False):
    data = data[~np.isnan(data)]
    quartiles = np.percentile(data, [25, 50, 75])
    q1, median, q3 = quartiles
    mean = np.mean(data)
    ax.boxplot(data, vert=False, patch_artist=True, boxprops=dict(facecolor=lcol, color=col),
               medianprops=dict(color=col), showfliers=outliers, positions=[position])
    ax.axvline(mean, color=col, linestyle='dashed', linewidth=1)
    return q1, median, q3