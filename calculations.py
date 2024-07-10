#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 15:58:28 2024

@author: tamarervin
"""

import numpy as np
import matplotlib.pyplot as plt

import astropy.units as u 
from astropy.coordinates import SkyCoord

def smooth(data, num):
    # pad array to deal with edges
    padded_data = np.pad(data, (num // 2, num // 2), mode='edge')
    # compute the moving average using convolution
    weights = np.ones(num) / num
    return np.convolve(data, np.ones(num)/num, mode='same')


def calculate_delta(data, smoothed_data):
    return data - smoothed_data


def calc_deltaB(df, num):
    Br_smo = smooth(df.Br, num)
    Bt_smo = smooth(df.Bt, num)
    Bn_smo = smooth(df.Bn, num)
    delta_Br = calculate_delta(df.Br, Br_smo)
    delta_Bt = calculate_delta(df.Bt, Bt_smo)
    delta_Bn = calculate_delta(df.Bn, Bn_smo)

    return delta_Br, delta_Bt, delta_Bn, Br_smo, Bt_smo, Bn_smo

def calculate_velocity_change(delta_b, nppin):
    return delta_b * 1e-9 / np.sqrt(1.25e-6 * nppin * 1e6 * 1.67e-27) / 1000.0

def calc_vA(B, nppin):
    return B * 1e-9 / np.sqrt(1.25e-6 * nppin * 1e6 * 1.67e-27) / 1000.0

def calculate_sigma_c(zp_square, zm_square):
    return (zp_square - zm_square) / (zp_square + zm_square)

def calc_sigma(dataframe, num=371):
    # smooth and calculate velocity fluctuation
    Vr_smo = smooth(dataframe.vr, num)
    Vt_smo = smooth(dataframe.vt, num)
    Vn_smo = smooth(dataframe.vn, num)
    delta_Vr = calculate_delta(dataframe.vr, Vr_smo)
    delta_Vt = calculate_delta(dataframe.vt, Vt_smo)
    delta_Vn = calculate_delta(dataframe.vn, Vn_smo)

    # smooth and calculate magnetic field fluctuation
    Br_smo = smooth(dataframe.Br, num)
    Bt_smo = smooth(dataframe.Bt, num)
    Bn_smo = smooth(dataframe.Bn, num)
    delta_Br = calculate_delta(dataframe.Br, Br_smo)
    delta_Bt = calculate_delta(dataframe.Bt, Bt_smo)
    delta_Bn = calculate_delta(dataframe.Bn, Bn_smo)
    
    # calculate Alfven speed
    Vra = calc_vA(dataframe.Br, dataframe.use_dens.values)
    Vta = calc_vA(dataframe.Bt, dataframe.use_dens.values)
    Vna = calc_vA(dataframe.Bn, dataframe.use_dens.values)
    vA = np.linalg.norm([Vra, Vta, Vna], axis=0)

    # calculate magnetic field in velocity units
    polarity = smooth(dataframe.polarity, num)
    delta_Vrb = calculate_velocity_change(delta_Br, dataframe.use_dens.values) * (-1 * polarity)
    delta_Vtb = calculate_velocity_change(delta_Bt, dataframe.use_dens.values) * (-1 * polarity)
    delta_Vnb = calculate_velocity_change(delta_Bn, dataframe.use_dens.values) * (-1 * polarity) 
    
    # calculate Zp and Zm
    Zpr = delta_Vr.values + delta_Vrb.values
    Zpt = delta_Vt.values + delta_Vtb.values
    Zpn = delta_Vn.values + delta_Vnb.values
    Zmr = delta_Vr.values - delta_Vrb.values
    Zmt = delta_Vt.values - delta_Vtb.values
    Zmn = delta_Vn.values - delta_Vnb.values

    # take norm
    Zpsquare = Zpr**2 + Zpt**2 + Zpn**2
    Zmsquare = Zmr**2 + Zmt**2 + Zmn**2

    deltav = delta_Vr**2 + delta_Vt**2 + delta_Vn**2 
    deltab = delta_Vrb**2 + delta_Vtb**2 + delta_Vnb**2 

    # background time average
    # numerator = delta_Vr * delta_Vrb + delta_Vt * delta_Vtb + delta_Vn * delta_Vnb
    num_avg = smooth(Zpsquare, num) - smooth(Zmsquare, num)
    # num_avg = smooth(numerator, num)
    deltav_avg = smooth(deltav, num)
    deltab_avg = smooth(deltab, num)
    denom = smooth(Zpsquare, num) + smooth(Zmsquare, num)

    # calculate sigma
    sigmac = num_avg / denom
    denom = smooth(deltav, num) + smooth(deltab, num)
    sigmar = (deltav_avg - deltab_avg) / denom
    
    # alfven ratio
    rA = smooth(deltav, num) / smooth(deltab, num)

    return sigmac, sigmar, vA, np.sqrt(Zpsquare), np.sqrt(Zmsquare), np.sqrt(deltav), np.sqrt(deltab), rA

def calc_components(dataframe, num=371):
    # smooth and calculate velocity fluctuation
    Vr_smo = smooth(dataframe.vr, num)
    Vt_smo = smooth(dataframe.vt, num)
    Vn_smo = smooth(dataframe.vn, num)
    delta_Vr = calculate_delta(dataframe.vr, Vr_smo)
    delta_Vt = calculate_delta(dataframe.vt, Vt_smo)
    delta_Vn = calculate_delta(dataframe.vn, Vn_smo)

    # smooth and calculate magnetic field fluctuation
    Br_smo = smooth(dataframe.Br, num)
    Bt_smo = smooth(dataframe.Bt, num)
    Bn_smo = smooth(dataframe.Bn, num)
    delta_Br = calculate_delta(dataframe.Br, Br_smo)
    delta_Bt = calculate_delta(dataframe.Bt, Bt_smo)
    delta_Bn = calculate_delta(dataframe.Bn, Bn_smo)
    
    # calculate Alfven speed
    Vra = calc_vA(dataframe.Br, dataframe.use_dens.values)
    Vta = calc_vA(dataframe.Bt, dataframe.use_dens.values)
    Vna = calc_vA(dataframe.Bn, dataframe.use_dens.values)
    vA = np.linalg.norm([Vra, Vta, Vna], axis=0)

    # calculate magnetic field in velocity units
    polarity = smooth(dataframe.polarity, num)
    delta_Vrb = calculate_velocity_change(delta_Br, dataframe.use_dens.values) * (-1 * polarity)
    delta_Vtb = calculate_velocity_change(delta_Bt, dataframe.use_dens.values) * (-1 * polarity)
    delta_Vnb = calculate_velocity_change(delta_Bn, dataframe.use_dens.values) * (-1 * polarity)
    
    # calculate Zp and Zm
    Zpr = delta_Vr.values + delta_Vrb.values
    Zpt = delta_Vt.values + delta_Vtb.values
    Zpn = delta_Vn.values + delta_Vnb.values
    Zmr = delta_Vr.values - delta_Vrb.values
    Zmt = delta_Vt.values - delta_Vtb.values
    Zmn = delta_Vn.values - delta_Vnb.values

    # take norm
    Zpsquare = Zpr**2 + Zpt**2 + Zpn**2
    Zmsquare = Zmr**2 + Zmt**2 + Zmn**2

    deltav = delta_Vr**2 + delta_Vt**2 + delta_Vn**2 
    deltab = delta_Vrb**2 + delta_Vtb**2 + delta_Vnb**2 

    return [np.sqrt(Zpr**2), np.sqrt(Zpt**2), np.sqrt(Zpn**2)], [np.sqrt(Zmr**2), np.sqrt(Zmt**2), np.sqrt(Zmn**2)], [np.sqrt(delta_Vr**2), np.sqrt(delta_Vt**2), np.sqrt(delta_Vn**2)], [np.sqrt(delta_Vrb**2), np.sqrt(delta_Vtb**2), np.sqrt(delta_Vnb**2)]
def turbulence(df, num=171):
    
   # smooth and calculate velocity fluctuation
   Vr_smo = df['vr'].rolling(window=num, min_periods=1).mean()
   Vt_smo = df['vt'].rolling(window=num, min_periods=1).mean()
   Vn_smo = df['vn'].rolling(window=num, min_periods=1).mean()
   delta_Vr = calculate_delta(df.vr, Vr_smo)
   delta_Vt = calculate_delta(df.vt, Vt_smo)
   delta_Vn = calculate_delta(df.vn, Vn_smo)
   
   # smooth and calculate magnetic field fluctuation
   Br_smo = df['Br'].rolling(window=num, min_periods=1).mean()
   Bt_smo = df['Bt'].rolling(window=num, min_periods=1).mean()
   Bn_smo = df['Bn'].rolling(window=num, min_periods=1).mean()
   delta_Br = calculate_delta(df.Br, Br_smo)
   delta_Bt = calculate_delta(df.Bt, Bt_smo)
   delta_Bn = calculate_delta(df.Bn, Bn_smo)
   
   # calculate magnetic field in velocity units
   delta_Vrb = calculate_velocity_change(delta_Br, df.use_dens.values)
   delta_Vtb = calculate_velocity_change(delta_Bt, df.use_dens.values)
   delta_Vnb = calculate_velocity_change(delta_Bn, df.use_dens.values)
   
   # calculate elsasser variables
   Zpr = delta_Vr.values + delta_Vrb.values
   Zpt = delta_Vt.values + delta_Vtb.values
   Zpn = delta_Vn.values + delta_Vnb.values
   Zmr = delta_Vr.values - delta_Vrb.values
   Zmt = delta_Vt.values - delta_Vtb.values
   Zmn = delta_Vn.values - delta_Vnb.values

   Zpsquare = Zpr**2 + Zpt**2 + Zpn**2
   Zmsquare = Zmr**2 + Zmt**2 + Zmn**2
   
   df['Zp'] = np.sqrt(Zpsquare)
   df['Zm'] = np.sqrt(Zmsquare)

   # calculate fluctuations
   df['deltav'] = np.sqrt(delta_Vr**2 + delta_Vt**2 + delta_Vn**2)
   df['deltab'] = np.sqrt(delta_Vrb**2 + delta_Vtb**2 + delta_Vnb**2)
   
   # calculate cross helicity
   df['num'] = delta_Vr * delta_Vrb + delta_Vt * delta_Vtb + delta_Vn * delta_Vnb
   df['dv2'] = df.deltav ** 2
   df['db2'] = df.deltab ** 2
   numerC = df['num'].rolling(window=num, min_periods=1).mean()
   denom = df['dv2'].rolling(window=num, min_periods=1).mean() + df['db2'].rolling(window=num, min_periods=1).mean()
   df['sigmac'] = 2 * numerC / denom
   
   # residual energy
   numerR = df['dv2'].rolling(window=num, min_periods=1).mean() - df['db2'].rolling(window=num, min_periods=1).mean()
   df['sigmar'] = numerR / denom
   
   # plotting
   fig, axs = plt.subplots(3, figsize=(12, 12))
   ax = axs[0]
   ax.plot(df.Time, df.num, c='red', label='$\sigma_C \; Numerator$')
   ax.plot(df.Time, numerC, c='k', label='Average')
   
   ax = axs[1]
   ax.plot(df.Time, df.dv2 + df.db2, c='red', label='$\sigma_C \; Denominator$')
   ax.plot(df.Time, denom, c='k', label='Average')
   
   ax = axs[2]
   ax.plot(df.Time, df.dv2 - df.db2, c='red', label='$\sigma_R \; Denominator$')
   ax.plot(df.Time, numerR, c='k', label='Average')
   
   for ax in axs:
       ax.legend()
   
   # drop unnecessary columns
   df = df.drop(['num', 'dv2', 'db2'], axis=1)
   
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


