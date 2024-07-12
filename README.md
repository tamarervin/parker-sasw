# Characteristics and Source Regions of Slow Alfvenic Solar Wind Observed by Parker Solar Probe

Repository containing code for analysis and reproduction of figures from Ervin et. al. (2024) paper: Characteristics and Source Regions of Slow Alfvenic Solar Wind Observed by Parker Solar Probe

[![DOI](https://zenodo.org/badge/827028789.svg)](https://zenodo.org/doi/10.5281/zenodo.12734996)


# Main Text

## Figure One
Distribution functions of solar wind speed binned by heliocentric distance. The dashed lines indicate different threshold percentages based on observations of wind speeds at 1 AU (see Section 3.1) as applied to Parker. The red, purple, and blue lines signify the uppermost 13%, 25%, and 38% of the wind respectively where 13% is the 500 km/s cutoff at 1 AU, 38% is the 400 km/s cutoff, and 25% falls in between. In our data stream identification, we use the red line, or the fastest 13% of wind measured from Parker to differentiate slow and fast solar winds. The velocities associated with each dashed line is listed.

- *Panel (a):* All Parker solar wind velocity data below 40 $R_{\odot}$ for Encounters 4-14 with associated dashed lines.
- *Panel (b) - (d):* All Parker solar wind velocity data for radial bins
  - *(b):* 10-20 $R_{\odot}$
  - *(c):* 20-30 $R_{\odot}$
  - *(d):* 30-40 $R_{\odot}$ with the associated thresholds.

## Figure Two
All usable E4 to E14 data (see Section 3.1) is compiled into normalized histograms of cross helicity ($\sigma_C$) against the residual energy ($\sigma_R$) to showcase the identification method. The criteria for categorization of "high" Alfvenicity and the radially dependent velocity thresholding method is described in Section 3.1. Panels (a), (b), and (c) plot the designated data for SSW, SASW, and FSW respectively.

- *Panel (a):* Plot of designated data for SSW.
- *Panel (b):* Plot of designated data for SASW.
- *Panel (c):* Plot of designated data for FSW.

## Figure Three
All usable E4 to E14 data (see Section 3.1) is compiled into these normalized histograms of alpha particle density ($N_\alpha$) against proton density ($N_p$), each scaled by $R^2$. Particle measurements from the SPAN-I instrument aboard Parker/SWEAP show the alpha-to-proton abundance ratios. The dashed lines indicate the typical "high helium abundance" threshold of 0.045 and the "low abundance" boundary of 0.015 (Kasper et. al. 2007; Kasper et. al. 2012). Panels (a), (b), and (c) plot the designated data for SSW, SASW, and FSW respectively.

- *Panel (a):* Plot of designated data for SSW.
- *Panel (b):* Plot of designated data for SASW.
- *Panel (c):* Plot of designated data for FSW.

## Figure Four
Overview of parameters determined from PFSS modeling and the associated estimated footpoints for the FSW, SASW, and the CH-like and non-CH SASW sub-categories. The boxplot shows the inner quartile range (25 - 75 %) percentile and the solid line shows the median value. The dashed vertical lines show the mean of each parameter for the type of wind stream. Panels (a), (b), and (c) show the photospheric footpoint field strength ($B_0$), intensity at the footpoint from AIA 193&#8491; images, and expansion factor based on Equation 3 respectively.

- *Panel (a):* Photospheric footpoint field strength ($B_0$).
- *Panel (b):* Intensity at the footpoint from AIA 193&#8491; images.
- *Panel (c):* Expansion factor based on Equation 3.

## Figure Five
Normalized histograms comparing intensity values from many AIA images with intensity values sampled at the footpoints of the 11 identified and modeled SASW streams. The AIA images chosen are images at a 24-hour cadence for two weeks around each observation.

- *Panel (a):* Intensity values from many AIA images.
- *Panel (b):* Intensity values sampled at the footpoints of the 11 identified and modeled SASW streams.

## Figure Six
Box plot overviews of compositional parameters for the 11 SASW and 5 FSW streams we modeled. The dashed vertical lines indicate the mean for the distribution with the corresponding color and the solid line shows the median.

- *Panel (a):* Scaled proton density from SWEAP/SPAN-I.
- *Panel (b):* Proton mass flux ($n_0 v_0 = (B_0 / B_r) N_P v_P$) calculated from in situ observations and the photospheric footpoint field strength from modeling results.
- *Panel (c):* Alpha abundance ratio ($A_{He}$ ) using alpha and proton measurements from SWEAP/SPAN-I. The dashed black lines indicate the high (0.045) and low (0.015) $A_{He}$ thresholds (Kasper et. al. 2007; Kasper et. al. 2012).
- *Panel (d):* Normalized differential streaming speed ($|v_{\alpha, p}|$ / $\mathrm{v_A}$).

# Appendix A

## Figure Seven
Overview of solar wind velocity observations from the Wind 3DP instrument from 2004 to 2024. 

- *Panel (a):* Time series of solar wind velocity from Wind at 1AU during the time period of this study.
- *Panel (b):* Distribution of velocity measurements from 2004 to 2024 from Wind.
- *Panel (c):* Distribution of velocity measurements during Solar Cycle 24 (January 2008 to December 2019).
- *Panel (d):* Distribution of velocity measurements from 2020 to 2023, covering the time period in this study. In panels (b) to (d), the dashed red and blue lines indicate the 500~{\kms} and 400~{\kms} speed cutoffs respectively. The legend shows the percentage of observations that fall above that threshold for each period.

# Appendix B

## Figures Eight to Eighteen
Comparison of in situ characteristics and footpoint estimates for the modeled SASW streams.

## Figures Nineteen to Twenty Three
Comparison of in situ characteristics and footpoint estimates for the modeled FSW streams.

# Appendix C

## Figure Twenty Four
Overview of error on estimated PFSS footpoints due to noise in the velocity measurement used for ballistic
propagation. We use three streams of different speeds to show how the effect varies based on $v_{sw}$. The top row (panels (a),
(b), (c)) show the overall error on the latitude and longitude of the footpoints. The middle row (panels (d), (e), (f)) show the
error on the source surface longitude as a function of time. The bottom row (panels (g) to (i)) shows the error on the footpoints
compared with the associated AIA image.

## Figure Twenty Five
Comparison of the error on the estimated footpoints due to changing the source surface height for streams of
varying speeds. Panels (a) to (c) show the overall error on the footpoints as a box plot. Panels (d) to (f) show the footpoints
associated with different source surface heights on the associated AIA image.

## Figure Twenty Six
Validation of our choice of source surface height (Rss) for the three different streams we have discussed in this section.
The rows corresponding to the streams from E7, E6, and E10 from top to bottom respectively. The trajectory of Parker at the
source surface is shown in blue (negative polarity) and red (positive polarity) based on the measurements from Parker/FIELDS.
The heliospheric current sheet from the PFSS model is compared with the trajectory and we show the percentage of correct
polarity the model predicted in the legend.

## Figure Twenty Seven
Comparison of the effect of the choice of input magnetogram for the lower PFSS boundary condition on the
estimated footpoints for streams of varied speeds. Panels (a) to (c) show the overall error on the estimated footpoint for each
of the wind streams of interest. Panels (d) to (f) show the footpoints and their error on the associated AIA images.
