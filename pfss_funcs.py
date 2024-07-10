
import pfsspy
import sunpy.map
import numpy as np
import astropy.coordinates
import astropy.units as u


def adapt2pfsspy(filepath, #must already exist on your computer
                 rss=2.5, # Source surface height
                 nr=60, # number of radial gridpoiints for model
                 realization="mean", #which slice of the adapt ensemble to choose
                 return_magnetogram = False # switch to true for function to return the input magnetogram
                ):

    # Load the FITS file into memory
    # ADAPT includes 12 "realizations" - model ensembles
    # pfsspy.utils.load_adapt is a specific function that knows
    # how to handel adapt maps
    adaptMapSequence = pfsspy.utils.load_adapt(filepath)
    # If realization = mean, just average them all together
    if realization == "mean" : 
        br_adapt_ = np.mean([m.data for m in adaptMapSequence],axis=0)
        adapt_map = sunpy.map.Map(br_adapt_,adaptMapSequence[0].meta)
    # If you enter an integer between 0 and 11, the corresponding
    # realization is selected
    elif isinstance(realization,int) : adapt_map = adaptMapSequence[realization]
    else : raise ValueError("realization should either be 'mean' or type int ") 
    
    # pfsspy requires that the y-axis be in sin(degrees) not degrees
    # pfsspy.utils.car_to_cea does this conversion
    adapt_map_strumfric = pfsspy.utils.car_to_cea(adapt_map)

    # Option to return the magnetogram
    if return_magnetogram : 
        return adapt_map_strumfric
    # Otherwise run the PFSS Model and return
    else :
        adapt_map_input = sunpy.map.Map(adapt_map_strumfric.data/1e5,
                                        adapt_map_strumfric.meta)
        peri_input = pfsspy.Input(adapt_map_input, nr, rss)
        peri_output = pfsspy.pfss(peri_input)
        return peri_output


# Define function which does the field line tracing
def pfss2flines(pfsspy_output, # pfsspy output object
                nth=18,nph=36, # number of tracing grid points
                rect=[-90,90,0,360], #sub-region of sun to trace (default is whole sun)
                trace_from_SS=False, # if False : start trace from photosphere, 
                                     #if True, start tracing from source surface
                skycoord_in=None, # Use custom set of starting grid poitns
                max_steps = 1000 # max steps tracer should take before giving up
                ) :
    
    # Tracing if grid
    if skycoord_in is None  :
        [latmin,latmax,lonmin,lonmax]=rect
        lons,lats = np.meshgrid(np.linspace(lonmin,lonmax,nph),
                                np.linspace(latmin,latmax,nth)
                                )
        if not trace_from_SS : alt = 1.0*u.R_sun # Trace up from photosphere
        else : alt = po.grid.rss*u.R_sun  # Trace down from ss
        alt = [alt]*len(lons.ravel())
        seeds = astropy.coordinates.SkyCoord(lons.ravel()*u.deg,
                               lats.ravel()*u.deg,
                               alt,
                               frame = pfsspy_output.coordinate_frame)
        
    # Tracing if custom set of points
    else : 
        skycoord_in.representation_type = "spherical"
        seeds = astropy.coordinates.SkyCoord(skycoord_in.lon,
                               skycoord_in.lat,
                               skycoord_in.radius,
                               frame = pfsspy_output.coordinate_frame)
        
    return pfsspy_output.trace(pfsspy.tracing.FortranTracer(max_steps=max_steps),seeds)