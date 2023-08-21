'''
QAA v6 
http://www.ioccg.org/groups/Software_OCA/QAA_v6_2014209.pdf 
http://www.ioccg.org/groups/Software_OCA/QAA_v6.xlsm

There are inconsistencies between the pdf definition, and the spreadsheet
implementation. Notably, the spreadsheet uses Rrs throughout rather than
rrs. As well, absorption / scatter values are slightly off, and the a_ph
calculation uses wavelength-specific absorption rather than solely the 443
band.

Here, we use the pdf definition in all cases except the a_ph calculation -
using wavelength absorption prevents higher band a_ph estimates from 
flattening (bands > ~500nm). Where exact bands are requested (e.g. the 
reference band lambda0), this implementation uses the nearest available
band. This impacts the exact absorption/scattering values used, as well as
the calculation of xi with the band difference in the exponent. 

551 is also used to find the closest 555nm band, in order to avoid using 
the 555nm band of MODIS (which is a land-focused band). 
'''

from ...utils import (
    optimize, get_required, set_outputs, 
    loadtxt, to_rrs, closest_wavelength,
)
from ...meta import (
    # g0_Gordon as g0, 
    # g1_Gordon as g1,
    g0_QAA as g0, 
    g1_QAA as g1,
)

from scipy.interpolate import CubicSpline as Interpolate
from pathlib import Path
import numpy as np


@set_outputs(['a', 'ap', 'ag', 'aph', 'apg', 'adg', 'b', 'bbp']) # Define the output product keys
@optimize([]) # Define any optimizable parameters
def model(Rrs, wavelengths, *args, **kwargs):
    wavelengths = np.array(wavelengths)
    required = [443, 490, 550, 670]
    tol = kwargs.get('tol', 21) # allowable difference from the required wavelengths
    Rrs = get_required(Rrs, wavelengths, required, tol) # get values as a function: Rrs(443)
    rrs = get_required(to_rrs(Rrs(None)), wavelengths, required, tol)

    if 'aph' in kwargs.keys():
        a_ph = get_required(kwargs['aph'],wavelengths,required,tol)
    if 'ad' in kwargs.keys():
        a_d = get_required(kwargs['ad'],wavelengths,required,tol)
    if 'ag' in kwargs.keys():
        a_g = get_required(kwargs['ag'],wavelengths,required,tol)


    absorb  = Interpolate( *loadtxt('../IOP/aw').T  )
    scatter = Interpolate( *loadtxt('../IOP/bbw').T )

    get_band   = lambda k: closest_wavelength(k, wavelengths, tol=tol, validate=False)
    functional = lambda v: get_required(v, wavelengths, [], tol)
    
    # Invert rrs formula to find u
    u = functional( (-g0 + (g0**2 + 4 * g1 * rrs(None)) ** 0.5) / (2 * g1) )
    
    lambda0 = get_band(wavelengths)
    a_w = absorb(lambda0)
    b_w = scatter(lambda0)

    a = a_w + a_ph(lambda0) + a_d(lambda0) + a_g(lambda0)
    b = (u(lambda0) * a) / (1 - u(lambda0)) - b_w

    b[b < 0] = 1e-5

    
    # Return all backscattering and absorption parameters
    return {
        'b'  : b + b_w,
        'bbp': b,
    }