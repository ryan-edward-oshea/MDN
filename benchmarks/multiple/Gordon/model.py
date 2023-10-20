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
    g0_Gordon as g0, 
    g1_Gordon as g1,
    # g0_QAA as g0, 
    # g1_QAA as g1,
)

from scipy.interpolate import CubicSpline as Interpolate
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


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

    aph = a_ph(lambda0)
    ad  = a_d(lambda0)
    ag  = a_g(lambda0)
    Rrs_def = Rrs(lambda0)
    rrs_def = rrs(lambda0)

    #Make plots of Rrs, aph, ad, ag, a and b
    if False:
        for i in range(np.shape(aph)[0]):
            # plt.figure()
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
            ax1.plot(wavelengths,np.squeeze(aph[i,:].T), 'r',label='aph')
            ax1.plot(wavelengths,np.squeeze(ad[i,:].T),  'g',label='ad')
            ax1.plot(wavelengths,np.squeeze(ag[i,:].T),  'b',label='ag')
            ax1.plot(wavelengths,np.squeeze(a[i,:].T),   'k',label='a')
            ax1.legend(loc="upper right")
            
            ax1.set_title('Absorption')
            
            ax2.plot(wavelengths,np.squeeze(Rrs_def[i,:].T),   'r',label='Rrs')
            ax2.plot(wavelengths,np.squeeze(rrs_def[i,:].T),   'k',label='rrs')
            ax2.legend(loc="upper right")
            ax2.set_title('Rrs')
    
            ax3.plot(wavelengths,np.squeeze(b[i,:].T),   'k',label='bbp')
            ax3.plot(wavelengths,np.squeeze(((u(lambda0) * a) / (1 - u(lambda0)))[i,:].T),   'r',label='b')
            ax3.plot(wavelengths,np.squeeze((u(lambda0))[i,:].T),   'g',label='u')
            ax3.plot(wavelengths,np.squeeze((u(lambda0)* a)[i,:].T),   'c',label='u*a')
    
            ax3.set_title('b')
    
            ax3.legend(loc="upper right")
            
            fig.savefig(f'bbp/meta/Gordon_absorption_{i}.png')
            plt.close('all')
        
    
    
    
    # Return all backscattering and absorption parameters
    return {
        'b'  : b + b_w,
        'bbp': b,
    }