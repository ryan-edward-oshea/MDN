'''
Gordon et al. 1988 "A semianalytic radiance model of ocean color": https://doi.org/10.1029/JD093iD09p10909
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
