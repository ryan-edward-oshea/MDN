'''
Copy this folder into a product directory, and rename it to be the algorithm name
'''

from ...utils import get_required, optimize, loadtxt, to_rrs, closest_wavelength
from scipy.interpolate import CubicSpline as Interpolate
from ....benchmarks.multiple.QAA.model import model as QAA 

import numpy as np
# Define any optimizable parameters
@optimize(['a', 'b','c'],[0.68,0.84,0.24])
def model(Rrs, wavelengths, *args, **kwargs):
	QAA_output = QAA(Rrs, wavelengths,args,kwargs)
	b_b = QAA_output['b']

	wavelengths = np.asarray(wavelengths)
	wavelength_location_778_75 = np.argmin((abs(wavelengths - 778.75))) #finds the nearest wavelength


	bb_slice = slice(wavelength_location_778_75,wavelength_location_778_75+1)
	b_b_sliced = b_b[:,bb_slice]
	b_b_742 = b_b_sliced

	required = [620, 665, 709]
	tol = kwargs.get('tol', 5) # allowable difference from the required wavelengths
	Rrs = get_required(Rrs, wavelengths, required, tol) # get values as a function: Rrs(443)
	rrs = get_required(to_rrs(Rrs(None)), wavelengths, required, tol)

	# Convert rrs to R(0-) by multiplying by pi, assuming lambertian, though these terms would cancel out through division, no matter what the true Q value was
	# R_0_minus = rrs*np.pi

	absorb  = Interpolate( *loadtxt('../IOP/aw').T  )
	scatter = Interpolate( *loadtxt('../IOP/bbw').T )

	get_band   = lambda k: closest_wavelength(k, wavelengths, tol=tol)

	a = kwargs.get('a', 0.68)
	b = kwargs.get('b', 0.84)
	c = kwargs.get('c', 0.24)

	gamma = a
	delta = b
	eta = c

	lambda0 = get_band(620)
	lambda1 = get_band(665)
	lambda2 = get_band(709)

	a_w_620 = absorb(lambda0)
	a_w_665 = absorb(lambda1)
	a_w_709 = absorb(lambda2)

	a_pc_star_620_specific_absorption_PC = .0095 # Average value for lake Loosdrecht
	a_chl_665 = rrs(709)/rrs(665)*(a_w_709 + b_b_742) - b_b_742 - a_w_665*pow(gamma,-1) # Equation 3

	# This value of a_ph* the specific absorption coefficient is highly varible, but we will use values from Simis et al.
	a_pc_620 =  rrs(709)/rrs(620) * (a_w_709+b_b_742) - b_b_742 -a_w_620*pow(delta,-1) - (eta*a_chl_665) #Equation 5
	PC = a_pc_620/a_pc_star_620_specific_absorption_PC # Equation 6
	return PC
