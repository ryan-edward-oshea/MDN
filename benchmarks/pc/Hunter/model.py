'''
Copy this folder into a product directory, and rename it to be the algorithm name
'''

from ...utils import get_required, optimize

# Define any optimizable parameters
# paper: Hyperspectral remote sensing of cyanobacterial pigments as indicators for cell populations and toxins in eutrophic lakes
@optimize(['a','b'], [-4.96,266])
def model(Rrs, wavelengths, *args, **kwargs):
	required = [600, 615, 725]
	tol = kwargs.get('tol', 5) # allowable difference from the required wavelengths
	Rrs = get_required(Rrs, wavelengths, required, tol) # get values as a function: Rrs(443)

	# Set default values for these parameters
	a = kwargs.get('a', -4.96)
	b = kwargs.get('b', 266)

	#Hunter PC
	hunter = a+b*Rrs(725)*(1/Rrs(615)-1/Rrs(600))
	return hunter