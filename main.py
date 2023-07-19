
#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from MDN.product_estimation import main
from MDN.meta import get_sensor_bands
from MDN.parameters import get_args
from MDN import image_estimates, get_sensor_bands,get_tile_data

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import numpy as np

plt.rc('text', usetex=True)


##### Do not change between these statements ########
def chunk_array(input_list,n):
	for i in range(0,len(input_list),n):
		yield input_list[i:i+n]
		
sensor = 'HICO'
min_in_out_val = 1e-6
specified_args = {
            'sensor'   : sensor,
            'removed_dataset' : "South_Africa",

            'plot_loss': False,
            'benchmark' : False,

            #Do not change between runs
            'n_iter'   : 31622,
            'n_mix'    : 5,
            'n_hidden' : 446, 
            'n_layers' : 5, 
            'lr'       : 1e-3,
            'l2'       : 1e-3,
            'epsilon'  : 1e-3,
            'batch'    : 128, 
            'use_HICO_aph':True,
            'n_rounds' : 10,
            'product' : 'aph,chl,tss,pc,ad,ag,cdom',
            'use_gpu' : False,
            'data_loc' : "/home/ryanoshea/in_situ_database/Working_in_situ_dataset/Augmented_Gloria_V3_2/",

            ## argument to set the exact minimum Rrs value and output value (so the Cholesky Decomposition succeeds)
            'min_in_out_val'  : min_in_out_val,
            }

specified_args_wavelengths = {
            'aph_wavelengths' :  get_sensor_bands(specified_args['sensor'] + '-aph'),
            'adag_wavelengths' :  get_sensor_bands(specified_args['sensor'] + '-adag'),
            }

specified_args.update(specified_args_wavelengths)

if not specified_args['benchmark']:
    specified_args['plot_loss'] = False
##### Do not change between these statements ########


####################
#  Change below here
####################

output_selector = 1
# Generates product estimates for random data
if output_selector == 0:
	random_image_data = np.random.rand(3, 3, len(get_sensor_bands(sensor)))
	random_image_data[random_image_data<min_in_out_val] = min_in_out_val
	products, slices  = image_estimates(random_image_data,**specified_args)
	print(products, type(products), slices)
	for product in slices:
		print("Product: ", product," Slice: ",slices[product]," Output shape:",np.shape(products[:,:,slices[product]]))
	

if output_selector == 1:
	tile_path = '/media/ryanoshea/BackUp/Scenes/Erie/HICO/20140908/unedited/H2014251184102.L1B_ISS/out/acolite.nc'
	bands, Rrs = get_tile_data(tile_path, sensor, allow_neg=False)

	inp_list = list(chunk_array(Rrs, 10))
	
	products_list = []
	for i,Rrs_block in enumerate(inp_list):
		print("Rrs block #:", i, ' of', len(inp_list) )
		products, slices  = image_estimates(Rrs_block,**specified_args)
		products_list.append(products)

			
	products = np.concatenate(products_list,axis=0)
	for product in slices:
		print("Product: ", product," Slice: ",slices[product]," Output shape:",np.shape(products[:,:,slices[product]]))
		
	print("Output products shape is:", np.shape(products))
	print("With slices:", slices)
	chla = products[:,:,slices['chl']]
	TSS = products[:,:,slices['tss']]
	cdom = products[:,:,slices['cdom']]
	print(chla,TSS,cdom)

	fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

	chl_im = ax1.imshow(chla,vmin=0.1, vmax=100, cmap='jet', aspect='auto',norm=LogNorm())
	fig.colorbar(chl_im, ax=ax1)
	ax1.set_title('Chl')
	TSS_im = ax2.imshow(TSS,vmin=0.1, vmax=100, cmap='jet', aspect='auto',norm=LogNorm())
	fig.colorbar(TSS_im, ax=ax2)
	ax2.set_title('TSS')
	cdom_im = ax3.imshow(cdom,vmin=0.1, vmax=1, cmap='jet', aspect='auto',norm=LogNorm())
	fig.colorbar(cdom_im, ax=ax3)
	ax3.set_title('CDOM')

	plt.show()
	






