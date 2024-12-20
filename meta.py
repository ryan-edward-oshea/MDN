import numpy as np 

SENSOR_LABEL = { # http://www.ioccg.org/sensors/seawifs.html
	'CZCS'   : 'Nimbus-7',
	'TM'     : 'Landsat-5',
	'ETM'    : 'Landsat-7',
	'OLI'    : 'Landsat-8', 
	'OSMI'   : 'Arirang-1',
	'POLDER' : 'POLDER',
    'GOCI'   : 'GOCI',
	'AER'    : 'AERONET',
	'OCTS'   : 'ADEOS-1',
	'SEAWIFS': 'OrbView-2',
	'VI'     : 'Suomi-NPP', 
	'MOS'    : 'MOS-1',
	'MOD'    : 'MODIS',
	'MODA'   : 'MODIS-Aqua',
	'MODT'   : 'MODIS-Terra',
	'MSI'    : 'Sentinel-2', 
	'S2A'    : 'Sentinel-2A', 
	'S2B'    : 'Sentinel-2B', 
	'S3A'    : 'Sentinel-3A', 
	'S3B'    : 'Sentinel-3B', 
	'OLCI'   : 'Sentinel-3',
	'MERIS'  : 'Envisat-1',
	'HICO'   : 'HICO',
	'PRISMA' : 'PRISMA',
	'PACE'   : 'PACE',
	'HYPER'  : '1nm Hyperspectral',
}

def get_sensor_label(sensor):
	sensor, *ext = sensor.split('-')
	assert(sensor in SENSOR_LABEL), f'Unknown sensor: {sensor}'
	
	label = SENSOR_LABEL[sensor]
	if 'pan' in ext:
		label += '+Pan'
	return label 

# --------------------------------------------------------------

SENSOR_BANDS = {
	'CZCS'      : [     443,           520, 550,                     670                                   ],
	'TM'        : [               490,      560,                     660                                   ],
	'ETM'       : [               483,      560,                     662                                   ],
	'ETM-pan'   : [               483,      560,                     662,                               706],
	'OLI'       : [     443,      482,      561,                     655                                   ],
	'OLI-SOLID' : [     443,      482,      561,                     655                                   ],
	'OLI-pan'   : [     443,      482,      561,   589,              655,                                  ],
	'OLI-full'  : [     443,      482,      561,                     655,                               865],
	'OLI-nan'   : [     443,      482,      561,   589,              655,                               865],
	'OLI-rho'   : [     443,      482,      561,                     655,                                   865, 1609],
	'OSMI'      : [412, 443,      490,      555,                                              765          ],
	'POLDER'    : [     443,      490,      565,                     670,                     765          ],
	'AER'       : [412, 442,      490,      560,                     668                                   ],
	'OCTS'      : [412, 443,      490, 520, 565,                     670,                     765          ],
	'SEAWIFS'   : [412, 443,      490, 510, 555,                     670,                     765          ],
	'VI'        : [410, 443,      486,      551,                     671,           745                    ], 
	'VI-sat'    : [     443,      486,      551,                     671,           745                    ],
    'VI-sat_no_NIR'    : [     443,      486,      551,                     671,                           ], 
	'VI-SOLID'  : [410, 443,      486,      551,                     671,                                  ], 
	'MOS'       : [408, 443,      485, 520, 570,      615, 650,      685,           750                    ],
	'MOD'       : [412, 443, 469, 488, 531, 551, 555,      645, 667, 678,           748                    ],
	'MOD-IOP'   : [412, 443, 469, 488, 531, 551, 555,      645, 667, 678,                                  ],
	'MOD-sat'   : [412, 443, 469, 488, 531, 551, 555,      645, 667, 678,                                  ],
	'MOD-sat'   : [     443, 469, 488, 531, 551, 555,      645,                     748                    ],
    'MODA-sat_no_NIR'   : [     443, 469, 488, 531, 551, 555,      645,                                    ],
	'MOD-poly'  : [412, 443,      488, 531, 551,                667, 678,           748                    ],
	'MOD-SOLID' : [412, 443,      488,      551,                667, 678,                                  ],
	'MSI'       : [     443,      490,      560,                     665,      705, 740,                783],
	'MSI-SOLID' : [     443,      490,      560,                     665,      705,                        ],
	'MSI-rho'   : [     443,      490,      560,                     665,      705, 740,                783, 865],
	'OLCI'      : [411, 442,      490, 510, 560,      619,      664, 673, 681, 708, 753,                778],
	'OLCI-rho'  : [411, 442,      490, 510, 560,      619,      664, 673, 681, 708, 753,           768, 778, 865, 884, 1016],
	'OLCI-full' : [411, 442,      490, 510, 560,      619,      664, 673, 681, 708, 753, 761, 764, 767, 778],
	'OLCI-poly' : [411, 442,      490, 510, 560,      619,      664,      681, 708, 753,                778],
	'OLCI-sam'  : [411, 442,      490, 510, 560,      619,      664, 673, 681, 708, 753,                   ],
	'OLCI-SOLID': [411, 442,      490, 510, 560,      619,      664, 673, 681,                             ],
    'OLCI-sat'      : [     442,      490, 510, 560,      619,      664, 673, 681, 708,                    ],
	'MERIS'     : [412, 442,      490, 510, 560,      620,      665,      681, 708, 753, 760,           778],

	'HICO-full' : [409, 415, 421, 426, 432, 438, 444, 449, 455, 461, 467, 472, 478, 484, 490, 495, 501, 507, 
				   512, 518, 524, 530, 535, 541, 547, 553, 558, 564, 570, 575, 581, 587, 593, 598, 604, 610, 
				   616, 621, 627, 633, 638, 644, 650, 656, 661, 667, 673, 679, 684, 690, 696, 701, 707, 713, 
				   719, 724, 730, 736, 742, 747, 753, 759, 764, 770, 776, 782, 787],
	'HICO-chl'  : [501, 507, 512, 518, 524, 530, 535, 541, 547, 553, 558, 564, 570, 575, 581, 587, 593, 598, 
				   604, 610, 616, 621, 627, 633, 638, 644, 650, 656, 661, 667, 673, 679, 684, 690, 696, 701, 
				   707, 713],
	'HICO-IOP'  : [409, 415, 421, 426, 432, 438, 444, 449, 455, 461, 467, 472, 478, 484, 490, 495, 501, 507, 
				   512, 518, 524, 530, 535, 541, 547, 553, 558, 564, 570, 575, 581, 587, 593, 598, 604, 610, 
				   616, 621, 627, 633, 638, 644, 650, 656, 661, 667, 673, 679, 684, 690], # absorption data becomes negative > 690nm
	'HICO'      : [409, 415, 421, 426, 432, 438, 444, 449, 455, 461, 467, 472, 478, 484, 490, 495, 501, 507, 
				   512, 518, 524, 530, 535, 541, 547, 553, 558, 564, 570, 575, 581, 587, 593, 598, 604, 610, 
				   616, 621, 627, 633, 638, 644, 650, 656, 661, 667, 673, 679, 684, 690, 696, 701, 707, 713, 719, 724],
    
    
    'HICO-aph'   : [409, 415, 421, 426, 432, 438, 444, 449, 455, 461, 467, 472, 478, 484, 490, 495, 501, 507, 
				   512, 518, 524, 530, 535, 541, 547, 553, 558, 564, 570, 575, 581, 587, 593, 598, 604, 610, 
				   616, 621, 627, 633, 638, 644, 650, 656, 661, 667, 673, 679, 684, 690],
    
    'PRISMA'     : [411, 419, 427, 434, 441, 449, 456, 464, 471, 478, 485, 493, 500, 507, 515, 523, 530,
                   538, 546, 554, 563, 571, 579, 588, 596, 605, 614, 623, 632, 641, 651, 660, 670, 679, 689, 
                   699, 709, 719,],
    
	'PRISMA-aph' : [411, 419, 427, 434, 441, 449, 456, 464, 471, 478, 485, 493, 500, 507, 515, 523, 530,
                   538, 546, 554, 563, 571, 579, 588, 596, 605, 614, 623, 632, 641, 651, 660, 670, 679, 689,],
    
    'PRISMA-adag': [412, 440, 488, 532, 560, 630, 650, 676, ], 
    
    'HICO-adag'  : [412, 440, 488, 532, 560, 630, 650, 676,],
    
    'PACE'       : [410, 415, 420, 425, 430, 435, 440, 445, 450, 455, 460, 465, 470, 475, 480, 485, 490, 495,
                   500, 505, 510, 515, 520, 525, 530, 535, 540, 545, 550, 555, 560, 565, 570, 575, 580, 585, 
                   590, 595, 600, 605, 610, 615, 620, 625, 630, 635, 640, 645, 650, 655, 660, 665, 670, 675, 
                   680, 685, 690, 695, 700, 705, 710, 715, 720, ],
    
    'PACE-rho'   : [410, 415, 420, 425, 430, 435, 440, 445, 450, 455, 460, 465, 470, 475, 480, 485, 490, 495,
                   500, 505, 510, 515, 520, 525, 530, 535, 540, 545, 550, 555, 560, 565, 570, 575, 580, 585, 
                   590, 595, 600, 605, 610, 615, 620, 625, 630, 635, 640, 645, 650, 655, 660, 665, 670, 675, 
                   680, 685, 690, 695, 700, 705, 710, 715, 720, ],
    
    'PACE-sat'   : [410, 415, 420, 425, 430, 435, 440, 445, 450, 455, 460, 465, 470, 475, 480, 485, 490, 495,
                    500, 505, 510, 515, 520, 525, 530, 535, 540, 545, 550, 555, 560, 565, 570, 575, 580, 585, 
                    590, 595, 600, 605, 610, 615,      625, 630, 635, 640, 645, 650, 655, 660, 665, 670, 675, 
                    680,               700, 705, 710, 715,       ],
                    
    'PACE-sat-rho': [410, 415, 420, 425, 430, 435, 440, 445, 450, 455, 460, 465, 470, 475, 480, 485, 490, 495,
                    500, 505, 510, 515, 520, 525, 530, 535, 540, 545, 550, 555, 560, 565, 570, 575, 580, 585, 
                    590, 595, 600, 605, 610, 615,      625, 630, 635, 640, 645, 650, 655, 660, 665, 670, 675, 
                    680,               700, 705, 710, 715,       ],
                    
    'PACE-aph'   : [410, 415, 420, 425, 430, 435, 440, 445, 450, 455, 460, 465, 470, 475, 480, 485, 490, 495,
                    500, 505, 510, 515, 520, 525, 530, 535, 540, 545, 550, 555, 560, 565, 570, 575, 580, 585,
                    590, 595, 600, 605, 610, 615, 620, 625, 630, 635, 640, 645, 650, 655, 660, 665, 670, 675,
                    680, 685, 690,],
    
    'PACE-adag' : [410, 440, 490, 530, 560, 630, 650, 675,], 
    
	'HYPER'     : list(range(400, 799)),
	'HYPER-nan' : list(range(400, 801)), 
    
    # 'PACE'      : [403, 405, 408, 410, 413, 415, 418, 420, 423, 425, 428, 430, 433, 435, 438, 440, 443, 445, 448, 
    #                450, 453, 455, 458, 460, 463, 465, 468, 470, 473, 475, 478, 480, 483, 485, 488, 490, 493, 495, 
    #                498, 500, 503, 505, 508, 510, 513, 515, 518, 520, 523, 525, 528, 530, 533, 535, 538, 540, 543, 
    #                545, 548, 550, 553, 555, 558, 560, 563, 565, 568, 570, 573, 575, 578, 580, 583, 585, 588, 590, 
    #                593, 595, 598, 600, 603, 605, 608, 610, 613, 615, 618, 620, 623, 625, 628, 630, 633, 635, 638, 
    #                640, 643, 645, 648, 650, 653, 655, 658, 660, 663, 665, 668, 670, 673, 675, 678, 680, 683, 685, 
    #                688, 690, 693, 695, 698, 700, 703, 705, 708, 710, 713, 715, 718, 720, 723, 725, 728, 730, 733, 
    #                735, 738, 740, 743, 745, 748, 750, 753, 755, 758, 760, 763, 765, 768, 770, 773, 775, 778, 780, 
    #                783, 785, 788, 790, 793, 795, 798, 800],
    
    # 'PACE'      : [410, 413, 415, 418, 420, 423, 425, 428, 430, 433, 435, 438, 440, 443, 445, 448, 
    #                450, 453, 455, 458, 460, 463, 465, 468, 470, 473, 475, 478, 480, 483, 485, 488, 490, 493, 495, 
    #                498, 500, 503, 505, 508, 510, 513, 515, 518, 520, 523, 525, 528, 530, 533, 535, 538, 540, 543, 
    #                545, 548, 550, 553, 555, 558, 560, 563, 565, 568, 570, 573, 575, 578, 580, 583, 585, 588, 590, 
    #                593, 595, 598, 600, 603, 605, 608, 610, 613, 615, 618, 620, 623, 625, 628, 630, 633, 635, 638, 
    #                640, 643, 645, 648, 650, 653, 655, 658, 660, 663, 665, 668, 670, 673, 675, 678, 680, 683, 685, 
    #                688, 690, 693, 695, 698, 700, 703, 705, 708, 710, 713, 715, 718, 720, 723,],

    'PACE'      : [410, 415, 420, 425, 430, 435, 440, 445,  
                   450, 455, 460, 465, 470, 475, 480, 485, 490, 495, 
                   500, 505, 510, 515, 520, 525, 530, 535, 540,  
                   545, 550, 555, 560, 565, 570, 575, 580, 585, 590, 
                   595, 600, 605, 610, 615, 620, 625, 630, 635,  
                   640, 645, 650, 655, 660, 665, 670, 675, 680, 685, 
                   690, 695, 700, 705, 710, 715, 720, ],
    
    'PACE-sat'   : [410, 415, 420, 425, 430, 435, 440, 445,  
                   450, 455, 460, 465, 470, 475, 480, 485, 490, 495, 
                   500, 505, 510, 515, 520, 525, 530, 535, 540,  
                   545, 550, 555, 560, 565, 570, 575, 580, 585, 590, 
                   595, 600, 605, 610, 615,      625, 630, 635,  
                   640, 645, 650, 655, 660, 665, 670, 675, 680,   
                             700, 705, 710, 715,       ],
    
    # 'PACE-aph'      : [410, 413, 415, 418, 420, 423, 425, 428, 430, 433, 435, 438, 440, 443, 445, 448, 
    #                450, 453, 455, 458, 460, 463, 465, 468, 470, 473, 475, 478, 480, 483, 485, 488, 490, 493, 495, 
    #                498, 500, 503, 505, 508, 510, 513, 515, 518, 520, 523, 525, 528, 530, 533, 535, 538, 540, 543, 
    #                545, 548, 550, 553, 555, 558, 560, 563, 565, 568, 570, 573, 575, 578, 580, 583, 585, 588, 590, 
    #                593, 595, 598, 600, 603, 605, 608, 610, 613, 615, 618, 620, 623, 625, 628, 630, 633, 635, 638, 
    #                640, 643, 645, 648, 650, 653, 655, 658, 660, 663, 665, 668, 670, 673, 675, 678, 680, 683, 685, 
    #                688, 690,],

    'PACE-aph'      : [410, 415, 420, 425, 430, 435, 440, 445, 450, 455, 460, 465, 470, 475, 480, 485, 490, 495, 
                       500, 505, 510, 515, 520, 525, 530, 535, 540, 545, 550, 555, 560, 565, 570, 575, 580, 585, 
                       590, 595, 600, 605, 610, 615, 620, 625, 630, 635, 640, 645, 650, 655, 660, 665, 670, 675, 
                       680, 685, 690,],
    
	'PACE-adag' : [412, 440, 488, 532, 560, 630, 650, 676,], #[409, 438, 490, 530, 558, 627, 650, 673,], #Twardowski 2004 AC9: 412, 440, 488, 532,555 or 560, 630, 650, 676, and 715  # Previous [444, 490, 547, 593, 644,] # 409, 444, 490,  535, 581, 627


}

duplicates = {
	'MOD' : ['MODA', 'MODT'],
	'MSI' : ['S2A', 'S2B'],
	'OLCI': ['S3A', 'S3B'],

}

# Add duplicate sensors
for sensor in list(SENSOR_BANDS.keys()):
	for sensor2, dups in duplicates.items():
		if sensor2  in sensor:
			for dup in dups:
				SENSOR_BANDS[sensor.replace(sensor2, dup)] = SENSOR_BANDS[sensor]


def get_sensor_bands(sensor, args=None):
	assert(sensor in SENSOR_BANDS), f'Unknown sensor: {sensor}'
	bands = set()
	if args is not None:

		# Specific bands can be passed via args in order to override those used
		if hasattr(args, 'bands'):
			return np.array(args.bands.split(',') if isinstance(bands, str) else args.bands)

		# The provided bands can change if satellite bands with certain products are requested
		elif args.sat_bands:
			product_keys = {
				'chl' : ['chl'],
				'IOP' : ['aph', 'a*ph', 'ag', 'ad'],
			}

			for key, products in product_keys.items():
				for product in args.product.split(','):
					if (f'{sensor}-{key}' in SENSOR_BANDS) and (product in products): 
						bands |= set(SENSOR_BANDS[f'{sensor}-{key}'])

			if len(bands) == 0 and f'{sensor}-sat' in SENSOR_BANDS:
				sensor = f'{sensor}-sat'

	if len(bands) == 0:
		bands = SENSOR_BANDS[sensor]
	return np.sort(list(bands)) 	

# --------------------------------------------------------------

# Ancillary parameters for certain models
ANCILLARY = [
	'humidity',    # Relative humidity (%)
	'ice_frac',    # Ice fraction (0=no ice, 1=all ice)
	'no2_frac',    # Fraction of tropospheric NO2 above 200m
	'no2_strat',   # Stratospheric NO2 (molecules/cm^2)
	'no2_tropo',   # Tropospheric NO2 (molecules/cm^2)
	'ozone',       # Ozone concentration (cm)
	'pressure',    # Surface pressure (millibars)
	'mwind',       # Meridional wind speed @ 10m (m/s)
	'zwind',       # Zonal wind speed @ 10m (m/s)
	'windangle',   # Wind direction @ 10m (degree)
	'windspeed',   # Wind speed @ 10m (m/s)
	'scattang',    # Scattering angle (degree)
	'senz',        # Sensor zenith angle (degree)
	'sola',        # Solar azimuth angle (degree)
	'solz',        # Solar zenith angle (degree)
	'water_vapor', # Precipitable water vapor (g/cm^2)
	'time_diff',   # Difference between in situ measurement and satellite overpass (in situ prior to overpass = negative)
]

# Ancillary parameters which are periodic (e.g. 0 degrees == 360 degrees)
PERIODIC = [
	'windangle',
]
