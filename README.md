# Product Estimation

from MDN import image_estimates, get_tile_data, get_sensor_bands
sensor = "<OLI, MSI, OLCI, or HICO>"

# Tile should be the output of an atmospheric correction program e.g. SeaDAS
bands, Rrs = get_tile_data("path/to/my/tile.nc", sensor, allow_neg=False) 
chla, idxs = image_estimates(Rrs, sensor=sensor)

# Or, with just random data:
import numpy as np 
random_data = np.random.rand(3, 3, len(get_sensor_bands(sensor)))
chla, idxs  = image_estimates(random_data, sensor=sensor)
```

Or, a .csv file may be given as input, with each row as a single sample. The .csv contents should be only the Rrs values to be estimated on (i.e. no header row or index column).

`python3 -m MDN --sensor <OLI, MSI, OLCI, or HICO> path/to/my/Rrs.csv`

*Note:* The user-supplied input values should correspond to R<sub>rs</sub> (units of 1/sr). 

Current performance is shown in the following scatter plots, with 50% of the data used for training and 50% for testing. Note that the models supplied in this repository are trained using 100% of the <i>in situ</i> data, and so observed performance may differ slightly. 

<p align="center">
	<img src=".res/S2B_benchmark.png?raw=true" height="311" width="721.5"></img>
	<br>
	<br>
	<img src=".res/OLCI_benchmark.png?raw=true" height="311" width="721.5"></img>
</p>



