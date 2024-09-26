# -*- coding: utf-8 -*-
"""
File Name:      gloria_processing_utils
Description:    This code file will be used to hand the samples from the GLORIA datasets. The package assumes that the
                data is the raw CSV form used for distribution by the website

Date Created:   August 29th, 2022
"""

import numpy as np
import pandas as pd
import re
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
import warnings
warnings.filterwarnings(action="ignore")

from .parameters import get_args
from .meta import get_sensor_bands
from .Rrs_manipulation.resample import resample, read
from .product_estimation import image_estimates

import os
this_dir, this_filename = os.path.split(__file__)
SRF_FOLDER = os.path.join(this_dir, "Rrs_manipulation", "Rsr")


'Create a dicitonary variable with the short string identifiers and column names of various parameters of interest'
GLORIA_VARIABLE_LOOKUP= {
    'chl': 'Chla',                  # Chlorophyll-s
    'tss': 'TSS',                   # Total Suspended Solids
    'cdom': 'aCDOM440',             # colored DISSOLVED ORGANIC MATERIALS
    'secchi': 'Secchi_depth',       # Secchi Disc Depth
    'pc': 'PC',                     # Phyco-cyanin
    'lat': 'Latitude',              # Latitude of the sample
    'lon': 'Longitude'              # Longitude of the sample
}



GLORIA_WQI_IOP = ['chl', 'tss', 'cdom', 'secchi', 'pc', 'lat', 'lon', 'aph', 'ad', 'ag', 'bbp', 'salinity']

SENSOR_NAME={
    "OLCI":"OLCI",
    "VI": "VIIRS"
}

'List variable with currently supported sensors'
SUPPORTED_SENSORS = ['OLI', 'MSI', 'OLCI', 'S3A', 'S3B', 'HICO', 'PRISMA', 'PACE', 'PACE-sat','HYPER']

def impute_data(x_train, y_train, n_neighbors=5):
    """
    A function to impute the missing values based on the other samples in the training data. This is an example of the
    static imputation used in Saranathan et al. [2024]

    :param x_train: [np.ndarray: nSamples X nBands]
    The training data

    :param y_train: [np.ndarray: nSamples X nParams]
    The test data

    :param n_neighbors: [int >= 0]
    The number of neighbors used in imputation

    :return:

    :param x_train: [np.ndarray: nSamples X nBands]
    The training data - with missing values filled in

    :param y_train: [np.ndarray: nSamples X nParams]
    The test data- with missing values filled in
    """
    assert isinstance(x_train, np.ndarray), "The variable <x_train> must be numpy array"
    assert isinstance(y_train, np.ndarray), "The variable <y_train> must be numpy array"
    assert x_train.shape[0] == y_train.shape[0], "The two variables must have the same number of samples"
    assert isinstance(n_neighbors, int) and (n_neighbors > 0), "The <n_neighbors> variable must be a positive integer"

    "Stack the data together to perform imputations"
    X = np.hstack((x_train, y_train))

    'Set up the data imputer and perform the imputation'
    imputer = KNNImputer(n_neighbors=n_neighbors)
    X = imputer.fit_transform(X)

    n_dim = (-1 *y_train.shape[1])
    return X[:, :n_dim], X[:, n_dim:]


def resample_Rrs(rrs_data, wvl_in, srf_folder=Path(SRF_FOLDER),
                    sensor="OLCI"):
    """
    This function is designed to resample the hyperspectral measurements available in the Augmented GLORIA dataset to
    the spectral resolution of a specific sensor

    :param rrs_data: [np.ndarray: nSamples X nBands]
    A numpy array where each row corresponds to a specific sample and each column corresponds to a specific band

    :param wvl_in: [np.ndarray: nBands]
    The wavlengths associated with the data in the Augmented Gloria dataset

    :param srf_folder: [pathlib.Path]
    The location where the spectral response functions of the various sensors of interest are stored

    :param sensor: [str in SUPPORTED_SENSORS] (Default: 'OLCI')
    This string variable is used to identify the sensor under process currently


    :return:
    rrs_resamp:[np.ndarray: nSamples X nBands]
    A numpy array where each row corresponds to a specific sample and each column corresponds to a specific band

    """
    assert sensor in SUPPORTED_SENSORS, f"The tool does not currently support the Sensor: {sensor}"
    assert srf_folder.is_dir(), f"No directory {srf_folder} found"
    assert (rrs_data.shape[1] == len(wvl_in)), [rrs_data.shape, wvl_in.shape]

    'Get the wavelengths of the sensor of interest'
    args = get_args()
    args.sensor = sensor
    wvl_out = get_sensor_bands(args.sensor, args)

    'Get the spectral response function for the appropriate sensor'
    rsr = read(f'{srf_folder}/%s_rsr.csv' % sensor.split('-')[0])
    rsr[np.isnan(rsr)] = 0
    rsr[rsr < -50] = 0  # Some sheets have e.g. -999 as a placeholder

    bands = rsr[0, 1:]
    waves = rsr[1:, 0]
    rsr = rsr[1:, 1:]

    # Rsr can extend past available hyperspectral data
    valid = np.logical_and(waves <= wvl_in.max(), waves >= wvl_in.min())
    # remove = np.any(rsr[~valid] > 0, axis=0)             # No part of rsr can lie outside
    remove = (rsr[~valid].sum(0) / rsr.sum(0)) > 0.1  # Less than 10% of rsr can lie outside

    print('Bands with responses outside of range [%s, %s]: \n' % (wvl_in.min(), wvl_in.max()),
          list(bands[remove]))
    bands = bands[~remove]
    waves = waves[valid]
    rsr = rsr[:, ~remove][valid]

    'Apply spectral response function to the Augmented GLORIA'
    print('\nCreating data for bands\n', list(bands))
    if not len(bands): return

    rsr_idx = [waves[rsr[:, band] > 0] for band in range(len(bands))]
    rsr_val = [rsr[rsr[:, band] > 0, band] for band in range(len(bands))]
    W = [np.ones(len(rsr_idx[band])) for band in range(len(rsr_val))]

    rrs_data_resamp = np.zeros((rrs_data.shape[0], len(rsr_idx)))
    for ii in tqdm(range(rrs_data.shape[0])):
        rrs_data_resamp[ii, :] = resample(rsr_val, rsr_idx, rrs_data[ii, :], wvl_in, W)

    'Extract the bands of interest for the sensor setup'
    closest_bands = [np.argmin(np.abs(bands - item)) for item in wvl_out]
    rrs_data_resamp = rrs_data_resamp[:, closest_bands]

    return rrs_data_resamp, wvl_out

def get_gloria_samples(sensor="OLCI", bg_var=['chl', 'tss', 'cdom'],
                       gloria_folder=Path("/Volumes/AMS_HDD/Spectral Data/Augmented_Gloria_V3.2/"),
                       rrs_name='AG_Rrs.csv',
                       bg_name='AG_meta_and_lab.csv',
                       flag_name='AugGLORIA_qc_flags.csv',
                       srf_folder=Path('/Users/arunsaranathan/SSAI/Code/Rrs_manipulation/Rsr'),
                       gloria_only = False,
                       rem_flagged=False,
                       pc_name='AG_Conc_PC.csv'
                       ):
    """
    This function can be used to read data in from the Augmented Gloria Dataset provided by Sundar. This dataset will
    have a much larger number of samples than the original in situ dataset

    :param sensor: [str in SUPPORTED_SENSORS] (Default: 'OLCI')
    This string variable is used to identify the sensor under process currently

    :param bg_var: [ [str,..,str] in ['chl', 'tss', 'cdom', 'secchi', 'PC']] (Default: ['tss', 'secchi'])
    This list variable contains string identifiers for the biochemical variables of interest
    [*bg_var= BioGeochemical variables]

    :param gloria_folder:[[pathlib.Path] (Default: <local_address>)
    Location of the folder with the GLORIA dataset

    :param rrs_name: [string] (Default: <local_address>)
    The of the file with the Rrs

    :param bg_name:[string] (Default: <local_address>)
    Name of the file with the biogeochemical variables

    :param flag_name: [string] (Default: GLORIA_qc_flags.xlsx)
    Name of the file with the different flags

    :param srf_folder:[pathlib.Path] (Default: <local_address>)
    Location of the surface response function

    :param gloria_only: [bool] (Default: False)
    This flag used to decide if only GLORIA samples are extracted

    :param rem_flagged: [bool] (Default:False)
    It's a boolean flag which decides an whether spectra flagged in the GLORIA dataset are removed

    :param bg_name:[string] (Default: <local_address>)
    Name of the file with the Phyco cyanin concentrations for the Augemented GLORIA dataset

    :return:
    rrs_resamp: [np.ndarray : nSamples X nBands]
    A numpy array where each row corresponds to a different sample and each column a specific wavelength in the sensor
    of interest

    bg_data_sel: [np.ndarray : nSamples X nParams]
    A numpy array where each row corresponds to a different sample and each column a specific measured IOP from the
    GLORIA dataset

    gid:[np.ndarray: nSamples X 1]
    The gloria id for the samples
    """

    assert sensor in SUPPORTED_SENSORS, f"The tool does not currently support the Sensor: {sensor}"
    assert isinstance(bg_var, list), "The variable <bg_Var> must be a list type variable"
    for item in bg_var:
        assert item in GLORIA_VARIABLE_LOOKUP, f"This tool does not currently support the variable: {item}"
    assert gloria_folder.is_dir(), f"No folder found at {gloria_folder}"
    assert isinstance(rrs_name, str), "The variable <rrs_name> must be a string"
    rrs_location = gloria_folder / rrs_name
    assert rrs_location.is_file(), f"No such file found: {rrs_location}"
    assert isinstance(bg_name, str), "The variable <bg_name> must be a string"
    bg_location = gloria_folder / bg_name
    assert bg_location.is_file(), f"No such file found: {bg_location}"
    assert isinstance(flag_name, str), "The variable <flag_name> msut be string variable"
    flag_loc = gloria_folder / flag_name
    assert srf_folder.is_dir(), f"No directory {srf_folder} found"
    assert isinstance(rem_flagged, bool), "The <rem_flagged> variable must be Boolean"
    if rem_flagged:
        assert flag_loc.is_file(), f"No file found at: {str(flag_loc)}"
    assert isinstance(gloria_only, bool), "The <gloria_only> variable must be Boolean"
    if 'pc' in bg_var:
        pc_location= gloria_folder.joinpath(pc_name)
        assert pc_location.is_file(), f"Cannot find file with PC concentrations at:{pc_location}"

    '------------------------------------------------------------------------------------------------------------------'
    'GET Rrs DATA'
    '------------------------------------------------------------------------------------------------------------------'
    "Get all the spectral data available as part of Augmented GLORIA"
    rrs_data = pd.read_csv(rrs_location, header=0, index_col=None)
    if gloria_only:
        rrs_data = rrs_data.loc[rrs_data['GLORIA_ID'].str.startswith('GID', na=False)]
    'iterating the columns'
    wvl_in = []
    for col in rrs_data.columns:
        wvl_in += [col]

    'Get sample ids'
    gid = np.asarray(rrs_data[wvl_in[0]])
    'Get sample spectra'
    wvl_in = wvl_in[1:-1]
    rrs_data = np.asarray(rrs_data[wvl_in], dtype=float)
    wvl_in =  np.asarray([int(re.findall(r'\d+', item)[0]) for item in wvl_in])
    #rrs_data = np.asarray(rrs_data, dtype=float)[:, 1:-1]

    'Resample the spectral samples in augmented gloria according to the sensor of interest'
    rrs_resamp, wvl_out = resample_Rrs(rrs_data, wvl_in, sensor=sensor, srf_folder=srf_folder)

    '------------------------------------------------------------------------------------------------------------------'
    'GET WATER QUALITY INDICATORS DATA'
    '------------------------------------------------------------------------------------------------------------------'
    'Open the biogeochemical variables file'
    bg_data = pd.read_csv(bg_location, header=0, index_col=None)
    if gloria_only:
        bg_data = bg_data.loc[bg_data['GLORIA_ID'].str.startswith('GID', na=False)]
    bg_data_sel = []
    for item in bg_var:
        if item != 'pc':
            bg_data_sel += [np.asarray(bg_data[GLORIA_VARIABLE_LOOKUP[item]], dtype=float)]
    bg_data_sel = (np.asarray(bg_data_sel)).T

    'Add Phyco-cyanin if needed'
    if 'pc' in bg_var:
        pc_data = pd.read_csv(pc_location, header=0, index_col=None)
        if gloria_only:
            pc_data = pc_data.loc[bg_data['GLORIA_ID'].str.startswith('GID', na=False)]

        bg_data_sel = np.hstack((bg_data_sel, np.asarray(pc_data.iloc[:, 1]).reshape((-1,1))))


    assert rrs_resamp.shape[0] == bg_data_sel.shape[0], f"SHAPE MISMATCH! Got {bg_data_sel.shape[0]} IOPs for " \
                                                        f"{rrs_resamp.shape[0]} samples"

    if rem_flagged:
        gloria_flags = pd.read_csv(flag_loc, header=0, index_col=None)
        if gloria_only:
            gloria_flags = gloria_flags.loc[gloria_flags['GLORIA_ID'].str.startswith('GID', na=False)]
        gloria_flags = np.asarray(gloria_flags.iloc[:, 1:3], dtype=np.float32)
        gloria_flags = np.asarray(1- np.nanmax(gloria_flags, axis=1), dtype=bool)

        rrs_resamp = rrs_resamp[gloria_flags, :]
        bg_data_sel = bg_data_sel[gloria_flags, :]
        gid = gid[gloria_flags]

    return rrs_resamp, bg_data_sel, gid

def get_gloria_trainTestData(sensor='HICO',  out_var=["chl", "tss", "cdom"], save_flag=False, load_exists=True,
                             rand_seed =42, rem_flagged=True, impute_flag=False,
                             gloria_folder = Path("/Volumes/AMS_HDD/Spectral Data/Augmented_Gloria_V2"),
                             flag_name='GLORIA_qc_flags.csv', rrs_name='GLORIA_Rrs.csv',
                             bg_name='GLORIA_meta_and_lab.csv',
                             save_folder = Path("/uncert_hyper_experiments_t2/data_extraction/data_products"),
                             save_name= "trainTest", gloria_only=False, pc_name='AG_Conc_PC.csv',
                             srf_folder=Path(SRF_FOLDER),
                             train_mode=True):
    """
    This function can be used generate a training and test set for the Augmented GLORIA dataset

    :param sensor: [str in SUPPORTED_SENSORS] (Default: 'OLCI')
    This string variable is used to identify the sensor under process currently

    :param out_var: [ [str,..,str] in ['chl', 'tss', 'cdom', 'secchi', 'PC']] (Default: ['tss', 'secchi'])
    This list variable contains string identifiers for the biochemical variables of interest
    [*bg_var= BioGeochemical variables]

    :param save_flag: [bool] (Default:True)
    It's a boolean flag which decides whether the training and test data are stored after extraction

    :param load_exists:[bool] (Default:True)
    It's a boolean flag which decides an existing version of the boolean flag is reloaded or not

    :param rand_seed: [int] (Default: 42 h/t to Douglas Adams)
    The random seed to for the train and test split

    :param rem_flagged: [bool] (Default:True)
    It's a boolean flag which decides an whether spectra flagged in the GLORIA dataset are removed

    :param impute_flag: [bool] (Default:True)
    It's a boolean flag which decides an whether data imputation is to be used on the training data.

    :param gloria_folder: [pathlib.Path] (Default: /Volumes/AMS_HDD/Spectral Data/Augmented_Gloria_V1/)
    The location of the folder with the Augmented GLORIA data

    :param rrs_name: [string] (Default: <local_address>)
    The of the file with the Rrs

    :param bg_name:[string] (Default: <local_address>)
    Name of the file with the biogeochemical variables

    :param flag_name: [string] (Default: GLORIA_qc_flags.xlsx)
    Name of the file with the different flags

    :param save_folder: (pathlib.Path) (Default: /Users/arunsaranathan/SSAI/Code/uncert_hyper_experiments_t2/data_creation/data_products)
    The location of the file with the GLORIA_qc_flags

    :param gloria_only: [bool] (Default: False)
    This flag used to decide if only GLORIA samples are extracted

    :param train_mode: [bool] (Default: True)
    This flag used to decide whether data is split into training and test data


    :return:
    x_train: [np.ndarray : nSamples X nBands]
    Training set independent variables, a numpy array where each row corresponds to a different sample and each column
    a specific wavelength in the sensor of interest

    y_train: [np.ndarray : nSamples X nParams]
    Training set dependent variables, a numpy array where each row corresponds to a different sample and each column a
    specific measured IOP from the GLORIA dataset

    x_test: [np.ndarray : nSamples X nBands]
    Test set independent variables, a numpy array where each row corresponds to a different sample and each column
    a specific wavelength in the sensor of interest

    y_test: [np.ndarray : nSamples X nParams]
    Test set dependent variables, a numpy array where each row corresponds to a different sample and each column a
    specific measured IOP from the GLORIA dataset

    gid: [np.ndarray: nSamples]
    The ID of each sample in the GLORIA dataset
    """
    assert sensor in SUPPORTED_SENSORS, f"The tool does not currently support the Sensor: {sensor}"
    assert isinstance(out_var, list), "The variable <out_Var> must be a list type variable"
    for item in out_var:
        assert item in GLORIA_VARIABLE_LOOKUP, f"This tool does not currently support the output variable: {item}"
    assert isinstance(save_flag, bool), "The <save_flag> variable must be Boolean"
    assert isinstance(load_exists, bool), "The <load_exits> variable must be Boolean"
    assert isinstance(impute_flag, bool), "The <impute_flag> variable must be Boolean"
    assert isinstance(rem_flagged, bool), "The <rem_flagged> variable must be Boolean"
    if rem_flagged:
        assert isinstance(flag_name, str), "The variable <flag_name> msut be string variable"
        flag_loc = gloria_folder / flag_name
        assert flag_loc.is_file(), f"No file found at: {str(flag_loc)}"
    assert isinstance(gloria_only, bool), "The <gloria_only> variable must be Boolean"
    assert isinstance(train_mode, bool), "The <train_mode> variable must be Boolean"

    'Add specific file name to create the base address of the file with the data'
    if not impute_flag:
        base_address = save_folder / "no_impute" / f"{save_name}_{sensor}_noImpute.npz"
    else:
        base_address = save_folder / f"{save_name}_{sensor}.npz"

    if not train_mode:
        temp = str(base_address)
        temp = temp.replace(".npz", "_fullData.npz")
        base_address = Path(temp)

    'IF specified load the existing data variables'
    if load_exists:
        'Check if a preset version exists'
        if base_address.is_file():
            data = np.load(str(base_address), allow_pickle=True)
            if train_mode:
                'Check if variables of interest are present'
                if (all(data['product'] == out_var)) and (data['rem_flagged'] == rem_flagged):
                    x_train, y_train, x_test, y_test, gid = data['x_train'], data['y_train'], data['x_test'], data['y_test'], data['gid']

                    return x_train, y_train, x_test, y_test, gid
            else:
                if (all(data['product'] == out_var)) and (data['rem_flagged'] == rem_flagged):
                    x_data, y_data, gid = data['x_data'], data['y_data'], data['gid']

                    return x_data, y_data, gid




    'It not a pre-loaded one get the data'
    x_data, y_data, gid = get_gloria_samples(sensor=sensor, bg_var=out_var, gloria_folder=gloria_folder,
                                        rrs_name=rrs_name, bg_name=bg_name, rem_flagged=rem_flagged,
                                        gloria_only=gloria_only, pc_name=pc_name, srf_folder=srf_folder,
                                        flag_name=flag_name,)


    'Replace nan Rrs by 0'
    x_data = np.nan_to_num(x_data)
    x_data[x_data <= 1.e-6]= 1e-6
    y_data[y_data <= 1.e-6]= np.nan

    'Split into training and test data'
    if train_mode:
        x_train, x_test, y_train, y_test, gid_train, gid_test = train_test_split(x_data, y_data, gid, test_size=0.5, random_state=rand_seed)


    'Impute missing values --- STATIC GOLD STANDARD'
    if impute_flag:
        if train_mode:
            x_train, y_train = impute_data(x_train, y_train)
        else:
            x_data, y_data = impute_data(x_data, y_data)

    if save_flag:
        'Make parents if they do not exist'
        (base_address.parents[0]).mkdir(parents=True, exist_ok=True)
        if train_mode:
            'Save train and test data'
            np.savez_compressed(base_address, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
                                product=out_var, rem_flagged=rem_flagged, gid_train=gid_train, gid_test=gid_test)
        else:
            'Save train and test data'
            np.savez_compressed(base_address, x_data=x_data, y_data=y_data, product=out_var,
                                rem_flagged=rem_flagged, gid=gid)


    if train_mode:
        return x_train, y_train, x_test, y_test, gid_train, gid_test
    else:
        return x_data, y_data, gid


if __name__ == "__main__":
    "To test the functionality set the gloria location as the folder with the unzipped GLORIA data"
    gloria_folder= Path("C:\\Users\\asaranat\\OneDrive - NASA\\Code\\oceanOptics_tutorials\\data\\GLORIA_2022\\")
    x_data, y_data, gid_data = get_gloria_trainTestData(sensor='OLCI', out_var=["chl", "tss", "cdom"],
                                                        save_flag=False, load_exists=False, rand_seed=42,
                                                        rem_flagged=True, gloria_folder=gloria_folder,
                                                        flag_name='GLORIA_qc_flags.csv', rrs_name='GLORIA_Rrs.csv',
                                                        bg_name='GLORIA_meta_and_lab.csv', gloria_only=True,
                                                        train_mode=False)

    'Generate estimates for these samples using a pretrained MDN'
    est_values = image_estimates(x_data.reshape((1, -1)), sensor="OLCI")

    print('finished')
