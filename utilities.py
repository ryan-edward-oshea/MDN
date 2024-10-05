# -*- coding: utf-8 -*-
"""
File Name:      utilities
Description:    This code file contains a set of supporting functions that are needed to generate the predictions and
                uncertainty estimations available for a specific models.

Date Created:   September 2nd, 2024
"""

import numpy as np
import sys
from tqdm import tqdm
from tqdm import tqdm

from .product_estimation import get_estimates
from .parameters import get_args
from .meta import get_sensor_bands
from .utils import mask_land
from .uncertainty_package_final.uncert_support_lib import get_sample_uncertainity

'Base properties for imshow'
ASPECT = 'auto'
cmap = 'jet'

def get_mdn_preds(test_x, args=None, sensor="OLCI", products="chl", mode="point", model_type='production', verbose=False):
    """
    This function is used to generate estimates from pre-trained MDN

    Inputs
    ------
        a) test_x: A numpy array with the data on which we want to generate the predictions. Each row of this matrix
                  corresponds to a test spectral samples.
        b) args: The arguments for the MDN. [Default: None]
        c) sensor: The sensor for which predictions are needed. This argument in only used if args is not provided.
                 [Default: "OLCI"]
        d) products: The products which will be predicted by the model. This argument is only used if args is not
                   provided. [Default: "chl"]
        e) mode:   A flag that signifies whether full MDN output suite is to be produced or just point estimates. The
                 modes currently supported are 'full' and 'point'
        f) model_type:  A flag that signifies whether we use a model trained on the full GLORIA data or a reduced test
                        set
        g) verbose:   A boolean flag that controls how much information is printed out to the console

    Outputs
    -------
        a) mdn_preds: In the 'full' mode the output predictions is a list which contains all the different output compo-
        nents of the MDN output. In the point mode it generates a numpy array that is the best estimate of the output
        for each sample (row) for each WQI (column)
        b) op_slices: A dictionary indicating the slices corresponding to the different output variables.
    """

    assert  model_type in ['production', 'testing'], f"Currently the toolbox only supports two model types" \
                                                     f" 'production' and 'testing'. Instead got '{model_type}'."

    if args == None:
        print("MDN model settings not provided by user!")
        print(f"Looking for model at {sensor} resolution predicting {products}!!")

        'Get the default arguments'
        kwargs = {'product': products,
                  'sat_bands': True if products == 'chl,tss,cdom,pc' else False,
                  # 'data_loc'     : 'D:/Data/Insitu',
                  'model_loc': "Weights" if model_type == 'production' else "Weights_test",
                  'sensor': sensor}

        if sensor == 'PRISMA' or sensor == 'HICO' or sensor == 'PACE' and products == 'aph,chl,tss,pc,ad,ag,cdom':
            min_in_out_val = 1e-6
            kwargs = {
                'allow_missing': False,
                'allow_nan_inp': False,
                'allow_nan_out': True,

                'sensor': sensor,
                'removed_dataset': "South_Africa,Trasimeno" if sensor == "PRISMA" else "South_Africa",
                'filter_ad_ag': False,
                'imputations': 5,
                'no_bagging': False,
                'plot_loss': False,
                'benchmark': False,
                'sat_bands': False,
                'n_iter': 31622,
                'n_mix': 5,
                'n_hidden': 446,
                'n_layers': 5,
                'lr': 1e-3,
                'l2': 1e-3,
                'epsilon': 1e-3,
                'batch': 128,
                'use_HICO_aph': True,
                'n_rounds': 10,
                'product': 'aph,chl,tss,pc,ad,ag,cdom',
                'use_gpu': False,
                'data_loc': "/home/ryanoshea/in_situ_database/Working_in_situ_dataset/Augmented_Gloria_V3_2/",
                'use_ratio': True,
                'min_in_out_val': min_in_out_val,

            }

            specified_args_wavelengths = {
                'aph_wavelengths': get_sensor_bands(kwargs['sensor'] + '-aph'),
                'adag_wavelengths': get_sensor_bands(kwargs['sensor'] + '-adag'),
            }

        args = get_args(kwargs, use_cmdline=False)
    else:
        if verbose:
            print(f"Based on provided arguments processing with model for {sensor}")
            print(f"Based on provided arguments processing model predicting {products}")

    assert mode in ['full', 'point'], f"The toolbox current only supports two modes 'full' and 'point'. " \
                                      f"Instead got mode: '{mode}'."

    'Get the predictions if the model does not exist the toolbox will throw up an error'
    if test_x is not None:
        outputs, op_slices = get_estimates(args, x_test=test_x, return_coefs=True,)

    'Return prodcuts based on user requirement'
    if mode == 'full':
        return outputs, op_slices
    else:
        return np.median(outputs['estimates'], axis=0), op_slices


    print('finished')

def arg_median(X, axis=0):
    """
    This function can be used to identify the location of the sample which corresponds to the median in the specific
    samples.

    Inputs
    ------
    :param X:[np.ndarray]
    A numpy array in which we want to find the position of the median from the samples

    :param axis: [int] (Default: 0)
    An integer axis along which we are doing this process

    Outputs
    ------

    A numpy array of the median location
    """
    assert isinstance(X, np.ndarray), "The variable <X> must be a numpy array"
    assert isinstance(axis, int) and axis >= 0, f"The variable <axis> must be a integer >= 0"
    assert axis <= len(X.shape), f"Given matrix has {len(X.shape)} dimensions, but asking meidan along axis {axis}" \
                                 f"dimension"

    'Find the median along axis of interest'
    amedian = np.nanmedian(X, axis=axis)

    'Find difference from median'
    aabs = np.abs((X.T - np.expand_dims(amedian.T, axis=-1))).T

    'Find the sample with smallest difference'
    return np.nanargmin(aabs, axis=axis)


def get_mdn_uncert_ensemble(ensmeble_distribution, estimates, scaler_y_list, scaler_mode="invert"):
    """
    This function accepts the a dictionary with the distribution details for the entire ensemble and calculates the
    uncertainty for the entire ensmeble

    Inputs
    ------
    :param ensmeble_distribution (list of dictionaries)
    A dictionary that has all the distribution information provided by Brandons MDN package

    :param estimates (np.ndarray)
    A numpy array that contains the estimates from each of the ensemble models for the MDN

    :param scaler_y_list (list of model scalers)
    To convert uncertianty to appropriate scale

    :param scaler_models (str from ['invert', 'non_invert']) [Default: "invert"]

    Outputs
    ------
    :param ensmeble_uncertainties (list)
    A list containing the uncertainties for the entire ensemble set
    """


    assert scaler_mode in ["invert", "non_invert"], f"Only two available options for <scaler_mode> are 'invert' and" \
                                                    f"'non_invert"

    'Get the location of the prediction closest to the median -- may need to select uncertainty of median'
    est_med_loc = arg_median(estimates, axis=0)

    'Create a variable to hold the uncertainties'
    ensmeble_uncertainties= []
    'create a counter to track model number'
    ctr = 0

    'Get the scaler'
    scaler_y = scaler_y_list[0]
    'iterate over models'
    for item in tqdm(ensmeble_distribution):
        dist = {'pred_wts': item[0], 'pred_mu': item[1],
                'pred_sigma': item[2]}

        'Use the MDN output to get the uncertainty estimates'
        aleatoric, epistemic = get_sample_uncertainity(dist)
        aleatoric, epistemic = np.sum(aleatoric, axis=1), np.sum(epistemic, axis=1)

        ensmeble_uncertainties += [np.sqrt(aleatoric + epistemic)]

    'Get sample level uncertainty'
    ensmeble_uncertainties = np.asarray(ensmeble_uncertainties)
    if estimates.shape[2] == 1:
        ensmeble_uncertainties = np.expand_dims(ensmeble_uncertainties, axis=2)
    final_uncertainties = []
    for ii in range(estimates.shape[1]):
        samp_uncert = []
        for jj in range(estimates.shape[2]):
            samp_uncert += [ensmeble_uncertainties[est_med_loc[ii, jj], ii, jj]]

        final_uncertainties += [np.asarray(samp_uncert)]

    'If needed invert the variance'
    estimates[~np.isfinite(estimates)] = 1e-6
    if scaler_mode == "invert":
        lim1 = np.asarray(scaler_y.transform(np.median(estimates+1e-6, axis=0))) - np.asarray(final_uncertainties)
        lim2 = np.asarray(scaler_y.transform(np.median(estimates+1e-6, axis=0))) + np.asarray(final_uncertainties)

        sd = np.squeeze(1 * (scaler_y.inverse_transform(lim2) - scaler_y.inverse_transform(lim1)))

        return sd
    else:
        return np.asarray(final_uncertainties)

def map_cube(img_data, wvl_bands, sensor, products='chl,tss,cdom', land_mask=False, landmask_threshold=0.0,
             flg_subsmpl=False, subsmpl_rate=10, flg_uncert=False, slices=None, scaler_mode="invert",
             block_size=10000):
    """
    This function is used tomap the pixels in a 3D numpy array, in terms of both parameters and the associated
    model uncertainty.

    :param img_data: [np.ndarray: nRow X nCols X nBands]
    The 3D array for which we need MDN predictions

    :param wvl_bands: [np.ndarray: nBands]
    The bands associated with the 3rd dimension of img_data

    :param sensor:
    The sensor for which we are creating the image maps.

    :param products: [str] (Default:"chl,tss,cdom")
    The products we want to estimate using this model.

    :param land_mask: [Bool] (default: False)
    Should a heuristic be applied to mask out the land pixels

    :param landmask_threshold: [-1 <= float <= 1] (default: 0.2)
    The value with which the land mask is being calculated.

    :param flg_subsmpl: [bool] (Default: False)
    Does the image have to be subsampled.

    :param subsmpl_rate: [int > 0] (Default: 10)
    The subsampling rate. Must be an integer. For e.g. if provided rate is 2, one pixel is chosen in each 2X2
    spatial bin.

    :param flg_uncert: [bool] (Default: False)
    Does uncertainty have to be estimated

    :param slices: [dict](Default: None)
    The indicies of the MDN outputs

    :param scaler_modes: [str in ['invert', 'non_invert']] (Default: 'non_invert')
    Is the uncertainty inverted using the MDN's intrinsic scaler

    :param block_size: [int] (Default: 10000)
    The size of the spectral block that is being processed for at once

    :return:
    model_preds: [np.ndarray]
    A prediction for each valid sample in the input image

    img_uncert: [np.ndarray] (OPTIONAL)
    Only generated when flg_ucncert is true. Encapsulates the prediction uncertainty for each sample for each output.

    op_slices: [dictionary]
    The output slices of the various products.
    """

    assert isinstance(img_data, np.ndarray), "The <image_data> variable must be a numpy array"
    assert len(img_data.shape) == 3, "The <image_data> variable must be a 3D numpy array"
    assert isinstance(land_mask, bool), "The <mask_land> parameter must be a boolean variable"
    assert isinstance(landmask_threshold, float) and (np.abs(landmask_threshold) <= 1), "The <landmask_threshold>" \
                                                                                        "must be in range [-1, 1]"
    assert isinstance(flg_subsmpl, bool), f"The variable <flg_subsmpl> must be boolean"
    if flg_subsmpl:
        assert isinstance(subsmpl_rate, int) and (subsmpl_rate > 0), f"The variable <subsmpl_rate> must be a " \
                                                                     f"positive integer"

    'Get the default arguments'
    kwargs = {'product': products,
              'sat_bands': True if products == 'chl,tss,cdom,pc' else False,
              'sensor': sensor}

    if sensor == 'PRISMA' or sensor == 'HICO' or sensor == 'PACE' and products == 'aph,chl,tss,pc,ad,ag,cdom':
        min_in_out_val = 1e-6
        kwargs = {
            'allow_missing': False,
            'allow_nan_inp': False,
            'allow_nan_out': True,

            'sensor': sensor,
            'removed_dataset': "South_Africa,Trasimeno" if sensor == "PRISMA" else "South_Africa",
            'filter_ad_ag': False,
            'imputations': 5,
            'no_bagging': False,
            'plot_loss': False,
            'benchmark': False,
            'sat_bands': False,
            'n_iter': 31622,
            'n_mix': 5,
            'n_hidden': 446,
            'n_layers': 5,
            'lr': 1e-3,
            'l2': 1e-3,
            'epsilon': 1e-3,
            'batch': 128,
            'use_HICO_aph': True,
            'n_rounds': 10,
            'product': 'aph,chl,tss,pc,ad,ag,cdom',
            'use_gpu': False,
            'data_loc': "/home/ryanoshea/in_situ_database/Working_in_situ_dataset/Augmented_Gloria_V3_2/",
            'use_ratio': True,
            'min_in_out_val': min_in_out_val,

        }

        specified_args_wavelengths = {
            'aph_wavelengths': get_sensor_bands(kwargs['sensor'] + '-aph'),
            'adag_wavelengths': get_sensor_bands(kwargs['sensor'] + '-adag'),
        }

    args = get_args(kwargs, use_cmdline=False)

    'Compare the model bands to the available bands '
    sensor_bands = get_sensor_bands(args.sensor)
    if any(sensor_bands != wvl_bands):
        valid_bands = []
        for item in sensor_bands:
            assert np.min(np.abs(np.asarray(wvl_bands) - item))<=5, f"The bands provided-{wvl_bands} do not " \
                                                                       f"agree with the sensor bands {sensor_bands}"
            valid_bands += [np.argmin(np.abs(np.asarray(wvl_bands) - item))]
    else:
        valid_bands = ~np.isnan(wvl_bands)

    'Only selecting the valid bands for this model'
    wvl_bands = np.asarray(wvl_bands)[valid_bands]
    img_data = img_data[:, :, valid_bands]

    'Sub-sample the image if needed'
    if flg_subsmpl:
        img_data = img_data[::subsmpl_rate, ::subsmpl_rate, :]

    'Apply the mask to find and remove the Land pixels or just remove nan values'
    if land_mask:
        'Get the mask which mask out the land pixels'
        # wvl_bands_m, img_data_m = get_tile_data(image_name, 'OLCI-no760', rhos=rhos_flag)
        img_mask = mask_land(img_data, wvl_bands, threshold=landmask_threshold)

        'Get the locations/spectra for the water pixels'
        water_pixels = np.where(img_mask == 0)
        water_spectra = img_data[water_pixels[0], water_pixels[1], :]
    else:
        'Get a simple mask removing pixels with Nan values'
        img_mask = np.asarray((np.isnan(np.min(img_data, axis=2))), dtype=np.float)

        'Get the locations/spectra for the water pixels'
        water_pixels = np.where(img_mask == 0)
        water_spectra = img_data[water_pixels[0], water_pixels[1], :]

    'Mask out the spectra with invalid pixels'
    if water_spectra.size != 0:
        water_spectra = np.expand_dims(water_spectra, axis=1)
        water_final = np.ma.masked_invalid(water_spectra.reshape((-1, water_spectra.shape[-1])))
        water_mask = np.any(water_final.mask, axis=1)
        water_final = water_final[~water_mask]
        # water_pixels = water_pixels[~water_mask]

        'Get the estimates and predictions for each sample'
        final_estimates = np.asarray([])
        final_uncertainties = np.asarray([])

        for ctr in range((water_final.shape[0] // block_size) + 1):
            'Get the data in the block'
            temp = water_final[(ctr * block_size):min((ctr + 1) * block_size, water_final.shape[0])]

            outputs, op_slices = get_mdn_preds(temp, args=args, mode="full")
            estimates = np.asarray(outputs['estimates'])
            if final_estimates.size == 0:
                final_estimates = np.median(estimates, axis=0)
            else:
                final_estimates = np.vstack((final_estimates, np.median(estimates, axis=0)))

            final_estimates = np.asarray(final_estimates)

            if flg_uncert:
                'Perform the Uncertainity estimation'
                ensmeble_uncertainties = get_mdn_uncert_ensemble(ensmeble_distribution=outputs['coefs'],
                                                                 estimates=estimates,
                                                                 scaler_y_list=outputs['scalery'],
                                                                 scaler_mode=scaler_mode)

                if final_uncertainties.size == 0:
                    final_uncertainties = ensmeble_uncertainties
                else:
                    final_uncertainties = np.vstack((final_uncertainties, ensmeble_uncertainties))

                final_uncertainties = np.asarray(final_uncertainties)

        'Create the parameter prediction map'
        model_preds = np.zeros((img_data.shape[0], img_data.shape[1], final_estimates.shape[-1]))
        model_preds[water_pixels[0][~water_mask], water_pixels[1][~water_mask], :] = np.asarray(final_estimates)

        if not flg_uncert:
            return model_preds, op_slices

        'Get the image uncertainty'
        img_uncert = np.zeros((img_data.shape[0], img_data.shape[1], final_estimates.shape[-1]))
        img_uncert[water_pixels[0][~water_mask], water_pixels[1][~water_mask],] = \
            np.squeeze(np.asarray(final_uncertainties))

    else:
        if not flg_uncert:
            return np.zeros((img_data.shape[:-1]))

        model_preds = np.zeros((img_data.shape[:-1]))
        img_uncert = np.zeros((img_data.shape[:-1]))

    return model_preds, img_uncert, op_slices

