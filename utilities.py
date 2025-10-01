# -*- coding: utf-8 -*-
"""
File Name:      utilities
Description:    This code file contains a set of supporting functions that are needed to generate the predictions and
                uncertainty estimations available for a specific models.

Date Created:   September 2nd, 2024
"""

import numpy as np
from tqdm import tqdm

from .meta import get_sensor_bands
from .parameters import get_args
from .product_estimation import get_estimates
from .uncertainty_package_final.uncert_support_lib import get_sample_uncertainity
from .utils import mask_land

'Base properties for imshow'
ASPECT = 'auto'
cmap = 'jet'


def get_mdn_preds(test_x, args=None, sensor="OLCI", products="chl", mode="point", model_type='production',
                  verbose=False):
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

    assert model_type in ['production', 'testing'], f"Currently the toolbox only supports two model types" \
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
            print(f"Based on provided arguments processing with model for {args.sensor}")
            print(f"Based on provided arguments processing model predicting {args.products}")

    assert mode in ['full', 'point'], f"The toolbox current only supports two modes 'full' and 'point'. " \
                                      f"Instead got mode: '{mode}'."

    'Get the predictions if the model does not exist the toolbox will throw up an error'
    if test_x is not None:
        outputs, op_slices = get_estimates(args, x_test=test_x, return_coefs=True, )

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


def get_mdn_uncert_ensemble(ensmeble_distribution, estimates, scaler_y_list, scaler_mode="invert",
                            uncert_mode="full", flg_uncert_limits=False):
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

    :param scaler_mode (str from ['invert', 'non_invert']) [Default: "invert"]
    This is flag that decides whether the uncertainty is in the scaled space in which the model works or is inverted
    back to the physical space.

    :param uncert_mode (str from ['full', 'select']) [Default: "select"]
    This is flag that decides whether the function returns the uncertainty corresponding to each ensemble or if it
    returns the uncertainty corresponding to the value closest to the median.

    :param flg_uncert_limits (bool) [Default: False]
    This is flag that decides whether the function returns the uncertainty as a composite metric or as upper and lower
    limits around the central value.

    Outputs
    ------
    :param ensmeble_uncertainties (list)
    A list containing the uncertainties for the entire ensemble set
    """

    assert scaler_mode in ["invert", "non_invert"], f"Only two available options for <scaler_mode> are 'invert' and" \
                                                    f"'non_invert'"
    assert uncert_mode in ["full", "select"], f"Only two available options for <uncert_mode> are 'full' and" \
                                              f"'select'"
    assert isinstance(flg_uncert_limits, bool), f"The variable <flg_uncert_limits> must be Boolean."

    'Create a variable to hold the uncertainties'
    ensemble_uncertainties = []

    'iterate over models'
    for item in tqdm(ensmeble_distribution):
        dist = {'pred_wts': item[0], 'pred_mu': item[1],
                'pred_sigma': item[2]}

        'Use the MDN output to get the uncertainty estimates'
        aleatoric, epistemic = get_sample_uncertainity(dist)
        aleatoric, epistemic = np.sum(aleatoric, axis=1), np.sum(epistemic, axis=1)

        ensemble_uncertainties += [np.sqrt(aleatoric + epistemic)]

    'Get sample level uncertainty'
    if uncert_mode is "select":
        'This model is the uncertainty associated with the median prediction'
        est_med_loc = np.argmin(np.abs(estimates - np.median(estimates, axis=0)[np.newaxis, :]),
                                axis=0)  # arg_median(estimates, axis=0)

        ensemble_uncertainties = np.asarray(ensemble_uncertainties)
        final_uncertainties = np.take_along_axis(ensemble_uncertainties, est_med_loc[None, ...], axis=0)[0]
        """if estimates.shape[2] == 1:
            ensemble_uncertainties = np.expand_dims(ensemble_uncertainties, axis=2)
        final_uncertainties = []
        for ii in range(estimates.shape[1]):
            samp_uncert = []
            for jj in range(estimates.shape[2]):
                samp_uncert += [ensemble_uncertainties[est_med_loc[ii, jj], ii, jj]]

            final_uncertainties += [np.asarray(samp_uncert)]"""

        'If needed invert the variance'
        estimates[~np.isfinite(estimates)] = 1e-6
        if scaler_mode == "invert":
            'Get the scaler'
            scaler_y = scaler_y_list[0]
            lim1 = np.asarray(scaler_y.transform(np.median(estimates + 1e-6, axis=0))) - np.asarray(final_uncertainties)
            lim2 = np.asarray(scaler_y.transform(np.median(estimates + 1e-6, axis=0))) + np.asarray(final_uncertainties)

            'Use the flag to decide whether the uncertainty is being returned as a limit or a composite metric'
            sd= {}
            if flg_uncert_limits:
                sd['low_lim'] = scaler_y.inverse_transform(lim1)
                sd['upp_lim'] = scaler_y.inverse_transform(lim2)
            else:
                sd['comp_unc'] = np.squeeze(1 * (scaler_y.inverse_transform(lim2) - scaler_y.inverse_transform(lim1)))

            return sd
        else:
            'No limit option in the unscaled mode as the uncertainties are symmetric in this mode'
            return np.asarray(final_uncertainties)
    else:
        if scaler_mode == "invert":
            'Create a variable to hold the uncertainties'
            sd = {}
            if flg_uncert_limits:
                sd['low_lim'] = {}
                sd['upp_lim'] = {}

            for ii, item in enumerate(estimates):
                'Get the scaler'
                scaler_y = scaler_y_list[ii]
                if len(item.shape) == 1:
                    item = item.reshape((-1, 1))
                lim1 = np.squeeze(np.asarray(scaler_y.transform(item))) - np.squeeze(
                    np.asarray(ensemble_uncertainties[ii]))
                lim2 = np.squeeze(np.asarray(scaler_y.transform(item))) + np.squeeze(
                    np.asarray(ensemble_uncertainties[ii]))

                if len(lim1.shape) == 1:
                    lim1 = lim1.reshape((-1, 1))
                    lim2 = lim2.reshape((-1, 1))

                if flg_uncert_limits:
                    sd['low_lim'][f"Model-{ii}"] = lim1
                    sd['upp_lim'][f"Model-{ii}"] = lim2
                else:
                    sd[f"Model-{ii}"] = np.squeeze(1 * (scaler_y.inverse_transform(lim2) - scaler_y.inverse_transform(lim1)))

            return sd
        else:
            return ensemble_uncertainties


def get_mdn_preds_uncertainties(test_x, args=None, sensor="OLCI", products="chl", model_type='production',
                                verbose=False, scaler_mode="invert", uncert_mode="full", flg_uncert_limits=False):
    """
    This function is used to generate estimates from pre-trained MDN

    Inputs
    ------
    :param test_x: (np.ndarray: nSamples X nBands)
    A numpy array with the data on which we want to generate the predictions. Each row of this matrix corresponds to a
    test spectral samples.

    :param args: (dict) [Default: None]
    The arguments for the MDN.

    :param sensor: (str) [Default: "OLCI"]
    The sensor for which predictions are needed. The provided string must be a valid sensor defined in the MDN package.
    This argument in only used if args is not provided to create the MDN args.

    :param products: (str) [Default: "chl"]
    The products which will be predicted by the model. This argument is only used if args is not provided.

    :param model_type: [Default: 'production']
    A flag that signifies whether we use a model trained on the full GLORIA data ('production') or a reduced training
    set to enable the creation of a air-gapped test/validation set ('testing'). It affects how the arguments for the
    MDN package are set

    :param verbose: (bool) [Default: False]
    A boolean flag that controls how much information is printed out to the console

    :param scaler_mode (str from ['invert', 'non_invert']) [Default: "invert"]
    This is flag that decides whether the uncertainty is in the scaled space in which the model works or is inverted
    back to the physical space.

    :param uncert_mode (str from ['full', 'select']) [Default: "select"]
    This is flag that decides whether the function returns the uncertainty corresponding to each ensemble or if it
    returns the uncertainty corresponding to the value closest to the median.

    :param flg_uncert_limits (bool) [Default: False]
    This is flag that decides whether the function returns the uncertainty as a composite metric or as upper and lower
    limits around the central value.

    Outputs
    -------
        a) mdn_preds: In the 'full' mode the output predictions is a list which contains all the different output compo-
        nents of the MDN output. In the point mode it generates a numpy array that is the best estimate of the output
        for each sample (row) for each WQI (column)
        b) op_slices: A dictionary indicating the slices corresponding to the different output variables.
    """

    assert model_type in ['production', 'testing'], f"Currently the toolbox only supports two model types" \
                                                    f" 'production' and 'testing'. Instead got '{model_type}'."
    assert scaler_mode in ["invert", "non_invert"], f"Only two available options for <scaler_mode> are 'invert' and" \
                                                    f"'non_invert'"
    assert uncert_mode in ["full", "select"], f"Only two available options for <uncert_mode> are 'full' and" \
                                              f"'select'"

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
            print(f"Based on provided arguments processing with model for {args.sensor}")
            print(f"Based on provided arguments processing model predicting {args.products}")

    'Get the predictions if the model does not exist the toolbox will throw up an error'
    mdn_preds_full, mdn_preds_desc = get_mdn_preds(test_x, args, sensor=args.sensor, products=args.product, mode="full")

    'Clean the MDN estimates'
    temp = mdn_preds_full['estimates']
    temp = np.nan_to_num(temp, 20000)
    temp[~np.isfinite(temp)] = 20000
    temp[temp <= args.min_in_out_val] = args.min_in_out_val
    temp[temp > 20000] = 20000
    temp[temp <= 1.e-6] = 1.e-6
    mdn_preds_full['estimates'] = temp

    'Get the predictions from the model'
    if scaler_mode == "invert":
        mdn_predictions = mdn_preds_full['estimates']
    else:
        'Create a variable to hold the MDN predictions'
        mdn_predictions = np.asarray([])

        'Get the scaled prediction of each model'
        for model in range(10):
            'Find the MDN component with the highest weight'
            max_weight_comp = mdn_preds_full['coefs'][model][0].argmax(axis=1)  # [0] here refers to MDN weights,
            # which is the first item of the MDN
            # predictions
            'Get the means corresponding to the component with the highest weight'
            mdn_pred_model_val = mdn_preds_full['coefs'][model][1]  # [0] here refers to mean of the individual
            # gaussians, which is the second item of the MDN
            # predictions
            'Select the mean corresponding the largest weight'
            mdn_pred_model_val = mdn_pred_model_val[np.arange(mdn_pred_model_val.shape[0]), max_weight_comp, :]

            'If the predictions variable is empty'
            if mdn_predictions.size == 0:
                mdn_predictions = mdn_pred_model_val
            else:
                mdn_predictions = np.dstack((mdn_predictions, mdn_pred_model_val))

        'Transpose the variable so the models is the first dimension'
        mdn_predictions = mdn_predictions.transpose((2, 0, 1))

    'Get all the corresponding uncertainties for this prediction'
    mdn_uncertainties = get_mdn_uncert_ensemble(mdn_preds_full['coefs'], np.asarray(mdn_preds_full['estimates']),
                                                mdn_preds_full['scalery'], scaler_mode=scaler_mode,
                                                uncert_mode=uncert_mode, flg_uncert_limits=flg_uncert_limits)

    if uncert_mode == "select":
        # Index of prediction closest to the median
        est_med_loc = np.argmin(
            np.abs(mdn_predictions - np.median(mdn_predictions, axis=0, keepdims=True)),
            axis=0
        )
        # Gather median predictions and uncertainties
        final_predictions = np.take_along_axis(mdn_predictions, est_med_loc[None, ...], axis=0)[0]

        if flg_uncert_limits:
            'Extract the lower and upper limit uncertainties'
            final_uncertainties_lb = np.asarray(mdn_uncertainties['low_lim'])
            final_uncertainties_ub = np.asarray(mdn_uncertainties['upp_lim'])


            return final_predictions, (final_uncertainties_lb, final_uncertainties_ub), mdn_preds_desc

        else:
            # If using composite uncertainties extract that
            final_uncertainties = np.asarray(mdn_uncertainties['comp_unc'])
            return final_predictions, final_uncertainties, mdn_preds_desc
    else:
        if flg_uncert_limits:
            final_uncertainties_lb = np.stack(
                [mdn_uncertainties['low_lim'][f"Model-{ii}"] for ii in range(len(mdn_uncertainties))], axis=0)
            final_uncertainties_ub = np.stack(
                [mdn_uncertainties['upp_lim'][f"Model-{ii}"] for ii in range(len(mdn_uncertainties))], axis=0)

            return mdn_predictions, (final_uncertainties_lb, final_uncertainties_ub), mdn_preds_desc
        else:
            return mdn_predictions, np.stack([mdn_uncertainties[f"Model-{i}"] for i in range(10)], axis=0), mdn_preds_desc






def map_cube_mdn_full(args, img_data, wvl_bands, land_mask=False, landmask_threshold=0.0, flg_subsmpl=False,
                      subsmpl_rate=10, scaler_mode="invert", block_size=10000, uncert_mode="select", flg_uncert_limits=False):
    """
    This function is used tomap the pixels in a 3D numpy array, in terms of both parameters and the associated
    model uncertainty.

    :param args: [dict]
    A dictionary that holds the settings of the MDN model that is being used for this process.

    :param img_data: [np.ndarray: nRow X nCols X nBands]
    The 3D array for which we need MDN predictions

    :param wvl_bands: [np.ndarray: nBands]
    The bands associated with the 3rd dimension of img_data

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

    :param uncert_mode (str from ['full', 'select']) [Default: "select"]
    This is flag that decides whether the function returns the uncertainty corresponding to each ensemble or if it
    returns the uncertainty corresponding to the value closest to the median.

    :param flg_uncert_limits (bool) [Default: False]
    This is flag that decides whether the function returns the uncertainty as a composite metric or as upper and lower
    limits around the central value.

    :param block_size: [int] (Default: 10000)
    The size of the spectral block that is being processed for at once

    :return:
    img_preds: [np.ndarray]
    A prediction of all WQIs estimated by the MDN for each valid sample in the input image

    img_uncert: [np.ndarray] (OPTIONAL)
    Encapsulates the prediction uncertainty for each sample for each output.

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

    'Compare the model bands to the available bands '
    sensor_bands = get_sensor_bands(args.sensor)
    if not np.array_equal(np.asarray(sensor_bands), np.asarray(wvl_bands)):
        valid_bands = []
        for item in sensor_bands:
            assert np.min(np.abs(np.asarray(wvl_bands) - item)) <= 5, f"The bands provided-{wvl_bands} do not " \
                                                                      f"agree with the sensor bands {sensor_bands}"
            valid_bands += [int(np.argmin(np.abs(np.asarray(wvl_bands) - item)))]

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
        img_mask = np.asarray((np.isnan(np.min(img_data, axis=2))), dtype=float)

        'Get the locations/spectra for the valid water pixels'
        water_pixels = np.where(img_mask == 0)
        water_spectra = img_data[water_pixels[0], water_pixels[1], :]

        'Flag spectra with majority -ve values'
        maj_negative = (water_spectra < 0.0001).sum(axis=1) > 5
        'Drop spectra with majority -ve values'
        water_pixels = tuple(item[~maj_negative] for item in water_pixels)
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
        if flg_uncert_limits:
            final_uncertainties_lb = np.asarray([])
            final_uncertainties_ub = np.asarray([])
        else:
            final_uncertainties = np.asarray([])

        for ctr in range((water_final.shape[0] // block_size) + 1):
            'Get the data in the block'
            temp = water_final[(ctr * block_size):min((ctr + 1) * block_size, water_final.shape[0])]
            temp[temp <= args.min_in_out_val] = args.min_in_out_val

            'Get the estimates and uncertainties'
            if flg_uncert_limits:
                block_estimates, (block_uncertainties_lb,  \
                    block_uncertainties_ub), op_slices= get_mdn_preds_uncertainties(temp,
                                                                        args=args, sensor=args.sensor,
                                                                        products=args.product,
                                                                        scaler_mode=scaler_mode,
                                                                        uncert_mode=uncert_mode,
                                                                        flg_uncert_limits=flg_uncert_limits,
                                                                        verbose=False)
                'Add this block of predictions to existing predictions and uncertainties'
                if final_estimates.size == 0 and final_uncertainties_lb.size == 0 and final_uncertainties_ub.size == 0:
                    final_estimates = block_estimates
                    final_uncertainties_lb = block_uncertainties_lb
                    final_uncertainties_ub = block_uncertainties_ub
                else:
                    final_estimates = np.vstack((final_estimates, block_estimates))
                    final_uncertainties_lb = np.vstack((final_uncertainties_lb, block_uncertainties_lb))
                    final_uncertainties_ub = np.vstack((final_uncertainties_ub, block_uncertainties_ub))
            else:
                block_estimates, block_uncertainties, op_slices = get_mdn_preds_uncertainties(temp, args=args, sensor=args.sensor,
                                                                                   products=args.product,
                                                                                   scaler_mode=scaler_mode,
                                                                                   uncert_mode= uncert_mode,
                                                                                   flg_uncert_limits=flg_uncert_limits,
                                                                                   verbose=False)

                'Add this block of predictions to existing predictions'
                if final_estimates.size == 0 and final_uncertainties.size == 0:
                    final_estimates = block_estimates
                    final_uncertainties = block_uncertainties
                else:
                    final_estimates = np.vstack((final_estimates, block_estimates))
                    final_uncertainties = np.vstack((final_uncertainties, block_uncertainties))

        'Create the parameter prediction cube'
        img_preds = np.zeros((img_data.shape[0], img_data.shape[1], final_estimates.shape[-1]))
        img_preds[water_pixels[0][~water_mask], water_pixels[1][~water_mask], :] = np.asarray(final_estimates)
        'Create the parameter prediction cube'
        if flg_uncert_limits:
            img_uncert_lb = np.zeros((img_data.shape[0], img_data.shape[1], final_estimates.shape[-1]))
            img_uncert_lb[water_pixels[0][~water_mask], water_pixels[1][~water_mask],] = \
                np.squeeze(np.asarray(final_uncertainties_lb))

            img_uncert_ub = np.zeros((img_data.shape[0], img_data.shape[1], final_estimates.shape[-1]))
            img_uncert_ub[water_pixels[0][~water_mask], water_pixels[1][~water_mask],] = \
                np.squeeze(np.asarray(final_uncertainties_ub))

            return img_preds, (img_uncert_lb, img_uncert_ub), op_slices
        else:
            img_uncert = np.zeros((img_data.shape[0], img_data.shape[1], final_estimates.shape[-1]))
            img_uncert[water_pixels[0][~water_mask], water_pixels[1][~water_mask],] = \
                np.squeeze(np.asarray(final_uncertainties))

            return img_preds, img_uncert, op_slices

    else:
        'Create and return cube of only 0 of the appropriate size'
        img_preds = np.zeros((img_data.shape[0], img_data.shape[1], args['data_ytrain_shape'][1]))
        'Create the parameter prediction cube'
        if flg_uncert_limits:
            img_uncert_lb = np.zeros((img_data.shape[0], img_data.shape[1], args['data_ytrain_shape'][1]))
            img_uncert_ub = np.zeros((img_data.shape[0], img_data.shape[1], args['data_ytrain_shape'][1]))
            return img_preds, (img_uncert_lb, img_uncert_ub), op_slices
        else:
            img_uncert = np.zeros((img_data.shape[0], img_data.shape[1], args['data_ytrain_shape'][1]))


            return img_preds, img_uncert. op_slices

