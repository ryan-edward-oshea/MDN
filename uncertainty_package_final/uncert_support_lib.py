# -*- coding: utf-8 -*-
"""
File Name:      uncert_support_lib
Description:    This file contains various code to support the extraction of uncertainty from the predictions of a MDN
                model.

Date Created:   September 8th, 2022
"""
'Import base python packages'
import numpy as np
import sys

from tqdm import tqdm

'Import user defined packages'
from .uncertainity_estimation import uncertainity_estimation

INTERP_MODE= ["nearest", "interp"]
NAN_MODE= ["any", "all"]
_NUMERIC_KINDS = set('uifc')

def get_sample_uncertainity(pred_dist, compress=False):
    """
    This function iterates over all data samples and extract from the data the uncertainities associated with each
     prediction. THIS FUNCTION ONLY WORKS FOR A SINGLE MODEL

    :param pred_dist: [nSamples:dict]
    A list of dicitonaries where each element is the distribution predicted by the MDN for a specific sample

    :param compress [bool] (Default:True)
    The variable is used to compress the distribution level uncertainties into a single measure

    :return:
    """

    'Estimate the different uncertainities for each sample'
    #aleatoric, epistemic = np.zeros((pred_dist['pred_mu'].shape[0],pred_dist['pred_mu'][0, :].shape[0], pred_dist['pred_mu'][0, :].shape[0])), \
    #                      np.zeros((pred_dist['pred_mu'].shape[0],pred_dist['pred_mu'][0, :].shape[0]))
    aleatoric, epistemic = np.squeeze(np.zeros((pred_dist['pred_mu'].shape))), np.squeeze(np.zeros((pred_dist['pred_mu'].shape)))
    for ii in tqdm(range(pred_dist['pred_wts'].shape[0])):
        #pi = pred_dist['pred_wts'][ii, :]
        #mu = pred_dist['pred_mu'][ii, :]
        #var = pred_dist['pred_sigma'][ii, :]

        aleatoric[ii, :], epistemic[ii, :] = uncertainity_estimation(nDim=pred_dist['pred_mu'][ii, :].shape[1],
                                           nDist=pred_dist['pred_mu'][ii, :].shape[0]).estimate_uncertainity(
            pred_dist['pred_wts'][ii, :], pred_dist['pred_mu'][ii, :], pred_dist['pred_sigma'][ii, :])

        """if aleatoric is None or epistemic is None:
            aleatoric = alt
            epistemic = eps
        else:
            aleatoric = np.vstack((aleatoric, alt))
            epistemic = np.vstack((epistemic, eps))"""


    if compress:
        uncert = np.sqrt(np.sum(aleatoric, axis=1) + np.sum(epistemic, axis=1))
        return uncert

    return aleatoric, epistemic


"""if __name__ == "__main__":
    'Define the base address where the data is present'
    base_folder= "/Users/arunsaranathan/SSAI/data/Insitu/"

    'Define the sensor to be processed and the wavelengths required for the sensor'
    sensor_option = "OLCI"

    '------------------------------------------------------------------------------------------------------------------'
    'STEP 1: Get the data in this folder'
    print('Getting the appropriate data')
    args = get_args()
    args.sensor = sensor_option
    data, chla, _, _ = get_data(args)
    '------------------------------------------------------------------------------------------------------------------'
    'STEP 2: Make the predictions on the available data'
    pred_dist, scalers, estimates = get_model_preds(args, data)

    '------------------------------------------------------------------------------------------------------------------'
    'STEP 3: Perform the Uncertainity estimation'
    aleatoric, epistemic = get_sample_uncertainity(pred_dist, sensor_type=sensor_option)

    print('Finished')"""














