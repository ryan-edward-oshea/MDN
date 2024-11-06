# -*- coding: utf-8 -*-

"""
File Name:      uncertainity_estimation
Description:    The code below can be used to extract the uncertainity associated with a specific prediction made by a
                Mixture Density Network (MDN). The uncertainity is further decomposed into "aleatoric" (noise)
                uncertainity and "epistemic" uncertainity as described in [1].

                [1] "Uncertainty-Aware Learning from Demonstration Using Mixture Density Networks with Sampling-Free
                Variance Modeling", Sungjoon Choi, Kyungjae Lee, Sungbin Lim, and Songhwai Oh, 2017.

Date Created:   September 1st, 2021
"""

'Import base packages'
import numpy as np


class uncertainity_estimation:
    def __init__(self, nDim, nDist):
        """
        For each object I am fixing the dimensionality of the vector which the MDN is predicting and the number of
        Gaussian component the mixture contains.

        :param nDim: [int]
        The number of dimensions must be a non-zero integer

        :param nDist: [int]
        The number of distributions must be a non-zero integer
        """
        assert isinstance(nDim, int) and (nDim >= 0), "The target vector dimension must be a positive integer"
        assert isinstance(nDist, int) and (nDist > 1), "The number of distributions for the MDN must be > 1"

        'Initialize the variables'
        self.__nDim = nDim
        self.__nDist = nDist

    def estimate_uncertainity(self, weights, means, variances):
        """
        This function takes the MDN output, with individual component means and variances and decomposes them into the
        uncertainity associated with each component.

        :param weights: [ndarray: __nDist]
        The probability/weight associated with each distribution

        :param means: [ndarray: __nDist X __nDim]
        A 2D numpy matrix which contains the means associated with the various distribution components predicted by the
        MDN. Each row contains the mean corresponding to one component.

        :param variances: [ndarray: __nDist X __nDim X __nDim]
        A 3D numpy matrix which contains the means associated with the various distribution components predicted by the
        MDN. Each row contains the mean corresponding to one component.


        :return: eps_variance: [ndarray: __nDist X __nDim X __nDim]
                alt_variance: [ndarray: __nDist X __nDim X __nDim]

        The estimated epistemic and aleatoric variance associated with each component
        """

        assert isinstance(means, np.ndarray) and isinstance(variances, np.ndarray) and isinstance(weights, np.ndarray),\
            "For now, all parameters are expected to be numpy matrices"
        'Check shape of weights matrix'
        assert len(weights.shape) == 1, "The mean variable must be a 2D matrix"
        assert (weights.shape[0] == self.__nDist), "The weights variable must have dimenision __nDist"
        assert np.round(weights.sum(), 5) == 1., "Since the weights term is a pdf it must sum to 1"

        'Check shape of means matrix'
        assert len(means.shape) == 2, "The mean variable must be a 2D matrix"
        assert (means.shape[0] == self.__nDist) and (means.shape[1] == self.__nDim), "The means variable must have " \
                                                                                      "dimenision __nDist X __nDim"

        'Check shape of variances matrix'
        assert len(variances.shape) == 3, "The variances variable must be a 3D matrix"
        assert (variances.shape[0] == self.__nDist) and \
               (variances.shape[1] == self.__nDim) and \
               (variances.shape[1] == self.__nDim), "The variances variable must have dimenision " \
                                                    "__nDist X __nDim X 1 X 1"

        'The aleatoric uncertainity: noise based uncertainity from Eqn (10) of [1]'
        if self.__nDim == 1:
            alt_uncert = np.squeeze(np.expand_dims(weights, axis=(1,2)) * variances)
        else:
            weighted_var = np.expand_dims(weights, axis=(1, 2)) * variances
            alt_uncert = np.asarray([np.diag(np.squeeze(weighted_var[ii, :, :])) for ii in range(weighted_var.shape[0])])

        'The epistemic uncertainity: knowledge based uncertainity from Eqn. (10) of [1]'
        eps_uncert = np.squeeze(np.square(means - np.sum(np.expand_dims(weights, axis=1) * means, axis=0)))
        eps_uncert = (eps_uncert.T * weights).T

        return alt_uncert, eps_uncert


if __name__ == "__main__":

    pi = np.array([0.21, 0.37, 0.42])
    mu = np.array([[-2.3], [1.8], [-3.1]])
    #var = np.array([[[1,1], [1,1]], [[2,2], [2,2]], [[3,3],[3,3]]])
    var = np.array([[[0.3]], [[1.2]], [[0.85]]])

    alt, eps = uncertainity_estimation(nDim=mu.shape[1], nDist=3).estimate_uncertainity(pi, mu, var)

    print('Finished')





