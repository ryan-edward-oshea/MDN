# -*- coding: utf-8 -*-
"""
File Name:      plot_utilities.py
Description:    This code file contains the helper functions needed to create high quality plots for the various experi-
                ments

Date Created:   September 2nd, 2024
"""
import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.ticker as ticker
from matplotlib.axes import Axes
import seaborn as sns

from .utils import get_tile_data, get_tile_geographic_info
from .metrics import mape, mdsa

'Set display parameters for MATPLOTLIB'
plt.rcParams.update({
"text.usetex": True,
"font.family": "sans-serif",
"font.sans-serif": ["Helvetica"]})
plt.rcParams['mathtext.default']='regular'
SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16
mrkSize = 25
ASPECT="auto"
cmap = "jet"

mpl.rcParams['xtick.labelsize'] = SMALL_SIZE
mpl.rcParams['ytick.labelsize'] = SMALL_SIZE

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


error_function = {
    "mape": mape,
    "mdsa":mdsa,
}

def colorbar(mappable, ticks_list=None, lbl_list=None,):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import matplotlib.pyplot as plt
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(mappable, cax=cax)
    if ticks_list is not None:
        cbar.set_ticks(ticks_list)
        if lbl_list is not None:
            cbar.set_ticklabels(lbl_list)
    plt.sca(last_axes)

    cbar.ax.tick_params(labelsize=BIGGER_SIZE)

    return cbar


def add_identity(ax, *line_args, **line_kwargs):
    '''
    Add 1 to 1 diagonal line to a plot.
    https://stackoverflow.com/questions/22104256/does-matplotlib-have-a-function-for-drawing-diagonal-lines-in-axis-coordinates

    Usage: add_identity(plt.gca(), color='k', ls='--')
    '''
    line_kwargs['label'] = line_kwargs.get('label', '_nolegend_')
    identity, = ax.plot([], [], *line_args, **line_kwargs)

    def callback(axes):
        low_x, high_x = ax.get_xlim()
        low_y, high_y = ax.get_ylim()
        lo = max(low_x, low_y)
        hi = min(high_x, high_y)
        identity.set_data([lo, hi], [lo, hi])

    callback(ax)
    ax.callbacks.connect('xlim_changed', callback)
    ax.callbacks.connect('ylim_changed', callback)

    ann_kwargs = {
        'transform': ax.transAxes,
        'textcoords': 'offset points',
        'xycoords': 'axes fraction',
        'fontname': 'monospace',
        'xytext': (0, 0),
        'zorder': 25,
        'va': 'top',
        'ha': 'left',
    }
    ax.annotate(r'$\mathbf{1:1}$', xy=(0.87, 0.99), size=16, **ann_kwargs)

def create_scatterplots_trueVsPred(y_true, y_pred, color=None, short_name=None, x_label=None, y_label=None, inplot_str=None,
                                   title="Model Performance", maxv_b=None, minv_b=None, ipython_mode=False):
    """
    This function creates scatter plots that can be used compares the true value of a predicted variable against the
    value predicted by a machine learning algorithm. Each variable is placed in a seperate subplot

    :param y_true: [np.ndarray: nSamples X nVariables]
    The true values of the variables. Each column corresponds to a single variable

    :param y_pred: [np.ndarray: nSamples X nVariables]
    The predicted value of the variables. Each column corresponds to a single variable

    :param short_name [list: nVariable](Default: None)
    The short name of the variables of interest

    :param x_label: [list: nVariables] (Default: None)
    The list of labels for the x-axis. Default is none. If provided must have a label for each variable

    :param y_label: [list: nVariables] (Default: None)
    The list of labels for the y-axis. Default is none. If provided must have a label for each variable

    :param inplot_str: [list: nVariables] (Default: None)
    A list of strings which will be placed inside each subplot. Can be used to place the error metrics of the
    predictions inside the subplot window

    :param title: [str] (Default: "Model Performance")
    The title to be placed at the top of the image file

    :param minv_b: [list: nVariables] (Default: [-1]* nVariables)
    The smallest value on the scatter plot

    :param maxv_b: [list: nVariables] (Default: [1]* nVariables)
    The largest value on the scatter plot

    :param ipython_mode:[bool] (Default: False)
    In the ipython_mode, the images are auto displayed and figure is not returned by the function

    :return:
    """

    'Check sizes of the true and predicted values are the same'
    assert y_true.shape == y_pred.shape, 'The arrays of the true and predicted values must have the same shape'
    'Check short names if provided else create appropriate short names'
    if short_name is not None:
        assert len(short_name) == y_true.shape[1], f"Expected {y_true.shape[1]} names. Got {len(short_name)}."
        assert all(isinstance(item, str) for item in short_name), "All elements of <short_names> must be strings"
    else:
        short_name = [f"Var-{ii+1}" for ii in range(len(short_name))]

    'If color vector is given check that is accurate'
    if color is not None:
        assert color.shape == y_true.shape, f"The color vector should be defined for each point"
        assert isinstance(color, np.ndarray), f"The color variable must be numeric"
        assert np.issubdtype(color.dtype, np.number), f"All entries of the color variable must be numeric"

    'Check the labels provided'
    if x_label is not None:
        assert len(x_label) == y_true.shape[1], f"Expected {y_true.shape[1]} names. Got {len(x_label)}."
        assert all(isinstance(item, str) for item in x_label), "All elements of <x_label> must be strings"
    else:
        x_label = [f"True Var-{ii+1}" for ii in range(len(short_name))]

    if y_label is not None:
        assert len(y_label) == y_true.shape[1], f"Expected {y_true.shape[1]} names. Got {len(y_label)}."
        assert all(isinstance(item, str) for item in y_label), "All elements of <y_label> must be strings"
    else:
        y_label = [f"Predicted Var-{ii + 1}" for ii in range(len(short_name))]

    'Check the labels provided'
    if inplot_str is not None:
        assert len(inplot_str) == y_true.shape[1], f"Expected {y_true.shape[1]} names. Got {len(inplot_str)}."
        assert all(isinstance(item, str) for item in inplot_str), "All elements of <inplot_str> must be strings"

    'Check the provided limits for each scatterplot'
    if maxv_b is not None:
        assert len(maxv_b) == y_true.shape[1], f"Need to define limits for {y_true.shape[1]} plots . " \
                                               f"Got {len(maxv_b)}."
        assert all (isinstance(item, int) for item in maxv_b), "The limits need to be integers"
    else:
        maxv_b = [1] * y_true.shape[1]

    'Check the provided limits for each scatterplot'
    if minv_b is not None:
        assert len(minv_b) == y_true.shape[1], f"Need to define limits for {y_true.shape[1]} plots . " \
                                               f"Got {len(minv_b)}."
        assert all(isinstance(item, int) for item in minv_b), "The limits need to be integers"
    else:
        minv_b = [-1] * y_true.shape[1]



    'Create the base figure and set its properties'
    fig1, axes = plt.subplots(nrows=1, ncols=y_true.shape[1], figsize=((7.5 * y_true.shape[1]), 7))
    axes = [ax for axs in np.atleast_1d(axes) for ax in np.atleast_1d(axs)]
    point_colors = ['xkcd:fresh green', 'xkcd:tangerine', 'xkcd:sky blue', 'xkcd:greyish blue', 'xkcd:goldenrod',
              'xkcd:clay', 'xkcd:bluish purple', 'xkcd:reddish']

    ctr = 0
    for (lbl, y1, y2) in zip(short_name, y_true.T, y_pred.T):
        if inplot_str is not None:
            str1 = inplot_str[ctr]
        else:
            str1 = None
        #print(str1)

        l_kws = {'color': point_colors[ctr], 'path_effects': [pe.Stroke(linewidth=4, foreground='k'), pe.Normal()],
                 'zorder': 22,
                 'lw': 1}
        s_kws = {'alpha': 0.4, 'color': point_colors[ctr]}  # , 'edgecolor': 'grey'}

        # curr_idx = 0

        #minv = -2 if lbl == 'cdom' else minv_b[ctr]  # int(np.nanmin(y_true_log)) - 1 if product != 'aph' else -4
        #maxv = 3 if lbl == 'tss' else 3 if lbl == 'chl' else maxv_b[ctr]  # int(np.nanmax(y_true_log)) + 1 if product != 'aph' else 1
        loc = ticker.LinearLocator(numticks=int(round((maxv - minv) / 0.5) + 1))
        # fmt = ticker.FuncFormatter(lambda i, _: r'$10$\textsuperscript{%.1f}' % i)
        fmt1 = ticker.FuncFormatter(lambda i, _: r'%1.1f' % (10**i))
        fmt2 = ticker.FuncFormatter(lambda i, _: r'%1.1f' % (10**i) if ((i /0.5) % 2 == 0) else '')

        axes[ctr].set_ylim((minv, maxv))
        axes[ctr].set_xlim((minv, maxv))
        axes[ctr].xaxis.set_major_locator(loc)
        axes[ctr].yaxis.set_major_locator(loc)
        axes[ctr].xaxis.set_major_formatter(fmt2)
        axes[ctr].yaxis.set_major_formatter(fmt1)
        axes[ctr].tick_params(axis='both', labelsize=SMALL_SIZE)

        valid = np.logical_and(np.isfinite(y1), np.isfinite(y2))
        if valid.sum():
            df = pd.DataFrame((np.vstack((np.log10(y1[valid] + 1e-6), np.log10(y2[valid] + 1e-6)))).T,
                              columns=['true', 'pred'])

            if color is not None:
                sns.regplot(x='true', y='pred', data=df, scatter=False,
                            ax=axes[ctr], scatter_kws=s_kws, line_kws=l_kws, fit_reg=True, truncate=False, robust=True,
                            ci=None)
                plt.scatter(np.log10(y1[valid] + 1e-6), np.log10(y2[valid] + 1e-6), c=color[:, ctr], edgecolor='k',
                            s=mrkSize, vmin=np.percentile(np.squeeze(color[:, ctr]), 10),
                            vmax=np.percentile(np.squeeze(color[:, ctr]), 90))
            else:
                sns.regplot(x='true', y='pred', data=df, scatter=True,
                            ax=axes[ctr], scatter_kws=s_kws, line_kws=l_kws, fit_reg=True, truncate=False, robust=True,
                            ci=None)


            kde = sns.kdeplot(x='true', y='pred', data=df,
                              shade=False, ax=axes[ctr], bw='scott', n_levels=4, legend=False, gridsize=100,
                              color=point_colors[ctr])

        invalid = np.logical_and(np.isfinite(y1), ~np.isfinite(y2))
        if invalid.sum():
            axes[ctr].scatter(np.log10(y1[invalid] + 1e-6), [minv] * (invalid).sum(), color='r',
                              alpha=0.4, label=r'$\mathbf{%s\ invalid}$' % (invalid).sum())
            axes[ctr].legend(loc='lower right', prop={'weight': 'bold', 'size': 16})

        add_identity(axes[ctr], ls='--', color='k', zorder=20)

        props = dict(boxstyle='round', facecolor='white', alpha=0.7)
        if str1 is not None:
            str1 = (str1.strip()).replace(',' ,'\n')
            axes[ctr].text(0.05, 0.95, str1, transform=axes[ctr].transAxes, fontsize=SMALL_SIZE*1, weight="bold",
                           verticalalignment='top', bbox=props)

        textstr1 = r'(N=' + f"{(y2[valid]).shape[0]})"
        axes[ctr].text(0.75, 0.1, textstr1, transform=axes[ctr].transAxes, fontsize=SMALL_SIZE*1, weight="bold",
                       verticalalignment='top', bbox=props)

        axes[ctr].set_xlabel(x_label[ctr].replace(' ', '\ '), fontsize=MEDIUM_SIZE*1, labelpad=10)
        axes[ctr].set_ylabel(y_label[ctr].replace(' ', '\ '), fontsize=MEDIUM_SIZE*1, labelpad=10)
        axes[ctr].set_aspect('equal', 'box')
        axes[ctr].set_title(short_name[ctr])
        axes[ctr].grid()

        ctr += 1

    plt.suptitle(title, fontsize=BIGGER_SIZE, weight="bold")

    if not ipython_mode:
        return fig1



def create_scatterplots_axis(ax, y_true, y_pred, color=None, short_name=None, x_label=None, y_label=None,
                             def_scatter_color="tab:green", inplot_str=None, maxv=None,
                             minv=None, ipython_mode=False, vmin=30., vmax=100.):
    """
    This function creates a single scatter plots that can be used to compare the true value of a predicted variable
    against the value predicted by a machine learning algorithm. The axis is defined by the user.

    :param ax: [matplotlib.axes.Axes]
    The matplotlib axis in which the plot will be created.

    :param y_true: [np.ndarray: nSamples X nVariables]
    The true values of the variables. Each column corresponds to a single variable

    :param y_pred: [np.ndarray: nSamples X nVariables]
    The predicted value of the variables. Each column corresponds to a single variable

    :param color: [np.ndarray: nSamples X nVariables] (Default: None)
    The option for the user to define the color for each scatter point. If none is provided the scatterplot will be
    created where each point uses the color provided by the variable <def_scatter_color>.

    :param short_name [list: nVariable](Default: None)
    The short name of the variables of interest

    :param x_label: [list: nVariables] (Default: None)
    The list of labels for the x-axis. Default is none. If provided must have a label for each variable

    :param y_label: [list: nVariables] (Default: None)
    The list of labels for the y-axis. Default is none. If provided must have a label for each variable

    :param inplot_str: [list: nVariables] (Default: None)
    A list of strings which will be placed inside each subplot. Can be used to place the error metrics of the
    predictions inside the subplot window

    :param minv_b: [list: nVariables] (Default: [-1]* nVariables)
    The smallest value on the scatter plot axis

    :param maxv_b: [list: nVariables] (Default: [1]* nVariables)
    The largest value on the scatter plot axis

    :param vmin: [list: nVariables] (Default: 30)
    If using a vector color bar set the lower limit on the colorbar

    :param maxv_b: [list: nVariables] (Default: 100)
    If using a vector color bar set the higher limit on the colorbar

    :param ipython_mode:[bool] (Default: False)
    In the ipython_mode, the images are auto displayed and figure is not returned by the function

    :return:
    """

    'Check that a valid matplotlib axis is provided'
    assert isinstance(ax, Axes), f"The variable <ax> needs to be a matplotlib.axes.Axes instead got {type(ax)}"

    'Check sizes of the true and predicted values are the same'
    assert  y_true.shape[1] == 1, f"This function is only designed to plot one variable" \
                                  f" instead recieved {y_true.shape[1]} (assumes rows are samples and columns " \
                                  f"are variables)."
    assert y_true.shape == y_pred.shape, 'The arrays of the true and predicted values must have the same shape'
    'Check short names if provided else create appropriate short names'
    if short_name is not None:
        assert len(short_name) == y_true.shape[1], f"Expected {y_true.shape[1]} names. Got {len(short_name)}."
        assert all(isinstance(item, str) for item in short_name), "All elements of <short_names> must be strings"
    else:
        short_name = [f"Var-{ii + 1}" for ii in range(len(short_name))]

    'If color vector is given check that is accurate'
    if color is not None:
        assert color.shape == y_true.shape, f"The color vector should be defined for each point"
        assert isinstance(color, np.ndarray), f"The color variable must be numeric"
        assert np.issubdtype(color.dtype, np.number), f"All entries of the color variable must be numeric"

    'Check the labels provided'
    if x_label is not None:
        assert len(x_label) == y_true.shape[1], f"Expected {y_true.shape[1]} names. Got {len(x_label)}."
        assert all(isinstance(item, str) for item in x_label), "All elements of <x_label> must be strings"
    else:
        x_label = [f"True Var-{ii + 1}" for ii in range(len(short_name))]

    if y_label is not None:
        assert len(y_label) == y_true.shape[1], f"Expected {y_true.shape[1]} names. Got {len(y_label)}."
        assert all(isinstance(item, str) for item in y_label), "All elements of <y_label> must be strings"
    else:
        y_label = [f"Predicted Var-{ii + 1}" for ii in range(len(short_name))]

    'Check the labels provided'
    if inplot_str is not None:
        assert len(inplot_str) == y_true.shape[1], f"Expected {y_true.shape[1]} names. Got {len(inplot_str)}."
        assert all(isinstance(item, str) for item in inplot_str), "All elements of <inplot_str> must be strings"

    'Check that an upper limit is provided for the scatterplot'
    if maxv is not None:
        assert isinstance(maxv, int), "The limits need to be integers"
    else:
        maxv = 1

    'Check that an upper limit is provided for the scatterplot'
    if minv is not None:
        assert isinstance(minv, int), "The limits need to be integers"
    else:
        minv = -1

    'Set the properties of the scatterplot, contours etc.'
    l_kws = {'color': def_scatter_color, 'path_effects': [pe.Stroke(linewidth=4, foreground='k'), pe.Normal()],
             'zorder': 22,
             'lw': 1}
    s_kws = {'alpha': 0.4, 'color': "black"}  # , 'edgecolor': 'grey'}

    'Set the format of the axis/ticker etc.'
    'Set axis tick locations'
    loc = ticker.LinearLocator(numticks=int(round((maxv - minv) / 0.5) + 1))
    'Set appropriate format for axis tick labels'
    fmt1 = ticker.FuncFormatter(lambda i, _: r'%1.1f' % (10 ** i))
    #fmt2 = ticker.FuncFormatter(lambda i, _: r'%1.1f' % (10 ** i) if ((i / 0.5) % 2 == 0) else '')
    'Set the max and min limits fir the axis as provided by the user'
    ax.set_ylim((minv, maxv))
    ax.set_xlim((minv, maxv))
    'Set the ticks'
    ax.xaxis.set_major_locator(loc)
    ax.yaxis.set_major_locator(loc)
    'Set the tick labels'
    ax.xaxis.set_major_formatter(fmt1)
    ax.yaxis.set_major_formatter(fmt1)
    ax.tick_params(axis='both', labelsize=SMALL_SIZE)


    'Check/process the string to be placed inside the scatter-plot. Primary use case is to display the regression' \
    'metrics corresponding to a specific scatterplot'
    if inplot_str is not None:
        str1 = inplot_str[0]
    else:
        str1 = None

    'Squeeze values'
    y_true, y_pred = np.squeeze(y_true), np.squeeze(y_pred)
    'Ensure that there are valid values'
    valid = np.logical_and(np.isfinite(y_true), np.isfinite(y_pred))
    if valid.sum():
        df = pd.DataFrame((np.vstack((np.log10(y_true[valid] + 1e-6),
                                      np.log10(y_pred[valid] + 1e-6)))).T, columns=['true', 'pred'])

        'Create the Seaborn regplot to show how good the regression is'
        'If point by point color is provided fill that in using matplotlib scatter as seaborn does not support that'
        if color is not None:
            sns.regplot(x='true', y='pred', data=df, scatter=False,
                        ax=ax, scatter_kws=s_kws, line_kws=l_kws, fit_reg=True, truncate=False, robust=True,
                        ci=None)
            sc1 = ax.scatter(np.log10(y_true[valid] + 1e-6), np.log10(y_pred[valid] + 1e-6), c=color[:, 0], edgecolor='k',
                        s=mrkSize, vmin=vmin, vmax=vmax)
            plt.colorbar(sc1, ax=ax)
        else:
            sns.regplot(x='true', y='pred', data=df, scatter=True,
                        ax=ax, scatter_kws=s_kws, line_kws=l_kws, fit_reg=True, truncate=False, robust=True,
                        ci=None)

        'Also use kdeplot to get the contours based on density'
        kde = sns.kdeplot(x='true', y='pred', data=df,
                          shade=False, ax=ax, bw='scott', n_levels=4, legend=False, gridsize=100,
                          color="tab:red")

    'Check and and place the invalid values'
    invalid = np.logical_and(np.isfinite(y_true), ~np.isfinite(y_pred))
    if invalid.sum():
        ax.scatter(np.log10(y_true[invalid] + 1e-6), [minv] * (invalid).sum(), color='r',
                          alpha=0.4, label=r'$\mathbf{%s\ invalid}$' % (invalid).sum())
        ax.legend(loc='lower right', prop={'weight': 'bold', 'size': 16})

    add_identity(ax, ls='--', color='k', zorder=20)

    'Place the str in the legend'
    props = dict(boxstyle='round', facecolor='white', alpha=0.7)
    if str1 is not None:
        str1 = (str1.strip()).replace(',', '\n')
        ax.text(0.05, 0.95, str1, transform=ax.transAxes, fontsize=SMALL_SIZE * 1, weight="bold",
                       verticalalignment='top', bbox=props)

    'Add a label to show the number of points'
    textstr1 = r'(N=' + f"{(y_pred[valid]).shape[0]})"
    ax.text(0.75, 0.1, textstr1, transform=ax.transAxes, fontsize=SMALL_SIZE * 1, weight="bold",
                   verticalalignment='top', bbox=props)

    ax.set_xlabel(x_label[0].replace(' ', '\ '), fontsize=MEDIUM_SIZE * 1, labelpad=10)
    ax.set_ylabel(y_label[0].replace(' ', '\ '), fontsize=MEDIUM_SIZE * 1, labelpad=10)
    ax.set_aspect('equal', 'box')
    ax.set_title(short_name[0])
    ax.grid()

def rgb_enhance(rgb: 'numpy.ndarray') -> 'numpy.ndaray':
    """ Rescale a rgb image to enhance the visual quality, adapted from:
    https://gis.stackexchange.com/questions/350663/automated-image-enhancement-in-python

    Parameters:
    rgb : numpy.ndarray of type float - size row*col*3

    Returns:
    rgb_enhanced: numpy.ndarray of type float - size row*col*3

    """

    import skimage.exposure as exposure
    import numpy as np

    rgb_vector = rgb.reshape([rgb.shape[0] * rgb.shape[1], rgb.shape[2]])
    rgb_vector = rgb_vector[~np.isnan(rgb_vector).any(axis=1)]

    # Get cutoff values based on standard deviations. Ideally these would be
    # on either side of each histogram peak and cutoff the tail.
    lims = []
    for i in range(3):
        x = np.mean(rgb_vector[:, i])
        sd = np.std(rgb_vector[:, i])
        low = x - (0.75 * sd)  # Adjust the coefficient here if the image doesn't look right
        high = x + (0.75 * sd)  # Adjust the coefficient here if the image doesn't look right
        if low < 0:
            low = 0
        if high > 1:
            high = 1
        lims.append((low, high))

    r = exposure.rescale_intensity(rgb[:, :, 0], in_range=lims[0])
    g = exposure.rescale_intensity(rgb[:, :, 1], in_range=lims[1])
    b = exposure.rescale_intensity(rgb[:, :, 2], in_range=lims[2])
    rgb_enhanced = np.dstack((r, g, b))

    return rgb_enhanced


def find_rgb_img(img, wvl_bands, PRISMA_mode=False):
    """
    This function can be used extract the RB composite from a image cube

    :param img: [np.ndarray: nRows X nCols X nBands]
    The image cube we are extracting the RGB image from.

    :param wvl_bands:[np.ndarray: nBands]
    The actual wavelength bands

    :param PRISMA_mode[bool] (Default: False)
    A variable that controls the exact way in which the RGB is rescaled for visualization.

    :return:
    """
    assert img.shape[2] == len(wvl_bands), " Wavelengths should be associated with each band in the cube"

    #img, wvl_bands, _, _ = extract_sensor_data(file_name, sensor, rhos=False)

    'Get the RGB Bands'
    rgb_bands = [640, 550, 440]
    img_rgb = np.zeros((img.shape[0], img.shape[1], 3))

    for ii in range(len(rgb_bands)):
        idx = np.argmin(np.abs(wvl_bands - rgb_bands[ii]))
        img_rgb[:, :, ii] = img[:, :, idx]

    if img_rgb.mean() >= 1:
        img_rgb /= 255.

    'Enhance the image for visualiztion'
    if not PRISMA_mode:
        img_rgb = rgb_enhance(img_rgb)
    else:
        'NORMALIZE THE RGB image for better clarity'
        interpolate = lambda data, hi=0.1: np.interp(data, [0, hi], [0, 1])

        for ii in range(img_rgb.shape[2]):
            temp = np.squeeze(img_rgb[:, :, ii])
            temp[temp < 0] = 0
            temp = interpolate(temp, 0.05)
            img_rgb[:, :, ii] = 1. * temp

    return img_rgb


def find_rgb_img_nc(file_name, sensor, rhos=True):
    """
    This function can be used extract the RB composite from a NetCDF file

    :param file_name: [str]
    The physical address of the file to be read

    :param sensor: [str]
    The sensor resoloution which the image is being read at

    :param rhos: [bool] (Default: True)
    The flag which decides whether the function uses rhos or Rrs

    :return:
    """
    'Get the image data and an RGB composite of the scene'
    if "L1B" not in str(file_name):
        wvl_bands, img = get_tile_data(file_name, sensor, rhos=rhos)
        wvl_bands = np.asarray(wvl_bands)
    else:
        import netCDF4

        f = netCDF4.Dataset(file_name)
        img = (f.groups['products']).variables['Lt']  # temperature variable
        wvl_bands = img.wavelengths

    #img, wvl_bands, _, _ = extract_sensor_data(file_name, sensor, rhos=False)

    'Get the RGB Bands'
    rgb_bands = [640, 550, 440]
    img_rgb = np.zeros((img.shape[0], img.shape[1], 3))

    for ii in range(len(rgb_bands)):
        idx = np.argmin(np.abs(wvl_bands - rgb_bands[ii]))
        img_rgb[:, :, ii] = img[:, :, idx]

    if img_rgb.mean() >= 1:
        img_rgb /= 255.

    'Enhance the image for visualiztion'
    if "PRISMA" not in sensor:
        img_rgb = rgb_enhance(img_rgb)
    else:
        'NORMALIZE THE RGB image for better clarity'
        interpolate = lambda data, hi=0.1: np.interp(data, [0, hi], [0, 1])

        for ii in range(img_rgb.shape[2]):
            temp = np.squeeze(img_rgb[:, :, ii])
            temp[temp < 0] = 0
            temp = interpolate(temp, 0.05)
            img_rgb[:, :, ii] = 0.7 * temp

    return img_rgb


def display_sat_rgb(file_name, sensor, figsize=(15, 5), title=None, ipython_mode=False):
    """
    This function can be used extract an RGB image by using the rhos data present in a netCDF file

    :param file_name: [str]
    The physical address of the file to be read

    :param sensor: [str]
    The sensor resoloution which the image is being read at

    :param figsize: (tuple with 2 ints)
    The size of the figure to be plotted

    :param title:[str]
    The title to be added to the matplotlib figure

    :param ipython_mode:[bool] (Default: False)
    In the ipython_mode, the images are auto displayed and figure is not returned by the function
    :return:
    """

    'Get the geographic information'
    lon, lat, extent = get_tile_geographic_info(file_name)
    'Get the rgb composite'
    rgb_img = find_rgb_img_nc(file_name, sensor)

    'Display the results'
    fig1, ax1 = plt.subplots(figsize=figsize)
    fig1.patch.set_visible(True)
    ord = 0


    img1 = ax1.imshow(rgb_img, extent=extent, aspect=ASPECT, zorder=ord)
    if title != None:
        ax1.set_title(title, fontsize=MEDIUM_SIZE, fontweight="bold")

    if not ipython_mode:
        return rgb_img, img1
    else:
        return rgb_img


def overlay_rgb_mdnProducts(rgb_img, model_preds, extent, img_uncert=None, product_name='Parameter',
                            figsize=(15, 5), pred_ticks= [-1, 0, 1, 2], pred_uncert_ticks = [-1, 0, 1, 2],
                            ipython_mode=False):
    """
    This function can be used to overlay the MDN-prediction maps over the RGB compostite of a satellite image for display

    :param rgb_img: [np.ndarray, rows X cols X 3]
    The RGB commposite of the scene

    :param model_preds: [np.ndarray, rows X cols]
    The MDN predictions for that location

    :param extent: [np.array]
    A descrtption of the extent of the location

    :param img_uncert:  [np.ndarray, rows X cols]
    The uncertainty associated with the MDN predictions for that location

    :param product_name: (string) (Default: "Parameter")
    The name of the product that has been predicted

    :param ipython_mode:[bool] (Default: False)
    In the ipython_mode, the images are auto displayed and figure is not returned by the function

    :return: fig1: A figure with appropriate plots
    """

    'Check data properties'
    assert rgb_img.shape[:2] == model_preds.shape[:2], f"The base RGB and prediction image should have the same" \
                                                       f" spatial dimensions"
    assert rgb_img.shape[2] == 3, "The <rgb_img> can only have three bands"
    if len(model_preds.shape) == 3:
        assert model_preds.shape[2] == 1, "This function is only set up to the overlay the predictions of a single " \
                                          "parameter at a time"

    assert len(extent) == 4, "Need to provide the spatial extent of the image to be displayed"
    if img_uncert is not None:
        assert rgb_img.shape[:2] == img_uncert.shape[
                                    :2], f"The base RGB and uncertainty image should have the same spatial dimensions"
        if len(img_uncert.shape) > 2:
            assert model_preds.shape[2] == 1, "This function is only set up to the overlay the predictions of a single " \
                                              "parameter at a time"


    'Create the basic figure and set its properties'
    if img_uncert is not None:
        fig1, (ax1, ax2) = plt.subplots(ncols=2, figsize=figsize, sharex=True, sharey=True)
    else:
        fig1, ax1= plt.subplots(figsize=figsize)

    fig1.patch.set_visible(True)
    ord = 0

    'Display the results - model predictions'
    model_preds = np.log10(model_preds + 1.e-6)
    img1 = ax1.imshow(rgb_img, extent=extent, aspect=ASPECT, zorder=ord)
    img2 = ax1.imshow(np.ma.masked_where(model_preds <= -5.9, model_preds), cmap=cmap,
                      extent=extent, aspect=ASPECT, zorder=ord + 1)
    ax1.set_title(product_name, fontsize=BIGGER_SIZE, fontweight="bold")
    'Apply colorbar'
    #pred_ticks = np.arange(np.floor(np.min(model_preds[model_preds > -5.9])), np.floor(np.max(model_preds))+1)
    pred_labels = [f'{(10**(i)):.2f}'  for i in pred_ticks]
    img2.set_clim(pred_ticks[0], pred_ticks[-1])
    colorbar(img2, ticks_list=pred_ticks, lbl_list=pred_labels)


    'Display the results - model uncertainty'
    if img_uncert is not None:
        img_uncert = np.log10(img_uncert + 1.e-6)
        img3 = ax2.imshow(rgb_img, extent=extent, aspect=ASPECT, zorder=ord)
        'Normalize uncertainty'
        img4 = ax2.imshow(np.ma.masked_where(img_uncert <= -5.9, img_uncert), cmap=cmap,
                      extent=extent, aspect=ASPECT, zorder=ord + 1)
        ax2.set_title(r"Total Uncertainty ($\sigma_{UNC}$)", fontsize=BIGGER_SIZE, fontweight="bold")
        img4.set_clim(pred_uncert_ticks[0], pred_uncert_ticks[-1])
        pred_uncert_labels = [f'{(10**(i)):.2f}' for i in pred_uncert_ticks]   #[f'{i:2.3f}' for i in pred_uncert_ticks]
        colorbar(img4, ticks_list=pred_uncert_ticks, lbl_list=pred_uncert_labels)

    if not ipython_mode:
        return fig1


def create_performance_plots(ax, y_true, y_pred, uncert, bb_limits = np.asarray([0, 0.94, 2.6, 6.4, 20, 56, 154]),
                             error_metric="mae", title=None, ipython_mode=False):
    """
    This function is used to create a barplot that shows the performance of the MDN model in differnt bins as defined
    by the variable <bin_limits>

    :param ax: [matplotlib.axes.Axes]
    The matplotlib axis in which the plot will be created.

    :param y_true: [np.ndarray: nSamples]
    The true values of the variable. [Assumes that there is only variable at time to this function]

    :param y_pred: [np.ndarray: nSamples]
    The predicted value of the variable. [Assumes that there is only variable at time to this function]

    :param uncert: [np.ndarray: nSamples] (Default: None)
    The predictive uncertainty value of the variable, if available. [Assumes that there is only variable at time to this function]

    :param bb_limits: [np.ndarray] (Default: np.asarray([0, 0.94, 2.6, 6.4, 20, 56, 154]))
    The limits of the bin as defined by the user. The values must be in ascending order, with first value of 0 and last
    value of np.inf. The default values are based on OWT for Chla

    :param error_meric: [str from ["mae", "rmse", "rmsle", "mdsa", "bias"]] (Default: mae)
    The error we want to use for the analysis

    :param ipython_mode:[bool] (Default: False)
    In the ipython_mode, the images are auto displayed and figure is not returned by the function

    :return:
    """

    'Check variable <ax>'
    assert isinstance(ax, Axes), f"The variable <ax> must be of type matplotlib.axes.Axes, instead got type {type(ax)}."

    'Check sizes of the true <y_true> and predicted <y_pred> values are the same'
    assert isinstance(y_true, np.ndarray), f"Function assumes the variable <y_true> must be a np.ndarray. Instead got" \
                                           f"{type(y_true)}."
    assert isinstance(y_pred, np.ndarray), f"Function assumes the variable <y_pred> must be a np.ndarray. Instead got" \
                                           f"{type(y_pred)}."
    y_true, y_pred = np.squeeze(y_true), np.squeeze(y_pred)
    assert y_true.shape == y_pred.shape, f'The arrays of the true and predicted values must have the same shape.' \
                                         f' Instead <y_true> is {y_true.shape}, and <y_pred> is {y_pred.shape}.'
    assert len(y_true.shape)==1, f"Function only supports plotting performance of a single variable, instead got " \
                                 f"{y_true.shape[1]} variables (columns)."

    'Similar checks for the uncertainty vector'
    assert isinstance(uncert, np.ndarray), f"Function assumes the variable <y_true> must be a np.ndarray. " \
                                           f"Instead got {type(uncert)}."
    uncert = np.squeeze(uncert)
    assert y_pred.shape == uncert.shape, f'The arrays of the predicted values and uncertainties must have the ' \
                                         f'same shape. Instead <y_pred> is {y_pred.shape}, ' \
                                         f'and <uncert> is {uncert.shape}.'

    if title is not None:
        assert isinstance(title, str), f"The variable title must be a string"

    'Check the binlimits'
    assert isinstance(bb_limits, np.ndarray), f"Function assumes the variable <bin_limits> must be a np.ndarray. " \
                                           f"Instead got {type(bb_limits)}"
    assert len(bb_limits.shape) == 1, f"The <bin_limits> variable must be a 1D vector, instead got an array of " \
                                       f"shape({bb_limits.shape})."
    assert np.all(bb_limits[:-1] <= bb_limits[1:]), f"The variables in <bin_limits> must be in ascending order. " \
                                                      f"Instead got <bin_limits> = {bb_limits}."
    assert all(np.isfinite(bb_limits)), f"All entries of <bin_limits> must be valid and finite,  " \
                                         f"instead got <bin_limits> = {bb_limits}."
    'Add inifinity at the end to create the final bin'
    bb_limits =np.append(bb_limits, np.inf)

    'Check the error metrics being used here'
    assert isinstance(error_metric, str), f"The variable <errpr_metric> must be a string, " \
                                          f"instead got {type(error_metric)}."
    assert error_metric in error_function.keys(), 'The function only supports the following error' \
                                                                     'metrics: ["mae", "mdsa", "bias"]' \
                                                                     f'. Instead got {error_metric}.'
    error_func = error_function.get(error_metric)

    'Iterate over the limits and get and plot statistics in each bin'
    full_labels = []
    bar_labels = [r"\% samples", r"\% error", "\% uncertainty"]
    for ctr in np.arange(1, bb_limits.shape[0]):
        "Get the upper and lower limitrs"
        ll_lim, up_lim = bb_limits[ctr-1], bb_limits[ctr]
        'Create label'
        full_labels += [f'{int(ll_lim):1.1f}-{up_lim:1.1f}']


        'Find the samples in this range'
        idx = np.where((ll_lim <= y_true) & (y_true < up_lim))[0]

        'Get the stats'
        if y_true[idx].size != 0:
            'Number of samples'
            nsamples = 100. * y_true[idx].shape[0] / y_true.shape[0]
            error_val = error_func(y_true[idx], y_pred[idx])
            median_uncert = 100. * np.median(uncert[idx] / y_true[idx])
        else:
            nsamples, error_val, median_uncert = 0., 0., 0.

        'Plot the bars as needed'
        t1_label = bar_labels[0] if ctr == 1 else '_nolegend_'
        samp_bar = ax.bar((5 * ctr) - 0.5, nsamples, width=0.5,
                          color='skyblue', edgecolor="steelblue", label=t1_label)

        t2_label = bar_labels[1] if ctr == 1 else '_nolegend_'
        error_bar = ax.bar((5 * ctr), error_val, width=0.5,
                           color='coral', edgecolor="orangered", label=t2_label)

        t3_label = bar_labels[2] if ctr == 1 else '_nolegend_'  # <- FIXED LINE
        uncert_bar = ax.bar((5 * ctr) + 0.5, median_uncert, width=0.5,
                            color='limegreen', edgecolor="darkgreen", label=t3_label)

    'Add in the plot ticks and ticklabels'
    ax.set_xticks(np.arange(5, 5*bb_limits.shape[0], 5))
    ax.set_xticklabels(full_labels, rotation=70)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.set_yticklabels([f'{int(item)}' for item in np.arange(0, 101, 20)])
    ax.tick_params(axis='both', labelsize=MEDIUM_SIZE * 1)
    ax.set_ylim([0., 100.])
    ax.legend(loc='upper right', framealpha=0.7, fontsize=MEDIUM_SIZE)

    if title is not None:
        ax.set_title(title, fontsize=MEDIUM_SIZE, fontweight="bold")
    ax.grid()

    return ax



if __name__ == "__main__":
    sensor = "OLCI"
    date = "08-29-2016"
    location = "lake_erie"

    tile_path = f"data/example_imagery/{sensor}/{date}/{location}/sat_cube.nc"

    img_rgb = find_rgb_img_nc(tile_path, sensor)
