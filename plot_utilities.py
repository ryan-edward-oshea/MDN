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
import seaborn as sns

from .utils import get_tile_data, get_tile_geographic_info

'Set display parameters for MATPLOTLIB'
plt.rcParams.update({
"text.usetex": True,
"font.family": "sans-serif",
"font.sans-serif": ["Helvetica"]})
plt.rcParams['mathtext.default']='regular'
SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16
mrkSize = 5
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

def create_scatterplots_trueVsPred(y_true, y_pred, short_name=None, x_label=None, y_label=None, inplot_str=None,
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
    colors = ['xkcd:fresh green', 'xkcd:tangerine', 'xkcd:sky blue', 'xkcd:greyish blue', 'xkcd:goldenrod',
              'xkcd:clay', 'xkcd:bluish purple', 'xkcd:reddish']


    ctr = 0
    for lbl, y1, y2 in zip(short_name, y_true.T, y_pred.T):
        str1 = inplot_str[ctr]
        #print(str1)

        l_kws = {'color': colors[ctr], 'path_effects': [pe.Stroke(linewidth=4, foreground='k'), pe.Normal()],
                 'zorder': 22,
                 'lw': 1}
        s_kws = {'alpha': 0.4, 'color': colors[ctr]}  # , 'edgecolor': 'grey'}

        # curr_idx = 0

        minv = -2 if lbl == 'cdom' else minv_b[ctr]  # int(np.nanmin(y_true_log)) - 1 if product != 'aph' else -4
        maxv = 3 if lbl == 'tss' else 3 if lbl == 'chl' else maxv_b[ctr]  # int(np.nanmax(y_true_log)) + 1 if product != 'aph' else 1
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
            sns.regplot(x='true', y='pred', data=df,
                        ax=axes[ctr], scatter_kws=s_kws, line_kws=l_kws, fit_reg=True, truncate=False, robust=True,
                        ci=None)
            kde = sns.kdeplot(x='true', y='pred', data=df,
                              shade=False, ax=axes[ctr], bw='scott', n_levels=4, legend=False, gridsize=100,
                              color=colors[ctr])

        invalid = np.logical_and(np.isfinite(y1), ~np.isfinite(y2))
        if invalid.sum():
            axes[ctr].scatter(np.log10(y1[invalid] + 1e-6), [minv] * (invalid).sum(), color='r',
                              alpha=0.4, label=r'$\mathbf{%s\ invalid}$' % (invalid).sum())
            axes[ctr].legend(loc='lower right', prop={'weight': 'bold', 'size': 16})

        add_identity(axes[ctr], ls='--', color='k', zorder=20)

        props = dict(boxstyle='round', facecolor='white', alpha=0.7)
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


def display_sat_rgb(file_name, sensor, figsize=(15, 5), title=None, ipython_mode=False, flipud=False):
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
    if flipud: rgb_img = np.flipud(rgb_img)
    
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

if __name__ == "__main__":
    sensor = "OLCI"
    date = "08-29-2016"
    location = "lake_erie"

    tile_path = f"data/example_imagery/{sensor}/{date}/{location}/sat_cube.nc"

    img_rgb = find_rgb_img_nc(tile_path, sensor)
