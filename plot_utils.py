from .metrics import slope, sspb, mdsa, rmsle, r_squared
from .meta import get_sensor_label,get_sensor_bands
from .utils import closest_wavelength, ignore_warnings
from collections import defaultdict as dd 
from pathlib import Path 
import numpy as np 
from matplotlib.pyplot import text
from .spectrum_rgb import get_spectrum_cmap

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
        lo = max(low_x,  low_y)
        hi = min(high_x, high_y)
        identity.set_data([lo, hi], [lo, hi])

    callback(ax)
    ax.callbacks.connect('xlim_changed', callback)
    ax.callbacks.connect('ylim_changed', callback)

    ann_kwargs = {
        'transform'  : ax.transAxes,
        'textcoords' : 'offset points', 
        'xycoords'   : 'axes fraction', 
        'fontname'   : 'monospace', 
        'xytext'     : (0,0), 
        'zorder'     : 25,     
        'va'         : 'top', 
        'ha'         : 'left', 
    }
    ax.annotate(r'$\mathbf{1:1}$', xy=(0.87,0.99), size=11, **ann_kwargs)

def create_stats_HPC(y_true, y_est, metrics=[mdsa, sspb, slope],label=None):
    ''' Create metrics for model comparison '''
    summary_metrics = {}
    for metric in metrics:
        # print(metric.__name__)
        label = metric.__name__#.replace('SSPB', 'Bias').replace('MdSA', 'Error')
        print(label, metric(y_true, y_est))
        summary_metrics[label] = metric(y_true, y_est)
    return summary_metrics

def _create_metric(metric, y_true, y_est, longest=None, label=None):
    ''' Create a position-aligned string which shows the performance via a single metric '''
    # if label == None:   label = metric.__name__.replace('SSPB', '\\beta').replace('MdSA', '\\varepsilon\\thinspace').replace('Slope','S\\thinspace')
    if label == None:   label = metric.__name__.replace('SSPB', 'Bias').replace('MdSA', 'Error')
    if longest == None: longest = len(label)

    ispct = metric.__qualname__ in ['mape', 'sspb', 'mdsa'] # metrics which are percentages
    diff  = longest-len(label.replace('^',''))
    space = r''.join([r'\ ']*diff + [r'\thinspace']*diff)
    prec  = (1 if abs(metric(y_true, y_est)) < 100 and metric.__name__ not in ['N'] else 0) if ispct or metric.__name__ in ['N'] else 3
    # prec  = 1 if abs(metric(y_true, y_est)) < 100 else 0
    stat  = f'{metric(y_true, y_est):.{prec}f}'
    perc  = r'$\small{\mathsf{\%}}$' if ispct else ''
    return rf'$\mathtt{{{label}}}{space}:$ {stat}{perc}'

def _create_stats(y_true, y_est, metrics, title=None):
    ''' Create stat box strings for all metrics, assuming there is only a single target feature '''
    longest = max([len(metric.__name__.replace('SSPB', 'Bias').replace('MdSA', 'Error').replace('^','')) for metric in metrics])
    statbox = [_create_metric(m, y_true, y_est, longest=longest) for m in metrics]
    
    if title is not None:
        statbox = [rf'$\mathbf{{\underline{{{title}}}}}$'] + statbox
    return statbox 

def _create_multi_feature_stats(y_true, y_est, metrics, labels=None):
    ''' Create stat box strings for a single metric, assuming there are multiple target features '''
    if labels == None: 
        labels = [f'Feature {i}' for i in range(y_true.shape[1])]
    assert(len(labels) == y_true.shape[1] == y_est.shape[1]), f'Number of labels does not match number of features: {labels} - {y_true.shape}'
    
    title   = metrics[0].__name__.replace('SSPB', 'Bias').replace('MdSA', 'Error')
    longest = max([len(label.replace('^','')) for label in labels])
    statbox = [_create_metric(metrics[0], y1, y2, longest=longest, label=lbl) for y1, y2, lbl in zip(y_true.T, y_est.T, labels)]
    statbox = [rf'$\mathbf{{\underline{{{title}}}}}$'] + statbox
    return statbox 

def add_stats_box(ax, y_true, y_est, metrics=[mdsa, sspb, slope], bottom_right=False, bottom=False, right=False, x=0.025, y=0.97, fontsize=16, label=None):
    ''' Add a text box containing a variety of performance statistics, to the given axis '''
    import matplotlib.pyplot as plt
    plt.rc('text', usetex=True)
    plt.rcParams['mathtext.default']='regular'

    create_box = _create_stats if len(y_true.shape) == 1 or y_true.shape[1] == 1 else _create_multi_feature_stats
    stats_box  = '\n'.join( create_box(y_true, y_est, metrics, label) )
    ann_kwargs = {
        'transform'  : ax.transAxes,
        'textcoords' : 'offset points', 
        'xycoords'   : 'axes fraction', 
        'fontname'   : 'monospace', 
        'xytext'     : (0,0), 
        'zorder'     : 25,     
        'va'         : 'top', 
        'ha'         : 'left', 
        'bbox'       : {
            'facecolor' : 'white',
            'edgecolor' : 'black', 
            'alpha'     : 0.7,
        }
    }

    ann = ax.annotate(stats_box, xy=(x,y), size=fontsize, **ann_kwargs)

    bottom |= bottom_right
    right  |= bottom_right

    # Switch location to (approximately) the bottom right corner
    if bottom or right or bottom_right:
        plt.gcf().canvas.draw()
        bbox_orig = ann.get_tightbbox(plt.gcf().canvas.renderer).transformed(ax.transAxes.inverted())

        new_x = bbox_orig.x0
        new_y = bbox_orig.y1
        if bottom:
            new_y = bbox_orig.y1 - bbox_orig.y0 + (1 - y)
            ann.set_y(new_y)
            new_y += 0.06
        if right:
            new_x = 1 - (bbox_orig.x1 - bbox_orig.x0) + x
            ann.set_x(new_x)
            new_x -= 0.04
        ann.xy = (new_x, new_y)
    return ann 
    

def draw_map(*lonlats, scale=0.2, world=False, us=True, eu=False, labels=[], ax=None, gray=False, res='i', **scatter_kws):
    ''' Helper function to plot locations on a global map '''
    import matplotlib.pyplot as plt
    from matplotlib.transforms import Bbox
    from mpl_toolkits.axes_grid1.inset_locator import TransformedBbox, BboxPatch, BboxConnector 
    from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, inset_axes
    from mpl_toolkits.basemap import Basemap
    from itertools import chain

    PLOT_WIDTH  = 8
    PLOT_HEIGHT = 6

    WORLD_MAP = {'cyl': [-90, 85, -180, 180]}
    US_MAP    = {
        'cyl' : [24, 49, -126, -65],
        'lcc' : [23, 48, -121, -64],
    }
    EU_MAP    = {
        'cyl' : [34, 65, -12, 40],
        'lcc' : [30.5, 64, -10, 40],
    }

    def mark_inset(ax, ax2, m, m2, MAP, loc1=(1, 2), loc2=(3, 4), **kwargs):
        """
        https://stackoverflow.com/questions/41610834/basemap-projection-geos-controlling-mark-inset-location
        Patched mark_inset to work with Basemap.
        Reason: Basemap converts Geographic (lon/lat) to Map Projection (x/y) coordinates

        Additionally: set connector locations separately for both axes:
            loc1 & loc2: tuple defining start and end-locations of connector 1 & 2
        """
        axzoom_geoLims = (MAP['cyl'][2:], MAP['cyl'][:2]) 
        rect = TransformedBbox(Bbox(np.array(m(*axzoom_geoLims)).T), ax.transData)
        pp   = BboxPatch(rect, fill=False, **kwargs)
        ax.add_patch(pp)
        p1 = BboxConnector(ax2.bbox, rect, loc1=loc1[0], loc2=loc1[1], **kwargs)
        ax2.add_patch(p1)
        p1.set_clip_on(False)
        p2 = BboxConnector(ax2.bbox, rect, loc1=loc2[0], loc2=loc2[1], **kwargs)
        ax2.add_patch(p2)
        p2.set_clip_on(False)
        return pp, p1, p2


    if world:
        MAP    = WORLD_MAP
        kwargs = {'projection': 'cyl', 'resolution': res}
    elif us:
        MAP    = US_MAP
        kwargs = {'projection': 'lcc', 'lat_0':30, 'lon_0':-98, 'resolution': res}#, 'epsg':4269}
    elif eu:
        MAP    = EU_MAP
        kwargs = {'projection': 'lcc', 'lat_0':48, 'lon_0':27, 'resolution': res}
    else:
        raise Exception('Must plot world, US, or EU')

    kwargs.update(dict(zip(['llcrnrlat', 'urcrnrlat', 'llcrnrlon', 'urcrnrlon'], MAP['lcc' if 'lcc' in MAP else 'cyl'])))
    if ax is None: f = plt.figure(figsize=(PLOT_WIDTH, PLOT_HEIGHT), edgecolor='w')
    m  = Basemap(ax=ax, **kwargs)
    ax = m.ax if m.ax is not None else plt.gca()

    if not world:
        m.readshapefile(Path(__file__).parent.joinpath('map_files', 'st99_d00').as_posix(), name='states', drawbounds=True, color='k', linewidth=0.5, zorder=11)
        m.fillcontinents(color=(0,0,0,0), lake_color='#9abee0', zorder=9)
        if not gray:
            m.drawrivers(linewidth=0.2, color='blue', zorder=9)
        m.drawcountries(color='k', linewidth=0.5)
    else:
        m.drawcountries(color='w')
    # m.bluemarble()
    if not gray:
        if us or eu: m.shadedrelief(scale=0.3 if world else 1)
        else:
            # m.arcgisimage(service='ESRI_Imagery_World_2D', xpixels = 2000, verbose= True)
            m.arcgisimage(service='World_Imagery', xpixels = 2000, verbose= True)
    else:
        pass
    # lats = m.drawparallels(np.linspace(MAP[0], MAP[1], 13))
    # lons = m.drawmeridians(np.linspace(MAP[2], MAP[3], 13))

    # lat_lines = chain(*(tup[1][0] for tup in lats.items()))
    # lon_lines = chain(*(tup[1][0] for tup in lons.items()))
    # all_lines = chain(lat_lines, lon_lines)
    
    # for line in all_lines:
    #     line.set(linestyle='-', alpha=0.0, color='w')

    if labels:
        colors = ['aqua', 'orangered',  'xkcd:tangerine', 'xkcd:fresh green', 'xkcd:clay', 'magenta', 'xkcd:sky blue', 'xkcd:greyish blue', 'xkcd:goldenrod', ]
        markers = ['o', '^', 's', '*',  'v', 'X', '.', 'x',]
        mod_cr = False
        assert(len(labels) == len(lonlats)), [len(labels), len(lonlats)]
        for i, (label, lonlat) in enumerate(zip(labels, lonlats)):
            lonlat = np.atleast_2d(lonlat)
            if 'color' not in scatter_kws or mod_cr:
                scatter_kws['color'] = colors[i]
                scatter_kws['marker'] = markers[i]
                mod_cr = True
            ax.scatter(*m(lonlat[:,0], lonlat[:,1]), label=label, zorder=12, **scatter_kws)    
        ax.legend(loc='lower left', prop={'weight':'bold', 'size':8}).set_zorder(20)

    else:
        for lonlat in lonlats:
            if len(lonlat):
                lonlat = np.atleast_2d(lonlat)
                s = ax.scatter(*m(lonlat[:,0], lonlat[:,1]), zorder=12, **scatter_kws)
                # plt.colorbar(s, ax=ax)
    hide_kwargs = {'axis':'both', 'which':'both'}
    hide_kwargs.update(dict([(k, False) for k in ['bottom', 'top', 'left', 'right', 'labelleft', 'labelbottom']]))
    ax.tick_params(**hide_kwargs)

    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1.5)
        ax.spines[axis].set_zorder(50)
    # plt.axis('off')

    if world:
        size = 0.35
        if us:
            loc = (0.25, -0.1) if eu else (0.35, -0.01)
            ax_ins = inset_axes(ax, width=PLOT_WIDTH*size, height=PLOT_HEIGHT*size, loc='center', bbox_to_anchor=loc, bbox_transform=ax.transAxes, axes_kwargs={'zorder': 5})
            
            scatter_kws.update({'s': 6})
            m2 = draw_map(*lonlats, labels=labels, ax=ax_ins, **scatter_kws)
            
            mark_inset(ax, ax_ins, m, m2, US_MAP, loc1=(1,1), loc2=(2,2), edgecolor='grey', zorder=3)
            mark_inset(ax, ax_ins, m, m2, US_MAP, loc1=[3,3], loc2=[4,4], edgecolor='grey', zorder=0)


        if eu:
            ax_ins = inset_axes(ax, width=PLOT_WIDTH*size, height=PLOT_HEIGHT*size, loc='center', bbox_to_anchor=(0.75, -0.05), bbox_transform=ax.transAxes, axes_kwargs={'zorder': 5})
            
            scatter_kws.update({'s': 6})
            m2 = draw_map(*lonlats, us=False, eu=True, labels=labels, ax=ax_ins, **scatter_kws)
            
            mark_inset(ax, ax_ins, m, m2, EU_MAP, loc1=(1,1), loc2=(2,2), edgecolor='grey', zorder=3)
            mark_inset(ax, ax_ins, m, m2, EU_MAP, loc1=[3,3], loc2=[4,4], edgecolor='grey', zorder=0)

    return m


def default_dd(d={}, f=lambda k: k):
    ''' Helper function to allow defaultdicts whose default value returned is the queried key '''

    class key_dd(dd):
        ''' DefaultDict which allows the key as the default value '''
        def __missing__(self, key):
            if self.default_factory is None:
                raise KeyError(key)
            val = self[key] = self.default_factory(key)
            return val 
    return key_dd(f, d)


@ignore_warnings
def plot_scatter(y_test, benchmarks, bands, labels, products, sensor, title=None, methods=None, n_col=3, img_outlbl='',args=None):
    import matplotlib.patheffects as pe 
    import matplotlib.ticker as ticker
    import matplotlib.pyplot as plt 
    import seaborn as sns 

    folder = Path('scatter_plots')
    folder.mkdir(exist_ok=True, parents=True)

    product_labels = default_dd({
        'chl' : 'Chl\\textit{a}',
        'aph' : 'a_{\\textit{ph}}',
        'ad' : 'a_{\\textit{nap}}',
        'ag' : 'a_{\\textit{cdom}}',
        'tss' : 'TSS',
        'pc' : 'PC',
        'cdom': 'CDOM',#'a_{\\textit{cdom}}(443)',   
    })
    
    product_units = default_dd({
        'chl' : '[mg m^{-3}]',
        'pc' : '[mg m^{-3}]',
        'tss' : '[g m^{-3}]',
        'aph' : '[m^{-1}]',
        'ad' : '[m^{-1}]',
        'ag' : '[m^{-1}]',

        'cdom': '[m^{-1}]',
    }, lambda k: '')

    model_labels = default_dd({
        'MDN' : 'MDN',
    })

    products = [p for i,p in enumerate(np.atleast_1d(products)) if i < y_test.shape[-1]]

    plt.rc('text', usetex=True)
    plt.rcParams['mathtext.default']='regular'
    # plt.rcParams['mathtext.fontset'] = 'stix'
    # plt.rcParams['font.family'] = 'cm'

    # Only plot certain bands
    if len(labels) > 3 :
        product_bands = {
            'default' :  [443,482,561,655] if 'OLI'  in sensor else [443,490,560,620,673,710]  if 'OLCI' in sensor  else [443,530,620,673,]  if args.use_HICO_aph  else  [409,443,478,535,620,673,690,724]  #[415,443,490,501,550,620,673,690] #[443,482,561,655]
            # 'aph'     : [443, 530], [409, 421, 432, 444, 455, 467, 478, 490, 501, 512, 524, 535, 547, 558, 570, 581, 593, 604, 616, 621, 633, 644, 650, 656, 667, 673, 679, 684, 690, 701, 713, 724,]
        }

        target     = [closest_wavelength(w, bands if not args.use_HICO_aph else get_sensor_bands(f'{sensor}-aph', args) ) for w in product_bands.get(products[0], product_bands['default'])]
        print('Target',target)
        plot_label = [w in target for w in bands]
        if args.use_HICO_aph: 
            plot_label_aph = [w in target for w in get_sensor_bands(f'{sensor}-aph', args)]
            #y_test=y_test[:,[ i in get_sensor_bands('HICO-aph', args) for i in  get_sensor_bands(sensor, args)]]
        if args.use_HICO_aph and (products[0]=='ad' or products[0]=='ag'): 
                product_bands = {
                    'default' :  [get_sensor_bands(f'{sensor}-adag', args)[0],get_sensor_bands(f'{sensor}-adag', args)[2],get_sensor_bands(f'{sensor}-adag', args)[4]] #[415,444,478,535,564,593,621]#list(get_sensor_bands('HICO-adag', args))
                }
                target     = [closest_wavelength(w, bands if not args.use_HICO_aph else get_sensor_bands(f'{sensor}-adag', args)) for w in product_bands.get(products[0], product_bands['default'])]
                plot_label_aph = [w in target for w in get_sensor_bands(f'{sensor}-adag', args)]
        plot_label = [w in target for w in bands]

        plot_order =  ['MDN','QAA','GIOP']
        plot_order = [val  for i,val in enumerate(plot_order) if val in benchmarks[products[0]].keys()]
        
        plot_bands = True
    else:
        plot_label = [True] * len(labels)
        plot_order = methods
        plot_bands = False

        if plot_order is None:
            if len(products) == 1:
                plot_order = ['MDN']
            if 'chl' in products and len(products) == 1:
            #     # benchmarks = benchmarks['chl']
            #     if 'MLP' in benchmarks:
                plot_order = ['MDN', 'Gilerson_2band', 'Smith_Blend']  if 'OLI' not in sensor else   ['MDN',]   #OC3 GIOP , 'Gilerson_2band', 'Smith_Blend'
            if 'tss' in products and len(products) == 1:
                #     # benchmarks = benchmarks['chl']
                #     if 'MLP' in benchmarks:
                    plot_order = ['MDN', 'Nechad', 'Novoa']
            if 'cdom' in products and len(products) == 1:
                #     # benchmarks = benchmarks['chl']
                #     if 'MLP' in benchmarks:
                    plot_order = ['MDN', 'QAA_CDOM', 'Ficek']
            if 'pc' in products and len(products) == 1:
                #     # benchmarks = benchmarks['chl']
                #     if 'MLP' in benchmarks:
                    plot_order = ['MDN','Schalles','Sim2005'] if 'OLI' not in sensor else  ['MDN',]  #'Schalles','Sim2005'
            if len(products) == 1:
                plot_order = [val  for i,val in enumerate(plot_order) if val in benchmarks[products[0]].keys()]
            #     else:
            #         plot_order = ['MDN']
            # elif len(products) == 3 and all(k in products for k in ['chl', 'tss', 'cdom']):
            #     n_col = 3
            #     plot_order = {
            #         'chl'  : ['MDN', 'OC3','Smith_Blend'],
            #         'tss'  : ['MDN', 'SOLID', 'Novoa'],
            #         'cdom' : ['MDN', 'Ficek', 'Mannino'],
            #     }
            #     plot_label = [True] * 3
            #     plot_bands = True
            # elif len(products) == 3 and all(k in products for k in ['chl', 'pc', 'ad']):
            #     n_col = 3
            #     plot_order = {
            #         'chl'  : ['MDN',],
            #         'pc'  : ['MDN',],
            #     #    'ad' : ['MDN',],
            #     }
            #     plot_label = [True] * 3
            #     plot_bands = True

    labels = [(p,label) for label in labels for p in products if p in label]
    print('Plotting labels:', [l for i,l in enumerate(labels)] ) #if plot_label[i]]
    assert(len(labels) == y_test.shape[-1]), [len(labels), y_test.shape]

    # plot_order = [p for p in plot_order if p in benchmarks]
    fig_size   = 5
    n_col      = max(n_col, sum(plot_label))
    n_row      = max(1,int(not plot_bands) + int(0.5 + len(plot_order) / (1 if plot_bands else n_col)) - int(not plot_bands))
    if isinstance(plot_order, dict): n_row = 3
    if plot_bands:
        n_col, n_row = n_row, n_col

    fig, axes = plt.subplots(n_row, n_col, figsize=(fig_size*n_col, fig_size*n_row+1))
    axes      = [ax for axs in np.atleast_1d(axes) for ax in np.atleast_1d(axs)]
    colors    = ['xkcd:sky blue', 'xkcd:tangerine', 'xkcd:fresh green', 'xkcd:greyish blue', 'xkcd:goldenrod',  'xkcd:clay', 'xkcd:bluish purple', 'xkcd:reddish']

    print('Order:', plot_order)
    print(f'Plot size: {n_row} x {n_col}')
    print(labels)

    curr_idx = 0
    full_ax  = fig.add_subplot(111, frameon=False)
    full_ax.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False, pad=10)

    estimate_label = 'Estimated' #'Satellite-derived'
    x_pre  = 'Measured'
    y_pre  = estimate_label.replace('-', '\\textbf{-}')
    plabel = f'{product_labels[products[0]]} {product_units[products[0]]}'
    xlabel = fr'$\mathbf{{{x_pre} {plabel} }}$'
    ylabel = fr'$\mathbf{{{y_pre}}}$'+'' +fr'$\mathbf{{ {plabel}}}$'
    if not isinstance(plot_order, dict):
        full_ax.set_xlabel(xlabel.replace(' ', '\ '), fontsize=20, labelpad=10)
        full_ax.set_ylabel(ylabel.replace(' ', '\ '), fontsize=20, labelpad=10)
    else:
        full_ax.set_xlabel(fr'$\mathbf{{{x_pre} Product}}$'.replace(' ', '\ '), fontsize=20, labelpad=10)

    s_lbl = title or get_sensor_label(sensor).replace('-',' ')
    n_pts = len(y_test)
    title = fr'$\mathbf{{\underline{{\large{{{s_lbl}}}}}}}$' + '\n' + fr'$\small{{\mathit{{N\small{{=}}}}{n_pts}}}$'
    # full_ax.set_title(title.replace(' ', '\ '), fontsize=24, y=1.06)

    if isinstance(plot_order, dict):
        full_ax.set_title(fr'$\mathbf{{\underline{{\large{{{s_lbl}}}}}}}$'.replace(' ', '\ '), fontsize=24, y=1.03)

    for plt_idx, (label, y_true) in enumerate(zip(labels, y_test.T)):
        


        product, title = label 
        if args.use_HICO_aph and (product=='aph' or product=='ag' or product=='ad'):
            if not plot_label_aph[plt_idx]: continue 
        else:
            if not plot_label[plt_idx]: continue 
        plabel = f'{product_labels[product]} {product_units[product]}'

        for est_idx, est_lbl in enumerate(plot_order[product] if isinstance(plot_order, dict) else plot_order):
            # if plt_idx >= (len(plot_order[product]) if isinstance(plot_order, dict) else benchmarks[product][est_lbl].shape[1]): continue
            if isinstance(plot_order, dict) and est_lbl not in benchmarks[product]: 
                axes[curr_idx].tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
                axes[curr_idx].axis('off')
                curr_idx += 1
                continue 
            if est_lbl =='GIOP' or est_lbl =='QAA':
                # valid_bool = []
                # wvls = get_sensor_bands(sensor, args)
                valid_bool = []
                required_wvl = get_sensor_bands(f'{sensor}-adag', args)
                wvls = get_sensor_bands(sensor, args)
                for required_wvl_i in required_wvl:
                    wvl_diff = [i.item() for i in np.abs(wvls - required_wvl_i)]
                    wvl_diff_bool = [wvl_diff_i < 4 for wvl_diff_i in wvl_diff]
                    valid_bool.append([wvl_diff_bool_i if np.argmin(wvl_diff) == i else False for  i,wvl_diff_bool_i in enumerate(wvl_diff_bool) ])
                
                valid_wvl_ad_ag =np.any(np.array(valid_bool),axis=0)
                
                if not args.use_HICO_aph or (np.shape(benchmarks[product][est_lbl])[1] !=  np.shape(get_sensor_bands(f'{sensor}-aph', args))[0] and args.use_HICO_aph and product=='aph')  or (np.shape(benchmarks[product][est_lbl])[1] !=  np.shape(get_sensor_bands(f'{sensor}-adag', args))[0] and args.use_HICO_aph and (product=='ad' or product=='ag' ) ):
                    benchmarks[product][est_lbl] = np.reshape(benchmarks[product][est_lbl],(-1,np.shape(get_sensor_bands(sensor, args))[0]))   #if not args.use_HICO_aph  else np.reshape(benchmarks[product][est_lbl],(-1,np.shape(get_sensor_bands('HICO-aph', args))[0])) #if np.shape(benchmarks[product][est_lbl])[0] == 1786*np.shape(get_sensor_bands(sensor, args))[0] else  np.reshape(benchmarks[product][est_lbl],(1786,np.shape(get_sensor_bands(sensor, args))[0])) #np.shape(benchmarks[product]['MDN']))
                #Resample to HICO_aph wavelengths
                if args.use_HICO_aph and np.shape(benchmarks[product][est_lbl])[1] ==  np.shape(get_sensor_bands(sensor, args))[0] and product == 'aph':
                    benchmarks[product][est_lbl] = benchmarks[product][est_lbl][:,[ i in get_sensor_bands(f'{sensor}-aph', args) for i in  get_sensor_bands(sensor, args)]]
                if args.use_HICO_aph and np.shape(benchmarks[product][est_lbl])[1] ==  np.shape(get_sensor_bands(sensor, args))[0] and (product == 'ad' or  product == 'ag'):
                    benchmarks[product][est_lbl] = benchmarks[product][est_lbl][:,valid_wvl_ad_ag ] #[ i in get_sensor_bands(f'{sensor}-adag', args) for i in  get_sensor_bands(sensor, args)]]            


            
            
            y_est = benchmarks[product][est_lbl] if isinstance(plot_order, dict) else benchmarks[product][est_lbl][..., plt_idx]
            ax    = axes[curr_idx]
            cidx  = (curr_idx % n_col) if plot_bands else curr_idx
            color = colors[cidx]

            first_row = curr_idx < n_col #(curr_idx % n_row) == 0
            last_row  = curr_idx >= ((n_row-1)*n_col) #((curr_idx+1) % n_row) == 0
            first_col = (curr_idx % n_col) == 0
            last_col  = ((curr_idx+1) % n_col) == 0
            print(curr_idx, first_row, last_row, first_col, last_col, est_lbl, product, plabel, sum(np.isfinite(y_true)))
            if not (('Scdom443' in products and len(products)==1) or ('Snap443' in products and len(products)==1)):

                y_est_log  = np.log10(y_est).flatten()
                y_true_log = np.log10(y_true).flatten()
            else:
                y_est_log  = (y_est).flatten()
                y_true_log = (y_true).flatten()                
            curr_idx  += 1

            l_kws = {'color': color, 'path_effects': [pe.Stroke(linewidth=4, foreground='k'), pe.Normal()], 'zorder': 22, 'lw': 1}
            s_kws = {'alpha': 0.4, 'color': color}#, 'edgecolor': 'grey'}

            if est_lbl == 'MDN':
                [i.set_linewidth(5) for i in ax.spines.values()]
                est_lbl = 'MDN'
                # est_lbl = 'MDN-I'
            else:
                est_lbl = est_lbl.replace('Mishra_','').replace('Gons_2band', 'Gons').replace('Gilerson_2band', 'GI2B').replace('Smith_','').replace('Cao_XGB','BST')#.replace('Cao_', 'Cao\ ')

            if product not in ['chl', 'tss', 'cdom','pc','nap','SCDOM','SNAP'] and last_col:
                ax2 = ax.twinx()
                ax2.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False, pad=0)
                ax2.grid(False)
                ax2.set_yticklabels([])
                if args.use_HICO_aph and product == 'aph':
                    bands_aph = get_sensor_bands('HICO-aph', args)
                    ax2.set_ylabel(fr'$\mathbf{{{bands_aph[plt_idx]:.0f}nm}}$' f"\n" f"N={sum(np.isfinite(y_true))}", fontsize=20)
                else:
                    if args.use_HICO_aph and (product == 'ad' or product == 'ag' ) :
                        bands_adag = get_sensor_bands('HICO-adag', args)
                        ax2.set_ylabel(fr'$\mathbf{{{bands_adag[plt_idx]:.0f}nm}}$' f"\n" f"N={sum(np.isfinite(y_true))}", fontsize=20)
                    else:
                        ax2.set_ylabel(fr'$\mathbf{{{bands[plt_idx]:.0f}nm}}$' f"\n" f"N={sum(np.isfinite(y_true))}", fontsize=20)



            minv = -3 if product == 'cdom' else 0.000 if product == 'Scdom443' else .005 if product == 'Snap443' else int(np.nanmin(y_true_log)) - 1 if product != 'aph' and product != 'ag' and product != 'ad'  else -4
            maxv = 3 if product == 'tss' else 4 if product == 'chl' else .035 if product == 'Scdom443' else .02 if product == 'Snap443' else int(np.nanmax(y_true_log)) + 1 if product != 'aph' and product != 'ag' and product != 'ad'  else 2
            loc  = ticker.LinearLocator(numticks=int(round(maxv-minv+1)))
            fmt  = ticker.FuncFormatter(lambda i, _: r'$10$\textsuperscript{%i}'%i)
            
            ax.set_ylim((minv, maxv))
            ax.set_xlim((minv, maxv))
            ax.xaxis.set_major_locator(loc)
            ax.yaxis.set_major_locator(loc)
            if not (('Scdom443' in products and len(products)==1) or ('Snap443' in products and len(products)==1)):
                ax.xaxis.set_major_formatter(fmt)
                ax.yaxis.set_major_formatter(fmt)
            else:
                if product == 'Scdom443':
                    ax.xaxis.set_ticks(np.arange(0, 0.03, 0.01))
                    ax.yaxis.set_ticks(np.arange(0, 0.03, 0.01))
                if product == 'Snap443':
                    ax.xaxis.set_ticks(np.arange(0.0025, 0.02, 0.01))
                    ax.yaxis.set_ticks(np.arange(0.0025, 0.02, 0.01))
            # if not last_row:                   ax.set_xticklabels([])
            # elif isinstance(plot_order, dict): ax.set_xlabel(fr'$\mathbf{{{x_pre}}}$'+'' +fr'$\mathbf{{ {plabel}}}$'.replace(' ', '\ '), fontsize=18)
            if not first_col:                  ax.set_yticklabels([])
            elif isinstance(plot_order, dict): 
                ylabel = fr'$\mathbf{{{y_pre}}}$'+'' +fr'$\mathbf{{ {plabel}}}$' + '\n' + fr'$\small{{\mathit{{N\small{{=}}}}{np.isfinite(y_true_log).sum()}}}$'
                ax.set_ylabel(ylabel.replace(' ', '\ '), fontsize=18)

            valid = np.logical_and(np.isfinite(y_true_log), np.isfinite(y_est_log))
            if valid.sum():
                sns.regplot(y_true_log[valid], y_est_log[valid], ax=ax, scatter_kws=s_kws, line_kws=l_kws, fit_reg=True, truncate=False, robust=True, ci=None)
                kde = sns.kdeplot(y_true_log[valid], y_est_log[valid], shade=False, ax=ax, bw='scott', n_levels=4, legend=False, gridsize=100, color='k')
                # kde.collections[2].set_alpha(0)

            invalid = np.logical_and(np.isfinite(y_true_log), ~np.isfinite(y_est_log))
            if invalid.sum():
                ax.scatter(y_true_log[invalid], [minv]*(invalid).sum(), color='r', alpha=0.4, label=r'$\mathbf{%s\ invalid}$' % (invalid).sum())
                ax.legend(loc='lower right', prop={'weight':'bold', 'size': 16})

            add_identity(ax, ls='--', color='k', zorder=20)

            if valid.sum():
                add_stats_box(ax, y_true[valid], y_est[valid],metrics=[mdsa,sspb,slope,rmsle])
                # if plot_order[est_idx] == 'MDN' and product == 'aph': 
                #     if plot_order[est_idx] not in args.summary_stats.keys(): args.summary_stats[ plot_order[est_idx]] = {}
                #     args.summary_stats[ plot_order[est_idx]][label[1]] = create_stats_HPC(y_true[valid], y_est[valid], metrics=[mdsa, sspb, slope],label=None)
                # else:
                if plot_order[est_idx] not in args.summary_stats.keys(): args.summary_stats[ plot_order[est_idx]] = {}
                args.summary_stats[ plot_order[est_idx]][label[1]] = create_stats_HPC(y_true[valid], y_est[valid], metrics=[mdsa, sspb, slope],label=None)


            if first_row or not plot_bands or (isinstance(plot_order, dict) and plot_order[product][est_idx] != 'MDN'):
                if est_lbl == 'BST':
                    # ax.set_title(fr'$\mathbf{{\large{{{est_lbl}}}}}$'+'\n'+r'$\small{\textit{(Cao\ et\ al.\ 2020)}}$', fontsize=18)
                    ax.set_title(r'$\small{\textit{(Cao\ et\ al.\ 2020)}}$' + '\n' + fr'$\mathbf{{\large{{{est_lbl}}}}}$', fontsize=18, linespacing=0.95)
                
                elif est_lbl == 'QAA_CDOM':
                    # ax.set_title(fr'$\mathbf{{\large{{{est_lbl}}}}}$'+'\n'+r'$\small{\textit{(Cao\ et\ al.\ 2020)}}$', fontsize=18)
                    ax.set_title( r'$\mathbf{{\large{QAA}}}' + r'$\small{\textit{CDOM}', fontsize=18, linespacing=0.95)
          
                elif est_lbl == 'Ficek':
                    # ax.set_title(fr'$\mathbf{{\large{{{est_lbl}}}}}$'+'\n'+r'$\small{\textit{(Cao\ et\ al.\ 2020)}}$', fontsize=18)
                    ax.set_title(fr'$\mathbf{{\large{{{est_lbl}}}}}$' + r'$\small{\textit{\ (et\ al.\ 2011)}}$', fontsize=18, linespacing=0.95)
                                
                elif est_lbl == 'Mannino':
                    # ax.set_title(fr'$\mathbf{{\large{{{est_lbl}}}}}$'+'\n'+r'$\small{\textit{(Cao\ et\ al.\ 2020)}}$', fontsize=18)
                    ax.set_title(fr'$\mathbf{{\large{{{est_lbl}}}}}$' + r'$\small{\textit{\ (et\ al.\ 2008)}}$', fontsize=18, linespacing=0.95)

                elif est_lbl == 'Novoa':
                    # ax.set_title(fr'$\mathbf{{\large{{{est_lbl}}}}}$'+'\n'+r'$\small{\textit{(Cao\ et\ al.\ 2020)}}$', fontsize=18)
                    ax.set_title(fr'$\mathbf{{\large{{{est_lbl}}}}}$' + r'$\small{\textit{\ (et\ al.\ 2017)}}$', fontsize=18, linespacing=0.95)

                else: ax.set_title(fr'$\mathbf{{\large{{{est_lbl}}}}}$' '\n' f'(N={sum(np.isfinite(y_true))})', fontsize=18) if first_col else  ax.set_title(fr'$\mathbf{{\large{{{est_lbl}}}}}$' , fontsize=18)
            # ax.x
            ax.tick_params(labelsize=18)
            ax.grid('on', alpha=0.3)
            from pylab import text
            #text(0.9,0.1,f'N={sum(np.isfinite(y_true))}',fontsize=12)
            
    u_label  = ",".join([o.split('_')[0] for o in plot_order]) if len(plot_order) < 10 else f'{n_row}x{n_col}'
    out_dir = args.config_name
    print('OUTLBL is: ',out_dir)
    import os
    os.makedirs(str(folder) + '/' + out_dir,exist_ok=True)
    filename = folder.joinpath(f'{out_dir}/{img_outlbl}{",".join(products)}_{sensor}_{n_pts}test_{u_label}.png')
    plt.tight_layout()
    # plt.subplots_adjust(wspace=0.35)
    plt.savefig(filename.as_posix(), dpi=600, bbox_inches='tight', pad_inches=0.1,)
    plt.show()

def plot_spectra(y_test, benchmarks, bands, labels, products, sensor, title=None, methods=None, n_col=3, img_outlbl='',args=None,y_full=None,slices=None):
    aph_wavelengths = get_sensor_bands('HICO-aph', args) if args.use_HICO_aph else get_sensor_bands(sensor, args)
    ag_wavelengths = get_sensor_bands('HICO-adag', args) if args.use_HICO_aph else get_sensor_bands(sensor, args)
    product_labels = default_dd({
        'chl' : 'Chl\\textit{a}',
        'aph' : '\\textit{a}_{ph}',
        'ad' : '\\textit{a}_{d}',
        'ag' : '\\textit{a}_{g}',
        'tss' : 'TSS',
        'pc' : 'PC',
        'cdom': '\\textit{a}_{CDOM}',   
    })
    
    product_units = default_dd({
        'chl' : '[mg m^{-3}]',
        'pc' : '[mg m^{-3}]',
        'tss' : '[g m^{-3}]',
        'aph' : '[m^{-1}]',
        'ad' : '[m^{-1}]',
        'ag' : '[m^{-1}]',

        'cdom': '[m^{-1}]',
    }, lambda k: '')
    
    font = {'family' : 'normal',
            'weight' : 'bold',
            'size'   : 20}
    
    import matplotlib.pyplot as plt
    from matplotlib.figure import figaspect
    import scipy.stats as stats

    folder = Path('scatter_plots')
    folder.mkdir(exist_ok=True, parents=True)
    colors = ['aqua', 'orangered',  'xkcd:tangerine', 'xkcd:fresh green', 'xkcd:clay', 'magenta', 'xkcd:sky blue', ]
    index = 1
    for index in range(len(y_test)):
        if sum(np.isnan(y_test[index,:])) or sum(np.isnan(y_full[:,slices['chl']][index,:])) or sum(np.isnan(y_full[:,slices['pc']][index,:])): continue
        if index not in [1297,2099,2263]: continue
        W, H = figaspect(0.4)
        fig = plt.figure(figsize=(1.5*W, 1.5*H))
        
        ax = fig.add_subplot(131)
        ax.grid(color='black',alpha=0.1)
        ax.set_axisbelow(True)
        ax.plot(aph_wavelengths,benchmarks['aph']['MDN'][index,:],label='MDN',color=colors[3])
        ax.plot(aph_wavelengths,benchmarks['aph']['GIOP'][index,:],label='GIOP',color=colors[1])
        ax.plot(aph_wavelengths,benchmarks['aph']['QAA'][index,:],label='QAA',color=colors[2])
        ax.plot(aph_wavelengths,y_test[index,:],label='Truth',color='k')
        plabel = f'{product_labels["aph"]} {product_units["aph"]}'
    
        if index  in [2263]: ax.set_title(fr'$\mathbf{{{plabel}}}$',fontsize=28)
        plabel_x = f'Wavelength \ [nm]'
        # ax.set_xlabel(fr'$\mathbf{{{plabel_x}}}$')
        chl_conc = str(round(y_full[:,slices['chl']][index][0],1))
        chl_conc = f'{chl_conc:>7}'.replace(" ","\ ")
        pc_conc =  str(round(y_full[:,slices['pc']][index][0],1))
        pc_conc = f'{pc_conc:>7}'.replace(" ","\ ")

        cdom_conc = str(round(y_full[:,slices['cdom']][index][0],1))
        cdom_conc = f'{cdom_conc:>5}'.replace(" ","\ ")

        tss_conc = str(round(y_full[:,slices['tss']][index][0],1))
        tss_conc = f'{tss_conc:>7}'.replace(" ","\ ")

        from pylab import text

        if index  in [2263]: ax.legend()
        ax.set_xlim([400,700])
    
        ax = fig.add_subplot(132)
    
        ax.grid(color='black',alpha=0.1)
        ax.set_axisbelow(True)
        ax.plot(ag_wavelengths,benchmarks['ag']['MDN'][index,:],label='MDN',color=colors[3])
        QAA =  np.reshape(benchmarks['ag']['QAA'],(-1,np.shape(get_sensor_bands(sensor, args))[0]))
        ax.plot(bands,QAA[index,:],label='QAA',color=colors[2])
        ax.plot(ag_wavelengths,y_full[:,slices['ag']][index,:],label='Truth',color='k')
        plabel = f'{product_labels["ag"]} {product_units["ag"]}'
        
        if index  in [2263]:ax.set_title(fr'$\mathbf{{{plabel}}}$',fontsize=28)
        plabel_x = f'Wavelength \ [nm]'
        if index not in [2099,2263]: ax.set_xlabel(fr'$\mathbf{{{plabel_x}}}$',fontsize=28)
        ax.set_xlim([400,700])
        chl_text = "Chl:" 
        text(0.65,0.865,f'Chl:\ \ \ \ {{{{{chl_conc}}}}}  \n PC: \ \ \ \ {{{pc_conc}}} \n CDOM: {{{cdom_conc}}} \nTSS: \ \ {tss_conc}   ',horizontalalignment='left',verticalalignment='center',transform=ax.transAxes,backgroundcolor='1.0',bbox=dict(facecolor='white',edgecolor='black',boxstyle='round'),fontdict=font)

        ax = fig.add_subplot(133)
    
        ax.grid(color='black',alpha=0.1)
        ax.set_axisbelow(True)
        ax.plot(ag_wavelengths,benchmarks['ad']['MDN'][index,:],label='MDN',color=colors[3])
        ax.plot(ag_wavelengths,y_full[:,slices['ad']][index,:],label='Truth',color='k')
        plabel = f'{product_labels["ad"]} {product_units["ad"]}'
        
        if index  in [2263]: ax.set_title(fr'$\mathbf{{{plabel}}}$',fontsize=28)
        plabel_x = f'Wavelength \ [nm]'
        # ax.set_xlabel(fr'$\mathbf{{{plabel_x}}}$')
        ax.set_xlim([400,700])

        out_dir = args.config_name
        print('OUTLBL is: ',out_dir)
        import os
        os.makedirs(str(folder) + '/' + out_dir,exist_ok=True)
        filename = folder.joinpath(f'{out_dir}/spectral_products/{img_outlbl}Spectral_products_{sensor}_{index}_test.png')    
        plt.tight_layout()
        # plt.subplots_adjust(wspace=0.35)
        plt.savefig(filename.as_posix(), dpi=400, bbox_inches='tight', pad_inches=0.1,)
        plt.show()  
        plt.close('all')
                                                

def plot_histogram(product_values,products,slices,locs):
    PC_loc = slices["pc"]
    chl_loc = slices["chl"]
    cdom_loc = slices["cdom"]
    tss_loc = slices["tss"]
    
    
    import matplotlib.pyplot as plt
    from matplotlib.figure import figaspect
    import scipy.stats as stats
    folder = Path('scatter_plots')
    folder.mkdir(exist_ok=True, parents=True)
    W, H = figaspect(0.4)
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_subplot(221)
    ax.grid(color='black',alpha=0.1)
    ax.set_axisbelow(True)
    colors = ['aqua', 'orangered',  'xkcd:tangerine', 'xkcd:fresh green', 'xkcd:clay', 'magenta', 'xkcd:sky blue', ]
    product_labels = default_dd({
        'chl' : 'chl\\textit{a}',
        'pc'  : 'PC',
        'tss'  : 'TSS',
        'cdom'  : 'CDOM',#'a_{\mathit{cdom}}(443)',

        'chl/pc': 'Chl\\textit{a}:PC',
        'pc/chl': 'PC:Chl\\textit{a}',

    })
    product_units = default_dd({
        'chl' : '[mg m^{-3}]',
        'pc' : '[mg m^{-3}]',

        'tss' : '[g m^{-3}]',
        'cdom'  : '[m^{-1}]',

        'aph' : '[m^{-1}]',
    }, lambda k: '')
    plt_idx = 0
    plabel = f'{product_labels["chl"]} {product_units["chl"]}'
    xlabel = fr'$\mathbf{{{plabel}}}$'

    bin_locations = np.linspace(-1,3)
    x_vals = np.log10(product_values[np.squeeze(~np.isnan(product_values[:,chl_loc])),chl_loc])
    # ax.set_title(f'N={sum(np.isfinite(x_vals))}')
    n,bins,patches = ax.hist(x_vals,bins=bin_locations,facecolor='xkcd:dark mint green',edgecolor='white',linewidth=0.5,density=False,log=False,alpha=0.75)
    text(min(bins*.75),max(n*.9),f'N={sum(np.isfinite(x_vals))[0]}',fontsize=12)

    ax.set_xlabel(xlabel.replace(' ', '\ '),fontsize=15)
    ylabel = fr'$\mathbf{{Frequency}}$'

    # ax.set_ylabel(ylabel.replace(' ', '\ '),fontsize=15)
    ax.tick_params(axis='both',which='minor',labelsize=13)
    ax.tick_params(axis='both',which='major',labelsize=13)
    ax.set_facecolor('xkcd:white')

    labels = ax.get_xticklabels(which='both')
    locs = ax.get_xticks()

    xtick_labels = [int(value) for value in locs] 
    xtick_labels = [fr'${{10^{ {value} }}}$' for value in xtick_labels]

    ax.set_xticks(locs)
    ax.set_xticklabels(xtick_labels)
    ax.set_xlim((-1,3))

    plt_idx = 1
    plabel = f'{product_labels["pc"]} {product_units["pc"]}'
    xlabel = fr'$\mathbf{{{plabel}}}$'

    ax = fig.add_subplot(222)
    ax.grid(color='black',alpha=0.1)
    ax.set_axisbelow(True)
    x_vals = np.log10(product_values[:,PC_loc])
    # ax.set_title(f'N={sum(np.isfinite(x_vals))}')
    # text(-0.75,200,f'N={sum(np.isfinite(x_vals))[0]}',fontsize=12)

    n,bins,patches  = ax.hist(x_vals,bins=bin_locations,facecolor='xkcd:cool blue',edgecolor='white',linewidth=0.5,density=False,log=False,alpha=0.75)
    text(min(bins*.75),max(n*.9),f'N={sum(np.isfinite(x_vals))[0]}',fontsize=12)
    ax.set_xlabel(xlabel.replace(' ', '\ '),fontsize=15)
    ax.tick_params(axis='both',which='minor',labelsize=13)
    ax.tick_params(axis='both',which='major',labelsize=13)
    ax.set_facecolor('xkcd:white')

    labels = ax.get_xticklabels(which='both')
    locs = ax.get_xticks()

    xtick_labels = [int(value) for value in locs] 
    xtick_labels = [fr'${{10^{ {value} }}}$' for value in xtick_labels]
    ax.set_xticks(locs)
    ax.set_xticklabels(xtick_labels)
    ax.set_xlim((-1,3))

    plt_idx = 2
    plabel = f'{product_labels["tss"]} {product_units["tss"]}'
    xlabel = fr'$\mathbf{{{plabel}}}$'

    ax = fig.add_subplot(223)
    ax.grid(color='black',alpha=0.1)
    ax.set_axisbelow(True)
    x_vals = np.log10(product_values[:,tss_loc])
    # ax.set_title(f'N={sum(np.isfinite(x_vals))}')
    # text(-0.75,200,f'N={sum(np.isfinite(x_vals))[0]}',fontsize=12)


    bin_locations = np.linspace(-1,3)

    n,bins,patches  = ax.hist(x_vals,bins=bin_locations,facecolor='xkcd:burnt orange',edgecolor='white',linewidth=0.5,density=False,log=False,alpha=0.75)
    text(min(bins*.75),max(n*.9),f'N={sum(np.isfinite(x_vals))[0]}',fontsize=12)
    ax.set_xlabel(xlabel.replace(' ', '\ '),fontsize=15)
    ax.tick_params(axis='both',which='minor',labelsize=13)
    ax.tick_params(axis='both',which='major',labelsize=13)
    ax.set_facecolor('xkcd:White')

    labels = ax.get_xticklabels(which='both')
    locs = ax.get_xticks()

    xtick_labels = [int(value) for value in locs] 
    xtick_labels = [fr'${{10^{ {value} }}}$' for value in xtick_labels]
    ax.set_xticks(locs)
    ax.set_xticklabels(xtick_labels)
    ax.set_xlim((-1,3))

    plt_idx = 3
    plabel = f'{product_labels["cdom"]} {product_units["cdom"]}'
    xlabel = fr'$\mathbf{{{plabel}}}$'

    ax = fig.add_subplot(224)
    ax.grid(color='black',alpha=0.1)
    ax.set_axisbelow(True)
    x_vals = np.log10(product_values[:,cdom_loc])
    bin_locations = np.linspace(-3,2)
    # ax.set_title(f'N={sum(np.isfinite(x_vals))}')
    # text(-0.75,200,f'N={sum(np.isfinite(x_vals))[0]}',fontsize=12)


    n,bins,patches  = ax.hist(x_vals,bins=bin_locations,facecolor='xkcd:red brown',edgecolor='white',linewidth=0.5,density=False,log=False,alpha=0.75)
    text(min(bins*.88),max(n*.9),f'N={sum(np.isfinite(x_vals))[0]}',fontsize=12)
    ax.set_xlabel(xlabel.replace(' ', '\ '),fontsize=15)
    ax.tick_params(axis='both',which='minor',labelsize=13)
    ax.tick_params(axis='both',which='major',labelsize=13)
    ax.set_facecolor('xkcd:White')

    labels = ax.get_xticklabels(which='both')
    locs = ax.get_xticks()

    xtick_labels = [int(value) for value in locs] 
    xtick_labels = [fr'${{10^{ {value} }}}$' for value in xtick_labels]
    ax.set_xticks(locs)
    ax.set_xticklabels(xtick_labels)
    ax.set_xlim((-3,2))


    filename = folder.joinpath(f'Product_histogram_{np.shape(product_values)}.jpg')
    font = {'family' : 'normal',
            'weight' : 'bold',
            'size'   : 20}
    fig.text(-0.02, 0.5, 'Frequency', va='center', rotation='vertical',**font)

    plt.tight_layout()
    plt.savefig(filename.as_posix(), dpi=600, bbox_inches='tight', pad_inches=0.1,)
    plt.close()

    print('Mean of {}: {} Median of {}:'.format(products[chl_loc],np.nanmean(product_values[:,chl_loc]),np.nanmedian(product_values[:,chl_loc])))
    print('Mean of {}: {} Median of {}:'.format(products[PC_loc],np.nanmean(product_values[:,PC_loc]),np.nanmedian(product_values[:,PC_loc])))

    print('Mean of {}: {} Median of {}:'.format(products[cdom_loc],np.nanmean(product_values[:,cdom_loc]),np.nanmedian(product_values[:,cdom_loc])))

    print('Mean of {}: {} Median of {}:'.format(products[tss_loc],np.nanmean(product_values[:,tss_loc]),np.nanmedian(product_values[:,tss_loc])))


    # print('Mean of {}: {} Median of {}:'.format(products[1],np.mean(product_values[:,PC_loc]),np.median(product_values[:,PC_loc])))
    # print('Mean of {}: {} Median of {}:'.format(products[1],np.mean(product_values[:,PC_loc]/product_values[:,chl_loc]),np.median(product_values[:,PC_loc]/product_values[:,chl_loc])))


@ignore_warnings
def plot_remote_insitu(y_remote, y_insitu, dictionary_of_matchups=None, products='chl', sensor='HICO',run_name="",args=None):
    y_remote_OG =y_remote
    y_insitu_OG=y_insitu

    import matplotlib.patheffects as pe
    import matplotlib.ticker as ticker
    import matplotlib.pyplot as plt
    import matplotlib
    import seaborn as sns
    from pylab import text

    folder = Path('scatter_plots')
    folder.mkdir(exist_ok=True, parents=True)
    n_row = 2
    n_col = 3
    fig_size   = 5
    plt_idx = 0
    plt.rc('text', usetex=True)
    plt.rcParams['mathtext.default']='regular'

    fig, axes = plt.subplots(n_row, n_col, figsize=(fig_size*n_col, fig_size*n_row))
    axes      = [ax for axs in np.atleast_1d(axes) for ax in np.atleast_1d(axs)]
    colors    = ['xkcd:sky blue', 'xkcd:tangerine', 'xkcd:fresh green', 'xkcd:greyish blue', 'xkcd:goldenrod',  'xkcd:clay', 'xkcd:bluish purple', 'xkcd:reddish', 'xkcd:neon purple']

    full_ax  = fig.add_subplot(111, frameon=False)
    full_ax.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False, pad=10)

    product_labels = default_dd({
        'chl' : 'Chl\\textit{a}',
        'pc'  : 'PC',
        'aph' :  'a_{\mathit{ph}}',
        'ad' :  'a_{\mathit{nap}}',
        'ag' : 'a_{\mathit{cdom}}',
        'ag443' : 'a_{\mathit{cdom}(443)}',
        'ad443' : 'a_{\mathit{nap}(443)}',

        'tss' : 'TSS',
        'rrs' : 'R_{\mathit{rs}}',

        })

    product_units = default_dd({
        'chl' : '[mg m^{-3}]',
        'PC' : '[mg m^{-3}]',
        'pc' : '[mg m^{-3}]',
        'rrs' : '[sr^{-1}]',

        'tss' : '[g m^{-3}]',
        
        'aph' : '[m^{-1}]',
        'ad' : '[m^{-1}]',
        'ag' : '[m^{-1}]',

    }, lambda k: '')

    font= {'size':15}
    for current_product in products:
        print(plt_idx,'current product:',current_product )
        # if current_product in ['aph']:
            
        y_remote = np.squeeze(np.asarray(y_remote_OG[0][:,:,y_remote_OG[1][current_product]]))#[:,0]
        y_insitu = np.squeeze(np.asarray(y_insitu_OG[0][:,:,y_insitu_OG[1][current_product]]))#[:,0]



        plabel_1 = f'{product_labels[products[plt_idx]]}'
        plabel_2 = f'{product_units[products[plt_idx]]}'
        xlabel = fr'$\mathbf{{{plabel_1}\textsuperscript{{e}}{plabel_2}}}$'
        ylabel = fr'$\mathbf{{{plabel_1}\textsuperscript{{r}}{plabel_2}}}$'


        s_lbl = get_sensor_label(sensor).replace('-',' ')
        n_pts = len(y_insitu)
        title = fr'$\mathbf{{\underline{{\large{{{s_lbl}}}}}}}$' + '\n' + fr'$\small{{\mathit{{N\small{{=}}}}{n_pts}}}$'
        # full_ax.set_title(title.replace(' ', '\ '), fontsize=24, y=1.04)

        curr_idx = 0
        cidx  = plt_idx
        color = colors[cidx]
        l_kws = {'color': color if current_product not in ['aph','ad','ag'] else 'k', 'path_effects': [pe.Stroke(linewidth=4, foreground='k'), pe.Normal()], 'zorder': 22, 'lw': 1}
        s_kws = {'alpha': 0.4 if current_product not in ['aph','ad','ag'] else 0.0, 'color': color}

        ax = axes[plt_idx]
        ax.set_xlabel(xlabel.replace(' ', '\ '), fontsize=20, labelpad=10)
        ax.set_ylabel(ylabel.replace(' ', '\ '), fontsize=20, labelpad=10)


        y_true_log = np.log10(y_insitu).flatten()
        y_est_log = np.log10(y_remote).flatten()
        minv = int(np.nanmin(y_true_log)) - 1
        maxv = int(np.nanmax(y_true_log)) + 1
        loc  = ticker.LinearLocator(numticks=maxv-minv+1)
        fmt  = ticker.FuncFormatter(lambda i, _: r'$10$\textsuperscript{%i}'%i)

        ax.set_ylim((minv, maxv))
        ax.set_xlim((minv, maxv))
        ax.xaxis.set_major_locator(loc)
        ax.yaxis.set_major_locator(loc)
        ax.xaxis.set_major_formatter(fmt)
        ax.yaxis.set_major_formatter(fmt)
        valid = np.logical_and(np.isfinite(y_true_log), np.isfinite(y_est_log))
        print('valid matchups:',sum(valid))
        
        bands  = np.array( get_sensor_bands(args.sensor, args) ) if args.use_HICO_aph == False and current_product == 'aph' else  np.array( get_sensor_bands('HICO-aph', args) ) if args.use_HICO_aph == True and current_product == 'aph' else  np.array( get_sensor_bands('HICO-adag', args) ) if args.use_HICO_aph == True and (current_product == 'ad' or current_product == 'ag') else np.array( get_sensor_bands(args.sensor, args) ) 
        norm = matplotlib.colors.Normalize(vmin=380.0, vmax=780.0)
        cmap   = get_spectrum_cmap()
        colors_wavelength = [cmap.to_rgba(nm)  for nm in bands]
        
        
        if valid.sum():
            sns.regplot(y_true_log[valid], y_est_log[valid], ax=ax, scatter_kws=s_kws, line_kws=l_kws, fit_reg=True, truncate=False, robust=True, ci=None)
            if current_product in ['aph','ad','ag']: 
                for i in range(np.shape(y_insitu)[1]):
                    y_true_log_wavelength = np.log10(y_insitu[:,i])
                    y_est_log_wavelength = np.log10(y_remote[:,i])
                    valid_wavelength = np.logical_and(np.isfinite(y_true_log_wavelength),np.isfinite(y_est_log_wavelength))
                    ax.scatter(y_true_log_wavelength[valid_wavelength], y_est_log_wavelength[valid_wavelength],color=colors_wavelength[i])
            kde = sns.kdeplot(y_true_log[valid], y_est_log[valid], shade=False, ax=ax, bw='scott', n_levels=4, legend=False, gridsize=100, color='k')
            if current_product not in ['aph','ad','ag']: kde.collections[2].set_alpha(0)




        if len(valid.flatten()) != valid.sum() and False:
            ax.scatter(y_true_log[~valid], [minv]*(~valid).sum(), color='r', alpha=0.4, label=r'$\mathbf{%s\ invalid, %s\ nan}$' % ((~valid).sum(), np.isnan(y_true_log[~valid]).sum()) ) 
            ax.legend(loc='lower right', prop={'weight':'bold', 'size': 16})

        add_identity(ax, ls='--', color='k', zorder=20)
        add_stats_box(ax, y_insitu.flatten()[valid], y_remote.flatten()[valid],metrics=[slope,r_squared])

        ax.tick_params(labelsize=12)
        ax.grid('on', alpha=0.3)

        filename = folder.joinpath(f'remote_vs_insitu_summary_{run_name}_{products}_{sensor}.jpg')
        plt.tight_layout()
        plt_idx = plt_idx+1

    plt.savefig(filename.as_posix(), dpi=600, bbox_inches='tight', pad_inches=0.1,)

    plt_idx = 0 

    n_row = 4
    n_col = 4
    fig_size   = 5
    plt_idx = 0
    plt.rc('text', usetex=True)
    plt.rcParams['mathtext.default']='regular'

    fig, axes = plt.subplots(n_row, n_col, figsize=(fig_size*n_col, fig_size*n_row))
    axes      = [ax for axs in np.atleast_1d(axes) for ax in np.atleast_1d(axs)]
    colors    = ['xkcd:sky blue', 'xkcd:tangerine', 'xkcd:fresh green', 'xkcd:greyish blue', 'xkcd:goldenrod',  'xkcd:clay', 'xkcd:bluish purple', 'xkcd:reddish', 'xkcd:neon purple']

    full_ax  = fig.add_subplot(111, frameon=False)
    full_ax.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False, pad=10)

    xlabel = fr'$\mathbf{{Band [nm]}}$'
    ylabel = fr'$\mathbf{{R\textsubscript{{rs}} [sr\textsuperscript{{-1}}]}}$'
    full_ax.set_xlabel(xlabel.replace(' ', '\ '), fontsize=20, labelpad=10)
    full_ax.set_ylabel(ylabel.replace(' ', '\ '), fontsize=20, labelpad=30)
    
    site_labels_of_interest = ['WE2','WE6','WE13', 'Lake Erie St. 970','St. Andrews Bay (SA11)\nApr. 14, 2010','Pensacola Bay (PB09)\nAug. 26, 2011','Pensacola Bay (PB05)\nAug. 26, 2011','Pensacola Bay (PB04)\nAug. 26, 2011', 'Pensacola Bay (PB14)\nJun. 02, 2011','Pensacola Bay (PB08)\nJun. 02, 2011','Choctawhatchee Bay (CH01)\nJul. 30, 2011','Choctawhatchee Bay (CH03)\nJul. 30, 2011','WE4','WE8','Gulf_Mexico 72','Gulf_Mexico 82']

    site_labels_of_interest_no_newline_dict = {
    'WE2' : 'WE2',
    'WE6' : 'WE6',
    'WE13' : 'WE13',
    'WE4' : 'WE4',
    'WE8' : 'WE8',

    'Lake Erie St. 970' : 'Lake Erie St. 970',
    'St. Andrews Bay (SA11)\nApr. 14, 2010' : 'St. Andrews Bay (SA11)',
    'Pensacola Bay (PB14)\nJun. 02, 2011' : 'Pensacola Bay (PB14)',
    'Pensacola Bay (PB06)\nJun. 02, 2011' : 'Pensacola Bay (PB06)',
    'Pensacola Bay (PB04)\nAug. 26, 2011' : 'Pensacola Bay (PB04)',
    'Pensacola Bay (PB08)\nJun. 02, 2011' : 'Pensacola Bay (PB08)',
    'Pensacola Bay (PB05)\nAug. 26, 2011' : 'Pensacola Bay (PB05)',

    'Pensacola Bay (PB09)\nAug. 26, 2011' : 'Pensacola Bay (PB09)',
    'Choctawhatchee Bay (CH01)\nJul. 30, 2011' : 'Choctawhatchee Bay (CH01)',
    'Choctawhatchee Bay (CH03)\nJul. 30, 2011' : 'Choctawhatchee Bay (CH03)',
    'Gulf_Mexico 72' : 'Gulf of Mexico 72',
    'Gulf_Mexico 82' :'Gulf of Mexico 82',
    }
    import datetime as dt
    def try_to_parse_date(input_text):
        for fmt in ('[\'%Y%m%d %H:%M\']','[\'%Y-%m-%d %H:%M\']','[\'%Y%m%d\']'):
            try:
                return dt.datetime.strptime(input_text,fmt)
            except ValueError:
                pass
        raise ValueError('No Valid date format found')

    round_digits = 1


    for plotting_label_current in site_labels_of_interest:
        if plotting_label_current in dictionary_of_matchups['plotting_labels']:
            index_of_plotting_label = np.where(plotting_label_current == dictionary_of_matchups['plotting_labels'])
            index = index_of_plotting_label[0][0]
        else:
            print("NOT IN DICTIONARY")
            continue

        ax = axes[plt_idx]
        first_row = plt_idx < n_col
        last_row  = plt_idx >= ((n_row-1)*n_col)
        first_col = (plt_idx % n_col) == 0
        last_col  = ((plt_idx+1) % n_col) == 0

        if not last_row:  ax.set_xticklabels([])
        if not first_col: ax.set_yticklabels([])

        chl_truth = round(np.asscalar(dictionary_of_matchups['chl'][index]),round_digits)
        PC_truth = round(np.asscalar(dictionary_of_matchups['PC'][index]),round_digits)
        chl_remote_estimate = round(np.asscalar(np.squeeze(np.asarray(y_remote_OG[0][:,:,y_remote_OG[1]["chl"]]))[index]),round_digits)
        PC_remote_estimate = round(np.asscalar(np.squeeze(np.asarray(y_remote_OG[0][:,:,y_remote_OG[1]["pc"]]))[index]),round_digits)

        chl_insitu_estimate = round(np.asscalar(np.squeeze(np.asarray(y_insitu_OG[0][:,:,y_insitu_OG[1]["chl"]]))[index]),round_digits)
        PC_insitu_estimate = round(np.asscalar(np.squeeze(np.asarray(y_insitu_OG[0][:,:,y_insitu_OG[1]["pc"]]))[index]),round_digits)

        text_label = fr'PC: {PC_truth}'+ '\n'  + fr'PC\textsuperscript{{e}}: {PC_insitu_estimate}' + '\n' + fr'PC\textsuperscript{{r}}: {PC_remote_estimate}'
        text(0.9,0.905,text_label,horizontalalignment='center',verticalalignment='center',transform=ax.transAxes,backgroundcolor='1.0',bbox=dict(facecolor='white',edgecolor='black',boxstyle='round'),fontdict=font)
        plot_label ='ABCDEFGHIJKLMNOPQRST'
        text(0.03,0.96,plot_label[plt_idx],horizontalalignment='center',verticalalignment='center',transform=ax.transAxes,backgroundcolor='1.0',bbox=dict(facecolor='white',edgecolor='black',boxstyle='round'),fontdict=font)

        insitu_Rrs = dictionary_of_matchups['insitu_Rrs_resampled_full'][index,:]
        insitu_Rrs_wvl = dictionary_of_matchups['insitu_Rrs_resampled_wvl_full'][0,:]
        retrieved_Rrs = dictionary_of_matchups['Rrs_retrieved_full'][index,:]
        retrieved_Rrs_wvl = dictionary_of_matchups['Rrs_retrieved_wvl_full'][0,:]
        site_label = dictionary_of_matchups['site_label'][index,:]
        plotting_label = str(dictionary_of_matchups['plotting_labels'][index,:])

        date_time = str(dictionary_of_matchups['insitu_datetime'][index,:])
        reformatted_datetime = try_to_parse_date(date_time) 
        reformatted_datetime = reformatted_datetime.strftime('%b, %d, %Y ')

        ax.plot(insitu_Rrs_wvl,insitu_Rrs,'-o',color='b', alpha=0.4,label=fr'Rrs')
        ax.plot(retrieved_Rrs_wvl,retrieved_Rrs,'-o',color='r', alpha=0.4,label=fr'\^{{R}}rs')
        ax.set_ylim((0.0,0.025))
        plotting_label = site_labels_of_interest_no_newline_dict[plotting_label_current]

        title = fr'$\mathbf{{{{\large{{{plotting_label}}}}}}}$' + '\n' + fr'$\small{{{reformatted_datetime}}}$'
        ax.tick_params(labelsize=20)
        ax.grid('on', alpha=0.3)

        ax.set_title(title.replace(' ', '\ '), fontsize=18)
        filename = folder.joinpath(f'remote_vs_insitu_{run_name}_{products}_{sensor}.jpg')
        plt.tight_layout()
        plt_idx = plt_idx+1
        if plt_idx ==1:
            ax.legend(loc='upper center',fontsize=16)
    plt.savefig(filename.as_posix(), dpi=600, bbox_inches='tight', pad_inches=0.1,)
    plt.close()
    
    
    #################3
    #A third plot of aph, ag, and ad
    n_row = 4
    n_col = 4
    fig_size   = 5
    plt_idx = 0
    plt.rc('text', usetex=True)
    plt.rcParams['mathtext.default']='regular'

    
    colors    = ['xkcd:sky blue', 'xkcd:tangerine', 'xkcd:fresh green', 'xkcd:greyish blue', 'xkcd:goldenrod',  'xkcd:clay', 'xkcd:bluish purple', 'xkcd:reddish', 'xkcd:neon purple']



    xlabel = fr'$\mathbf{{Band [nm]}}$'
    ylabel = fr'$\mathbf{{a\textsubscript{{ph}} [m\textsuperscript{{-1}}]}}$'
    cdom="cdom"
    nap="nap"
    rs="rs"
    ylabel_ag = fr'$\mathbf{{a\textsubscript{{\mathit{cdom}}} [m\textsuperscript{{-1}}]}}$'        
    ylabel_ad = fr'$\mathbf{{a\textsubscript{{\mathit{{nap}}}} [m\textsuperscript{{-1}}]}}$'
    ylabel_rrs = fr'$\mathbf{{R\textsubscript{{\mathit{rs}}} [sr\textsuperscript{{-1}}]}}$'

    PRODUCT_CURRENT = 'ag'
    plabel_1 = f'{product_labels[PRODUCT_CURRENT]}'
    plabel_2 = f'{product_units[PRODUCT_CURRENT]}'
    ylabel_ag = fr'$\mathbf{{{plabel_1}{plabel_2}}}$'
    
    PRODUCT_CURRENT = 'ad'
    plabel_1 = f'{product_labels[PRODUCT_CURRENT]}'
    plabel_2 = f'{product_units[PRODUCT_CURRENT]}'
    ylabel_ad = fr'$\mathbf{{{plabel_1}{plabel_2}}}$'
    
    PRODUCT_CURRENT = 'rrs'
    plabel_1 = f'{product_labels[PRODUCT_CURRENT]}'
    plabel_2 = f'{product_units[PRODUCT_CURRENT]}'
    ylabel_rrs = fr'$\mathbf{{{plabel_1}{plabel_2}}}$'
        
    PRODUCT_CURRENT = 'aph'
    plabel_1 = f'{product_labels[PRODUCT_CURRENT]}'
    plabel_2 = f'{product_units[PRODUCT_CURRENT]}'
    ylabel = fr'$\mathbf{{{plabel_1}{plabel_2}}}$'
    
    # full_ax.set_ylabel(ylabel.replace(' ', '\ '), fontsize=20, labelpad=30) 
    # full_ax2 = full_ax.twinx()
    # full_ax2.set_ylabel(ylabel.replace(' ', '\ '), fontsize=20, labelpad=30) 
    # full_ax.set_ylabel(ylabel.replace(' ', '\ '), fontsize=20, labelpad=30) 

    aph_resampled = dictionary_of_matchups['insitu_aph_resampled']
    aph_wvl_resampled = dictionary_of_matchups['insitu_aph_resampled_wvl']
    aph_wvl_resampled = aph_wvl_resampled[0,:]
    
    ad_resampled = dictionary_of_matchups['insitu_ad_resampled']
    ad_wvl_resampled = dictionary_of_matchups['insitu_ad_resampled_wvl']
    ad_wvl_resampled = ad_wvl_resampled[0,:]
    
    ag_resampled = dictionary_of_matchups['insitu_ag_resampled']
    ag_wvl_resampled = dictionary_of_matchups['insitu_ag_resampled_wvl']
    ag_wvl_resampled = ag_wvl_resampled[0,:]
    #sum(np.logical_and(~np.isnan(ag_resampled).all(axis=1),~np.isnan(dictionary_of_matchups['Rrs_retrieved_full']).all(axis=1)))
    aph_remote_estimate = (np.squeeze(np.asarray(y_remote_OG[0][:,:,y_remote_OG[1]["aph"]]))[index])

    current_product = 'aph'
    bands  = np.array( get_sensor_bands(args.sensor, args) ) if args.use_HICO_aph == False and current_product == 'aph' else  np.array( get_sensor_bands('HICO-aph', args) ) if args.use_HICO_aph == True and current_product == 'aph' else  np.array( get_sensor_bands('HICO-adag', args) ) if args.use_HICO_aph == True and (current_product == 'ad' or current_product == 'ag') else np.array( get_sensor_bands(args.sensor, args) ) 
    bands_ad_ag = np.array( get_sensor_bands('HICO-adag', args) )
    site_labels_of_interest_1 = ['St. Andrews Bay (SA11)\nApr. 14, 2010','Pensacola Bay (PB09)\nAug. 26, 2011','Pensacola Bay (PB05)\nAug. 26, 2011','Pensacola Bay (PB04)\nAug. 26, 2011'] #, 'Pensacola Bay (PB14)\nJun. 02, 2011','Pensacola Bay (PB08)\nJun. 02, 2011','Choctawhatchee Bay (CH01)\nJul. 30, 2011','Choctawhatchee Bay (CH03)\nJul. 30, 2011','WE4','WE8','Gulf_Mexico 72','Gulf_Mexico 82'
    site_labels_of_interest_2 = ['Pensacola Bay (PB14)\nJun. 02, 2011','Pensacola Bay (PB08)\nJun. 02, 2011','Choctawhatchee Bay (CH01)\nJul. 30, 2011','Choctawhatchee Bay (CH03)\nJul. 30, 2011']
    
    for site_label_set,site_labels_of_interest in enumerate([site_labels_of_interest_1, site_labels_of_interest_2]):
        counter = 0
        i = []
        plt_idx=0
        fig, axes = plt.subplots(n_row, n_col, figsize=(fig_size*n_col, fig_size*n_row))
        axes      = [ax for axs in np.atleast_1d(axes) for ax in np.atleast_1d(axs)]
        full_ax  = fig.add_subplot(111, frameon=False)
        full_ax.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False, pad=10)
        full_ax.set_xlabel(xlabel.replace(' ', '\ '), fontsize=28, labelpad=10)

        for plotting_label_current in site_labels_of_interest:
            if plotting_label_current in dictionary_of_matchups['plotting_labels']:
                index_of_plotting_label = np.where(plotting_label_current == dictionary_of_matchups['plotting_labels'])
                index = index_of_plotting_label[0][0]
            else:
                print("NOT IN DICTIONARY")
                continue
            
            if plt_idx > 3: continue

            
            cdom_remote_estimate = round(np.asscalar(np.squeeze(np.asarray(y_remote_OG[0][:,:,y_remote_OG[1]["cdom"]]))[index]),round_digits)
            cdom_insitu_estimate = round(np.asscalar(np.squeeze(np.asarray(y_insitu_OG[0][:,:,y_insitu_OG[1]["cdom"]]))[index]),round_digits)
    
            aph_remote_estimate =  np.squeeze(np.asarray(y_remote_OG[0][:,:,y_remote_OG[1]["aph"]]))[index]
            aph_insitu_estimate =  np.squeeze(np.asarray(y_insitu_OG[0][:,:,y_insitu_OG[1]["aph"]]))[index]
    
            ad_remote_estimate =  np.squeeze(np.asarray(y_remote_OG[0][:,:,y_remote_OG[1]["ad"]]))[index]
            ad_insitu_estimate =  np.squeeze(np.asarray(y_insitu_OG[0][:,:,y_insitu_OG[1]["ad"]]))[index]
            
            ag_remote_estimate =  np.squeeze(np.asarray(y_remote_OG[0][:,:,y_remote_OG[1]["ag"]]))[index]
            ag_insitu_estimate =  np.squeeze(np.asarray(y_insitu_OG[0][:,:,y_insitu_OG[1]["ag"]]))[index]
    

            if np.any(np.isnan(aph_resampled),axis=1)[index] or np.any(np.isnan(aph_remote_estimate)): continue
            linewidth_truth=8
            linewidth_insitu=linewidth_truth-1
            linewidth_remote=linewidth_truth-2

        
            ax0 = axes[plt_idx]
            
            ax0.plot(dictionary_of_matchups['insitu_Rrs_resampled_wvl'][0,:],dictionary_of_matchups['insitu_Rrs_resampled'][index,:],'c',label=fr'$\mathit{{In \ situ}}$',linewidth=linewidth_insitu)
            ax0.plot(dictionary_of_matchups['Rrs_retrieved_wvl'][0,:],dictionary_of_matchups['Rrs_retrieved'][index,:],'r',label='Remote',linewidth=linewidth_remote)
            ax0.set_ylim([0,0.015])
            ax0.set_xlim([400,700])
            ax0.grid()
            ax0.set_yticks([0.0025, 0.0075,0.0125], minor=False)
            ax0.set_yticks([0.005, 0.01, 0.015], minor=True)
            ax0.yaxis.grid(True, which='major')
            ax0.yaxis.grid(True, which='minor')
            ax0.set_xticks([400, 500, 600,700], minor=False)
            ax0.set_xticks([450, 550, 650], minor=True)
            ax0.xaxis.grid(True, which='major')
            ax0.xaxis.grid(True, which='minor')
            
            ax0.set_xticklabels([])
            if plt_idx == 0: 
                # ax0.legend(fontsize=26)
                ax0.set_ylabel(ylabel_rrs.replace(' ', '\ '), fontsize=34, labelpad=30) 
                # ax.tick_params(axis='x', labelsize=8)
                ax0.tick_params(axis='y', labelsize=26)
            else:
                ax0.set_xticklabels([])
                ax0.set_yticklabels([])            
            
            
            plt_idx_1_5 = plt_idx+n_col
            ax = axes[plt_idx_1_5]
            ax.plot(aph_wvl_resampled,aph_resampled[index,:], 'k',label='Measured (a)',linewidth=linewidth_truth)
            e="e"
            ax.plot(bands,aph_insitu_estimate,'c',label=fr'$\mathit{{In \ situ}} \  (a^{e})$',linewidth=linewidth_insitu)
            r="r"
            ax.plot(bands,aph_remote_estimate,'r',label=fr'Remote ($a^{r}$)',linewidth=linewidth_remote)
            ax.set_ylim([0,0.6])
            ax.set_xlim([400,700])
            ax.grid()
            ax.set_xticklabels([])

            if plt_idx_1_5 == n_col: 
                leg = ax.legend(fontsize=26,frameon=True)
                leg.get_frame().set_edgecolor('m')
                leg.get_frame().set_linewidth(2)

                ax.set_ylabel(ylabel.replace(' ', '\ '), fontsize=34, labelpad=30) 
                ax.tick_params(axis='y', labelsize=26)
            else:
                ax.set_xticklabels([])
                ax.set_yticklabels([])        
                
            # ax.set_ylim([0,1.5])
            # ax.set_xlim([400,700])
            ax.set_yticks([0.2, 0.4, 0.6], minor=True)
            ax.set_yticks([0.1, 0.3, 0.5], minor=False)
            ax.yaxis.grid(True, which='major')
            ax.yaxis.grid(True, which='minor')
            ax.set_xticks([400, 500, 600,700], minor=False)
            ax.set_xticks([450, 550, 650], minor=True)
            ax.xaxis.grid(True, which='major')
            ax.xaxis.grid(True, which='minor')
            
            #####################
            #ax2=ax.twinx()
            plt_idx_2=plt_idx_1_5+n_col
    
    
            ax2 = axes[plt_idx_2]
            # ax2.plot(ad_wvl_resampled,ad_resampled[index,:], 'k',label=fr'$a\textsubscript{{d}}$')
            ax2.plot(ag_wvl_resampled,ag_resampled[index,:], 'k',label=fr'$a\textsubscript{{g}}$',linewidth=linewidth_truth)
            
            # ax2.plot(bands_ad_ag,ad_remote_estimate,'r')
            ax2.plot(bands_ad_ag,ag_insitu_estimate,'c',linewidth=linewidth_insitu)
            ax2.plot(bands_ad_ag,ag_remote_estimate,'r',linewidth=linewidth_remote)
            ag_measurement=dictionary_of_matchups['cdom'][index][0]
            ax2.plot(443,ag_measurement,'k.',markersize=15,label=fr'$cdom\textsubscript{{443}}$')
            ax2.plot(443,cdom_remote_estimate,'r.',markersize=15)
            ax2.plot(443,cdom_insitu_estimate,'c.',markersize=15)
    
            # ax2.plot(bands_ad_ag,ad_insitu_estimate,'g')
            import math 
            from .utils import convert_point_slope_to_spectral_cdom, convert_spectral_cdom_to_point_slope
            # wavelengths = [412,443,500,550,600,650,700]
            # CDOM=1.5
            # SCDOM = 0.0182
            # spectral_CDOM = convert_point_slope_to_spectral_cdom(CDOM,SCDOM,wavelengths,reference_CDOM_wavelength=440)
            # plt.plot(wavelengths,spectral_CDOM)
            # plt.ylabel('ag | ad')
            # plt.xlabel('Wavelength (nm)')

            # plt.show()
            allowed_error=20
            CDOM_truth,SCDOM_truth = convert_spectral_cdom_to_point_slope(ag_wvl_resampled,ag_resampled[index,:],reference_CDOM_wavelength=443,spectral_min_max=[400,700],allowed_error=allowed_error)

            CDOM_insitu,SCDOM_insitu = convert_spectral_cdom_to_point_slope(bands_ad_ag,ag_insitu_estimate,reference_CDOM_wavelength=443,spectral_min_max=[400,700],allowed_error=allowed_error)
            CDOM_remote,SCDOM_remote = convert_spectral_cdom_to_point_slope(bands_ad_ag,ag_remote_estimate,reference_CDOM_wavelength=443,spectral_min_max=[400,700],allowed_error=allowed_error)
            font_cdom=22
            CDOM_truth_rounded=np.round(ag_measurement,1)
            plabel = product_labels['ag443']
            plabel = f'{plabel}'
            xlabel = fr'$\mathbf{{ {plabel} }}$  : (CDOM) : S-CDOM'
            ax2.text(426,2.26,xlabel,color='k',fontsize=font_cdom-3.5,fontweight='extra bold')
            ax2.text(472,2.05,f'{CDOM_truth:.2f} : ({CDOM_truth_rounded:.1f}) : {SCDOM_truth:.4f}',color='k',fontsize=font_cdom,fontweight='extra bold')
            ax2.text(472,1.85,f'{CDOM_insitu:.2f} : ({cdom_insitu_estimate:.1f}) : {SCDOM_insitu:.4f}',color='c',fontsize=font_cdom,fontweight='extra bold')
            ax2.text(472,1.65,f'{CDOM_remote:.2f} : ({cdom_remote_estimate:.1f}) : {SCDOM_remote:.4f}',color='r',fontsize=font_cdom,fontweight='extra bold')



    
            if plt_idx_2 == 2*n_col: 
                # ax2.legend()
                ax2.set_ylabel(ylabel_ag.replace(' ', '\ '), fontsize=34, labelpad=30) 
                ax2.set_xticklabels([])
                ax2.tick_params(axis='y', labelsize=26)
            else:
                ax2.set_xticklabels([])
                ax2.set_yticklabels([])
                
    
            ax2.set_ylim([0,2.5])
            ax2.set_xlim([400,700])
            ax2.grid()
            ax2.set_yticks([1.0, 2.0], minor=True)
            ax2.set_yticks([0.5,1.5,2.5], minor=False)
            ax2.yaxis.grid(True, which='major')
            ax2.yaxis.grid(True, which='minor')
            ax2.set_xticks([400, 500, 600,700], minor=False)
            ax2.set_xticks([450, 550, 650], minor=True)
            ax2.xaxis.grid(True, which='major')
            ax2.xaxis.grid(True, which='minor')
            
            #####################
    
            plt_idx_3=plt_idx_2+n_col
    
    
            ax3 = axes[plt_idx_3]
            ax3.plot(ad_wvl_resampled,ad_resampled[index,:], 'k',label=fr'$a\textsubscript{{d}}$',linewidth=linewidth_truth)
            # ax3.plot(ag_wvl_resampled,ag_resampled[index,:], 'k-.',label=fr'$a\textsubscript{{g}}$')
            ax3.plot(bands_ad_ag,ad_insitu_estimate,'c',linewidth=linewidth_insitu)
            ax3.plot(bands_ad_ag,ad_remote_estimate,'r',linewidth=linewidth_remote)
            # ax3.plot(bands_ad_ag,ag_remote_estimate,'r-.')
            ag_measurement=dictionary_of_matchups['cdom'][index][0]
            # ax3.plot(443,ag_measurement,'k.',markersize=12,label=fr'$cdom\textsubscript{{443}}$')
            # ax3.plot(443,cdom_remote_estimate,'r.',markersize=12)
            # ax3.plot(443,cdom_insitu_estimate,'g.',markersize=12)
    
            # ax3.plot(bands_ad_ag,ag_insitu_estimate,'g-.')
            
            NAP_truth,SNAP_truth = convert_spectral_cdom_to_point_slope(ad_wvl_resampled,ad_resampled[index,:],reference_CDOM_wavelength=443,spectral_min_max=[400,700],allowed_error=allowed_error)
            NAP_insitu,SNAP_insitu = convert_spectral_cdom_to_point_slope(bands_ad_ag,ad_insitu_estimate,reference_CDOM_wavelength=443,spectral_min_max=[400,700],allowed_error=allowed_error)
            NAP_remote,SNAP_remote = convert_spectral_cdom_to_point_slope(bands_ad_ag,ad_remote_estimate,reference_CDOM_wavelength=443,spectral_min_max=[400,700],allowed_error=allowed_error)
            # font_cdom=18
            plabel = product_labels['ad443']
            plabel = f'{plabel}'
            xlabel = fr'$\mathbf{{ {plabel} }}$  :  S-NAP'          
            ax3.text(446,1.34,xlabel,color='k',fontsize=font_cdom,fontweight='extra bold')
            ax3.text(490,1.2,f'{NAP_truth:.2f} : {SNAP_truth:.4f}',color='k',fontsize=font_cdom,fontweight='extra bold')
            ax3.text(490,1.1,f'{NAP_insitu:.2f} :  {SNAP_insitu:.4f}',color='c',fontsize=font_cdom,fontweight='extra bold')
            ax3.text(490,1.0,f'{NAP_remote:.2f} :  {SNAP_remote:.4f}',color='r',fontsize=font_cdom,fontweight='extra bold')
            
            
            if plt_idx_3 == 3*n_col: 
                ax3.set_ylabel(ylabel_ad.replace(' ', '\ '), fontsize=34, labelpad=30) 
                ax3.tick_params(axis='y', labelsize=26)
            else:
                ax3.set_yticklabels([])
                
            ax3.tick_params(axis='x', labelsize=26)

            ax3.set_ylim([0,1.5])
            ax3.set_xlim([400,700])
            ax3.set_yticks([0.25, 0.75, 1.25], minor=False)
            ax3.set_yticks([0.5, 1.0, 1.5], minor=True)
            ax3.yaxis.grid(True, which='major')
            ax3.yaxis.grid(True, which='minor')
            ax3.set_xticks([400, 500, 600,700], minor=False)
            ax3.set_xticks([450, 550, 650], minor=True)
            ax3.xaxis.grid(True, which='major')
            ax3.xaxis.grid(True, which='minor')
            
            # ax3.grid()
            
            ax0.set_title(dictionary_of_matchups['plotting_labels'][index][0].replace('_',''),fontsize=30-site_label_set*4)
            plt_idx=plt_idx+1
            # if plt_idx < 24: ax.set_xticks([])
            counter = counter+1
           
        plt.tight_layout()
        filename = folder.joinpath(f'pb_matchups_1_{run_name}_{products}_{sensor}_{site_label_set}.jpg')
        plt.savefig(filename.as_posix(), dpi=400, bbox_inches='tight', pad_inches=0.1,)
    
        # plt.show()
    

    
    
    
    
    
    
    
    
    