from .metrics import slope, sspb, mdsa 
from .meta import get_sensor_label,get_sensor_bands
from .utils import closest_wavelength, ignore_warnings
from collections import defaultdict as dd 
from pathlib import Path 
import numpy as np 

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
            'default' :  [443,482,561,655] if 'OLI'  in sensor else [443,490,560,620,673,710]  if 'OLCI' in sensor  else [409,443,478,535,620,673,690,]  if args.use_HICO_aph  else  [409,443,478,535,620,673,690,724]  #[415,443,490,501,550,620,673,690] #[443,482,561,655]
            # 'aph'     : [443, 530], [409, 421, 432, 444, 455, 467, 478, 490, 501, 512, 524, 535, 547, 558, 570, 581, 593, 604, 616, 621, 633, 644, 650, 656, 667, 673, 679, 684, 690, 701, 713, 724,]
        }

        target     = [closest_wavelength(w, bands if not args.use_HICO_aph else get_sensor_bands('HICO-aph', args) ) for w in product_bands.get(products[0], product_bands['default'])]
        print('Target',target)
        plot_label = [w in target for w in bands]
        if args.use_HICO_aph: 
            plot_label_aph = [w in target for w in get_sensor_bands('HICO-aph', args)]
            #y_test=y_test[:,[ i in get_sensor_bands('HICO-aph', args) for i in  get_sensor_bands(sensor, args)]]
        if args.use_HICO_aph and (products[0]=='ad' or products[0]=='ag'): 
                product_bands = {
                    'default' :  list(get_sensor_bands('HICO-adag', args))
                }
                target     = [closest_wavelength(w, bands if not args.use_HICO_aph else get_sensor_bands('HICO-adag', args)) for w in product_bands.get(products[0], product_bands['default'])]
                plot_label_aph = [w in target for w in get_sensor_bands('HICO-adag', args)]

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
                if not args.use_HICO_aph or (np.shape(benchmarks[product][est_lbl])[1] !=  np.shape(get_sensor_bands('HICO-aph', args))[0] and args.use_HICO_aph and product=='aph')  or (np.shape(benchmarks[product][est_lbl])[1] !=  np.shape(get_sensor_bands('HICO-adag', args))[0] and args.use_HICO_aph and (product=='ad' or product=='ag' ) ):
                    benchmarks[product][est_lbl] = np.reshape(benchmarks[product][est_lbl],(-1,np.shape(get_sensor_bands(sensor, args))[0]))   #if not args.use_HICO_aph  else np.reshape(benchmarks[product][est_lbl],(-1,np.shape(get_sensor_bands('HICO-aph', args))[0])) #if np.shape(benchmarks[product][est_lbl])[0] == 1786*np.shape(get_sensor_bands(sensor, args))[0] else  np.reshape(benchmarks[product][est_lbl],(1786,np.shape(get_sensor_bands(sensor, args))[0])) #np.shape(benchmarks[product]['MDN']))
                #Resample to HICO_aph wavelengths
                if args.use_HICO_aph and np.shape(benchmarks[product][est_lbl])[1] ==  np.shape(get_sensor_bands(sensor, args))[0] and product == 'aph':
                    benchmarks[product][est_lbl] = benchmarks[product][est_lbl][:,[ i in get_sensor_bands('HICO-aph', args) for i in  get_sensor_bands(sensor, args)]]
                if args.use_HICO_aph and np.shape(benchmarks[product][est_lbl])[1] ==  np.shape(get_sensor_bands(sensor, args))[0] and (product == 'ad' or  product == 'ag'):
                    benchmarks[product][est_lbl] = benchmarks[product][est_lbl][:,[ i in get_sensor_bands('HICO-adag', args) for i in  get_sensor_bands(sensor, args)]]            
                    
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



            minv = -3 if product == 'cdom' else 0.000 if product == 'Scdom443' else .005 if product == 'Snap443' else int(np.nanmin(y_true_log)) - 1 if product != 'aph'  else -4
            maxv = 3 if product == 'tss' else 4 if product == 'chl' else .035 if product == 'Scdom443' else .02 if product == 'Snap443' else int(np.nanmax(y_true_log)) + 1 if product != 'aph'  else 2
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
                add_stats_box(ax, y_true[valid], y_est[valid])
                if plot_order[est_idx] == 'MDN' and product == 'aph': 
                    args.summary_stats[label[1]] = create_stats_HPC(y_true[valid], y_est[valid], metrics=[mdsa, sspb, slope],label=None)
                else: 
                    args.summary_stats[label[1]] = create_stats_HPC(y_true[valid], y_est[valid], metrics=[mdsa, sspb, slope],label=None)


            if first_row or not plot_bands or (isinstance(plot_order, dict) and plot_order[product][est_idx] != 'MDN'):
                if est_lbl == 'BST':
                    # ax.set_title(fr'$\mathbf{{\large{{{est_lbl}}}}}$'+'\n'+r'$\small{\textit{(Cao\ et\ al.\ 2020)}}$', fontsize=18)
                    ax.set_title(r'$\small{\textit{(Cao\ et\ al.\ 2020)}}$' + '\n' + fr'$\mathbf{{\large{{{est_lbl}}}}}$', fontsize=18, linespacing=0.95)
                
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
    plt.savefig(filename.as_posix(), dpi=100, bbox_inches='tight', pad_inches=0.1,)
    plt.show()
