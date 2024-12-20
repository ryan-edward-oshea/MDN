from pathlib import Path
from sklearn import preprocessing
from tqdm  import trange 
from collections import defaultdict as dd

import numpy as np 
import pickle as pkl 
import hashlib 

from .model import MDN
from .meta  import get_sensor_bands, SENSOR_LABEL, ANCILLARY, PERIODIC
from .utils import get_labels, get_data, generate_config, using_feature, split_data, _load_datasets, compress, get_matchups, find_wavelength, OWT_classification,  convert_point_slope_to_spectral_cdom, convert_spectral_cdom_to_point_slope
from .metrics import performance, mdsa, sspb, msa
from .plot_utils import plot_scatter, plot_histogram, plot_spectra, plot_remote_insitu, add_identity
from .benchmarks import run_benchmarks
from .parameters import get_args
from .transformers import TransformerPipeline, generate_scalers

from MDN.meta import get_sensor_bands
from MDN.benchmarks.multiple.GIOP_initialized.model    import model as GIOP_init
from MDN.benchmarks.multiple.GIOP.model                import model as GIOP_default
from MDN.benchmarks.multiple.QAA_initialized.model     import model as QAA_init
from MDN.benchmarks.multiple.QAA.model                 import model as QAA_default
from MDN.benchmarks.multiple.Gordon.model              import model as Gordon_init
np.seterr(invalid='ignore')


def return_spectral_array(multispectral_variable, args=None, wavelengths_adg=None, wavelengths=None, normalize = True, ftol=1e-4, xtol=1e-3):
    adg_local = []
    if not wavelengths_adg: wavelengths_adg             = get_sensor_bands(args.sensor+'-adag', args)
    if not wavelengths:     wavelengths                 = get_sensor_bands(args.sensor, args)

    for i in multispectral_variable:
        adg_remote,Sadg_remote  = convert_spectral_cdom_to_point_slope(wavelengths_adg,i,reference_CDOM_wavelength=443,spectral_min_max=[400,700],allowed_error=100,ftol=ftol, xtol=xtol)
        spectral_adg_remote     = convert_point_slope_to_spectral_cdom(adg_remote,Sadg_remote,wavelengths,reference_CDOM_wavelength=443)
        if normalize:
            adg_local.append(spectral_adg_remote/adg_remote)
        else:
            adg_local.append(spectral_adg_remote)
    return np.array(adg_local)

def get_bbp_estimates(Rrs,estimates,slices,args,outputs = ['Gordon'],ftol=1e-4, xtol=1e-3):
        sensor                        = args.sensor
        wavelengths                   = get_sensor_bands(args.sensor, args)
        aph_wavelengths_len           = len(get_sensor_bands(args.sensor.split('-')[0]+'-aph', args))
        GIOP_initialization           =  {'chl' : estimates[:,slices['chl']],
                                          'aph' : estimates[:,slices['aph']],
                                          'ad'  : return_spectral_array(estimates[:,slices['ad']],args,normalize=True,ftol=ftol, xtol=xtol)[:,0:aph_wavelengths_len], #[:,0:50]
                                          'ag'  : return_spectral_array(estimates[:,slices['ag']],args,normalize=True,ftol=ftol, xtol=xtol)[:,0:aph_wavelengths_len], #[:,0:50]
                                          }

        QAA_initialization            =  {'chl' : estimates[:,slices['chl']],
                                          'aph' : estimates[:,slices['aph']],
                                          'ad'  : return_spectral_array(estimates[:,slices['ad']],args,normalize=False,ftol=ftol, xtol=xtol)[:,0:aph_wavelengths_len],#[:,0:50],
                                          'ag'  : return_spectral_array(estimates[:,slices['ag']],args,normalize=False,ftol=ftol, xtol=xtol)[:,0:aph_wavelengths_len],#[:,0:50]
                                          }

        GIOP_initialization['adg']    = (GIOP_initialization['ad'] + GIOP_initialization['ag'])/2
        bbp_estimates = {}
        
        if 'GIOP' in outputs:
            # bbp_estimates['GIOP-MDN']     = GIOP_init(   Rrs[:,0:50], wavelengths[0:50], sensor,**GIOP_initialization)
            # bbp_estimates['GIOP-default'] = GIOP_default(Rrs[:,0:50], wavelengths[0:50], sensor)

            bbp_estimates['GIOP-MDN']     = GIOP_init(   Rrs[:,0:aph_wavelengths_len], wavelengths[0:aph_wavelengths_len], sensor,**GIOP_initialization)
            bbp_estimates['GIOP-default'] = GIOP_default(Rrs[:,0:aph_wavelengths_len], wavelengths[0:aph_wavelengths_len], sensor)
            
        if 'QAA' in outputs:
            bbp_estimates['QAA-MDN']     = QAA_init(    Rrs[:,0:aph_wavelengths_len], wavelengths[0:aph_wavelengths_len], sensor,**QAA_initialization)
            bbp_estimates['QAA-default'] = QAA_default( Rrs[:,0:aph_wavelengths_len], wavelengths[0:aph_wavelengths_len], sensor)
            
        if 'Gordon' in outputs:
            bbp_estimates['Gordon-MDN']      = Gordon_init( Rrs[:,0:aph_wavelengths_len], wavelengths[0:aph_wavelengths_len], sensor,**QAA_initialization)

        return bbp_estimates

def get_estimates(args, x_train=None, y_train=None, x_test=None, y_test=None, output_slices=None, dataset_labels=None, x_sim=None, y_sim=None, return_model=False, return_coefs=False):
    ''' 
    Estimate all target variables for the given x_test. If a model doesn't 
    already exist, creates a model with the given training data. 
    '''        
    # Add x/y scalers to the args object
    generate_scalers(args, x_train, x_test)

    if args.verbose: 
        print(f'\nUsing {len(args.wavelengths)} wavelength(s) in the range [{args.wavelengths[0]}, {args.wavelengths[-1]}]')
        if x_train is not None: print_dataset_stats(x=x_train, label='Train')
        if y_train is not None: print_dataset_stats(y=y_train, label='Train')
        if x_test  is not None: print_dataset_stats(x=x_test,  label='Test')
        if y_test  is not None: print_dataset_stats(y=y_test,  label='Test')
    
    # Add a few additional variables to be stored in the generated config file
    if x_train is not None: setattr(args, 'data_xtrain_shape', x_train.shape)
    if y_train is not None: setattr(args, 'data_ytrain_shape', y_train.shape)
    if x_test  is not None: setattr(args, 'data_xtest_shape',  x_test.shape)
    if y_test  is not None: setattr(args, 'data_ytest_shape',  y_test.shape)
    if dataset_labels is not None: 
        sets_str  = ','.join(sorted(map(str, np.unique(dataset_labels))))
        sets_hash = hashlib.sha256(sets_str.encode('utf-8')).hexdigest()
        setattr(args, 'datasets_hash', sets_hash)

    model_path = generate_config(args, create=x_train is not None)    
    print("Model Path:",model_path.name)
    args.config_name = model_path.name
    
    predict_kwargs = {
        'avg_est'             : getattr(args, 'avg_est', False),
        'threshold'           : getattr(args, 'threshold', None),
        'confidence_interval' : getattr(args, 'CI', None),
        'use_gpu'             : True,#getattr(args, 'use_gpu', False),
        'chunk_size'          : getattr(args, 'chunk_size', 1e3),
        'return_coefs'        : True,
    }

    x_full, y_full   = x_train, y_train
    x_valid, y_valid = None, None

    outputs = dd(list)
    for round_num in trange(args.n_rounds, disable=args.verbose or (args.n_rounds == 1) or args.silent):
        args.curr_round = round_num
        curr_round_seed = args.seed+round_num if args.seed is not None else None
        np.random.seed(curr_round_seed)

        # 75% of rows used in bagging
        #if using_feature(args, 'bagging') and x_train is not None and args.n_rounds > 1:
            #(x_train, y_train), (x_valid, y_valid), train_idxs, test_idxs  = split_data(x_full, y_full, n_train=0.75, seed=curr_round_seed) 
            #print("bagging?")

        datasets = {k: dict(zip(['x','y'], v)) for k,v in {
            'train' : [x_train, y_train],
            'valid' : [x_valid, y_valid],
            'test'  : [x_test, y_test],
            'full'  : [x_full, y_full],
            'sim'   : [x_sim, y_sim],
        }.items() if v[0] is not None}

        model_kwargs = {
            'n_mix'      : args.n_mix, 
            'hidden'     : [args.n_hidden] * args.n_layers, 
            'lr'         : args.lr,
            'l2'         : args.l2,
            'n_iter'     : args.n_iter,
            'batch'      : args.batch,
            'imputations': args.imputations,
            'epsilon'    : args.epsilon,
            'scalerx'    : TransformerPipeline([S(*args, **kwargs) for S, args, kwargs in args.x_scalers]),
            'scalery'    : TransformerPipeline([S(*args, **kwargs) for S, args, kwargs in args.y_scalers]),
            'model_path' : model_path.joinpath(f'Round_{round_num}'),
            'no_load'    : args.no_load,
            'no_save'    : args.no_save,
            'seed'       : curr_round_seed,
            'verbose'    : args.verbose,
        }

        model = MDN(**model_kwargs)
        model.fit(x_train, y_train, output_slices, args=args, datasets=datasets)

        if return_model:
            outputs['model'].append(model)

        if return_coefs:
            outputs['scalerx'].append(model.scalerx)
            outputs['scalery'].append(model.scalery)

        if x_test is not None:
            (estimates, *confidence), coefs = model.predict(x_test, **predict_kwargs)
            #Add on bbp outputs if aph/ad/ag are available
            if all( [ product in args.product.split(',') for product in  ['aph','ad','ag']]):
                bbp_estimates = get_bbp_estimates(Rrs=x_test,estimates=estimates,slices=model.output_slices,args=args,ftol=1e-1, xtol=1e-1)
                bbp_gordon    = bbp_estimates['Gordon-MDN']['bbp']
                if 'bbp' not in model.output_slices.keys():
                    model.output_slices['bbp'] = slice(np.shape(estimates)[1],np.shape(estimates)[1]+np.shape(bbp_gordon)[1])
                estimates     = np.concatenate((estimates, bbp_gordon),axis=1)

            outputs['estimates'].append(estimates)

            if return_coefs:
                outputs['coefs'].append(coefs)

            if len(confidence):
                upper, lower = confidence
                outputs['upper_bound'].append(upper) 
                outputs['lower_bound'].append(lower)

            if args.verbose and y_test is not None:
                median = np.median(outputs['estimates'], axis=0)
                labels = get_labels(args.wavelengths, output_slices, n_out=y_test.shape[1])
                for lbl, y1, y2 in zip(labels, y_test.T, median.T):
                    print( performance(f'{lbl:>7s} Median', y1, y2) )
                print(f'--- Done round {round_num} ---\n')

        if hasattr(model, 'session'): model.session.close()

    # Create compressed model archive
    compress(model_path)

    if len(outputs) == 1:
        outputs = list(outputs.values())[0]
    return outputs, model.output_slices


def apply_model(x_test, use_cmdline=False, **kwargs):
	''' Apply a model (defined by kwargs and default parameters) to x_test '''
	args = get_args(kwargs, use_cmdline=use_cmdline)
	preds, idxs = get_estimates(args, x_test=x_test)
	return np.median(preds, 0), idxs


def image_estimates(data, sensor=None, function=apply_model, rhos=False, anc=False, **kwargs):
    ''' 
    Takes data of shape [Height, Width, Wavelengths] and returns the outputs of the 
    given function for that image, in the same [H, W] shape. 
    rhos and anc models are not yet available.  
    '''
    def ensure_feature_dim(v):
        if len(v.shape) == 1:
            v = v[:, None]
        return v 

    if isinstance(data, list):
        assert(all([data[0].shape == d.shape for d in data])), (
            f'Not all inputs have the same shape: {[d.shape for d in data]}')
        data = np.dstack(data)

    assert(sensor is not None), (
        f'Must pass sensor name to image_estimates function. Options are: {list(SENSOR_LABEL.keys())}')
    assert(sensor in SENSOR_LABEL), (
        f'Requested sensor {sensor} unknown. Must be one of: {list(SENSOR_LABEL.keys())}')
    assert(len(data.shape) == 3), (
        f'Expected data to have 3 dimensions (height, width, feature). Found shape: {data.shape}')

    args = get_args(sensor=sensor, **kwargs)
    print(args)
    expected_features = len(get_sensor_bands(sensor, args)) + (len(ANCILLARY)+len(PERIODIC) if anc or rhos else 0)
    assert(data.shape[-1] == expected_features), (
        f'Got {data.shape[-1]} features; expected {expected_features} features for sensor {sensor}')
    
    im_shape = data.shape[:-1] 
    im_data  = np.ma.masked_invalid(data.reshape((-1, data.shape[-1])))
    im_mask  = np.any(im_data.mask, axis=1)
    im_data  = im_data[~im_mask]
    estimate = function(im_data, sensor=sensor, **kwargs) if im_data.size else np.zeros((0, 1))

    # Need to handle function which return extra information (e.g. a dictionary mapping output feature slices)
    remaining = None
    if isinstance(estimate, tuple):
        estimate, *remaining = estimate 

    estimate = ensure_feature_dim(estimate)
    est_mask = np.tile(im_mask[:,None], (1, estimate.shape[-1]))
    est_data = np.ma.array(np.zeros(est_mask.shape)*np.nan, mask=est_mask, hard_mask=True)
    est_data.data[~im_mask] = estimate
    est_data = est_data.reshape(im_shape + est_data.shape[-1:])

    # Let the user handle the extra information of the function they passed, if there was any
    if remaining is not None and len(remaining):
        if len(remaining) == 1: 
            remaining = remaining[0]
        return est_data, remaining
    return est_data


def print_dataset_stats(**kwargs):
    ''' Print datasets shape & min / max stats per feature '''
    label = kwargs.pop('label', '')
    for k, arr in kwargs.items():
        if arr is not None:
            print(f'\n{label} {k.title()}'.strip()+'\n\t'.join(['']+[f'{k}: {v}'.replace("'", "") for k, v in {
                'Shape'   : np.array(arr).shape,
                'N Valid' : getattr(np.isfinite(arr).sum(0), 'min' if np.array(arr).shape[1] > 10 else 'tolist')(),
                'Minimum' : [f'{a:>6.2f}' for a in np.nanmin(arr, 0)],
                'Maximum' : [f'{a:>6.2f}' for a in np.nanmax(arr, 0)],
            }.items()]), '\n')

            if hasattr(arr, 'head'):
                print('First sample:')
                print(arr.head(1).to_string(index=False), '\n---------------------------\n')


def generate_estimates(args, bands, x_train, y_train, x_test, y_test, slices, locs=None):
    estimates, slices = get_estimates(args, x_train, y_train, x_test, y_test, slices)
    estimates = np.median(estimates, 0)
    benchmarks = run_benchmarks(args.sensor, x_test, y_test, x_train, y_train, slices, args)
    for p in slices: 
        if p not in benchmarks: benchmarks[p] = {}
        benchmarks[p].update({'MDN' : estimates[..., slices[p]]})
    return benchmarks


def main(kwargs,plot_matchups=False,run_bbp=False):
    args = get_args(kwargs,use_cmdline=False)
    if plot_matchups:
        import pickle

        # # args.no_load=False
        # dictionary_of_matchups = get_matchups(args.sensor)    
        # # print(dictionary_of_matchups.keys())
        # # print(dictionary_of_matchups['site_label'])
        # insitu_Rrs = np.reshape(dictionary_of_matchups['insitu_Rrs_resampled'],(1,-1,len(get_sensor_bands(args.sensor))))
        # remote_Rrs = np.reshape(dictionary_of_matchups['Rrs_retrieved'],(1,-1,len(get_sensor_bands(args.sensor))))

        
        # with open('insitu_Rrs.pickle', 'wb') as insitu_Rrs_file:
        #     pickle.dump(insitu_Rrs, insitu_Rrs_file, protocol=pickle.HIGHEST_PROTOCOL)
        # with open('remote_Rrs.pickle', 'wb') as remote_Rrs_file:
        #     pickle.dump(remote_Rrs, remote_Rrs_file, protocol=pickle.HIGHEST_PROTOCOL)
        # with open('dictionary_of_matchups.pickle', 'wb') as dictionary_of_matchups_file:
        #     pickle.dump(dictionary_of_matchups, dictionary_of_matchups_file, protocol=pickle.HIGHEST_PROTOCOL)

        with open('insitu_Rrs.pickle', 'rb') as insitu_Rrs_file:
            insitu_Rrs = pickle.load(insitu_Rrs_file)
            with open('remote_Rrs.pickle', 'rb') as remote_Rrs_file:
                remote_Rrs = pickle.load(remote_Rrs_file)  
                with open('dictionary_of_matchups.pickle', 'rb') as dictionary_of_matchups_file:
                    dictionary_of_matchups = pickle.load(dictionary_of_matchups_file)  
            
                    print("In situ shape",np.shape(insitu_Rrs),"Remote shape",np.shape(remote_Rrs))
                        
                    estimates_in_situ = image_estimates(data=insitu_Rrs,args=args,**kwargs)
                    estimates_remote  = image_estimates(data=remote_Rrs,args=args,**kwargs)
                    plot_remote_insitu(y_remote=estimates_remote, y_insitu=estimates_in_situ,dictionary_of_matchups=dictionary_of_matchups,products=['chl','tss','pc','aph','ag','ad'],sensor=args.sensor,run_name="sample",args=args)
        assert(0)

    #args.benchmark = True
    # If a file was given, estimate the product for the Rrs contained within
    if args.filename:
        filename = Path(args.filename)
        assert(filename.exists()), f'Expecting "{filename}" to be path to Rrs data, but it does not exist.'

        bands = get_sensor_bands(args.sensor, args)
        if filename.is_file(): x_test = np.loadtxt(args.filename, delimiter=',')
        else:                  x_test, *_ = _load_datasets(['Rrs'], [filename], bands)

        print(f'Generating estimates for {len(x_test)} data points ({x_test.shape})')
        print_dataset_stats(rrs=x_test, label='Input')

        estimates, slices = get_estimates(args, x_test=x_test)
        estimates = np.median(estimates, 0)
        print_dataset_stats(estimates=estimates, label='MDN')

        labels    = get_labels(bands, slices, estimates.shape[1])
        estimates = np.append([labels], estimates, 0).astype(str)
        filename  = filename.parent.joinpath(f'MDN_{filename.stem}.csv').as_posix()
        
        print(f'Saving estimates at location "{filename}"')
        np.savetxt(filename, estimates, delimiter=',', fmt='%s')

    # Save data used with the given args
    elif args.save_data:
        x_data, y_data, slices, locs, lat_lon_data = get_data(args)

        valid  = np.any(np.isfinite(x_data), 1) # Remove samples which are completely nan
        x_data = x_data[valid].astype(str)
        y_data = y_data[valid].astype(str)
        locs   = np.array(locs)[valid].astype(str)
        wvls   = list(get_sensor_bands(args.sensor, args).astype(int).astype(str))
        lbls   = get_labels(get_sensor_bands(args.sensor, args), slices, y_data.shape[1])
        data_full = np.append(np.append(locs, y_data, 1), x_data, 1)
        data_full = np.append([['dataset', 'index']+lbls+wvls], data_full, 0)
        filename  = f'{args.sensor}_data_full.csv'
        np.savetxt(filename, data_full, delimiter=',', fmt='%s')
        print(f'Saved data with shape {data_full.shape} to {filename}')

    # Train a model with partial data, and benchmark on remaining
    elif args.benchmark and not run_bbp:
        import matplotlib.pyplot as plt
        plot_aph_vs_chl=False
        if plot_aph_vs_chl:
            print(args)
            input('Hold')
            if args.dataset == 'sentinel_paper':
                setattr(args, 'fix_tchl', True)
                setattr(args, 'seed', 1234)

            np.random.seed(args.seed)
            
            bands   = get_sensor_bands(args.sensor, args)
            n_train = 0.75 if args.dataset != 'sentinel_paper' else 1000
            x_data, y_data, slices, locs, lat_lon_data = get_data(args)

            #(x_train, y_train), (x_test, y_test), train_idxs, test_idxs = split_data(x_data, y_data, n_train=n_train, seed=args.seed)

            #benchmarks = generate_estimates(args, bands, x_train, y_train, x_test, y_test, slices, locs)
            
            

            
            aph_idx = 3
            aph_truth_values = y_data[:,slices['aph']][:,aph_idx]
            chl_truths = y_data[:,slices['chl']]
            fig1 = plt.figure()
            ax1 = fig1.add_subplot(111)
            plt.scatter(np.squeeze(chl_truths),aph_truth_values,label='Full Dataset',c='k')
            #colormap = plt.cm.nipy_spectral #I suggest to use nipy_spectral, Set1,Paired
            import cycler
            import matplotlib as mpl
            #ax1.set_color_cycle([colormap(i) for i in np.linspace(0, 1,number_of_plots)])
            # n = 100
            # color = plt.cm.viridis(np.linspace(0, 1,n))
            # mpl.rcParams['axes.prop_cycle'] = cycler.cycler('color', color)
            colormap = plt.cm.nipy_spectral
            colors = [colormap(i) for i in np.linspace(0, 1,50)]
            ax1.set_prop_cycle('color', colors)
            for i,loc in enumerate(np.unique(locs[:,0])):
                print(i,loc)
                if i == 36:
                    plt.scatter(np.squeeze(chl_truths[locs[:,0] == loc]),aph_truth_values[locs[:,0] == loc],label=i,c='m')                    
                else:
                    plt.scatter(np.squeeze(chl_truths[locs[:,0] == loc]),aph_truth_values[locs[:,0] == loc],label=i)
            plt.legend()
            plt.axis([.1, 10000,.001, 100])
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel('Chl a (mgm-3)')
            plt.ylabel('aph (443) (m-1)')
            plt.grid()
            plt.show()
            
            fig2 = plt.figure()
            #Plot the aph data for the Dutch Lakes
            ax1 = fig2.add_subplot(111)

            x_data[np.squeeze(np.isnan(chl_truths)) * np.squeeze([locs[:,0] == 'SeaBASS3'])]
            plt.plot(get_sensor_bands(args.sensor, args),y_data[np.squeeze(~np.isnan(aph_truth_values)) * np.squeeze([locs[:,0] == 'SeaBASS3'])].T[0:len(get_sensor_bands(args.sensor, args)),:])
            plt.show()
            
            fig3 = plt.figure()
            loc = 'SeaBASS3'
            chl_truths_seabass = np.log10(np.squeeze(chl_truths[locs[:,0] == loc]))
            aph_truths_seabass =  np.log10(aph_truth_values[locs[:,0] == loc])
            chl_0_1 = np.logical_and(chl_truths_seabass > 0, chl_truths_seabass < 1)
            aph_neg_1_0 = np.logical_and(aph_truths_seabass>-.3, aph_truths_seabass<0.1)
            chl_aph_ranges = np.logical_and(chl_0_1,aph_neg_1_0)
            seabass_locs = locs[locs[:,0]=='SeaBASS3'][:,1][np.where(chl_aph_ranges)] #SeaBASS2_discover_aq_2011_442, ECU/MILLER/GEOCAPE_CBODAQ/DISCOVER_AQ_2011/archive/CBODAQ_OpticsProfiler_01.sb, ag, agp, cgp, bbp
            
            plt.scatter(10**chl_truths_seabass[chl_aph_ranges],10**aph_truths_seabass[chl_aph_ranges])
            plt.show()

            assert(0)               
        print(args)
        # input('Hold')
        if args.dataset == 'sentinel_paper':
            setattr(args, 'fix_tchl', True)
            setattr(args, 'seed', 1234)
        import numpy  as np

        np.random.seed(args.seed)
        
        bands   = get_sensor_bands(args.sensor, args)
        n_train = 0.5 if args.dataset != 'sentinel_paper' else 1000
        x_data, y_data, slices, locs, lat_lon_data = get_data(args)
        # input('Hold')
        (x_train, y_train), (x_test, y_test), train_idxs, test_idxs = split_data(x_data, y_data, n_train=n_train, seed=args.seed)

        benchmarks = generate_estimates(args, bands, x_train, y_train, x_test, y_test, slices, locs)
        # aph_idx = 3
        # aph_estimates_MDN_lowest_wvl = benchmarks['aph']['MDN'][:,aph_idx]
        # aph_truth_values = y_test[:,slices['aph']][:,aph_idx]
        # chl_truths = y_test[:,slices['chl']][:]
        # plt.scatter(np.squeeze(chl_truths),aph_truth_values)
        # plt.xscale('log')
        # plt.yscale('log')
        # plt.xlabel('Chl (mgm-3)')
        # plt.ylabel('aph (m-1)')

        # plt.plot()
        
        # symmetric_accuracy = 100 * (np.exp((np.abs(np.log(aph_estimates_MDN_lowest_wvl / aph_truth_values)))) - 1)
        # locs_high_conc = np.logical_and(aph_truth_values > 0.5,aph_truth_values < 5)
        # sorted_symmetric_accuracy_high_conc, sorted_loc_idx_with_high_conc, aph_truths_sorted_accuracy = zip(*sorted(zip(symmetric_accuracy[locs_high_conc], locs[test_idxs[locs_high_conc]][:,0],aph_truth_values[locs_high_conc]))) 
        
        # print(sorted_symmetric_accuracy_high_conc)
        # print(sorted_loc_idx_with_high_conc)
        # print(aph_truths_sorted_accuracy)


        labels     = get_labels(bands, slices, y_test.shape[1], wavelengths_ad_ag= get_sensor_bands(f'{args.sensor}-adag', args) if args.use_HICO_aph else None, wavelengths_aph=get_sensor_bands(f'{args.sensor}-aph', args) if args.use_HICO_aph else None,use_HICO_aph=args.use_HICO_aph) #get_labels(get_sensor_bands('HICO-aph', args) if args.use_HICO_aph else bands, slices, y_test.shape[1])
        products   = args.product.split(',')
        args.summary_stats = {}
        #Split by product
        for product in products:
                plot_scatter(y_test[:,slices[product]], benchmarks, bands, labels[slices[product]], product, args.sensor,args=args)
                if product in ['aph'] and True: 
                    plot_spectra(y_test[:,slices[product]], benchmarks, bands, labels[slices[product]], product, args.sensor,args=args,y_full=y_test,slices=slices)
        
        import pickle
        file = open('scatter_plots/' + args.config_name+'/args.pkl', 'wb') #file = open('/home/ryanoshea/in_situ_database/Working_in_situ_dataset/scatter_plots/b89adcedb5c1a74c06fcf0e7668925df18633db48a5fbef55a2c9b830526de74'+'/args.pkl', 'rb')
        pickle.dump(args,file)
        return args
        #saves the plots and config to a folder with the models name

    #Test a Bbp model out
    elif run_bbp:
         

        
        from .plot_utils import plot_bbp_error,plot_bbp_spectra
        
        import numpy  as np
        from   pylab import text
        from   spectral import spectral_angles
        import matplotlib.pyplot as plt
        import math, random
        random.seed(43)
        
        
        products_str = "aph,chl,tss,pc,ad,ag,cdom,anw" #,anw
        
        removed_dataset_holder        = args.removed_dataset
        args.removed_dataset          = "SeaBASS_bb\\Zimmerman_Richard\\Seagrass_Mapping_Florida\\Fwtic2010\\requested_files\\ODU\\archive"
        args.product                  = "bbp"
        
        x_data, y_data, slices, locs, lat_lon_data = get_data(args)
       
        sensor                        = "HICO"
        Rrs                           = x_data 
        wavelengths                   = get_sensor_bands(args.sensor, args)
        wavelengths_adg               = get_sensor_bands(args.sensor+'-adag', args)
        wavelengths_aph               = get_sensor_bands(args.sensor+'-aph', args)
        wavelengths_bbp               = get_sensor_bands('bbp', args)

        args.removed_dataset          = removed_dataset_holder
        args.product                  = products_str 
        
        estimates, slices             = get_estimates(args, x_test = Rrs, output_slices=slices, dataset_labels=locs[:,0])
        estimates                     = np.median(estimates, 0)
        
        bbp_dictionary                = get_bbp_estimates(Rrs=Rrs,estimates=estimates,slices=slices,args=args,outputs=['Gordon','GIOP','QAA'])
        
        resample_wavelength_locations = [] 
        found_bbp_wavelengths         = []
        found_bbp_wavelengths_index   = []
        
        for i,bbp_wavelength in enumerate(wavelengths_bbp):
            try:
                resample_wavelength_locations.append(find_wavelength(bbp_wavelength,wavelengths[0:50],tol=5))
                found_bbp_wavelengths.append(bbp_wavelength)
                found_bbp_wavelengths_index.append(i)
            except:
                print(bbp_wavelength, "not found")
        
        def get_OWT_classification(Rrs,wavelengths,k=3,shuffle=True,sensor='HICO'):
            classes              = OWT_classification(Rrs,wavelengths,sensor= sensor)
            classes_dict         = {OWT:  [index for index,OWT_bool in enumerate(classes==OWT) if OWT_bool]  for OWT in range(1,8)}
            if shuffle:  
                for i in range(1,8): random.shuffle(classes_dict[i])
            
            return classes, classes_dict
        
        classes, classes_dict = get_OWT_classification(Rrs,wavelengths,k=3)


        # plot_bbp_error(bbp_Gordon,bbp_default_QAA,bbp_default_GIOP,bbp_truth,found_bbp_wavelengths,label='Gordon')
        plot_bbp_error(bbp_dictionary,resample_wavelength_locations,found_bbp_wavelengths,found_bbp_wavelengths_index,bbp_truth=y_data,classes=classes,locs=locs,label='Gordon')
        plot_bbp_spectra(wavelengths_aph,found_bbp_wavelengths,found_bbp_wavelengths_index,resample_wavelength_locations,bbp_dictionary,bbp_truth=y_data,classes_dict=classes_dict,plot_prefix='MULT_bbp')
        
        
        #Spectral Plots of Hyper bbp
        data_loc_holder                            = args.data_loc
        args.data_loc                              = "/home/ryanoshea/in_situ_database/Working_in_situ_dataset/Brice_Grunert_hyper_bbp_processed/"
        args.product                               = "bbp"

        x_data, y_data, slices, locs, lat_lon_data = get_data(args)
        Rrs                                        = x_data*np.pi
        args.product                               = products_str
        args.data_loc                              = data_loc_holder
        estimates, slices                          = get_estimates(args, x_test = Rrs, output_slices=slices, dataset_labels=locs[:,0])
        estimates                                  = np.median(estimates, 0)
        bbp_dictionary                             = get_bbp_estimates(Rrs=Rrs,estimates=estimates,slices=slices,args=args,outputs=['Gordon','GIOP','QAA'])
        wavelengths_bbp                            = np.array(range(430,705,5))
        classes, classes_dict                      = get_OWT_classification(Rrs,wavelengths,k=3)
         
        plot_bbp_spectra(wavelengths_aph,found_bbp_wavelengths,found_bbp_wavelengths_index,resample_wavelength_locations,bbp_dictionary,bbp_truth=y_data,classes_dict=classes_dict,plot_prefix='hyper_bbp')


        
        assert(0)
            
        
        
    # Otherwise, train a model with all data (if not already existing)

    else:
        if 'latlon' in args.product.split(','):  
            x_data, y_data, slices, locs, latlons = get_data(args)
            latlons = [i.split(',') for i in latlons]
            import pandas as pd
            latlons_df = {'latlons' : latlons}
            latlons_df = pd.DataFrame(latlons_df)
            latlons_df2 = pd.DataFrame(latlons_df['latlons'].to_list(), columns=['lat','lon']).to_csv('lat_lons.csv',index=False)
        else:
            x_data, y_data, slices, locs, lat_lon_data = get_data(args)
        import numpy as np
        lat_lon_data = np.squeeze(lat_lon_data)
        np.savetxt("lat_lon_data.csv",lat_lon_data,delimiter = ",")
        products   = args.product.split(',') 
        plot_histogram(y_data,products,slices,locs)
        
        import pickle
        with open('HICO_simultaneuos_MDN.pickle', 'wb') as handle:
                pickle.dump([x_data, y_data, slices, locs], handle, protocol=pickle.HIGHEST_PROTOCOL)        
        get_estimates(args, x_data, y_data, output_slices=slices, dataset_labels=locs[:,0])
