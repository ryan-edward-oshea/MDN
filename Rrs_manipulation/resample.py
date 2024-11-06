# from scipy.interpolate import Akima1DInterpolator as Akima
from scipy.interpolate import CubicSpline as Akima

from tqdm import tqdm
import pandas as pd 
import numpy as np
import os 

'''
rsr funcs:
https://oceancolor.gsfc.nasa.gov/docs/rsr/
'''

def read(filename):
    if 'CCNY' in filename:
        data = np.loadtxt(filename, delimiter=',', dtype=str)
        keys = ['Rrs', 'ap_']
        head = data[0]
        cols = [any([k in h for k in keys]) for h in head]
        head = np.array([''.join([h.replace(k, '') for k in keys if k in h]) for h in head])
        data = data[1:, cols]
        data = np.append(head[None, cols], data, axis=0).astype(np.float32)
    else:
        data = pd.read_csv(filename, header=None)
        data = data.apply(pd.to_numeric, errors='coerce').to_numpy()
    return data

def resample(rsr, rsr_idx, data, data_idx, W):
    valid = np.isfinite(data)
    if data_idx[valid].size:
        idx_pairs = np.where(np.diff(np.hstack(([False],np.diff(data_idx[valid])<5,[False]))))[0].reshape(-1,2)
        if idx_pairs.size:
            start_longest_seq = idx_pairs[np.diff(idx_pairs,axis=1).argmax(),0]
            end_longest_seq = idx_pairs[np.diff(idx_pairs,axis=1).argmax(),1]
            if data_idx[valid][end_longest_seq] - data_idx[valid][start_longest_seq]  < 50:
                start_longest_seq = -1
                end_longest_seq = -1
        else:
            start_longest_seq = -1
            end_longest_seq = -1
        j=-1
        valid_hyp=[]
        for i,valid_i in enumerate(valid):
            if valid_i:
                j=j+1
            if j >=start_longest_seq and j <=end_longest_seq:
                valid_hyp.append(valid_i)
            else:
                valid_hyp.append(False)
        # print(np.shape(valid),np.shape(valid_hyp))
        valid=np.asarray(valid_hyp)
        # print(data_idx[valid])
        #valid = np.asarray([ valid_i if i >=start_longest_seq and i <=end_longest_seq else False  for i,valid_i in enumerate(valid) ])
    if valid.sum() and data_idx[valid].size:
        min_avail = data_idx[valid].min()
        max_avail = data_idx[valid].max()
        # finds the longest chain of consecutive numbers, uses these as new valid range
        
        # print(valid,sum(valid),data_idx[valid], data[valid])
        # if np.max(np.diff(data_idx[valid]))>10:
        #     print("DATA IS NOT HYPERSPECTRAL",data_idx[valid],np.max(np.diff(data_idx[valid])))


        if sum(valid)>1:
            #Breaks the loop if the data is not hyperspectral...
            
            interp    = Akima(data_idx[valid], data[valid])
            averaged  = []
            for band in range(len(rsr)):                
                curr_rsr  = rsr[band]
                curr_idx  = rsr_idx[band]
                if (curr_rsr[(curr_idx >= min_avail) & (curr_idx <= max_avail)].sum() / curr_rsr.sum()) > 0.9:
                    averaged.append(np.nansum(interp(curr_idx) * curr_rsr * W[band]) / np.nansum(curr_rsr * W[band]))
                else:
                    averaged.append(np.nan)        
            return averaged
    return [np.nan for band in range(len(rsr))]

def create(name, folder, sensor, LUT, filtered=False, use_f0=False, square=False):
    # sensor   = 'S2B'
    # LUT      = True  # Create training data (LUT) or testing data (in situ)
    # filtered = False  # Create filtered dataset for LUT based on QAA bbp error


    if LUT:    
        # folder  = 'Train/Generated' 
        # name    = 'Rrs'
        hyp_idx = read('%s/HYP_wavelengths' % folder)
        hyp_val = read('%s/HYP/%s.csv' % (folder, name))
        if hyp_val.shape[0] < hyp_val.shape[1]:
            hyp_val = hyp_val.T
        assert(hyp_val.shape[1] == len(hyp_idx)), [hyp_val.shape, hyp_idx.shape]
        # valid = hyp_idx <= 715
        # hyp_idx = hyp_idx[valid]
        # hyp_val = hyp_val[:, valid]
        # print(hyp_val.shape)         

        # b_wvl, bbw = np.loadtxt('IOP/bbw', delimiter=',').T 
        # b_wvl, bbw = np.loadtxt('IOP/aw', delimiter=',').T 
        # hyp_val += bbw             

        if filtered:
            keep = read('Train/LUT_filtered_keep.csv').astype(np.bool)
            print(keep.shape, keep[:10])
            print('Keeping %s / %s LUT samples after filter' % (keep.sum(), hyp_val.shape[0]))
            hyp_val = hyp_val[keep]

    else:     
        # folder  = 'Train/IOCCG'
        # folder  = 'Test/Full'
        # name    = 'a_p'
        insitu  = read('%s/HYP/%s.csv' % (folder, name))
        hyp_idx = insitu[0]
        hyp_val = insitu[1:]
        # hyp_val[np.isnan(hyp_val)] = 0
        assert(hyp_idx[0] >= 100), hyp_idx[0]
        # hyp_val[np.isnan(hyp_val)] = 0
        # hyp_idx_lut = read('Train/Full/HYP_wavelengths')
        # valid = np.logical_and(hyp_idx_lut >= hyp_idx.min(), hyp_idx_lut <= hyp_idx.max())
        # hyp_idx_lut = hyp_idx_lut[valid]
        # # hyp_idx_lut = hyp_idx_lut[hyp_idx_lut < 715]
        # hyp_val = Akima(hyp_idx, hyp_val.T)(hyp_idx_lut).T
        # hyp_idx = hyp_idx_lut

        # b_wvl, bbw = np.loadtxt('IOP/bbw', delimiter=',').T 
        # b_wvl, bbw = np.loadtxt('IOP/aw', delimiter=',').T 
        # hyp_val += bbw[valid]      

    nLw = False
    if name == 'nLw':
        f0   = np.loadtxt('/home/ryanoshea/in_situ_database/Working_in_situ_dataset/IOP/f0.txt', delimiter=',')
        nLw  = True
        name = 'Rrs'

    rsr = read('Rsr/%s_rsr.csv' % sensor)
    rsr[np.isnan(rsr)] = 0
    rsr[rsr < -50] = 0 # Some sheets have e.g. -999 as a placeholder

    bands = rsr[0, 1:]
    waves = rsr[1:, 0]
    rsr   = rsr[1:,1:]

    if square:
        # https://oceancolor.gsfc.nasa.gov/docs/ocssw/atmocor2_8c_source.html
        # https://oceancolor.gsfc.nasa.gov/docs/ocssw/nlw__outband_8c_source.html
        # https://oceancolor.gsfc.nasa.gov/docs/ocssw/brdf_8c_source.html#l00040
        # unsure what to do about brdf correction..
        # brdf is applied _after_ the outband correction, which means there's no way to calculate the nominal
        # values from the full, or vice versa, as you would need the original nLw prior to corrections due to 
        # it being used within the correction itself.
        part_f0 = np.array([188.75411818181817, 201.44287272727271, 179.27043636363638, 151.67255454545455])
        full_f0 = np.array([189.55582113809504, 200.4591433751529, 182.0736679998659, 154.94912082061634])
        # full_f0 = np.array([189.652, 200.396, 182.079, 155.038]) # incorrect, but what seadas uses... # S3
        a0 = np.array([9.931e-1, 9.771e-1, 1.047e0, 9.682e-1])
        a1 = np.array([2.053e-3, 4.611e-2, -2.140e-2, -6.174e-3])
        a2 = np.array([-1.642e-4, -5.545e-3, 1.682e-3, 7.386e-4])
        # rsr = np.logical_and((bands.round()-5)[None,:] <= waves[:, None], waves[:, None] <= (bands.round()+5)[None,:]).astype(float) # S2

    # Rsr can extend past available hyperspectral data
    valid  = np.logical_and(waves <= hyp_idx.max(), waves >= hyp_idx.min())
    # remove = np.any(rsr[~valid] > 0, axis=0)             # No part of rsr can lie outside
    remove = (rsr[~valid].sum(0) / rsr.sum(0)) > 0.1     # Less than 10% of rsr can lie outside 
    
    print('Bands with responses outside of range [%s, %s]: \n' % (hyp_idx.min(), hyp_idx.max()), list(bands[remove]))
    bands = bands[~remove]
    waves = waves[valid]
    rsr   = rsr[:, ~remove][valid]
    print('\nCreating data for bands\n', list(bands))
    if not len(bands): return

    rsr_idx = [waves[rsr[:, band] > 0] for band in range(len(bands))]
    rsr_val = [rsr[rsr[:, band] > 0, band] for band in range(len(bands))]

    target_dir = os.path.join(folder, sensor.upper() + ('-S' if square else ''))
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    W = [np.ones(len(rsr_idx[band])) for band in range(len(rsr_val))]
    if use_f0:
        W = Akima(*np.loadtxt('/home/ryanoshea/in_situ_database/Working_in_situ_dataset/IOP/f0.txt', delimiter=',').T)
        W = [W(rsr_idx[band]) for band in range(len(rsr_val))]

    np.savetxt(os.path.join(target_dir, '%s_wvl.csv' % os.path.basename(name)), bands, delimiter=',')
    with open(os.path.join(target_dir, os.path.basename(name) + '.csv'), 'w+') as fn:
        for d in tqdm(hyp_val):
            vals = resample(rsr_val, rsr_idx, d, hyp_idx, W)
            if nLw:
                vals = [v/f for v,f in zip(vals, resample(rsr_val, rsr_idx, f0[:,1], f0[:,0], W))]
            if square:
                assert(sensor == 'OLI')
                nlw = np.array(vals[:4]) * full_f0
                ratio = nlw[1] / nlw[2]
                x = (a2 * ratio + a1) * ratio + a0
                R = (nlw * x) / part_f0
                vals = list(R) + list(vals[4:])
            # if np.any(np.isnan(vals)): print('Nan')
            fn.write(','.join([str(v) for v in vals]) + '\n')


if __name__ == '__main__':
    from glob import glob 
    # sensors = ['OLI','PACE','HICO','PRISMA','OLCI','HYPER']
    sensors = ['OLCI', 'MSI']
    #sensors = ['OLI', 'S2B', 'S2A', 'MSI', 'VI', 'OLCI', 'MODA', 'MODT', 'HICO', 'TM', 'ETM']
    #sensors += ['HICO', 'MOS', 'ETM800', 'TM', 'ETM', 'CZCS', 'OCTS', 'MERIS', 'SeaWiFS', 'PRISMA','S3A','S3B']
    root    = 'Test'
    rewrite = True
    square  = False 
    base_folder = 'C:\\Users\\asaranat\\OneDrive - NASA\\Data\\spectral_data\\Augmented_Gloria_V3_3\\'
    datasets = sorted(os.listdir(base_folder))
    #datasets = ['DutchLakes']
    folders = [f'{base_folder}{f}' for f in datasets if os.path.isdir(f'{base_folder}{f}' )]
    print(folders)
    input('Hold')
    import traceback
    for sensor in sensors:
        for folder in folders:
            print('\n---', folder, '-', sensor, '---')
            try:
                print(glob(os.path.join(folder, 'HYP', '*.csv')))
                #input('HOLD')
                for f in glob(os.path.join(folder, 'HYP', '*.csv')):
                    f = os.path.basename(f).replace('.csv', '')
                    if '_old' in f: continue
    
                    print('\t---',f,os.path.exists(os.path.join(folder, sensor, f'{f}.csv')) )
                    if not os.path.exists(os.path.join(folder, sensor, f'{f}.csv')) or rewrite:
                        print('Creating')
                        create(f, folder, LUT='Train' in folder, sensor=sensor, square=square)
            except Exception as e:
                # print(f"Failure to process due to error: {e}")
                print(f'Failure to process due to error:  {e}\n{traceback.format_exc()}')
                print('folder',folder)
                input('Waiting to acknowledge error')
#
