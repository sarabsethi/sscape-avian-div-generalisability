import os
import numpy as np
from analysis_libs import get_aggregated_feats, null_corr_rs, corr
from tqdm import tqdm

analysed_data_dir = 'analysed_data'
if not os.path.exists(analysed_data_dir):
    os.makedirs(analysed_data_dir)

for corr_type in ['spearman', 'pearson']:
    feat_mode = 'mean'
    abs_rs = False
    n_null_perms = 100

    parsed_data_dir = 'parsed_pc_data'
    parsed_data_fs = os.listdir(parsed_data_dir)
    parsed_data_fs = [f for f in parsed_data_fs if 'safe-' not in f]
    #parsed_data_fs = parsed_data_fs[:2]
    #parsed_data_fs = ['india_maad.npy', 'us_maad.npy']

    for savef_ix, savef in enumerate(parsed_data_fs):
        savef_real_rs = []

        np.random.seed(42)

        pc_data_savef = os.path.join(parsed_data_dir, savef)

        print('Loading parsed PC data from {}'.format(pc_data_savef))
        all_pcs = np.load(pc_data_savef, allow_pickle=True)
        all_pcs = np.asarray(all_pcs)

        aud_feat_dims = all_pcs[0].audio_feats.shape[all_pcs[0].audio_feats.ndim-1]

        print('Getting diversity metrics from point counts')
        bio_divs = []
        aud_feat_mat = np.empty((len(all_pcs), aud_feat_dims))
        for pc_ix, pc in enumerate(all_pcs):
            bio_divs.append(len(np.unique(pc.avi_spec_comm)))
            aud_feat_mat[pc_ix,:] = get_aggregated_feats(pc, feat_mode)


        print('Calculating all null and actual correlations')
        all_real_rs = []
        all_null_rs = []
        for f_ix, f in enumerate(tqdm(aud_feat_mat.T)):
            #if np.std(f) == 0: continue
            real_r, real_p = corr(f, bio_divs, type=corr_type)
            if abs_rs: real_r = np.abs(real_r)

            null_rs = null_corr_rs(f, bio_divs, n_perms=n_null_perms, abs_rs=abs_rs, type=corr_type)

            all_real_rs.append(real_r)
            all_null_rs.append(null_rs)

        all_real_rs = np.asarray(all_real_rs)
        all_null_rs = np.asarray(all_null_rs)

        all_real_rs = np.nan_to_num(all_real_rs)
        all_null_rs = np.nan_to_num(all_null_rs)

        np.save(os.path.join(analysed_data_dir, '{}_corr_{}'.format(corr_type, savef)), [savef, all_real_rs, all_null_rs])
