import os
import numpy as np
import matplotlib.pyplot as plt
from analysis_libs import get_nice_featureset_name
from matplotlib.ticker import MaxNLocator


all_feat_types = ['vggish', 'maad']
all_feat_cols = ['blue', 'orange']
all_feat_dims = [128, 60]
feat_mode = 'mean'
pval_thresh = 0.05

analysed_data_dir = 'analysed_data'
corr_type = 'pearson'
all_datasets = ['india', 'safe', 'taiwan', 'us']

fig = plt.figure(figsize=(6,5))

for feat_ix, feat_type in enumerate(all_feat_types):

    feat_sig_count = [0] * all_feat_dims[feat_ix]
    feat_sig_count = np.asarray(feat_sig_count)

    for ds_ix, dataset in enumerate(all_datasets):

        combined_null_rs = []
        for f_t in all_feat_types:
            analysed_f = os.path.join(analysed_data_dir, '{}_corr_{}_{}.npy'.format(corr_type, dataset, f_t))
            savef, all_real_rs, all_null_rs = np.load(analysed_f, allow_pickle=True)
            combined_null_rs.extend(all_null_rs)

        combined_null_rs = np.asarray(combined_null_rs)
        combined_null_rs = combined_null_rs.flatten()

        aud_feat_dims = len(all_real_rs)
        sorted_null_rs = np.sort(np.abs(combined_null_rs))
        sig_mult_hyp_corr = 1 - (0.05/aud_feat_dims)
        sig_ix = int(sig_mult_hyp_corr*len(sorted_null_rs))
        sig_level = sorted_null_rs[sig_ix]
        print('significant results at {}/{} (R={})'.format(sig_ix, len(sorted_null_rs), sig_level))

        analysed_f = os.path.join(analysed_data_dir, '{}_corr_{}_{}.npy'.format(corr_type, dataset, feat_type))
        savef, all_real_rs, all_null_rs = np.load(analysed_f, allow_pickle=True)

        sig_ixs = np.where((np.abs(all_real_rs) > sig_level))[0]
        feat_sig_count[sig_ixs] += 1

    best_feat_ixs = np.where((feat_sig_count == np.max(feat_sig_count)))[0]
    print('feature ixs {}, corr with {} datasets'.format(best_feat_ixs, np.max(feat_sig_count)))

    for ds_ix, dataset in enumerate(all_datasets):
        analysed_f = os.path.join(analysed_data_dir, '{}_corr_{}_{}.npy'.format(corr_type, dataset, feat_type))
        savef, all_real_rs, all_null_rs = np.load(analysed_f, allow_pickle=True)
        print(all_real_rs[best_feat_ixs])

    w = 0.3
    bins = np.arange(len(all_datasets)+2)-0.5-(w/2)+(w*feat_ix)
    hist_data, _, _ = plt.hist(feat_sig_count, density=False, alpha=0.7, label=get_nice_featureset_name(feat_type), bins=bins, rwidth=w, facecolor=all_feat_cols[feat_ix])

    print('{}: {}'.format(feat_type, hist_data))

plt.gca().axvline(x=len(all_datasets), ymin=0, ymax=1, color='k', linestyle='--', label='Number of datasets')

plt.ylabel('Number of features correlated \nwith avian richess', fontsize=14)
plt.xlabel('Number of datasets', fontsize=14)
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.tight_layout()
plt.legend(loc='upper right', framealpha=1)

figs_dir = 'figures'
if not os.path.exists(figs_dir):
    os.makedirs(figs_dir)
plt.savefig(os.path.join(figs_dir,'fig_richness_corrs_num_datasets.svg'))

plt.show()
