import os
import numpy as np
import matplotlib.pyplot as plt
from analysis_libs import get_nice_featureset_short_name, get_nice_dataset_name

all_feat_types = ['vggish', 'maad']
all_feat_cols = ['blue', 'orange']
feat_mode = 'mean'
pval_thresh = 0.05

analysed_data_dir = 'analysed_data'
corr_type = 'pearson'
all_datasets = ['india', 'safe', 'taiwan', 'us']

fig, axs = plt.subplots(2, 2, sharex=True, sharey=False, figsize=(7,5))
axs = np.ravel(axs)

for ds_ix, dataset in enumerate(all_datasets):
    plt.sca(axs[ds_ix])

    combined_null_rs = []
    num_tests = 0
    for f_t in all_feat_types:
        analysed_f = os.path.join(analysed_data_dir, '{}_corr_{}_{}.npy'.format(corr_type, dataset, f_t))
        savef, all_real_rs, all_null_rs = np.load(analysed_f, allow_pickle=True)
        combined_null_rs.extend(all_null_rs)
        num_tests += len(all_real_rs)

    combined_null_rs = np.asarray(combined_null_rs)
    combined_null_rs = combined_null_rs.flatten()
    print(len(combined_null_rs))

    sorted_null_rs = np.sort(np.abs(combined_null_rs))
    sig_mult_hyp_corr = 1 - (0.05/num_tests)
    sig_ix = int(sig_mult_hyp_corr*len(sorted_null_rs))
    sig_level = sorted_null_rs[sig_ix]
    print('significant results at {}/{} (pearson\'s r={})'.format(sig_ix, len(sorted_null_rs), sig_level))

    for feat_ix, feat_type in enumerate(all_feat_types):
        analysed_f = os.path.join(analysed_data_dir, '{}_corr_{}_{}.npy'.format(corr_type, dataset, feat_type))
        savef, all_real_rs, all_null_rs = np.load(analysed_f, allow_pickle=True)

        sig_ixs = np.where((np.abs(all_real_rs) > sig_level))[0]
        num_sig = len(sig_ixs)
        print('{}: {} sig feats'.format(dataset, num_sig))

        m_bins = (np.arange(25)/20) - 0.6
        plt.hist(all_real_rs, density=False, bins=m_bins, alpha=0.7, facecolor=all_feat_cols[feat_ix], label=get_nice_featureset_short_name(feat_type))

    if ds_ix == 0:
        plt.legend(loc='upper right')

    plt.gca().axvline(x=sig_level, ymin=0, ymax=1, color='k', linestyle='--', label='$\it{p=0.05}$')
    plt.gca().axvline(x=-sig_level, ymin=0, ymax=1, color='k', linestyle='--')

    plt.title(get_nice_dataset_name(dataset))

    plt.xticks([-0.6, 0, 0.6])
    plt.yticks([])


fig.supxlabel('Corr. with avian richnesss ({}\'s R)'.format(corr_type.capitalize()), fontsize=14)
fig.supylabel('Number of acoustic features', fontsize=14)
plt.tight_layout()

figs_dir = 'figures'
if not os.path.exists(figs_dir):
    os.makedirs(figs_dir)
plt.savefig(os.path.join(figs_dir,'fig_richness_feat_corrs.svg'))

plt.show()
