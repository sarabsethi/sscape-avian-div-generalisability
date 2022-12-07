import os
import numpy as np
import matplotlib.pyplot as plt
from analysis_libs import get_nice_feat_name, get_nice_dataset_short_name, get_biodivs_and_feats, lighten_color
from scipy.stats import pearsonr


all_feat_types = ['vggish', 'maad']
selected_feat_ixs = [14, 43]
# 43rd soundscape index is HFC
all_feat_cols = ['blue', 'orange']

parsed_data_dir = 'parsed_pc_data'
all_datasets = ['india', 'safe', 'taiwan', 'us']

fig, axs = plt.subplots(2, 4, sharex=True, sharey='row', figsize=(7,5))
axs = np.ravel(axs, order='F')
ax_ix = 0

for dataset in all_datasets:
    for ft_ix, feat_type in enumerate(all_feat_types):
        plt.sca(axs[ax_ix])
        ax_ix += 1

        pc_data_savef = os.path.join(parsed_data_dir, '{}_{}.npy'.format(dataset, feat_type))
        all_pcs = np.load(pc_data_savef, allow_pickle=True)

        bio_divs, aud_feat_mat = get_biodivs_and_feats(all_pcs)

        cf_ix = selected_feat_ixs[ft_ix]
        color = all_feat_cols[ft_ix]
        feat_vals = aud_feat_mat[:, cf_ix]

        print('Plotting {}: Feat {}'.format(dataset, get_nice_feat_name(feat_type, cf_ix)))

        plt.scatter(bio_divs, feat_vals, s=3, label=get_nice_feat_name(feat_type, cf_ix), color=lighten_color(color,1.4), alpha=0.3)
        plt.title('{}'.format(get_nice_dataset_short_name(dataset)))

        if ax_ix <= 2:
            plt.ylabel(get_nice_feat_name(feat_type, cf_ix))

        a, b = np.polyfit(bio_divs, feat_vals, 1)

        r, p = pearsonr(bio_divs, feat_vals)
        print('{}, pearson r^2 = {}'.format(get_nice_feat_name(feat_type, cf_ix), r**2))
        plt.gca().plot(bio_divs, a*bio_divs+b, c=color)

      
        plt.gca().text(0.95, 0.95, '$r^2$ = {}'.format(round(r**2,3)), horizontalalignment='right',verticalalignment='top',transform=plt.gca().transAxes)

fig.supxlabel('Avian richness', fontsize=14)
fig.supylabel('Mean acoustic feature value', fontsize=14)
plt.tight_layout()
plt.subplots_adjust(wspace=0.1, hspace=0.2)

figs_dir = 'figures'
if not os.path.exists(figs_dir):
    os.makedirs(figs_dir)
plt.savefig(os.path.join(figs_dir,'fig_richness_feat_scatters.svg'))

plt.show()
