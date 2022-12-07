import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from analysis_libs import get_aggregated_feats, pca_transform_feats, get_nice_dataset_name, get_nice_featureset_name
from sklearn.metrics.pairwise import pairwise_distances
from scipy.spatial.distance import squareform
from skbio.stats.distance import mantel

# Method for pdist function to calculate distance between community richnesses
def spec_rich_distance(vec1, vec2):
    return (np.sum(vec2) - np.sum(vec1))


parsed_data_dir = 'parsed_pc_data'

# n_pca_comps = None means don't do PCA
n_pca_comps = None
spec_richness = False

# 'std' for standard deviation of feats, 'mean' for averages
feat_mode = 'mean'
feat_type = 'vggish'

data_fs = os.listdir(parsed_data_dir)
data_fs = [f for f in data_fs if 'safe-' not in f and feat_type in f]

num_sites = len(data_fs)
plt_rows = 4
fig, axs = plt.subplots(int((num_sites-1) / plt_rows) + 1, np.min([num_sites, plt_rows]), sharex=False, sharey=True, figsize=(13,5))
axs = np.ravel(axs)

for f_ix, f in enumerate(data_fs):

    pc_data_savef = os.path.join(parsed_data_dir, f)

    tqdm.write('Loading parsed PC data from {}'.format(pc_data_savef))
    all_pcs = np.load(pc_data_savef, allow_pickle=True)
    all_pcs = np.asarray(all_pcs)

    all_specs_comm_names = []
    for pc in all_pcs:
        for spec in pc.avi_spec_comm:
            if spec not in all_specs_comm_names:
                all_specs_comm_names.append(spec)
    all_specs_comm_names = np.unique(np.asarray(all_specs_comm_names))

    # Set up empty matrices for the audio features and the species matrix
    test_feats = all_pcs[0].audio_feats
    if test_feats.ndim > 1:
        aud_feat_dims = test_feats.shape[1]
    else:
        aud_feat_dims = test_feats.shape[0]
    aud_feat_mat = np.empty((len(all_pcs), aud_feat_dims))
    spec_mat = np.empty((len(all_pcs), len(all_specs_comm_names)))

    # Populate the audio feature and species matrices
    fail = False
    for pc_ix, pc in enumerate(all_pcs):
        feats = get_aggregated_feats(pc, feat_mode)

        if feats is None:
            fail = True
            break

        aud_feat_mat[pc_ix,:] = get_aggregated_feats(pc, feat_mode)

        # Now species matrix
        spec_vec = np.zeros(len(all_specs_comm_names))
        # Presence/absence - 1 indicates species present 0 otherwise
        for s_ix, spec_name in enumerate(all_specs_comm_names):
            if spec_name in pc.avi_spec_comm:
                spec_vec[s_ix] = 1
        # Make sure sum of presences = sum of species (sanity check)
        assert(np.sum(spec_vec) == len(pc.avi_spec_comm))
        spec_mat[pc_ix,:] = spec_vec

    if fail: continue

    aud_feat_mat = pca_transform_feats(all_pcs, n_pca_comps, aud_feat_mat)

    # Calculate pairwise distances
    tqdm.write('Calculating pairwise distances of spec_mat and aud_feat_mat')

    # Faster version of pdist - squareform is it's own inverse so takes it back to reduced vector here
    aud_feat_dists = squareform(pairwise_distances(aud_feat_mat, metric = 'euclidean', n_jobs = -1), checks=False)

    #aud_feat_dists = np.log(aud_feat_dists)

    if spec_richness:
        spec_dists = squareform(pairwise_distances(spec_mat, metric = spec_rich_distance, n_jobs = -1), checks=False)
    else:
        spec_dists = squareform(pairwise_distances(spec_mat, metric = 'jaccard', n_jobs = -1), checks=False)

    mantel_coeff, mantel_p, mantel_n = mantel(squareform(spec_dists), squareform(aud_feat_dists), alternative='greater', method='spearman')
    print('{}: mantel coeff = {}, p = {}, n = {}'.format(pc_data_savef, mantel_coeff, mantel_p, mantel_n))

    # Plot results
    plt.sca(axs[f_ix])
    ax = axs[f_ix]
    ax.hexbin(aud_feat_dists, spec_dists, bins='log', cmap='Greys')
    
    ax.set_title(get_nice_dataset_name(f))

if spec_richness:
    fig.supylabel('Species richness distances', fontsize=14)
else:
    fig.supylabel('Species community distances', fontsize=14)

fig.supxlabel('Soundscape distances ({})'.format(get_nice_featureset_name(feat_type).lower()), fontsize=14)

plt.tight_layout()

figs_dir = 'figures'
if not os.path.exists(figs_dir):
    os.makedirs(figs_dir)
plt.savefig(os.path.join(figs_dir,'fig_pdist_corrs.svg'))

plt.show()
