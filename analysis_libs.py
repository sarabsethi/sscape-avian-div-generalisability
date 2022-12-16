import numpy as np
from sklearn.decomposition import PCA
from scipy.stats import spearmanr, pearsonr


def get_nice_featureset_name(ugly_name):
    if ugly_name == 'vggish':
        return 'Learned features'
    if ugly_name == 'maad':
        return 'Soundscape indices'

    return ugly_name


def get_nice_featureset_short_name(ugly_name):
    if ugly_name == 'vggish':
        return 'LFs'
    if ugly_name == 'maad':
        return 'SSIs'

    return ugly_name


def get_nice_dataset_name(ugly_name):
    if 'safe' in ugly_name:
        return "Malaysia (tropical)"
    if 'us' in ugly_name:
        return "USA (temperate)"
    if 'india' in ugly_name:
        return "India (montane tropical)"
    if 'taiwan' in ugly_name:
        return "Taiwan (tea)"

    return ugly_name


def get_nice_dataset_short_name(ugly_name):
    if 'safe' in ugly_name:
        return "Malaysia"
    if 'us' in ugly_name:
        return "USA"
    if 'india' in ugly_name:
        return "India"
    if 'taiwan' in ugly_name:
        return "Taiwan"

    return ugly_name


def get_nice_feat_name(feat_type, feat_ix):
    if feat_type == 'maad':
        feat_abbrv = ['ZCR', 'MEANt', 'VARt', 'SKEWt', 'KURTt', 'LEQt', 'BGNt', 'SNRt', 'MED', 'Ht', 'ACTtFraction', 'ACTtCount', 'ACTtMean', 'EVNtFraction', 'EVNtMean', 'EVNtCount', 'MEANf', 'VARf', 'SKEWf', 'KURTf', 'NBPEAKS', 'LEQf', 'ENRf', 'BGNf', 'SNRf', 'Hf', 'EAS', 'ECU', 'ECV', 'EPS', 'EPS_KURT', 'EPS_SKEW', 'ACI', 'NDSI', 'rBA', 'AnthroEnergy', 'BioEnergy', 'BI', 'ROU', 'ADI', 'AEI', 'LFC', 'MFC', 'HFC', 'ACTspFract', 'ACTspCount', 'ACTspMean', 'EVNspFract', 'EVNspMean', 'EVNspCount', 'TFSD', 'H_Havrda', 'H_Renyi', 'H_pairedShannon', 'H_gamma', 'H_GiniSimpson', 'RAOQ', 'AGI', 'ROItotal', 'ROIcover'][feat_ix]

        if feat_abbrv == 'HFC':
            return 'HFC (high frequency energy)'
        if feat_abbrv == 'ROIcover':
            return 'Spectrogram cover %'
        else:
            return feat_abbrv

    if feat_type == 'vggish':
        return 'Learned feature {}'.format(feat_ix)


# Take a colour and return a lighter version of it
def lighten_color(color, amount=0.5):
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])


# Get all unique species in a dataset
def get_dataset_unq_specs(all_pcs):
    all_avi_specs = []
    for pc in all_pcs:
        all_avi_specs.extend(pc.avi_spec_comm)
    
    unq_specs = np.unique(all_avi_specs)
    return unq_specs


# Get an array representing avian richness and audio features for all point counts in a dataset
def get_biodivs_and_feats(all_pcs, feat_mode='mean'):
    aud_feat_dims = all_pcs[0].audio_feats.shape[all_pcs[0].audio_feats.ndim-1]

    bio_divs = []
    aud_feat_mat = np.empty((len(all_pcs), aud_feat_dims))
    for pc_ix, pc in enumerate(all_pcs):
        bio_divs.append(len(np.unique(pc.avi_spec_comm)))
        aud_feat_mat[pc_ix,:] = get_aggregated_feats(pc, feat_mode)

    return np.asarray(bio_divs), aud_feat_mat


# Get the average acoustic feature for a given point count
def get_aggregated_feats(pc, feat_mode):
    # First audio features
    feats = np.asarray(pc.audio_feats)

    # Calculate either the mean or standard deviation of the audio features
    if feat_mode == 'std':
        if feats.ndim == 1:
            print('Can\'t calculate std of {} shape audio feats'.format(feats.shape))

        feats = np.std(feats, axis=0)

    elif feat_mode == 'mean':
        if feats.ndim > 1:
            feats = np.mean(feats, axis=0)

    return feats


# Calculate null correlation coefficients 
def null_corr_rs(X, Y, n_perms, abs_rs=True, type='spearman'):
    null_f = np.copy(X)
    null_rs = []
    for n in range(n_perms):
        np.random.shuffle(null_f)

        if type == 'spearman':
            null_r, null_p = spearmanr(null_f, Y)
        elif type == 'pearson':
            null_r, null_p = pearsonr(null_f, Y)

        if abs_rs:
            null_r = abs(null_r)
        null_rs.append(null_r)

    null_rs = np.asarray(null_rs)
    return null_rs


# Calculate real correlation coefficients
def corr(X, Y, type='spearman'):
    if type == 'spearman':
        r, p = spearmanr(X, Y)
    elif type == 'pearson':
        r, p = pearsonr(X, Y)

    return r, p


# Fit a PCA embedding to all audio features in a dataset
def pca_fit_feats(all_pcs, n_pca_comps):
    if n_pca_comps is None: return None

    # Calculate global dimensionality reduction
    print('Calculating and applying dim red n_pca_comps={}'.format(n_pca_comps))
    all_feats = []
    for pc in all_pcs:
        if pc.audio_feats.ndim == 1:
            all_feats.append(pc.audio_feats)
        else:
            all_feats.extend(pc.audio_feats)

    all_feats = np.asarray(all_feats)

    pca = PCA(n_components=n_pca_comps)
    pca.fit(all_feats)
    #print(np.cumsum(pca.explained_variance_ratio_))

    return pca


# Transform the audio features to a lower dim PCA embedding for a given dataset
def pca_transform_feats(all_pcs, n_pca_comps, aud_feat_mat):
    # If n_pca_comps is None then don't apply PCA
    if n_pca_comps is None:
        print('n_pca_comps is None, not applying PCA')
        return aud_feat_mat

    pca = pca_fit_feats(all_pcs, n_pca_comps)

    return pca.transform(aud_feat_mat)


def get_secs_per_audio_feat(feat_name):
    target_secs_per_feat = feat_name.split('_')[-1].split('s')[0]
    if target_secs_per_feat.startswith('0'): target_secs_per_feat = float(target_secs_per_feat)/100
    else: target_secs_per_feat = float(target_secs_per_feat)

    actual_secs_per_feat = 0.96 * int(target_secs_per_feat / 0.96)
    return round(actual_secs_per_feat,2)
