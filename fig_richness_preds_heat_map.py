import os
import numpy as np
import matplotlib.pyplot as plt
from analysis_libs import get_nice_dataset_name, get_nice_dataset_short_name, get_biodivs_and_feats
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import pearsonr 

feat_type = 'vggish'
feat_mode = 'mean'

fig = plt.figure(figsize=(6,5))

parsed_data_dir = 'parsed_pc_data'
parsed_data_fs = os.listdir(parsed_data_dir)

dataset_names = ['india', 'safe', 'taiwan', 'us']
parsed_data_fs = ['{}_{}.npy'.format(dsn, feat_type) for dsn in dataset_names]

all_train_bio_divs = []
all_test_bio_divs = []
all_train_aud_feat_mats = []
all_test_aud_feat_mats = []

train_ds_ixs = []
test_ds_ixs = []

for savef_ix, savef in enumerate(parsed_data_fs):
    if 'maad' in savef:
        aud_feat_dims = 60
    elif 'vggish' in savef:
        aud_feat_dims = 128
    elif 'combined' in savef:
        aud_feat_dims = 188

    np.random.seed(42)

    pc_data_savef = os.path.join(parsed_data_dir, savef)

    print('Loading parsed PC data from {}'.format(pc_data_savef))
    all_pcs = np.load(pc_data_savef, allow_pickle=True)
    all_pcs = np.asarray(all_pcs)

    print('Getting diversity metrics from point counts')
    bio_divs, aud_feat_mat = get_biodivs_and_feats(all_pcs)

    #train_test_shuffle = False if 'us_' in savef else True
    train_test_shuffle = True
    X_train, X_test, y_train, y_test = train_test_split(aud_feat_mat, bio_divs, test_size=0.3, shuffle=train_test_shuffle, random_state=42)

    all_train_aud_feat_mats.append(X_train)
    all_test_aud_feat_mats.append(X_test)

    all_train_bio_divs.append(y_train)
    all_test_bio_divs.append(y_test)

    train_ds_ixs.append([savef_ix] * len(y_train))
    test_ds_ixs.append([savef_ix] * len(y_test))

dataset_names = [get_nice_dataset_name(n) for n in parsed_data_fs]
dataset_short_names = [get_nice_dataset_short_name(n) for n in parsed_data_fs]

all_scores = []
samp_sizes = []
for fit_ds_name, fit_train_aud_feat_mat, fit_train_bio_div in zip(dataset_names, all_train_aud_feat_mats, all_train_bio_divs):
    print('Fitting RF regression model to {}'.format(fit_ds_name))

    regr = RandomForestRegressor(random_state=42)
    regr.fit(fit_train_aud_feat_mat, fit_train_bio_div)

    eval_scores = []
    for eval_ds_name, eval_test_aud_feat_mat, eval_test_bio_div in zip(dataset_names, all_test_aud_feat_mats, all_test_bio_divs):
        pred_bio_div = regr.predict(eval_test_aud_feat_mat)

        #e_score = r2_score(eval_test_bio_div, pred_bio_div, force_finite=True)
        e_score = r2_score(eval_test_bio_div, pred_bio_div)

        print('Score on {}: {}'.format(eval_ds_name, e_score))
        eval_scores.append(e_score)

    all_scores.append(eval_scores)
    samp_sizes.append(len(fit_train_bio_div))

all_scores = np.vstack(all_scores)
print(all_scores)

diag_vals = np.diagonal(all_scores)
samp_sizes = np.asarray(samp_sizes)
wd_r, wd_p = pearsonr(diag_vals, samp_sizes)
print('Correlation between diagonal R^2 and sample sizes (within-dataset): coeff = {}, p = {}'.format(round(wd_r, 3), round(wd_p, 3)))

off_diag_vals = np.reshape(all_scores[~np.eye(all_scores.shape[0], dtype=bool)], (-1, all_scores.shape[0]-1))
mean_off_diag_vals = np.mean(off_diag_vals, axis=1)
cd_r, cd_p = pearsonr(mean_off_diag_vals, samp_sizes)
print('Correlation between mean off-diagonal R^2 and sample sizes: coeff = {}, p = {}'.format(round(cd_r, 3), round(cd_p, 3)))

plt.matshow(all_scores, vmin=0, vmax=1, cmap='Greys', fignum=0)
plt.xticks(np.arange(len(dataset_short_names)), dataset_short_names)
plt.yticks(np.arange(len(dataset_short_names)), dataset_short_names)
plt.xlabel('Test dataset', fontsize=14)
plt.gca().xaxis.set_label_position('top')
plt.ylabel('Train dataset', fontsize=14)

divider = make_axes_locatable(plt.gca())
cax = divider.append_axes("right", size="5%", pad=0.2)
plt.colorbar(cax=cax, extend='min'  ).set_label(label='Coeff. of determination ($R^2$)', size=14)

figs_dir = 'figures'
if not os.path.exists(figs_dir):
    os.makedirs(figs_dir)

savefig_name = 'fig_richness_preds_heat_map.svg'
if feat_type != 'vggish':
    savefig_name = '{}_{}'.format(feat_type, savefig_name)
plt.savefig(os.path.join(figs_dir,savefig_name), bbox_inches='tight')

plt.show()
