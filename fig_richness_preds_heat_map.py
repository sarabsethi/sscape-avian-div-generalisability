import os
import numpy as np
import matplotlib.pyplot as plt
from analysis_libs import get_nice_dataset_name, get_nice_dataset_short_name, get_biodivs_and_feats
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from mpl_toolkits.axes_grid1 import make_axes_locatable

feat_type = 'vggish'
feat_mode = 'mean'

fig = plt.figure(figsize=(6,5))

parsed_data_dir = 'parsed_pc_data'
parsed_data_fs = os.listdir(parsed_data_dir)

parsed_data_fs = [f for f in parsed_data_fs if 'safe-' not in f and feat_type in f]
#parsed_data_fs = parsed_data_fs[:2]

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
for fit_ds_name, fit_train_aud_feat_mat, fit_train_bio_div in zip(dataset_names, all_train_aud_feat_mats, all_train_bio_divs):
    print('Fitting RF regression model to {}'.format(fit_ds_name))

    regr = RandomForestRegressor(random_state=42)
    regr.fit(fit_train_aud_feat_mat, fit_train_bio_div)

    eval_scores = []
    for eval_ds_name, eval_test_aud_feat_mat, eval_test_bio_div in zip(dataset_names, all_test_aud_feat_mats, all_test_bio_divs):
        pred_bio_div = regr.predict(eval_test_aud_feat_mat)

        e_score = r2_score(eval_test_bio_div, pred_bio_div, force_finite=True)

        print('Score on {}: {}'.format(eval_ds_name, e_score))
        eval_scores.append(e_score)

    all_scores.append(eval_scores)

all_scores = np.vstack(all_scores)
print(all_scores)

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
plt.savefig(os.path.join(figs_dir,'fig_richness_preds_heat_map.svg'), bbox_inches='tight')

plt.show()
