import os
import numpy as np
from analysis_libs import get_dataset_unq_specs

parsed_data_dir = 'parsed_pc_data'

tot_count_fs = 0
tot_count_pcs = 0

all_data_fs = os.listdir(parsed_data_dir)
#all_data_fs = [f for f in all_data_fs if 'safe-' in f]
all_data_fs = [f for f in all_data_fs if 'safe-' not in f]

for f in all_data_fs:
    p = os.path.join(parsed_data_dir, f)

    all_pcs = np.load(p, allow_pickle=True)

    all_sites = []
    if 'us' not in f:
        all_sites = np.asarray([pc.site for pc in all_pcs if pc.site is not None])
    num_sites = len(np.unique(all_sites))

    unq_specs = get_dataset_unq_specs(all_pcs)

    all_dts = [pc.dt for pc in all_pcs]
    earliest_dt = np.min(all_dts).strftime('%Y-%m-%d')
    latest_dt = np.max(all_dts).strftime('%Y-%m-%d')

    all_rec_dur_mins = []
    if 'vggish' in f:
        all_rec_dur_mins = [pc.audio_feats.shape[0] * 0.96 / 60 for pc in all_pcs]

    if 'safe-' not in f and 'vggish' in f:  
        tot_count_fs += 1
        tot_count_pcs += len(all_pcs)
    if 'safe-' in f and 'vggish' in f:
        tot_count_fs += 1
        tot_count_pcs += len(all_pcs)

    print('{}: {} PCs, {} sites, {} species, {} - {}, mean_dur = {}'.format(f, len(all_pcs), num_sites, len(unq_specs), earliest_dt, latest_dt, np.mean(all_rec_dur_mins)))

print('Total count: {} files: {} PCs'.format(tot_count_fs, tot_count_pcs))
