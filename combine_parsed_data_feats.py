import os 
import numpy as np
from tqdm import tqdm
from skimage.measure import block_reduce

all_datasets = ['safe', 'india', 'us', 'taiwan']

parsed_data_dir = 'parsed_pc_data'
parsed_data_fs = os.listdir(parsed_data_dir)
parsed_data_fs = [f for f in parsed_data_fs if 'safe-' not in f]

for dataset in all_datasets:
    parsed_maad_path = os.path.join(parsed_data_dir, '{}_maad.npy'.format(dataset))
    parsed_vggish_path = os.path.join(parsed_data_dir, '{}_vggish.npy'.format(dataset))

    print('Loading maad PC data from {}'.format(parsed_maad_path))
    all_maad_pcs = np.load(parsed_maad_path, allow_pickle=True)
    all_maad_pcs = np.asarray(all_maad_pcs)

    print('Loading vggish PC data from {}'.format(parsed_vggish_path))
    all_vggish_pcs = np.load(parsed_vggish_path, allow_pickle=True)
    all_vggish_pcs = np.asarray(all_vggish_pcs)

    all_vggish_pc_ids = np.asarray([pc.id for pc in all_vggish_pcs])

    print('Combining mean features for: {}'.format(dataset))

    all_combined_pcs = []
    for maad_pc in tqdm(all_maad_pcs):
        matching_vggish_ix = np.where((all_vggish_pc_ids == maad_pc.id))[0][0]
        vggish_pc = all_vggish_pcs[matching_vggish_ix]

        mean_maad_feats = np.asarray(maad_pc.audio_feats)
        if mean_maad_feats.ndim > 1: 
            mean_maad_feats = np.mean(mean_maad_feats, axis=0)
        mean_vggish_feats = np.mean(vggish_pc.audio_feats, axis=0)

        assert(mean_vggish_feats.shape[0] + mean_maad_feats.shape[0] == 188)     

        combined_feats = np.hstack((mean_vggish_feats, mean_maad_feats))

        comb_pc = maad_pc 
        comb_pc.audio_feats = combined_feats
        all_combined_pcs.append(comb_pc)
    
    np.save(os.path.join('parsed_pc_data','{}_combined.npy'.format(dataset)), all_combined_pcs)


        