import pandas as pd
from datetime import datetime
from tqdm import tqdm
import os
from sklearn.cluster import AffinityPropagation
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from tqdm import tqdm
from analysis_libs import get_secs_per_audio_feat
import urllib.request
import json

try:
    import sounddevice as sd
except ImportError:
    pass

class AudioFeat:

    def __init__(self, feat_vec, pc, offs_secs):
        self.feat_vec = feat_vec
        self.pc = pc
        self.offs_secs = offs_secs
        self.dur_secs = pc.secs_per_audio_feat

    def play_audio(self, audio_dir='/mnt/e/pc_recordings', print_str = None, blocking=False):
        sr, data = self.get_audio_data(audio_dir)

        sd.wait()
        if print_str is None:
            tqdm.write('Playing {} {} (+{}s)'.format(self.pc.site.name, self.pc.dt, self.offs_secs))
        else:
            tqdm.write(print_str)
        sd.play(data, sr, blocking=blocking)

        return sr, data

    def get_audio_data(self, audio_dir='/mnt/e/pc_recordings'):
        full_wav_path = os.path.join(audio_dir, self.pc.audio_fname + '.wav')

        sr, data = wavfile.read(full_wav_path)
        if data.ndim == 1:
            tqdm.write('Audio is mono - copying track to stereo (may not work!)')
            data = np.vstack([data] * 2).T

        samp_idx_start = int(self.offs_secs*sr)
        samp_idx_end = int((self.offs_secs+self.dur_secs)*sr)
        snippet_data = data[samp_idx_start:samp_idx_end,:]

        return sr, np.asarray(snippet_data)

class Taxon:
    '''
    Taxon data: links to GBIF database to get name, rank, phylogeny info etc
    '''

    def parse_excel_vals(self, gbif_sql_conn, t_row):
        #gbif_return = gbif_helper.web_gbif_validate(t_row['Taxon name'],t_row['Taxon type'])
        gbif_return = gbif_helper.local_gbif_validate(gbif_sql_conn,t_row['Taxon name'],t_row['Taxon type'])

        if gbif_return['status'] == 'found':
            self.tree_hier = gbif_return['hier']
            self.comm_name = t_row['Name']
            self.name = gbif_return['canon'][2]
            self.rank = gbif_return['canon'][3]

            self.is_avi = False
            self.is_herp = False
            group_str = t_row['Group']
            if not pd.isnull(group_str):
                if group_str.lower() == 'avifauna': self.is_avi = True
                if group_str.lower() == 'herpetofauna': self.is_herp = True

            return True
        else:
            tqdm.write('Couldn\'t match taxon {} ({})'.format(t_row['Taxon name'],t_row['Taxon type']))
            return False

    def get_red_list_status(self, force_remote=False):
        f_path = os.path.join('red_list_api','{}.json'.format(self.comm_name.lower().replace(' ','-')))

        if not os.path.exists(f_path) or force_remote:
            #token = '9bb4facb6d23f48efbf424bb05c0c1ef1cf6f468393bc745d42179ac4aca5fee' # Test API token
            token = 'd556dd8abceb3d69c463ef8bd00fbf76e5290905049800ad29d21206214efe98'
            enc_name = urllib.parse.quote(self.name)
            url = 'https://apiv3.iucnredlist.org/api/v3/species/{}?token={}'.format(enc_name,token)
            print('Fetching Red List info from {}'.format(url))

            req = urllib.request.Request(url)
            r = urllib.request.urlopen(req).read()
            cont = json.loads(r.decode('utf-8'))

            with open(f_path, 'w') as json_file:
                json.dump(cont, json_file)

        with open(f_path) as json_load_file:
            data = json.load(json_load_file)

            try:
                return data['result'][0]['category']
            except:
                #print('error parsing Red List API result: {}'.format(data))
                return None

    def __str__(self):
        return str(self.__dict__)


class Site:
    '''
    Monitoring site: has lat, long, elevation info
    '''
    def parse_excel_vals(self, s_row):
        # Read in basic information
        self.name = s_row['Location name']
        self.lat = s_row['Latitude']
        self.long = s_row['Longitude']
        self.elev = s_row['Elevation']

        return True

    def get_agb(self):
        mean_agbs = np.asarray([0.270780356, 0.674964289, 0.714754937, 1.103958403, 2.071538492, 2.381011351, 2.542379803, 2.707194682, 2.854348647, 2.854348647, 3.017740631, 3.566314216, 6.804037942, 6.804037942])
        site_names = np.asarray(['OP Belian', 'OP3 843', 'C Matrix', 'D Matrix', 'E1 648', 'D100 641', 'E100 edge', 'C10 621', 'Riparian 1', 'Riparian 2', 'B1 602', 'B10', 'VJR 1', 'VJR 2'])

        agb = mean_agbs[site_names == self.name]

        if len(agb) > 0: return agb[0]
        else: return -1

    def get_abbrv_name(self):
        if 'matrix' in self.name.lower():
            return 'SL'
        if 'OP' in self.name:
            return 'OP'
        if 'VJR' in self.name:
            return 'OG'
        if 'Riparian' in self.name:
            return 'RP'
        if self.name in ['E1 648', 'D100 641', 'E100 edge', 'C10 621', 'B1 602', 'B10']:
            return 'LF'


    def __str__(self):
        return str(self.__dict__)


class PointCount:
    '''
    Point count data: has ID, audio filename, site, datetime, weather, herp, avi info.
    '''

    def parse_bird_comm_data(self, id, dt, avi_spec_comm):
        # Read in basic information
        self.id = id
        self.dt = dt
        self.avi_spec_comm = avi_spec_comm

    def parse_spec_comm_data(self, id, dt, avi_spec_comm, herp_spec_comm):
        # Read in basic information
        self.id = id
        self.dt = dt
        self.avi_spec_comm = avi_spec_comm
        self.herp_spec_comm = herp_spec_comm

    def link_pc_audio_feats(self, feat_name, audio_feats):
        self.audio_feat_name = feat_name

        if 'raw_audioset_feats' in feat_name:
            self.secs_per_audio_feat = get_secs_per_audio_feat(feat_name)
        
        self.audio_feats = np.asarray(audio_feats)

    def parse_rec_excel_data(self, r_row, valid_sites):
        # Read in basic information
        self.id = r_row['Point_count_ID']
        self.audio_fname = r_row['Audio_file']
        self.weather = r_row['Weather']
        self.herp = r_row['Adi_Syamin']
        self.avi = r_row['Jani']

        # Check point count ID and audio file are not null
        if pd.isnull(r_row['Point_count_ID']) or pd.isnull(r_row['Audio_file']):
            tqdm.write('({}) Invalid ID or audio file ({}) (notes: {})'.format(self.id, self.audio_fname, r_row['Notes']))
            return False

        # Check site name is valid and match to site object
        try:
            self.site = [m for m in valid_sites if r_row['Site']==m.name][0]
        except Exception:
            tqdm.write('({}) Couldn\'t match {} to list of valid sites'.format(self.id, r_row['Site']))
            return False

        # Collate date and time into datetime object
        try:
            self.dt = datetime.combine(r_row['Date'].date(), r_row['Time'])
        except Exception:
            tqdm.write('({}) Invalid datetime {} {} (notes: {})'.format(self.id, r_row['Date'].date(), r_row['Time'], r_row['Notes']))
            return False

        return True

    def link_audio_feats(self, feat_name, feats_db_helper):
        self.audio_feat_name = feat_name
        self.secs_per_audio_feat = get_secs_per_audio_feat(feat_name)

        self.audio_feats = feats_db_helper.get_audio_feats(self.audio_fname, feat_name)
        if self.audio_feats is not None:
            return True
        else:
            self.audio_feats = np.asarray([])
            tqdm.write('({}) Couldn\'t find calculated {} for {}'.format(self.id, feat_name, self.audio_fname))
            return False

    def parse_pc_excel_data(self, all_data_df, valid_taxa):
        pc_data_df = all_data_df[all_data_df['Point_count_ID'] == self.id]

        # Check the data has been entered - 'no data' will indicate if there were actually no species
        if len(pc_data_df) == 0:
            tqdm.write('({}) No PC data found (avi={}, herp={}, site={}, dt={})'.format(self.id, self.avi, self.herp, self.site.name, self.dt))
            return False

        # Remove null and 'no data' indicators
        pc_data_df = pc_data_df[pc_data_df['Species_common_name'].notna()]
        pc_data_df = pc_data_df[pc_data_df['Species_common_name'] != 'No data']

        # Match records to valid taxa list and build species community
        self.avi_spec_comm = []
        self.herp_spec_comm = []
        self.tot_spec_comm = []
        for r_idx, r_row in pc_data_df.iterrows():
            #if r_row['Audio_visual'] is 'V': continue

            try:
                taxon = [t for t in valid_taxa if r_row['Species_common_name'] == t.comm_name][0]
            except Exception:
                taxon = None
                tqdm.write('({}) Couldn\'t match {} to list of valid taxa'.format(self.id, r_row['Species_common_name']))

            if taxon:
                if self.avi and taxon.is_avi: self.avi_spec_comm.append(taxon)
                elif self.herp and taxon.is_herp: self.herp_spec_comm.append(taxon)
                else: tqdm.write('{}: pc avi {} pc herp {} taxon avi {} taxon herp {}'.format(taxon.comm_name,self.avi,self.herp,taxon.is_avi,taxon.is_herp))
                self.tot_spec_comm.append(taxon)

        self.avi_spec_comm = list(set(self.avi_spec_comm))
        self.herp_spec_comm = list(set(self.herp_spec_comm))
        self.tot_spec_comm = list(set(self.tot_spec_comm))
        return True


    def af_prop_cluster_feats(self):
        self.af_prop_clust = AffinityPropagation().fit(self.audio_feats)

        self.af_prop_clust.affinity_matrix_ = []
        return self.af_prop_clust

    def gmm_cluster_feats(self):
        n_comps = 100
        cov_type = 'full'

        self.gmm_clust = GaussianMixture(n_components=n_comps,
                                covariance_type=cov_type,random_state=10).fit(self.audio_feats)
        self.dp_gmm_clust = BayesianGaussianMixture(n_components=n_comps,
                                covariance_type=cov_type,random_state=10).fit(self.audio_feats)

        '''
        # Plotting script to visualise weights between GMM and DP-GMM
        plt.plot(sorted(self.gmm_clust.weights_,reverse=True),label='GMM')
        plt.plot(sorted(self.dp_gmm_clust.weights_,reverse=True), label='DP-GMM')

        plt.legend()
        plt.ylabel('GMM component weight')
        plt.xlabel('GMM component number')
        plt.show()
        '''

        return self.gmm_clust, self.dp_gmm_clust

    def __str__(self):
        return str(self.__dict__)
