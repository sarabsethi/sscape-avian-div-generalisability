# sscape-avian-div-generalisability

Supporting code for manuscript on soundscape approaches to monitoring of avian diversity.

## Setup

All code was developed using an Anaconda environment based on Python v3.8, but other versions may also work. 

Additional libraries required:
* ```tqdm```
* ```skbio```

All point count data and pre-computed acoustic features is stored on Zenodo at XXX. Download the ZIP file and unzip the contents to a directory called ```parsed_pc_data``` in this repository.

To check that data was placed properly, run ```python parsed_pc_data_stats.py``` which will provide summary statistics on each of the datasets.

## Compute univariate correlations

To compute the univariate correlations between acoustic feature vectors and species richness (and associated null correlations) run: ```python calc_corrs_with_nulls.py```. Results are saved to a directory named ```analysed_data``` as ```.npy``` files.

## Reproduce figures

Each file beginning with ```fig_``` reproduces a figure in the manuscript. Parameters such as which featureset to use, which correlation metric etc are changeable at the top of each file.

* Fig 1a: ```fig_richness_feat_corrs.py```
* Fig 1b: ```fig_richness_corrs_num_datasets.py```
* Fig 1c: ```fig_richness_feat_scatters.py```
* Fig 1d: ```fig_richness_preds_heat_map.py```
* Fig 1e: ```fig_richness_preds_heat_map.py``` with ```feat_type = 'maad'```

* Fig S1: ```fig_pdist_corrs.py``` with ```spec_richness = True```
* Fig S2: ```fig_pdist_corrs.py``` with ```feat_type = 'maad'```

Figures are saved to a directory named ```figures``` in SVG vector format.
