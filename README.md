
Scripts to perform graph representation learning modelling on molecular interactions networks of transcriptomics (protein-protein interactions) and metabolomics (metabolite-metabolite interactions) data from the PPMI and the luxPARK cohort, respectively. The models on metabolomics data can be applied to the entire luxPARK cohort, or to a subset of *de novo* patients.

### Data pre-processing, prior to ML modelling

#### ppmi_analyses 

##### ppmi_data4ML_class.R
This script performs unsupervised filters to generate data for ML modelling of snapshot data (BL) from RNAseq data.

##### Building_network_data_4ML.ipynb
A jupyter notebok that shows how to build a PPI network from STRING database files matching a matrix of pre-processed transcriptomics data.

* ppmi_data4ML_class.R employs as input transcriptomics and phenotypical data resulting from previous pre-processing scripts described in repository *statistical_analyses_cross_long_PD* for **parsing data** and **Baseline (T0) PD/HC** (ppmi_filter_gene_expression.R, ppmi_norm_gene_expression.R, ppmi_generate_pathway_level.R). 


#### luxpark_analyses 

##### Building_network_data.ipynb *
A jupyter notebok that shows how to build a PPI network from STITCH or KEGG database files matching a matrix of (log-transformed) metabolomics data.

* Unsupervised filters were not applied to metabolomics data. Building_network_data.ipynb employs as input metabolomics and phenotypical data resulting from previous pre-processing scripts described in repository *statistical_analyses_cross_long_PD* for **parsing data** and **Baseline (T0) PD/HC** (lx_extract_visit.R, lx_denovo_filter.R, lx_generate_pathway_level.R). 



### Modelling

##### cv_wandb_4ML.py
Main script to perform the training, hyperparameter tunning and cross-validation of GCN, ChebyNet and GAT models using molecular interaction networks to classify the omics profiles as signals on a graph (i.e., graph classification). The script requires a config file for the hyperparameter search (.yaml). Weights & biases are used to monitor the training.

##### cv_results.py
Extracts the cross-validated results by looking at the minimum validation loss and generates figures and results tables based on node and edge importance.

##### features_plot.py
Generates barplot with most relevant nodes and their functional annotation.
