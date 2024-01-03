# Graph representation learning using molecular interaction networks for modelling omics data in Parkinson's disease 

<div align="center"><tr><td align="center" width="9999">
<img src="meta_data/MolInteraction_schema.png" alt="Schema of molecular interaction network modelling pipeline" width="80%" align="center">
</td></tr></div>
<!--![SSN GNN schema](meta_data/MolInteraction_schema.png) -->

This repository contains an implementation to perform graph representation learning modelling using molecular interaction networks of transcriptomics (protein-protein interactions) and metabolomics (metabolite-metabolite interactions), which is able to learn PD-specific fingerprints from the spatial distribution of molecular relationships in an end-to-end fashion. The scripts apply the graph representation learning modelling pipeline on networks of molecular interactions, where transcriptomics and metabolomics data from the PPMI and the LuxPARK cohort, respectively, are projected. 

If something is not clear or you have any question, please [open an Issue](https://gitlab.lcsb.uni.lu/elisa.gomezdelope/GRL_molecular_interactions_PD/-/issues).


## Repository structure
The analyses on both PPMI ans LuxPARK cohorts include some pre-processing steps prior to the modelling pipeline.

The main script to run the modelling pipeline, including model building, training, hyperparameter tunning and cross-validation, is the file executed by the wandb agent: `cv_wandb.py` (or `cv_wandb_DENOVO.py`). This file includes all the code necessary to extract the vector containing the omics profile of each patient, project it in the molecular interaction graph (PPI or MMI), read the hyperparameters defined from the wandb agent, train, and evaluate a GCN, GAT or ChebyNet model accordingly. The files it requires are in the same directory:
* `utils.py`: Many utility functions, including those for network construction, training and evaluation, feature relevance, etc. 
* `plot_utils.py`: Functions to create plots about the training and validation, as well as to project the node (sample) embeddings in 2D and 3D. They were used for debugging and experimentation.
* `wandb_config_*.yaml`: Config file for the hyperparameter search of each model.

In addition, `cv_wandb.py` requires as input information from the topology of the molecular interaction networks, given by the following lines of code (to be replaced if pipeline implemented on a different dataset):

```python
rna_file = "../data/data_rnaseq_ppi_4ML.csv"
labels_file = "../data/labels_4ML.csv"
ppi_score_file = "../data/ppi_score_4ML.csv"
adj_file = "../data/A_ppi_weighted_4ML.csv" 
```
```python
metab_file = "../data/data_metab_mmi_STITCH_700.csv"
labels_file = "../data/labels_4ML.csv"
mmi_score_file = "../data/mmi_score_STITCH_700.csv"
adj_file = "../data/A_mmi_weighted_STITCH_700.csv"
```
* `rna_file`, `metab_file`: Matrix of gene expression and metabolomics abundance (n x m). Rows being samples and columns representing molecules.
* `labels_file`: Patient/control binary labels (n x 2), columns being sample id, diagnosis.
* `ppi_score_file`, `mmi_score_file`: Edge list (e x 3), columns being molecule1, molecule2, weight. Each row defines an edge (interaction) in the network.
* `adj_file`: Weighted adjacency matrix (n x n).

<!--![MMI GNN schema](meta_data/schema_nn_mmi.png) -->
<div align="center"><tr><td align="center" width="9999">
<img src="meta_data/schema_nn_mmi.png" alt="Architecture of the GNN models using molecular interaction networks" width="70%" align="center">
</td></tr></div>
Other scripts used for the modelling pipeline:
* `cv_results.py`: Extracts the cross-validated results by looking at the minimum validation loss and generates figures and results tables based on node and edge importance.
* `features_plot.py`: Generates barplot with most relevant nodes and their functional annotation.
* `pw_embeddings.py`: Adds a "pathway embeddings layer" using a masked linear layer with sparse mask (defined by molecules' pathway membership) and weight matrix, mimicking the representation of a pathway functional embedding.

### Data pre-processing

In the PPMI cohort:
* `Building_network_data_4ML.ipynb`: A jupyter notebok that shows how to build a fully connected PPI network from STRING database files matching a matrix of pre-processed transcriptomics data.
* `ppmi_data4ML_class.R`: This script performs unsupervised filters to generate data for ML modelling of snapshot data (BL) from RNAseq data.
*`ppmi_data4ML_class.R` employs as input transcriptomics and phenotypical data resulting from previous pre-processing scripts described in repository [statistical_analyses_cross_long_PD](https://gitlab.lcsb.uni.lu/elisa.gomezdelope/statistical_analyses_cross_long_pd) for **parsing data** and **Baseline (T0) PD/HC** (ppmi_filter_gene_expression.R, ppmi_norm_gene_expression.R, ppmi_generate_pathway_level.R). 

In the LuxPARK cohort:
* `STITCH_cleansing.ipynb`, `KEGG_cleansing.ipynb`: Perform prunning in the original network of interactions to get only edges with medium confidence score in STITCH (>700). They also filter molecules mapping both to the metabolomics matrix and the MMI network, merging annotations based on multiple identifiers: PubChem, KEGG, ChEBI, inchikeys, CAS, SMILES.
* `Building_network_data.ipynb`: A jupyter notebok that shows how to build a fully connected MMI network from pre-processed STITCH or KEGG database files matching a matrix of (log-transformed) metabolomics data.

*Unsupervised filters were not applied to metabolomics data. 
`Building_network_data.ipynb` employs as input metabolomics and phenotypical data resulting from previous pre-processing scripts described in repository [statistical_analyses_cross_long_PD](https://gitlab.lcsb.uni.lu/elisa.gomezdelope/statistical_analyses_cross_long_pd) for **parsing data** and **Baseline (T0) PD/HC** (lx_extract_visit.R, lx_denovo_filter.R, lx_generate_pathway_level.R). 



