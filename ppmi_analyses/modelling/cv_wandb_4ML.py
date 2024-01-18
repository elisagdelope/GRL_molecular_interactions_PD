## Imports

import argparse
import wandb
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import time
import sys
import os
import networkx as nx
import random
from utils import MyOmicsDataset
from utils import *
from models import *
from plot_utils import training_plots, pca_plot2d, pca_plot3d
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from datetime import date


if __name__ == '__main__':
    # set random seeds
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42) 
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='sweep_config', type=str)  # , default='configs/default.yaml'
    args, unknown = parser.parse_known_args()
    # set up wandb
    sweep_run = wandb.init(config=args.sweep_config, entity="elisa-gomezdelope")
    myconfig = wandb.config
    print('Config file from wandb:', myconfig)
    # set CUDA
    if torch.cuda.is_available():
        print('cuda available')
        dtypeFloat = torch.cuda.FloatTensor
        dtypeLong = torch.cuda.LongTensor
        device = torch.device('cuda')
        torch.cuda.manual_seed(42)
    else:
        print('cuda not available')
        dtypeFloat = torch.FloatTensor
        dtypeLong = torch.LongTensor
        device = torch.device('cpu')

    # I/O
    OUT_DIR = "../results/wandb/"
    datestamp = date.today().strftime('%Y%m%d')
    OUT_DIR = OUT_DIR + myconfig.model_name + "_" + datestamp + "/"
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)
    rna_file = "../data/data_rnaseq_ppi_4ML.csv"
    labels_file = "../data/labels_4ML.csv"
    ppi_score_file = "../data/ppi_score_4ML.csv"
    adj_file = "../data/A_ppi_weighted_4ML.csv" 
    rna = pd.read_csv(rna_file, index_col=0)
    labels = pd.read_csv(labels_file, index_col=0)
    Adj = pd.read_csv(adj_file, index_col=0)
    print("Dimensions of dataset:", rna.shape)
    print("Dimensions of labels:", labels.shape)
    print("Dimensions of adjacency matrix:", Adj.shape)

    # set the same gene/node order in A, X
    Adj = Adj[rna.columns].reindex(rna.columns)
    # set the same sample order in X, y
    labels = labels.reindex(rna.index)

        # X and y
    labels = labels.reindex(rna.index) # set the same sample order in X, y
    y = labels["DIAGNOSIS"] 
    y = y[y.index.isin(rna.index.to_list())]

    undersampling=True
    if undersampling:
        rus = RandomUnderSampler(random_state=42)
        X, y = rus.fit_resample(rna, y)
        X.index = rna.index[rus.sample_indices_]
        y.index = rna.index[rus.sample_indices_]
        labels_dict = y.to_dict()
        labels_dict = {k: int(v) for k, v in labels_dict.items()}
        y = np.array(y)
        X = np.array(X)
        X_indices = rna.index[rus.sample_indices_]
    else:
        labels_dict = y.to_dict()
        labels_dict = {k: int(v) for k, v in labels_dict.items()}
        y = np.array(y)
        X = np.array(rna)
        X_indices = rna.index

    features_name = rna.columns
    X_mapping = {gene: i for i, gene in enumerate(rna.columns)} # Create a mapping from gene IDs (nodes) to column indices in X (and Adj)

    # display network showing profile of 1 patient
    display_network=False
    if display_network:
        i_subject=0
        rnaseq_dict = rna.iloc[i_subject,:].to_dict() # dict for subject 0 gene: expression
        display_graph_individual(Adj, labels_dict =rnaseq_dict, save_fig=True, path="./", name_file="ppi_PPMI0_graph.png", plot_title="PPI network of genes in PPMI cohort")

    # Create graph structure from the adjacency matrix 'A' (shared across all samples)
    G = nx.from_numpy_matrix(np.array(Adj), create_using=nx.Graph)
    edge_index = torch.tensor(list(G.edges), dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(Adj.values, dtype=torch.float)[edge_index[0], edge_index[1]]

    # cv
    folds=myconfig.n_folds
    fold_v_performance = {'Fold': [], 'AUC': [], 'Accuracy': [], 'Recall': [], 'Specificity': [], 'F1': [],'N_epoch': [], 'Loss': []}
    fold_test_performance = {'Fold': [], 'AUC': [], 'Accuracy': [], 'Recall': [], 'Specificity': [], 'F1': [], 'N_epoch': []}
    features_track = {'Fold': [], 'Relevant_Nodes': [], 'Relevant_Edges_source': [], 'Relevant_Edges_dest': []}

    for fold, (train_msk, test_msk, val_msk) in enumerate(zip(*k_fold(X, y, folds))):
        # define data splits
        X_train = X[train_msk]
        y_train = y[train_msk]
        scaler = StandardScaler()
        # fit scaler to train, transform the whole X
        scaler.fit(X_train)
        print(X_train.shape)
        X_processed = scaler.transform(X)

        # Create a mapping from gene IDs (nodes) to column indices in X (and Adj)
        X_mapping = {gene: i for i, gene in enumerate(rna.columns)}
        # Create a graph from the adjacency matrix 'A', get edge index and edge_attr tensors (shared across all samples) from the weights in Adj
        G = nx.from_numpy_matrix(np.array(Adj), create_using=nx.Graph)
        edge_index = torch.tensor(list(G.edges), dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(Adj.values, dtype=torch.float)[edge_index[0], edge_index[1]]

        # Create PyG Data objects for each sample 
        dataset = []
        for sample_idx in range(X_processed.shape[0]):      
            # x : expression profile for the sample; y : label; edge_index: graph structure; edge_attr: edge weights (confidence score from ppi)
            data = Data(x=torch.tensor(X_processed[sample_idx], dtype=torch.float).unsqueeze(1), 
                        edge_index=edge_index, edge_attr=edge_attr, y=torch.tensor(y, dtype=torch.long)[sample_idx])
            dataset.append(data)

        train_data = [dataset[i] for i in torch.where(train_msk)[0]]
        val_data = [dataset[i] for i in torch.where(val_msk)[0]]
        test_data = [dataset[i] for i in torch.where(test_msk)[0]]

        train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=16, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=16, shuffle=True)

        # model
        model = generate_model(myconfig.model_name, myconfig, data.num_node_features)
        model = model.to(device)
        # compute class weights for loss function
        class_weights = class_weight.compute_class_weight(class_weight='balanced',
                                                          classes=np.unique(y_train),
                                                          y=y_train)
        class_weights = torch.tensor(class_weights, dtype=torch.float)
        criterion = nn.CrossEntropyLoss(weight=class_weights, reduction="mean")
        criterion.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=myconfig.lr, weight_decay=myconfig.weight_decay)  #lr=0.001, weight_decay=5e-4) 
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', myconfig.lrscheduler_factor, #factor=0.9, 
                                                   threshold=0.0001, patience=15,
                                                   verbose=True)
        n_epochs = myconfig.n_epochs # 150
        losses, performance, best_epoch, best_loss, best_model = training(device, model, optimizer, scheduler, criterion, train_loader, val_loader, test_loader, n_epochs, fold, wandb)

        # feature importance: top k%
        topk_percent = 1
        top_nodes, top_edges = calculate_feature_importance(device, model, train_loader, nodes_mapping= X_mapping, 
        top_k_nodes=round(0.01*topk_percent*dataset[0].x.shape[0]), 
        top_k_edges=round(0.01*topk_percent*dataset[0].edge_index.shape[1]))

        fold_v_performance, fold_test_performance, features_track = update_overall_metrics(fold, best_epoch, top_nodes, top_edges, performance, losses, fold_v_performance, fold_test_performance, features_track)

        # log performance and loss in wandb
        eval_info = {f'best_val_loss-{fold}': losses[best_epoch][1],  # val_loss at best epoch
                     f'best_val_Accuracy-{fold}': performance["Accuracy"][best_epoch][1],
                     f'best_val_AUC-{fold}': performance["AUC"][best_epoch][1],
                     f'best_val_Recall-{fold}': performance["Recall"][best_epoch][1],
                     f'best_val_Specificity-{fold}': performance["Specificity"][best_epoch][1],
                     f'best_val_F1-{fold}': performance["F1"][best_epoch][1],
                     f'best_train_AUC-{fold}': performance["AUC"][best_epoch][0],
                     f'best_test_Accuracy-{fold}': performance["Accuracy"][best_epoch][2],
                     f'best_test_AUC-{fold}': performance["AUC"][best_epoch][2],
                     f'best_test_Recall-{fold}': performance["Recall"][best_epoch][2],
                     f'best_test_Specificity-{fold}': performance["Specificity"][best_epoch][2],
                     f'best_test_F1-{fold}': performance["F1"][best_epoch][2],
 #                    f'features-{fold}': len(feat_names),
                     }
        wandb.log(eval_info)
        # reset parameters
        print('*resetting model parameters*')
        for name, module in model.named_children():
            module.reset_parameters()

    cv_metrics_to_wandb(fold_v_performance, fold_test_performance, wandb)
    print("sweep", sweep_run.name, pd.DataFrame.from_dict(fold_v_performance))
    print("sweep", sweep_run.name, pd.DataFrame.from_dict(fold_test_performance))
    print("sweep", sweep_run.name, pd.DataFrame.from_dict(features_track))
    # exports & plots performance & losses
    pd.DataFrame.from_dict(features_track).to_csv(OUT_DIR + sweep_run.name + "_features_track.csv", index=False)
    pd.DataFrame.from_dict(fold_test_performance).to_csv(OUT_DIR + sweep_run.name + "_test_performance.csv", index=False)
    pd.DataFrame.from_dict(fold_v_performance).to_csv(OUT_DIR + sweep_run.name + "_val_performance.csv", index=False)
 




