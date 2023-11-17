import torch
import pandas as pd
import numpy as np
import math
import os
from tqdm import tqdm
from torch_geometric.data import Data, InMemoryDataset
from torchmetrics import MetricCollection, AUROC, Accuracy, Precision, Recall, Specificity,F1Score
from sklearn.decomposition import PCA
from sklearn.utils import class_weight
from sklearn.model_selection import StratifiedKFold
from torch.optim import lr_scheduler
import copy
import matplotlib.pyplot as plt
from torch_geometric.explain import Explainer, GNNExplainer
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.colors import LogNorm
import torch_geometric.utils as pyg_utils
import networkx as nx
from models import *

class MyOmicsDataset(InMemoryDataset):

    def __init__(self, root, X_file, graph_file, labels_file, transform=None, pre_transform=None, pre_filter=None): #
        self.X_file = X_file
        self.graph_file = graph_file
        self.labels_file = labels_file
        super(MyOmicsDataset, self).__init__(root, transform, pre_transform, pre_filter=None)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        """ If this file exists in raw_dir, the download is not triggered.
            (The download func. is not implemented here)
        """
        return [self.X_file, self.graph_file, self.labels_file]

    @property
    def processed_file_names(self):
        """ If these files are found in processed_dir, processing is skipped"""
        return 'data.pt'

    def download(self):
        """ Download to `self.raw_dir`.
            Not implemented here
        """
        # path = download_url(url, self.raw_dir)
        pass

    def process(self):

        # load node attributes: gene expression
        X_df = pd.read_csv(self.raw_paths[0], index_col=0) # index is geneid

        # load graph: ppi
        ppi = pd.read_csv(self.raw_paths[1])

        # load labels
        labels = pd.read_csv(self.raw_paths[2], index_col=0)
        labels_dict = dict(zip(labels.index, labels[labels.columns[0]]))

        # convert df into tensors
        # map index to gene id in X_df -> {geneid: index} in node features
        X_mapping = {gene: i for i, gene in enumerate(X_df.columns)}
        src = [X_mapping[protein] for protein in ppi.iloc[:,0]] # get source nodes from first column
        dst = [X_mapping[protein] for protein in ppi.iloc[:,1]] # get destination nodes from second column
        edge_index = torch.tensor([src, dst])
        edge_attr = torch.tensor(ppi.iloc[:,2], dtype=torch.int) # get edge attributes from third column

        # create data objects
        data_list=[]
        for subject in tqdm(X_df.index.tolist()):
            ft = torch.tensor(X_df.loc[subject].astype(float).values, dtype=torch.float32).unsqueeze(1) # take the row corresponding for each subject (X_df: samples x genes]
            label = labels_dict[subject] # take the value corresponding for each subject (labels_dict is {subject: label})
            graph = Data(x=ft, edge_index=edge_index, edge_attr=edge_attr, y=label) # edge_index values are corresponding to the index in the X_df matrix  , subject=subject
            data_list.append(graph)

        # Apply the functions specified in pre_filter and pre_transform
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        # save processed data
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])



def train_epoch(device, model, loader, criterion, optimizer, metric):
    """ Train step of model on training dataset (one epoch)
    """
    model.to(device)
    model.train()
    criterion.to(device)
    emb_epoch = []
    batch_loss = 0.0
    for step, data in enumerate(loader):
        data.to(device)
        optimizer.zero_grad()  # Clear gradients
        # Perform a single forward pass
        if "_uw" in str(model.__class__.__name__):  # for unweighted models
            y_hat = model(x=data.x, edge_index=data.edge_index, batch=data.batch)
        elif "GAT" in str(model.__class__.__name__):
            y_hat = model(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, batch=data.batch)
        else:
            y_hat = model(x=data.x, edge_index=data.edge_index, edge_weight=data.edge_attr, batch=data.batch)
        loss = criterion(y_hat, data.y)  # Compute the loss
        loss.backward()  # Derive gradients
        optimizer.step()  # Update parameters based on gradients
        # track loss & embeddings
        batch_loss += loss.detach().cpu().numpy().item()
        #emb_epoch.append(h)
        # track performance
        y_hat = y_hat[:, 1]  # get positive class "probability"
        batch_acc = metric(y_hat.cpu(), data.y.cpu()) # applies sigmoid internally
    epoch_loss = batch_loss / (step + 1)
    #emb_epoch = torch.cat(emb_epoch).detach().cpu().numpy()  # tensor of all batches of the epoch
    train_acc = metric.compute()
    return epoch_loss, train_acc #, emb_epoch


def evaluate_epoch(device, model, loader, criterion, metric):
    """ Evaluate step of model on validation data
    """
    model.eval()
    batch_vloss = 0.0
    for step, data in enumerate(loader):
        data.to(device)
        # Perform a single forward pass
        if "_uw" in str(model.__class__.__name__):  # for unweighted models
            y_hat = model(x=data.x, edge_index=data.edge_index, batch=data.batch)
        elif "GAT" in str(model.__class__.__name__):
            y_hat = model(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, batch=data.batch)
        else:
            y_hat = model(x=data.x, edge_index=data.edge_index, edge_weight=data.edge_attr, batch=data.batch)
        vloss = criterion(y_hat, data.y)  # Compute the loss
        # track loss
        batch_vloss += vloss.detach().cpu().numpy()
        # track performance
        y_hat = y_hat[:, 1]  # get positive class "probability"
        batch_vacc = metric(y_hat.cpu(), data.y.cpu()) # applies sigmoid internally
    epoch_loss = batch_vloss / (step + 1)
    val_acc = metric.compute()
    return epoch_loss, val_acc


def test_epoch(device, model, loader, metric):
    """ Evaluate step of model on test data
    """
    model.eval()
    for step, data in enumerate(loader):
        data.to(device)
        # Perform a single forward pass
        if "_uw" in str(model.__class__.__name__):  # for unweighted models
            y_hat = model(x=data.x, edge_index=data.edge_index, batch=data.batch)
        elif "GAT" in str(model.__class__.__name__):
            y_hat = model(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, batch=data.batch)
        else:
            y_hat = model(x=data.x, edge_index=data.edge_index, edge_weight=data.edge_attr, batch=data.batch)
        y_hat = y_hat[:, 1]  # get label
        batch_perf = metric(y_hat.cpu(), data.y.cpu())
    test_acc = metric.compute()
    return test_acc

def training(device, model, optimizer, scheduler, criterion, loader_train, loader_val, loader_test, n_epochs, fold, wandb):
    """ Full training process
    """
    losses = []
    #embeddings = []
    perf_metrics = {'Accuracy': [], 'AUC': [], 'Recall': [], 'Specificity': [], 'F1': []}
    train_metrics = MetricCollection({
        'Accuracy': Accuracy(task="binary"),
        'AUC': AUROC(task="binary", num_classes=2),
        'Recall': Recall(task="binary", num_classes=2),
        'Specificity': Specificity(task="binary", num_classes=2),
        'F1': F1Score(task="binary", num_classes=2),
    })
    val_metrics = MetricCollection({
        'Accuracy': Accuracy(task="binary"),
        'AUC': AUROC(task="binary", num_classes=2),
        'Recall': Recall(task="binary", num_classes=2),
        'Specificity': Specificity(task="binary", num_classes=2),
        'F1': F1Score(task="binary", num_classes=2),
    })
    test_metrics = MetricCollection({
                'Accuracy': Accuracy(task="binary"),
                'AUC': AUROC(task="binary", num_classes=2),
                'Recall': Recall(task="binary", num_classes=2),
                'Specificity': Specificity(task="binary", num_classes=2),
                'F1': F1Score(task="binary", num_classes=2),
    })
    # Define the custom x axis metric
    wandb.define_metric(f'epoch_fold-{fold}')
    # Define which metrics to plot against that x-axis
    wandb.define_metric(f'val/loss-{fold}', step_metric=f'epoch_fold-{fold}')
    wandb.define_metric(f'train/loss-{fold}', step_metric=f'epoch_fold-{fold}')
    wandb.define_metric(f'val/Accuracy-{fold}', step_metric=f'epoch_fold-{fold}')
    wandb.define_metric(f'val/AUC-{fold}', step_metric=f'epoch_fold-{fold}')
    wandb.define_metric(f'val/Recall-{fold}', step_metric=f'epoch_fold-{fold}')
    wandb.define_metric(f'val/Specificity-{fold}', step_metric=f'epoch_fold-{fold}')
    wandb.define_metric(f'val/F1-{fold}', step_metric=f'epoch_fold-{fold}')
    wandb.define_metric(f'train/Accuracy-{fold}', step_metric=f'epoch_fold-{fold}')
    wandb.define_metric(f'train/AUC-{fold}', step_metric=f'epoch_fold-{fold}')
    wandb.define_metric(f'train/Recall-{fold}', step_metric=f'epoch_fold-{fold}')
    wandb.define_metric(f'train/Specificity-{fold}', step_metric=f'epoch_fold-{fold}')
    wandb.define_metric(f'train/F1-{fold}', step_metric=f'epoch_fold-{fold}')
    wandb.define_metric(f'test/AUC-{fold}', step_metric=f'epoch_fold-{fold}')
    wandb.define_metric(f'test/Accuracy-{fold}', step_metric=f'epoch_fold-{fold}')
    wandb.define_metric(f'test/Recall-{fold}', step_metric=f'epoch_fold-{fold}')
    wandb.define_metric(f'test/Specificity-{fold}', step_metric=f'epoch_fold-{fold}')
    wandb.define_metric(f'test/F1-{fold}', step_metric=f'epoch_fold-{fold}')


    for epoch in range(n_epochs):
        # train
        train_loss, train_perf = train_epoch(device, model, loader_train, criterion, optimizer, train_metrics)
        # validation
        val_loss, val_perf = evaluate_epoch(device, model, loader_val, criterion, val_metrics)
        # scheduler step
        scheduler.step(val_loss)
        # track losses & embeddings
        losses.append([train_loss, val_loss])
        #embeddings.append(epoch_embeddings)
        test_perf = test_epoch(device, model, loader_test, test_metrics)
        for m in perf_metrics.keys():
            perf_metrics[m].append([train_perf[m].detach().numpy().item(), val_perf[m].detach().numpy().item(), test_perf[m].detach().numpy().item()])
        # log performance and loss in wandb
        wandb.log({f'epoch_fold-{fold}': epoch,
                    f'val/loss-{fold}': val_loss,
                   f'train/loss-{fold}': train_loss,
                   f'val/Accuracy-{fold}': val_perf["Accuracy"].detach().numpy().item(),
                   f'val/AUC-{fold}': val_perf["AUC"].detach().numpy().item(),
                   f'val/Recall-{fold}': val_perf["Recall"].detach().numpy().item(),
                   f'val/Specificity-{fold}': val_perf["Specificity"].detach().numpy().item(),
                   f'val/F1-{fold}': val_perf["F1"].detach().numpy().item(),
                   f'train/Accuracy-{fold}': train_perf["Accuracy"].detach().numpy().item(),
                   f'train/AUC-{fold}': train_perf["AUC"].detach().numpy().item(),
                   f'train/Recall-{fold}': train_perf["Recall"].detach().numpy().item(),
                   f'train/Specificity-{fold}': train_perf["Specificity"].detach().numpy().item(),
                   f'train/F1-{fold}': train_perf["F1"].detach().numpy().item(),
                   f'test/AUC-{fold}': test_perf["AUC"].detach().numpy().item(),
                   f'test/Accuracy-{fold}': test_perf["AUC"].detach().numpy().item(),
                   f'test/Recall-{fold}': test_perf["Recall"].detach().numpy().item(),
                   f'test/Specificity-{fold}': test_perf["Specificity"].detach().numpy().item(),
                   f'test/F1-{fold}': test_perf["F1"].detach().numpy().item()
                   }) #, step=epoch)
        if epoch % 5 == 0:
            print(f"Epoch {epoch}",
                  f"Loss train {train_loss}",
                  f"Loss validation {val_loss}",
                  f"Acc train {train_perf}",
                  f"Acc validation {val_perf};")
        train_metrics.reset()
        val_metrics.reset()
        test_metrics.reset()

        # identify best model based on max validation AUC
        if epoch < 1:
            best_loss = losses[epoch][1]
            best_model = copy.deepcopy(model)
            best_epoch = epoch
        else:
            if best_loss < losses[epoch][1]:
                continue
            else:
                best_loss = losses[epoch][1]
                best_model = copy.deepcopy(model)
                best_epoch = epoch
    return losses, perf_metrics, best_epoch, best_loss, best_model # , embeddings


def embeddings_2pca(embeddings):
    """ Generates 3-dimensional pca from d-dimensional embeddings.
        Returns a pandas dataframe with the 3-d pc.
    """
    pca = PCA(n_components=3, random_state=42)
    pca_result = pca.fit_transform(embeddings)
    pca_df = pd.DataFrame()
    pca_df['pca-v1'] = pca_result[:, 0]
    pca_df['pca-v2'] = pca_result[:, 1]
    pca_df['pca-v3'] = pca_result[:, 2]
    print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))

    return pca_df



def k_fold(x, y, folds):
    skf = StratifiedKFold(folds, shuffle=True, random_state=12345)

    test_indices, train_indices = [], []
    test_mask, train_mask = [], []
    mask_array = torch.zeros(x.shape[0], dtype=torch.bool)
    for _, idx in skf.split(torch.zeros(x.shape[0]), y):
        test_indices.append(torch.from_numpy(idx).to(torch.long))
        mask_array = torch.zeros(x.shape[0], dtype=torch.bool)
        mask_array[test_indices[-1]] = True
        test_mask.append(mask_array)

    val_indices = [test_indices[i - 1] for i in range(folds)]
    val_mask = [test_mask[i - 1] for i in range(folds)]

    for i in range(folds):
        train_mask_indices = torch.ones(x.shape[0], dtype=torch.bool)
        train_mask_indices[test_indices[i]] = 0
        train_mask_indices[val_indices[i]] = 0
        train_indices.append(train_mask_indices.nonzero(as_tuple=False).view(-1))
        mask_array = torch.zeros(x.shape[0], dtype=torch.bool)
        mask_array[train_indices[-1]] = True
        train_mask.append(mask_array)

    #return train_indices, test_indices, val_indices
    return train_mask, test_mask, val_mask


def calculate_feature_importance(device, model, loader_train, nodes_mapping=None, top_k_nodes=124, top_k_edges=1891):
    """
    Explainability at node and edge level as per GNN-Explainer model from the “GNNExplainer: Generating Explanations for Graph Neural Networks” paper for identifying compact subgraph structures that play a crucial role in the predictions made by a GNN
    :param model:
    :param loader_train:
    :param names_list:
    :param save_fig:
    :param name_file:
    :param path:
    :param top_k_nodes:
    :param top_k_edges:
    :return: topk_nodes, topk_edges
    """
    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=100),
        explanation_type='model',
        node_mask_type='attributes',
        edge_mask_type='object',
        model_config=dict(
            mode='multiclass_classification',
            task_level='graph',
            return_type='log_probs',
        ),
    )
    all_explanations=[]
    for data in loader_train:
        data = data.to(device)
        # Calculate explanations for each graph in the batch
        if "_uw" in str(model.__class__.__name__):  # for unweighted models
            explanations = explainer(x=data.x, edge_index=data.edge_index, batch=data.batch)
        elif "GAT" in str(model.__class__.__name__):
            explanations = explainer(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, batch=data.batch)
        else:
            explanations = explainer(x=data.x, edge_index=data.edge_index, edge_weight=data.edge_attr, batch=data.batch)
        all_explanations.append(explanations)
    print(f'Generated explanations in {explanations.available_explanations}')

    # top k relevant nodes/edges (avg relevance across all individuals/graphs)
    nodes_relevance, edges_relevance, relevant_subgraph_edge_index = relevant_nodes_edges(all_explanations)
    topk_nodes = nodes_relevance.iloc[0:top_k_nodes,:]
    topk_edges = edges_relevance.iloc[0:top_k_edges,:]
    if nodes_mapping is not None:
        topk_nodes.loc[:, 'node_name'] = topk_nodes['node_index'].apply(lambda x: [key for key, value in nodes_mapping.items() if value == x][0])

        edge_mapping = {idx: (edge[0], edge[1]) for idx, edge in enumerate(relevant_subgraph_edge_index.t().tolist())}
        topk_edges.loc[:, 'source'] = topk_edges['edge_index'].apply(lambda edge_idx: edge_mapping[edge_idx][0])
        topk_edges.loc[:, 'destination'] = topk_edges['edge_index'].apply(lambda edge_idx: edge_mapping[edge_idx][1])
        topk_edges.loc[:, 'source'] = topk_edges['source'].apply(lambda x: [key for key, value in nodes_mapping.items() if value == x][0])
        topk_edges.loc[:, 'destination'] = topk_edges['destination'].apply(lambda x: [key for key, value in nodes_mapping.items() if value == x][0])

    return(topk_nodes, topk_edges)

def relevant_nodes_edges(explanations_list):
    """
    Aggregate node and edge importance scores from multiple graphs and calculate average importance per graph.

    Args:
        explanations_list (list): List of explanations for each graph.
    Returns:
        node_scores_df (pd.DataFrame): DataFrame with node importance scores.
        edge_scores_df (pd.DataFrame): DataFrame with edge importance scores.
    """
    all_node_masks=[]
    all_edge_masks=[]
    for explanations in explanations_list:
    # explanations contains multiple graphs (as many as 16)
        node_mask = explanations.node_mask
        edge_mask = explanations.edge_mask
        edge_index = explanations.edge_index
        batch = explanations.batch

        # Iterate through each unique graph in the batch
        unique_graphs = torch.unique(batch)
        for graph_id in unique_graphs:
            # Identify nodes belonging to the current graph
            nodes_in_graph = (batch == graph_id).nonzero().squeeze()
            # Identify edges belonging to the current graph
            edge_mask_in_graph = (edge_index[0] >= nodes_in_graph[0]) & (edge_index[0] < nodes_in_graph[-1])
            edge_index_in_graph = edge_index[:, edge_mask_in_graph]
            relevant_subgraph_edge_index = edge_index_in_graph - nodes_in_graph[0]
            # store node_masks and edge_masks for each graph
            all_node_masks.append(node_mask[nodes_in_graph])
            all_edge_masks.append(edge_mask[edge_mask_in_graph])

    # Aggregate node importance scores and average by all graphs (to get avg node importance per graph)
    node_importance_scores = torch.sum(torch.stack(all_node_masks), dim=0)
    node_importance_scores = node_importance_scores/len(all_node_masks)
    sorted_node_indices = torch.argsort(node_importance_scores.view(-1), descending=True)
    node_scores_df = pd.DataFrame({'node_index': sorted_node_indices.cpu().numpy(),
                                'node_avg_mask_score': node_importance_scores[sorted_node_indices].cpu().numpy()[:,0]})
    #node_scores_df.iloc[0:top_k_nodes,:]

    # Aggregate edge importance scores
    edge_importance_scores = torch.sum(torch.stack(all_edge_masks), dim=0)
    edge_importance_scores = edge_importance_scores/len(all_edge_masks)
    sorted_edge_indices = torch.argsort(edge_importance_scores.view(-1), descending=True)
    edge_scores_df = pd.DataFrame({'edge_index': sorted_edge_indices.cpu().numpy(),
                                'edge_avg_mask_score': edge_importance_scores[sorted_edge_indices].cpu().numpy()})
    #edge_scores_df.iloc[0:top_k_edges,:]

    return(node_scores_df, edge_scores_df, relevant_subgraph_edge_index)


def quantile_map(value, quantiles):
    """
    Mapping function that assigns colors based on quantiles
    """
    for i in range(len(quantiles) - 1):
        if quantiles[i] <= value <= quantiles[i + 1]:
            return i / (len(quantiles) - 1)
    return 1.0  # For values outside the quantile range

def display_graph_individual(adj_df, labels_dict=None, save_fig=False, path="./", name_file="graph.png", plot_title=None):
    """Draw the graph given an adjacency matrix"""
    fig, ax = plt.subplots(figsize=(14, 14))
    G = nx.from_pandas_adjacency(adj_df)
    weights = nx.get_edge_attributes(G, 'weight').values()
    pos = nx.spring_layout(G, seed=12)
    if labels_dict is None:
        # Default node color if labels_dict is not provided
        node_colors = "blue"
    else:
        # Define the number of quantiles and quantile boundaries
        label_values = np.array(list(labels_dict.values()))
        label_values = np.log1p(label_values)
        num_quantiles = 20
        quantiles = np.percentile(label_values, np.linspace(0, 100, num_quantiles + 1))
        colormap = plt.get_cmap("viridis")
        mapped_values = [quantile_map(val, quantiles) for val in label_values]
        norm = Normalize(vmin=0, vmax=1) # Normalize the mapped values
        node_colors = [colormap(norm(val)) for val in mapped_values]

        # Draw the graph
        nx.draw(G, pos=pos, with_labels=False,
                node_color=node_colors, node_size=80,
                width=list(weights), ax=ax)

        # Color bar as a legend
        sm = ScalarMappable(cmap=colormap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, orientation="horizontal", shrink=0.25)
        cbar.set_label('Normalized log-transformed expression',  fontsize=16)
        cbar.set_ticks([])
        #cbar.ax.tick_params(labelsize=14)

    plt.title(plot_title, fontsize=24)
    plt.tight_layout()
    plt.show()
    if save_fig:
        fig.savefig(path + name_file)
    # # Log the image to wandb: Convert the graph image to a PIL Image
    # image = Image.frombytes('RGB', fig.canvas.get_width_height(),
    #                         fig.canvas.tostring_rgb())
    # wandb.log({graph: wandb.Image(image), "caption": "Graph Visualization"})
    plt.close(fig)


def cv_metrics_to_wandb(dict_val_metrics, dict_test_metrics, wandb):
    for key in dict_val_metrics.keys():
        val_values = dict_val_metrics[key]
        mean_val = np.mean(val_values)
        std_val = np.std(val_values)
        wandb.run.summary[f"mean_val_{key}"] = mean_val
        wandb.run.summary[f"std_val_{key}"] = std_val
        wandb.run.summary[f"values_val_{key}"] = np.array(val_values)
        wandb.log({f"mean_val_{key}": mean_val, f"std_val_{key}": std_val}, commit=False)
        if key in dict_test_metrics.keys():
            test_values = dict_test_metrics[key]
            mean_test = np.mean(test_values)
            std_test = np.std(test_values)
            wandb.run.summary[f"mean_test_{key}"] = mean_test
            wandb.run.summary[f"std_test_{key}"] = std_test
            wandb.run.summary[f"values_test_{key}"] = np.array(test_values)
            wandb.log({f"mean_test_{key}": mean_test, f"std_test_{key}": std_test}, commit=False)

def generate_model(model_name, config, n_features):
    models_dict = {"GCNN_3p_uw": GCNN_3p_uw(n_features, config.cl1_hidden_units, config.cl2_hidden_units, config.ll_out_units, config.dropout),
                   "GCNN_3p": GCNN_3p(n_features, config.cl1_hidden_units, config.cl2_hidden_units, config.ll_out_units, config.dropout),
                   "Cheb_GCNN_3p_uw": Cheb_GCNN_3p_uw(n_features, config.cl1_hidden_units, config.cl2_hidden_units, config.K_cheby, config.ll_out_units, config.dropout),
                   "Cheb_GCNN_3p": Cheb_GCNN_3p(n_features, config.cl1_hidden_units, config.cl2_hidden_units, config.K_cheby, config.ll_out_units, config.dropout),
                   "GAT_3p": GAT_3p(n_features, config.cl1_hidden_units, config.cl2_hidden_units, config.heads, config.ll_out_units, config.dropout),
                   "GAT_3p_uw": GAT_3p_uw(n_features, config.cl1_hidden_units, config.cl2_hidden_units, config.heads, config.ll_out_units, config.dropout)
    }
    # "MLP2": MLP2(n_features, config.cl1_hidden_units, config.cl2_hidden_units, config.ll_out_units, config.dropout),
    model = models_dict[model_name]
    print(model)
    return model

def update_overall_metrics(fold, fold_best_epoch, top_nodes, top_edges, fold_performance, fold_losses, dict_val_metrics, dict_test_metrics, features_track):
    dict_val_metrics["Fold"].append(fold)
    dict_val_metrics["N_epoch"].append(fold_best_epoch)
    dict_test_metrics["Fold"].append(fold)
    dict_test_metrics["N_epoch"].append(fold_best_epoch)
    features_track["Fold"].append(fold)
    features_track["Relevant_Nodes"].append(top_nodes["node_name"].tolist())
    features_track["Relevant_Edges_source"].append(top_edges["source"].tolist())
    features_track["Relevant_Edges_dest"].append(top_edges["destination"].tolist())
    for m in fold_performance.keys():
        dict_val_metrics[m].append(fold_performance[m][fold_best_epoch][1])
        dict_test_metrics[m].append(fold_performance[m][fold_best_epoch][2])
    dict_val_metrics["Loss"].append(fold_losses[fold_best_epoch][1])
    return (dict_val_metrics, dict_test_metrics, features_track)



def display_explanation_subgraph(nodes, edges, G, labels_dict=None, save_fig=False, path="./", name_file="explanation_graph.png", plot_title=None):
    """
    Highlight relevant nodes and edges in graph
    """
    # TODO!



def relevant_nodes_edges_subgraphs(explanations_list):
    explanatory_subgraphs=[] # list of explanatory subgraphs
    explanatory_subgraphs_nx=[] # list of explanatory subgraphs in nx format
    all_node_masks=[]
    all_edge_masks=[]
    i=0
    for explanations in explanations_list:
    # explanations contains multiple graphs (as many as 16)
        node_mask = explanations.node_mask
        edge_mask = explanations.edge_mask
        edge_index = explanations.edge_index
        batch = explanations.batch

        # Iterate through each unique graph in the batch
        unique_graphs = torch.unique(batch)
        for graph_id in unique_graphs:
            # Identify nodes belonging to the current graph
            nodes_in_graph = (batch == graph_id).nonzero().squeeze()
            # Identify edges belonging to the current graph
            edge_mask_in_graph = (edge_index[0] >= nodes_in_graph[0]) & (edge_index[0] < nodes_in_graph[-1])
            edge_index_in_graph = edge_index[:, edge_mask_in_graph]

            # Create the relevant subgraph for the current graph
            relevant_subgraph = Data(
                x=node_mask[nodes_in_graph],
                edge_index=edge_index_in_graph - nodes_in_graph[0],  # Adjust indices
                edge_attr=edge_mask[edge_mask_in_graph],
                # Add other relevant attributes if needed
            )

            ##### test
            out = copy.copy(relevant_subgraph)
            test_edgemask = edge_mask[edge_mask_in_graph] > 0
            test_nodemask= node_mask[nodes_in_graph].sum(dim=-1) > 0
            if test_edgemask is not None:
                for key, value in relevant_subgraph.items():
                    if key == 'edge_index':
                        out.edge_index = value[:, test_edgemask]
                    elif relevant_subgraph.is_edge_attr(key):
                        out[key] = value[test_edgemask]
            if test_nodemask is not None:
                out = out.subgraph(test_nodemask)

            nx_G = pyg_utils.to_networkx(relevant_subgraph)
            explanatory_subgraphs_nx.append(nx_G)
            explanatory_subgraphs.append(relevant_subgraph)
            all_node_masks.append(node_mask[nodes_in_graph]) # store all node_masks
            all_edge_masks.append(edge_mask[edge_mask_in_graph]) # store all edge_masks
            i+=1
            print(i)

    # Aggregate node importance scores and average by all graphs (so we get an average importance per graph)
    node_importance_scores = torch.sum(torch.stack(all_node_masks), dim=0)
    node_importance_scores = node_importance_scores/len(all_node_masks)
    # create sorted df with index, score
    sorted_node_indices = torch.argsort(node_importance_scores.view(-1), descending=True)
    node_scores_df = pd.DataFrame({'node_index': sorted_node_indices.numpy(),
                                'node_avg_mask_score': node_importance_scores[sorted_node_indices].numpy()[:,0]})
    #node_scores_df.iloc[0:top_k_nodes,:]

    # Aggregate edge importance scores
    edge_importance_scores = torch.sum(torch.stack(all_edge_masks), dim=0)
    edge_importance_scores = edge_importance_scores/len(all_edge_masks)
    # create sorted df with index, score
    sorted_edge_indices = torch.argsort(edge_importance_scores.view(-1), descending=True)
    edge_scores_df = pd.DataFrame({'edge_index': sorted_edge_indices.numpy(),
                                'edge_avg_mask_score': edge_importance_scores[sorted_edge_indices].numpy()})
    #edge_scores_df.iloc[0:top_k_edges,:]

    return(node_scores_df, edge_scores_df, explanatory_subgraphs_nx, explanatory_subgraphs)



def training_nowandb(device, model, optimizer, scheduler, criterion, loader_train, loader_val, loader_test, n_epochs, fold):
    """ Full training process
    """
    losses = []
    #embeddings = []
    perf_metrics = {'Accuracy': [], 'AUC': [], 'Recall': [], 'Specificity': [], 'F1': []}
    train_metrics = MetricCollection({
        'Accuracy': Accuracy(task="binary"),
        'AUC': AUROC(task="binary", num_classes=2),
        'Recall': Recall(task="binary", num_classes=2),
        'Specificity': Specificity(task="binary", num_classes=2),
        'F1': F1Score(task="binary", num_classes=2),
    })
    val_metrics = MetricCollection({
        'Accuracy': Accuracy(task="binary"),
        'AUC': AUROC(task="binary", num_classes=2),
        'Recall': Recall(task="binary", num_classes=2),
        'Specificity': Specificity(task="binary", num_classes=2),
        'F1': F1Score(task="binary", num_classes=2),
    })
    test_metrics = MetricCollection({
                'Accuracy': Accuracy(task="binary"),
                'AUC': AUROC(task="binary", num_classes=2),
                'Recall': Recall(task="binary", num_classes=2),
                'Specificity': Specificity(task="binary", num_classes=2),
                'F1': F1Score(task="binary", num_classes=2),
    })
    for epoch in range(n_epochs):
        # train
        train_loss, train_perf = train_epoch(device, model, loader_train, criterion, optimizer, train_metrics)
        # validation
        val_loss, val_perf = evaluate_epoch(device, model, loader_val, criterion, val_metrics)
        # scheduler step
        scheduler.step(val_loss)
        # track losses & embeddings
        losses.append([train_loss, val_loss])
        #embeddings.append(epoch_embeddings)
        test_perf = test_epoch(device, model, loader_test, test_metrics)
        for m in perf_metrics.keys():
            perf_metrics[m].append([train_perf[m].detach().numpy().item(), val_perf[m].detach().numpy().item(), test_perf[m].detach().numpy().item()])

        if epoch % 5 == 0:
            print(f"Epoch {epoch}",
                  f"Loss train {train_loss}",
                  f"Loss validation {val_loss}",
                  f"Acc train {train_perf}",
                  f"Acc validation {val_perf};")
        train_metrics.reset()
        val_metrics.reset()
        test_metrics.reset()

        # identify best model based on max validation AUC
        if epoch < 1:
            best_loss = losses[epoch][1]
            best_model = copy.deepcopy(model)
            best_epoch = epoch
        else:
            if best_loss < losses[epoch][1]:
                continue
            else:
                best_loss = losses[epoch][1]
                best_model = copy.deepcopy(model)
                best_epoch = epoch
    return losses, perf_metrics, best_epoch, best_loss, best_model # , embeddings


def plot_connected_graph(G_connected, annotation=None, d=10, fig_save=True, labels=True, path=None, plotname="cv_relevantedges"):
# plot largest connected component of relevant edges with labels for high degree nodes
    degrees = dict(G_connected.degree())
    edge_widths = [next(iter(data.values())) for _, _, data in G_connected.edges(data=True)] # retrieve the edge attribute

    plt.figure(figsize=(7, 7))
    pos = nx.spring_layout(G_connected, k=0.05, seed=16)
    nx.draw(G_connected, pos, with_labels=False, node_size=40, node_color='tab:blue', edge_color='gray', width=edge_widths, edge_cmap=plt.cm.Blues)
    if labels:
        selected_nodes = [node for node, degree in degrees.items() if degree >= d]
        G_selected = G_connected.subgraph(selected_nodes)
        edge_widths = [next(iter(data.values())) for _, _, data in G_selected.edges(data=True)] # retrieve the edge attribute
        nx.draw(G_selected, pos, with_labels=False, node_size=60, node_color='orange', edge_color='black', width=edge_widths)
        # Draw the labels and arrows for selected nodes
        if annotation is not None:
            node_labels = {node: annotation.loc[annotation['ensembl_gene_id'] == node, 'wikigene_name'].values[0] if node in selected_nodes else '' for node in G_connected.nodes}
        else:
            node_labels = {node: node if node in selected_nodes else '' for node in G_connected.nodes}
        for node, label in node_labels.items():
            if node in selected_nodes:
                xy = pos[node]
                xytext = (xy[0] +0.3 , xy[1] +0.01)  # Adjust the x,y-coordinate of the label
                plt.annotate(label, xy=xy, xytext=xytext, arrowprops=dict(arrowstyle='->', connectionstyle="arc3,rad=0.1", color='crimson'), fontsize=12, fontname='Arial', bbox=dict(boxstyle="round,pad=0.3", alpha=0.4), fontweight='bold') #, fontweight='bold'
    if fig_save:
        plt.savefig(path + "/" + plotname + ".png", bbox_inches='tight', dpi=300)
        plt.savefig(path + "/" + plotname + ".pdf", bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()
        
def plot_largest_connected_components(G, k=0.2, labels=True, min_nodes=4, d=None, annotation=None, fig_save=True, path=".", plotname="cv_overlappingedges_components"):
    components = [comp for comp in nx.connected_components(G) if len(comp) >= min_nodes]

    # Generate a layout for the full graph
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, k=k, seed=13)

    for i, component in enumerate(components):
        G_subgraph = G.subgraph(component)
        color = plt.cm.tab10(i % 10) # Generate a unique color for each subgraph
        edge_widths = [next(iter(data.values())) for _, _, data in G_subgraph.edges(data=True)] 
        nx.draw(G_subgraph, pos, with_labels=False, node_size=200, node_color=[color] * len(G_subgraph.nodes), edge_color='gray', width=edge_widths, edge_cmap=plt.cm.Blues)
        if labels:
            degrees = dict(G_subgraph.degree())
            if d is not None:
                selected_nodes = [node for node, degree in degrees.items() if degree >= d]
            else:
                selected_nodes = [node for node, degree in degrees.items()]
            if annotation is not None:
                node_labels = {node: annotation.loc[annotation['ensembl_gene_id'] == node, 'wikigene_name'].values[0] if node in selected_nodes else '' for node in G_subgraph.nodes}
            else:
                node_labels = {node: node if node in selected_nodes else '' for node in G_subgraph.nodes}
            for node, label in node_labels.items():
                xy = pos[node]
                xytext = (xy[0] - 0.04, xy[1] + 0.03)  # Adjust the y-coordinate of the label
                plt.annotate(label, xy=xy, xytext=xytext, fontsize=14, fontname='Arial', bbox=dict(boxstyle="round,pad=0.3", alpha=0.2), fontweight='bold')
    plt.title(f'Largest connected components (at least {min_nodes} nodes) occurring at least twice across the 10-fold CV')
    plt.tight_layout()  # Adjust layout for better spacing
    if fig_save:
        plt.savefig(path + "/" + plotname + ".png", bbox_inches='tight', dpi=300)
        plt.savefig(path + "/" + plotname + ".pdf", bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()
    





def plot_largest_connected_components2(G, labels=True, min_nodes=4, d=None, annotation=None, fig_save=True, path=None, plotname="cv_overlappingedges_components"):
    components = [comp for comp in nx.connected_components(G) if len(comp) >= min_nodes]

    # Generate a layout for the full graph
    pos = nx.layout.kamada_kawai_layout(G)
    all_subgraphs = []  # to store all subgraphs for combined plot
    num_subplots = len(components)
    num_rows = math.ceil(math.sqrt(num_subplots))
    num_cols = math.ceil(num_subplots / num_rows)

    for i, component in enumerate(components):
        plt.figure(figsize=(10, 8))
        G_subgraph = G.subgraph(component)
        all_subgraphs.append(G_subgraph)  # store subgraph for combined plot
        color = plt.cm.tab10(i % 10) # Generate a unique color for each subgraph
        edge_widths = [next(iter(data.values())) for _, _, data in G_subgraph.edges(data=True)] 
        nx.draw(G_subgraph, pos, with_labels=False, node_size=200, node_color=[color] * len(G_subgraph.nodes), edge_color='gray', width=edge_widths, edge_cmap=plt.cm.Blues)
        if labels:
            degrees = dict(G_subgraph.degree())
            if d is not None:
                selected_nodes = [node for node, degree in degrees.items() if degree >= d]
            else:
                selected_nodes = [node for node, degree in degrees.items()]
            if annotation is not None:
                node_labels = {node: annotation.loc[annotation['ensembl_gene_id'] == node, 'wikigene_name'].values[0] if node in selected_nodes else '' for node in G_subgraph.nodes}
            else:
                node_labels = {node: node if node in selected_nodes else '' for node in G_subgraph.nodes}
            for node, label in node_labels.items():
                xy = pos[node]
                xytext = (xy[0], xy[1] + 0.03)  # Adjust the y-coordinate of the label
                plt.annotate(label, xy=xy, xytext=xytext, fontsize=14, fontname='Arial', bbox=dict(boxstyle="round,pad=0.3", alpha=0.2), fontweight='bold')
        plt.tight_layout()  # Adjust layout for better spacing
        if fig_save:
            # Save each subgraph separately
            subgraph_name = f"{plotname}_subgraph_{i + 1}"
            subgraph_path = os.path.join(path, subgraph_name)
            plt.savefig(subgraph_path + ".png", bbox_inches='tight', dpi=300)
            plt.savefig(subgraph_path + ".pdf", bbox_inches='tight', dpi=300)
            plt.close()
        else:
            plt.show()
        

    # Create a combined plot with all subgraphs
    plt.figure(figsize=(15, 12))
    d=2
    for i, G_subgraph in enumerate(all_subgraphs):
        plt.subplot(num_rows, num_cols, i + 1)
        color = plt.cm.tab10(i % 10)
        nx.draw(G_subgraph, pos, with_labels=False, node_size=200, node_color=[color] * len(G_subgraph.nodes), edge_color='gray', width=edge_widths, edge_cmap=plt.cm.Blues)
        if labels:
            degrees = dict(G_subgraph.degree())
            if d is not None:
                selected_nodes = [node for node, degree in degrees.items() if degree >= d]
            else:
                selected_nodes = [node for node, degree in degrees.items()]
            if annotation is not None:
                node_labels = {node: annotation.loc[annotation['ensembl_gene_id'] == node, 'wikigene_name'].values[0] if node in selected_nodes else '' for node in G_subgraph.nodes}
            else:
                node_labels = {node: node if node in selected_nodes else '' for node in G_subgraph.nodes}
            for node, label in node_labels.items():
                xy = pos[node]
                xytext = (xy[0] - 0.03, xy[1] + 0.02)  # Adjust the y-coordinate of the label
                plt.annotate(label, xy=xy, xytext=xytext, fontsize=12, fontname='Arial', bbox=dict(boxstyle="round,pad=0.3", alpha=0.2), fontweight='bold')
    plt.tight_layout()
    if fig_save:
        # Save each subgraph separately
        subgraph_name = f"{plotname}_subgraph_{i + 1}"
        subgraph_path = os.path.join(path, subgraph_name)
        plt.savefig(os.path.join(path, f"{plotname}_subgraphs_combined.png"), bbox_inches='tight', dpi=300)
        plt.savefig(os.path.join(path, f"{plotname}_subgraphs_combined.pdf"), bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()


