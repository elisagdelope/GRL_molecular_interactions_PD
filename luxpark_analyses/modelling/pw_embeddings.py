import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split



def broadcast(src, other, dim):
    # Source: torch_scatter
    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:
        for _ in range(0, dim):
            src = src.unsqueeze(0)
    for _ in range(src.dim(), other.dim()):
        src = src.unsqueeze(-1)
    src = src.expand_as(other)
    return src


class SparseMaskedLinear_v2(nn.Module):
    """ Masked linear layer with sparse mask AND sparse weight matrix (faster and more memory efficient) """
    def __init__(self, in_features, out_features, sparse_mask, bias=True, device=None, dtype=None):
        """
        in_features: number of input features
        out_features: number of output features
        sparse_mask: torch tensor of shape (n_connections, 2), where indices[:, 0] index the input neurons
                     and indices[:, 1] index the output neurons
        """
        # Reference: https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/linear.py
        self.sparse_mask = sparse_mask
        self.sparse_mask = sparse_mask
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.nn.init.normal_(torch.empty((sparse_mask.shape[0]), **factory_kwargs)))  # Shape=(n_connections,)
        self.use_bias = bias
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, **factory_kwargs))
    def forward(self, input):
        # weight shape: (out_features, in_features)
        x = input[:, self.sparse_mask[:, 0]]  # Shape=(batch_size, n_connections)
        src = x * self.weight[None, :]  # Shape=(batch_size, n_connections)
        # Reduce via scatter sum
        out = torch.zeros((x.shape[0], self.out_features), dtype=x.dtype, device=x.device)
        index = broadcast(self.sparse_mask[:, 1], src, dim=-1)
        out = out.scatter_add_(dim=-1, index=index, src=src)
        if self.use_bias:
            out = out + self.bias
        return out


class pw_linear(nn.Module):
    def __init__(self, sparse_mask, in_mask_f, out_mask_f, h_f, out_f):
        super(pw_linear, self).__init__()
        torch.manual_seed(42)
        self.sparse_masked = SparseMaskedLinear_v2(in_mask_f, out_mask_f, sparse_mask, device=device)
        self.lin1 = nn.Linear(out_mask_f, h_f)
        self.lin2 = nn.Linear(h_f, out_f)
        self.bn1 = nn.BatchNorm1d(out_mask_f)
        self.bn2 = nn.BatchNorm1d(h_f)
    def forward(self, data):
        #x = data.x
        x = data
        x = F.relu(self.sparse_masked(x))
        x = self.bn1(x)
        x = F.dropout(x, p=0.4, training=self.training)
        x = F.relu(self.lin1(x))
        x = self.bn2(x)
        x = F.dropout(x, p=0.2, training=self.training)
        out = self.lin2(x)
        return x, out

class mlp(nn.Module):
    def __init__(self, in_f, h_f, out_f):
        super(mlp, self).__init__()
        torch.manual_seed(42)
        self.lin1 = nn.Linear(in_f, h_f)
        self.lin2 = nn.Linear(h_f, out_f)
        self.bn1 = nn.BatchNorm1d(h_f)
    def forward(self, data):
        #x = data.x
        x = data
        x = F.relu(self.lin1(x))
        x = self.bn1(x)
        x = F.dropout(x, p=0.2, training=self.training)
        out = self.lin2(x)
        return x, out


class pw_Cheb_3p_uw(nn.Module):
    def __init__(self, sparse_mask, in_mask_f, out_mask_f, CL1_F, CL2_F, K, out_f):
        super(pw_Cheb_3p_uw, self).__init__()
        torch.manual_seed(42)
        self.sparse_masked = SparseMaskedLinear_v2(in_mask_f, out_mask_f, sparse_mask, device=device)
        # graph CL1
        self.conv1 = ChebConv(in_channels=out_mask_f, out_channels=CL1_F, K=K)
        self.bn1 = nn.BatchNorm1d(CL1_F)
        # graph CL2
        self.conv2 = ChebConv(in_channels=CL1_F, out_channels=CL2_F, K=K)
        self.bn2 = nn.BatchNorm1d(CL2_F)
        # FC1
        self.lin1 = nn.Linear(CL2_F * 3, out_f)
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        # pw mask:
        x = self.sparse_masked(x)
        # node embeddings:
        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        #x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        # graph embedding: pooling at graph level (sum over all nodes embeddings)
        batch = torch.zeros(data.x.shape[0], dtype=int) if data.batch is None else data.batch
        # x = global_add_pool(x, data.batch)
        x0 = global_add_pool(x, data.batch)
        x1 = global_mean_pool(x, data.batch)
        x2 = global_max_pool(x, data.batch)
        x = torch.cat([x0, x1, x2], dim=-1)
        out = self.lin1(x)
        return x, out  # returns the embedding x & prediction out



from utils_bkp import *
from models_bkp import *
from plot_utils import training_plots, pca_plot2d, pca_plot3d
from torch.optim import lr_scheduler

if __name__ == '__main__':
    # I/O
    INPUT_DIR = "../data/"
    OUTPUT_DIR = '../results/'
    in_cv_file = "data_cv_metab_4ML_DIAGNOSIS.tsv"
    in_test_file = "data_test_metab_4ML_DIAGNOSIS.tsv"
    out_file = "data_metab_4ML_DIAGNOSIS.csv"
    annotation_file = "chemical_annotation.tsv"
    labels_file = "labels.csv"

    # Reading the files
    df_cv = pd.read_table(INPUT_DIR + in_cv_file, index_col=0)
    df_test = pd.read_table(INPUT_DIR + in_test_file, index_col=0)
    pw = pd.read_table(INPUT_DIR + annotation_file)
    labels = pd.read_csv(INPUT_DIR + labels_file)

    # transform data
#    df = pd.concat([df_cv, df_test], axis=0)
    y_train = np.array(df_cv['DIAGNOSIS'])
    x_train = np.array(df_cv.drop(['DIAGNOSIS'], axis=1))
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)

    y_test = np.array(df_test['DIAGNOSIS'])
    x_test = np.array(df_test.drop(['DIAGNOSIS'], axis=1))
    x_test = scaler.transform(x_test)

    # Split data into training and test sets
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.15, stratify=y_train, random_state=42)
    x_train, x_val = torch.tensor(x_train).float(), torch.tensor(x_val).float()
    y_train, y_val = torch.tensor(y_train), torch.tensor(y_val)
    from torch.utils.data import TensorDataset, DataLoader
    train_dataset = TensorDataset(x_train, y_train)
    val_dataset = TensorDataset(x_val, y_val)

    # create mask (each molecule belongs to one or none pathway)
    pw = pw[["SUB_PATHWAY", "ANALYSIS_ID"]]
    pw = pw[pw['ANALYSIS_ID'].isin(df_cv.columns)]
    pw.fillna(value='None', inplace=True)
    pw_mask = pw.pivot_table(index='ANALYSIS_ID', columns='SUB_PATHWAY', aggfunc='size', fill_value=0).astype(int)

    # generate embeddings
    in_features = pw_mask.shape[0]
    out_features = pw_mask.shape[1]
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    sparse_mask = torch.tensor(np.array(pw_mask)).nonzero()
    sparse_layer = SparseMaskedLinear_v2(in_features, out_features, sparse_mask)

    # Forward pass
    #y_sparse = sparse_layer(x)
    model = pw_linear(32, 2, sparse_layer)
    model = model.to(device)
    n_epochs = 50
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.05, betas=(0.9, 0.95))
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.9999999, patience=10, verbose=True)
    criterion = nn.CrossEntropyLoss()
    model = model.to(device)
    losses, performance = training_mlp(device, model, optimizer, scheduler, criterion, data, n_epochs)


    # Print the shapes of each set
    print("Training set shape:", train_data.shape)
    print("Validation set shape:", val_data.shape)
    print("Test set shape:", test_data.shape)