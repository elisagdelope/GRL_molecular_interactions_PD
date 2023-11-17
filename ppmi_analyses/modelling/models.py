import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, ChebConv, GATConv, GATv2Conv
from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool
import torch.nn.functional as F


class Cheb_GCNN_3p(nn.Module):
    def __init__(self, in_f, CL1_F, CL2_F, K, out_f, p_dropout): #, DL1_F, DL2_F
        super(Cheb_GCNN_3p, self).__init__()
        # graph CL1
        self.conv1 = ChebConv(in_channels=in_f, out_channels=CL1_F, K=K)
        # graph CL2
        self.conv2 = ChebConv(in_channels=CL1_F, out_channels=CL2_F, K=K)
        # FC1
        self.lin1 = nn.Linear(CL2_F * 3, out_f)
        self.bn1 = nn.BatchNorm1d(CL1_F)
        self.p_dropout = p_dropout      

    def forward(self, x, edge_index, edge_weight, batch):
        #x, edge_index = data.x, data.edge_index
        # node embeddings:
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = self.bn1(x)
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index, edge_weight))
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        # graph embedding: pooling at graph level (sum over all nodes embeddings)
        # x = global_add_pool(x, data.batch)
        x0 = global_add_pool(x, batch)
        x1 = global_mean_pool(x, batch)
        x2 = global_max_pool(x, batch)
        x = torch.cat([x0, x1, x2], dim=-1)
        out = self.lin1(x)
        return out  # returns the embedding x & prediction out


class Cheb_GCNN_3p_uw(nn.Module):
    def __init__(self, in_f, CL1_F, CL2_F, K, out_f, p_dropout): #, DL1_F, DL2_F
        super(Cheb_GCNN_3p_uw, self).__init__()
        # graph CL1
        self.conv1 = ChebConv(in_channels=in_f, out_channels=CL1_F, K=K)
        # graph CL2
        self.conv2 = ChebConv(in_channels=CL1_F, out_channels=CL2_F, K=K)
        # FC1
        self.lin1 = nn.Linear(CL2_F * 3, out_f)
        self.bn1 = nn.BatchNorm1d(CL1_F)
        self.p_dropout = p_dropout      

    def forward(self, x, edge_index, batch):
        #x, edge_index = data.x, data.edge_index
        # node embeddings:
        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        # graph embedding: pooling at graph level (sum over all nodes embeddings)
        # x = global_add_pool(x, data.batch)
        x0 = global_add_pool(x, batch)
        x1 = global_mean_pool(x, batch)
        x2 = global_max_pool(x, batch)
        x = torch.cat([x0, x1, x2], dim=-1)
        out = self.lin1(x)
        return out  # returns the embedding x & prediction out

class GAT_3p_uw(nn.Module):
    def __init__(self, in_f, CL1_F, CL2_F, heads, out_f, p_dropout):
        super(GAT_3p_uw, self).__init__()
        # graph CL1
        self.gat1 = GATv2Conv(in_channels=in_f, out_channels=CL1_F, heads=heads)
        # graph CL2
        self.gat2 = GATv2Conv(in_channels=CL1_F * heads, out_channels=CL2_F, heads=1, concat=False)
        # FC1
        self.lin1 = nn.Linear(CL2_F * 3, out_f)
        self.bn1 = nn.BatchNorm1d(CL1_F * heads)
        self.bn2 = nn.BatchNorm1d(CL2_F)
        self.p_dropout = p_dropout

    def forward(self, x, edge_index, batch):
        #x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        # node embeddings:
        x = F.relu(self.gat1(x, edge_index))
        x = self.bn1(x)
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        x = F.relu(self.gat2(x, edge_index))
        x = self.bn2(x)
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        # graph embedding: pooling at graph level (sum over all nodes embeddings)
        # x = global_add_pool(x, data.batch)
        x0 = global_add_pool(x, batch)
        x1 = global_mean_pool(x, batch)
        x2 = global_max_pool(x, batch)
        x = torch.cat([x0, x1, x2], dim=-1)
        out = self.lin1(x)
        return out # returns the embedding x & prediction out


class GAT_3p(nn.Module):
    def __init__(self, in_f, CL1_F, CL2_F, heads, out_f, p_dropout):
        super(GAT_3p, self).__init__()
        # graph CL1
        self.gat1 = GATv2Conv(in_channels=in_f, out_channels=CL1_F, heads=heads, edge_dim=1)
        # graph CL2
        self.gat2 = GATv2Conv(in_channels=CL1_F * heads, out_channels=CL2_F, heads=1, concat=False, edge_dim=1)
        # FC1
        self.lin1 = nn.Linear(CL2_F * 3, out_f)
        self.bn1 = nn.BatchNorm1d(CL1_F * heads)
        self.bn2 = nn.BatchNorm1d(CL2_F)
        self.p_dropout = p_dropout

    def forward(self, x, edge_index, edge_attr, batch):
        #x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        # node embeddings:
        x = F.relu(self.gat1(x, edge_index, edge_attr))
        x = self.bn1(x)
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        x = F.relu(self.gat2(x, edge_index, edge_attr))
        x = self.bn2(x)
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        # graph embedding: pooling at graph level (sum over all nodes embeddings)
        # x = global_add_pool(x, data.batch)
        x0 = global_add_pool(x, batch)
        x1 = global_mean_pool(x, batch)
        x2 = global_max_pool(x, batch)
        x = torch.cat([x0, x1, x2], dim=-1)
        out = self.lin1(x)
        return out # returns the embedding x & prediction out

class GCNN_3p_uw(nn.Module):
    def __init__(self, in_f, CL1_F, CL2_F, out_f, p_dropout):
        super(GCNN_3p_uw, self).__init__()
        # graph CL1
        self.conv1 = GCNConv(in_channels=in_f, out_channels=CL1_F)
        # graph CL2
        self.conv2 = GCNConv(in_channels=CL1_F, out_channels=CL2_F)
        # FC1
        self.lin1 = nn.Linear(CL2_F * 3, out_f)
        self.bn1 = nn.BatchNorm1d(CL1_F)
        self.bn2 = nn.BatchNorm1d(CL2_F)
        self.p_dropout = p_dropout  
        
    def forward(self, x, edge_index, batch):
        #x, edge_index = data.x, data.edge_index
        # node embeddings:
        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        # graph embedding: pooling at graph level (sum over all nodes embeddings)
        # x = global_add_pool(x, data.batch)
        x0 = global_add_pool(x, batch)
        x1 = global_mean_pool(x, batch)
        x2 = global_max_pool(x, batch)
        x = torch.cat([x0, x1, x2], dim=-1)
        out = self.lin1(x)
        return out


class GCNN_3p(nn.Module):
    def __init__(self, in_f, CL1_F, CL2_F, out_f, p_dropout):
        super(GCNN_3p, self).__init__()
        # graph CL1
        self.conv1 = GCNConv(in_channels=in_f, out_channels=CL1_F)
        # graph CL2
        self.conv2 = GCNConv(in_channels=CL1_F, out_channels=CL2_F)
        # FC1
        self.lin1 = nn.Linear(CL2_F * 3, out_f)
        self.bn1 = nn.BatchNorm1d(CL1_F)
        self.bn2 = nn.BatchNorm1d(CL2_F)
        self.p_dropout = p_dropout  
        
    def forward(self, x, edge_index, edge_weight, batch):
        #x, edge_index = data.x, data.edge_index
        # node embeddings:
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = self.bn1(x)
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index, edge_weight))
        x = self.bn2(x)
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        # graph embedding: pooling at graph level (sum over all nodes embeddings)
        # x = global_add_pool(x, data.batch)
        x0 = global_add_pool(x, batch)
        x1 = global_mean_pool(x, batch)
        x2 = global_max_pool(x, batch)
        x = torch.cat([x0, x1, x2], dim=-1)
        out = self.lin1(x)
        return out

class GCNN(nn.Module):
    def __init__(self, in_f, CL1_F, CL2_F, out_f, p_dropout):
        super(GCNN, self).__init__()
        # graph CL1
        self.conv1 = GCNConv(in_channels=in_f, out_channels=CL1_F)
        # graph CL2
        self.conv2 = GCNConv(in_channels=CL1_F, out_channels=CL2_F)
        # FC1
        self.lin1 = nn.Linear(CL2_F, out_f)
        self.bn1 = nn.BatchNorm1d(CL1_F)
        self.bn2 = nn.BatchNorm1d(CL2_F)
        self.p_dropout = p_dropout  
        
    def forward(self, x, edge_index, edge_weight, batch):
        #x, edge_index = data.x, data.edge_index
        # node embeddings:
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = self.bn1(x)
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index, edge_weight))
        x = self.bn2(x)
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        # graph embedding: pooling at graph level (sum over all nodes embeddings)
        x = global_add_pool(x, batch)
        out = self.lin1(x)
        return out

class GCNN_3p_old(nn.Module):
    def __init__(self, in_f, CL1_F, CL2_F, out_f):
        super(GCNN_3p_old, self).__init__()
        # graph CL1
        self.conv1 = GCNConv(in_channels=in_f, out_channels=CL1_F)
        # graph CL2
        self.conv2 = GCNConv(in_channels=CL1_F, out_channels=CL2_F)
        # FC1
        self.lin1 = nn.Linear(CL2_F * 3, out_f)

    def forward(self, x, edge_index, batch):
        #x, edge_index = data.x, data.edge_index
        # node embeddings:
        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index))

        # graph embedding: pooling at graph level (sum over all nodes embeddings)
        batch = torch.zeros(data.x.shape[0], dtype=int) if data.batch is None else data.batch
        # x = global_add_pool(x, data.batch)
        x0 = global_add_pool(x, data.batch)
        x1 = global_mean_pool(x, data.batch)
        x2 = global_max_pool(x, data.batch)
        x = torch.cat([x0, x1, x2], dim=-1)
        out = self.lin1(x)
        return x, out  # returns the embedding x & prediction out

class GCNN3L(nn.Module):
    def __init__(self, in_f, CL1_F, CL2_F, CL3_F, out_f):
        super(GCNN3L, self).__init__()

        # graph CL1
        self.conv1 = GCNConv(in_channels=in_f, out_channels=CL1_F)
        # graph CL2
        self.conv2 = GCNConv(in_channels=CL1_F, out_channels=CL2_F)
        # graph CL3
        self.conv3 = GCNConv(in_channels=CL2_F, out_channels=CL3_F)
        # FC1
        self.lin1 = nn.Linear(CL3_F, out_f)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # node embeddings:
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))

        # graph embedding: pooling at graph level (sum over all nodes embeddings)
        data.batch = torch.zeros(data.x.shape[0], dtype=int) if data.batch is None else data.batch
        x = global_add_pool(x, data.batch)

        out = self.lin1(x)

        return x, out  # returns the embedding x & prediction out


class Cheb_GCNN(nn.Module):
    def __init__(self, in_f, CL1_F, CL2_F, K, out_f, p_dropout): #, DL1_F, DL2_F
        super(Cheb_GCNN, self).__init__()
        # graph CL1
        self.conv1 = ChebConv(in_channels=in_f, out_channels=CL1_F, K=K)
        # graph CL2
        self.conv2 = ChebConv(in_channels=CL1_F, out_channels=CL2_F, K=K)
        # FC1
        self.lin1 = nn.Linear(CL2_F, out_f)
        self.bn1 = nn.BatchNorm1d(CL1_F)
        self.p_dropout = p_dropout        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        # node embeddings:
        x = F.relu(self.conv1(x, edge_index))
       # x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        # graph embedding: pooling at graph level (sum over all nodes embeddings)      
        batch = torch.zeros(data.x.shape[0],dtype=int) if data.batch is None else data.batch
        x = global_add_pool(x, data.batch)
        out = self.lin1(x)
        return x, out # returns the embedding x & prediction out

class GCNN2(nn.Module):
    def __init__(self, in_f ,CL1_F, CL2_F, out_f):
        super(GCNN2, self).__init__()
        # graph CL1
        self.conv1 = GCNConv(in_channels=in_f, out_channels=CL1_F)
        # graph CL2
        self.conv2 = GCNConv(in_channels=CL1_F, out_channels=CL2_F)
        # FC1
        self.lin1 = nn.Linear(CL2_F, out_f)
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        # node embeddings:
        x = F.relu(self.conv1(x, edge_index))
        #x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        # graph embedding: pooling at graph level (sum over all nodes embeddings)
        batch = torch.zeros(data.x.shape[0],dtype=int) if data.batch is None else data.batch
        x = global_add_pool(x, data.batch)
        out = self.lin1(x)
        return x, out # returns the embedding x & prediction out

class GAT(nn.Module):
    def __init__(self, in_f, CL1_F, CL2_F, heads_1, heads_2, out_f):
        super(GAT, self).__init__()
        # graph CL1
        self.conv1 = GATConv(in_channels=in_f, out_channels=CL1_F, heads=heads_1) # dropout=0.5
        # graph CL2
        self.conv2 = GATConv(in_channels=CL1_F*heads_1, out_channels=CL2_F, heads=heads_2, concat=False)
        # FC1
        self.lin1 = nn.Linear(CL2_F*heads_2, out_f)
    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        # node embeddings:
        x = F.elu(self.conv1(x, edge_index))
        #x = F.dropout(x, p=0.5, training=self.training)
        x = F.elu(self.conv2(x, edge_index))
        # graph embedding: pooling at graph level (sum over all nodes embeddings)
        data.batch = torch.zeros(data.x.shape[0], dtype=int) if data.batch is None else data.batch
        x = global_add_pool(x, data.batch)
        out = self.lin1(x)
        return x, out # returns the embedding x & prediction out