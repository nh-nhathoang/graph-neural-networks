import torch
import torch.nn as nn
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU, Dropout
from torch_geometric.nn import GINConv, TransformerConv
from torch_geometric.nn import global_add_pool, global_mean_pool

class GIN(torch.nn.Module):
    def __init__(self, dim_h, node_feature):
        super(GIN, self).__init__()
        self.conv1 = GINConv(
            Sequential(Linear(node_feature, dim_h),
                       BatchNorm1d(dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU()))
        self.conv2 = GINConv(
            Sequential(Linear(dim_h, dim_h), BatchNorm1d(dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU()))
        self.conv3 = GINConv(
            Sequential(Linear(dim_h, dim_h), BatchNorm1d(dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU()))

        self.lin1 = Linear(dim_h*3, dim_h*2)
        self.lin2 = Linear(dim_h*2, dim_h*2)
        self.lin4 = Linear(dim_h*2, 1)   
        self.relu = nn.ReLU()

    def forward(self, x, edge_index, batch):
        # Node embeddings 
        h1 = self.conv1(x, edge_index)
        h2 = self.conv2(h1, edge_index)
        h3 = self.conv3(h2, edge_index)

        # Graph-level readout
        h1 = global_add_pool(h1, batch)
        h2 = global_add_pool(h2, batch)
        h3 = global_add_pool(h3, batch)

        # Concatenate graph embeddings
        h = torch.cat((h1, h2, h3), dim=1)
        
        # Classifier
        h = self.relu(self.lin1(h))
        #h = F.dropout(h, p=0.0, training=self.training)
        h = self.relu(self.lin2(h))
        #h = F.dropout(h, p=0.0, training=self.training)
        h = self.lin4(h)
        
        return h.squeeze(-1)

class Transformer(torch.nn.Module):
    def __init__(self, dim_h, node_feature, heads=4):
        super(Transformer, self).__init__()
        self.conv1 = TransformerConv(node_feature, dim_h, heads=heads)
        self.conv2 = TransformerConv(dim_h * heads, dim_h, heads=heads)
        self.conv3 = TransformerConv(dim_h * heads, dim_h, heads=heads)

        self.lin1 = Linear(dim_h * heads * 3, dim_h * 2)
        self.lin2 = Linear(dim_h * 2, dim_h * 2)
        self.lin3 = Linear(dim_h * 2, 1)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index, batch):
        h1 = self.conv1(x, edge_index)
        h1 = self.relu(h1)
        h2 = self.conv2(h1, edge_index)
        h2 = self.relu(h2)
        h3 = self.conv3(h2, edge_index)
        h3 = self.relu(h3)

        h1 = global_mean_pool(h1, batch)
        h2 = global_mean_pool(h2, batch)
        h3 = global_mean_pool(h3, batch)

        h = torch.cat((h1, h2, h3), dim=1)

        h = self.relu(self.lin1(h))
        #h = F.dropout(h, p=0.0, training=self.training)
        h = self.relu(self.lin2(h))
        #h = F.dropout(h, p=0.0, training=self.training)
        h = self.lin3(h)

        return h.squeeze(-1) 
