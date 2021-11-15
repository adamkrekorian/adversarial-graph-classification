import torch
from torch import nn
from torch.nn import functional as F

from torch_geometric.nn import SAGEConv, global_mean_pool

import torch
import torch.nn.functional as F


class GraphSAGE(nn.Module):
    def __init__(self, dim_features, dim_target, config):
        super().__init__()

        num_layers = config['num_layers']
        dim_embedding = config['dim_embedding']
        self.aggregation = config['aggregation']  # can be mean or max

        if self.aggregation == 'max':
            self.fc_max = nn.Linear(dim_embedding, dim_embedding)

        self.layers = nn.ModuleList([])
        for i in range(num_layers):
            dim_input = dim_features if i == 0 else dim_embedding

            conv = SAGEConv(dim_input, dim_embedding)
            # Overwrite aggregation method (default is set to mean
            conv.aggr = self.aggregation

            self.layers.append(conv)

        # For graph classification
        self.fc1 = nn.Linear(num_layers * dim_embedding, dim_embedding)
        self.fc2 = nn.Linear(dim_embedding, dim_target)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x_all = []

        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            if self.aggregation == 'max':
                x = torch.relu(self.fc_max(x))
            x_all.append(x)

        x = torch.cat(x_all, dim=1)
        x = global_mean_pool(x, batch)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class GraphSAGEAdj(nn.Module):
    def __init__(self, dim_features, dim_target, config):
        super(GraphSAGEAdj, self).__init__()
        dim_embedding = config['dim_embedding']
        self.aggregation = config['aggregation']  # can be mean or max

        if self.aggregation == 'max':
            self.fc_max = nn.Linear(dim_embedding, dim_embedding)

        self.gc1 = nn.Linear(1,32)
        self.gc2 = nn.Linear(32, 32)
        self.gc3 = nn.Linear(32, 32)

        self.fc1 = nn.Linear(3 * dim_embedding, dim_embedding)
        self.fc2 = nn.Linear(dim_embedding, dim_target)

    def forward(self, data):
        a_hat, x, batch = data.a_hat, data.x, data.batch
        inp = torch.matmul(a_hat, x)


        x_all = []
        y = F.relu(self.gc1(inp))
        x_all.append(y)
        y = F.relu(self.gc2(y))
        x_all.append(y)
        y = F.relu(self.gc3(y))
        x_all.append(y)

        x = torch.cat(x_all, dim=1)
        x = global_mean_pool(x, batch)


        f = F.relu(self.fc1(x))
        f = self.fc2(f)

        return f
