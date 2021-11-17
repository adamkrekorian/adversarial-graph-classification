import torch.cuda

from models.graph_classifiers.GraphSAGE import GraphSAGEAdj
from evaluation.dataset_getter import DatasetGetter

import torch.nn as nn
from config.base import Grid, Config

import csv

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def compute_adj_matrix(data):
    d_hat = torch.eye(len(data.x)) + torch.diag(torch.bincount(data.edge_index[0], minlength = len(data.x)))

    adj_mat = torch.zeros((len(data.x), len(data.x)))

    for i in range(len(data.edge_index)):
        adj_mat[data.edge_index[0, i], data.edge_index[1, i]] += 1

    adj_mat.requires_grad = True

    a_tilde = torch.eye(len(data.x)) + adj_mat

    a_tilde.retain_grad()

    # d_hat_pow = torch.matrix_power(d_hat,)

    evals, evecs = torch.eig(d_hat, eigenvectors=True)  # get eigendecomposition
    evals = evals[:, 0]  # get real part of (real) eigenvalues

    evpow = evals ** (-1 / 2)  # raise eigenvalues to fractional power

    # build exponentiated matrix from exponentiated eigenvalues
    d_hat_pow = torch.matmul(evecs, torch.matmul(torch.diag(evpow), torch.inverse(evecs)))

    d_hat_pow.requires_grad = True

    a_hat = torch.matmul(torch.matmul(d_hat_pow, a_tilde), d_hat_pow)

    a_hat.retain_grad()
    return a_hat, adj_mat

def add_dummy_node(data):
    # add node
    shp = list(data.x.shape)
    shp[0] += 1
    data.x = torch.ones(shp, dtype=data.x.dtype)

    # add batch

    shp = list(data.batch.shape)
    shp[0] += 1
    data.batch = torch.zeros(shp, dtype=data.batch.dtype)
    return data

def add_edge(data, e1, e2):
    shp = list(data.edge_index.shape)
    shp[1] += 2
    tmp = torch.ones(shp, dtype=data.edge_index.dtype)

    tmp[:, :-2] = data.edge_index
    tmp[0, -2] = e1
    tmp[1, -2] = e2
    tmp[0, -1] = e2
    tmp[1, -1] = e1
    data.edge_index = tmp

    return data




# Load Dataset
config_file = "config_GraphSAGE.yml"
dataset_name = "REDDIT-BINARY"

model_configurations = Grid(config_file, dataset_name)
dataset_getter = DatasetGetter()

model_config = Config.from_dict(model_configurations[0])
dataset_class = model_config.dataset
dataset = dataset_class()

train_loader, val_loader = dataset_getter.get_train_val(dataset,
                                                        model_configurations[0]['batch_size'],
                                                        shuffle=False)

# Load Model
saved_model_path = "../graph-sage-binary-maxpool.pt"


net = GraphSAGEAdj(dim_features=dataset.dim_features, dim_target=dataset.dim_target, config=model_configurations[0])
net.load_state_dict(torch.load(saved_model_path))

net.to(device)
net.eval()

# Generate Adversarial Node

criterion = nn.CrossEntropyLoss()

# Metrics
metrics = {'num_incorrect': 0, 'num_successful': 0, 'num_unsuccessful': 0}

# add a node to data.x
for data in val_loader:
    num_add_nodes = 0
    num_add_edges = 0

    add_dummy_node(data)
    num_add_nodes += 1

    data['a_hat'], adj_matrix = compute_adj_matrix(data)
    output = net(data)

    if (data.y[0] != torch.argmax(output)):
        metrics['num_incorrect'] += 1
        print('Incorrect Classification')
        continue

    its = 0
    while its < data.num_nodes and data.y[0] == torch.argmax(output):
        data['a_hat'], adj_matrix = compute_adj_matrix(data)
        output = net(data)

        loss = criterion(output, data.y)
        # grads = torch.autograd.grad(output[:, 1], data['a_hat'], retain_graph=True)

        grads = torch.autograd.grad(loss, adj_matrix, retain_graph=True)
        node_to_add = torch.argmin(grads[0][:, -1])

        if grads[0][node_to_add, -1] < -0.001:
            data = add_edge(data, node_to_add, len(grads[0])-1)
            num_add_edges += 1
        else:
            add_dummy_node(data)
            num_add_nodes += 1
        its +=1

    if num_add_nodes+num_add_edges < data.num_nodes:
        metrics['num_successful'] += 1
    else:
        metrics['num_unsuccessful'] += 1
    print(f'Label: {data.y.item()}, Nodes: {data.num_nodes}, Edges {data.num_edges}')
    print(f'Percent Additional Nodes: {100*num_add_nodes/data.num_nodes:.3f}%, Percent Additional Edges: {100*num_add_edges/data.num_edges:.3f}%')

w = csv.writer(open("metrics-maxpool.csv", "w"))
for key, val in metrics.items():
    w.writerow([key, val])
