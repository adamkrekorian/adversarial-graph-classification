import torch.cuda

from models.graph_classifiers.GraphSAGE import GraphSAGEAdjMean
from evaluation.dataset_getter import DatasetGetter

import torch.nn as nn
from config.base import Grid, Config
import csv
import matplotlib.pyplot as plt
import numpy as np

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
saved_model_path = "../graph-sage-binary.pt"


net = GraphSAGEAdjMean(dim_features=dataset.dim_features, dim_target=dataset.dim_target, config=model_configurations[0])
net.load_state_dict(torch.load(saved_model_path))

net.to(device)
net.eval()

# Generate Adversarial Node

criterion = nn.CrossEntropyLoss()

metrics = {'num_incorrect': 0, 'num_successful': 0, 'num_unsuccessful': 0}

nodes_added_per_attack = []
edges_added_per_attack = []

nodes_in_original = []
edges_in_original = []
success_labels = []

failed_nodes_in_original = []
failed_edges_in_original = []
failed_labels = []

# add a node to data.x
for data in val_loader:
    num_add_nodes = 0
    num_add_edges = 0

    failed = False

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

        if num_add_nodes> 10 and num_add_edges == 0:
            failed = True
            break

    if num_add_nodes+num_add_edges < data.num_nodes and not failed:
        metrics['num_successful'] += 1
        nodes_added_per_attack.append(num_add_nodes)
        edges_added_per_attack.append(num_add_edges)

        nodes_in_original.append(len(data.x))
        edges_in_original.append(len(data.edge_index[0]))
        success_labels.append(data.y.item())
    else:
        metrics['num_unsuccessful'] += 1
        failed_nodes_in_original.append(len(data.x))
        failed_edges_in_original.append(len(data.edge_index[0]))
        failed_labels.append(data.y.item())
    print(f'Label: {data.y.item()}, Nodes: {data.num_nodes}, Edges {data.num_edges}')
    print(f'Percent Additional Nodes: {100*num_add_nodes/data.num_nodes:.3f}%, Percent Additional Edges: {100*num_add_edges/data.num_edges:.3f}%')

nodes_added_per_attack_np = np.array(nodes_added_per_attack)
edges_added_per_attack_np = np.array(edges_added_per_attack)

nodes_in_original_np = np.array(nodes_in_original)
edges_in_original_np = np.array(edges_in_original)
success_labels_np = np.array(success_labels)

failed_nodes_in_original_np = np.array(failed_nodes_in_original)
failed_edges_in_original_np = np.array(failed_edges_in_original)
failed_labels_np = np.array(failed_labels)

#Show success rate of adversarial attacks - what % of the time can we succeed? When do we fail
boundaries = [0, 200, 500, 1000, 2000, 5000, 10000]
plt.figure(figsize=(8,8))
sums = np.array([0,0])
for i in range(len(boundaries)-1):
    labels = ('Successful','Unsuccessful')
    success_in_range = nodes_in_original_np + edges_in_original_np >= boundaries[i]
    success_in_range1 = nodes_in_original_np + edges_in_original_np<boundaries[i+1]
    success_in_range = success_labels_np[np.logical_and(success_in_range, success_in_range1)]

    failed_in_range = failed_nodes_in_original_np + failed_edges_in_original_np >= boundaries[i]
    failed_in_range1 = failed_nodes_in_original_np + failed_edges_in_original_np < boundaries[i + 1]
    failed_in_range = failed_labels_np[np.logical_and(failed_in_range, failed_in_range1)]
    old_sums = sums
    sums = np.array([len(success_in_range), len(failed_in_range)])
    plt.bar(labels, sums, label=str(boundaries[i])+'-'+ str(boundaries[i+1]), bottom=old_sums)
    sums = old_sums + sums
plt.legend()
plt.title('Success of adversarial attacks, by input nodes+edges')
plt.ylabel('Count')
plt.savefig('success_rate.png')

#Show success rate of adversarial attacks by class
plt.figure(figsize=(8,8))
labels = ('Successful','Unsuccessful')
sums = np.array([len(success_labels_np[success_labels_np == 0]), len(failed_labels_np[failed_labels_np == 0])])
old_sums = sums
plt.bar(labels, sums, label='Class 0')
sums = np.array([len(success_labels_np[success_labels_np == 1]), len(failed_labels_np[failed_labels_np == 1])])
plt.bar(labels, sums, label='Class 1', bottom=old_sums)
plt.legend()
plt.title('Success of adversarial attacks, by class')
plt.ylabel('Count')
plt.savefig('success_rate_class.png')

#Show some data on the adversarial attacks- nodes vs edges added for each successful attack, broken down by the number of nodes in the original graph
plt.figure(figsize=(8,8))
node_pct = nodes_added_per_attack_np/nodes_in_original_np
edge_pct = edges_added_per_attack_np/edges_in_original_np

plt.scatter(node_pct[success_labels_np==0], edge_pct[success_labels_np==0], label='Class 0')
plt.scatter(node_pct[success_labels_np==1], edge_pct[success_labels_np==1], label='Class 1')
plt.legend()
plt.title('Nodes added as a % of input nodes vs edges added as a % of input edges')
plt.xlabel("% of nodes added")
plt.ylabel("% of edges added")
plt.savefig('node_added_pct.png')

plt.figure(figsize=(8,8))
plt.scatter(nodes_added_per_attack_np[success_labels_np==0], edges_added_per_attack_np[success_labels_np==0], label='Class 0')
plt.scatter(nodes_added_per_attack_np[success_labels_np==1], edges_added_per_attack_np[success_labels_np==1], label='Class 1')
plt.legend()
plt.xlabel("Nodes added")
plt.ylabel("Edges added")
plt.title('Nodes added vs edges added by class')
plt.savefig('node_added.png')

#Maybe N+E added vs number in original graph?
ne_added = nodes_added_per_attack_np + edges_added_per_attack_np
ne_original = nodes_in_original_np + edges_in_original_np

plt.figure(figsize=(8,8))
plt.scatter(ne_added[success_labels_np==0], ne_original[success_labels_np==0], label='Class 0')
plt.scatter(ne_added[success_labels_np==1], ne_original[success_labels_np==1], label='Class 1')
plt.legend()
plt.xlabel("Number of nodes + edges added")
plt.ylabel("Number of nodes + edges in original graph")
plt.title('Nodes and edges added vs nodes and edges in the original graph by class')
plt.savefig('node_added_ratio.png')



w = csv.writer(open("metrics-mean.csv", "w"))
for key, val in metrics.items():
    w.writerow([key, val])

