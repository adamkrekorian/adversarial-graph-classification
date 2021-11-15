import torch.cuda

from models.graph_classifiers.GraphSAGE import GraphSAGEAdj
from evaluation.dataset_getter import DatasetGetter
from config.base import Grid, Config

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def compute_adj_matrix(data):
    d_hat = torch.eye(len(data.x)) + torch.diag(torch.bincount(data.edge_index[0]))

    adj_mat = torch.zeros((len(data.x), len(data.x)))

    for i in range(len(data.edge_index)):
        adj_mat[data.edge_index[0, i], data.edge_index[1, i]] += 1

    adj_mat.requires_grad = True

    a_tilde = torch.eye(len(data.x)) + adj_mat

    # d_hat_pow = torch.matrix_power(d_hat,)

    evals, evecs = torch.eig(d_hat, eigenvectors=True)  # get eigendecomposition
    evals = evals[:, 0]  # get real part of (real) eigenvalues

    evpow = evals ** (-1 / 2)  # raise eigenvalues to fractional power

    # build exponentiated matrix from exponentiated eigenvalues
    d_hat_pow = torch.matmul(evecs, torch.matmul(torch.diag(evpow), torch.inverse(evecs)))

    a_hat = torch.matmul(torch.matmul(d_hat_pow, a_tilde), d_hat_pow)
    return a_hat, adj_mat

def add_dummy_node(data):
    print(data)



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

net = GraphSAGEAdj(dim_features=dataset.dim_features, dim_target=dataset.dim_target, config=model_configurations[0])
net.load_state_dict(torch.load(saved_model_path))
net.to(device)

# Generate Adversarial Node

# add a node to data.x
for data in train_loader:
    add_dummy_node(data)


    data['a_hat'], adj_matrix = compute_adj_matrix(data)

# add every edge to edge_index
# compute adjacency matrix on data
# forward pass of GraphSAGE
# torch.autograd.grad(output[0][:, 0], adj_mat, retain_graph=True) gradient of the output with respect to the adjacency matrix
    # all the intermediate tensors require grad

