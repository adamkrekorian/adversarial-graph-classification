import torch.cuda

from experiments.EndToEndExperiment import EndToEndExperiment
from config.base import Grid
from evaluation.dataset_getter import DatasetGetter
from log.Logger import Logger
from models.graph_classifiers.GraphSAGE import GraphSAGEAdj

from config.base import Config


device = 'cuda' if torch.cuda.is_available() else 'cpu'

saved_model_path = "../graph-sage-binary.pt"


net = GraphSAGEAdj(1,2)
net.load_state_dict(net.load(saved_model_path))
net.to(device)

#