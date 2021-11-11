from experiments.EndToEndExperiment import EndToEndExperiment
from config.base import Grid
from evaluation.dataset_getter import DatasetGetter
from log.Logger import Logger

config_file = "config_GraphSAGE.yml"
dataset_name = "REDDIT-BINARY"

model_configurations = Grid(config_file, dataset_name)

exp_path = "test"

exp = EndToEndExperiment(model_configurations[0], exp_path)

dataset_getter = DatasetGetter()
logger = Logger("./log-out.txt", 'w')

train_acc, val_acc = exp.run_valid(dataset_getter, logger)

print(train_acc, val_acc)
