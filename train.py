import xnmt
from xnmt import xnmt_run_experiment

experiment = xnmt_run_experiment.RunExp(config_file="train_preproc.yaml")
experiment.run()