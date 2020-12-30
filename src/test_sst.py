import torch
from omegaconf import OmegaConf
import pdb

device = torch.device('cpu')
models = OmegaConf.load('../latest_silero_models.yml')
pdb.set_trace()
