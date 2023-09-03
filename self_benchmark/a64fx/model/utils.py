import torch
import random
import numpy as np
from .DistSAGE import DistSAGE

def create_model_and_optimizer(config: dict):
    model = None
    optimizer = None
    if config['model_name'] == 'sage':
        model = DistSAGE(config['in_channels'], config['hidden_channels'], config['out_channels'], 
                         config['num_layers'], config['dropout'], config['num_bits'], config['is_pre_delay'])

        # wrap model with ddp
        model = torch.nn.parallel.DistributedDataParallel(model)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])

    return model, optimizer

def set_random_seed(random_seed):
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
