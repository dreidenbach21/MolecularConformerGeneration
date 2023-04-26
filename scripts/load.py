import sys
import resource

# Get the current resource limits
# soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_AS)

# # Increase the soft limit to 4GB
# new_soft_limit = 100 * 1024**3
# print(soft_limit, hard_limit, new_soft_limit)
# resource.setrlimit(resource.RLIMIT_AS, (new_soft_limit, hard_limit))

print(sys.path)
import os
print(os.environ['PYTHONPATH'])
# os.environ["SHM_PAGE_SIZE"] = "10485760" 10mb
import hydra
from omegaconf import DictConfig, OmegaConf
import ipdb
import torch
import wandb
import random
import logging
from utils.torsional_diffusion_data_all import load_torsional_data  # , QM9_DIMS, DRUG_DIMS
from model.vae import VAE
import datetime
import torch.multiprocessing as mp
# import multiprocessing as mp
# from multiprocessing import set_start_method
#mp.set_start_method('spawn') # use 'spawn' method instead of 'fork'
#mp.set_sharing_strategy('file_system') 

def load_data(cfg):
    # geom_path = "/home/dreidenbach/data/GEOM/rdkit_folder/"
    # qm9_path = geom_path + "qm9/"
    # drugs_path = geom_path + "drugs/"
    #mp.set_start_method('spawn') # use 'spawn' method instead of 'fork'
    #mp.set_sharing_strategy('file_system') 
    print("Loading QM9...")
    # with open(geom_path + "qm9_safe_v2.pickle", 'rb') as f:
    #     qm9 = pickle.load(f)
    train_loader, train_data = load_torsional_data(batch_size=cfg['train_batch_size'], mode='train', limit_mols=2000, num_workers = 5)
    #ipdb.set_trace()
    val_loader, val_data = load_torsional_data(batch_size=cfg['val_batch_size'], mode='val', limit_mols=200, num_workers = mp.cpu_count())
    ipdb.set_trace()
    # val_loader, val_data = None, None
    print("Loading QM9 --> Done")
    return train_loader, train_data, val_loader, val_data

@hydra.main(config_path="../configs", config_name="config.yaml")
def main(cfg: DictConfig): #['encoder', 'decoder', 'vae', 'optimizer', 'losses', 'data', 'coordinates', 'wandb']
    # ipdb.set_trace()
    import datetime
    now = datetime.datetime.now()
    suffix = f"_{now.strftime('%m-%d_%H-%M-%S')}"
    coordinate_type = cfg.coordinates
    NAME = cfg.wandb['name'] + suffix
    train_loader, train_data, val_loader, val_data = load_data(cfg.data)
  
if __name__ == "__main__":
    # mp.set_sharing_strategy('file_system')
    main()
