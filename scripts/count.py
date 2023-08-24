import sys
import os
import hydra
from omegaconf import DictConfig, OmegaConf
import ipdb
import torch
import wandb
import random
import logging
from utils.data_count import qm9_count_local # , QM9_DIMS, DRUG_DIMS
from model.vae import VAE
import datetime
import torch.multiprocessing as mp
from model.benchmarker import *

def load_data(cfg):
    #mp.set_start_method('spawn') # use 'spawn' method instead of 'fork'
    #mp.set_sharing_strategy('file_system') 
    print("Loading DRUGs...")
    train_loader, train_data = qm9_count_local(batch_size=cfg['train_batch_size'], mode='train', num_workers= 16) #cook_drugs_local
    val_loader, val_data = qm9_count_local(batch_size=cfg['val_batch_size'], mode='val', num_workers = 16)
    print("Loading DRUGS --> Done")
    return train_loader, train_data, val_loader, val_data

@hydra.main(config_path="../configs", config_name="config_drugs.yaml")
def main(cfg: DictConfig): #['encoder', 'decoder', 'vae', 'optimizer', 'losses', 'data', 'coordinates', 'wandb']
    train_loader, train_data, val_loader, val_data = load_data(cfg.data)
    ipdb.set_trace()


if __name__ == "__main__":
    # mp.set_sharing_strategy('file_system')
    main()
