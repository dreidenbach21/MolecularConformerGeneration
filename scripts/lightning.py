import glob
import os

import dgl
import dgl.nn.pytorch as dglnn

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# import torchmetrics.functional as MF
import tqdm
# from ogb.nodeproppred import DglNodePropPredDataset
import lightning.pytorch as pl
# from pytorch_lightning import LightningDataModule, LightningModule, Trainer
# from pytorch_lightning.callbacks import ModelCheckpoint
import sys
import hydra
from omegaconf import DictConfig, OmegaConf
from model.lightning_vae import VAE
import ipdb
import wandb
import random
import logging
from utils.torsional_diffusion_data_all import * 
import datetime
# from model.benchmarker import *
# import torch.distributed as dist
# from dgl.data import split_dataset
from dgl.dataloading import GraphDataLoader
# import dgl.multiprocessing as mp
# from torch.nn.parallel import DistributedDataParallel

drugs_types = {'H': 0, 'Li': 1, 'B': 2, 'C': 3, 'N': 4, 'O': 5, 'F': 6, 'Na': 7, 'Mg': 8, 'Al': 9, 'Si': 10,
               'P': 11, 'S': 12, 'Cl': 13, 'K': 14, 'Ca': 15, 'V': 16, 'Cr': 17, 'Mn': 18, 'Cu': 19, 'Zn': 20,
               'Ga': 21, 'Ge': 22, 'As': 23, 'Se': 24, 'Br': 25, 'Ag': 26, 'In': 27, 'Sb': 28, 'I': 29, 'Gd': 30,
               'Pt': 31, 'Au': 32, 'Hg': 33, 'Bi': 34}
# from torchmetrics import Accuracy

def collate(samples):
    A, B = map(list, zip(*samples))
    A_graph = dgl.batch([x[0] for x in A])
    geo_A = dgl.batch([x[1] for x in A])
    Ap = dgl.batch([x[2] for x in A])
    A_cg = dgl.batch([x[3] for x in A])
    geo_A_cg = dgl.batch([x[4] for x in A])
    frag_ids = [x[5] for x in A]
    
    B_graph = dgl.batch([x[0] for x in B])
    geo_B = dgl.batch([x[1] for x in B])
    Bp = dgl.batch([x[2] for x in B])
    B_cg = dgl.batch([x[3] for x in B])
    geo_B_cg = dgl.batch([x[4] for x in B])
    B_frag_ids = [x[5] for x in B]
    return (A_graph, geo_A, Ap, A_cg, geo_A_cg, frag_ids), (B_graph, geo_B, Bp, B_cg, geo_B_cg, B_frag_ids)

def load_data(mode = 'train', data_dir='/home/dannyreidenbach/data/DRUGS/drugs/',dataset='drugs', limit_mols=0,
              log_dir='./test_run', num_workers=0, restart_dir=None, seed=0,
              split_path='/home/dannyreidenbach/data/DRUGS/split.npy',std_pickles=None):
    types = drugs_types
    data = ConformerDataset(data_dir, split_path, mode, dataset=dataset,
                                   types=types, transform=None,
                                   num_workers=num_workers,
                                   limit_molecules=limit_mols, #args.limit_train_mols,
                                   cache_path=None, #args.cache,
                                   name=f'{dataset}_{mode}_{limit_mols}_final',
                                   pickle_dir=std_pickles,
                                   raw_dir='/home/dannyreidenbach/data/DRUGS/dgl', 
                                   save_dir='/home/dannyreidenbach/data/DRUGS/dgl',
                                   use_diffusion_angle_def=False,
                                   boltzmann_resampler=None)
    return data

def get_dataloader(dataset, seed=None, batch_size=400, num_workers=0, mode = 'train'):
    if mode == 'train':
        use_ddp = True
    else:
        use_ddp = False
    dataloader = dgl.dataloading.GraphDataLoader(dataset, use_ddp=use_ddp, batch_size=batch_size,
                                                 shuffle=True, drop_last=False, num_workers=num_workers, collate_fn = collate)
    print("Data Loader",mode)
    return dataloader

@hydra.main(config_path="../configs", config_name="config_drugs.yaml")
def main(cfg: DictConfig): 
    import datetime
    now = datetime.datetime.now()
    suffix = f"_{now.strftime('%m-%d_%H-%M-%S')}"
    coordinate_type = cfg.coordinates
    NAME = cfg.wandb['name'] + suffix
    # wandb_run = wandb.init(
    #     project=cfg.wandb.project,
    #     name=NAME,
    #     notes=cfg.wandb.notes,
    #     config = cfg,
    #     save_code = True
    # )
    model = VAE(cfg.vae, cfg.encoder, cfg.decoder, cfg.losses, coordinate_type = cfg.coordinates, device = device)
    train_dataset = load_data(mode = 'train')
    val_dataset = load_data(mode = 'val')
    train_loader = get_dataloader(train_dataset)
    val_loader = get_dataloader(train_dataset)
    trainer = Trainer(
        accelerator="gpu",
        devices=[0, 1, 2, 3],
        max_epochs=10,
        # callbacks=[checkpoint_callback],
        strategy="ddp",
    )
    trainer.fit(model = model, train_dataloaders=train_loader)

if __name__ == "__main__":
    main()