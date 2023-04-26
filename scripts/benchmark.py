import sys
import os
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
from model.benchmarker import *

def load_data(cfg):
    # geom_path = "/home/dreidenbach/data/GEOM/rdkit_folder/"
    # qm9_path = geom_path + "qm9/"
    # drugs_path = geom_path + "drugs/"
    #mp.set_start_method('spawn') # use 'spawn' method instead of 'fork'
    #mp.set_sharing_strategy('file_system') 
    print("Loading QM9...")
    # with open(geom_path + "qm9_safe_v2.pickle", 'rb') as f:
    #     qm9 = pickle.load(f)
    train_loader, train_data = load_torsional_data(batch_size=cfg['train_batch_size'], mode='train', limit_mols=2000, num_workers = mp.cpu_count())
    #ipdb.set_trace()
    val_loader, val_data = load_torsional_data(batch_size=cfg['val_batch_size'], mode='val', limit_mols=200, num_workers = mp.cpu_count())
    ipdb.set_trace()
    # val_loader, val_data = None, None
    print("Loading QM9 --> Done")
    return train_loader, train_data, val_loader, val_data

@hydra.main(config_path="../configs", config_name="config.yaml")
def main(cfg: DictConfig): #['encoder', 'decoder', 'vae', 'optimizer', 'losses', 'data', 'coordinates', 'wandb']
    import datetime
    now = datetime.datetime.now()
    suffix = f"_{now.strftime('%m-%d_%H-%M-%S')}"
    coordinate_type = cfg.coordinates
    NAME = cfg.wandb['name'] + suffix
    # train_loader, train_data, val_loader, val_data = load_data(cfg.data)
    # assert(1 == 0)
    F = cfg.encoder["coord_F_dim"]
    D = cfg.encoder["latent_dim"]
    model = VAE(cfg.vae, cfg.encoder, cfg.decoder, cfg.losses, coordinate_type, device = "cuda").cuda()
    
    print("CUDA CHECK", next(model.parameters()).is_cuda)
    print("# of Encoder Params = ", sum(p.numel()
          for p in model.encoder.parameters() if p.requires_grad))
    print("# of Decoder Params = ", sum(p.numel()
          for p in model.decoder.parameters() if p.requires_grad))
    print("# of VAE Params = ", sum(p.numel()
          for p in model.parameters() if p.requires_grad))
    
    
    runner = BenchmarkRunner()
    runner.generate(model, rdkit_only = True)

    print("Benchmark Test Complete")


if __name__ == "__main__":
    # mp.set_sharing_strategy('file_system')
    main()
