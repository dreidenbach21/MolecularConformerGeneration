# %load_ext autoreload
# %autoreload 2
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import sys
import pickle
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict, Counter

import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolTransforms
from rdkit.Chem import Draw
from rdkit.Chem.Draw import IPythonConsole
from rdkit import rdBase
import time
import py3Dmol
print( time.asctime())
import torch
print(torch.__version__, torch.cuda.is_available())
IPythonConsole.molSize = 400,400
IPythonConsole.drawOptions.addAtomIndices = True

sys.path.insert(0, '/home/dreidenbach/code/mcg')
sys.path.insert(0, '/home/dreidenbach/code/mcg/coagulation')
sys.path.insert(0, '/home/dreidenbach/code/mcg/coagulation/model')
sys.path.insert(0, '/home/dreidenbach/code/mcg/coagulation/utils')

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
from model.benchmarker import *
import glob
# config_path = "/home/dreidenbach/code/mcg/coagulation/configs/config_qm9.yaml"
# config = OmegaConf.load(config_path)

from hydra.experimental import compose, initialize_config_dir
from omegaconf import DictConfig

# Set the path to your config directory and config name
config_dir = "/home/dreidenbach/code/mcg/coagulation/configs"
config_name = "config_qm9.yaml"

# Initialize the Hydra config system
initialize_config_dir(config_dir)

# Load the config using Hydra
cfg = compose(config_name)

runner = BenchmarkRunner(true_mols = '/data/dreidenbach/data/torsional_diffusion/QM9/test_mols.pkl',
                                valid_mols = '/data/dreidenbach/data/torsional_diffusion/QM9/test_smiles.csv',
                                save_dir = '/data/dreidenbach/data/torsional_diffusion/QM9/test_set',
                                batch_size = cfg.data['train_batch_size'],
                                dataset='QM9',
                                name = 'qm9_full_V3_check') #! drugs_full
model = VAE(cfg.vae, cfg.encoder, cfg.decoder, cfg.losses, cfg.coordinates, device = "cuda").cuda()
if model is not None:
    print("# of Encoder Params = ", sum(p.numel()
          for p in model.encoder.parameters() if p.requires_grad))
    print("# of Decoder Params = ", sum(p.numel()
          for p in model.decoder.parameters() if p.requires_grad))
    print("# of VAE Params = ", sum(p.numel()
          for p in model.parameters() if p.requires_grad))
weights_path = "/home/dreidenbach/code/notebooks/model_chkpt/QM9_med_vae_kl_dist_anneal_mixing_05-04_03-40-56_4.pt"

chkpt = torch.load(weights_path)
if "model" in chkpt:
    chkpt = chkpt['model']
model.load_state_dict(chkpt, strict = False)
runner.generate(model, use_wandb = False, rdkit_only = False)
runner.generate(model, use_wandb = False, rdkit_only = True)
# start = runner.generated_molecules

