# %load_ext autoreload
# %autoreload 2
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="8"
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
config_name = "config_drugs.yaml"

# Initialize the Hydra config system
initialize_config_dir(config_dir)

# Load the config using Hydra
cfg = compose(config_name)

runner = BenchmarkRunner(true_mols = '/data/dreidenbach/data/torsional_diffusion/XL/test_mols.pkl',
                                valid_mols = '/data/dreidenbach/data/torsional_diffusion/XL/test_smiles.csv',
                                save_dir = '/data/dreidenbach/data/torsional_diffusion/XL/test_set',
                                batch_size = 100,
                                dataset='xl',
                                name = 'xl_full')
model = VAE(cfg.vae, cfg.encoder, cfg.decoder, cfg.losses, cfg.coordinates, device = "cuda").cuda()
# weights_path = "./home/dreidenbach/code/notebooks/model_chkpt/DRUGS_full_base_softplus_05-09_00-57-12_0_temp.pt" #1/5 data
# weights_path = "/home/dreidenbach/code/notebooks/model_chkpt/DRUGS_inst_2_5_confs_4gpu_05-12_18-21-40_4_temp.pt" #! bench2
# weights_path = "/home/dreidenbach/code/notebooks/model_chkpt/DRUGS_asia_4gp_distributed_full_item_swap_05-12_22-41-37_2_0_final.pt" #! bench
weights_path = "/home/dreidenbach/code/notebooks/model_chkpt/DRUGS_us_4gp_all_distributed_asian_force_second_05-13_01-41-29_3_0_final.pt" #! bench3
#DRUGS_asia_4gp_distributed_full_05-12_01-18-08_0_0_temp.pt" 
# #DRUGS_all_single_05-11_07-03-12_0_temp.pt" 
# #DRUGS_full_base_softplus_all_05-09_08-53-36_0_temp.pt"
chkpt = torch.load(weights_path)
if "model" in chkpt:
    chkpt = chkpt['model']
model.load_state_dict(chkpt, strict = False)
runner.generate(model, use_wandb = False, rdkit_only = False)
# print("\n\n\n\n\n\n\n\n\n\n")
# runner.generate(model, use_wandb = False, rdkit_only = True)
# start = runner.generated_molecules

