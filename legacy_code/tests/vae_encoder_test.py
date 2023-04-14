import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="5"
import sys
import pickle
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict, Counter

import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolTransforms
from rdkit import rdBase
import time
import py3Dmol
import torch
import copy
import ipdb

sys.path.insert(0, '/home/dreidenbach/code/mcg')
sys.path.insert(0, '/home/dreidenbach/code/mcg/coagulation')
# sys.path.insert(0, '/home/dreidenbach/code/mcg/EquiBind')
from ecn import *
from ecn_layers import *
from iegmn import *
from model_utils import *
from embedding import *
from molecule_utils import *
from ecn_3d_layers import *
from ecn_3d import *
from equivariant_model_utils import *
from vae import *

ecn_model_params = {
  "geometry_reg_step_size": 0.001,
  "geometry_regularization": True,
#   "standard_norm_order": True,
#   "A_evolve": True, # whether or not the coordinates are changed in the EGNN layers
#   "B_evolve": True,
#   "B_no_softmax": False,
#   "A_no_softmax": False,
  "n_lays": 3,  # 5 in  good run
  "debug": False,
  "shared_layers": False, # False in good run
  "noise_decay_rate": 0.5,
  "noise_initial": 1,
  "use_edge_features_in_gmn": True,
  "use_mean_node_features": True,
  "atom_emb_dim": 64, #residue_emb_dim
  "latent_dim": 64, #! D
  "coord_F_dim":32, #! F 3DLinker used 12 and 5 layers
  "dropout": 0.1,
  "nonlin": 'lkyrelu', # ['swish', 'lkyrelu']
  "leakyrelu_neg_slope": 1.0e-2, # 1.0e-2 in  good run
  "cross_msgs": True,
  "layer_norm": 'BN', # ['0', 'BN', 'LN'] # BN in good run
  "layer_norm_coords": '0', # ['0', 'LN'] # 0 in good run
  "final_h_layer_norm": '0', # ['0', 'GN', 'BN', 'LN'] # 0 in good run
  "pre_crossmsg_norm_type":  '0', # ['0', 'GN', 'BN', 'LN']
  "post_crossmsg_norm_type": '0', # ['0', 'GN', 'BN', 'LN']
  "use_dist_in_layers": True,
  "skip_weight_h": 0.5, # 0.5 in good run
  "x_connection_init": 0.25, # 0.25 in good run
  "random_vec_dim": 0, # set to 0 to have no stochasticity
  "random_vec_std": 1,
  "use_scalar_features": False, # Have a look at lig_feature_dims in process_mols.py to see what features we are talking about.
  "num_A_feats":  None,# leave as None to use all ligand features. Have a look at lig_feature_dims in process_mols.py to see what features we are talking about. If this is 1, only the first of those will be used.
  "normalize_coordinate_update": True,
  "weight_sharing":True,
}

geom_path = "/home/dreidenbach/data/GEOM/rdkit_folder/"
qm9_path = geom_path + "qm9/"
drugs_path = geom_path + "drugs/"
print("Loading QM9...")
with open(geom_path + "qm9_safe_v2.pickle", 'rb') as f:
    qm9 = pickle.load(f)
print("Loading QM9 --> Done")
def init_graph_match(rd_mol, kit_mol, D = 64, F = 32):
    A = mol2graph(rd_mol).to('cuda:0')
    B = mol2graph(kit_mol, use_rdkit_coords = True).to('cuda:0')

    A_frags, A_frag_ids, A_adj, A_out, A_bond_break, A_cg_bonds, A_cg_map = coarsen_molecule(rd_mol)
    A_cg = conditional_coarsen_3d(A, A_frag_ids, A_cg_map, radius=4, max_neighbors=None, latent_dim_D = D, latent_dim_F =F).to('cuda:0')
    B_frags, B_frag_ids, B_adj, B_out, B_bond_break, B_cg_bonds, B_cg_map = coarsen_molecule(kit_mol)
    B_cg = conditional_coarsen_3d(B, B_frag_ids, B_cg_map, radius=4, max_neighbors=None, latent_dim_D = D, latent_dim_F =  F).to('cuda:0')
    
    geometry_graph_A = get_geometry_graph(rd_mol).to('cuda:0')
    geometry_graph_B = get_geometry_graph(kit_mol).to('cuda:0')
    
    Ap = create_pooling_graph(A, A_frag_ids).to('cuda:0')
    Bp = create_pooling_graph(B, B_frag_ids).to('cuda:0')
    
    geometry_graph_A_cg = get_coarse_geometry_graph(A_cg, A_cg_map).to('cuda:0')
    geometry_graph_B_cg = get_coarse_geometry_graph(B_cg, B_cg_map).to('cuda:0')
    
    return A,B,geometry_graph_A,geometry_graph_B, Ap, Bp, A_cg, B_cg, geometry_graph_A_cg, geometry_graph_B_cg

if __name__ =="__main__":
    smi = 'CC[C@@](C)(O)[C@@H](C)OC'
    rd_mol = qm9[smi]["conformers"][0]["rd_mol"]
    kit_mol = copy.deepcopy(rd_mol)

    A,B,geometry_graph_A,geometry_graph_B, Ap, Bp, A_cg, B_cg, geometry_graph_A_cg, geometry_graph_B_cg = init_graph_match(rd_mol, 
                                                                                                                kit_mol, D = ecn_model_params["latent_dim"], F = ecn_model_params["coord_F_dim"])
    print(A)
    print(Ap)
    print(A_cg)
    edge_dim = A.edata['feat'].shape[1]
    print("edge feature dim", edge_dim)
    model = ECN3D(device = "cuda", #cpu
              A_input_edge_feats_dim = edge_dim,
              B_input_edge_feats_dim = edge_dim, 
              **ecn_model_params).cuda()
    encoder = Encoder(model, ecn_model_params["latent_dim"], ecn_model_params["coord_F_dim"]).cuda()
    print("# of Params = ", sum(p.numel() for p in encoder.parameters() if p.requires_grad))
    if True:
        for name, param in encoder.named_parameters():
            if param.requires_grad:
                print( name, param.shape, param.numel())


    results, geom_losses, geom_loss_cg, full_trajectory, full_trajectory_cg = encoder(A, B, geometry_graph_A, geometry_graph_B, Ap, Bp, A_cg, B_cg, geometry_graph_A_cg, geometry_graph_B_cg, epoch=0)
    # results = {
    #         "Z_V": Z_V,
    #         "Z_h": Z_h,
    #         "v_A": v_A,
    #         "v_B": v_B,
    #         "h_A": h_A,
    #         "h_B": h_B,

    #         "prior_mean_V": prior_mean_V,
    #         "prior_mean_h": prior_mean_h,
    #         "prior_logvar_V": prior_logvar_V,
    #         "prior_logvar_h": prior_logvar_h,

    #         "posterior_mean_V": posterior_mean_V,
    #         "posterior_mean_h": posterior_mean_h,
    #         "posterior_logvar_V": posterior_logvar_V,
    #         "posterior_logvar_h": posterior_logvar_h,

    #     }

    print(results["Z_V"].shape, results["Z_h"].shape)
    kl_v = encoder.kl(results["posterior_mean_V"], results["posterior_logvar_V"], results["prior_mean_V"], results["prior_logvar_V"], coordinates = True)
    kl_h = encoder.kl(results["posterior_mean_h"], results["posterior_logvar_h"], results["prior_mean_h"], results["prior_logvar_h"], coordinates = False)
    print("Test Success!")
