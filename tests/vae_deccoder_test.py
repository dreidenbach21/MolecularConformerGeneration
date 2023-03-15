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
from decoder_utils import *
from geometry_utils import *

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
  "layer_norm": 'LN', # ['0', 'BN', 'LN'] # BN in good run #TODO: batch norm has issues with only 1 sample during training only
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
    
    return A,B,geometry_graph_A,geometry_graph_B, Ap, Bp, A_cg, B_cg, geometry_graph_A_cg, geometry_graph_B_cg, A_frag_ids, A_bond_break

if __name__ =="__main__":
    smiles = ['CC[C@@](C)(O)[C@@H](C)OC', 'C[C@@]12COCC=C[C@@H]1C2', 'CC[C@@](C)(O)[C@@H](C)OC']
    data = []
    frag_ids = []
    bond_breaks = []
    for smi in smiles:
        rd_mol = qm9[smi]["conformers"][0]["rd_mol"]
        kit_mol = copy.deepcopy(rd_mol)
        A,B,geometry_graph_A,geometry_graph_B, Ap, Bp, A_cg, B_cg, geometry_graph_A_cg, geometry_graph_B_cg, A_ids, A_bond_break = init_graph_match(rd_mol, kit_mol, D = ecn_model_params["latent_dim"], F = ecn_model_params["coord_F_dim"])
        data.append((A,B,geometry_graph_A,geometry_graph_B, Ap, Bp, A_cg, B_cg, geometry_graph_A_cg, geometry_graph_B_cg))
        frag_ids.append(A_ids)
        bond_breaks.append(A_bond_break)
    As, Bs, geoAs, geoBs, Aps, Bps, Acgs, Bcgs, geoAcgs, geoBcgs = [], [], [], [], [], [], [], [], [], []
    for d in data:
        A,B,geometry_graph_A,geometry_graph_B, Ap, Bp, A_cg, B_cg, geometry_graph_A_cg, geometry_graph_B_cg = d
        As.append(A)
        Bs.append(B)
        geoAs.append(geometry_graph_A)
        geoBs.append(geometry_graph_B)
        Aps.append(Ap)
        Bps.append(Bp)
        Acgs.append(A_cg)
        Bcgs.append(B_cg)
        geoAcgs.append(geometry_graph_A_cg)
        geoBcgs.append(geometry_graph_B_cg)

    A_batch = dgl.batch(As).to('cuda:0')
    B_batch = dgl.batch(Bs).to('cuda:0')
    geoA_batch = dgl.batch(geoAs).to('cuda:0')
    geoB_batch = dgl.batch(geoBs).to('cuda:0')
    Ap_batch = dgl.batch(Aps).to('cuda:0')
    Bp_batch = dgl.batch(Bps).to('cuda:0')
    Acg_batch = dgl.batch(Acgs).to('cuda:0')
    Bcg_batch = dgl.batch(Bcgs).to('cuda:0')
    geoAcg_batch = dgl.batch(geoAcgs).to('cuda:0')
    geoBcg_batch = dgl.batch(geoBcgs).to('cuda:0')

    edge_dim = A.edata['feat'].shape[1]
    print("edge feature dim", edge_dim)
    model = ECN3D(device = "cuda", #cpu
              A_input_edge_feats_dim = edge_dim,
              B_input_edge_feats_dim = edge_dim, 
              **ecn_model_params).cuda()
    encoder = Encoder(model, ecn_model_params["latent_dim"], ecn_model_params["coord_F_dim"]).cuda()
    print("# of Params = ", sum(p.numel() for p in encoder.parameters() if p.requires_grad))
    # if True:
    #     for name, param in encoder.named_parameters():
    #         if param.requires_grad:
    #             print( name, param.shape, param.numel())


    results, geom_losses, geom_loss_cg, full_trajectory, full_trajectory_cg = encoder(A_batch, B_batch, geoA_batch, geoB_batch, Ap_batch, Bp_batch, Acg_batch, Bcg_batch, geoAcg_batch, geoBcg_batch, epoch=0)
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
    print("Encoder Success!")

    decode_matcher = IEGMN_Bidirectional(device = "cuda", #cpu
              A_input_edge_feats_dim = edge_dim,
              B_input_edge_feats_dim = edge_dim, 
              **ecn_model_params).cuda()
    decoder =  Decoder(decode_matcher, ecn_model_params["latent_dim"], ecn_model_params["coord_F_dim"], model.atom_embedder).cuda()
    print("Decoder Init Success")
    print("# of Params = ", sum(p.numel() for p in decoder.parameters() if p.requires_grad))

    generated_molecule, rdkit_reference = decoder(Acg_batch, B_batch, frag_ids, bond_breaks)

    for G, R, T in zip(dgl.unbatch(generated_molecule), dgl.unbatch(rdkit_reference), dgl.unbatch(A_batch)):
        gx = G.ndata['x_cc'].cpu().detach().numpy()
        rx = R.ndata['x_ref'].cpu().detach().numpy()
        tx = T.ndata['x_ref'].cpu().detach().numpy()

        Rot, trans = rigid_transform_Kabsch_3D(rx.T, gx.T)
        lig_coords = ((Rot @ (rx).T).T + trans.squeeze())
        print('kabsch RMSD between rdkit ligand and Generated ligand is ', np.sqrt(np.sum((lig_coords - gx) ** 2, axis=1).mean()).item())

        Rot, trans = rigid_transform_Kabsch_3D(rx.T, tx.T)
        lig_coords = ((Rot @ (rx).T).T + trans.squeeze())
        print('kabsch RMSD between rdkit ligand and True ligand is ', np.sqrt(np.sum((lig_coords - tx) ** 2, axis=1).mean()).item())

        Rot, trans = rigid_transform_Kabsch_3D(tx.T, gx.T)
        lig_coords = ((Rot @ (tx).T).T + trans.squeeze())
        print('kabsch RMSD between True ligand and Generated ligand is ', np.sqrt(np.sum((lig_coords - gx) ** 2, axis=1).mean()).item())
        print()
    print("Decoder Autoregressive Preliminary Test Success")
