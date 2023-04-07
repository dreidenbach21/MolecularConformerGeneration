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
  "geom_reg_steps":1,
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
  "layer_norm": 'BN', # ['0', 'BN', 'LN'] # BN in good run #TODO: batch norm has issues with only 1 sample during training only
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
  "weight_sharing":False,
}

geom_path = "/home/dreidenbach/data/GEOM/rdkit_folder/"
qm9_path = geom_path + "qm9/"
drugs_path = geom_path + "drugs/"
print("Loading QM9...")
with open(geom_path + "qm9_safe_v2.pickle", 'rb') as f:
    qm9 = pickle.load(f)
print("Loading QM9 --> Done")

# #optimizer: Adam
# optimizer_params:
#   lr: 1.0e-4
#   weight_decay: 1.0e-4 # 1.0e-5 in good run
# clip_grad: 100 # leave empty for no grad clip

# scheduler_step_per_batch: False
# lr_scheduler:  ReduceLROnPlateau # leave empty to use none
# lr_scheduler_params:
#   factor: 0.6
#   patience: 60
#   min_lr: 8.0e-6
#   mode: 'max'
#   verbose: True

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
  
def dummy_data_loader():
  smiles = ['CC[C@@](C)(O)[C@@H](C)OC',
            'C[C@@]12COCC=C[C@@H]1C2',
            'C[C@@]12C[C@H]3CN1[C@]32CO',
            'CO[C@H]1C[C@H]2C[C@@]2(C)C1',
            # 'C1=C[C@@]23C[C@H]4CN2[C@@H]1[C@H]43',
            'O[C@H]1CC[C@H]2CC=C[C@H]21',
            'CC[C@@H]1[C@H](O)C=C[C@@H]1O',
            'O[C@@H]1[C@H](O)[C@@H]2C[C@H]12',
            'CC[C@H]1OC[C@@H](C)[C@H]1O',
            'O=C[C@@]12O[C@H]3[C@@H]4[C@@H]([C@@H]41)[C@H]32',
            # 'C1OC[C@H]2[C@H]3C[C@H]3[C@H]2O1',
            'C[C@H]1[C@H]2C[C@H]3C(=O)[C@@H]1N23',
            'N#C[C@@H]1N=C(N)[C@@H]2O[C@H]12',
            'c1coc([C@H]2CN2)n1',
            'O=C[C@@H]1[C@@H]2C[C@H]3[C@@H]2[C@@]13O',
            'C#Cc1cc(C)[nH]c1N',
            'CCC#CC[C@H]1C[C@H]1C',
            'CCC1CC(C)(C)C1',
            'C[C@@]12C=CC[C@]1(C=O)O2',
            'CO[C@@]1(CCO)C[C@H]1C']
  data = []
  frag_ids = []
  # bond_breaks = []
  for smi in smiles:
    rd_mol = qm9[smi]["conformers"][0]["rd_mol"]
    kit_mol = copy.deepcopy(rd_mol)
    A,B,geometry_graph_A,geometry_graph_B, Ap, Bp, A_cg, B_cg, geometry_graph_A_cg, geometry_graph_B_cg, A_ids, A_bond_break = init_graph_match(rd_mol, kit_mol, D = ecn_model_params["latent_dim"], F = ecn_model_params["coord_F_dim"])
    data.append((A,B,geometry_graph_A,geometry_graph_B, Ap, Bp, A_cg, B_cg, geometry_graph_A_cg, geometry_graph_B_cg))
    frag_ids.append(A_ids)
    # bond_breaks.append(A_bond_break)
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
  return frag_ids, A_batch, B_batch, geoA_batch, geoB_batch, Ap_batch, Bp_batch, Acg_batch, Bcg_batch, geoAcg_batch, geoBcg_batch

if __name__ =="__main__":
  
  # frag_ids, A_batch, B_batch, geoA_batch, geoB_batch, Ap_batch, Bp_batch, Acg_batch, Bcg_batch, geoAcg_batch, geoBcg_batch = dummy_data_loader()
  edge_dim = 15 #A_batch.edata['feat'].shape[1]

  ecn = ECN3D(device = "cuda", #cpu
          A_input_edge_feats_dim = edge_dim,
          B_input_edge_feats_dim = edge_dim, 
          **ecn_model_params).cuda()
  # encoder = Encoder(ecn, ecn_model_params["latent_dim"], ecn_model_params["coord_F_dim"]).cuda()
  atom_embedder = ecn.atom_embedder
  ecn_model_params['layer_norm'] = 'LN' # Have to switch for AR #! EB used BN but we switched all to LN
  iegmn = IEGMN_Bidirectional(device = "cuda", #cpu
            A_input_edge_feats_dim = edge_dim,
            B_input_edge_feats_dim = edge_dim, 
            **ecn_model_params).cuda()
  # decoder =  Decoder(iegmn, ecn_model_params["latent_dim"], ecn_model_params["coord_F_dim"], model.atom_embedder).cuda()
  F = ecn_model_params["coord_F_dim"]
  D = ecn_model_params["latent_dim"]
  model = VAE(ecn, D, F, iegmn, atom_embedder, device = "cuda")
  print("# of Encoder Params = ", sum(p.numel() for p in model.encoder.parameters() if p.requires_grad))
  print("# of Decoder Params = ", sum(p.numel() for p in model.decoder.parameters() if p.requires_grad))
  print("# of Atom Embedder Params = ", sum(p.numel() for p in atom_embedder.parameters() if p.requires_grad))
  print("# of VAE Params = ", sum(p.numel() for p in model.parameters() if p.requires_grad))
  optim = torch.optim.AdamW(model.parameters(), lr=1e-3)#3) #! 1e-2 was giving us nan blow ups for 3 smiles look into gradient clipping
  # loss.backward() #! EquiBIND
  # if self.args.clip_grad != None:
  #     torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.args.clip_grad, norm_type=2)
  # self.optim.step()
  # self.after_optim_step()  # overwrite this function to do stuff before zeroing out grads
  # self.lr_scheduler.total_warmup_steps > self.lr_scheduler._step)):  # step per batch if that is what we want to do or if we are using a warmup schedule and are still in the warmup period
  #         self.step_schedulers() --> self.lr_scheduler.step()
  # self.optim.zero_grad()
  # self.optim_steps += 1
  # torch.autograd.set_detect_anomaly(True)
  loss_log_name = "train_vae_losses_test_bn_ln_ar_522"
  loss_log = []
  for epoch in range(10000):
    print("Forward Pass", epoch)
    # ipdb.set_trace()
    frag_ids, A_batch, B_batch, geoA_batch, geoB_batch, Ap_batch, Bp_batch, Acg_batch, Bcg_batch, geoAcg_batch, geoBcg_batch = dummy_data_loader()
    generated_molecule, rdkit_reference, dec_results, channel_selection_info, KL_terms, enc_out = model(frag_ids, A_batch, B_batch, geoA_batch, geoB_batch, Ap_batch, Bp_batch, Acg_batch, Bcg_batch, geoAcg_batch, geoBcg_batch, epoch=epoch)
    loss, losses = model.loss_function(generated_molecule, rdkit_reference, dec_results, channel_selection_info, KL_terms, enc_out, step = epoch)
    loss_log.append(losses)
    # if torch.isnan(loss).any():
    #   ipdb.set_trace()
    #   generated_molecule, rdkit_reference, dec_results, KL_terms, enc_out = model(frag_ids, A_batch, B_batch, geoA_batch, geoB_batch, Ap_batch, Bp_batch, Acg_batch, Bcg_batch, geoAcg_batch, geoBcg_batch, epoch=epoch)
    #   loss, losses = model.loss_function(generated_molecule, rdkit_reference, dec_results, KL_terms, step = epoch)
    print(f"LOSS = {loss}")
    loss.backward()
    # for name, p in model.named_parameters(): 
    #   if p.requires_grad and p.grad is not None:
    #     # ipdb.set_trace()
    #     print("LOG", name, torch.min(p.grad).item(), torch.max(p.grad).item(), torch.min(p.data).item(), torch.max(p.data).item())
    #     if "mean" in name or "logvar" in name:
    #       print(p.data)
    # print(f"LOSS = {loss}")
    parameters = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
    norm_type = 2
    total_norm = 0.0
    for p in parameters:
      param_norm = p.grad.detach().data.norm(2)
      total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    print(f"LOSS = {loss}")
    print("TOTAL GRADIENT NORM", total_norm)

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=500, norm_type=2)
    parameters = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
    norm_type = 2
    total_norm = 0.0
    for p in parameters:
      param_norm = p.grad.detach().data.norm(2)
      total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    print(f"LOSS = {loss}")
    print("CLIPPED TOTAL GRADIENT NORM", total_norm)

    optim.step()
    optim.zero_grad()

    if epoch > 0 and epoch % 100 == 0:
      for G, R, T in zip(dgl.unbatch(generated_molecule), dgl.unbatch(rdkit_reference), dgl.unbatch(A_batch)):
        gx = G.ndata['x_cc'].cpu().detach().numpy()
        rx = R.ndata['x_ref'].cpu().detach().numpy()
        # tx = T.ndata['x_ref'].cpu().detach().numpy()
        tx = T.ndata['x_true'].cpu().detach().numpy()
        # Rot, trans = rigid_transform_Kabsch_3D(rx.T, gx.T)
        # lig_coords = ((Rot @ (rx).T).T + trans.squeeze())
        # print('RMSD between rdkit ligand and Generated ligand is ', np.sqrt(np.sum((rx - gx) ** 2, axis=1).mean()).item())
        # print('kabsch RMSD between rdkit ligand and Generated ligand is ', np.sqrt(np.sum((lig_coords - gx) ** 2, axis=1).mean()).item())
        # Rot, trans = rigid_transform_Kabsch_3D(rx.T, tx.T)
        # lig_coords = ((Rot @ (rx).T).T + trans.squeeze())
        # print('RMSD between rdkit ligand and True ligand is ', np.sqrt(np.sum((rx - tx) ** 2, axis=1).mean()).item())
        # print('kabsch RMSD between rdkit ligand and True ligand is ', np.sqrt(np.sum((lig_coords - tx) ** 2, axis=1).mean()).item())
        Rot, trans = rigid_transform_Kabsch_3D(tx.T, gx.T)
        lig_coords = ((Rot @ (tx).T).T + trans.squeeze())
        print('RMSD between True ligand and Generated ligand is ', np.sqrt(np.sum((tx - gx) ** 2, axis=1).mean()).item())
        print('kabsch RMSD between True ligand and Generated ligand is ', np.sqrt(np.sum((lig_coords - gx) ** 2, axis=1).mean()).item())
    print()
    print("Reloading Data")
    if epoch > 0 and epoch %10 == 0:
      with open(f'./{loss_log_name}.pkl', 'wb') as f:
        pickle.dump(loss_log, f)
    # frag_ids, A_batch, B_batch, geoA_batch, geoB_batch, Ap_batch, Bp_batch, Acg_batch, Bcg_batch, geoAcg_batch, geoBcg_batch = dummy_data_loader()
  print("Training Complete")
  with open(f'./{loss_log_name}.pkl', 'wb') as f:
    pickle.dump(loss_log, f)
