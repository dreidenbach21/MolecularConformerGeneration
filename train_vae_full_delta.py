import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="6"
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
from vae_delta import *
from decoder_utils_delta import *
from geometry_utils import *
from torsional_diffusion_data import load_torsional_data, QM9_DIMS, DRUG_DIMS

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
# with open(geom_path + "qm9_safe_v2.pickle", 'rb') as f:
#     qm9 = pickle.load(f)
batch_size = 25 #25
train_limit = 100 #100
train_loader, train_data = load_torsional_data(batch_size = batch_size, mode= 'train', limit_mols = train_limit)
val_limit = 25 #25
val_loader, val_data = load_torsional_data(batch_size = batch_size, mode= 'val', limit_mols = val_limit)
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

if __name__ =="__main__":
  
  # frag_ids, A_batch, B_batch, geoA_batch, geoB_batch, Ap_batch, Bp_batch, Acg_batch, Bcg_batch, geoAcg_batch, geoBcg_batch = dummy_data_loader()
  edge_dim = 20#15 #A_batch.edata['feat'].shape[1]

  ecn = ECN3D(device = "cuda", **ecn_model_params).cuda()
  atom_embedder = ecn.atom_embedder
  ecn_model_params['layer_norm'] = 'LN' # Have to switch for AR #! EB used BN but we switched all to LN
  iegmn = IEGMN_Bidirectional_Delta(device = "cuda", **ecn_model_params).cuda()
  F = ecn_model_params["coord_F_dim"]
  D = ecn_model_params["latent_dim"]
  model = VAE_Delta(ecn, D, F, iegmn, atom_embedder, device = "cuda")
  print("CUDA CHECK", next(model.parameters()).is_cuda)
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
  torch.autograd.set_detect_anomaly(True)
  # train_loss_log_name = "torsional_diffusion_test_geomol_1000_minpostclamp" + "_train"
  # val_loss_log_name = "torsional_diffusion_test_geomol2_1000_minpostclamp" + "_val"
  train_loss_log_name = "n1_ref_test_dist_delta" + "_train"
  val_loss_log_name = "n1_ref_test_dist_delta" + "_val"
  train_loss_log_total, val_loss_log_total = [], []
  for epoch in range(10000):
    print("\n\n\n\n\nEpoch", epoch)
    train_loss_log, val_loss_log = [], []
    for A_batch, B_batch in train_loader:
      A_graph, geo_A, Ap, A_cg, geo_A_cg, frag_ids = A_batch
      B_graph, geo_B, Bp, B_cg, geo_B_cg = B_batch
      
      A_graph, geo_A, Ap, A_cg, geo_A_cg = A_graph.to('cuda:0'), geo_A.to('cuda:0'), Ap.to('cuda:0'), A_cg.to('cuda:0'), geo_A_cg.to('cuda:0')
      B_graph, geo_B, Bp, B_cg, geo_B_cg = B_graph.to('cuda:0'), geo_B.to('cuda:0'), Bp.to('cuda:0'), B_cg.to('cuda:0'), geo_B_cg.to('cuda:0')

      generated_molecule, rdkit_reference, dec_results, channel_selection_info, KL_terms, enc_out = model(frag_ids, A_graph, B_graph, geo_A, geo_B, Ap, Bp, A_cg, B_cg, geo_A_cg, geo_B_cg, epoch=epoch)
    # ipdb.set_trace()
      loss, losses = model.loss_function(generated_molecule, rdkit_reference, dec_results, channel_selection_info, KL_terms, enc_out, geo_A, step = epoch)
      train_loss_log.append(losses)
      print(f"Train LOSS = {loss}")
      loss.backward()

      for name, p in model.named_parameters(): 
        if p.requires_grad and p.grad is not None and (torch.isnan(p.grad).any() or torch.isnan(p.data).any()):
          print("[LOG]", name, torch.min(p.grad).item(), torch.max(p.grad).item(), torch.min(p.data).item(), torch.max(p.data).item())

      parameters = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
      norm_type = 2
      total_norm = 0.0
      for p in parameters:
        param_norm = p.grad.detach().data.norm(2)
        total_norm += param_norm.item() ** 2
      total_norm = total_norm ** 0.5
      print(f"Train LOSS = {loss}")
      print("TOTAL GRADIENT NORM", total_norm)

      torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1000, norm_type=2) #500
      parameters = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
      norm_type = 2
      total_norm = 0.0
      for p in parameters:
        param_norm = p.grad.detach().data.norm(2)
        total_norm += param_norm.item() ** 2
      total_norm = total_norm ** 0.5
      print(f"Train LOSS = {loss}")
      print("CLIPPED TOTAL GRADIENT NORM", total_norm)

      optim.step()
      optim.zero_grad()

    train_loss_log_total.append(train_loss_log)
    with open(f'./{train_loss_log_name}.pkl', 'wb') as f:
      pickle.dump(train_loss_log_total, f)
    
    print("\n\n\n\n\n Validation")
    with torch.no_grad():
      model.flip_teacher_forcing()
      for A_batch, B_batch in val_loader:
        A_graph, geo_A, Ap, A_cg, geo_A_cg, frag_ids = A_batch
        B_graph, geo_B, Bp, B_cg, geo_B_cg = B_batch

        A_graph, geo_A, Ap, A_cg, geo_A_cg = A_graph.to('cuda:0'), geo_A.to('cuda:0'), Ap.to('cuda:0'), A_cg.to('cuda:0'), geo_A_cg.to('cuda:0')
        B_graph, geo_B, Bp, B_cg, geo_B_cg = B_graph.to('cuda:0'), geo_B.to('cuda:0'), Bp.to('cuda:0'), B_cg.to('cuda:0'), geo_B_cg.to('cuda:0')

        generated_molecule, rdkit_reference, dec_results, channel_selection_info, KL_terms, enc_out = model(frag_ids, A_graph, B_graph, geo_A, geo_B, Ap, Bp, A_cg, B_cg, geo_A_cg, geo_B_cg, epoch=epoch)
      # ipdb.set_trace()
        loss, losses = model.loss_function(generated_molecule, rdkit_reference, dec_results, channel_selection_info, KL_terms, enc_out, geo_A, step = epoch)
        val_loss_log.append(losses)
        print(f"Val LOSS = {loss}")
      val_loss_log_total.append(val_loss_log)
      model.flip_teacher_forcing()
      with open(f'./{val_loss_log_name}.pkl', 'wb') as f:
        pickle.dump(val_loss_log_total, f)

    # frag_ids, A_batch, B_batch, geoA_batch, geoB_batch, Ap_batch, Bp_batch, Acg_batch, Bcg_batch, geoAcg_batch, geoBcg_batch = dummy_data_loader()
  print("Training Complete")
  # with open(f'./{train_loss_log_name}.pkl', 'wb') as f:
  #     pickle.dump(train_loss_log_total, f)
