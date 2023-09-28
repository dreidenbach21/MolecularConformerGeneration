from utils.equivariant_model_utils import *
from utils.geometry_utils import *
from decoder_layers import IEGMN_Bidirectional
from decoder_delta_layers import IEGMN_Bidirectional_Delta
from decoder_double_delta_layers import IEGMN_Bidirectional_Double_Delta
from collections import defaultdict
import numpy as np
import copy
from torch import nn
import torch.nn.functional as F
import dgl
from tqdm import tqdm
from utils.neko_fixed_attention import neko_MultiheadAttention

import ipdb

class DecoderNoAR(nn.Module):
    def __init__(self, atom_embedder, coordinate_type, n_lays, debug, device, shared_layers, noise_decay_rate, cross_msgs, noise_initial,
                 use_edge_features_in_gmn, use_mean_node_features, atom_emb_dim, latent_dim, coord_F_dim,
                 dropout, nonlin, leakyrelu_neg_slope, random_vec_dim=0, random_vec_std=1, use_scalar_features=True,
                 num_A_feats=None, save_trajectories=False, weight_sharing = True, conditional_mask=False, verbose = False, **kwargs):
        super(DecoderNoAR, self).__init__()
        # self.mha = torch.nn.MultiheadAttention(embed_dim = 3, num_heads = 1, batch_first = True)
        self.verbose = verbose
        self.mha = neko_MultiheadAttention(embed_dim = 3, num_heads = 1, batch_first = True)
        self.double = False
        self.device = device
        if coordinate_type == "delta":
            self.iegmn = IEGMN_Bidirectional_Delta(n_lays, debug, device, shared_layers, noise_decay_rate, cross_msgs, noise_initial,
                 use_edge_features_in_gmn, use_mean_node_features, atom_emb_dim, latent_dim, coord_F_dim,
                 dropout, nonlin, leakyrelu_neg_slope, random_vec_dim, random_vec_std, use_scalar_features,
                 save_trajectories, weight_sharing, conditional_mask, **kwargs).cuda() #iegmn
        # elif coordinate_type == "double":
        #     self.double = True #True
        #     self.iegmn = IEGMN_Bidirectional_Double_Delta(n_lays, debug, device, shared_layers, noise_decay_rate, cross_msgs, noise_initial,
        #          use_edge_features_in_gmn, use_mean_node_features, atom_emb_dim, latent_dim, coord_F_dim,
        #          dropout, nonlin, leakyrelu_neg_slope, random_vec_dim, random_vec_std, use_scalar_features,
        #          save_trajectories, weight_sharing, conditional_mask, **kwargs).cuda() #iegmn
        # else:
        #     self.iegmn = IEGMN_Bidirectional(n_lays, debug, device, shared_layers, noise_decay_rate, cross_msgs, noise_initial,
        #             use_edge_features_in_gmn, use_mean_node_features, atom_emb_dim, latent_dim, coord_F_dim,
        #             dropout, nonlin, leakyrelu_neg_slope, random_vec_dim, random_vec_std, use_scalar_features,
        #             save_trajectories, weight_sharing, conditional_mask, **kwargs).cuda() #iegmn
        self.atom_embedder = atom_embedder # taken from the FG encoder
        D, F = latent_dim, coord_F_dim
        
        # self.h_channel_selection = Scalar_MLP(D, 2*D, D)
        # self.mse = nn.MSELoss()
        # norm = "ln"
        if kwargs['cc_norm'] == "bn":
            self.eq_norm = VNBatchNorm(F)
            self.inv_norm = nn.BatchNorm1d(D)
            self.eq_norm_2 = VNBatchNorm(3)
            self.inv_norm_2 = nn.BatchNorm1d(D)
        elif kwargs['cc_norm'] == "ln":
            self.eq_norm = VNLayerNorm(F)
            self.inv_norm = nn.LayerNorm(D)
            self.eq_norm_2 = VNLayerNorm(3)
            self.inv_norm_2 = nn.LayerNorm(D)
        else:
            assert(1 == 0)
        
        # self.feed_forward_V = Vector_MLP(F, 2*F, 2*F, F, leaky = False, use_batchnorm = False)
        self.feed_forward_V = nn.Sequential(VNLinear(F, 2*F), VN_MLP(2*F, F, F, F, leaky = False, use_batchnorm = False))
        # self.feed_forward_h = Scalar_MLP(D, 2*D, D)

        # self.feed_forward_V_3 = Vector_MLP(3, F, F, 3, leaky = False, use_batchnorm = False)
        self.feed_forward_V_3 = nn.Sequential(VNLinear(3, F), VN_MLP(F, 3, 3, 3, leaky = False, use_batchnorm = False))
        # self.feed_forward_h_3 = Scalar_MLP(D, 2*D, D)
        self.teacher_forcing = False #kwargs['teacher_forcing']
        self.mse_none = nn.MSELoss(reduction ='none')
    
    def get_node_mask(self, ligand_batch_num_nodes, receptor_batch_num_nodes, device):
        rows = ligand_batch_num_nodes.sum()
        cols = receptor_batch_num_nodes.sum()
        mask = torch.zeros(rows, cols, device=device)
        partial_l = 0
        partial_r = 0
        for l_n, r_n in zip(ligand_batch_num_nodes, receptor_batch_num_nodes):
            mask[partial_l: partial_l + l_n, partial_r: partial_r + r_n] = 1
            partial_l = partial_l + l_n
            partial_r = partial_r + r_n
        return mask

    def get_queries_and_mask(self, Acg, B, N, F, splits):
        # Each batch isa CG bead
        queries = []
        prev = 0
        info = list(Acg.ndata['cg_to_fg'].flatten().cpu().numpy().astype(int)) # This is used to help with the padding
        # for x in info:
        #     queries.append(B_batch.ndata['x'][prev:prev+ x, :])
        #     prev += x
        # import ipdb; ipdb.set_trace()
        splits = sum(splits, []) # flatten
        for x in splits:
            queries.append(B.ndata['x'][list(x), :])

        Q = torch.nn.utils.rnn.pad_sequence(queries, batch_first=True)
        n_max = Q.shape[1]
        attn_mask = torch.ones((N, n_max, F), dtype=torch.bool).to(self.device)
        for b in range(attn_mask.shape[0]):
            attn_mask[b, :info[b], :] = False
            attn_mask[b, info[b]:, :] = True
        return Q, attn_mask, info

    def channel_selection(self, A_cg, B, cg_frag_ids):
        # ! FF + Add Norm
        Z_V = A_cg.ndata["Z_V"]# N x F x 3
        # Z_h = A_cg.ndata["Z_h"] # N x D
        N, F, _ = Z_V.shape
        # _, D = Z_h.shape
        if self.verbose: print("[CC] V input", torch.min(Z_V).item(), torch.max(Z_V).item())
        # if self.verbose: print("[CC] h input", torch.min(Z_h).item(), torch.max(Z_h).item())
        Z_V_ff = self.feed_forward_V(Z_V)
        # Z_h_ff = self.feed_forward_h(Z_h)
        if self.verbose: print("[CC] V input FF", torch.min(Z_V_ff).item(), torch.max(Z_V_ff).item())
        # if self.verbose: print("[CC] h input FF", torch.min(Z_h_ff).item(), torch.max(Z_h_ff).item())
        Z_V = Z_V_ff + Z_V
        # Z_h = Z_h_ff + Z_h
        if self.verbose: print("[CC] V input FF add", torch.min(Z_V).item(), torch.max(Z_V).item())
        # if self.verbose: print("[CC] h input FF add", torch.min(Z_h).item(), torch.max(Z_h).item())
        Z_V = self.eq_norm(Z_V)
        # Z_h = self.inv_norm(Z_h)
        if self.verbose: print("[CC] V add norm", torch.min(Z_V).item(), torch.max(Z_V).item())
        # if self.verbose: print("[CC] h add norm", torch.min(Z_h).item(), torch.max(Z_h).item())
        # Equivariant Channel Selection
        Q, attn_mask, cg_to_fg_info = self.get_queries_and_mask(A_cg, B, N, F, cg_frag_ids)
        K = Z_V
        V = Z_V
        attn_out, attn_weights = self.mha(Q, K, V, attn_mask = attn_mask)
        res = []
        for idx, k in enumerate(cg_to_fg_info):
            res.append( attn_out[idx, :k, :]) # Pull from the parts that were not padding
        res = torch.cat(res, dim = 0)
        x_cc = res # n x 3
        # Invariant Channel Selection
        # h_og = B.ndata['rd_feat'] # n x d = 17
        # h_og_lifted = self.h_channel_selection(self.atom_embedder(h_og)) # n x D
        # h_cc = h_og_lifted + torch.repeat_interleave(Z_h, torch.tensor(cg_to_fg_info).to(self.device), dim = 0) # n x D
        # Second Add Norm
        if self.verbose: print("[CC] V cc attn update", torch.min(x_cc).item(), torch.max(x_cc).item())
        # if self.verbose: print("[CC] h cc mpnn update", torch.min(h_cc).item(), torch.max(h_cc).item())
        x_cc_ff = self.feed_forward_V_3(x_cc.unsqueeze(2)).squeeze(2)
        # h_cc_ff = self.feed_forward_h_3(h_cc)
        if self.verbose: print("[CC] V input FF 3", torch.min(x_cc_ff).item(), torch.max(x_cc_ff).item())
        # if self.verbose: print("[CC] h input FF 3", torch.min(h_cc_ff).item(), torch.max(h_cc_ff).item())
        x_cc = x_cc_ff + x_cc
        # h_cc = h_cc_ff + h_cc
        if self.verbose: print("[CC] V cc add 2", torch.min(x_cc).item(), torch.max(x_cc).item())
        # if self.verbose: print("[CC] h cc add 2", torch.min(h_cc).item(), torch.max(h_cc).item())
        x_cc = self.eq_norm_2(x_cc.unsqueeze(2)).squeeze(2)
        # h_cc = self.inv_norm_2(h_cc)
        # if self.verbose: print("same") #! the above makes it worse for some reason for bn
        h_cc = self.atom_embedder(B.ndata['ref_feat']) #! testing
        if self.verbose: print("[CC] V cc add norm --> final", torch.min(x_cc).item(), torch.max(x_cc).item())
        if self.verbose: print("[CC] h cc add norm --> final", torch.min(h_cc).item(), torch.max(h_cc).item())
        if self.verbose: print()
        return x_cc, h_cc # we have selected the features for all coarse grain beads in parallel


    def step(self, latent, prev, geo_latent = None, geo_current = None, no_conditioning = True):
        coords_A, h_feats_A, coords_B, h_feats_B = self.iegmn(latent, prev, mpnn_only = no_conditioning,
                                                                                            geometry_graph_A = geo_latent,
                                                                                            geometry_graph_B = geo_current,
                                                                                            teacher_forcing=False,
                                                                                            atom_embedder=self.atom_embedder)
        return  coords_A, h_feats_A, coords_B, h_feats_B

    # def align(self, source, target):
    #     with torch.no_grad():
    #         lig_coords_pred = target
    #         lig_coords = source
    #         if source.shape[0] == 1:
    #             return source
    #         lig_coords_pred_mean = lig_coords_pred.mean(dim=0, keepdim=True)  # (1,3)
    #         lig_coords_mean = lig_coords.mean(dim=0, keepdim=True)  # (1,3)

    #         A = (lig_coords_pred - lig_coords_pred_mean).transpose(0, 1) @ (lig_coords - lig_coords_mean)+1e-7 #added noise to help with gradients
    #         if torch.isnan(A).any() or torch.isinf(A).any():
    #             print(torch.max(A))
    #             import ipdb; ipdb.set_trace()
                
    #         U, S, Vt = torch.linalg.svd(A)

    #         corr_mat = torch.diag(torch.tensor([1, 1, torch.sign(torch.det(A))], device=lig_coords_pred.device))
    #         rotation = (U @ corr_mat) @ Vt
    #         translation = lig_coords_pred_mean - torch.t(rotation @ lig_coords_mean.t())  # (1,3)
    #     return (rotation @ lig_coords.t()).t() + translation
    
    def forward(self, cg_mol_graph, rdkit_mol_graph, cg_frag_ids, true_geo_batch, rd_geo_batch):
        rdkit_reference = copy.deepcopy(rdkit_mol_graph.to('cpu')).to('cuda:0') 
        X_cc, H_cc = self.channel_selection(cg_mol_graph, rdkit_mol_graph, cg_frag_ids)
        rdkit_mol_graph.ndata['x_cc'] = X_cc
        rdkit_mol_graph.ndata['feat_cc'] = H_cc
        
        latent = rdkit_mol_graph
        geo_latent = true_geo_batch

        #! Pass RDKit in so ref is RDKit and x_cc is the channel selection
        try:
            coords_A, h_feats_A, coords_B, h_feats_B= self.step(latent, rdkit_reference, geo_latent, rd_geo_batch, no_conditioning = False)
        except Exception as e:
            print(e)
            import ipdb; ipdb.set_trace()
            test = 1
            coords_A, h_feats_A, coords_B, h_feats_B= self.step(latent, rdkit_reference, geo_latent, rd_geo_batch, no_conditioning = False)
        ref_coords_A = latent.ndata['x_true']

        returns = (coords_A, h_feats_A, coords_B, h_feats_B, ref_coords_A, X_cc, H_cc)

        latent.ndata['x_cc'] = coords_A
        latent.ndata['feat_cc'] = h_feats_A
        return latent, rdkit_reference, returns

