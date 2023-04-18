from utils.equivariant_model_utils import *
from utils.geometry_utils import *
from decoder_layers import IEGMN_Bidirectional
from collections import defaultdict
import numpy as np
import copy
from torch import nn
import torch.nn.functional as F
import dgl
from utils.neko_fixed_attention import neko_MultiheadAttention

import ipdb

class Decoder(nn.Module):
    def __init__(self, atom_embedder, n_lays, debug, device, shared_layers, noise_decay_rate, cross_msgs, noise_initial,
                 use_edge_features_in_gmn, use_mean_node_features, atom_emb_dim, latent_dim, coord_F_dim,
                 dropout, nonlin, leakyrelu_neg_slope, random_vec_dim=0, random_vec_std=1, use_scalar_features=True,
                 num_A_feats=None, save_trajectories=False, weight_sharing = True, conditional_mask=False, **kwargs):
        super(Decoder, self).__init__()
        # self.mha = torch.nn.MultiheadAttention(embed_dim = 3, num_heads = 1, batch_first = True)
        self.mha = neko_MultiheadAttention(embed_dim = 3, num_heads = 1, batch_first = True)
        self.iegmn = IEGMN_Bidirectional(n_lays, debug, device, shared_layers, noise_decay_rate, cross_msgs, noise_initial,
                 use_edge_features_in_gmn, use_mean_node_features, atom_emb_dim, latent_dim, coord_F_dim,
                 dropout, nonlin, leakyrelu_neg_slope, random_vec_dim, random_vec_std, use_scalar_features,
                 save_trajectories, weight_sharing, conditional_mask, **kwargs).cuda() #iegmn
        self.atom_embedder = atom_embedder # taken from the FG encoder
        D, F = latent_dim, coord_F_dim
        self.h_channel_selection = Scalar_MLP(D, 2*D, D)
        self.mse = nn.MSELoss()
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
        self.device = device
        # self.feed_forward_V = Vector_MLP(F, 2*F, 2*F, F, leaky = False, use_batchnorm = False)
        self.feed_forward_V = nn.Sequential(VNLinear(F, 2*F), VN_MLP(2*F, F, F, F, leaky = False, use_batchnorm = False))
        self.feed_forward_h = Scalar_MLP(D, 2*D, D)

        # self.feed_forward_V_3 = Vector_MLP(3, F, F, 3, leaky = False, use_batchnorm = False)
        self.feed_forward_V_3 = nn.Sequential(VNLinear(3, F), VN_MLP(F, 3, 3, 3, leaky = False, use_batchnorm = False))
        self.feed_forward_h_3 = Scalar_MLP(D, 2*D, D)
        self.teacher_forcing = kwargs['teacher_forcing']
    
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
        Z_h = A_cg.ndata["Z_h"] # N x D
        N, F, _ = Z_V.shape
        _, D = Z_h.shape
        print("[CC] V input", torch.min(Z_V).item(), torch.max(Z_V).item())
        print("[CC] h input", torch.min(Z_h).item(), torch.max(Z_h).item())
        # ipdb.set_trace()
        Z_V_ff = self.feed_forward_V(Z_V)
        # Z_h_ff = self.feed_forward_h(Z_h)
        print("[CC] V input FF", torch.min(Z_V_ff).item(), torch.max(Z_V_ff).item())
        # print("[CC] h input FF", torch.min(Z_h_ff).item(), torch.max(Z_h_ff).item())
        Z_V = Z_V_ff + Z_V
        # Z_h = Z_h_ff + Z_h
        print("[CC] V input FF add", torch.min(Z_V).item(), torch.max(Z_V).item())
        # print("[CC] h input FF add", torch.min(Z_h).item(), torch.max(Z_h).item())
        Z_V = self.eq_norm(Z_V)
        # Z_h = self.inv_norm(Z_h)
        print("[CC] V add norm", torch.min(Z_V).item(), torch.max(Z_V).item())
        # print("[CC] h add norm", torch.min(Z_h).item(), torch.max(Z_h).item())
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
        print("[CC] V cc attn update", torch.min(x_cc).item(), torch.max(x_cc).item())
        # print("[CC] h cc mpnn update", torch.min(h_cc).item(), torch.max(h_cc).item())
        x_cc_ff = self.feed_forward_V_3(x_cc.unsqueeze(2)).squeeze(2)
        # h_cc_ff = self.feed_forward_h_3(h_cc)
        print("[CC] V input FF 3", torch.min(x_cc_ff).item(), torch.max(x_cc_ff).item())
        # print("[CC] h input FF 3", torch.min(h_cc_ff).item(), torch.max(h_cc_ff).item())
        x_cc = x_cc_ff + x_cc
        # h_cc = h_cc_ff + h_cc
        print("[CC] V cc add 2", torch.min(x_cc).item(), torch.max(x_cc).item())
        # print("[CC] h cc add 2", torch.min(h_cc).item(), torch.max(h_cc).item())
        x_cc = self.eq_norm_2(x_cc.unsqueeze(2)).squeeze(2)
        # h_cc = self.inv_norm_2(h_cc)
        # print("same") #! the above makes it worse for some reason for bn
        h_cc = self.atom_embedder(B.ndata['ref_feat']) #! testing
        print("[CC] V cc add norm --> final", torch.min(x_cc).item(), torch.max(x_cc).item())
        print("[CC] h cc add norm --> final", torch.min(h_cc).item(), torch.max(h_cc).item())
        print()
        return x_cc, h_cc # we have selected the features for all coarse grain beads in parallel


    def autoregressive_step(self, latent, prev = None, t = 0, geo_latent = None, geo_current = None):
        # ipdb.set_trace()
        coords_A, h_feats_A, coords_B, h_feats_B, geom_losses, full_trajectory = self.iegmn(latent, prev, mpnn_only = t==0, geometry_graph_A = geo_latent, geometry_graph_B = geo_current, teacher_forcing=self.teacher_forcing, atom_embedder=self.atom_embedder)
        # coords_A held in latent.ndata['x_now']
        return  coords_A, h_feats_A, coords_B, h_feats_B, geom_losses, full_trajectory
    
    def sort_ids(self, all_ids, all_order):
        result = []
        for ids, order in zip(all_ids, all_order):
            sorted_frags = list(zip(ids, order))
            sorted_frags.sort(key = lambda x: x[1])
            frag_ids = [x for x, y in sorted_frags]
            result.append(frag_ids)
        return result

    def isolate_next_subgraph(self, final_molecule, id_batch, true_geo_batch):
        molecules = dgl.unbatch(final_molecule)
        geos = dgl.unbatch(true_geo_batch)
        result = []
        geo_result = []
        # valid = []
        # check = True
        for idx, ids in enumerate(id_batch):
            if ids is None:
                # valid.append((idx, False))
                # check = False
                continue
            # else:
                # valid.append((idx, True))
            fine = molecules[idx]
            subg = dgl.node_subgraph(fine, ids)
            result.append(subg)

            subgeo = dgl.node_subgraph(geos[idx], ids)
            geo_result.append(subgeo)
        return dgl.batch(result).to(self.device), dgl.batch(geo_result).to(self.device) #valid, check

    
    def gather_current_molecule(self, final_molecule, current_molecule_ids, progress, true_geo_batch):
        if current_molecule_ids is None or len(current_molecule_ids) == 0: #or torch.sum(progress) == 0
            return None, None
        if torch.sum(progress) == 0:
            return final_molecule, true_geo_batch
        molecules = dgl.unbatch(final_molecule)
        geos = dgl.unbatch(true_geo_batch)
        result = []
        geo_result = []
        for idx, ids in enumerate(current_molecule_ids):
            if progress[idx] == 0:
                continue
            fine = molecules[idx]
            subg = dgl.node_subgraph(fine, ids)
            result.append(subg)

            subgeo = dgl.node_subgraph(geos[idx], ids)
            geo_result.append(subgeo)
        return dgl.batch(result).to(self.device), dgl.batch(geo_result).to(self.device)

    def update_molecule(self, final_molecule, id_batch, coords_A, h_feats_A, latent):
        num_nodes = final_molecule.batch_num_nodes()
        start = 0
        prev = 0
        for idx, ids in enumerate(id_batch):
            if ids is None:
                start += num_nodes[idx]
                continue
            updated_ids = [start + x for x in ids]
            start += num_nodes[idx]
            cids = [prev + i for i in range(len(ids))]
            prev += len(ids)

            final_molecule.ndata['x_cc'][updated_ids, :] = coords_A[cids, :]
            final_molecule.ndata['feat_cc'][updated_ids, :] = h_feats_A[cids, :]

    def add_reference(self, ids, refs, progress):
        batch_idx = 0
        for atom_ids, ref_list in zip(ids, refs):
            for idx, bead in enumerate(atom_ids):
                # print(idx, bead)
                if len(bead) == 1:
                    print("\n start", atom_ids)
                    bead.add(int(ref_list[idx].item()))
                    print("update with reference", atom_ids)
                    progress[batch_idx] += 1
            batch_idx += 1
        lens = [sum([len(y) for y in x]) for x in ids]
        check = all([a-b.item() == 0 for a, b in zip(lens,progress)])
        # ipdb.set_trace()
        assert(check)
        return ids, progress

    def distance_loss(self, generated_coords_all, geometry_graphs, true_coords = None):
        geom_loss = []
        for geometry_graph, generated_coords in zip(dgl.unbatch(geometry_graphs), generated_coords_all):
            src, dst = geometry_graph.edges()
            src = src.long()
            dst = dst.long()
            d_squared = torch.sum((generated_coords[src] - generated_coords[dst]) ** 2, dim=1)
            geom_loss.append(1/len(src) * torch.sum((d_squared - geometry_graph.edata['feat'] ** 2) ** 2))
            # geom_loss.append(torch.sum((d_squared - geometry_graph.edata['feat'] ** 2) ** 2))
        print("          [AR Distance Loss Step]", geom_loss)
        # if true_coords is not None:
        #     for a, b, c, d in zip (generated_coords_all, geom_loss, true_coords, dgl.unbatch(geometry_graphs)):
        #         print("          Aligned MSE", a.shape, self.rmsd(a, c, align = True))
        #         print("          Gen", a)
        #         print("          True", c)
        #         print("          distance", b)
        #         print("          edges", d.edges())
        return torch.mean(torch.tensor(geom_loss))

    def align(self, source, target):
        # Rot, trans = rigid_transform_Kabsch_3D_torch(input.T, target.T)
        # lig_coords = ((Rot @ (input).T).T + trans.squeeze())
        # Kabsch RMSD implementation below taken from EquiBind
        # if source.shape[0] == 2:
        #     return align_sets_of_two_points(target, source) #! Kabsch seems to work better and it is ok
        lig_coords_pred = target
        lig_coords = source
        if source.shape[0] == 1:
            return source
        lig_coords_pred_mean = lig_coords_pred.mean(dim=0, keepdim=True)  # (1,3)
        lig_coords_mean = lig_coords.mean(dim=0, keepdim=True)  # (1,3)

        A = (lig_coords_pred - lig_coords_pred_mean).transpose(0, 1) @ (lig_coords - lig_coords_mean)+1e-7 #added noise to help with gradients

        U, S, Vt = torch.linalg.svd(A)

        corr_mat = torch.diag(torch.tensor([1, 1, torch.sign(torch.det(A))], device=lig_coords_pred.device))
        rotation = (U @ corr_mat) @ Vt
        translation = lig_coords_pred_mean - torch.t(rotation @ lig_coords_mean.t())  # (1,3)
        return (rotation @ lig_coords.t()).t() + translation
        # return lig_coords
    
    def rmsd(self, generated, true, align = False):
        if align:
            true = self.align(true, generated)
        loss = self.mse(true, generated)
        return loss

    def forward(self, cg_mol_graph, rdkit_mol_graph, cg_frag_ids, true_geo_batch):
        # ipdb.set_trace()
        rdkit_reference = copy.deepcopy(rdkit_mol_graph)
        X_cc, H_cc = self.channel_selection(cg_mol_graph, rdkit_mol_graph, cg_frag_ids)
        rdkit_mol_graph.ndata['x_cc'] = X_cc
        rdkit_mol_graph.ndata['feat_cc'] = H_cc
        bfs = cg_mol_graph.ndata['bfs'].flatten()
        ref = cg_mol_graph.ndata['bfs_reference_point'].flatten()
        X_cc = copy.deepcopy(X_cc.detach())
        H_cc = copy.deepcopy(H_cc.detach())
        # ipdb.set_trace()
        bfs_order = []
        ref_order = []
        start = 0
        for i in cg_mol_graph.batch_num_nodes():
            bfs_order.append(bfs[start : start + i])
            ref_order.append(ref[start: start + i])
            start += i
        # ipdb.set_trace()
        progress = rdkit_mol_graph.batch_num_nodes().cpu()
        # print("progress", progress)
        final_molecule = rdkit_mol_graph
        frag_ids = self.sort_ids(cg_frag_ids, bfs_order)
        # print(frag_ids)
        frag_ids, progress = self.add_reference(frag_ids, ref_order, progress) #done
        # ipdb.set_trace()
        frag_batch = defaultdict(list) # keys will be time steps
        # TODO: build similar object for reference atoms as they are already in BFS order
        # TODO: does this work or do we need to sort them like hte BFS
        # TODO: print out what the references look like adn the frag ids after sorting to make sure they line up and look realistic
        max_nodes = max(cg_mol_graph.batch_num_nodes()).item()
        for t in range(max_nodes):
            for idx, frag in enumerate(frag_ids): # iterate over moelcules
                ids = None
                if t < len(frag):
                    ids = list(frag[t])
                frag_batch[t].append(ids)

        # ipdb.set_trace()
        current_molecule_ids = None
        current_molecule = None
        geo_current = None
        returns = []
        for t in range(max_nodes):
            # ipdb.set_trace()
            print("[Auto Regressive Step]")
            id_batch = frag_batch[t]
            # print("ID", id_batch)
            latent, geo_latent = self.isolate_next_subgraph(final_molecule, id_batch, true_geo_batch)
            # if not check:
            #     current_molecule = self.adaptive_batching(current_molecule)
            # ipdb.set_trace()
            coords_A, h_feats_A, coords_B, h_feats_B, geom_losses, full_trajectory = self.autoregressive_step(latent, current_molecule, t, geo_latent, geo_current)
            ref_coords_A = latent.ndata['x_true']
            ref_coords_B = current_molecule.ndata['x_true'] if current_molecule is not None else None
            ref_coords_B_split = [x.ndata['x_true'] for x in dgl.unbatch(current_molecule)] if current_molecule is not None else None
            model_predicted_B = current_molecule.ndata['x_cc'] if current_molecule is not None else None
            model_predicted_B_split = [x.ndata['x_cc'] for x in dgl.unbatch(current_molecule)] if current_molecule is not None else None
            gen_input = latent.ndata['x_cc']
            print("[AR step end] geom losses total from decoder", t, geom_losses)
            dist_loss = self.distance_loss([x.ndata['x_cc'] for x in dgl.unbatch(latent)], geo_latent, [x.ndata['x_true'] for x in dgl.unbatch(latent)])
            print()
            returns.append((coords_A, h_feats_A, coords_B, h_feats_B, geom_losses, full_trajectory, id_batch, ref_coords_A, ref_coords_B, ref_coords_B_split, gen_input, model_predicted_B, model_predicted_B_split, dist_loss))
            # print("ID Check", returns[0][6])
            # print(f'{t} A MSE = {torch.mean(torch.sum(ref_coords_A - coords_A, dim = 1)**2)}')
            # if ref_coords_B is not None:
            # if torch.gt(coords_A, 1000).any() or  torch.lt(coords_A, -1000).any() or (coords_B is not None and torch.gt(coords_B, 1000).any()) or (coords_B is not None and torch.lt(coords_B, -1000).any()):
            #     # ipdb.set_trace()
            #     print("Caught Explosion in Decode step")
                # X_cc, H_cc = self.channel_selection(cg_mol_graph, rdkit_mol_graph, cg_frag_ids)
                # coords_A, h_feats_A, coords_B, h_feats_B, geom_losses, full_trajectory = self.autoregressive_step(latent, current_molecule, t, geo_latent, geo_current)

                # print(f'{t} B MSE = {torch.mean(torch.sum(ref_coords_B - coords_B, dim = 1)**2)}')
            self.update_molecule(final_molecule, id_batch, coords_A, h_feats_A, latent)
            progress -= torch.tensor([len(x) if x is not None else 0 for x in id_batch])
            if t == 0:
                current_molecule_ids = copy.deepcopy(id_batch)
            else:
                for idx, ids in enumerate(id_batch):
                    if ids is None:
                        continue
                    ids = [x for x in ids if x not in set(current_molecule_ids[idx])] #! added to prevent overlap of reference
                    current_molecule_ids[idx].extend(ids)
            # ipdb.set_trace() # erroring on dgl.unbatch(final_molecule) for some odd reason --> fixed by adding an else above
            current_molecule, geo_current = self.gather_current_molecule(final_molecule, current_molecule_ids, progress, true_geo_batch)

        return final_molecule, rdkit_reference, returns, (X_cc, H_cc)

