from ecn_3d import *
from equivariant_model_utils import *
from decoder_utils import IEGMN_Bidirectional
from collections import defaultdict
import ipdb

class Encoder(nn.Module):
    def __init__(self, ecn, D, F):
        super(Encoder, self).__init__()
        self.ecn = ecn
        
        self.posterior_mean_V = Vector_MLP(2*F, 2*F, 2*F, F)
        self.posterior_mean_h = Scalar_MLP(2*D, 2*D, D)
        self.posterior_logvar_V = Scalar_MLP(2*F*3, 2*F*3, F)# need to flatten to get equivariant noise N x F x 1
        self.posterior_logvar_h = Scalar_MLP(2*D, 2*D, D)

        self.prior_mean_V = Vector_MLP(F,F,F,F)
        self.prior_mean_h = Scalar_MLP(D, D, D)
        self.prior_logvar_V = Scalar_MLP(F*3, F*3, F)
        self.prior_logvar_h = Scalar_MLP(D, D, D)

    def forward(self, A_graph, B_graph, geometry_graph_A, geometry_graph_B, A_pool, B_pool, A_cg, B_cg, geometry_graph_A_cg, geometry_graph_B_cg, epoch):
        (v_A, h_A), (v_B, h_B), geom_losses, geom_loss_cg, full_trajectory, full_trajectory_cg = self.ecn(A_graph, B_graph, geometry_graph_A, geometry_graph_B, A_pool, B_pool, A_cg, B_cg, geometry_graph_A_cg, geometry_graph_B_cg, epoch)
        # ipdb.set_trace()
        posterior_input_V = torch.cat((v_A, v_B), dim = 1) # N x 2F x 3
        posterior_input_h = torch.cat((h_A, h_B), dim = 1) # N x 2D

        prior_mean_V = self.prior_mean_V(v_B)
        prior_mean_h = self.prior_mean_h(h_B)
        prior_logvar_V = self.prior_logvar_V(v_B.reshape((v_B.shape[0], -1))).unsqueeze(2)
        prior_logvar_h = self.prior_logvar_h(h_B)

        posterior_mean_V = self.posterior_mean_V(posterior_input_V)
        posterior_mean_h = self.posterior_mean_h(posterior_input_h)
        posterior_logvar_V = self.posterior_logvar_V(posterior_input_V.reshape((posterior_input_V.shape[0], -1))).unsqueeze(2)
        posterior_logvar_h = self.posterior_logvar_h(posterior_input_h)

        Z_V = self.reparameterize(posterior_mean_V, posterior_logvar_V)
        Z_h = self.reparameterize(posterior_mean_h, posterior_logvar_h)

        A_cg.ndata["Z_V"] = Z_V
        A_cg.ndata["Z_h"] = Z_h

        results = {
            "Z_V": Z_V,
            "Z_h": Z_h,
            "v_A": v_A,
            "v_B": v_B,
            "h_A": h_A,
            "h_B": h_B,

            "prior_mean_V": prior_mean_V,
            "prior_mean_h": prior_mean_h,
            "prior_logvar_V": prior_logvar_V,
            "prior_logvar_h": prior_logvar_h,

            "posterior_mean_V": posterior_mean_V,
            "posterior_mean_h": posterior_mean_h,
            "posterior_logvar_V": posterior_logvar_V,
            "posterior_logvar_h": posterior_logvar_h,

        }
        return results, geom_losses, geom_loss_cg, full_trajectory, full_trajectory_cg

    def reparameterize(self, mean, logvar):
        sigma = 1e-12 + torch.exp(logvar / 2)
        eps = torch.randn_like(mean)
        return mean + eps*sigma

    # https://github.com/NVIDIA/NeMo/blob/b9cf05cf76496b57867d39308028c60fef7cb1ba/nemo/collections/nlp/models/machine_translation/mt_enc_dec_bottleneck_model.py#L217
    def kl(self, z_mean, z_logvar, z_mean_prior, z_logvar_prior, coordinates = False):
        # posterior = torch.distributions.Normal(loc=z_mean, scale=torch.exp(0.5 * z_logv))
        # prior = torch.distributions.Normal(loc=z_mean_prior, scale=torch.exp(0.5 * z_logv_prior))
        # loss = (reconstruction( MSE = negative log prob) + self.kl_beta * KL)
        # Dkl( P || Q) # i added the .sum()
        # ipdb.set_trace()
        p_std = 1e-12 + torch.exp(z_logvar / 2)
        q_std = 1e-12 + torch.exp(z_logvar_prior / 2)
        q_mean = z_mean_prior
        p_mean = z_mean
        var_ratio = (p_std / q_std).pow(2).sum(-1)
        t1 = ((p_mean - q_mean) / q_std).pow(2).sum(-1)
        if coordinates:
            var_ratio = var_ratio.sum(-1)
            t1 = t1.sum(-1)
        kl =  0.5 * (var_ratio + t1 - 1 - var_ratio.log()) # shape = number of CG beads
        return kl.mean()

class Decoder(nn.Module):
    def __init__(self, iegmn, D, F, atom_embedder, device = "cuda"):
        super(Decoder, self).__init__()
        self.mha = torch.nn.MultiheadAttention(embed_dim = 3, num_heads = 1, batch_first = True) # requires mask and reshaping on the batch dim due to dgl
        self.iegmn = iegmn
        self.h_channel_selection = atom_embedder # taken from the FG encoder
        self.device = device
    
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
        # ipdb.set_trace()
        Z_V = A_cg.ndata["Z_V"]# N x F x 3
        Z_h = A_cg.ndata["Z_h"] # N x D
        N, F, _ = Z_V.shape
        _, D = Z_h.shape

        Q, attn_mask, cg_to_fg_info = self.get_queries_and_mask(A_cg, B, N, F, cg_frag_ids)
        K = Z_V
        V = Z_V
        attn_out, attn_weights = self.mha(Q, K, V, attn_mask = attn_mask)

        res = []
        for idx, k in enumerate(cg_to_fg_info):
            res.append( attn_out[idx, :k, :]) # Pull from the parts that were not padding
        res = torch.cat(res, dim = 0)
        x_cc = res # n x 3

        h_og = B.ndata['rd_feat'] # n x d = 17
        h_og_lifted = self.h_channel_selection(h_og) # n x D
        h_cc = h_og_lifted + torch.repeat_interleave(Z_h, torch.tensor(cg_to_fg_info).to(self.device), dim = 0) # n x D
        return x_cc, h_cc # we have selected the features for all coarse grain beads in parallel


    def autoregressive_step(self, latent, prev = None, t = 0):
        # ipdb.set_trace()
        coords_A, h_feats_A, coords_B, h_feats_B, geom_losses, full_trajectory = self.iegmn(latent, prev, mpnn_only = t==0)
        # coords_A held in latent.ndata['x_now']
        return  coords_A, h_feats_A, coords_B, h_feats_B, geom_losses, full_trajectory

    def split_subgraph(self, X, H, fine, frag_ids, bfs_order, start = 0):
        # ipdb.set_trace()
        result = []
        for idx, atom_ids in enumerate(frag_ids):
            og_atom_ids = list(atom_ids)
            atom_ids = [x+start for x in atom_ids]
            coords = X[atom_ids, :]
            feats = H[atom_ids, :]
            # result.append( (coords, feats, bfs_order[idx]))
            subg = fine.subgraph(og_atom_ids)
            subg.ndata['x_cc'] = coords
            subg.ndata['feat_cc'] = feats #! here the rdkit Fine is being overwritten with the result of our attention operation as wanted
            # TODO: must update the edge features.
            # TODO: better off creating a new function to create a subgraph instead of resetting stuff
            result.append( (subg, bfs_order[idx]))
        result.sort(key=lambda x: x[-1])
        return result


    def forward(self, cg_mol_graph, rdkit_mol_graph, cg_frag_ids):
        ipdb.set_trace()
        X_cc, H_cc = self.channel_selection(cg_mol_graph, rdkit_mol_graph, cg_frag_ids)

        ACGs = dgl.unbatch(cg_mol_graph) # list of graphs
        Bs = dgl.unbatch(rdkit_mol_graph)
        subgraphs = []
        bulk_atoms = 0
        for idx in range(len(Bs)):
            subgraphs.append(self.split_subgraph(X_cc, H_cc, Bs[idx], cg_frag_ids[idx], ACGs[idx].ndata['bfs'].flatten(), start = bulk_atoms))
            bulk_atoms += Bs[idx].num_nodes()
        # ipdb.set_trace()
        ar_batches = defaultdict(list)
        ar_batches_prev = defaultdict(list)
        for sg in subgraphs:
            for idx, val in enumerate(sg):
                assert(idx == val[1])
                ar_batches[idx].append(val[0])
            for idx in range(1, len(sg)-1):
                ar_batches_prev[idx].extend(ar_batches[idx-1])
        dgl_ar_batches = [dgl.batch(ar_batches[k]) for k in range(len(ar_batches))]
        # ipdb.set_trace()
        dgl_ar_batches_prev = [dgl.batch(ar_batches_prev[k]) if len(ar_batches_prev[k]) > 0 else None for k in range(len(ar_batches_prev))]
        
        for t in range(len(dgl_ar_batches)):
            latent = dgl_ar_batches[t]
            prev = dgl_ar_batches_prev[t] # TODO will have to figure out how to combine and propagate stuff later on instead just feeding in previous single subgraph
            # TODO merge with valid graphs of latent t-1. design this better
            # ! See dgl.merge and add_eges

            #TODO weird edge error likely due to poor data creation
            coords_A, h_feats_A, coords_B, h_feats_B, geom_losses, full_trajectory = self.autoregressive_step(latent, prev, t) # TODO: implent distance reularization with ground truth distance graph
            ipdb.set_trace()
            latent.ndata['x_cc'] = coords_A
            latent.ndata['feat_cc'] = h_feats_A
            # prev = latent
        return dgl_ar_batches




    
