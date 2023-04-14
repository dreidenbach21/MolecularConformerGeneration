from ecn_3d import *
from equivariant_model_utils import *
from geometry_utils import *
from decoder_utils_delta import IEGMN_Bidirectional_Delta
from collections import defaultdict
import numpy as np
import copy
from torch import nn
import torch.nn.functional as F
import ipdb
from neko_fixed_attention import neko_MultiheadAttention

class VAE_Delta(nn.Module):
    def __init__(self, ecn, D, F, iegmn, atom_embedder, device = "cuda"):
        super(VAE_Delta, self).__init__()
        self.encoder = Encoder(ecn, D, F).to(device)
        self.decoder = Decoder(iegmn, D, F, atom_embedder, device).to(device)
        self.D = D 
        self.F = F 
        self.kl_v_beta = 1#e-4 2.5
        self.kl_h_beta = 0#1e-4
        self.kl_reg_beta = 1
        self.align_kabsch_weight = 20
        self.ar_rmsd_weight = 10 #1
        self.mse = nn.MSELoss()
        # self.mse_sum= nn.MSELoss(reduction='sum')
        self.device = device
        # self.weight_decay_v = 0.1
        # self.weight_decay_h = 0.005
        self.lambda_x_cc = 0
        self.lambda_h_cc = 0 #1e-2

    def flip_teacher_forcing(self):
        self.decoder.teacher_forcing = not self.decoder.teacher_forcing

    def forward(self, frag_ids, A_graph, B_graph, geometry_graph_A, geometry_graph_B, A_pool, B_pool, A_cg, B_cg, geometry_graph_A_cg, geometry_graph_B_cg, epoch):
        enc_out = self.encoder(A_graph, B_graph, geometry_graph_A, geometry_graph_B, A_pool, B_pool, A_cg, B_cg, geometry_graph_A_cg, geometry_graph_B_cg, epoch)
        results, geom_losses, geom_loss_cg, full_trajectory, full_trajectory_cg = enc_out
        print("[ENC] encoder output geom loss adn geom cg loss", geom_losses, geom_loss_cg)
        kl_v = self.encoder.kl(results["posterior_mean_V"], results["posterior_logvar_V"], results["prior_mean_V"], results["prior_logvar_V"], coordinates = True)
        # kl_v_reg = self.encoder.kl(results["prior_mean_V"], results["prior_logvar_V"], torch.zeros_like(results["prior_mean_V"]), torch.zeros_like(results["prior_logvar_V"]), coordinates = True)
        # kl_v_reg = self.encoder.kl(results["posterior_mean_V"], results["posterior_logvar_V"], torch.zeros_like(results["posterior_mean_V"]), torch.zeros_like(results["posterior_logvar_V"]), coordinates = True)
        # kl_v_reg = self.encoder.kl(results["prior_mean_V"], results["prior_logvar_V"], results["prior_mean_V"], torch.zeros_like(results["prior_logvar_V"]), coordinates = True)
        # kl_h = self.encoder.kl(results["posterior_mean_h"], results["posterior_logvar_h"], results["prior_mean_h"], results["prior_logvar_h"], coordinates = False)
        kl_v_reg = 0
        kl_h = 0
        dec_out = self.decoder(A_cg, B_graph, frag_ids, geometry_graph_A)
        generated_molecule, rdkit_reference, dec_results, channel_selection_info = dec_out
        return generated_molecule, rdkit_reference, dec_results, channel_selection_info, (kl_v, kl_h, kl_v_reg), enc_out
    
    def distance_loss(self, generated_molecule, geometry_graphs):
        geom_loss = []
        for geometry_graph, generated_mol in zip(dgl.unbatch(geometry_graphs), dgl.unbatch(generated_molecule)):
            src, dst = geometry_graph.edges()
            src = src.long()
            dst = dst.long()
            generated_coords = generated_mol.ndata['x_cc']
            d_squared = torch.sum((generated_coords[src] - generated_coords[dst]) ** 2, dim=1)
            geom_loss.append(1/len(src) * torch.sum((d_squared - geometry_graph.edata['feat'] ** 2) ** 2)) #TODO: scaling hurt performance
            # geom_loss.append(torch.sum((d_squared - geometry_graph.edata['feat'] ** 2) ** 2))
        print("[Distance Loss]", geom_loss)
        return torch.mean(torch.tensor(geom_loss))

    def loss_function(self, generated_molecule, rdkit_reference, dec_results, channel_selection_info, KL_terms, enc_out, geometry_graph, step = 0):
        kl_v, kl_h, kl_v_reg = KL_terms
        print("[Loss Func] KL V", kl_v)
        print("kl h", kl_h)
        print("KL prior reg kl", kl_v_reg)
    
        kl_loss = self.kl_v_beta*kl_v + self.kl_h_beta*kl_h + self.kl_reg_beta*kl_v_reg
        if step < 5:#0:
            kl_loss = 0.1*kl_loss
        
        x_cc, h_cc = channel_selection_info
        # print("[Loss Func] Channel Selection Norms (x,h): ", torch.norm(x_cc, 2), torch.norm(h_cc, 2))
        x_true = rdkit_reference.ndata['x_true']
        print("[Loss Func] Channel Selection Norms (x,h): ", torch.norm(x_cc, 2), torch.norm(h_cc, 2), torch.norm(x_true, 2))
        # cc_loss = self.lambda_cc*self.mse_sum(x_cc, x_true) #self.lambda_cc*torch.norm(x_cc-x_true, 2)**2
        x_cc_loss = (torch.norm(x_cc, 2) - torch.norm(x_true, 2))**2
        print("[Loss Func] X CC norm diff loss", x_cc_loss)
        x_cc_loss = []
        start = 0
        for natoms in generated_molecule.batch_num_nodes():
            x_cc_loss.append(self.rmsd(x_cc[start: start + natoms], x_true[start: start + natoms], align = True))
            start += natoms
        print("[Loss Func] aligned X CC loss", x_cc_loss)
        x_cc_loss = sum(x_cc_loss)
        # ipdb.set_trace()
        h_cc_loss = (torch.norm(h_cc, 2))**2
        print("[Loss Func] h CC norm loss", h_cc_loss)
        cc_loss = self.lambda_x_cc*x_cc_loss + self.lambda_h_cc*h_cc_loss
        print()
        # final_gen_coords = generated_molecule.ndata['x_cc']
        # true_coords = generated_molecule.ndata['x_true']
        # rdkit_coords = rdkit_reference.ndata['x_ref']
        ar_rmsd, final_align_rmsd, ar_dist_loss = self.coordinate_loss(dec_results, generated_molecule, align =  True)
        print()
        print("[Loss Func] Auto Regressive MSE", ar_rmsd)
        print("[Loss Func] Kabsch Align MSE", final_align_rmsd)
        print("[Loss Func] step", step, "KL Loss", kl_loss)
        print()
        loss =  self.ar_rmsd_weight*ar_rmsd + self.align_kabsch_weight*final_align_rmsd + kl_loss #+ cc_loss
        results, geom_losses, geom_loss_cg, full_trajectory, full_trajectory_cg = enc_out
        # ipdb.set_trace()
        # l2_v = torch.norm(results["posterior_logvar_V"], 2)**2
        # l2_v2 = torch.norm(results["posterior_mean_V"], 2)**2
        # l2_vp = torch.norm(results["prior_logvar_V"], 2)**2
        # l2_vp2 = torch.norm(results["prior_mean_V"], 2)**2

        l2_v = torch.norm(self.std(results["posterior_logvar_V"]), 2)**2
        l2_v2 = torch.norm(results["posterior_mean_V"], 2)**2
        l2_vp = torch.norm(self.std(results["prior_logvar_V"]), 2)**2
        l2_vp2 = torch.norm(results["prior_mean_V"], 2)**2
        l2_d = torch.norm(results["posterior_mean_V"]-results["prior_mean_V"], 2)**2
        # l2_h = torch.norm(results["posterior_logvar_h"], 2)**2
        # print("log variance norm: v, h: ", l2_v, l2_h)
        # l2_loss = 0 #self.weight_decay_v*l2_v+self.weight_decay_h*l2_h
        # loss += l2_loss
        # ipdb.set_trace()
        rdkit_loss = sum([x.ndata['rdkit_loss'][0] for x in dgl.unbatch(rdkit_reference)])
        distance_lambda = 10
        distance_loss = distance_lambda*self.distance_loss(generated_molecule, geometry_graph)
        ar_dist_lambda = 10
        ar_dist_loss = ar_dist_lambda*ar_dist_loss
        print("[Loss Func] distance", distance_loss)
        print("[Loss Func] ar distance", ar_dist_loss)
        loss += distance_loss + ar_dist_loss
        
        return loss, (ar_rmsd.cpu(), final_align_rmsd.cpu(), kl_loss.cpu(), x_cc_loss.cpu(), h_cc_loss.cpu(),
                     l2_v.cpu(), l2_v2.cpu(), l2_vp.cpu(), l2_vp2.cpu(), l2_d.cpu(), rdkit_loss.cpu(), distance_loss.cpu(), ar_dist_loss.cpu())

    def std(self, input):
        return 1e-12 + torch.exp(input / 2)
        #  return 1e-12 + F.softplus(input / 2)

    def align(self, source, target):
        # Rot, trans = rigid_transform_Kabsch_3D_torch(input.T, target.T)
        # lig_coords = ((Rot @ (input).T).T + trans.squeeze())
        # Kabsch RMSD implementation below taken from EquiBind
        # if source.shape[0] == 2: #! Kabsch seems to work better for RMSD still
        #     return align_sets_of_two_points(target, source)
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
    
    def ar_loss_step(self, coords, coords_ref, chunks, align = False, step = 1, first_step = 1):
        loss = 0
        start = 0
        for chunk in chunks:
            sub_loss = self.rmsd(coords[start: start + chunk, :], coords_ref[start:start+chunk, :], align)
            print("       ", sub_loss.cpu().item(), coords[start: start + chunk, :].shape)
            if coords[start: start + chunk, :].shape[0] == 1:
                print("       ", coords[start: start + chunk, :], coords_ref[start: start + chunk, :])
            loss += sub_loss
            start += chunk
        if step == 0:
            loss *= first_step
        return loss

    def coordinate_loss(self, dec_results, generated_molecule, align = False):# = None, final_gen_coords = None, true_coords = None):
        loss = 0
        dist_losses = 0
        for step, info in enumerate(dec_results):
            coords_A, h_feats_A, coords_B, h_feats_B, geom_losses, _, id_batch, ref_coords_A, ref_coords_B, ref_coords_B_split, gen_input, ar_con_input, dist_loss = info
            # ipdb.set_trace()
            # print("ID")
            # num_molecules = len([x for x in id_batch if x is not None])
            num_molecule_chunks = [len(x) for x in id_batch if x is not None]
            # ga = self.mse_sum(coords_A, ref_coords_A)/num_molecules # Generative accuracy #TDO: was self.rmsd
            ga = self.ar_loss_step(coords_A, ref_coords_A, num_molecule_chunks, align, step)
            print("Generative", coords_A.shape, ga)
            print("Dist Loss", dist_loss, "\n")
            # if torch.gt(ga, 1000).any() or torch.isinf(ga).any() or torch.isnan(ga).any():
            #     print("Chunks", num_molecule_chunks)
            #     print("generative input", gen_input)
            #     print("Bad Coordinate Check A", coords_A)
            loss += ga
            dist_losses += dist_loss
            if coords_B is None:
                assert(step == 0)
                print()
                continue
            # arc = self.mse_sum(coords_B, ref_coords_B)/num_molecules # AR consistency
            num_molecule_chunks = [x.shape[0] for x in ref_coords_B_split]
            arc = self.ar_loss_step(coords_B, ref_coords_B, num_molecule_chunks, align)
            print("AR Consistency", coords_B.shape, arc)
            if torch.gt(arc, 1000).any() or torch.isinf(arc).any() or torch.isnan(arc).any():
                print("Chunks", num_molecule_chunks)
                print("conditional input", ar_con_input)
                print("Bad Coordinate Check B", coords_B)
            print()
            loss += arc
        # molecules = dgl.unbatch(generated_molecule)
        align_loss = [self.rmsd(m.ndata['x_cc'],m.ndata['x_true'], align = True) for m in dgl.unbatch(generated_molecule)]
        # for m in molecules:
        #     print('LOSS kabsch RMSD between generated ligand and true ligand is ', np.sqrt(np.sum((m.ndata['x_cc'].cpu().detach().numpy() - self.align(m.ndata['x_true'], m.ndata['x_cc']).cpu().detach().numpy()) ** 2, axis=1).mean()).item())
        #     print('LOSS-2 kabsch RMSD between generated ligand and true ligand is ', np.sqrt(self.mse(m.ndata['x_cc'], self.align(m.ndata['x_true'], m.ndata['x_cc'])).cpu().detach().numpy()).item())
        #     print('LOSS-3 kabsch RMSD between generated ligand and true ligand is ', np.sqrt(self.mse_loss(m.ndata['x_cc'], m.ndata['x_true'], align=True).cpu().detach().numpy()).item())
        print("Align MSE Loss", align_loss)
        align_loss = sum(align_loss)
        return loss, align_loss, dist_losses
            

class Encoder(nn.Module):
    def __init__(self, ecn, D, F):
        super(Encoder, self).__init__()
        self.ecn = ecn
        
        self.posterior_mean_V = VN_MLP(2*F, F, F, F, use_batchnorm = False) 
        # self.posterior_mean_V = Vector_MLP(2*F, 2*F, 2*F, F, use_batchnorm = False) 
        self.posterior_mean_h = Scalar_MLP(2*D, 2*D, D, use_batchnorm = False)
        self.posterior_logvar_V = Scalar_MLP(2*F*3, 2*F*3, F, use_batchnorm = False)# need to flatten to get equivariant noise N x F x 1
        self.posterior_logvar_h = Scalar_MLP(2*D, 2*D, D, use_batchnorm = False)

        self.prior_mean_V = VN_MLP(F,F,F,F, use_batchnorm = False)
        # self.prior_mean_V = Vector_MLP(F,F,F,F, use_batchnorm = False)
        self.prior_mean_h = Scalar_MLP(D, D, D, use_batchnorm = False)
        self.prior_logvar_V = Scalar_MLP(F*3, F*3, F, use_batchnorm = False)
        self.prior_logvar_h = Scalar_MLP(D, D, D, use_batchnorm = False)
        self.bn = VNBatchNorm(F)

    def forward(self, A_graph, B_graph, geometry_graph_A, geometry_graph_B, A_pool, B_pool, A_cg, B_cg, geometry_graph_A_cg, geometry_graph_B_cg, epoch):
        (v_A, h_A), (v_B, h_B), geom_losses, geom_loss_cg, full_trajectory, full_trajectory_cg = self.ecn(A_graph, B_graph, geometry_graph_A, geometry_graph_B, A_pool, B_pool, A_cg, B_cg, geometry_graph_A_cg, geometry_graph_B_cg, epoch)
        # ipdb.set_trace()
        print("[Encoder] ecn output V A", torch.min(v_A).item(), torch.max(v_A).item())
        print("[Encoder] ecn output V B", torch.min(v_B).item(), torch.max(v_B).item())
        # if torch.isnan(torch.max(v_A)): #TODO How come we get NaN issue stemming from here
        #     import ipdb; ipdb.set_trace()
        #     (v_A, h_A), (v_B, h_B), geom_losses, geom_loss_cg, full_trajectory, full_trajectory_cg = self.ecn(A_graph, B_graph, geometry_graph_A, geometry_graph_B, A_pool, B_pool, A_cg, B_cg, geometry_graph_A_cg, geometry_graph_B_cg, epoch)
        posterior_input_V = torch.cat((v_A, v_B), dim = 1) # N x 2F x 3
        posterior_input_h = torch.cat((h_A, h_B), dim = 1) # N x 2D

        prior_mean_V = self.prior_mean_V(v_B)
        prior_mean_h = self.prior_mean_h(h_B)
        prior_logvar_V = self.prior_logvar_V(v_B.reshape((v_B.shape[0], -1))).unsqueeze(2)
        prior_logvar_h = self.prior_logvar_h(h_B)

        # TODO: try clamping the prior logv to 0 or min clamp the posterior
        prior_logvar_V = torch.clamp(prior_logvar_V, max = 0)

        posterior_mean_V = self.posterior_mean_V(posterior_input_V)
        # print("[Encoder] pre BN posterior mean V", torch.min(posterior_mean_V).item(), torch.max(posterior_mean_V).item()) #, torch.sum(posterior_mean_V, dim = 1))
        # posterior_mean_V = self.bn(posterior_mean_V) #TODO: Trying VN Batch Norm
        posterior_mean_h = self.posterior_mean_h(posterior_input_h)
        posterior_logvar_V = self.posterior_logvar_V(posterior_input_V.reshape((posterior_input_V.shape[0], -1))).unsqueeze(2)
        posterior_logvar_h = self.posterior_logvar_h(posterior_input_h)

        print("[Encoder] posterior mean V", torch.min(posterior_mean_V).item(), torch.max(posterior_mean_V).item()) #, torch.sum(posterior_mean_V, dim = 1))
        print("[Encoder] posterior logvar V", torch.min(posterior_logvar_V).item(), torch.max(posterior_logvar_V).item()) #, torch.sum(posterior_logvar_V,  dim = 1))
        print("[Encoder] posterior mean h", torch.min(posterior_mean_h).item(), torch.max(posterior_mean_h).item(), torch.sum(posterior_mean_h).item())
        print("[Encoder] posterior logvar h", torch.min(posterior_logvar_h).item(), torch.max(posterior_logvar_h).item(), torch.sum(posterior_logvar_h).item())

        print("[Encoder] prior mean V", torch.min(prior_mean_V).item(), torch.max(prior_mean_V).item()) #, torch.sum(prior_mean_V,  dim = 1))
        print("[Encoder] prior logvar V", torch.min(prior_logvar_V).item(), torch.max(prior_logvar_V).item()) #, torch.sum(prior_logvar_V,  dim = 1))
        print("[Encoder] prior mean h", torch.min(prior_mean_h).item(), torch.max(prior_mean_h).item(), torch.sum(prior_mean_h).item())
        print("[Encoder] prior logvar h", torch.min(prior_logvar_h).item(), torch.max(prior_logvar_h).item(), torch.sum(prior_logvar_h).item())
        Z_V = self.reparameterize(posterior_mean_V, posterior_logvar_V)
        # if torch.gt(Z_V, 100).any() or  torch.lt(Z_V, -100).any():
        #         ipdb.set_trace()
        #         print("Caught Explosion in VAE step")
        Z_h = self.reparameterize(posterior_mean_h, posterior_logvar_h)
        print("[Encoder] Z_V post vae step", torch.min(Z_V).item(), torch.max(Z_V).item())

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

    def reparameterize(self, mean, logvar, scale = 1.0):
        # scale = 0.3 # trying this
        sigma = 1e-12 + torch.exp(scale*logvar / 2)
        # sigma = 1e-12 + F.softplus(scale*logvar / 2)
        eps = torch.randn_like(mean)
        # if torch.gt(mean + eps*sigma, 100).any() or  torch.lt(mean + eps*sigma, -100).any():
        #         ipdb.set_trace()
        #         print("Caught Explosion in VAE step")
        return mean + eps*sigma

    # https://github.com/NVIDIA/NeMo/blob/b9cf05cf76496b57867d39308028c60fef7cb1ba/nemo/collections/nlp/models/machine_translation/mt_enc_dec_bottleneck_model.py#L217
    def kl(self, z_mean, z_logvar, z_mean_prior, z_logvar_prior, coordinates = False):
        # posterior = torch.distributions.Normal(loc=z_mean, scale=torch.exp(0.5 * z_logv))
        # prior = torch.distributions.Normal(loc=z_mean_prior, scale=torch.exp(0.5 * z_logv_prior))
        # loss = (reconstruction( MSE = negative log prob) + self.kl_beta * KL)
        # Dkl( P || Q) # i added the .sum()
        # ipdb.set_trace() #TODO look into softplus instead of exp
        p_std = 1e-12 + torch.exp(z_logvar / 2)
        q_std = 1e-12 + torch.exp(z_logvar_prior / 2)
        # p_std = 1e-12 + F.softplus(z_logvar / 2)
        # q_std = 1e-12 + F.softplus(z_logvar_prior / 2)
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
    def __init__(self, iegmn, D, F, atom_embedder, device = "cuda", norm="ln", teacher_forcing = True):
        super(Decoder, self).__init__()
        # self.mha = torch.nn.MultiheadAttention(embed_dim = 3, num_heads = 1, batch_first = True)
        self.mha = neko_MultiheadAttention(embed_dim = 3, num_heads = 1, batch_first = True)
        self.iegmn = iegmn
        self.atom_embedder = atom_embedder # taken from the FG encoder
        self.h_channel_selection = Scalar_MLP(D, 2*D, D)
        self.mse = nn.MSELoss()
        # norm = "ln"
        if norm == "bn":
            self.eq_norm = VNBatchNorm(F)
            self.inv_norm = nn.BatchNorm1d(D)
            self.eq_norm_2 = VNBatchNorm(3)
            self.inv_norm_2 = nn.BatchNorm1d(D)
        elif norm == "ln":
            self.eq_norm = VNLayerNorm(F)
            self.inv_norm = nn.LayerNorm(D)
            self.eq_norm_2 = VNLayerNorm(3)
            self.inv_norm_2 = nn.LayerNorm(D)
        else:
            assert(1 == 0)
        self.device = device
        # TODO: try Vector_MLP repalce with VN_MLP for traditional MLP formulation not 3DLinkers
        # self.feed_forward_V = Vector_MLP(F, 2*F, 2*F, F, leaky = False, use_batchnorm = False)
        self.feed_forward_V = nn.Sequential(VNLinear(F, 2*F), VN_MLP(2*F, F, F, F, leaky = False, use_batchnorm = False))
        self.feed_forward_h = Scalar_MLP(D, 2*D, D)

        # self.feed_forward_V_3 = Vector_MLP(3, F, F, 3, leaky = False, use_batchnorm = False)
        self.feed_forward_V_3 = nn.Sequential(VNLinear(3, F), VN_MLP(F, 3, 3, 3, leaky = False, use_batchnorm = False))
        self.feed_forward_h_3 = Scalar_MLP(D, 2*D, D)
        self.teacher_forcing = teacher_forcing
    
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
        # lens = [sum([len(y) for y in x]) for x in ids]
        # check = all([a-b.item() == 0 for a, b in zip(lens,progress)])
        # ipdb.set_trace()
        # assert(check)
        return ids, progress

    def distance_loss(self, generated_coords_all, geometry_graphs, true_coords = None):
        geom_loss = []
        for geometry_graph, generated_coords in zip(dgl.unbatch(geometry_graphs), generated_coords_all):
            src, dst = geometry_graph.edges()
            src = src.long()
            dst = dst.long()
            if len(src) == 0 or len(dst) == 0:
                continue
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

    def get_reference(self, subgraph, molecule, ref_ids):
        # references = []
        result = []
        info = [(m, x) for m, x in zip(dgl.unbatch(molecule), ref_ids) if x is not None]
        for g, mid in zip(dgl.unbatch(subgraph), info):
            m, id = mid
            if id is None:
                continue
            r = id[0]
            if r == -1:
                # references.append(torch.zeros_like(g.ndata['x_cc']))
                g.ndata['reference_point'] = torch.zeros_like(g.ndata['x_cc'])
            else:
                point = molecule.ndata['x_cc'][r].reshape(1,-1)
                # references.append(point.repeat(g.ndata['x_cc'].shape[0]))
                g.ndata['reference_point'] = point.repeat(g.ndata['x_cc'].shape[0], 1)
            result.append(g)
        return dgl.batch(result).to(self.device) #, references

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
        frag_ids, progress = self.add_reference(frag_ids, ref_order, progress) #! Do not need reference when we do delta coordinates?
        # ipdb.set_trace()
        frag_batch = defaultdict(list) # keys will be time steps
       
        max_nodes = max(cg_mol_graph.batch_num_nodes()).item()
        for t in range(max_nodes):
            for idx, frag in enumerate(frag_ids): # iterate over moelcules
                ids = None
                if t < len(frag):
                    ids = list(frag[t])
                frag_batch[t].append(ids)
        ref_batch = defaultdict(list)
        # ipdb.set_trace()
        for t in range(max_nodes):
            for idx, refs in enumerate(ref_order): # iterate over moelcules
                ids = None
                if t < len(refs):
                    ids = [int(refs[t].item())]
                ref_batch[t].append(ids)

        current_molecule_ids = None
        current_molecule = None
        geo_current = None
        returns = []
        for t in range(max_nodes):
            # ipdb.set_trace()
            print("[Auto Regressive Step]")
            id_batch = frag_batch[t]
            r_batch = ref_batch[t]
            # print("ID", id_batch)
            # if t == 1:
            #     ipdb.set_trace()
            latent, geo_latent = self.isolate_next_subgraph(final_molecule, id_batch, true_geo_batch)
            latent = self.get_reference(latent, final_molecule, r_batch)
            # ipdb.set_trace()
            # if not check:
            #     current_molecule = self.adaptive_batching(current_molecule)
            # ipdb.set_trace()
            coords_A, h_feats_A, coords_B, h_feats_B, geom_losses, full_trajectory = self.autoregressive_step(latent, current_molecule, t, geo_latent, geo_current)
            ref_coords_A = latent.ndata['x_true']
            ref_coords_B = current_molecule.ndata['x_true'] if current_molecule is not None else None
            ref_coords_B_split = [x.ndata['x_true'] for x in dgl.unbatch(current_molecule)] if current_molecule is not None else None
            ar_con_input = current_molecule.ndata['x_cc'] if current_molecule is not None else None
            gen_input = latent.ndata['x_cc']
            print("[AR step end] geom losses total from decoder", t, geom_losses)
            dist_loss = self.distance_loss([x.ndata['x_cc'] for x in dgl.unbatch(latent)], geo_latent, [x.ndata['x_true'] for x in dgl.unbatch(latent)])
            # print("True Distance Check")
            # _ = self.distance_loss([x.ndata['x_true'] for x in dgl.unbatch(latent)], geo_latent)
            print()
            returns.append((coords_A, h_feats_A, coords_B, h_feats_B, geom_losses, full_trajectory, id_batch, ref_coords_A, ref_coords_B, ref_coords_B_split, gen_input, ar_con_input, dist_loss))
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
