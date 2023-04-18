from encoder import *
from decoder import *

class VAE(nn.Module):
    def __init__(self, kl_params, encoder_params, decoder_params, loss_params, device = "cuda"):
        super(VAE, self).__init__()
        self.encoder = Encoder(**encoder_params).to(device)
        self.decoder = Decoder(self.encoder.atom_embedder, **decoder_params).to(device)
        self.mse = nn.MSELoss()
        self.mse_none = nn.MSELoss(reduction ='none')
        self.device = device
        F = encoder_params["coord_F_dim"]
        D = encoder_params["latent_dim"]
        
        self.kl_free_bits = kl_params['kl_free_bits']
        self.kl_prior_logvar_clamp = kl_params['kl_prior_logvar_clamp']
        self.kl_softplus = kl_params['kl_softplus']

        self.kl_v_beta = loss_params['kl_weight']
        self.kl_h_beta = 0
        # self.kl_reg_beta = 1
        self.lambda_global_mse = loss_params['global_mse_weight']
        self.lambda_ar_mse = loss_params['ar_mse_weight']
        self.lambda_x_cc = loss_params['x_cc_weight']
        self.lambda_h_cc = loss_params['h_cc_weight']
        self.lambda_distance = loss_params['distance_weight']
        self.lambda_ar_distance = loss_params['ar_distance_weight']
        self.ar_loss_direction = loss_params['ar_loss_bottom_up']
        self.loss_params = loss_params

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

    # def flip_teacher_forcing(self):
    #     self.decoder.teacher_forcing = not self.decoder.teacher_forcing

    def forward(self, frag_ids, A_graph, B_graph, geometry_graph_A, geometry_graph_B, A_pool, B_pool, A_cg, B_cg, geometry_graph_A_cg, geometry_graph_B_cg, epoch):
        enc_out = self.forward_vae(A_graph, B_graph, geometry_graph_A, geometry_graph_B, A_pool, B_pool, A_cg, B_cg, geometry_graph_A_cg, geometry_graph_B_cg, epoch)
        results, geom_losses, geom_loss_cg, full_trajectory, full_trajectory_cg = enc_out
        print("[ENC] encoder output geom loss adn geom cg loss", geom_losses, geom_loss_cg)
        kl_v = self.kl(results["posterior_mean_V"], results["posterior_logvar_V"], results["prior_mean_V"], results["prior_logvar_V"], coordinates = True)
        # kl_v_reg = self.kl(results["prior_mean_V"], results["prior_logvar_V"], torch.zeros_like(results["prior_mean_V"]), torch.zeros_like(results["prior_logvar_V"]), coordinates = True)
        # kl_v_reg = self.kl(results["posterior_mean_V"], results["posterior_logvar_V"], torch.zeros_like(results["posterior_mean_V"]), torch.zeros_like(results["posterior_logvar_V"]), coordinates = True)
        # kl_v_reg = self.kl(results["prior_mean_V"], results["prior_logvar_V"], results["prior_mean_V"], torch.zeros_like(results["prior_logvar_V"]), coordinates = True)
        # kl_h = self.kl(results["posterior_mean_h"], results["posterior_logvar_h"], results["prior_mean_h"], results["prior_logvar_h"], coordinates = False)
        # kl_v_reg = 0
        kl_h = 0
        dec_out = self.decoder(A_cg, B_graph, frag_ids, geometry_graph_A)
        generated_molecule, rdkit_reference, dec_results, channel_selection_info = dec_out
        return generated_molecule, rdkit_reference, dec_results, channel_selection_info, (kl_v, kl_h), enc_out #, kl_v_reg), enc_out
    
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
        # kl_v, kl_h, kl_v_reg = KL_terms
        kl_v, kl_h = KL_terms
        print("[Loss Func] KL V", kl_v)
        print("[Loss Func] kl h", kl_h)
        kl_loss = self.kl_v_beta*kl_v + self.kl_h_beta*kl_h # + self.kl_reg_beta*kl_v_reg
        # if step < 0:#50:
        #     kl_loss = 0.1*kl_loss
        x_cc, h_cc = channel_selection_info
        x_true = rdkit_reference.ndata['x_true']
        print("[Loss Func] Channel Selection Norms (x,h): ", torch.norm(x_cc, 2), torch.norm(h_cc, 2), torch.norm(x_true, 2))
        x_cc_loss = (torch.norm(x_cc, 2) - torch.norm(x_true, 2))**2
        print("[Loss Func] X CC norm diff loss", x_cc_loss)
        x_cc_loss = []
        start = 0
        for natoms in generated_molecule.batch_num_nodes():
            x_cc_loss.append(self.rmsd(x_cc[start: start + natoms], x_true[start: start + natoms], align = True))
            start += natoms
        print("[Loss Func] aligned X CC loss", x_cc_loss)
        x_cc_loss = sum(x_cc_loss)
        # # ipdb.set_trace()
        # h_cc_loss = (torch.norm(h_cc, 2))**2
        # print("[Loss Func] h CC norm loss", h_cc_loss)
        # cc_loss = self.lambda_x_cc*x_cc_loss + self.lambda_h_cc*h_cc_loss
        print()
        # final_gen_coords = generated_molecule.ndata['x_cc']
        # true_coords = generated_molecule.ndata['x_true']
        # rdkit_coords = rdkit_reference.ndata['x_ref']
        ar_mse, global_mse, ar_dist_loss = self.coordinate_loss(dec_results, generated_molecule, align =  True)
        print()
        print("[Loss Func] Auto Regressive MSE", ar_mse)
        print("[Loss Func] Kabsch Align MSE", global_mse)
        print("[Loss Func] step", step, "KL Loss", kl_loss)
        print()
        loss =  self.lambda_ar_mse*ar_mse + self.lambda_global_mse*global_mse + kl_loss #+ cc_loss
        results, geom_losses, geom_loss_cg, full_trajectory, full_trajectory_cg = enc_out
        
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
        distance_loss = self.lambda_distance*self.distance_loss(generated_molecule, geometry_graph)
        ar_dist_loss = self.lambda_ar_distance*ar_dist_loss
        print("[Loss Func] distance", distance_loss)
        print("[Loss Func] ar distance", ar_dist_loss)
        loss += distance_loss + ar_dist_loss
        
        loss_results = {
            'kl': kl_loss.cpu(),
            'global_distance': distance_loss.cpu(),
            'ar_distance': ar_dist_loss.cpu(),
            'global_mse': self.lambda_global_mse*global_mse.cpu(),
            'ar_mse': self.lambda_ar_mse*ar_mse.cpu(),
            'channel_selection_coords_align': x_cc_loss.cpu(),
            'rdkit_aligned_mse': rdkit_loss.cpu(),
            'L2 Norm Squared Posterior LogV': l2_v.cpu(),
            'L2 Norm Squared Posterior Mean': l2_v2.cpu(),
            'L2 Norm Squared Prior LogV': l2_vp.cpu(),
            'L2 Norm Squared Prior Mean': l2_vp2.cpu(),
            'L2 Norm Squared (Posterior - Prior) Mean': l2_d.cpu(),
        }
        return loss, loss_results #(ar_mse.cpu(), final_align_rmsd.cpu(), kl_loss.cpu(), x_cc_loss.cpu(), h_cc_loss.cpu(),l2_v.cpu(), l2_v2.cpu(), l2_vp.cpu(), l2_vp2.cpu(), l2_d.cpu(), rdkit_loss.cpu(), distance_loss.cpu(), ar_dist_loss.cpu())

    def std(self, input):
        if self.kl_softplus:
            return 1e-12 + F.softplus(input / 2)
        return 1e-12 + torch.exp(input / 2)

    def align(self, source, target):
        # Rot, trans = rigid_transform_Kabsch_3D_torch(input.T, target.T)
        # lig_coords = ((Rot @ (input).T).T + trans.squeeze())
        # Kabsch RMSD implementation below taken from EquiBind
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
    
    def rmsd(self, generated, true, align = False, no_reduction = False):
        if align:
            true = self.align(true, generated)
        if no_reduction:
            loss = self.mse_none(true, generated)
        else:
            loss = self.mse(true, generated)
        return loss
    
    def ar_loss_step(self, coords, coords_ref, chunks, condition_coords, condition_coords_ref, chunk_condition, align = False, step = 1, first_step = 1):
        loss = 0
        start = 0
        bottom_up = self.loss_params['ar_loss_bottom_up']
        if condition_coords is not None and bottom_up:
            start_A, start_B = 0, 0
            for chunk_A, chunk_B in zip(chunks, chunk_condition):
                A, A_true = coords[start_A: start_A + chunk_A, :], coords_ref[start_A:start_A+chunk_A, :]
                B, B_true = condition_coords[start_B: start_B + chunk_B, :], condition_coords_ref[start_B:start_B+chunk_B, :]
                if A.shape[0] == 2: # when we force reference we can remove the reference form B since its in A
                    b_rows = B.shape[0]
                    common_rows = torch.all(torch.eq(B_true[:, None, :], A_true[None, :, :]), dim=-1).any(dim=-1)
                    B, B_true = B[~common_rows], B_true[~common_rows]
                    assert(B.shape[0] == B_true.shape[0] and (B.shape[0] == b_rows - 1 or B.shape[0] == b_rows))
                AB = torch.cat([A, B], dim = 0)
                AB_true = torch.cat([A_true, B_true], dim = 0)
                unmasked_loss = self.rmsd(AB, AB_true, align, True)
                mask = torch.cat([torch.ones_like(A), torch.zeros_like(B)], dim=0)
                masked_loss = torch.masked_select(unmasked_loss, mask.bool()).mean()
                loss += masked_loss
                start_A += chunk_A
                start_B += chunk_B
                print("       AR loss and A shape, B shape", masked_loss.cpu().item(), A.shape, B.shape)
        else:
            for chunk in chunks:
                sub_loss = self.rmsd(coords[start: start + chunk, :], coords_ref[start:start+chunk, :], align)
                print("       ", sub_loss.cpu().item(), coords[start: start + chunk, :].shape)
                if coords[start: start + chunk, :].shape[0] == 1 or sub_loss.cpu().item()>3:
                    print("       \n", coords[start: start + chunk, :], coords_ref[start: start + chunk, :])
                loss += sub_loss
                start += chunk
        # if step == 0:
        #     loss *= first_step
        return loss

    def coordinate_loss(self, dec_results, generated_molecule, align = False):# = None, final_gen_coords = None, true_coords = None):
        #TODO: implement new losses with correct hydra control
        loss = 0
        dist_losses = 0
        for step, info in enumerate(dec_results):
            coords_A, h_feats_A, coords_B, h_feats_B, geom_losses, _, id_batch, ref_coords_A, ref_coords_B, ref_coords_B_split, gen_input, model_predicted_B, model_predicted_B_split, dist_loss = info
            num_molecule_chunks = [len(x) for x in id_batch if x is not None]
            num_molecule_chunks_condition = [x.shape[0] for x in ref_coords_B_split] if step > 0 else None # first step has no conditioning
            # import ipdb; ipdb.set_trace()
            print("Chunks", num_molecule_chunks, num_molecule_chunks_condition)
            ga = self.ar_loss_step(coords_A, ref_coords_A, num_molecule_chunks, model_predicted_B, ref_coords_B, num_molecule_chunks_condition, align, step)
            print("Generative", coords_A.shape, ga)
            print("Dist Loss", dist_loss, "\n")
            loss += ga
            dist_losses += dist_loss
            # TODO: do we want to do this kind of forced identity function? 
            # if coords_B is None:
            #     assert(step == 0)
            #     print()
            #     continue
            # num_molecule_chunks = [x.shape[0] for x in ref_coords_B_split]
            # arc = self.ar_loss_step(coords_B, ref_coords_B, num_molecule_chunks, align)
            # print("AR Consistency", coords_B.shape, arc)
            # if torch.gt(arc, 1000).any() or torch.isinf(arc).any() or torch.isnan(arc).any():
            #     print("Chunks", num_molecule_chunks)
            #     print("conditional input", ar_con_input)
            #     print("Bad Coordinate Check B", coords_B)
            # print()
            # loss += arc
        align_loss = [self.rmsd(m.ndata['x_cc'],m.ndata['x_true'], align = True) for m in dgl.unbatch(generated_molecule)]
        print("Align MSE Loss", align_loss)
        align_loss = sum(align_loss)
        return loss, align_loss, dist_losses
    
    def forward_vae(self, A_graph, B_graph, geometry_graph_A, geometry_graph_B, A_pool, B_pool, A_cg, B_cg, geometry_graph_A_cg, geometry_graph_B_cg, epoch):
        (v_A, h_A), (v_B, h_B), geom_losses, geom_loss_cg, full_trajectory, full_trajectory_cg = self.encoder(A_graph, B_graph, geometry_graph_A, geometry_graph_B, A_pool, B_pool, A_cg, B_cg, geometry_graph_A_cg, geometry_graph_B_cg, epoch)
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

        if self.kl_prior_logvar_clamp > -1:
            prior_logvar_V = torch.clamp(prior_logvar_V, max = self.kl_prior_logvar_clamp)

        posterior_mean_V = self.posterior_mean_V(posterior_input_V)
        # print("[Encoder] pre BN posterior mean V", torch.min(posterior_mean_V).item(), torch.max(posterior_mean_V).item()) #, torch.sum(posterior_mean_V, dim = 1))
        # posterior_mean_V = self.bn(posterior_mean_V) #made blow up worse: Trying VN Batch Norm
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

        print("\n[Enocoder] prior logvar mlp weight norms", torch.norm(self.prior_logvar_V.linear.weight), torch.norm(self.prior_logvar_V.linear2.weight))
        print("\n[Enocoder] posterior logvar mlp weight norms", torch.norm(self.posterior_logvar_V.linear.weight), torch.norm(self.posterior_logvar_V.linear2.weight))
        Z_V = self.reparameterize(posterior_mean_V, posterior_logvar_V)
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
        if self.kl_softplus:
            sigma = 1e-12 + F.softplus(scale*logvar / 2)
        else:
            sigma = 1e-12 + torch.exp(scale*logvar / 2)
        eps = torch.randn_like(mean)
        return mean + eps*sigma

    # https://github.com/NVIDIA/NeMo/blob/b9cf05cf76496b57867d39308028c60fef7cb1ba/nemo/collections/nlp/models/machine_translation/mt_enc_dec_bottleneck_model.py#L217
    def kl(self, z_mean, z_logvar, z_mean_prior, z_logvar_prior, coordinates = False):
        free_bits_per_dim = self.kl_free_bits/z_mean[0].numel()
        if self.kl_softplus:
            p_std = 1e-12 + F.softplus(z_logvar / 2)
            q_std = 1e-12 + F.softplus(z_logvar_prior / 2)
        else:
            p_std = 1e-12 + torch.exp(z_logvar / 2)
            q_std = 1e-12 + torch.exp(z_logvar_prior / 2)
        q_mean = z_mean_prior
        p_mean = z_mean
        var_ratio = (p_std / q_std).pow(2)
        t1 = ((p_mean - q_mean) / q_std).pow(2)
        kl =  0.5 * (var_ratio + t1 - 1 - var_ratio.log()) # shape = number of CG beads
        kl = torch.clamp(kl, min = free_bits_per_dim)
        kl = kl.sum(-1)
        if coordinates:
            kl = kl.sum(-1)
        return kl.mean()
            
