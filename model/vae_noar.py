from encoder_no_ar import *
from decoder_no_ar import *

class VAENoAr(nn.Module):
    def __init__(self, kl_params, encoder_params, decoder_params, loss_params, coordinate_type, device = "cuda"):
        super(VAENoAr, self).__init__()
        self.encoder = Encoder(**encoder_params) #.to(device)
        self.decoder = DecoderNoAR(self.encoder.atom_embedder, coordinate_type, **decoder_params) #.to(device)
        self.mse = nn.MSELoss()
        self.mse_none = nn.MSELoss(reduction ='none')
        self.device = device
        F = encoder_params["coord_F_dim"]
        D = encoder_params["latent_dim"]
        
        self.kl_free_bits = kl_params['kl_free_bits']
        self.kl_prior_logvar_clamp = kl_params['kl_prior_logvar_clamp']
        self.kl_softplus = kl_params['kl_softplus']
        # self.use_mim = kl_params['use_mim']

        self.kl_v_beta = loss_params['kl_weight']
        self.kl_lambda = 1e-8
        # self.kl_h_beta = 0
        # self.kl_reg_beta = 1
        self.lambda_global_mse = loss_params['global_mse_weight']
        # self.lambda_ar_mse = loss_params['ar_mse_weight']
        self.lambda_x_cc = loss_params['x_cc_weight']
        # self.lambda_h_cc = loss_params['h_cc_weight']
        self.lambda_distance = loss_params['distance_weight']
        self.lambda_angle = 1
        self.lambda_dihedral = 1
        # self.lambda_ar_distance = loss_params['ar_distance_weight']
        # self.ar_loss_direction = loss_params['ar_loss_bottom_up']
        self.loss_params = loss_params

        self.posterior_mean_V = nn.Sequential(VN_MLP(2*F, 2*F, 2*F, 2*F, use_batchnorm = False), VN_MLP(2*F, F, F, F, use_batchnorm = False))
        self.posterior_logvar_V = Scalar_MLP(2*F*3, 2*F*3, F, use_batchnorm = False)# need to flatten to get equivariant noise N x F x 1
        self.prior_mean_V = nn.Sequential(VN_MLP(F, F, F, F, use_batchnorm = False), VN_MLP(F, F, F, F, use_batchnorm = False))
        self.prior_logvar_V = Scalar_MLP(F*3, F*3, F, use_batchnorm = False)

    def forward(self, frag_ids, A_graph, B_graph, geometry_graph_A, geometry_graph_B, A_pool, B_pool, A_cg, B_cg, geometry_graph_A_cg, geometry_graph_B_cg, epoch, validation = False):
        enc_out = self.forward_vae(A_graph, B_graph, geometry_graph_A, geometry_graph_B, A_pool, B_pool, A_cg, B_cg, geometry_graph_A_cg, geometry_graph_B_cg, epoch, validation)
        results = enc_out
        natoms = A_graph.batch_num_nodes()
        nbeads = A_cg.batch_num_nodes()
        kl_v, kl_v_un_clamped = torch.tensor([0]).to(results["prior_mean_V"].device), torch.tensor([0]).to(results["prior_mean_V"].device)
        if not validation:
            kl_v, kl_v_un_clamped = self.kl(results["posterior_mean_V"], results["posterior_logvar_V"], results["prior_mean_V"], results["prior_logvar_V"], natoms, nbeads, coordinates = True)
            # import ipdb; ipdb.set_trace()
            dec_out = self.decoder(A_cg, B_graph, frag_ids, geometry_graph_A, geometry_graph_B)
        else:
            dec_out = self.decoder(B_cg, B_graph, frag_ids, geometry_graph_B, geometry_graph_B)
        kl_h = 0
        generated_molecule, rdkit_reference, (coords_A, h_feats_A, coords_B, h_feats_B, ref_coords_A, X_cc, H_cc) = dec_out
        return generated_molecule, rdkit_reference, (coords_A, h_feats_A, coords_B, h_feats_B, ref_coords_A), (X_cc, H_cc), (kl_v, kl_h, kl_v_un_clamped), enc_out
    
    def distance_loss(self, generated_molecule, geometry_graphs, key = "x_cc"):
        geom_loss = []
        for geometry_graph, generated_mol in zip(dgl.unbatch(geometry_graphs), dgl.unbatch(generated_molecule)):
            src, dst = geometry_graph.edges()
            src = src.long()
            dst = dst.long()
            generated_coords = generated_mol.ndata[key]
            d_squared = torch.sum((generated_coords[src] - generated_coords[dst]) ** 2, dim=1)
            geom_loss.append(1/len(src) * torch.sum((d_squared - geometry_graph.edata['feat'] ** 2) ** 2).unsqueeze(0)) #TODO: scaling hurt performance
        return torch.mean(torch.cat(geom_loss))

    
    def calc_torsion(self, pos, k):
        #  Code from geo mol https://github.com/PattanaikL/GeoMol/blob/main/model/utils.py#L189C1-L199C51
        # TODO this works for batches so do this
        p0, p1, p2, p3 = k
        p0, p1, p2, p3 = pos[p0], pos[p1], pos[p2], pos[p3]
        s1 = p1 - p0
        s2 = p2 - p1
        s3 = p3 - p2
        sin_d_ = torch.linalg.norm(s2, dim=-1) * torch.sum(s1 * torch.cross(s2, s3, dim=-1), dim=-1)
        cos_d_ = torch.sum(torch.cross(s1, s2, dim=-1) * torch.cross(s2, s3, dim=-1), dim=-1)
        return torch.atan2(sin_d_, cos_d_ + 1e-10)
    
    def calc_dihedral(self, pos, k): #a, b):
        """
        Compute angle between two batches of input vectors in radians
        """
        # TODO make this work for batches
        a,b,c = k
        a_pos = pos[a]
        b_pos = pos[b]
        c_pos = pos[c]
    
        AB = a_pos - b_pos
        CB = c_pos - b_pos
        inner_product = (AB * CB).sum(dim=-1)
        # norms
        # ABn = torch.linalg.norm(AB, dim=-1)
        # BCn = torch.linalg.norm(AB, dim=-1)
        # protect denominator during division
        den = torch.sqrt((AB*AB).sum(-1)*(CB*CB).sum(-1)) + 1e-10
        cos = inner_product / den
        cos = torch.clamp(cos, min=-1.0, max=1.0) # some values are -1.005
        if torch.any(torch.isnan(cos)) or torch.any(torch.isnan(torch.acos(cos))):
            import ipdb; ipdb.set_trace()
            test= 1
        return torch.acos(cos)
    
    def angle_loss(self, generated_molecule, angles, key = 'x_cc'):
        predicted_torsion_angles, true_torsion_angles = [], []
        predicted_angles, true_angles = [], []
        for idx, gmol in enumerate(dgl.unbatch(generated_molecule)):
            torsion_angles, dihedral_angles = angles[idx]
            pos = gmol.ndata[key] #TODO CGECK IF TRUE ANGLES LINES UP IN CALCULATION
            W,X,Y,Z = [], [], [], []
            # import ipdb; ipdb.set_trace()
            for k, v in torsion_angles.items():
                # predicted_torsion_angles.append(self.calc_torsion(pos, k))
                w,x,y,z = k
                W.append(w)
                X.append(x)
                Y.append(y)
                Z.append(z)
                true_torsion_angles.append(v)
            # import ipdb; ipdb.set_trace()
            predicted_torsion_angles.append(self.calc_torsion(pos, (W,X,Y,Z)))
            assert(predicted_torsion_angles[-1].shape[0] == len(torsion_angles))
            # X,Y,Z = [], [], []
            # for k, v in dihedral_angles.items():
            #     x,y,z = k
            #     X.append(x)
            #     Y.append(y)
            #     Z.append(z)
            #     true_angles.append(v)
            # predicted_angles.append(self.calc_dihedral(pos, (X,Y,Z)))
            # assert(predicted_angles[-1].shape[0] == len(dihedral_angles))
        
        predicted_torsion_angles = torch.cat(predicted_torsion_angles, dim = 0)
        true_torsion_angles = torch.tensor(true_torsion_angles).cuda()
        # predicted_angles = torch.cat(predicted_angles, dim = 0)
        # true_angles = torch.tensor(true_angles).cuda()
        cosine_torsion_difference = torch.cos(predicted_torsion_angles - true_torsion_angles)
        # cosine_difference = torch.cos(predicted_angles - true_angles)
        # import ipdb; ipdb.set_trace()
        return 1-torch.mean(cosine_torsion_difference), None #1-torch.mean(cosine_difference)
            
    def loss_function(self, generated_molecule, rdkit_reference, channel_selection_info, KL_terms, enc_out, geometry_graph, angles, log_latent_stats = True):
        kl_v, kl_h, kl_v_unclamped = KL_terms
        results = enc_out
        x_cc, h_cc = channel_selection_info
        if log_latent_stats:
            l2_v = torch.norm(self.std(results["posterior_logvar_V"]), 2)**2
            l2_v2 = torch.norm(results["posterior_mean_V"], 2)**2
            l2_vp = torch.norm(self.std(results["prior_logvar_V"]), 2)**2
            l2_vp2 = torch.norm(results["prior_mean_V"], 2)**2
            l2_d = torch.norm(results["posterior_mean_V"]-results["prior_mean_V"], 2)**2

        # kl_loss = self.kl_v_beta*kl_v
        kl_loss = self.kl_lambda*kl_v
        if log_latent_stats:
            # import ipdb; ipdb.set_trace()
            self.update_adaptive_lambda(kl_v.item())
        
        global_mse_loss = self.lambda_global_mse*self.coordinate_loss(generated_molecule)
        global_mse_loss_unmasked = self.lambda_global_mse*self.coordinate_loss(generated_molecule, mask_Hs= False)
        # global_mse_rdkit_loss = torch.cat([self.rmsd(m.ndata['x_cc'], rd.ndata['x_ref'], align = True).unsqueeze(0) for (m, rd) in zip(dgl.unbatch(generated_molecule), dgl.unbatch(rdkit_reference))]).mean()
        rdkit_loss = [self.rmsd(m.ndata['x_ref'], m.ndata['x_true'], align = True).unsqueeze(0) for m in dgl.unbatch(rdkit_reference)]
        rdkit_loss = self.lambda_global_mse*torch.cat(rdkit_loss).mean()
        
        distance_loss = self.distance_loss(generated_molecule, geometry_graph)
        # distance_loss_rd = self.distance_loss(generated_molecule, geometry_graph, key='x_ref')
        # distance_loss_true = self.distance_loss(generated_molecule, geometry_graph, key='x_true')
        scaled_distance_loss = self.lambda_distance*distance_loss
        
        ta_loss, _ = self.angle_loss(generated_molecule, angles)
        # ta_loss_rd, _ = self.angle_loss(generated_molecule, angles, key='x_ref')
        # ta_loss_true, _ = self.angle_loss(generated_molecule, angles, key='x_true')
        angle_loss = self.lambda_distance*self.lambda_angle*ta_loss #+ self.lambda_dihedral*di_loss
        # import ipdb; ipdb.set_trace()
        loss =  global_mse_loss + kl_loss + scaled_distance_loss + angle_loss #+ 0.2*global_mse_loss_unmasked this made it worse
        # loss = 10*ta_loss + kl_loss
        
        if log_latent_stats:
            loss_results = {
                'latent reg loss': kl_loss.item(),
                'kl_unclamped': self.kl_v_beta*kl_v_unclamped.item(),
                'global_distance': distance_loss.item(),
                # 'global_distance_rdkit': distance_loss_rd.item(),
                # 'global_distance_true': distance_loss_true.item(),
                'scaled_distance_loss':scaled_distance_loss.item(),
                'global_mse': global_mse_loss.item(),
                'global_mse_unmasked': global_mse_loss_unmasked.item(),
                # 'global_mse_rdkit': global_mse_rdkit_loss.item(),
                'torsion_angle_loss': ta_loss.item(),
                # 'torsion_angle_loss_rdkit': ta_loss_rd.item(),
                # 'torsion_angle_loss_true': ta_loss_true.item(),
                'torsion_angle_loss_scaled': angle_loss.item(),
                # 'dihedral_angle_loss': self.lambda_dihedral*di_loss.item(),
                # 'channel_selection_coords_align': x_cc_loss.item(),
                'rdkit_aligned_mse': rdkit_loss.item(),
                'L2 Norm Squared Posterior LogV': l2_v.item(),
                'L2 Norm Squared Posterior Mean': l2_v2.item(),
                'L2 Norm Squared Prior LogV': l2_vp.item(),
                'L2 Norm Squared Prior Mean': l2_vp2.item(),
                'L2 Norm Squared (Posterior - Prior) Mean': l2_d.item(),
                'unscaled kl': kl_v.item(),
                'unscaled unclamped kl': kl_v_unclamped.item(),
                # 'beta_kl': self.kl_v_beta,
                'adaptive kl lambda': self.kl_lambda
            }
        else:
            loss_results = {
                'latent reg loss': kl_loss.item(),
                'kl_unclamped': self.kl_v_beta*kl_v_unclamped.item(),
                'global_mse': global_mse_loss.item(),
                'global_mse_unmasked': global_mse_loss_unmasked.item(),
                # 'global_mse_rdkit': global_mse_rdkit_loss.item(),
                'torsion_angle_loss': ta_loss.item(),
                # 'torsion_angle_loss_rdkit': ta_loss_rd.item(),
                # 'torsion_angle_loss_true': ta_loss_true.item(),
                'torsion_angle_loss_scaled': angle_loss.item(),
                # 'dihedral_angle_loss': self.lambda_dihedral*di_loss.item(),
                'rdkit_aligned_mse': rdkit_loss.item(),
                'unscaled kl': kl_v.item(),
                'unscaled unclamped kl': kl_v_unclamped.item(),
                # 'beta_kl': self.kl_v_beta,
            }
        return loss, loss_results 
    
    def std(self, input):
        if self.kl_softplus:
            return 1e-6 + F.softplus(input / 2)
        return 1e-12 + torch.exp(input / 2)

    def align(self, source, target):
        with torch.no_grad():
            lig_coords_pred = target
            lig_coords = source
            if source.shape[0] == 1:
                return source
            lig_coords_pred_mean = lig_coords_pred.mean(dim=0, keepdim=True)  # (1,3)
            lig_coords_mean = lig_coords.mean(dim=0, keepdim=True)  # (1,3)

            A = (lig_coords_pred - lig_coords_pred_mean).transpose(0, 1) @ (lig_coords - lig_coords_mean) 
            A = A + torch.eye(A.shape[0]).to(A.device) * 1e-5 #added noise to help with gradients
            if torch.isnan(A).any() or torch.isinf(A).any():
                print("\n\n\n\n\n\n\n\n\n\nThe SVD tensor contains NaN or Inf values")
                # import ipdb; ipdb.set_trace()
            U, S, Vt = torch.linalg.svd(A)
            # corr_mat = torch.diag(1e-7 + torch.tensor([1, 1, torch.sign(torch.det(A))], device=lig_coords_pred.device))
            corr_mat = torch.diag(torch.tensor([1, 1, torch.sign(torch.det(A))], device=lig_coords_pred.device))
            rotation = (U @ corr_mat) @ Vt
            translation = lig_coords_pred_mean - torch.t(rotation @ lig_coords_mean.t())  # (1,3)
        return (rotation @ lig_coords.t()).t() + translation
    
    def rmsd(self, generated, true, mask = None, align = False, no_reduction = False):
        if mask is not None:
            rows_with_Hs = torch.where(mask[:, 0] == 1)[0]
            generated = generated[~rows_with_Hs]
            true = true[~rows_with_Hs]
        if align:
            true = self.align(true, generated)
        if no_reduction:
            loss = self.mse_none(true, generated)
        else:
            loss = self.mse(true, generated) #TODO this should have reduction sum change also for rdkit
        return loss

    def coordinate_loss(self, generated_molecule, mask_Hs = True):
        if mask_Hs:
            cutoff = 35 # For drugs and 5 for qm9
            global_mse_loss = [self.rmsd(m.ndata['x_cc'], m.ndata['x_true'], m.ndata['ref_feat'][:, :cutoff], align = True).unsqueeze(0) for m in dgl.unbatch(generated_molecule)]
        else:
            global_mse_loss = [self.rmsd(m.ndata['x_cc'], m.ndata['x_true'], align = True).unsqueeze(0) for m in dgl.unbatch(generated_molecule)]
        global_mse_loss = torch.cat(global_mse_loss).mean()
        return global_mse_loss
    
    def forward_vae(self, A_graph, B_graph, geometry_graph_A, geometry_graph_B, A_pool, B_pool, A_cg, B_cg, geometry_graph_A_cg, geometry_graph_B_cg, epoch, validation = False):
        (v_A, h_A), (v_B, h_B) = self.encoder(A_graph, B_graph, geometry_graph_A, geometry_graph_B, A_pool, B_pool, A_cg, B_cg, geometry_graph_A_cg, geometry_graph_B_cg, epoch, prior_only = validation)
        if not validation:
            posterior_input_V = torch.cat((v_A, v_B), dim = 1) # N x 2F x 3
            posterior_mean_V = self.posterior_mean_V(posterior_input_V)
            posterior_logvar_V = self.posterior_logvar_V(posterior_input_V.reshape((posterior_input_V.shape[0], -1))).unsqueeze(2)

        prior_mean_V = self.prior_mean_V(v_B)
        # ! Setting Prior log var to 0 so std = 1 no more clamping
        prior_logvar_V = self.prior_logvar_V(v_B.reshape(v_B.shape[0],-1)).unsqueeze(2) #TODO check if this works
        # prior_logvar_V = torch.zeros((v_B.shape[0], v_B.shape[1])).unsqueeze(2).to(v_B.device) # N x F x 1

        if validation:
            Z_V = self.reparameterize(prior_mean_V, prior_logvar_V, mean_only=True)
            # Z_h = self.reparameterize(prior_mean_h, prior_logvar_h, mean_only=True)
        else:
            Z_V = self.reparameterize(posterior_mean_V, posterior_logvar_V)
            # Z_h = self.reparameterize(posterior_mean_h, posterior_logvar_h)

        A_cg.ndata["Z_V"] = Z_V
        # A_cg.ndata["Z_h"] = Z_h
        B_cg.ndata["Z_V"] = Z_V
        # B_cg.ndata["Z_h"] = Z_h

        results = {
            "Z_V": Z_V,
            # "Z_h": Z_h,
            "v_A": v_A,
            "v_B": v_B,
            "h_A": h_A,
            "h_B": h_B,

            "prior_mean_V": prior_mean_V,
            # "prior_mean_h": prior_mean_h,
            "prior_logvar_V": prior_logvar_V,
            # "prior_logvar_h": prior_logvar_h,
        }
        if not validation:
            results.update({
            "posterior_mean_V": posterior_mean_V,
            # "posterior_mean_h": posterior_mean_h,
            "posterior_logvar_V": posterior_logvar_V,
            # "posterior_logvar_h": posterior_logvar_h,
            })
        return results

    def reparameterize(self, mean, logvar, scale = 1.0, mean_only = False):
        if mean_only:
            return mean
        if self.kl_softplus:
            sigma = 1e-6 + F.softplus(scale*logvar / 2)
        else:
            sigma = 1e-12 + torch.exp(scale*logvar / 2)
        eps = torch.randn_like(mean)
        return mean + eps*sigma

    # https://github.com/NVIDIA/NeMo/blob/b9cf05cf76496b57867d39308028c60fef7cb1ba/nemo/collections/nlp/models/machine_translation/mt_enc_dec_bottleneck_model.py#L217
    def kl(self, z_mean, z_logvar, z_mean_prior, z_logvar_prior, natoms, nbeads, coordinates = False):
        assert len(natoms) == len(nbeads)
        # ! Look into budget per molecule
        free_bits_per_dim = self.kl_free_bits/z_mean[0].numel()
        if self.kl_softplus:
            p_std = 1e-6 + F.softplus(z_logvar / 2)
            q_std = 1e-6 + F.softplus(z_logvar_prior / 2)
        else:
            p_std = 1e-12 + torch.exp(z_logvar / 2)
            q_std = 1e-12 + torch.exp(z_logvar_prior / 2)
        q_mean = z_mean_prior
        p_mean = z_mean
        var_ratio = (p_std / q_std).pow(2)
        t1 = ((p_mean - q_mean) / q_std).pow(2)
        kl =  0.5 * (var_ratio + t1 - 1 - var_ratio.log()) # shape = number of CG beads
        pre_clamp_kl = kl
        kl = torch.clamp(kl, min = free_bits_per_dim)
        kl = kl.sum(-1)
        if coordinates:
            kl = kl.sum(-1)
        # Here kl is [N]
        # return kl.mean()
        return self.kl_loss(kl, natoms, nbeads), self.kl_loss(pre_clamp_kl, natoms, nbeads)
    
    def kl_loss(self, kl, natoms, nbeads):
        B = len(natoms)
        start = 0
        loss = []
        for atom, coarse in zip(natoms, nbeads):
            kl_chunk = kl[start: start + coarse].sum().unsqueeze(0)
            loss.append(1/atom * kl_chunk)
            start += coarse
        # import ipdb; ipdb.set_trace()
        total_loss = 1/B * torch.sum(torch.cat(loss))
        return total_loss
    
    def kl_built_in(self, z_mean, z_logvar, z_mean_prior, z_logvar_prior, natoms, nbeads, coordinates = False):
        assert len(natoms) == len(nbeads)
        free_bits_per_dim = self.kl_free_bits/z_mean[0].numel()
        if self.kl_softplus:
            posterior_std = 1e-12 + F.softplus(z_logvar / 2)
            prior_std = 1e-12 + F.softplus(z_logvar_prior / 2)
        else:
            posterior_std = 1e-12 + torch.exp(z_logvar / 2)
            prior_std = 1e-12 + torch.exp(z_logvar_prior / 2)
        posterior = torch.distributions.Normal(loc = z_mean, scale = posterior_std)
        prior = torch.distributions.Normal(loc = z_mean_prior, scale = prior_std)
        pre_clamp_kl = torch.distributions.kl.kl_divergence(posterior, prior)
        kl = torch.clamp(pre_clamp_kl, min = free_bits_per_dim)
        kl = kl.sum(-1)
        if coordinates:
            kl = kl.sum(-1)
        # Here kl is [N]
        return self.kl_loss(kl, natoms, nbeads), self.kl_loss(pre_clamp_kl, natoms, nbeads)
    
    def mim(self, z, z_mean, z_logvar, z_mean_prior, z_logvar_prior, natoms, nbeads, coordinates = False):
        assert len(natoms) == len(nbeads)
        free_bits_per_dim = self.kl_free_bits/z_mean[0].numel()
        z_logvar = torch.clamp(z_logvar, min = -6) #! minimum uncertainty
        if self.kl_softplus:
            posterior_std = 1e-12 + F.softplus(z_logvar / 2)
            prior_std = 1e-12 + F.softplus(z_logvar_prior / 2)
        else:
            posterior_std = 1e-12 + torch.exp(z_logvar / 2)
            prior_std = 1e-12 + torch.exp(z_logvar_prior / 2)
        posterior = torch.distributions.Normal(loc = z_mean, scale = posterior_std)
        prior = torch.distributions.Normal(loc = z_mean_prior, scale = prior_std)
        log_q_z_given_x = self.kl_loss(posterior.log_prob(z).sum(-1).sum(-1), natoms, nbeads) #.sum(-1).sum(-1).mean()
        log_p_z = self.kl_loss(prior.log_prob(z).sum(-1).sum(-1), natoms, nbeads)
        loss_terms = -0.5 * (log_q_z_given_x + log_p_z)
        return loss_terms

    def update_adaptive_lambda(
        self,
        # current_lambda,
        current_rate,
        target_rate=0.5,
        delta=1.0e-3,
        epsilon=1.0e-2,
        lower_limit=1e-10,
        upper_limit=1e10,
    ):
        current_lambda = self.kl_lambda
        if current_rate > (1 + epsilon) * target_rate:
            out = current_lambda * (1 + delta)
        elif current_rate < 1 / (1 + epsilon) * target_rate:
            out = current_lambda * 1 / (1 + delta)
        else:
            out = current_lambda
        # self.kl_lambda = torch.clamp(out, min=lower_limit, max=upper_limit)
        self.kl_lambda = out
        if out < lower_limit:
            self.kl_lambda = lower_limit
        elif out > upper_limit:
            self.kl_lambda = upper_limit
        return self.kl_lambda
