from encoder_no_ar import *
from decoder_no_ar import *
import math
from scipy.spatial.transform import Rotation as R
import numpy as np
import ot

class VAENoArAngleOt(nn.Module):
    def __init__(self, kl_params, encoder_params, decoder_params, loss_params, coordinate_type, device = "cuda"):
        super(VAENoArAngleOt, self).__init__()
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
        self.kl_lambda = 1e-8 #1e-8 #! -8 for DRUGS
        self.mask_cut = 35 #35 # QM9 = 5 DRUGS=35
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
        angle_feat_dim = 7*D+1#+12
        # self.angle_predictor = nn.Sequential(Scalar_MLP(angle_feat_dim, 2*angle_feat_dim, 2*angle_feat_dim, use_batchnorm = False), Scalar_MLP(2*angle_feat_dim, angle_feat_dim, 1, use_batchnorm = False))
        self.angle_predictor = nn.Sequential(Scalar_MLP(angle_feat_dim, 2*angle_feat_dim, 3*angle_feat_dim, use_batchnorm = True),
                                             Scalar_MLP(3*angle_feat_dim, 3*angle_feat_dim, 2*angle_feat_dim, use_batchnorm = True),
                                             Scalar_MLP(2*angle_feat_dim, angle_feat_dim, 2, use_batchnorm = True))
    
    def new_angle_loss(self, y_pred, y_true, weight_reg=1):
        # Calculate the cosine difference between predicted and target angles
        cos_diff = torch.cos(y_true - y_pred)
        print("dtheta: ", y_pred[:20])
        print(torch.mean(y_pred))
        print("dtheta*: ", y_true[:20])
        print(torch.mean(y_true))
        print("cos diff: ", cos_diff[:20])
        print("1 - cos diff mean: ",1-torch.mean(cos_diff))
        loss = 1 - cos_diff
        # penalty_0 = torch.pi/ (torch.abs((y_pred - torch.pi)%(2*math.pi)) + 1e-3)
        penalty_0 = torch.abs(-(1+1e-3) - torch.pi/(y_pred - torch.pi + 1e-3) - torch.pi/(-y_pred - torch.pi + 1e-3))
        # -π/(-x - π + 0.001) - π/(x - π + 0.001) - 1 - 0.001
        penalty_pi = torch.pi/ (torch.abs(y_pred) + 1e-3)
        threshold_upper = math.pi/2
        threshold_lower = -math.pi/2
        # Create boolean masks for the conditions
        mask_upper = y_true <= threshold_upper
        mask_lower = y_true >= threshold_lower
        penalty = torch.where(mask_upper & mask_lower, penalty_0, penalty_pi)
        print("penalty: ", penalty[:20])
        print("penalty_0: ", penalty_0[:20])
        print("penalty_pi: ", penalty_pi[:20])
        # exceeded_threshold = (torch.abs(self.regression_outputs) > (2*math.pi)).float()
        # inside_range = torch.logical_and(self.regression_outputs >= -2*math.pi, self.regression_outputs <= 2*math.pi)
        # weight_reg = ((torch.abs(self.regression_outputs) - (2*math.pi)) ** 2) * exceeded_threshold
        loss = loss * penalty 
        print("loss:", loss[:20])
        # print("weight reg:", weight_reg[:20])
        print(torch.mean(loss))#, torch.mean(1e-5*weight_reg))
        if torch.mean(loss).item() < 0:
            import ipdb; ipdb.set_trace()
        return torch.mean(loss)
        # return torch.mean(loss[inside_range]) + 10*torch.mean(loss[~inside_range])  #+ torch.mean(1e-5*weight_reg) +  torch.mean(1-cos_diff)#
    
    def build_angle_batch(self, A_graph, angle_A, frag_ids):
        predicted_torsion_angles, true_torsion_angles = [], []
        predicted_angles, true_angles = [], []
        rd_feats, rd_feats_flipped = [], []
        count = 0
        for idx, gmol in enumerate(dgl.unbatch(A_graph)):
            torsion_angles, dihedral_angles = angle_A[idx]
            chunking = frag_ids[idx]
            pos = gmol.ndata['x_cc']
            W,X,Y,Z = [], [], [], []
            molecule_feat = gmol.ndata['feat_cc'].mean(0).reshape(-1) # (D, )
            for k, v in torsion_angles.items():
                # predicted_torsion_angles.append(self.calc_torsion(pos, k))
                feats = [molecule_feat]# Starting with the entire molecules
                feats_flipped = [molecule_feat]
                w,x,y,z = k
                W.append(w)
                X.append(x)
                Y.append(y)
                Z.append(z)
                true_torsion_angles.append(v)
                chunk_feats = []
                for chunk in chunking:
                    if x in chunk:
                        chunk_feats.append(gmol.ndata['feat_cc'][list(chunk), :].mean(0).reshape(-1))
                    elif y in chunk:
                        chunk_feats.append(gmol.ndata['feat_cc'][list(chunk), :].mean(0).reshape(-1))
                assert(len(chunk_feats) == 2)
                feats.extend(chunk_feats) # Adding features of TA substructure spliots
                feats_flipped.extend([chunk_feats[-1], chunk_feats[0]])
                # locs = [gmol.ndata['x_cc'][loc, :].reshape(-1) for loc in [w,x,y,z]] # 3D points for TA
                pfeats = [gmol.ndata['feat_cc'][loc, :].reshape(-1) for loc in [w,x,y,z]] # features for TA atoms
                # feats.extend(locs)
                feats.extend(pfeats)
                # locs = [gmol.ndata['x_cc'][loc, :].reshape(-1) for loc in [z,y,x,w]] # 3D points for TA
                pfeats = [gmol.ndata['feat_cc'][loc, :].reshape(-1) for loc in [z,y,x,w]] # features for TA atoms
                # feats_flipped.extend(locs)
                feats_flipped.extend(pfeats)
                
                rd_feats.append(torch.cat(feats)) # (460,) tensor
                rd_feats_flipped.append(torch.cat(feats_flipped))
            
            predicted_torsion_angles.append(self.calc_torsion(pos, (W,X,Y,Z)))
            for i in range(len(W)):
                index = i + count
                rd_feats[index] = torch.cat((rd_feats[index], predicted_torsion_angles[-1][i].view(1)))
                rd_feats_flipped[index] = torch.cat((rd_feats_flipped[index], predicted_torsion_angles[-1][i].view(1)))
                
            count += len(W)
            assert(predicted_torsion_angles[-1].shape[0] == len(torsion_angles))

        predicted_torsion_angles = torch.cat(predicted_torsion_angles, dim = 0)
        true_torsion_angles = torch.tensor(true_torsion_angles).cuda()
        regression_targets = true_torsion_angles - predicted_torsion_angles #predicted_torsion_angles - true_torsion_angles
        cosine_torsion_difference = torch.cos(regression_targets)
        classification_targets = 1-cosine_torsion_difference
        classification_targets = (classification_targets >= 1).float()
        rd_feats = torch.stack(rd_feats)
        rd_feats_flipped = torch.stack(rd_feats_flipped)
        return rd_feats, rd_feats_flipped, classification_targets, regression_targets

    def angle_forward(self, generated_molecule, angles, angles_mask, frag_ids):
        rd_feats, rd_feats_flipped, classification_targets, regression_targets = self.build_angle_batch(generated_molecule, angles, frag_ids)
        angle_deltas = 0.5*self.angle_predictor(rd_feats) + 0.5*self.angle_predictor(rd_feats_flipped)
        # import ipdb; ipdb.set_trace() # 
        angle_deltas = torch.tanh(angle_deltas)
        angle_deltas = torch.atan2(angle_deltas[:,0], angle_deltas[:,1] + 1e-10)
        # angle_deltas = self.angle_predictor(rd_feats)
        print("angle deltas:", angle_deltas[:20])
        # raw_pred_delta = angles_deltas.sequeeze(1)
        # true_deltas = []
        all_true_angles = []
        # angle_deltas = angle_deltas.squeeze(1) % (2*math.pi)
        # angle_deltas = torch.tanh(angle_deltas.squeeze(1))*(math.pi)
        # print("tanh angle * pi deltas:", angle_deltas[:20])
        # angle_deltas = torch.nn.functional.gelu(angle_deltas.squeeze(1))
        # print("gelu angle deltas:", angle_deltas[:20])
        
        # angle_deltas = angle_deltas.squeeze(1) #! needed for pure regression
        self.regression_outputs = angle_deltas
        # penalty = torch.mean(penalty_weight * (y_pred - threshold) ** 2 * exceeded_threshold)
        angle_deltas = angle_deltas  % (2*math.pi) #torch.clamp(angle_deltas, -1, 1)*(math.pi)
        print("moded angle deltas:", angle_deltas[:20])
        
        count = 0
        final_molecules = []
        for idx, gmol in enumerate(dgl.unbatch(generated_molecule)):
            # edge_mask = gmol.edata['mask_edges'] # (|E|,) binary mask has a bug since its manually set to [A, a, a, A ...] not what dgl does [A, a , ..., A, a]
            pos = gmol.ndata['x_cc']
            ogpos = gmol.ndata['x_cc']
            torsion_angles, _ = angles[idx]
            chunking = frag_ids[idx]
            mask_rotate = angles_mask[idx][0].T #!0 for error in data creation (angles x nodes)
            # angle_idx = 0
            if mask_rotate.shape[0] != len(torsion_angles):
                print("Torsion Angle Discrepancy", mask_rotate.shape[0], len(torsion_angles))
            for sub_count, info in enumerate(torsion_angles.items()):
                angle_indices, true_angle = info
                all_true_angles.append(true_angle)
                a,b,c,d = angle_indices
                u, v  = None, None
                # sub_count = 0
                used_index = set()
                for angle_idx in range(len(mask_rotate)):
                    if angle_idx in used_index or angle_idx >= mask_rotate.shape[0]:
                        # sub_count += 1
                        continue
                    # try:
                    mask = mask_rotate[angle_idx]
                    # except Exception as e: #! if we are in a discrepancy case we have to make sure the mask lines up to the torsion angle
                    #     print(e)
                    #     import ipdb; ipdb.set_trace()
                    #     test = mask_rotate.shape
                # angle_idx += 1
                    u, v = c, b
                    if not mask[u] and mask[v]: # calc torsion is the same for any order 
                        u, v = c, b
                        flip = False
                        used_index.add(angle_idx)
                        break
                    elif mask[u] and not mask[v]:
                        u, v = b, c
                        flip = True
                        used_index.add(angle_idx)
                        break
                    else:
                        u, v = None, None
                        # print("Mask does not line up", angle_idx, mask_rotate.shape[0], len(torsion_angles))
                        # sub_count += 1
                        continue
                # assert not mask[u]
                # assert mask[v]
                if u is None or v is None:
                    # test = mask_rotate.shape
                    print("Torsion Angle Discrepancy: not every angle can be split so ignoring")
                    continue
                
        
                rot_vec = pos[v] - pos[u] #pos[u] - pos[v]
                # rot_vec = rot_vec * angle_deltas[count+sub_count]/ torch.norm(rot_vec) #np.linalg.norm(rot_vec)
                # pred_angle = rd_feats[count+sub_count, -1].item() % (2*math.pi)
                # og_true = true_angle
                # true_angle = true_angle % (2*math.pi)
                
                # true_delta = true_angle - pred_angle
                pred_delta = angle_deltas[count+sub_count]% (2*torch.pi)
                rot_vec = rot_vec * pred_delta/ torch.norm(rot_vec)
                # count += 1
                rot_mat = torch.tensor(R.from_rotvec(rot_vec.cpu().detach().numpy()).as_matrix()).to(pos.device).float()
                # pos[mask] = (pos[mask] - pos[v]) @ rot_mat.T + pos[v]
                # ogpos = pos[mask]
                # ognpos = pos[~mask]
                
                output = torch.zeros_like(pos).cuda()
                result = (pos[mask] - pos[v]) @ rot_mat.T + pos[v]
                output[mask] = result
                pos = pos * ~mask.reshape(-1,1).cuda() + output
                # assert(self.calc_torsion(pos, (a,b,c,d)).item() - og_true < 1e-3)
                # check = self.calc_torsion(pos, (a,b,c,d)).item()
                # if np.abs(check - og_true) > 1e-2:
                #     if np.abs(check - true_angle) > 1e-2:
                #         import ipdb; ipdb.set_trace()
                #         test = 1
                # check = self.calc_torsion(pos, (a,b,c,d)) # this gets us back to the true angle

                    
                # gpos = pos[mask]
                # npos = pos[~mask]
            test = 1
            gmol.ndata['x_cc'] = pos
            final_molecules.append(gmol)
            count += len(torsion_angles)
        self.generated_angle_deltas = (angle_deltas + torch.pi) % (2 * torch.pi) - torch.pi #! ML generated delta
        # self.true_angle_deltas =  torch.tensor(true_deltas).to(angles_deltas.device)
        #  (angle_diff + np.pi) % (2 * np.pi) - np.pi
        ctrue_angles = (torch.tensor(all_true_angles).to(rd_feats.device)  + torch.pi) % (2 * torch.pi) - torch.pi
        ccur_angles = (rd_feats[:, -1].reshape(-1)  + torch.pi) % (2 * torch.pi) - torch.pi
        self.true_angle_deltas =  ctrue_angles - ccur_angles
        print("current angles", ccur_angles[:20])
        print("generated_angle_deltas", self.generated_angle_deltas[:20])
        print("true_angle_deltas", self.true_angle_deltas[:20])
        self.true_angle_deltas = (self.true_angle_deltas + torch.pi) % (2 * torch.pi) - torch.pi
        return dgl.batch(final_molecules)

    def forward(self, frag_ids, A_graph, B_graph, geometry_graph_A, geometry_graph_B, A_pool, B_pool, A_cg, B_cg, geometry_graph_A_cg, geometry_graph_B_cg, epoch, validation = False):
        enc_out = self.forward_vae(A_graph, B_graph, geometry_graph_A, geometry_graph_B, A_pool, B_pool, A_cg, B_cg, geometry_graph_A_cg, geometry_graph_B_cg, epoch, validation)
        results = enc_out
        natoms = A_graph.batch_num_nodes()
        nbeads = A_cg.batch_num_nodes()
        kl_v, kl_v_un_clamped = torch.tensor([0]).to(results["prior_mean_V"].device), torch.tensor([0]).to(results["prior_mean_V"].device)
        if not validation:
            kl_v, kl_v_un_clamped = self.kl(results["posterior_mean_V"], results["posterior_logvar_V"], results["prior_mean_V"], results["prior_logvar_V"], natoms, nbeads, coordinates = True)
            dec_out = self.decoder(A_cg, B_graph, frag_ids, geometry_graph_A, geometry_graph_B)
        else:
            dec_out = self.decoder(B_cg, B_graph, frag_ids, geometry_graph_B, geometry_graph_B)
        kl_h = 0
        generated_molecule, rdkit_reference, (coords_A, h_feats_A, coords_B, h_feats_B, ref_coords_A, X_cc, H_cc) = dec_out
        return generated_molecule, rdkit_reference, (coords_A, h_feats_A, coords_B, h_feats_B, ref_coords_A), (X_cc, H_cc), (kl_v, kl_h, kl_v_un_clamped), enc_out
    
    def distance_loss(self, generated_molecule, geometry_graphs, key = "x_cc", maskHs= True):
        geom_loss = []
        for geometry_graph, generated_mol in zip(dgl.unbatch(geometry_graphs), dgl.unbatch(generated_molecule)):
            src, dst = geometry_graph.edges()
            src = src.long()
            dst = dst.long()
            true_comp = geometry_graph.edata['feat'] ** 2
            # import ipdb; ipdb.set_trace()
            if maskHs:
                mask = generated_mol.ndata['ref_feat'][:, :self.mask_cut]
                filter_rows = torch.where(mask[:, 0] == 1)[0]
                cut_mask = ~((src.unsqueeze(1) == filter_rows) | (dst.unsqueeze(1) == filter_rows)).any(dim=1)
                src = src[cut_mask]
                dst = dst[cut_mask]
                true_comp = true_comp[cut_mask] #! TODO check if this works
            generated_coords = generated_mol.ndata[key]
            d_squared = torch.sum((generated_coords[src] - generated_coords[dst]) ** 2, dim=1)
            geom_loss.append(1/len(src) * torch.sum((d_squared - true_comp) ** 2).unsqueeze(0))
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
        # if torch.any(torch.isnan(cos)) or torch.any(torch.isnan(torch.acos(cos))):
        #     import ipdb; ipdb.set_trace()
        #     test= 1
        return torch.acos(cos)
    
    def angle_loss(self, generated_molecule, angles, key = 'x_cc'):
        predicted_torsion_angles, true_torsion_angles = [], []
        predicted_angles, true_angles = [], []
        for idx, gmol in enumerate(dgl.unbatch(generated_molecule)):
            torsion_angles, dihedral_angles = angles[idx]
            pos = gmol.ndata[key] #TODO CGECK IF TRUE ANGLES LINES UP IN CALCULATION
            W,X,Y,Z = [], [], [], []
            for k, v in torsion_angles.items():
                # predicted_torsion_angles.append(self.calc_torsion(pos, k))
                w,x,y,z = k
                W.append(w)
                X.append(x)
                Y.append(y)
                Z.append(z)
                true_torsion_angles.append(v)
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
        cosine_torsion_difference = torch.cos(true_torsion_angles - predicted_torsion_angles) #! trying order flip if that does anything
        # cosine_difference = torch.cos(predicted_angles - true_angles)
        return 1-torch.mean(cosine_torsion_difference), None #1-torch.mean(cosine_difference)
            
    def rmsd_ot_step(self, molecule_batch, key = 'x_cc'):
        costs = []
        for molecules in molecule_batch:
            mcost = []
            gens = [x.ndata[key] for x in molecules]
            trues = [x.ndata['x_true'] for x in molecules]
            masks = [m.ndata['ref_feat'][:, :self.mask_cut] for m in molecules]
            for idx, gen in enumerate(gens):
                cost = []
                for true in trues:
                    mask = masks[idx]
                    cost.append(self.rmsd(gen, true, mask, align = True).unsqueeze(0))
                mcost.append(cost)
            mcost = torch.stack([torch.cat(inner_list) for inner_list in mcost])
            costs.append(mcost)
        return costs
    
    def angle_ot_step(self, molecule_batch, angle_batch, key = 'x_cc'):
        costs = []
        for bmolecules, bangles in zip(molecule_batch, angle_batch):
            mcost = []
            gens = [x.ndata[key] for x in bmolecules]
            # trues = [x.ndata['x_true'] for x in bmolecules]
            tas = [t[0] for t in bangles]
            for idx, gen in enumerate(gens):
                cost = []
                pos = gen
                W,X,Y,Z = [], [], [], []
                for k, v in tas[0].items():
                    w,x,y,z = k
                    W.append(w)
                    X.append(x)
                    Y.append(y)
                    Z.append(z)
                    # true_torsion_angles.append(v)
                predicted_torsion_angles = self.calc_torsion(pos, (W,X,Y,Z))
                for ta in tas:
                    true_torsion_angles = []
                    for k, v in ta.items():
                        w,x,y,z = k
                        # W.append(w)
                        # X.append(x)
                        # Y.append(y)
                        # Z.append(z)
                        true_torsion_angles.append(v)
                    # predicted_torsion_angles = self.calc_torsion(pos, (W,X,Y,Z))
                    if len(true_torsion_angles) != predicted_torsion_angles.shape[0]:
                        print(len(true_torsion_angles), predicted_torsion_angles.shape)
                        import ipdb; ipdb.set_trace()
                    cost.append((1 - torch.cos(torch.tensor(true_torsion_angles).cuda()-predicted_torsion_angles)).mean().unsqueeze(0))
                mcost.append(cost)
            # import ipdb; ipdb.set_trace()
            try:
                mcost = torch.stack([torch.cat(inner_list) for inner_list in mcost])
            except Exception as e:
                print(e)
                import ipdb; ipdb.set_trace()
                test = 1
            costs.append(mcost)
        return costs     
    
    def dist_ot_step(self, molecule_batch, geo_batch, key = "x_cc", maskHs= True):
        costs = []
        for molecules, geos in zip(molecule_batch, geo_batch):
            cost = []
            for mole, geo in zip(molecules, geos):
                # for geo in geos:
                mask = mole.ndata['ref_feat'][:, :self.mask_cut]
                cost.append(self.geo_loss(mole, geo, mask, maskHs))
            costs.append(torch.stack(cost))
        return costs
    
    def dist_ot_step2(self, molecule_batch, geo_batch, key = "x_cc", maskHs= True):
        costs = []
        for molecules, geos in zip(molecule_batch, geo_batch):
            mcost = []
            for mole in molecules:
                cost = []
                for geo in geos:
                    # for geo in geos:
                    mask = mole.ndata['ref_feat'][:, :self.mask_cut]
                    cost.append(self.geo_loss(mole, geo, mask, maskHs))
                mcost.append(cost)
            # costs.append(torch.stack(cost))
            mcost = torch.stack([torch.cat(inner_list) for inner_list in mcost])
            costs.append(mcost)
        return costs
    
    def geo_loss(self, generated_mol, geometry_graph, mask, maskHs = True, key='x_cc'):
        src, dst = geometry_graph.edges()
        src = src.long()
        dst = dst.long()
        true_comp = geometry_graph.edata['feat'] ** 2
        # import ipdb; ipdb.set_trace()
        if maskHs:
            # mask = generated_mol.ndata['ref_feat'][:, :35]
            filter_rows = torch.where(mask[:, 0] == 1)[0]
            cut_mask = ~((src.unsqueeze(1) == filter_rows) | (dst.unsqueeze(1) == filter_rows)).any(dim=1)
            src = src[cut_mask]
            dst = dst[cut_mask]
            true_comp = true_comp[cut_mask] #! TODO check if this works
        generated_coords = generated_mol.ndata[key]
        d_squared = torch.sum((generated_coords[src] - generated_coords[dst]) ** 2, dim=1)
        return 1/len(src) * torch.sum((d_squared - true_comp) ** 2).unsqueeze(0)
                    
                    
        
    def loss_function(self, generated_molecule, rdkit_reference, channel_selection_info, KL_terms, enc_out, geometry_graph, angles, log_latent_stats = True, nmols = None):
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
            self.update_adaptive_lambda(kl_v.item())
        
        # import ipdb; ipdb.set_trace()
        gmols = dgl.unbatch(generated_molecule)
        start = 0
        bgmols = []
        for nmol in nmols:
            bgmols.append(gmols[start: start + nmol])
            start += nmol
        # import ipdb; ipdb.set_trace()
        rdmols = dgl.unbatch(rdkit_reference)
        start = 0
        brdmols = []
        for nmol in nmols:
            brdmols.append(rdmols[start: start + nmol])
            start += nmol
        # import ipdb; ipdb.set_trace()
        distgs = dgl.unbatch(geometry_graph)
        start = 0
        bdists = []
        for nmol in nmols:
            bdists.append(distgs[start: start + nmol])
            start += nmol
        # import ipdb; ipdb.set_trace()
        start = 0
        bangles = []
        for nmol in nmols:
            bangles.append(angles[start: start + nmol])
            start += nmol
        # import ipdb; ipdb.set_trace()
        rmsd_costs = self.rmsd_ot_step(bgmols)
        rd_rmsd_costs = self.rmsd_ot_step(bgmols, key = 'x_ref')
        # import ipdb; ipdb.set_trace()
        # distance_costs = self.dist_ot_step(bgmols, bdists)
        # import ipdb; ipdb.set_trace()
        distance_costs = self.dist_ot_step2(bgmols, bdists)
        angle_costs = self.angle_ot_step(bgmols, bangles)
        # angle_costs = self.angle_ot_step(bgmols, bangles)
        # import ipdb; ipdb.set_trace()
        loss = 0
        closs = 0
        rdcloss = 0
        angloss = 0
        dloss = 0
        # rd_closs = 0
        # H_2 = np.ones(self.n_model_confs) / self.n_model_confs
        H_1 = [np.ones(nmol) / nmol for nmol in nmols]
        ot_mat_list = []
        for idx in range(len(nmols)):
            # min_rmsd = torch.min(rmsd_costs[idx], dim=1)[0] # for Hausdorff
            cost_mat_i = self.lambda_global_mse*rmsd_costs[idx].detach().cpu().numpy() + self.lambda_distance*distance_costs[idx].detach().cpu().numpy() + self.lambda_angle*angle_costs[idx].detach().cpu().numpy()
            ot_mat = ot.emd(a=H_1[idx], b=H_1[idx], M=np.max(np.abs(cost_mat_i)) + cost_mat_i, numItermax=10000)
            ot_mat_attached = torch.tensor(ot_mat, device="cuda", requires_grad=False).float()
            # ot_mat_list.append(ot_mat_attached)
            live_cost = self.lambda_global_mse*rmsd_costs[idx]+ self.lambda_distance*distance_costs[idx] + self.lambda_angle*angle_costs[idx]
            loss += torch.sum(ot_mat_attached * live_cost)
            closs += torch.sum(ot_mat_attached * rmsd_costs[idx])
            rdcloss += torch.sum(ot_mat_attached * rd_rmsd_costs[idx])
            angloss += torch.sum(ot_mat_attached * angle_costs[idx])
            dloss += torch.sum(ot_mat_attached * distance_costs[idx])
            # rd_closs += torch.sum(ot_mat_attached * self.lambda_global_mse*rmsd_costs[idx])
        mloss = loss / len(nmols) 
        closs = closs / len(nmols)
        rdcloss = rdcloss / len(nmols)
        angloss = angloss / len(nmols)
        dloss = dloss / len(nmols)
        loss = mloss + kl_loss
        # global_mse_loss = self.lambda_global_mse*self.coordinate_loss(generated_molecule)
        # global_mse_loss_unmasked = self.lambda_global_mse*self.coordinate_loss(generated_molecule, mask_Hs= False)
        # # global_mse_rdkit_loss = torch.cat([self.rmsd(m.ndata['x_cc'], rd.ndata['x_ref'], align = True).unsqueeze(0) for (m, rd) in zip(dgl.unbatch(generated_molecule), dgl.unbatch(rdkit_reference))]).mean()
        # rdkit_loss = [self.rmsd(m.ndata['x_ref'], m.ndata['x_true'], align = True).unsqueeze(0) for m in dgl.unbatch(rdkit_reference)]
        # rdkit_loss = self.lambda_global_mse*torch.cat(rdkit_loss).mean()
        
        # distance_loss = self.distance_loss(generated_molecule, geometry_graph)
        # # distance_loss_rd = self.distance_loss(generated_molecule, geometry_graph, key='x_ref')
        # # distance_loss_true = self.distance_loss(generated_molecule, geometry_graph, key='x_true')
        # scaled_distance_loss = self.lambda_distance*distance_loss
        
        # ta_loss, _ = self.angle_loss(generated_molecule, angles)
        # # ta_loss_rd, _ = self.angle_loss(generated_molecule, angles, key='x_ref')
        # # ta_loss_true, _ = self.angle_loss(generated_molecule, angles, key='x_true')
        # angle_loss = self.lambda_angle*ta_loss #+ self.lambda_dihedral*di_loss
        # angle_regression_loss = self.new_angle_loss(self.generated_angle_deltas, self.true_angle_deltas)
        # # loss =  global_mse_loss + scaled_distance_loss + angle_loss + kl_loss + angle_regression_loss
        # loss = angle_regression_loss
        
        if log_latent_stats:
            loss_results = {
                'latent reg loss': kl_loss.item(),
                'global ot mse': closs.item(),
                'rdkit ot mse': rdcloss.item(),
                "angle loss": angloss.item(),
                'train ot loss': loss.item(),
                'distance loss': dloss.item(),
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
                'global ot mse': closs.item(),
                'rdkit ot mse': rdcloss.item(),
                "angle loss": angloss.item(),
                'train ot loss': loss.item(),
                'distance loss': dloss.item(),
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
            cutoff = 35 # For drugs and 5 for qm9 m.ndata['ref_feat'][:, :cutoff]
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

    def update_adaptive_lambda(
        self,
        # current_lambda,
        current_rate,
        target_rate=5,#5.0 (DRUGS),
        delta=5.0e-3,# 1
        epsilon=0.5, #1.0e-2, #0.5,
        lower_limit=1e-10,
        upper_limit=10,#1e10,
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
