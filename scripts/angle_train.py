import sys
# Print the current Python path
# print(sys.path)
import os
# print(os.environ['PYTHONPATH']) #sounce ./set_path.sh
import hydra
from omegaconf import DictConfig, OmegaConf
import ipdb
import torch
import wandb
import random
import logging
from utils.data_utils_noar import cook_drugs_angles
from utils.equivariant_model_utils import *
from model.vae_noar import VAENoAr
import datetime
from model.benchmarker import *
import glob
import time
from utils.embedding import AtomEncoder, A_feature_dims, AtomEncoderTorsionalDiffusion


def load_data(cfg):
    print("Loading DRUGS...")
    train_loader, train_data = cook_drugs_angles(batch_size=cfg['train_batch_size'], mode='train', limit_mols=cfg['train_data_limit'])
    # print("Loading Val DRUGS...")
    print("Skipping Validation")
    # val_loader, val_data = cook_drugs_angles(batch_size=cfg['val_batch_size'], mode='val', limit_mols=cfg['val_data_limit'])
    print("Loading All DRUGS --> Done")
    return train_loader, train_data, [], []
    # return train_loader, train_data, val_loader, val_data

def calc_torsion(pos, k):
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

def build_batch(A_graph, angle_A, frag_ids):
    predicted_torsion_angles, true_torsion_angles = [], []
    predicted_angles, true_angles = [], []
    rd_feats = []
    count = 0
    # key = 'x_ref'
    for idx, gmol in enumerate(dgl.unbatch(A_graph)):
        # rd_mol = []
        torsion_angles, dihedral_angles = angle_A[idx]
        chunking = frag_ids[idx]
        # if idx >= 2:
        #     break
        
        pos = gmol.ndata['x_ref']
        W,X,Y,Z = [], [], [], []
        molecule_feat = gmol.ndata['ref_feat'].mean(0).reshape(-1) # (D, )
        # import ipdb; ipdb.set_trace()
        for k, v in torsion_angles.items():
            # predicted_torsion_angles.append(self.calc_torsion(pos, k))
            feats = [molecule_feat]# Starting with the entire molecules
            # import ipdb; ipdb.set_trace()
            w,x,y,z = k
            W.append(w)
            X.append(x)
            Y.append(y)
            Z.append(z)
            true_torsion_angles.append(v)
            chunk_feats = []
            for chunk in chunking:
                if x in chunk:
                    chunk_feats.append(gmol.ndata['ref_feat'][list(chunk), :].mean(0).reshape(-1))
                elif y in chunk:
                    chunk_feats.append(gmol.ndata['ref_feat'][list(chunk), :].mean(0).reshape(-1))
            assert(len(chunk_feats) == 2)
            feats.extend(chunk_feats) # Adding features of TA substructure spliots
            # import ipdb; ipdb.set_trace()
            locs = [gmol.ndata['x_ref'][loc, :].reshape(-1) for loc in [w,x,y,z]] # 3D points for TA
            pfeats = [gmol.ndata['ref_feat'][loc, :].reshape(-1) for loc in [w,x,y,z]] # features for TA atoms
            feats.extend(locs)
            feats.extend(pfeats)
            
            rd_feats.append(torch.cat(feats)) # (460,) tensor
            # import ipdb; ipdb.set_trace()
        
        predicted_torsion_angles.append(calc_torsion(pos, (W,X,Y,Z)))
        # import ipdb; ipdb.set_trace()
        for i in range(len(W)):
            index = i + count
            rd_feats[index] = torch.cat((rd_feats[index], predicted_torsion_angles[-1][i].view(1)))
            # rd_feats[index].append(predicted_torsion_angles[i])
            
        count += len(W)
        # import ipdb; ipdb.set_trace()
        # rd_feats.append(rd_mol)
        assert(predicted_torsion_angles[-1].shape[0] == len(torsion_angles))
    # import ipdb; ipdb.set_trace()
    predicted_torsion_angles = torch.cat(predicted_torsion_angles, dim = 0)
    true_torsion_angles = torch.tensor(true_torsion_angles).cuda()
    # predicted_angles = torch.cat(predicted_angles, dim = 0)
    # true_angles = torch.tensor(true_angles).cuda()
    regression_targets = predicted_torsion_angles - true_torsion_angles
    cosine_torsion_difference = torch.cos(regression_targets)
    classification_targets = 1-cosine_torsion_difference
    classification_targets = (classification_targets >= 1).float()
    rd_feats = torch.stack(rd_feats)
    # import ipdb; ipdb.set_trace()
    print()
    print(1-cosine_torsion_difference)
    print(1-torch.mean(cosine_torsion_difference), "Torsional Angle Loss according to VAE")
    pi_tensor = torch.full_like(classification_targets, 3.141592653589793).to(cosine_torsion_difference.device)
    mean_expression = torch.mean(torch.cos(regression_targets - (pi_tensor*classification_targets)))
    print(1- mean_expression, "Torsional Angle Loss according to VAE if classification was perfect")
    return rd_feats, classification_targets, regression_targets
    
    
@hydra.main(config_path="../configs", config_name="config_drugs.yaml")
def main(cfg: DictConfig): #['encoder', 'decoder', 'vae', 'optimizer', 'losses', 'data', 'coordinates', 'wandb']
    import datetime
    now = datetime.datetime.now()
    suffix = f"_{now.strftime('%m-%d_%H-%M-%S')}"
    coordinate_type = cfg.coordinates
    NAME = cfg.wandb['name'] + suffix
    # wandb_run = wandb.init(
    #     project=cfg.wandb.project,
    #     name=NAME,
    #     notes=cfg.wandb.notes,
    #     config = cfg,
    #     save_code = True
    # )
    # save_code(wandb_run)
    
    train_loader, train_data, val_loader, val_data = load_data(cfg.data)
    del train_data
    F = cfg.encoder["coord_F_dim"]
    D = cfg.encoder["latent_dim"]
    atom_embedder = AtomEncoderTorsionalDiffusion(emb_dim=64, feature_dim =75).cuda()
    # import ipdb; ipdb.set_trace()
    # model = VAENoAr(cfg.vae, cfg.encoder, cfg.decoder, cfg.losses, coordinate_type, device = "cuda").cuda()
    feat_dim = 75 #64
    angle_feat_dim = 7*feat_dim+13
    # regressor = Scalar_MLP(angle_feat_dim, angle_feat_dim, 1, use_batchnorm = False).cuda()
    # optim = torch.optim.AdamW(list(regressor.parameters()) + list(atom_embedder.parameters()), lr= 1e-3)
    # loss_fn = nn.MSELoss()
    use_classifier = True
    classifier = Scalar_MLP(angle_feat_dim, angle_feat_dim, 1, use_batchnorm = False).cuda()
    # optim = torch.optim.AdamW(list(classifier.parameters()) + list(atom_embedder.parameters()), lr= 1e-3)
    optim = torch.optim.AdamW(classifier.parameters(), lr= 1e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    
    model = classifier
    for epoch in range(100*cfg.data['epochs']):
        print("Epoch", epoch)
        count = 0
        print(f"{len(train_loader)} batches")
        for A_batch, B_batch in train_loader:
            # A_graph, geo_A, angle_A, angle_mask_A, Ap, A_cg, geo_A_cg, frag_ids = A_batch
            B_graph, _, angle_B, angle_mask_B, _, _, _, B_frag_ids = B_batch
            # A_graph, geo_A, Ap, A_cg, geo_A_cg = A_graph.to('cuda:0'), geo_A.to('cuda:0'), Ap.to('cuda:0'), A_cg.to('cuda:0'), geo_A_cg.to('cuda:0')
            B_graph = B_graph.to('cuda:0')
            # start_time = time.time()
            assert(B_graph.ndata['ref_feat'].shape[1] == 75)
            # import ipdb; ipdb.set_trace()
            # B_graph.ndata['ref_feat'] = atom_embedder(B_graph.ndata['ref_feat'])
            batch = build_batch(B_graph, angle_B, B_frag_ids)
            rd_feats, classification_targets, regression_targets = batch
            # import ipdb; ipdb.set_trace()
            outs = model(rd_feats).squeeze(1) #! Do we want to use GEOMOL formulation
            if use_classifier:
                loss = loss_fn(outs, classification_targets)
            else:
                loss = loss_fn(outs, regression_targets)
            print(outs)
            print(classification_targets)
            print(f"Epoch {epoch} | Train LOSS = {loss}, {loss.device}, {loss.shape}")
            for ratio in [0.1, 0.2, 0.3, 0.5, 0.6, 0.7]:
                print(f"Ratio {ratio}")
                predicted = (torch.sigmoid(outs) >= ratio).float()
                pi_tensor = torch.full_like(predicted, 3.141592653589793).to(predicted.device)
                print(1- torch.mean(torch.cos(regression_targets - (pi_tensor*predicted))), "Torsional Angle Loss if using this classifiers outputs")
                
                total = classification_targets.shape[0]
                correct = (predicted == classification_targets).byte().sum().item()
                print(f"{correct} correct out of {total} total")
                incorrect_rotations = ((predicted == 1) & (classification_targets == 0)).sum().item()
                missed_rotations = ((predicted == 0) & (classification_targets == 1)).sum().item()
                hits = ((predicted == 1) & (classification_targets == 1)).sum().item()
                stays = ((predicted == 0) & (classification_targets == 0)).sum().item()
                print(f"{correct} correct out of {total} total: {incorrect_rotations} rotated when not supposed to and {missed_rotations} missed rotations")
                print(f"{hits} hits (1==1) and {stays} stays (0==0)")
            print()
            if count == 100:
                import ipdb; ipdb.set_trace()
                stop = 1
            loss.backward()
            # wandb.log(losses)
            optim.step()
            optim.zero_grad()
            
            # del A_graph, geo_A, angle_A, angle_mask_A, Ap, A_cg, geo_A_cg, frag_ids
            del B_graph, angle_B, angle_mask_B, B_frag_ids
            # del generated_molecule, rdkit_reference, channel_selection_info, KL_terms, enc_out, losses
            # if count > 0 and count %10 == 0:
            #     torch.cuda.empty_cache()
            #     model_path = f'/home/dannyreidenbach/mcg/coagulation/scripts/model_ckpt/{NAME}_{epoch}_temp.pt'
                # torch.save(model.state_dict(), model_path)
            count+=1

        # print("Validation")
        # val_loss = 0
        # with torch.no_grad():
        #     for A_batch, B_batch in val_loader:
        #         A_graph, geo_A, angle_A, angle_mask_A, Ap, A_cg, geo_A_cg, frag_ids = A_batch 
        #         B_graph, geo_B, angle_B, angle_mask_B, Bp, B_cg, geo_B_cg, B_frag_ids = B_batch

        #         A_graph, geo_A, Ap, A_cg, geo_A_cg = A_graph.to('cuda:0'), geo_A.to('cuda:0'), Ap.to('cuda:0'), A_cg.to('cuda:0'), geo_A_cg.to('cuda:0')
        #         B_graph, geo_B, Bp, B_cg, geo_B_cg = B_graph.to('cuda:0'), geo_B.to('cuda:0'), Bp.to('cuda:0'), B_cg.to('cuda:0'), geo_B_cg.to('cuda:0')

        #         generated_molecule, rdkit_reference, _, channel_selection_info, KL_terms, enc_out = model(B_frag_ids, A_graph, B_graph, geo_A, geo_B, Ap, Bp, A_cg, B_cg, geo_A_cg, geo_B_cg, epoch=epoch, validation = True)
        #         loss, losses = model.loss_function(generated_molecule, rdkit_reference, channel_selection_info, KL_terms, enc_out, geo_A, angle_A, log_latent_stats = False)
        #         losses['Val Loss'] = loss.cpu()
        #         wandb.log({'val_' + key: value for key, value in losses.items()})
        #         print(f"Val LOSS = {loss}, {type(loss)}")
            
    #     # scheduler.step(val_loss)
    #     scheduler.step()
    #     model_path = f'/home/dreidenbach/code/mcg/coagulation/scripts/model_ckpt/{NAME}_{epoch}.pt'
    #     torch.save(model.state_dict(), model_path)
        
    # print("No Ar Training Complete")


if __name__ == "__main__":
    main()
