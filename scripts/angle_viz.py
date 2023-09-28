import sys
# Print the current Python path
# print(sys.path)
import os
# print(os.environ['PYTHONPATH']) #sounce ./set_path.sh
# os.environ["OMP_NUM_THREADS"] = "64"  # Replace "4" with the desired number of threads
# Set the number of MKL threads (MKL_NUM_THREADS)
# os.environ["MKL_NUM_THREADS"] = "32" 
import hydra
from omegaconf import DictConfig, OmegaConf
import ipdb
import torch
import wandb
import random
import logging
from utils.data_utils_noar import cook_drugs_angles
from model.vae_noar_angle import VAENoArAngle
import datetime
from model.benchmarker import *
import glob
import time


def load_data(cfg):
    print("Loading DRUGS...")
    train_loader, train_data = cook_drugs_angles(batch_size=cfg['train_batch_size'], mode='train', limit_mols=cfg['train_data_limit'])
    return train_loader, train_data

def save_code(wandb_run): 
    code_dir = "/home/dannyreidenbach/mcg/coagulation/model" 
    # Create an artifact from the code directory
    code_artifact = wandb.Artifact("model", type="code") 
    # Add all the .py files in the code directory to the artifact using glob
    for file_path in glob.glob(code_dir + '/*.py'):
        code_artifact.add_file(file_path)
    wandb_run.log_artifact(code_artifact)
    
    code_artifact = wandb.Artifact("utils", type="code") 
    code_dir = "/home/dannyreidenbach/mcg/coagulation/utils" 
    # Add all the .py files in the utils directory to the artifact using glob
    for file_path in glob.glob(code_dir + '/*.py'):
        code_artifact.add_file(file_path) 
    wandb_run.log_artifact(code_artifact)

# import subprocess

# def print_gpu_usage():
#     command = "nvidia-smi"
#     result = subprocess.run(command, shell=True, stdout=subprocess.PIPE)
#     print(result.stdout.decode('utf-8'))

# def get_memory_usage():
#     # cmd = "nvidia-smi | grep python | awk '{ print $6 }'"
#     cmd = "nvidia-smi | grep python | awk '{ print $8 }' | sed 's/MiB//'"
#     output = subprocess.check_output(cmd, shell=True)
#     return int(output.strip().decode()) #int(output.strip())
    
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
    
    train_loader, train_data = load_data(cfg.data)
    F = cfg.encoder["coord_F_dim"]
    D = cfg.encoder["latent_dim"]
    # import ipdb; ipdb.set_trace()
    # model = VAENoArAngle(cfg.vae, cfg.encoder, cfg.decoder, cfg.losses, coordinate_type, device = "cuda").cuda()
    
    # print("CUDA CHECK", next(model.parameters()).is_cuda)
    # print("# of Encoder Params = ", sum(p.numel()
    #       for p in model.encoder.parameters() if p.requires_grad))
    # print("# of Decoder Params = ", sum(p.numel()
    #       for p in model.decoder.parameters() if p.requires_grad))
    # print("# of VAE Params = ", sum(p.numel()
    #       for p in model.parameters() if p.requires_grad))
    # if cfg.optimizer.optim == 'adamw':
    #     optim = torch.optim.AdamW(model.parameters(), lr= cfg.optimizer.lr)
    # elif cfg.optimizer.optim == 'adam':
    #     optim = torch.optim.Adam(model.parameters(), lr= cfg.optimizer.lr)
    # else:
    #     assert(1 == 0)
    # self.optim.step()
    # self.after_optim_step()  # overwrite this function to do stuff before zeroing out grads
    # self.lr_scheduler.total_warmup_steps > self.lr_scheduler._step)):  # step per batch if that is what we want to do or if we are using a warmup schedule and are still in the warmup period
    #         self.step_schedulers() --> self.lr_scheduler.step()
    # self.optim.zero_grad()
    # self.optim_steps += 1
    # torch.autograd.set_detect_anomaly(True)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min', factor=0.1, patience=1, verbose=True)
    # scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=1, gamma=0.9) #0.8

    # import ipdb; ipdb.set_trace()
    results = []
    for data in train_data:
        A_batch, B_batch = data
        A_graph, geo_A, angle_A, angle_mask_A, Ap, A_cg, geo_A_cg, frag_ids = A_batch
        B_graph, geo_B, angle_B, angle_mask_B, Bp, B_cg, geo_B_cg, B_frag_ids = B_batch
        torsions = angle_A[0]
        result = []
        for k, v in torsions.items():
            a,b,c,d = k
            angle = calc_torsion(B_graph.ndata['x'], k).item()
            result.append((v,angle))
            # tangle = calc_torsion(A_graph.ndata['x'], k)
            # tangle = calc_torsion(B_graph.ndata['x'], (1, 0, 2, 3))
        results.append((result, A_graph.ndata['x'].shape, torsions))
    import ipdb; ipdb.set_trace()
    save_dir = '/data/dreidenbach/data/torsional_diffusion/DRUGS/test_set'
    with open(os.path.join(save_dir, f'one_fifth_angles.pkl'), 'wb') as handle:
            pickle.dump(results, handle)
            
        
        
        
    # for epoch in range(100*cfg.data['epochs']):
    #     train_loss_log, val_loss_log = [], []
    #     count = 0
    #     print(f"{len(train_loader)} batches")
    #     for A_batch, B_batch in train_loader:
    #         A_graph, geo_A, angle_A, angle_mask_A, Ap, A_cg, geo_A_cg, frag_ids = A_batch
    #         B_graph, geo_B, angle_B, angle_mask_B, Bp, B_cg, geo_B_cg, B_frag_ids = B_batch
    #         # A_graph, geo_A, Ap, A_cg, geo_A_cg = A_graph.to('cuda:0'), geo_A.to('cuda:0'), Ap.to('cuda:0'), A_cg.to('cuda:0'), geo_A_cg.to('cuda:0')
            # B_graph, geo_B, Bp, B_cg, geo_B_cg = B_graph.to('cuda:0'), geo_B.to('cuda:0'), Bp.to('cuda:0'), B_cg.to('cuda:0'), geo_B_cg.to('cuda:0')
            # start_time = time.time()
            # # import ipdb; ipdb.set_trace()
            # generated_molecule, rdkit_reference, _, channel_selection_info, KL_terms, enc_out = model(frag_ids, A_graph, B_graph, geo_A, geo_B, Ap, Bp, A_cg, B_cg, geo_A_cg, geo_B_cg, epoch=epoch)
            # print("Forward Pass", time.time() - start_time)
            # start_time = time.time()
            # generated_molecule = model.angle_forward(generated_molecule, angle_A, angle_mask_A, frag_ids)
            # print("Angle Pass", time.time() - start_time)
            # start_time = time.time()
            # if epoch > 1000:# or count >= 100:
            #     import ipdb; ipdb.set_trace()
            # loss, losses = model.loss_function(generated_molecule, rdkit_reference, channel_selection_info, KL_terms, enc_out, geo_A, angle_A)
            # print("Loss Calc", time.time() - start_time)
            # print(f"Train LOSS = {loss}, {loss.device}, {loss.shape}")


if __name__ == "__main__":
    main()
