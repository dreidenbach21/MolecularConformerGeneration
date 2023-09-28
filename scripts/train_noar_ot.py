import sys
# Print the current Python path
# print(sys.path)
import os
# print(os.environ['PYTHONPATH']) #sounce ./set_path.sh
os.environ["OMP_NUM_THREADS"] = "12"  # Replace "4" with the desired number of threads
# Set the number of MKL threads (MKL_NUM_THREADS)
# os.environ["MKL_NUM_THREADS"] = "32" 
import hydra
from omegaconf import DictConfig, OmegaConf
import ipdb
import torch
import wandb
import random
import logging
from utils.data_utils_noar import cook_drugs_ot, cook_qm9_ot
from model.vae_noar_angle_ot import VAENoArAngleOt
import datetime
from model.benchmarker import *
import glob
import time


def load_data(cfg):
    print("Loading DRUGS...") # cook_qm9_ot cook_drugs_ot
    train_loader, train_data = cook_qm9_ot(batch_size=cfg['train_batch_size'], mode='train', limit_mols=cfg['train_data_limit'])
    print("Loading Val DRUGS...")
    # print("Skipping Validation")
    val_loader, val_data = cook_qm9_ot(batch_size=cfg['val_batch_size'], mode='val', limit_mols=cfg['val_data_limit'])
    # print("Loading All DRUGS --> Done")
    # return train_loader, train_data
    return train_loader, train_data, val_loader, val_data

def load_drug_data(cfg):
    print("Loading DRUGS...") # cook_qm9_ot cook_drugs_ot
    train_loader, train_data = cook_drugs_ot(batch_size=cfg['train_batch_size'], mode='train', limit_mols=cfg['train_data_limit'])
    print("Loading Val DRUGS...")
    # print("Skipping Validation")
    val_loader, val_data = cook_drugs_ot(batch_size=cfg['val_batch_size'], mode='val', limit_mols=cfg['val_data_limit'])
    # print("Loading All DRUGS --> Done")
    # return train_loader, train_data
    return train_loader, train_data, val_loader, val_data

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
    

@hydra.main(config_path="../configs", config_name="config_drugs.yaml")
# @hydra.main(config_path="../configs", config_name="config_qm9.yaml")
def main(cfg: DictConfig): #['encoder', 'decoder', 'vae', 'optimizer', 'losses', 'data', 'coordinates', 'wandb']
    import datetime
    now = datetime.datetime.now()
    suffix = f"_{now.strftime('%m-%d_%H-%M-%S')}"
    coordinate_type = cfg.coordinates
    NAME = cfg.wandb['name'] + suffix
    wandb_run = wandb.init(
        project=cfg.wandb.project,
        name=NAME,
        notes=cfg.wandb.notes,
        config = cfg,
        save_code = True
    )
    # save_code(wandb_run)
    
    train_loader, train_data, val_loader, val_data = load_drug_data(cfg.data)#load_data(cfg.data) load_drug_data
    # del train_data
    # del val_data
    F = cfg.encoder["coord_F_dim"]
    D = cfg.encoder["latent_dim"]
    # import ipdb; ipdb.set_trace()
    model = VAENoArAngleOt(cfg.vae, cfg.encoder, cfg.decoder, cfg.losses, coordinate_type, device = "cuda").cuda()
    
    weights_path = "/home/dreidenbach/code/mcg/coagulation/scripts/model_ckpt/DRUGS_noar_ot_all_09-22_02-18-40_2.pt"
    chkpt = torch.load(weights_path)
    if "model" in chkpt:
        chkpt = chkpt['model']
    model.load_state_dict(chkpt, strict = False)
    
    print("CUDA CHECK", next(model.parameters()).is_cuda)
    print("# of Encoder Params = ", sum(p.numel()
          for p in model.encoder.parameters() if p.requires_grad))
    print("# of Decoder Params = ", sum(p.numel()
          for p in model.decoder.parameters() if p.requires_grad))
    print("# of VAE Params = ", sum(p.numel()
          for p in model.parameters() if p.requires_grad))
    if cfg.optimizer.optim == 'adamw':
        optim = torch.optim.AdamW(model.parameters(), lr= cfg.optimizer.lr)
    elif cfg.optimizer.optim == 'adam':
        optim = torch.optim.Adam(model.parameters(), lr= cfg.optimizer.lr)
    else:
        assert(1 == 0)
    # self.optim.step()
    # self.after_optim_step()  # overwrite this function to do stuff before zeroing out grads
    # self.lr_scheduler.total_warmup_steps > self.lr_scheduler._step)):  # step per batch if that is what we want to do or if we are using a warmup schedule and are still in the warmup period
    #         self.step_schedulers() --> self.lr_scheduler.step()
    # self.optim.zero_grad()
    # self.optim_steps += 1
    # torch.autograd.set_detect_anomaly(True)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min', factor=0.1, patience=1, verbose=True)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=1, gamma=0.9) #0.8


    
    kl_annealing = True
    kl_weight = 0#1e-8 #1e-5
    kl_annealing_rate = 1e-3#1e-3
    kl_annealing_interval = 1 # at 200 we saw PC #! at 200 we still see dying
    kl_cap = 5e-3 #1e-1
    
    dist_weight = 1e-6 #0 #1e-6
    dist_annealing_rate =1e-1#3 #0.2 #0.05
    dist_cap = 1 #0.5
    for epoch in range(100*cfg.data['epochs']):
        print("Epoch", epoch)
        if kl_annealing and epoch > 0 and epoch % kl_annealing_interval == 0:
            kl_weight += kl_annealing_rate
            kl_weight = min(kl_weight, kl_cap)
            
            dist_weight += dist_annealing_rate
            dist_weight = min(dist_weight, dist_cap)
        if kl_annealing:
            model.kl_v_beta = kl_weight
            model.lambda_distance = dist_weight
        train_loss_log, val_loss_log = [], []
        count = 0
        print(f"{len(train_loader)} batches")
        for A_batch, B_batch, nmols in train_loader:
            A_graph, geo_A, angle_A, angle_mask_A, Ap, A_cg, geo_A_cg, frag_ids = A_batch
            B_graph, geo_B, angle_B, angle_mask_B, Bp, B_cg, geo_B_cg, B_frag_ids = B_batch
            A_graph, geo_A, Ap, A_cg, geo_A_cg = A_graph.to('cuda:0'), geo_A.to('cuda:0'), Ap.to('cuda:0'), A_cg.to('cuda:0'), geo_A_cg.to('cuda:0')
            B_graph, geo_B, Bp, B_cg, geo_B_cg = B_graph.to('cuda:0'), geo_B.to('cuda:0'), Bp.to('cuda:0'), B_cg.to('cuda:0'), geo_B_cg.to('cuda:0')
            start_time = time.time()
            # import ipdb; ipdb.set_trace()
            generated_molecule, rdkit_reference, _, channel_selection_info, KL_terms, enc_out = model(frag_ids, A_graph, B_graph, geo_A, geo_B, Ap, Bp, A_cg, B_cg, geo_A_cg, geo_B_cg, epoch=epoch)
            print("Forward Pass", time.time() - start_time)
            start_time = time.time()
            # generated_molecule = model.angle_forward(generated_molecule, angle_A, angle_mask_A, frag_ids)
            # print("Angle Pass", time.time() - start_time)
            # start_time = time.time()
            if epoch > 1000:# or count >= 100:
                import ipdb; ipdb.set_trace()
            loss, losses = model.loss_function(generated_molecule, rdkit_reference, channel_selection_info, KL_terms, enc_out, geo_A, angle_A, nmols= nmols)
            print("Loss Calc", time.time() - start_time)
            print(f"Train LOSS = {loss}, {loss.device}, {loss.shape}")
            loss.backward()
            losses['Train Loss'] = loss.cpu()
            wandb.log(losses)
            parameters = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
            norm_type = 2
            total_norm = 0.0
            for p in parameters:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            print(f"Train LOSS = {loss}, {type(loss)}")
            print("TOTAL GRADIENT NORM", total_norm, "clip mag:",cfg.optimizer.clip_grad_norm)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.optimizer.clip_grad_norm, norm_type=2) #50
            optim.step()
            optim.zero_grad()
            
            del A_graph, geo_A, angle_A, angle_mask_A, Ap, A_cg, geo_A_cg, frag_ids
            del B_graph, geo_B, angle_B, angle_mask_B, Bp, B_cg, geo_B_cg, B_frag_ids
            del generated_molecule, rdkit_reference, channel_selection_info, KL_terms, enc_out, losses
            # if count > 0 and count %10 == 0:
            #     torch.cuda.empty_cache()
            #     model_path = f'/home/dannyreidenbach/mcg/coagulation/scripts/model_ckpt/{NAME}_{epoch}_temp.pt'
                # torch.save(model.state_dict(), model_path)
            count+=1

        print("Validation")
        val_loss = 0
        with torch.no_grad():
            for A_batch, B_batch, nmols in val_loader:
                A_graph, geo_A, angle_A, angle_mask_A, Ap, A_cg, geo_A_cg, frag_ids = A_batch 
                B_graph, geo_B, angle_B, angle_mask_B, Bp, B_cg, geo_B_cg, B_frag_ids = B_batch

                A_graph, geo_A, Ap, A_cg, geo_A_cg = A_graph.to('cuda:0'), geo_A.to('cuda:0'), Ap.to('cuda:0'), A_cg.to('cuda:0'), geo_A_cg.to('cuda:0')
                B_graph, geo_B, Bp, B_cg, geo_B_cg = B_graph.to('cuda:0'), geo_B.to('cuda:0'), Bp.to('cuda:0'), B_cg.to('cuda:0'), geo_B_cg.to('cuda:0')

                generated_molecule, rdkit_reference, _, channel_selection_info, KL_terms, enc_out = model(B_frag_ids, A_graph, B_graph, geo_A, geo_B, Ap, Bp, A_cg, B_cg, geo_A_cg, geo_B_cg, epoch=epoch, validation = True)
                # generated_molecule = model.angle_forward(generated_molecule, angle_A, angle_mask_A, frag_ids)
                loss, losses = model.loss_function(generated_molecule, rdkit_reference, channel_selection_info, KL_terms, enc_out, geo_A, angle_A, nmols= nmols, log_latent_stats = False)
            
                losses['Val Loss'] = loss.cpu()
                wandb.log({'val_' + key: value for key, value in losses.items()})
                print(f"Val LOSS = {loss}, {type(loss)}")
            
        # scheduler.step(val_loss)
        scheduler.step()
        model_path = f'/home/dreidenbach/code/mcg/coagulation/scripts/model_ckpt/{NAME}_{epoch}.pt'
        torch.save(model.state_dict(), model_path)
        
    print("No Ar Training Complete")


if __name__ == "__main__":
    main()
