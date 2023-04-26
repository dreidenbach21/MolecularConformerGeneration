import sys
# Print the current Python path
# print(sys.path)
import os
# print(os.environ['PYTHONPATH'])
import hydra
from omegaconf import DictConfig, OmegaConf
import ipdb
import torch
import wandb
import random
import logging
from utils.torsional_diffusion_data_all import load_torsional_data  # , QM9_DIMS, DRUG_DIMS
from model.vae import VAE
import datetime
from model.benchmarker import *


def load_data(cfg):
    print("Loading QM9...")
    train_loader, train_data = load_torsional_data(batch_size=cfg['train_batch_size'], mode='train', limit_mols=cfg['train_data_limit'])
    val_loader, val_data = load_torsional_data(batch_size=cfg['val_batch_size'], mode='val', limit_mols=cfg['val_data_limit'])
    print("Loading QM9 --> Done")
    return train_loader, train_data, val_loader, val_data

# def save_code():
#     code_dir = "/home/dannyreidenbach/mcg/coagulation/model"
#     # Create an artifact from the code directory
#     code_artifact = wandb.Artifact("code", type="code")
#     # Add all the files in the code directory to the artifact
#     for root, dirs, files in os.walk(code_dir):
#         for file in files:
#             file_path = os.path.join(root, file)
#             code_artifact.add_file(file_path)
#     code_dir = "/home/dannyreidenbach/mcg/coagulation/utils"
#     for root, dirs, files in os.walk(code_dir):
#         for file in files:
#             file_path = os.path.join(root, file)
#             code_artifact.add_file(file_path)
#     wandb_run.log_artifact(code_artifact)
    
@hydra.main(config_path="../configs", config_name="config_qm9.yaml")
def main(cfg: DictConfig): #['encoder', 'decoder', 'vae', 'optimizer', 'losses', 'data', 'coordinates', 'wandb']
    import datetime
    now = datetime.datetime.now()
    suffix = f"_{now.strftime('%m-%d_%H-%M-%S')}"
    coordinate_type = cfg.coordinates
    NAME = cfg.wandb['name'] + suffix
    wandb.init(
        project=cfg.wandb.project,
        name=NAME,
        notes=cfg.wandb.notes,
        config = cfg,
        save_code = "all"
    )
    # save_code()
    train_loader, train_data, val_loader, val_data = load_data(cfg.data)
    BENCHMARK = BenchmarkRunner()
    F = cfg.encoder["coord_F_dim"]
    D = cfg.encoder["latent_dim"]
    model = VAE(cfg.vae, cfg.encoder, cfg.decoder, cfg.losses, coordinate_type, device = "cuda").cuda()
    
    print("CUDA CHECK", next(model.parameters()).is_cuda)
    print("# of Encoder Params = ", sum(p.numel()
          for p in model.encoder.parameters() if p.requires_grad))
    print("# of Decoder Params = ", sum(p.numel()
          for p in model.decoder.parameters() if p.requires_grad))
    print("# of VAE Params = ", sum(p.numel()
          for p in model.parameters() if p.requires_grad))
    if cfg.optimizer.optim == 'adamw':
        optim = torch.optim.AdamW(model.parameters(), lr= cfg.optimizer.lr)
    else:
        assert(1 == 0)
    # self.optim.step()
    # self.after_optim_step()  # overwrite this function to do stuff before zeroing out grads
    # self.lr_scheduler.total_warmup_steps > self.lr_scheduler._step)):  # step per batch if that is what we want to do or if we are using a warmup schedule and are still in the warmup period
    #         self.step_schedulers() --> self.lr_scheduler.step()
    # self.optim.zero_grad()
    # self.optim_steps += 1
    torch.autograd.set_detect_anomaly(True)
    train_loss_log_name = NAME + "_train"
    val_loss_log_name =  NAME + "_val"
    train_loss_log_total, val_loss_log_total = [], []
    
    kl_annealing = False
    kl_weight = 1e-6
    kl_annealing_rate = 1e-5
    kl_annealing_interval = 1
    kl_cap = 1e-3
    for epoch in range(cfg.data['epochs']):
        print("Epoch", epoch)
        if kl_annealing and epoch > 0 and epoch % kl_annealing_interval == 0:
            kl_weight += kl_annealing_rate
            kl_weight = min(kl_weight, kl_cap)
        if kl_annealing:
            model.kl_v_beta = kl_weight
        train_loss_log, val_loss_log = [], []
        for A_batch, B_batch in train_loader:
            A_graph, geo_A, Ap, A_cg, geo_A_cg, frag_ids = A_batch
            B_graph, geo_B, Bp, B_cg, geo_B_cg, B_frag_ids = B_batch

            A_graph, geo_A, Ap, A_cg, geo_A_cg = A_graph.to('cuda:0'), geo_A.to(
                'cuda:0'), Ap.to('cuda:0'), A_cg.to('cuda:0'), geo_A_cg.to('cuda:0')
            B_graph, geo_B, Bp, B_cg, geo_B_cg = B_graph.to('cuda:0'), geo_B.to(
                'cuda:0'), Bp.to('cuda:0'), B_cg.to('cuda:0'), geo_B_cg.to('cuda:0')

            generated_molecule, rdkit_reference, dec_results, channel_selection_info, KL_terms, enc_out, AR_loss = model(
                frag_ids, A_graph, B_graph, geo_A, geo_B, Ap, Bp, A_cg, B_cg, geo_A_cg, geo_B_cg, epoch=epoch)
            # ipdb.set_trace()
            loss, losses = model.loss_function(generated_molecule, rdkit_reference, dec_results, channel_selection_info, KL_terms, enc_out, geo_A, AR_loss, step=epoch)
            # train_loss_log.append(losses)
            # ipdb.set_trace()
            print(f"Train LOSS = {loss}")
            loss.backward()
            losses['Train Loss'] = loss.cpu()
            wandb.log(losses)

            for name, p in model.named_parameters():
                if p.requires_grad and p.grad is not None and (torch.isnan(p.grad).any() or torch.isnan(p.data).any()):
                    print("[LOG]", name, torch.min(p.grad).item(), torch.max(p.grad).item(), torch.min(p.data).item(), torch.max(p.data).item())

            parameters = [p for p in model.parameters(
            ) if p.grad is not None and p.requires_grad]
            norm_type = 2
            total_norm = 0.0
            for p in parameters:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            print(f"Train LOSS = {loss}")
            print("TOTAL GRADIENT NORM", total_norm)

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.optimizer.clip_grad_norm, norm_type=2) 
            optim.step()
            optim.zero_grad()

        print("Validation")
        with torch.no_grad():
            for A_batch, B_batch in val_loader:
                A_graph, geo_A, Ap, A_cg, geo_A_cg, frag_ids = A_batch
                B_graph, geo_B, Bp, B_cg, geo_B_cg, B_frag_ids = B_batch

                A_graph, geo_A, Ap, A_cg, geo_A_cg = A_graph.to('cuda:0'), geo_A.to('cuda:0'), Ap.to('cuda:0'), A_cg.to('cuda:0'), geo_A_cg.to('cuda:0')
                B_graph, geo_B, Bp, B_cg, geo_B_cg = B_graph.to('cuda:0'), geo_B.to('cuda:0'), Bp.to('cuda:0'), B_cg.to('cuda:0'), geo_B_cg.to('cuda:0')

                generated_molecule, rdkit_reference, dec_results, channel_selection_info, KL_terms, enc_out, AR_loss = model(
                        B_frag_ids, A_graph, B_graph, geo_A, geo_B, Ap, Bp, A_cg, B_cg, geo_A_cg, geo_B_cg, epoch=epoch, validation = True)
                # ipdb.set_trace()
                loss, losses = model.loss_function(generated_molecule, rdkit_reference, dec_results, channel_selection_info, KL_terms, enc_out, geo_A, AR_loss, step=epoch, log_latent_stats = False)
                # train_loss_log.append(losses)
                losses['Val Loss'] = loss.cpu()
                wandb.log({'val_' + key: value for key, value in losses.items()})
                print(f"Val LOSS = {loss}")
            
            print("Test Benchmarks")
            runner.generate(model)
            # TODO: Save Model
        


    print("Training Complete")


if __name__ == "__main__":
    main()
