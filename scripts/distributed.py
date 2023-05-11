import sys
import os
import hydra
from omegaconf import DictConfig, OmegaConf
import ipdb
import torch
import wandb
import random
import logging
from utils.torsional_diffusion_data_all import * 
from model.parallel_vae import VAE
import datetime
from model.benchmarker import *
import glob
import torch.distributed as dist
from dgl.data import split_dataset
from dgl.dataloading import GraphDataLoader
import dgl.multiprocessing as mp
import torch
from torch.nn.parallel import DistributedDataParallel

drugs_types = {'H': 0, 'Li': 1, 'B': 2, 'C': 3, 'N': 4, 'O': 5, 'F': 6, 'Na': 7, 'Mg': 8, 'Al': 9, 'Si': 10,
               'P': 11, 'S': 12, 'Cl': 13, 'K': 14, 'Ca': 15, 'V': 16, 'Cr': 17, 'Mn': 18, 'Cu': 19, 'Zn': 20,
               'Ga': 21, 'Ge': 22, 'As': 23, 'Se': 24, 'Br': 25, 'Ag': 26, 'In': 27, 'Sb': 28, 'I': 29, 'Gd': 30,
               'Pt': 31, 'Au': 32, 'Hg': 33, 'Bi': 34}

def init_model(cfg, seed, device):
    torch.manual_seed(seed)
    model = VAE(cfg.vae, cfg.encoder, cfg.decoder, cfg.losses, coordinate_type = cfg.coordinates, device = device).to(device)
    model = DistributedDataParallel(model, device_ids=[device], output_device=device) #, find_unused_parameters=True)
    return model

def init_process_group(world_size, rank, port):
    # Generate a random port number
    print("INIT Process", world_size, rank, port)
    dist.init_process_group(
        backend='nccl',
        init_method=f'tcp://127.0.0.1:{port}',
        world_size=world_size,
        rank=rank)

def collate(samples):
    A, B = map(list, zip(*samples))
    A_graph = dgl.batch([x[0] for x in A])
    geo_A = dgl.batch([x[1] for x in A])
    Ap = dgl.batch([x[2] for x in A])
    A_cg = dgl.batch([x[3] for x in A])
    geo_A_cg = dgl.batch([x[4] for x in A])
    frag_ids = [x[5] for x in A]
    
    B_graph = dgl.batch([x[0] for x in B])
    geo_B = dgl.batch([x[1] for x in B])
    Bp = dgl.batch([x[2] for x in B])
    B_cg = dgl.batch([x[3] for x in B])
    geo_B_cg = dgl.batch([x[4] for x in B])
    B_frag_ids = [x[5] for x in B]
    return (A_graph, geo_A, Ap, A_cg, geo_A_cg, frag_ids), (B_graph, geo_B, Bp, B_cg, geo_B_cg, B_frag_ids)

def load_data(mode = 'train', data_dir='/home/dannyreidenbach/data/DRUGS/drugs/',dataset='drugs', limit_mols=0,
              log_dir='./test_run', num_workers=1, restart_dir=None, seed=0,
              split_path='/home/dannyreidenbach/data/DRUGS/split.npy',std_pickles=None):
    types = drugs_types
    data = ConformerDataset(data_dir, split_path, mode, dataset=dataset,
                                   types=types, transform=None,
                                   num_workers=num_workers,
                                   limit_molecules=limit_mols, #args.limit_train_mols,
                                   cache_path=None, #args.cache,
                                   name=f'{dataset}_{mode}_{limit_mols}_final',
                                   pickle_dir=std_pickles,
                                   raw_dir='/home/dannyreidenbach/data/DRUGS/dgl', 
                                   save_dir='/home/dannyreidenbach/data/DRUGS/dgl',
                                   use_diffusion_angle_def=False,
                                   boltzmann_resampler=None)
    return data

def get_dataloader(dataset, seed, batch_size=300, num_workers=1, mode = 'train'):
    if mode == 'train':
        use_ddp = True
    else:
        use_ddp = False
    dataloader = dgl.dataloading.GraphDataLoader(dataset, use_ddp=use_ddp, batch_size=batch_size,
                                                 shuffle=True, drop_last=False, num_workers=num_workers, collate_fn = collate)
    print("Data Loader", mode)
    return dataloader

def get_contributing_params(y, top_level=True):
    nf = y.grad_fn.next_functions if top_level else y.next_functions
    for f, _ in nf:
        try:
            yield f.variable
        except AttributeError:
            pass  # node has no tensor
        if f is not None:
            yield from get_contributing_params(f, top_level=False)

def run(cfg, name, port, rank, world_size, train_dataset, val_dataset, seed=0):
    init_process_group(world_size, rank, port)
    # Assume the GPU ID to be the same as the process ID
    device = torch.device('cuda:{:d}'.format(rank))
    torch.cuda.set_device(device)

    model = init_model(cfg, seed, device)

    train_loader = get_dataloader(train_dataset,seed)
    val_loader = get_dataloader(train_dataset,seed)
    NAME = name
    print("CUDA CHECK", NAME, next(model.parameters()).is_cuda)
    # print("# of Encoder Params = ", sum(p.numel()
    #       for p in model.encoder.parameters() if p.requires_grad))
    # print("# of Decoder Params = ", sum(p.numel()
    #       for p in model.decoder.parameters() if p.requires_grad))
    # print("# of VAE Params = ", sum(p.numel()
    #       for p in model.parameters() if p.requires_grad))
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
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=1, gamma=0.8)


    
    kl_annealing = True
    kl_weight = 1e-5
    kl_annealing_rate = 1e-3
    kl_annealing_interval = 1
    kl_cap = 1e-1
    
    dist_weight = 1e-6
    dist_annealing_rate = 0.1
    dist_cap = 0.5
    for epoch in range(cfg.data['epochs']):
        print("Epoch", epoch)
        # The line below ensures all processes use a different
        # random ordering in data loading for each epoch.
        model.train()
        train_loader.set_epoch(epoch)
        
        
        # if kl_annealing and epoch > 0 and epoch % kl_annealing_interval == 0:
        #     kl_weight += kl_annealing_rate
        #     kl_weight = min(kl_weight, kl_cap)
            
        #     dist_weight += dist_annealing_rate
        #     dist_weight = min(dist_weight, dist_cap)
        # if kl_annealing:
        #     model.module.kl_v_beta = kl_weight
        #     model.module.lambda_distance = dist_weight
        count = 0
        
        for A_batch, B_batch in train_loader:
            A_graph, geo_A, Ap, A_cg, geo_A_cg, frag_ids = A_batch
            B_graph, geo_B, Bp, B_cg, geo_B_cg, B_frag_ids = B_batch
            A_graph, geo_A, Ap, A_cg, geo_A_cg = A_graph.to(device), geo_A.to(
                device), Ap.to(device), A_cg.to(device), geo_A_cg.to(device)
            B_graph, geo_B, Bp, B_cg, geo_B_cg = B_graph.to(device), geo_B.to(
                device), Bp.to(device), B_cg.to(device), geo_B_cg.to(device)
            print("forward", rank, count)
            generated_molecule = model(rank, frag_ids, A_graph, B_graph, geo_A, geo_B, Ap, Bp, A_cg, B_cg, geo_A_cg, geo_B_cg)
            
            contributing_parameters = set(get_contributing_params(generated_molecule))
            all_parameters = set(model.parameters())
            non_contributing = all_parameters - contributing_parameters
            print(non_contributing)
            
            loss, losses = model.module.loss_function(generated_molecule, rank, geo_A)
            print(f"Train LOSS = {loss}")
            loss.backward()
            losses['Train Loss'] = loss.cpu()
            if rank == 0:
                wandb.log(losses)
            parameters = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
            norm_type = 2
            total_norm = 0.0
            for p in parameters:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            print(f"Rank {rank} count {count} Train LOSS = {loss}")
            print("TOTAL GRADIENT NORM", total_norm)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.optimizer.clip_grad_norm, norm_type=2) 
            optim.step()
            optim.zero_grad()
            
            del A_graph, geo_A, Ap, A_cg, geo_A_cg, frag_ids
            del B_graph, geo_B, Bp, B_cg, geo_B_cg, B_frag_ids
            del generated_molecule, model.module.storage[rank]#rdkit_reference, dec_results, channel_selection_info, KL_terms, enc_out, AR_loss, losses
            if count > 0 and count %10 == 0:
                torch.cuda.empty_cache()
                # model_path = f'/home/dannyreidenbach/mcg/coagulation/scripts/model_ckpt/{NAME}_{epoch}_temp.pt'
                # torch.save(model.state_dict(), model_path)
            count+=1

        print("Validation")
        val_loss = 0
        model.eval()
        with torch.no_grad():
            for A_batch, B_batch in val_loader:
                A_graph, geo_A, Ap, A_cg, geo_A_cg, frag_ids = A_batch
                B_graph, geo_B, Bp, B_cg, geo_B_cg, B_frag_ids = B_batch

                A_graph, geo_A, Ap, A_cg, geo_A_cg = A_graph.to(device), geo_A.to(device), Ap.to(device), A_cg.to(device), geo_A_cg.to(device)
                B_graph, geo_B, Bp, B_cg, geo_B_cg = B_graph.to(device), geo_B.to(device), Bp.to(device), B_cg.to(device), geo_B_cg.to(device)

                generated_molecule = model(rank, B_frag_ids, A_graph, B_graph, geo_A, geo_B, Ap, Bp, A_cg, B_cg, geo_A_cg, geo_B_cg, validation = True)
                loss, losses = model.module.loss_function(generated_molecule, rank, geo_A, log_latent_stats = False)
                # train_loss_log.append(losses)
                losses['Val Loss'] = loss.cpu()
                # val_loss += losses['Val Loss']
                if rank == 0:
                    wandb.log({'val_' + key: value for key, value in losses.items()})
                print(f"Val LOSS = {loss}")
            
            # print("Test Benchmarks")
            # BENCHMARK.generate(model)
            
        # scheduler.step(val_loss)
        scheduler.step()
        model_path = f'/home/dannyreidenbach/mcg/coagulation/scripts/model_ckpt/{NAME}_{epoch}_{rank}.pt'
        torch.save(model.module.state_dict(), model_path)
        
    print("Training Complete", rank, world_size, seed)
    dist.destroy_process_group()
    
@hydra.main(config_path="../configs", config_name="config_drugs.yaml")
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
    num_gpus = 4
    procs = []
    train_dataset = load_data(mode = 'train')
    val_dataset = load_data(mode = 'val')
    port = random.randint(10000, 20000)
    for rank in range(num_gpus):
        p = mp.Process(target=run, args=(cfg, NAME, port, rank, num_gpus, train_dataset, val_dataset))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()

# Thumbnail credits: DGL
# sphinx_gallery_thumbnail_path = '_static/blitz_5_graph_classification.png'
if __name__ == '__main__':
    main()