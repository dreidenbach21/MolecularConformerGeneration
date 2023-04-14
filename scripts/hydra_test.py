import hydra
from omegaconf import DictConfig, OmegaConf
import ipdb
import torch
import wandb
import random
import logging

@hydra.main(config_path="../configs", config_name="config.yaml")
def main(cfg: DictConfig):
    # print(cfg.pretty())
    # ipdb.set_trace()
    logging.info(OmegaConf.to_yaml(cfg))
    logging.info(f'{torch.__version__} {torch.cuda.is_available()}')
    x = torch.ones((10000, 10000)).cuda()
    logging.info(x.shape)
    # print("t", x.shape)
    # ipdb.set_trace()
    # # start a new wandb run to track this script
    # wandb.init(
    #     # set the wandb project where this run will be logged
    #     project=cfg.wandb.project,
    #     name=cfg.wandb.name,
    #     notes=cfg.wandb.notes
    #     # # track hyperparameters and run metadata
    #     # config={
    #     # "learning_rate": 0.02,
    #     # "architecture": "CNN",
    #     # "dataset": "CIFAR-100",
    #     # "epochs": 10,
    #     # }
    # )
    # # simulate training
    # epochs = 10
    # offset = random.random() / 5
    # for epoch in range(2, epochs):
    #     acc = 1 - 2 ** -epoch - random.random() / epoch - offset
    #     loss = 2 ** -epoch + random.random() / epoch + offset
        
    #     # log metrics to wandb
    #     wandb.log({"acc": acc, "loss": loss})
        
    # # [optional] finish the wandb run, necessary in notebooks
    # wandb.finish()

if __name__ == "__main__":
    main()
