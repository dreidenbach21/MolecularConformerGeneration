import hydra
from omegaconf import DictConfig, OmegaConf
import ipdb
import torch

@hydra.main(config_path="../configs", config_name="config.yaml")
def main(cfg: DictConfig):
    # print(cfg.pretty())
    # ipdb.set_trace()
    print(OmegaConf.to_yaml(cfg))
    print(torch.__version__, torch.cuda.is_available())
    x = torch.ones((10000, 10000)).cuda()
    ipdb.set_trace()

if __name__ == "__main__":
    main()
