"""Entry point to the application."""

import logging
import os
import hydra
from hydra.core.hydra_config import HydraConfig
import torch
import torch.multiprocessing as mp
from omegaconf import DictConfig, OmegaConf

import trainers.trainer
from data.dataset import load_data, graph_partition

# logging.basicConfig(level = logging.INFO)

@hydra.main(config_path="conf", config_name="node_classification", version_base=None)
def main(cfg: DictConfig) -> None:
    """Run the specified application"""

    print(OmegaConf.to_yaml(cfg))

    # get the hydra output directory
    hydra_output_dir = HydraConfig.get().runtime.output_dir

    if cfg.app == "partition_data":
        graph, _, _ = load_data(**cfg.dataset.download)
        graph_partition(graph, **cfg.dataset.partition)
        return
    elif cfg.app == "train":
        train = trainers.trainer
        # set up the distributed training environment
        if cfg.distributed.backend == "gloo":
            n_devices = torch.cuda.device_count()
            devices = [f"{i}" for i in range(n_devices)]

            if "CUDA_VISIBLE_DEVICES" in os.environ:
                devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
                n_devices = len(devices)

            start_id = cfg.node_rank * cfg.parts_per_node
            end_id = min(start_id + cfg.parts_per_node, cfg.num_partitions)

            process = []
            torch.multiprocessing.set_start_method('spawn')
            for i in range(start_id, end_id):
                os.environ["CUDA_VISIBLE_DEVICES"] = devices[i%len(devices)]
                p = mp.Process(target=train.init_process, args=(i, cfg, hydra_output_dir))
                p.start()
                process.append(p)
            for p in process:
                p.join()
        else:
            raise ValueError(
                f"Backend {cfg.distributed.backend} is not supported."
            )
    else:
        raise ValueError(
            f"Backend {cfg.app} is not supported."
        )

if __name__ == "__main__":
    main() # pylint: disable=no-value-for-parameter
