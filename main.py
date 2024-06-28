"""Entry point to the application."""

import logging
import os
import hydra
from hydra.core.hydra_config import HydraConfig
import torch
import torch.multiprocessing as mp
from omegaconf import DictConfig, OmegaConf

from data.dataset import load_data, graph_partition, rel_graph_partition, load_partition, load_user_item_data
import trainers.fedgnn_trainer
import trainers.pos_neg_hetero_trainer
import trainers.pos_neg_hetero_trainer_networking
import trainers.trainer
import trainers.rel_trainer
import trainers.user_item_trainer


from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends import default_backend

# logging.basicConfig(level = logging.INFO)

@hydra.main(config_path="conf", config_name="news_recommendation", version_base=None)
def main(cfg: DictConfig) -> None:
    """Run the specified application"""

    print(OmegaConf.to_yaml(cfg))
    # get the hydra output directory
    hydra_output_dir = HydraConfig.get().runtime.output_dir
    
    # temporary code to handle keys locally for test
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
        backend=default_backend()
    )
    public_key = private_key.public_key()
    with open("private_key.pem", "wb") as private_file:
        private_file.write(
            private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            )
        )
    with open("public_key.pem", "wb") as public_file:
        public_file.write(
            public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            )
        )

    if cfg.app == "partition_data":
        graph, _= load_data(**cfg.dataset.download, cfg=cfg)
        # graph_partition(graph, **cfg.dataset.partition)
        # load_partition(cfg.dataset.partition.partition_dir, cfg.dataset.partition.dataset_name, 0)
        return
    elif cfg.app == "partition_relational_data":
        rel_graph_partition(
            cfg.dataset_name, cfg.partition_dir, cfg.num_partitions, cfg
         )   
        return
    elif cfg.app == "partition_user_item_data":
        graph, itemList = load_user_item_data(**cfg.dataset.download)
        graph_partition(graph, **cfg.dataset.partition)
        temp = load_partition(cfg.dataset.partition.partition_dir, cfg.dataset.partition.dataset_name, 0, task="edge_prediction")
        return
    elif cfg.app == "hetero_pos_neg_train":
        if cfg.federated:
            train = trainers.pos_neg_hetero_trainer_networking
            if cfg.distributed.backend == "gloo":
                # if cfg.master: 
                    # rank = 0
                    # init_master(cfg, hydra_output_dir)
                # else:
                    n_devices = torch.cuda.device_count()
                    devices = [f"{i}" for i in range(n_devices)]

                    if "CUDA_VISIBLE_DEVICES" in os.environ:
                        devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
                        n_devices = len(devices)
                        
                    torch.multiprocessing.set_start_method('spawn')
                    os.environ["CUDA_VISIBLE_DEVICES"] = devices[0]
                    master_p = mp.Process(target=train.init_master, args=(cfg, hydra_output_dir))
                    worker_p = mp.Process(target=train.init_process, args=(1, cfg, hydra_output_dir))
                    
                    master_p.start()
                    worker_p.start()
                    master_p.join()
                    worker_p.join()
        else:
            train = trainers.pos_neg_hetero_trainer
            if cfg.distributed.backend == "gloo":
                n_devices = torch.cuda.device_count()
                devices = [f"{i}" for i in range(n_devices)]

                if "CUDA_VISIBLE_DEVICES" in os.environ:
                    devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
                    n_devices = len(devices)
                    
                torch.multiprocessing.set_start_method('spawn')
                os.environ["CUDA_VISIBLE_DEVICES"] = devices[0]
                p = mp.Process(target=train.init_process, args=(0, cfg, hydra_output_dir))
                p.start()
                p.join()
        
    elif cfg.app == "fedgnn":
        train = trainers.fedgnn_trainer
        if cfg.distributed.backend == "gloo":
            n_devices = torch.cuda.device_count()
            devices = [f"{i}" for i in range(n_devices)]

            if "CUDA_VISIBLE_DEVICES" in os.environ:
                devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
                n_devices = len(devices)
                
            torch.multiprocessing.set_start_method('spawn')
            os.environ["CUDA_VISIBLE_DEVICES"] = devices[0]
            p = mp.Process(target=train.init_process, args=(0, cfg, hydra_output_dir))
            p.start()
            p.join()
    elif cfg.app == "train":
        # train = trainers.rel_trainer # TODO change depending on cfg
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
            # end_id = int(start_id + cfg.num_partitions / cfg.parts_per_node)
            
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
