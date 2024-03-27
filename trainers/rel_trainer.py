import os
import time

from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from relbench.data import RelBenchDataset
from data.dataset import load_rel_partition
from relbench.datasets import get_dataset
from relbench.data.database import Database

from torch_geometric.data import HeteroData
from relbench.external.graph import get_node_train_table_input, make_pkey_fkey_graph, NodeTrainTableInput
from relbench.data import NodeTask
from data.rel_dataset import DistrRelBenchDataset

from torch_frame.config.text_embedder import TextEmbedderConfig
from torch_geometric.distributed.local_feature_store import LocalFeatureStore
from torch_geometric.distributed.local_graph_store import LocalGraphStore

from text_embedder import GloveTextEmbedding
from inferred_stypes import dataset2inferred_stypes
from omegaconf import DictConfig
from comm_utils import get_boundary_nodes_pyg


def train(
    graph: HeteroData,
    node_dict: Dict,
    local_dict: Dict,
    table_input: NodeTrainTableInput,
    dataset: DistrRelBenchDataset,
    task: NodeTask,
    cfg: DictConfig,
    hydra_output_dir: str,
    results_dir: str,
) -> None:
    """Implement end-to-end training process
    
    Parameters
    ----------
    TODO
    
    
    
    Returns
    -------
    None
    """
    
    get_boundary_nodes_pyg(graph, table_input, node_dict, local_dict)

    
    # TODO
    # - get boundary nodes
    # - send and receive embeddings
    # - start training by layer
    # - msg passing in-between layers
    # - model update
    # - evaluate model
    # - save model
    
    raise NotImplementedError


def init_process(rank, cfg, hydra_output_dir):
    """Initialize the distributed environment"""

    os.environ["MASTER_ADDR"] = cfg.distributed.master_addr
    os.environ["MASTER_PORT"] = cfg.distributed.master_port
    dist.init_process_group(
        cfg.distributed.backend, rank=rank, world_size=cfg.num_partitions
    )

   

    # dataset: RelBenchDataset = get_dataset(cfg.dataset_name, process=True)
    # path = os.path.join(os.getcwd(), "data")
    # databases = dataset.shardDataset(num_shards=cfg.num_partitions, folder=path)

    # TODO
    # 1. Load the partition directly (using rank)
    
    dataset, task, node_dict, local_dict = load_rel_partition(partition_dir=(f"{cfg.partition_dir}/{cfg.dataset_name}"), dataset_name=cfg.dataset_name, task_name=cfg.task.name, part_id=rank)
    col_to_stype_dict = dataset2inferred_stypes[cfg.dataset_name]
    graph, col_stats_dict = make_pkey_fkey_graph(
        dataset.db,
        col_to_stype_dict=col_to_stype_dict,
        text_embedder_cfg=TextEmbedderConfig(
            text_embedder=GloveTextEmbedding(device=cfg.device), batch_size=256
        ),
        cache_dir=os.path.join(cfg.partition_dir, f"materialized_cache/{rank}"),
    )

    table_input = get_node_train_table_input(table=task.train_table, task=task)
    # graph = None
    # table_input = None
    
    os.makedirs("results/", exist_ok=True)
    os.makedirs(
        f"results/{cfg.dataset.partition.dataset_name}_{cfg.model.conv_layer._target_}_{cfg.dataset.partition.num_parts}/",
        exist_ok=True,
    )
    os.makedirs(
        f"results/{cfg.dataset.partition.dataset_name}_{cfg.model.conv_layer._target_}_{cfg.dataset.partition.num_parts}/rank_{rank}/",
        exist_ok=True,
    )

    results_dir = f"results/{cfg.dataset.partition.dataset_name}_{cfg.model.conv_layer._target_}_{cfg.dataset.partition.num_parts}/rank_{rank}/"
    
    start_time = time.time()
    
    train(
        graph,
        node_dict,
        local_dict,
        table_input,
        dataset,
        task,
        cfg,
        hydra_output_dir,
        results_dir,
    )   
    
    print(f"Rank {rank:2} | Total time taken: {time.time() - start_time:2.4f} s")
    dist.destroy_process_group()
