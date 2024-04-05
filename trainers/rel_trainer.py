import os
import time

from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from relbench.data import RelBenchDataset
from data.dataset import load_rel_partition
from relbench.datasets import get_dataset
from relbench.data.database import Database
from models.rel_model import RelModel

from torch_geometric.data import HeteroData
from relbench.external.graph import get_node_train_table_input, make_pkey_fkey_graph, NodeTrainTableInput
from relbench.data import NodeTask
from data.rel_dataset import DistrRelBenchDataset

from torch_frame.config.text_embedder import TextEmbedderConfig
from torch_geometric.distributed.local_feature_store import LocalFeatureStore
from torch_geometric.distributed.local_graph_store import LocalGraphStore
from torch_geometric.loader import NeighborLoader

from text_embedder import GloveTextEmbedding
from inferred_stypes import dataset2inferred_stypes
from hydra.utils import instantiate
from omegaconf import DictConfig
from comm_utils import get_boundary_nodes_pyg, send_and_receive_embeddings_pyg


def set_torch_seed(seed):
    """Set the seed for torch"""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# def get_in_subgraph(
#     graph: HeteroData, table_input: NodeTrainTableInput
# ):
#     nodes, filter_edge_idx, inv, edge_mask = 
#     subgraph_edge_attr = graph.edge_attr[edge_mask] if graph.edge_attr is not None else graph.edge_attr
#     sub_g = HeteroData(x=graph.x[nodes], edge_index=filter_edge_idx,
#                                  edge_attr=subgraph_edge_attr)
    

def train(
    graph: HeteroData,
    node_dict: Dict,
    local_dict: Dict,
    dataset: DistrRelBenchDataset,
    col_stats_dict: Dict,
    task: NodeTask,
    cfg: DictConfig,
    hydra_output_dir: str,
    results_dir: str,
) -> None:
    """Implement end-to-end training process
    
    Parameters
    ----------    
    
    Returns
    -------
    None
    """
    
    # set the seed
    set_torch_seed(cfg.seed)
    
    # setup the model
    model = RelModel(
        data=graph,
        col_stats_dict=col_stats_dict,
        num_layers=cfg.num_layers,
        channels=cfg.channels,
        out_channels=cfg.out_channels,
        aggr=cfg.model.aggregator_type,
        norm="batch_norm",
    )#.to(device) ???
    # TODO maybe setup optimizer later ? 
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate[0])
    device = cfg.device
    if device == "cuda":
        model = model.cuda()
        
    # TODO incorporate performance stores
    # perf_stores = [PerformanceStore()]
    # perf_store = perf_stores[0]
    
    rank, cluster_size = dist.get_rank(), size.get_world_size()
    os.makedirs(os.path.join(hydra_output_dir, "results"), exist_ok=True)
    if rank == 0:
        os.makedirs(os.path.join(hydra_output_dir, "checkpoint"), exist_ok=True)
    
    # get the boundary nodes lists
    boundary_nodes = get_boundary_nodes_pyg(graph, node_dict, local_dict)

    # where to store embeddings ? Where in relbench?
    # => in relbench's data graph there is an "embedding" field for each table
    # TODO find out how many nodes we have locally
    # TODO should we store per table? or just for "TableInput" nodes ? 
    # => last answer should be in relbench paper
    
    
    local_dict["feat_0"] = torch.zeros((len(nodes)), num_feat)
    # TODO retrieve inner node indices, find out what torch.arrange does
    inner_node_indices =  torch.arrange(node_dict["part_id"] == rank)
    local_dict["feat_0"][inner_node_indices] = 0# TODO the features we know? here we have not instanciated the task yet ! 
    
    
    # Share the zeroth embedding of all nodes to their neighbors
    send_and_receive_embeddings_pyg(
        boundary_nodes, node_dict, "feat_0", 
    )
    
    # Create the 3 full-batches
    batch_dict: Dict[str, HeteroData] = {}
    for split, table in [
        ("train", task.train_table),
        ("val", task.val_table),
        ("test", task.test_table),
    ]:
        table_input = get_node_train_table_input(table=table, task=task)
        entity_table = table_input.nodes[0]
        
        batch = list(NeighborLoader(
            graph,
            num_neighbors=[-1], # one layer batch with all neighbors
            time_attr="time",
            input_nodes=table_input.nodes,
            input_time=table_input.time,
            transform=table_input.transform,
            batch_size=graph.num_nodes, # whole graph
            temporal_strategy=cfg.temporal_strategy,
            shuffle=split == "train",
            num_workers=4,
            persistent_workers=4 > 0,
        ))
        assert(len(batch) == 1)
        batch_dict[split] = batch[0]
        
    # TODO adapt for pyg graph
    # emb_data_thread = threading.Thread(
    #     target=construct_graph_and_features_to_compute_next_embedding,
    #     args=(emb_data_dict, task, graph, node_dict, prev_feat_tag, inner_node_indices),
    # )
    # emb_data_thread.start()
    
    # train for one layer
    # TODO
    
    # send and receive embeddings, prepare next
    # TODO
    
    # Do each remaining layer
    # TODO
    
    # Display and save results
    # TODO
    
    
    # === -- notes & questions to myself -- ===
    # are "features" what we computed in the "task" ? 
    # we do message passing because inner_nodes have the most accurate embeddings,
    # we can't compute the true embedding of boundary nodes locally.
    # TODO raises the question of are the inner_nodes in my partitioning
    # true inner nodes? For users yes, but the rest I am not sure. 
    # Hopefully we have "more" inner nodes than thought

    
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

    # table_input = get_node_train_table_input(table=task.train_table, task=task)
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
        dataset,
        col_stats_dict,
        task,
        cfg,
        hydra_output_dir,
        results_dir,
    )   
    
    print(f"Rank {rank:2} | Total time taken: {time.time() - start_time:2.4f} s")
    dist.destroy_process_group()
