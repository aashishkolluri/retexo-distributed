import os
import time
import copy
import logging

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
from torch.nn import L1Loss, BCEWithLogitsLoss
from relbench.data import RelBenchDataset
from data.dataset import load_rel_partition
from relbench.datasets import get_dataset
from relbench.data.database import Database
from relbench.data.task_base import TaskType
from models.rel_model import RelModel, IntermediateRelModel, FirstLayerRelModel

from torch_geometric.data import HeteroData
from relbench.external.graph import get_node_train_table_input, make_pkey_fkey_graph, NodeTrainTableInput
from relbench.data import NodeTask
from data.rel_dataset import DistrRelBenchDataset

from torch_frame.config.text_embedder import TextEmbedderConfig
from torch_geometric.distributed.local_feature_store import LocalFeatureStore
from torch_geometric.distributed.local_graph_store import LocalGraphStore
from torch_geometric.loader import NeighborLoader

from performance import PerformanceStore
from text_embedder import GloveTextEmbedding
from inferred_stypes import dataset2inferred_stypes
from hydra.utils import instantiate
from omegaconf import DictConfig
from comm_utils import get_boundary_nodes_pyg, send_and_receive_embeddings_pyg, sync_model, aggregate_metrics, MultiThreadReducerCentralized
from trainers.worker_trainer import WorkerTrain

logger = logging.getLogger(__name__)
comm_volume_perf_store = PerformanceStore()

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

def setup_model(entire_model, curr_layer, device):
    """Set up the model"""
    # train the nth layer of the model
    curr_model = entire_model.get_nth_layer(curr_layer)
    if device == "cuda":
        curr_model = curr_model.cuda()
    return curr_model

    
def train_for_one_layer(
    model: IntermediateRelModel,
    task,
    batch_dict: Dict[str, HeteroData],
    x_dicts: Tuple[Dict],
    optimizer,
    tune_metric: str,   
    loss_fn,
    entity_table,
    cfg,
    all_num_nodes,
    total_task_nodes,
    num_rounds,
    clamp_min, 
    clamp_max,
    worker_trainer,
    results_dir):
    """Train for one layer"""
    if x_dicts is None:
        x_dicts = [None, None, None]
    if cfg.device == "cuda":
        batch_train = batch_dict["train"].to(torch.device("cuda"))
        batch_val = batch_dict["val"].to(torch.device("cuda"))
        batch_test = batch_dict["test"].to(torch.device("cuda"))
    
    rank = dist.get_rank()
    if rank == 0:
        # True for classification, False for regression
        higher_is_better = task.task_type == TaskType.BINARY_CLASSIFICATION 
        best_params = model.state_dict()
        best_val_loss = float("inf")
        # best_val_acc = 0.0
        # best_test_acc = 0.0
        best_val_metric = 0 if higher_is_better else float("inf")
        best_epoch = 0

    reducer = MultiThreadReducerCentralized(
        model, cfg.sleep_time, comm_volume_perf_store, cfg.measure_dv
    )
    
    def train_one_round() -> float:
        model.train()

        optimizer.zero_grad()
        if x_dicts[0] is None: # First layer
            pred, x_dict_train = model( 
                batch_train,
                task.entity_table,
            )
        else: 
            pred, x_dict_train = model(
                batch_train[task.entity_table].seed_time,
                x_dicts[0],
                batch_train.edge_index_dict,
                task.entity_table,
            )
        pred = pred.view(-1) if pred.size(1) == 1 else pred        
        # pred = pred[batch_train]
        expected = (batch_train[entity_table].y).detach()
        loss = loss_fn(pred, expected)
        train_loss = loss.detach().item()
        loss.backward()

        return x_dict_train, train_loss
        
    # do all rounds on current layer
    for training_round in range(num_rounds):
        
        start_time = time.time()
        x_dict_train, train_loss = train_one_round()
        time_round = time.time() - start_time
        
        agg_time_s = time.time()
        with torch.no_grad():
            reducer.aggregate_hetero_grad(model, batch_train.node_types, all_num_nodes)
        agg_time = time.time() - agg_time_s
        
        optimizer.step()
        
        
        val_pred, x_dict_val = test(batch_val, x_dicts[1], model, cfg.device, clamp_min, clamp_max, task)
        val_metrics = task.evaluate(val_pred, task.val_table)
        
        # sync the val metrics
        for k, v in val_metrics.items():
            val_metrics[k] = v * (task.val_table.df.shape[0] / total_task_nodes["val"])
        val_metrics = aggregate_metrics(val_metrics)
            
        if rank == 0:
            # store the model with best val accuracy
            
            if (higher_is_better and val_metrics[tune_metric] > best_val_metric) or (
                not higher_is_better and val_metrics[tune_metric] < best_val_metric
            ):
                best_val_metric = val_metrics[tune_metric]
                best_params = copy.deepcopy(model.state_dict())
                best_val_loss = train_loss
                best_epoch = training_round


        # TODO log every n round
        if (
            training_round 
        ) % cfg.log_every == 0 or training_round == num_rounds - 1:
            # evaluate the model
            test_pred, x_dict_test = test(batch_test, x_dicts[2], model, cfg.device, clamp_min, clamp_max, task)
            test_metrics = task.evaluate(test_pred)
            # aggregate the test metrics
            for k, v in test_metrics.items():
                test_metrics[k] = v * (task.test_table.df.shape[0]/ total_task_nodes["test"])
            test_metrics =  aggregate_metrics(test_metrics)
            
            print(
                f"Rank {rank:2} | Training Round {training_round:2} |"
            )

            with open(results_dir + "accuracy.txt", "a+") as f:
                f.write(
                    f'Epoch {training_round}, train loss: {train_loss}, {tune_metric}: {val_metrics[tune_metric]}, {test_metrics[tune_metric]}\n'
                )
            # log the metrics
            print_str = f"Rank {rank:2} | Train"
            print_str += f" | train loss: {train_loss:2.4f}"
            print(print_str)
            print_str = f"Rank {rank:2} | Val  "
            for k, v in val_metrics.items():
                print_str += f" | {k}: {v:2.4f} "
            print(print_str)
            print_str = f"Rank {rank:2} | Test "
            for k, v in test_metrics.items():
                print_str += f" | {k}: {v:2.4f} "
            print(print_str)

    if rank == 0:
        model.load_state_dict(best_params)
    sync_model(model)
    
    test_pred, x_dict_test = test(batch_test, x_dicts[2], model, cfg.device, clamp_min, clamp_max, task)
    test_metrics = task.evaluate(test_pred)

    for k, v in test_metrics.items():
        test_metrics[k] = v * (task.test_table.df.shape[0]/ total_task_nodes["test"])
    test_metrics = aggregate_metrics(test_metrics)

    if rank == 0:
        print(f"Best model at epoch {best_epoch} | train loss {best_val_loss:5.4f}")
        print(f"Best model val metrics: {best_val_metric}")
        print(f"Best model test metrics: {test_metrics}")
        # print(f"Best test accuracy achieved: {best_test_acc}")
        print("-------------------------------------------" * 3)
        with open(results_dir + "best_stats.txt", "a+") as f:
            f.write(
                f'Epoch {best_epoch}, {best_val_metric}, {test_metrics}\n'
            )
    
    time.sleep(5)
    return x_dict_train, x_dict_val, x_dict_test

@torch.no_grad()
def test(batch, x_dict, model, device, clamp_min, clamp_max, task) -> np.ndarray:
    model.eval()
    

    pred_list = []
    batch = batch.to(device)
    with torch.no_grad():
        if x_dict is None: # First layer
            pred, x_dict = model( 
                batch,
                task.entity_table,
            )
        else: 
            pred, x_dict = model(
                batch[task.entity_table].seed_time,
                x_dict,
                batch.edge_index_dict,
                task.entity_table,
            )
   
    if task.task_type == TaskType.REGRESSION:
        pred = torch.clamp(pred, clamp_min, clamp_max)
    if task.task_type == TaskType.BINARY_CLASSIFICATION:
        pred = torch.sigmoid(pred)
    
    pred = pred.view(-1) if pred.size(1) == 1 else pred
    
    detached = pred.detach().cpu()
    pred_list.append(detached)
    
    res = torch.cat(pred_list, dim=0).numpy()
    return res, x_dict


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
        aggr=cfg.model.conv_layer.aggregator_type,
        norm="batch_norm",
    )
   
    perf_stores = [PerformanceStore()]
    perf_store = perf_stores[0]
    
    device = cfg.device
    if device == "cuda":
        model = model.cuda()

    
    rank, cluster_size = dist.get_rank(), dist.get_world_size()
    os.makedirs(os.path.join(hydra_output_dir, "results"), exist_ok=True)
    if rank == 0:
        os.makedirs(os.path.join(hydra_output_dir, "checkpoint"), exist_ok=True)

    logger.info(
        "Process %d has %d nodes and %d edges",
        rank,
        graph.num_nodes,
        graph.num_edges,
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
            persistent_workers=False,
        ))
        assert(len(batch) == 1)
        batch_dict[split] = batch[0] 
        
    batch_dict["train"].to(device)

    
    # get the boundary nodes lists
    boundary_nodes, all_num_nodes, all_task_nodes, available = get_boundary_nodes_pyg(batch_dict["train"], task, local_dict)
    
    # set up the model for the first layer
    curr_model = setup_model(model, 0, cfg.device)
    sync_model(curr_model)
    
    if isinstance(cfg.learning_rate, float):
        cfg.learning_rate = [cfg.learning_rate]
    optimizer = torch.optim.Adam(curr_model.parameters(), lr=cfg.learning_rate[0])
   
    if task.task_type == TaskType.REGRESSION:
        tune_metric = "mae"
        loss_fn = L1Loss()
        clamp_min, clamp_max = np.percentile(
            task.train_table.df[task.target_col].to_numpy(), [2, 98]
        )
    elif task.task_type == TaskType.BINARY_CLASSIFICATION:
        tune_metric = "roc_auc"
        loss_fn = BCEWithLogitsLoss()
        clamp_min, clamp_max = None, None
    
    worker_trainer = WorkerTrain()
    if isinstance(cfg.num_rounds, int):
        cfg.num_rounds = [cfg.num_rounds]
   
    prev_feat_tag = "feat_0"
    
    # train for first layer (only the encoders) 
    x_dicts = train_for_one_layer(
        curr_model,
        task,
        batch_dict,
        None,
        optimizer,
        tune_metric,   
        loss_fn,
        entity_table,
        cfg,
        all_num_nodes,
        all_task_nodes,
        cfg.num_rounds[0],
        clamp_min,
        clamp_max,
        worker_trainer,
        results_dir
    )
    
    
    for curr_layer in range(1, cfg.model.n_layers + 1):
        
        # send_and_receive_embeddings_pyg(
        #     batch_dict["train"], boundary_nodes, x_dicts[0], local_dict, available
        # )
            
        curr_model = setup_model(model, curr_layer, cfg.device)
        sync_model(curr_model)
        
        # reset the optimizer
        optimizer = instantiate(
            cfg.optimizer,
            lr=cfg.learning_rate[curr_layer]
            if curr_layer < len(cfg.learning_rate)
            else cfg.learning_rate[0],
            params=curr_model.parameters(),
        )
            
        # - train for one layer
        x_dicts = train_for_one_layer(
            curr_model,
            task,
            batch_dict,
            x_dicts,
            optimizer,
            tune_metric,   
            loss_fn,
            entity_table,
            cfg,
            all_num_nodes,
            all_task_nodes,
            cfg.num_rounds[0],
            clamp_min,
            clamp_max,
            worker_trainer,
            results_dir
        )
    
    # save the model
    if rank == 0:
        torch.save(
            model.state_dict(), os.path.join(hydra_output_dir, "checkpoint", "model.pt")
        )
        print(
            f"\nModel saved at {os.path.join(hydra_output_dir, 'checkpoint', 'model.pt')}"
        )

    


def init_process(rank, cfg, hydra_output_dir):
    """Initialize the distributed environment"""

    os.environ["MASTER_ADDR"] = cfg.distributed.master_addr
    os.environ["MASTER_PORT"] = cfg.distributed.master_port
    dist.init_process_group(
        cfg.distributed.backend, rank=rank, world_size=cfg.num_partitions
    )

    # 1. Load the partition directly (using rank)
    # rank = (rank + 1) % dist.get_world_size()
    
    dataset, task, node_dict, local_dict = load_rel_partition(partition_dir=(f"{cfg.partition_dir}"), dataset_name=cfg.dataset_name, task_name=cfg.task.name, part_id=rank)
    col_to_stype_dict = dataset2inferred_stypes[cfg.dataset_name]
    graph, col_stats_dict = make_pkey_fkey_graph(
        dataset.db,
        col_to_stype_dict=col_to_stype_dict,
        text_embedder_cfg=TextEmbedderConfig(
            text_embedder=GloveTextEmbedding(device=cfg.device), batch_size=256
        ),
        cache_dir=os.path.join(cfg.partition_dir, f"materialized_cache/{rank}"),
    )
    
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

    # TODO I removed GlobalId from s_types to compare with raw graph
        
    # dataset: RelBenchDataset = get_dataset(cfg.dataset_name, process=True)
    # task: NodeTask = dataset.get_task(cfg.task.name, process=True)

    # col_to_stype_dict = dataset2inferred_stypes[cfg.dataset_name]
    # graph, col_stats_dict = make_pkey_fkey_graph(
    #     dataset.db,
    #     col_to_stype_dict=col_to_stype_dict,
    #     text_embedder_cfg=TextEmbedderConfig(
    #         text_embedder=GloveTextEmbedding(device=cfg.device), batch_size=256
    #     ),
    #     cache_dir=os.path.join(cfg.partition_dir, f"materialized_cache/{rank}"),
    # )
    
    # dir = '/home/ubuntu/retexo-distributed/outputs/2024-04-19/07-42-14/checkpoint/model.pt'
    # state = torch.load(dir)
    # model = RelModel(
    #     data=graph,
    #     col_stats_dict=col_stats_dict,
    #     num_layers=cfg.num_layers,
    #     channels=cfg.channels,
    #     out_channels=cfg.out_channels,
    #     aggr=cfg.model.conv_layer.aggregator_type,
    #     norm="batch_norm",
    # ).to(cfg.device)
    # model.load_state_dict(state)
    
    # loader_dict: Dict[str, NeighborLoader] = {}
    # for split, table in [
    #     ("train", task.train_table),
    #     ("val", task.val_table),
    #     ("test", task.test_table),
    # ]:
    #     table_input = get_node_train_table_input(table=table, task=task)

    #     entity_table = table_input.nodes[0]
    #     loader_dict[split] = list(NeighborLoader(
    #         graph,
    #         num_neighbors=[-1], # one layer batch with all neighbors
    #         time_attr="time",
    #         input_nodes=table_input.nodes,
    #         input_time=table_input.time,
    #         transform=table_input.transform,
    #         batch_size=graph.num_nodes, # whole graph
    #         temporal_strategy=cfg.temporal_strategy,
    #         shuffle=split == "train",
    #         persistent_workers=False,
    #     ))[0]
        
    # # test(batch: HeteroData, x_dict, model, device, clamp_min, clamp_max, task)
        
    # clamp_min, clamp_max = np.percentile(
    #     task.train_table.df[task.target_col].to_numpy(), [2, 98]
    # )
    # val_pred, useless = test(loader_dict["val"], None, model, cfg.device, clamp_min, clamp_max, task)
    # val_metrics = task.evaluate(val_pred, task.val_table)
    # print(f"Best Val metrics: {val_metrics}")

    # test_pred, useless = test(loader_dict["test"], None, model, cfg.device, clamp_min, clamp_max, task)
    # test_metrics = task.evaluate(test_pred)
    # print(f"Best test metrics: {test_metrics}")
    
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
