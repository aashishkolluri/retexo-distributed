"""Implement trainer that oversees end-to-end training process"""

import pickle
import os
import logging
import time
import threading
import copy
from typing import Any, Dict, List, Tuple
import dgl # type: ignore
import torch
from torch.optim import lr_scheduler
import torch.distributed as dist
from torch import Tensor
from dgl.distributed import GraphPartitionBook # type: ignore
from hydra.utils import instantiate
from omegaconf import DictConfig

from performance import PerformanceStore
from data.dataset import load_partition
from trainers.worker_trainer import WorkerTrain

from comm_utils import (
    get_boundary_nodes,
    send_and_receive_embeddings,
    aggregate_metrics,
    sync_model,
    MultiThreadReducerCentralized,
)

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


def get_in_out_graph(
    graph: dgl.DGLGraph, node_dict: Dict
) -> Tuple[dgl.DGLGraph, dgl.DGLGraph]:
    """Get the in and out graph"""
    in_graph = dgl.node_subgraph(graph, node_dict["inner_node"].bool())
    in_graph.ndata.clear()
    in_graph.edata.clear()

    out_graph = graph.clone()
    out_graph.ndata.clear()
    out_graph.edata.clear()
    in_nodes = torch.arange(in_graph.num_nodes())
    out_graph.remove_edges(out_graph.out_edges(in_nodes, form="eid"))
    return in_graph, out_graph


def setup_model(entire_model, curr_layer, device):
    """Set up the model"""
    # train the nth layer of the model
    curr_model = entire_model.get_nth_layer(curr_layer)
    if device == "cuda":
        curr_model = curr_model.cuda()
    return curr_model


def get_scheduler(num_rounds, optimizer):
    """Get the scheduler for learning rate"""

    def lr_lambda(current_step: int):
        if current_step < 0:
            return float(current_step) / float(max(1, 0))
        return max(
            0.0,
            float(num_rounds - current_step) / float(max(1, num_rounds - 0)),
        )

    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return scheduler


def train_for_one_layer(
    curr_model,
    traindata,
    valdata,
    testdata,
    optimizer,
    metrics,
    loss_function,
    cfg,
    worker_trainer,
    num_tr_val_te,
    num_local_tr_val_te,
    num_rounds,
    perf_store,
    results_dir,
):
    """Training loop for one layer"""
    num_train, num_val, num_test = num_tr_val_te
    num_local_train, num_local_val, num_local_test = num_local_tr_val_te
    if cfg.device == "cuda":
        traindata = [data.to(torch.device("cuda")) for data in traindata]
        valdata = [data.to(torch.device("cuda")) for data in valdata]
        testdata = [data.to(torch.device("cuda")) for data in testdata]

    rank = dist.get_rank()

    if rank == 0:
        best_params = curr_model.state_dict()
        best_val_loss = float("inf")
        best_val_acc = 0.0
        best_test_acc = 0.0
        best_val_metrics = None
        best_epoch = 0

    reducer = MultiThreadReducerCentralized(
        curr_model, cfg.sleep_time, comm_volume_perf_store, cfg.measure_dv
    )

    # start training
    for training_round in range(num_rounds):
        # start the round
        start_time = time.time()
        # train the model
        worker_trainer.train_model(
            curr_model,
            traindata,
            optimizer,
            loss_function,
            cfg.local_epochs,
        )
        time_round = time.time() - start_time
        perf_store.add_local_train_time(time_round)

        agg_time_s = time.time()
        with torch.no_grad():
            reducer.aggregate_grad(curr_model, num_local_train, num_train)
        agg_time = time.time() - agg_time_s
        perf_store.add_grad_reduce_time(agg_time)

        optimizer.step()

        if cfg.best_val_model:
            val_metrics = worker_trainer.evaluate(
                curr_model, valdata, metrics, loss_function
            )
            # sync the val metrics
            for k, v in val_metrics.items():
                val_metrics[k] = v * (num_local_val / num_val)
            aggregate_metrics(val_metrics)

            if rank == 0:
                # store the model with best val accuracy
                if val_metrics["accuracy"] > best_val_acc:
                    best_val_loss = val_metrics["loss"]
                    best_val_metrics = val_metrics
                    best_val_acc = val_metrics["accuracy"]
                    best_params = copy.deepcopy(curr_model.state_dict())
                    best_epoch = training_round

        if (
            training_round + 1
        ) % cfg.log_every == 0 or training_round == num_rounds - 1:
            # evaluate the model
            train_metrics = worker_trainer.evaluate(
                curr_model, traindata, metrics, loss_function
            )
            val_metrics = worker_trainer.evaluate(
                curr_model, valdata, metrics, loss_function
            )
            test_metrics = worker_trainer.evaluate(
                curr_model, testdata, metrics, loss_function
            )
            # aggregate the test metrics
            for k, v in test_metrics.items():
                test_metrics[k] = v * (num_local_test / num_test)
            aggregate_metrics(test_metrics)

            if rank == 0:
                if test_metrics["accuracy"] > best_test_acc:
                    best_test_acc = test_metrics["accuracy"]

            print(
                f"Rank {rank:2} | Training Round {training_round:2} | compute {time_round:2.4f} s | reduce {agg_time:2.4f} s"
            )

            with open(results_dir + "accuracy.txt", "a+") as f:
                f.write(
                    f'Epoch {training_round}, {val_metrics["accuracy"].item()}, {test_metrics["accuracy"].item()}\n'
                )
            # log the metrics
            print_str = f"Rank {rank:2} | Train"
            for k, v in train_metrics.items():
                print_str += f" | {k}: {v:2.4f}"
            print(print_str)
            print_str = f"Rank {rank:2} | Val  "
            for k, v in val_metrics.items():
                print_str += f" | {k}: {v:2.4f}"
            print(print_str)
            print_str = f"Rank {rank:2} | Test "
            for k, v in test_metrics.items():
                print_str += f" | {k}: {v:2.4f}"
            print(print_str)

    if cfg.best_val_model:
        if rank == 0:
            curr_model.load_state_dict(best_params)
        sync_model(curr_model)

        test_metrics = worker_trainer.evaluate(
            curr_model, testdata, metrics, loss_function
        )
        for k, v in test_metrics.items():
            test_metrics[k] = v * (num_local_test / num_test)
        aggregate_metrics(test_metrics)

        if rank == 0:
            print(f"Best model at epoch {best_epoch} | loss {best_val_loss:5.4f}")
            print(f"Best model val metrics: {best_val_metrics}")
            print(f"Best model test metrics: {test_metrics}")
            print(f"Best test accuracy achieved: {best_test_acc}")
            print("-------------------------------------------" * 3)
            with open(results_dir + "best_stats.txt", "a+") as f:
                f.write(
                    f'Epoch {best_epoch}, {best_val_metrics["accuracy"].item()}, {test_metrics["accuracy"].item()}, {best_test_acc}\n'
                )

    time.sleep(5) # for printing purposes

def construct_graph_and_features_to_compute_next_embedding(
    emb_data_dict, task, graph, node_dict, prev_feat_tag, inner_node_indices
):
    """Construct the graph and features for the next layer parallel to training"""
    prev_feats = node_dict[prev_feat_tag]
    graph.ndata["feat"] = prev_feats
    emb_data = task.construct_graph_and_features(graph, inner_node_indices)
    emb_data_dict["emb_data"] = emb_data

def train(
    graph: dgl.DGLGraph,
    node_dict: Dict,
    num_feat: int,
    num_classes: int,
    num_tr_val_te: Tuple[int, int, int],
    graph_partition_book: GraphPartitionBook,
    cfg: DictConfig,
    hydra_output_dir: str,
    results_dir: str,
) -> None:
    """Implement end-to-end training process"""

    # set the seed
    set_torch_seed(cfg.seed)

    # set up the model
    model = instantiate(cfg.model, input_dim=num_feat, output_dim=num_classes)
    device = cfg.device
    if device == "cuda":
        model = model.cuda()
    perf_stores = [PerformanceStore()]
    perf_store = perf_stores[0]

    rank, _ = dist.get_rank(), dist.get_world_size()

    os.makedirs(os.path.join(hydra_output_dir, "results"), exist_ok=True)
    if rank == 0:
        os.makedirs(os.path.join(hydra_output_dir, "checkpoint"), exist_ok=True)

    # get the in and out graph
    in_graph, _ = get_in_out_graph(graph, node_dict)

    logger.info(
        "Process %d has %d nodes and %d edges, with fractions %f inner nodes and %f inner edges",
        rank,
        graph.num_nodes(),
        graph.num_edges(),
        in_graph.num_nodes() / graph.num_nodes(),
        in_graph.num_edges() / graph.num_edges(),
    )

    # get the boundary node lists
    boundary_nodes = get_boundary_nodes(node_dict, graph_partition_book)

    # get the inner node indices
    inner_node_indices = torch.arange(node_dict["inner_node"].int().sum())

    # get the number of local samples for train val and test
    num_local_train = node_dict["train_mask"].int().sum()
    num_local_val = node_dict["val_mask"].int().sum()
    num_local_test = node_dict["test_mask"].int().sum()

    # get the dataloaders and model for the first layer
    # create zeroth embedding for all nodes
    node_dict["feat_0"] = torch.zeros((len(node_dict[dgl.NID]), num_feat))
    node_dict["feat_0"][inner_node_indices] = node_dict["feat"]

    # send and receive the zeroth embedding of all nodes to their graph neighbors
    emb_share_time_s = time.time()
    send_and_receive_embeddings(
        boundary_nodes, node_dict, "feat_0", comm_volume_perf_store, cfg.measure_dv
    )
    perf_store.set_emb_broadcast_time(time.time() - emb_share_time_s)

    # set up the task
    task = instantiate(cfg.task)
    # get the induced train, val, and test subgraphs on the training nodes
    traindata, valdata, testdata = task.get_tr_val_te_data(
        graph, node_dict, "feat_0"
    )
    # set up the model for the first layer
    curr_model = setup_model(model, 0, cfg.device)
    sync_model(curr_model)

    if isinstance(cfg.learning_rate, float):
        cfg.learning_rate = [cfg.learning_rate]
    # set up the optimizer
    optimizer = instantiate(
        cfg.optimizer, lr=cfg.learning_rate[0], params=curr_model.parameters()
    )
    # set up the loss function
    loss_function = task.get_loss_function()
    # set up the metrics
    metrics = task.get_evaluation_metrics()
    # set up the worker trainer
    worker_trainer = WorkerTrain()
    if isinstance(cfg.num_rounds, int):
        cfg.num_rounds = [cfg.num_rounds]

    # get the data ready to compute embeddings for next layer in parallel
    prev_feat_tag = "feat_0"
    emb_data_dict: Dict[str, Tuple[Any, Tensor, Tensor]] = {}
    emb_data_thread = threading.Thread(
        target=construct_graph_and_features_to_compute_next_embedding,
        args=(emb_data_dict, task, graph, node_dict, prev_feat_tag, inner_node_indices),
    )
    emb_data_thread.start()

    # train the first layer
    os.makedirs(results_dir + f"layer_0/", exist_ok=True)
    train_for_one_layer(
        curr_model,
        traindata,
        valdata,
        testdata,
        optimizer,
        metrics,
        loss_function,
        cfg,
        worker_trainer,
        num_tr_val_te,
        tuple([num_local_train, num_local_val, num_local_test]),
        cfg.num_rounds[0],
        perf_store,
        results_dir + f"layer_0/",
    )

    for curr_layer in range(1, cfg.model.n_layers):
        perf_stores.append(PerformanceStore())
        perf_store = perf_stores[curr_layer]

        # get the feat_1 of the nodes
        curr_shape_feat = (
            node_dict["inner_node"].int().sum(),
            curr_model.agg_layer._out_feats,
        )
        # get the embeddings (features) to train the current layer
        compute_embs_time_s = time.time()
        emb_data_thread.join()
        emb_data = emb_data_dict["emb_data"]
        if device == "cuda":
            emb_data = [data.to(torch.device("cuda")) for data in emb_data] # type: ignore
        curr_feats = worker_trainer.get_embeddings(
            curr_model, emb_data, curr_shape_feat, hidden_layer=True
        )
        curr_feat_tag = "feat_" + str(curr_layer)
        node_dict[curr_feat_tag] = torch.zeros(
            (len(node_dict[dgl.NID]), curr_feats.shape[1])
        )
        node_dict[curr_feat_tag][inner_node_indices] = curr_feats
        perf_store.add_compute_local_embs_time(time.time() - compute_embs_time_s)

        # send and receive the embeddings of all nodes to their graph neighbors
        emb_share_time_s = time.time()
        send_and_receive_embeddings(
            boundary_nodes,
            node_dict,
            curr_feat_tag,
            comm_volume_perf_store,
            cfg.measure_dv,
        )
        perf_store.set_emb_broadcast_time(time.time() - emb_share_time_s)

        # set up the dataloaders and model for the current layer
        traindata, valdata, testdata = task.get_tr_val_te_data(
            graph, node_dict, curr_feat_tag
        )
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
        # prepare the data to compute embeddings for next layer in parallel to training
        prev_feat_tag = curr_feat_tag
        if curr_layer < cfg.model.n_layers - 1:
            emb_data_dict.clear()
            emb_data_thread = threading.Thread(
                target=construct_graph_and_features_to_compute_next_embedding,
                args=(
                    emb_data_dict,
                    task,
                    graph,
                    node_dict,
                    prev_feat_tag,
                    inner_node_indices,
                ),
            )
            emb_data_thread.start()

        # train the current layer
        os.makedirs(results_dir + f"layer_{curr_layer}/", exist_ok=True)
        train_for_one_layer(
            curr_model,
            traindata,
            valdata,
            testdata,
            optimizer,
            metrics,
            loss_function,
            cfg,
            worker_trainer,
            num_tr_val_te,
            tuple([num_local_train, num_local_val, num_local_test]),
            cfg.num_rounds[curr_layer]
            if curr_layer < len(cfg.num_rounds)
            else cfg.num_rounds[0],
            perf_store,
            results_dir + f"layer_{curr_layer}/",
        )

    per_metrics = []
    total_comp_comm_time = 0.0
    total_emb_broadcast_time = 0.0
    avg_comp_comm_time = 0.0
    for i, perf_store in enumerate(perf_stores):
        per_metrics.append(perf_store.get_necessary_time_metrics())
        total_comp_comm_time += per_metrics[-1]["total_local_train_time"]
        total_comp_comm_time += per_metrics[-1]["total_grad_reduce_time"]
        total_comp_comm_time += per_metrics[-1]["total_compute_local_embs_time"]
        total_comp_comm_time += per_metrics[-1]["total_emb_broadcast_time"]
        total_emb_broadcast_time += per_metrics[-1]["total_emb_broadcast_time"]
        avg_comp_comm_time += per_metrics[-1]["avg_local_train_time"]
        avg_comp_comm_time += per_metrics[-1]["avg_grad_reduce_time"]
        print(f"\nRank {rank:2} | Training time metrics for layer {i}")
        for k, v in per_metrics[-1].items():
            print(f"Rank {rank:2} | {k:2}: {v:2.4f}")
    print(
        f"\nRank {rank:2} | total_comp_comm_time: {total_comp_comm_time:2.4f} | avg_comp_comm_time: {avg_comp_comm_time:2.4f} | total_emb_broadcast_time: {total_emb_broadcast_time:2.4f}"
    )

    per_metrics.append(
        {
            "total_comp_comm_time": total_comp_comm_time,
            "avg_comp_comm_time": avg_comp_comm_time,
            "total_emb_broadcast_time": total_emb_broadcast_time,
        }
    )

    if cfg.measure_dv:
        print(f"\nRank {rank:2} | Communication volume metrics")
        cv = comm_volume_perf_store.get_communication_volume()
        cv_message_passing_t = comm_volume_perf_store.get_cv_message_passing_t()
        cv_grad_reduce_t = comm_volume_perf_store.get_cv_grad_reduce_t()
        per_metrics.append(comm_volume_perf_store.get_necessary_cv_metrics())
        print(f"Rank {rank:2} | total communication_volume: {cv:2d}")
        print(f"Rank {rank:2} | cv_message_passing_t: {cv_message_passing_t:2d}")
        print(f"Rank {rank:2} | cv_grad_reduce_t: {cv_grad_reduce_t:2d}")

    # save the performance metrics
    with open(os.path.join(hydra_output_dir, "results", "perf_metrics.pkl"), "wb") as f:
        pickle.dump(per_metrics, f)

    for i in range(len(per_metrics) - 2):
        with open(results_dir + f"layer_{i}/perf_metrics.txt", "a+") as f:
            # total_local_train_time, total_grad_reduce_time, avg_local_train_time, avg_grad_reduce_time
            f.write(
                f"{per_metrics[i]['total_local_train_time']}, {per_metrics[i]['total_grad_reduce_time']}, {per_metrics[i]['avg_local_train_time']}, {per_metrics[i]['avg_grad_reduce_time']}\n"
            )

    if cfg.measure_dv:
        with open(results_dir + "total_perf_metrics.txt", "a+") as f:
            # 'total_comp_comm_time' 'avg_comp_comm_time' 'total_cv_message_passing_t' 'total_cv_grad_reduce_t', 'total_communication_volume'
            f.write(
                f"{per_metrics[-2]['total_comp_comm_time']}, {per_metrics[-2]['avg_comp_comm_time']}, {per_metrics[-2]['total_emb_broadcast_time']}, {per_metrics[-1]['total_cv_message_passing_t']}, {per_metrics[-1]['total_cv_grad_reduce_t']}, {per_metrics[-1]['total_communication_volume']}\n"
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

    # load the partition
    (
        sub_graph,
        node_dict,
        num_feat,
        num_classes,
        num_train,
        num_val,
        num_test,
        graph_partition_book,
    ) = load_partition(
        cfg.dataset.partition.partition_dir, cfg.dataset.partition.dataset_name, rank
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
    train(
        sub_graph,
        node_dict,
        num_feat,
        num_classes,
        tuple([num_train, num_val, num_test]),
        graph_partition_book,
        cfg,
        hydra_output_dir,
        results_dir,
    )
    print(f"Rank {rank:2} | Total time taken: {time.time() - start_time:2.4f} s")

    dist.destroy_process_group()
