"""Implement communication utilities for GNN models"""

from typing import Dict, List
from multiprocessing.pool import ThreadPool
import time
import dgl # type: ignore
from dgl.distributed import GraphPartitionBook # type: ignore
import torch.distributed as dist
import torch
import torch.nn as nn


def send_and_receive_embeddings(
    boundary_node_lists: List[torch.Tensor], node_info_dict: Dict, layer_tag: str, perf_store=None, measure_comm=False
):
    """Send and receive embeddings to nodes

    Parameters
    ----------
    boundary_node_lists : List[torch.Tensor]
        Boundary node lists for each partition (worker)
    node_info_dict : Dict
        Node information dictionary containing the corresponding embeddings
    layer_tag : str
        for instance "feat_0", "feat_1", "feat_2", etc. that are keys in node_info_dict

    """

    world_size = dist.get_world_size()
    rank = dist.get_rank()

    # send_reqs = []
    for part_id in range(world_size):
        if part_id == rank:
            continue

        boundary_node_t = boundary_node_lists[part_id]
        features_corr = node_info_dict[layer_tag][boundary_node_t]
        send_req = dist.isend(features_corr, dst=part_id)

        recv_boundary_node_indices = torch.arange(len(node_info_dict["part_id"]))[
            node_info_dict["part_id"] == part_id
        ]
        new_features = torch.zeros(
            len(recv_boundary_node_indices),
            node_info_dict[layer_tag].shape[1],
            device=node_info_dict[layer_tag].device,
        )
        dist.recv(new_features, src=part_id)
        send_req.wait()
        if measure_comm:
            # add cv for send and recv
            cv = get_comm_size_param(features_corr)
            perf_store.add_cv_message_passing_t(cv)
            cv = get_comm_size_param(new_features)
            perf_store.add_cv_message_passing_t(cv)
        node_info_dict[layer_tag][recv_boundary_node_indices] = new_features


def get_boundary_nodes(node_info_dict: Dict, gpb: GraphPartitionBook):
    """Get the boundary nodes"""

    rank, size = dist.get_rank(), dist.get_world_size()
    device = "cuda"
    boundary = [None] * size

    for i in range(1, size):
        left = (rank - i + size) % size
        right = (rank + i) % size
        belong_right = node_info_dict["part_id"] == right
        num_right = belong_right.sum().view(-1)
        if dist.get_backend() == "gloo":
            num_right = num_right.cpu()
            num_left = torch.tensor([0])
        else:
            num_left = torch.tensor([0], device=device)
        req = dist.isend(num_right, dst=right)
        dist.recv(num_left, src=left)
        start = gpb.partid2nids(right)[0].item()
        v = node_info_dict[dgl.NID][belong_right] - start
        if dist.get_backend() == "gloo":
            v = v.cpu()
            u = torch.zeros(num_left, dtype=torch.long) # type: ignore

        req.wait()
        req = dist.isend(v, dst=right)
        dist.recv(u, src=left)
        # u, _ = torch.sort(u)

        if dist.get_backend() == "gloo":
            boundary[left] = u
        req.wait()

    return boundary

def aggregate_model(model: nn.Module):
    """Aggregate the model across workers

    Parameters
    ----------
    model : nn.Module
        Model to aggregate

    Returns
    -------
    nn.Module
        Aggregated model
    """

    world_size = dist.get_world_size()

    for param in model.parameters():
        dist.all_reduce(param.data, op=dist.ReduceOp.SUM)
        param.data /= world_size

    return model

class MultiThreadReducer:
    """Multi-threaded reducer for aggregating gradients"""

    def __init__(self, model, sleep_time=0.1):
        self.model = model
        self._handles = []
        self._stream = None
        self._group = {}
        self.thread_pool = None
        self.sleep_time = sleep_time
        cnt = 0
        for _, (name, param) in enumerate(self.model.named_parameters()):
            cnt+=1
            self._group[name] = dist.new_group()
        self.thread_pool = ThreadPool(processes=cnt)
        self._stream = torch.cuda.Stream(device='cuda')

    def _reduce(self, param, name):
        def create_stream():
            self._stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self._stream):
                group = self._group[name]
                time.sleep(self.sleep_time)
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, group=group)
        self._handles.append(self.thread_pool.apply_async(create_stream))

    def aggregate_grad(self, model: nn.Module, num_local_train, num_train):
        """Aggregate the model across workers using thread pool"""
        for _, (name, param) in enumerate(model.named_parameters()):
            param.grad = param.grad * (num_local_train / num_train)
            self._reduce(param, name)
        for handle in self._handles:
            handle.wait()
        self._handles.clear()
        torch.cuda.current_stream().wait_stream(self._stream)

class MultiThreadReducerCentralized:
    """Multi-threaded reducer for aggregating gradients in a centralized manner"""

    def __init__(self, model, sleep_time=0.1, perf_store=None, measure_comm=False):
        self.model = model
        self._handles = []
        self._stream = None
        self._group = {}
        self.thread_pool = None
        self.sleep_time = sleep_time
        cnt = 0
        for _, (name, param) in enumerate(self.model.named_parameters()):
            cnt+=1
            self._group[name] = dist.new_group()
        self.thread_pool = ThreadPool(processes=cnt)
        self._stream = torch.cuda.Stream(device='cuda')
        self.measure_comm = measure_comm
        self.comm_vol_store = perf_store

    def _reduce(self, rank, world_size, param, name):
        def create_stream():
            self._stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self._stream):
                group = self._group[name]
                time.sleep(2*self.sleep_time)
                param_list = None
                if rank == 0:
                    param_list = []
                    for _ in range(world_size):
                        param_list.append(torch.zeros_like(param.grad))
                dist.gather(param.grad, param_list, dst=0, group=group)
                # aggregate the gradients
                if rank == 0:
                    param.grad = torch.sum(torch.stack(param_list), dim=0)
                # broadcast the aggregated gradients from rank 0 to all worker
                dist.broadcast(param.grad, src=0, group=group)
                if self.measure_comm:
                    # add cv for grad reduce
                    if rank == 0:
                        cv = 2 * get_comm_size_param(param.grad) * (world_size - 1)
                    else:
                        cv = 2 * get_comm_size_param(param.grad)
                    self.comm_vol_store.add_cv_grad_reduce_t(cv)
        self._handles.append(self.thread_pool.apply_async(create_stream))

    def aggregate_grad(self, model: nn.Module, num_local_train, num_train):
        """Aggregate the model across workers using thread pool"""
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        for _, (name, param) in enumerate(model.named_parameters()):
            param.grad = param.grad * (num_local_train / num_train)
            self._reduce(rank, world_size, param, name)
        for handle in self._handles:
            handle.wait()
        self._handles.clear()
        torch.cuda.current_stream().wait_stream(self._stream)

def aggregate_metrics(metrics: Dict):
    """Aggregate the metrics across workers

    Parameters
    ----------
    metrics : Dict
        Metrics to aggregate

    Returns
    -------
    Dict
        Aggregated metrics
    """

    for k, v in metrics.items():
        dist.all_reduce(v, op=dist.ReduceOp.SUM)

    return metrics

def sync_model(model: nn.Module):
    """Sync the model across workers

    Parameters
    ----------
    model : nn.Module
        Model to sync

    Returns
    -------
    nn.Module
        Synced model
    """

    for param in model.parameters():
        dist.broadcast(param.data, src=0)

def get_comm_size_param(param):
    """Get the communication size of a parameter

    Parameters
    ----------
    param : torch.Tensor
        Parameter

    Returns
    -------
    int
        Communication size of the parameter
    """

    return param.numel() * param.element_size()
