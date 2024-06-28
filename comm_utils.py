"""Implement communication utilities for GNN models"""

from typing import Dict, List
from multiprocessing.pool import ThreadPool
import time
import dgl # type: ignore
from dgl.distributed import GraphPartitionBook # type: ignore
from relbench.external.graph import get_node_train_table_input, make_pkey_fkey_graph, NodeTrainTableInput
from torch_geometric.distributed.local_feature_store import LocalFeatureStore
from torch_geometric.distributed.local_graph_store import LocalGraphStore
from torch_geometric.data import HeteroData
import os

import torch.distributed as dist
import torch
import torch.nn as nn
import numpy as np

from torch_frame import stype


from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend



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

def send_and_receive_embeddings_pyg(
    graph: HeteroData, boundary_node_lists: List[torch.Tensor], feats_dict: Dict, local_dict: Dict, available: List[torch.Tensor]
):
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    # send_reqs = []
    for part_id in range(world_size):
        if part_id == rank:
            continue
        
        for node_type in graph.node_types:    
            # belong_right = local_dict[node_type].iloc[relevant_indices]["part_id"] == right
            
            relevant_indices = graph[node_type]["n_id"].to("cpu")
            boundary_node_t = boundary_node_lists[node_type][part_id]
            # local_mask = torch.isin(torch.tensor(local_dict[node_type].iloc[relevant_indices]["GlobalId"].to_numpy()), boundary_node_t)
            # boundary_mask = torch.isin(boundary_node_t, torch.tensor(local_dict[node_type].iloc[relevant_indices]["GlobalId"].to_numpy()))
            local_mask = torch.isin(torch.tensor(local_dict[node_type].iloc[relevant_indices]["GlobalId"].to_numpy()), boundary_node_t)
            # local_mask = torch.isin(torch.tensor(local_dict[node_type]["GlobalId"].to_numpy()), boundary_node_t)
            embeddings =  feats_dict[node_type][local_mask[:graph[node_type]["tf"].num_rows]].to("cpu")

            # features_corr =  graph[node_type]["tf"][local_mask[:graph[node_type]["tf"].num_rows]]
            
            # ignoring node types with no embeddings:
            # if stype.embedding not in features_corr.feat_dict:
            #     continue
            
            # embeddings = features_corr.feat_dict[stype.embedding].values
            # can use features_corr.embeddings.values
            # embeddings = torch.ones(
            #     1, device="cuda"
            # )
            send_req = dist.isend(embeddings, dst=part_id)
            
            # recv_mask = torch.isin(torch.tensor(local_dict[node_type]["part_id"].to_numpy()), part_id)
            recv_ids = available[node_type][part_id]
            new_features = torch.zeros(
                recv_ids.shape[0],
                feats_dict[node_type].shape[1], 
                device=embeddings.device
                # dtype=torch.long
            )
            
            # new_features = torch.zeros(1, device="cuda")
            dist.recv(new_features, src=part_id)
            send_req.wait()
            
            recv_mask = torch.isin(torch.tensor(local_dict[node_type].iloc[relevant_indices]["GlobalId"].to_numpy()), recv_ids)
            
            # TODO is it enough to just update "values" ??? (what about other attributes of embeddings?)
            # feats_dic[node_type]["tf"][recv_mask].feat_dict[stype.embedding].values = new_features
            feats_dict[node_type][recv_mask[:graph[node_type]["tf"].num_rows]] = new_features.to("cuda")
    
def get_boundary_nodes(node_info_dict: Dict, gpb: GraphPartitionBook):
    """
    Get the boundary nodes
    
    Here each worker sends the number of boundary nodes followed by the boundary
    node to every other worker. 
    Similarly, it receives those information from every other worker.
    """

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


def get_boundary_nodes_pyg(graph: HeteroData, task, local_dict: Dict):
    """Get the boundary nodes"""

    rank, size = dist.get_rank(), dist.get_world_size()
    device = "cuda"
    
    num_nodes = {}
    available = {}
    boundary = {}
    for node_type in graph.node_types:
        boundary[node_type] = [None] * size 
        num_nodes[node_type] = [None] * size 
        available[node_type] = [None] * size
        
    num_task_nodes = { 
            "train": [None] * size, 
            "val": [None] * size, 
            "test": [None] * size
            }

    for i in range(1, size):
        left = (rank - i + size) % size
        right = (rank + i) % size
        
        for node_type in graph.node_types:
            # only take those in graph
            relevant_indices = graph[node_type]["n_id"].to("cpu").tolist()
            
            belong_right = local_dict[node_type].iloc[relevant_indices]["part_id"] == right
            ids = torch.tensor(local_dict[node_type].iloc[relevant_indices]["GlobalId"][belong_right].to_numpy())
            
            num_right = torch.tensor(belong_right.to_numpy()).sum().view(-1)
            if dist.get_backend() == "gloo":
                num_right = num_right.cpu()
                num_left = torch.tensor([0])
            else:  
                num_left = torch.tensor([0], device=device)
            req = dist.isend(num_right, dst=right)
            dist.recv(num_left, src=left)

            v = ids
            if dist.get_backend() == "gloo":
                v = v.cpu()
                u = torch.zeros(num_left, dtype=torch.long) # type: ignore

            req.wait()
            req = dist.isend(v, dst=right)
            dist.recv(u, src=left)
            
            if dist.get_backend() == "gloo":
                boundary[node_type][left] = u
            # req.wait()
            
            # Reply the available boundary nodes
            
            # mask u with only the available nodes
            relevant_indices = graph[node_type]["n_id"].to("cpu")
            masked_u = torch.isin(u, torch.tensor(local_dict[node_type].iloc[relevant_indices]["GlobalId"].to_numpy()))
            
            # local_mask = torch.isin(torch.tensor(local_dict[node_type]["GlobalId"][graph[node_type]["n_id"].tolist()].to_numpy()).unique(), u)
            
            if dist.get_backend() == "gloo":
                masked_u = masked_u.cpu()
                masked_v = torch.zeros(num_right, dtype=torch.bool) # type: ignore

            req.wait()
            req = dist.isend(masked_u, dst=left)
            dist.recv(masked_v, src=right)
            
            # reduce u to only the available nodes
            available_ids = v[masked_v]
            
            if dist.get_backend() == "gloo":
                available[node_type][right] = available_ids
            
        
            # Exchange num of nodes:
            
            num_nodes_local = torch.tensor(graph[node_type].num_nodes).cpu()
            num_nodes_dist =torch.zeros(1, dtype=torch.long)
            
            num_nodes[node_type][rank] = num_nodes_local
            
            req.wait()
            req = dist.isend(num_nodes_local, dst=right)
            dist.recv(num_nodes_dist, src=left)
            
            num_nodes[node_type][left] = num_nodes_dist
        
        # Exchange num of task nodes:
        
        for split, table in [
            ("train", task.train_table),
            ("val", task.val_table),
            ("test", task.test_table),
        ]:
            
            num_task_nodes_local = torch.tensor(table.df.shape[0]).cpu()
            num_task_nodes_dist =torch.zeros(1, dtype=torch.long)
            
            num_task_nodes[split][rank] = num_task_nodes_local
            
            req.wait()
            req = dist.isend(num_task_nodes_local, dst=right)
            dist.recv(num_task_nodes_dist, src=left)
            
            num_task_nodes[split][left] = num_task_nodes_dist
            
        req.wait()
            
    all_num_nodes = {}
    all_total_sum = 0
    all_local_sum = 0
    for type in graph.node_types:
        
        sum = 0

        for i in range(size):
            sum += num_nodes[type][i].item()
            all_total_sum += num_nodes[type][i].item()
  
        all_local_sum += num_nodes[type][rank].item()
        
        all_num_nodes[type] = {"total": sum, "local": num_nodes[type][rank].item()}
    all_num_nodes["all"] = {"total": all_total_sum, "local": all_local_sum} 
    
    train_sum = 0
    val_sum = 0
    test_sum = 0
    for i in range(size):
        train_sum += num_task_nodes["train"][i].item()
        val_sum += num_task_nodes["val"][i].item()
        test_sum += num_task_nodes["test"][i].item()
        
    all_task_nodes = {"train": train_sum, "val": val_sum, "test": test_sum}
    

    return boundary, all_num_nodes, all_task_nodes, available

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

    def __init__(self, model, sleep_time=0.1, pubkey=None, perf_store=None, measure_comm=False):
        self.model = model
        self._handles = []
        self._stream = None
        self._group = {}
        self.thread_pool = None
        self.sleep_time = sleep_time
        cnt = 0
        # for _, (name, param) in enumerate(self.model.named_parameters()):
        #     cnt+=1
        #     self._group[name] = dist.new_group()
        # self.thread_pool = ThreadPool(processes=cnt)
        self._stream = torch.cuda.Stream(device='cuda')
        self.measure_comm = measure_comm
        self.comm_vol_store = perf_store
        self.pubkey = pubkey
        
        # TODO (for local tests)
        with open("private_key.pem", "rb") as private_file:
            self.private_key = serialization.load_pem_private_key(
                private_file.read(),
                password=None,
                backend=default_backend()
            )

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


    def _reduce_encrypted(self, rank, world_size, model: nn.Module, shapes, num_elements, name, encrypted_grads_iv_key):
        def create_stream(shapes, num_elements, name, encrypted_grads_iv_key):
            self._stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self._stream):
                group = None#self._group[name]
                time.sleep(2 * self.sleep_time)
                encrypted_list = None
                if rank == 0:
                    encrypted_list = []
                    for _ in range(world_size):
                        # Placeholder tensor for gathering encrypted gradients
                        encrypted_list.append(torch.zeros_like(encrypted_grads_iv_key))

                # Gather encrypted gradients, IV, and keys
                dist.gather(encrypted_grads_iv_key, encrypted_list, dst=0, group=group)
                
                # Decrypt and aggregate the gradients
                if rank == 0:
                    decrypted_list = []
                    for encrypted in encrypted_list:
                        encrypted_grads_iv_key = encrypted.numpy()
                        iv = encrypted_grads_iv_key[-34:-32].tobytes()
                        encrypted_key = encrypted_grads_iv_key[-32:].tobytes()
                        encrypted_grads = encrypted_grads_iv_key[:-34] 
                        
                        symmetric_key = self.private_key.decrypt(
                            encrypted_key,
                            padding.OAEP(
                                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                                algorithm=hashes.SHA256(),
                                label=None
                            )
                        )
                        # Decrypt gradients
                        cipher = Cipher(algorithms.AES(symmetric_key), modes.CFB(iv), backend=default_backend())
                        decryptor = cipher.decryptor()
                        decrypted_bytes = decryptor.update(encrypted_grads.tobytes()) + decryptor.finalize()
                        decrypted_array = np.frombuffer(decrypted_bytes, dtype=np.float32)
                        decrypted_list.append(decrypted_array)

                    # Aggregate decrypted gradients
                    aggregated_grads = np.sum(decrypted_list, axis=0)

                    # Convert aggregated gradients back to tensor and set to param.grad
                    start = 0
                    i = 0
                    for param in model.parameters():
                        if param.grad is None:
                            continue
                        # num_elem = param.grad.numel()
                        grad_array = aggregated_grads[start:start + num_elements[i]].reshape(shapes[i])
                        param.grad.copy_(torch.tensor(grad_array).view_as(param.grad))
                        start += num_elements[i]
                        i+=1

                # Broadcast the aggregated gradients from rank 0 to all workers
                for _, param in enumerate(model.parameters()):
                    if param.grad is not None:
                        dist.broadcast(param.grad, src=0, group=group)

                if self.measure_comm:
                    # Add communication volume for gradient reduction
                    if rank == 0:
                        cv = 2 * get_comm_size_param(param.grad) * (world_size - 1)
                    else:
                        cv = 2 * get_comm_size_param(param.grad)
                    self.comm_vol_store.add_cv_grad_reduce_t(cv)

        self._handles.append(self.thread_pool.apply_async(create_stream, (shapes, num_elements, name, encrypted_grads_iv_key)))

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
        
    def aggregate_hetero_grad(self, model: nn.Module, node_types, all_num_nodes):
        """Aggregate the model across workers using thread pool"""
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        for _, (name, param) in enumerate(model.named_parameters()):
            if param.grad is None:
                continue
            type = extract_node_type(name, node_types)
            param.grad = param.grad * (all_num_nodes[type]["local"] / all_num_nodes[type]["total"])
            self._reduce(rank, world_size, param, name)
        for handle in self._handles:
            handle.wait()
        self._handles.clear()
        torch.cuda.current_stream().wait_stream(self._stream)
        
    def secure_aggregation(self, model: nn.Module, node_types, round):
        """Aggregate the model across workers using encrypted gradients"""
        # TODO: Implement secure aggregations
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        
        # create symmetric key for encryption
        symmetric_key = os.urandom(32)
        iv = os.urandom(16)
        cipher = Cipher(algorithms.AES(symmetric_key), modes.CFB(iv), backend=default_backend())
        encryptor = cipher.encryptor()
        
        all_grads = []
        shapes = []
        num_elements = []
        indexes = []
        
        for i, (name, param) in enumerate(model.named_parameters()):
            if param.grad is None:
                continue
            type = extract_node_type(name, node_types)
            param.grad = param.grad * 1. # TODO scale correctly#(all_num_nodes[type]["local"] / all_num_nodes[type]["total"])
            
            all_grads.append(param.grad.cpu().numpy())
            shapes.append(param.grad.shape)
            num_elements.append(param.grad.numel())
            indexes.append(i)
            
        all_grads_array = np.concatenate([grad.flatten() for grad in all_grads])
        encrypted_grads = encryptor.update(all_grads_array.tobytes()) + encryptor.finalize()

        encrypted_key = self.pubkey.encrypt(
            symmetric_key,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        # here reduce the encrypted gradients
        num_params = len(indexes)
        metadata = np.array([num_params] + indexes, dtype=np.int32)
        metadata_bytes = metadata.tobytes()
        encrypted_grads_iv_key = torch.tensor(np.frombuffer(metadata_bytes + encrypted_grads + iv + encrypted_key, dtype=np.float32))
        dist.send(encrypted_grads_iv_key, dst=0, tag=round)
        
        for param in model.parameters():
            if param.grad is not None:
                dist.broadcast(param.grad, src=0)
        # self._reduce_encrypted(rank, world_size, model, shapes, num_elements, 'encrypted_grads', encrypted_grads_iv_key)
        
        # for handle in self._handles:
        #     handle.wait()
        # self._handles.clear()
        # torch.cuda.current_stream().wait_stream(self._stream)
        
    def master_aggregate_gradients(self, cfg, layer):
        """Aggregate the gradients on the master node"""
        world_size = cfg.num_partitions + 1
        
        all_grads = []
        num_elements = []
        shapes = []
        
        dummy_iv = os.urandom(16)
        dummy_key = os.urandom(32)
    
        # with torch.no_grad():
        for _, (name, param) in enumerate(self.model.named_parameters()):
            all_grads.append(torch.zeros_like(param).cpu().numpy())
            shapes.append(param.shape)
            num_elements.append(param.numel())
    
        all_grads_array = np.concatenate([grad.flatten() for grad in all_grads])
        target_tensor = torch.tensor(np.frombuffer(all_grads_array.tobytes() + dummy_iv + dummy_key, dtype=np.float32))
        
        for round in range(cfg.num_rounds[layer]):
            encrypted_list = []
            for _ in range(1, world_size):  # Exclude master itself
                # Placeholder tensor for gathering encrypted gradients
                encrypted_grads_iv_key = torch.zeros(target_tensor.shape[0], dtype=torch.float32)  
                
                # Adjust the size as needed
                dist.recv(encrypted_grads_iv_key, src=_, tag=round)
                encrypted_list.append(encrypted_grads_iv_key.numpy())
              
            decrypted_list = []
            for encrypted in encrypted_list:
                # Extract metadata
                num_params = int.from_bytes(encrypted[:1], 'little')
                metadata_size = 1 + num_params# * 4
                indexes = np.frombuffer(encrypted[1:metadata_size], dtype=np.int32)
                
                gradients_size = 0
                for i, (name, param) in enumerate(self.model.named_parameters()):
                    if i in indexes:
                        gradients_size += param.numel()
                
                offset = metadata_size + gradients_size
                iv = encrypted[offset:offset+4].tobytes()
                encrypted_key = encrypted[offset+4:offset+68].tobytes()
                encrypted_grads = encrypted[metadata_size:offset]

                symmetric_key = self.private_key.decrypt(
                    encrypted_key,
                    padding.OAEP(
                        mgf=padding.MGF1(algorithm=hashes.SHA256()),
                        algorithm=hashes.SHA256(),
                        label=None
                    )
                )
                
                # Decrypt gradients
                cipher = Cipher(algorithms.AES(symmetric_key), modes.CFB(iv), backend=default_backend())
                decryptor = cipher.decryptor()
                decrypted_bytes = decryptor.update(encrypted_grads.tobytes()) + decryptor.finalize()
                decrypted_array = np.frombuffer(decrypted_bytes, dtype=np.float32)
                decrypted_list.append(decrypted_array)

            # Aggregate decrypted gradients
            aggregated_grads = np.sum(decrypted_list, axis=0)

            # Convert aggregated gradients back to tensor and set to param.grad
            start = 0
            # i = 0
            index = 0
            for param in self.model.parameters():
                if index in indexes:
                    grad_array = aggregated_grads[start:start + param.numel()].reshape(param.shape)
                    param.grad = torch.tensor(grad_array).view(param.shape)
                    start += param.numel()
                    # i += 1
                index+=1


            # Broadcast the aggregated gradients from rank 0 to all workers
            for (name, param) in self.model.named_parameters():
                if param.grad is not None:
                    dist.broadcast(param.grad, src=0)

def extract_node_type(name, node_types):
    tokens = name.split(".")
    for type in node_types:
        if type in tokens:
            return type
    return "all"

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
        t = torch.tensor(v).cuda()
        dist.all_reduce(t, op=dist.ReduceOp.SUM, async_op=False)
        metrics[k] = t.item() 

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
