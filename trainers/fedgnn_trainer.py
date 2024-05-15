import time, logging, os
from typing import Any, Dict, List, Tuple

import dgl # type: ignore
import torch
from torch.nn.functional import mse_loss
from torch.optim import lr_scheduler, SGD, Adam
import torch.distributed as dist
from torch import Tensor
from torch.utils.data import DataLoader
from dgl.distributed import GraphPartitionBook # type: ignore

from models.fedgnn import FedGNNModel
from comm_utils import (
    get_boundary_nodes,
    send_and_receive_embeddings,
    aggregate_metrics,
    sync_model,
    MultiThreadReducerCentralized,
)

from hydra.utils import instantiate
from omegaconf import DictConfig
from dgl.data import MovieLensDataset

from performance import PerformanceStore

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


def setup_model(entire_model, curr_layer, device):
    """Set up the model"""
    # train the nth layer of the model
    curr_model = entire_model.get_nth_layer(curr_layer)
    if device == "cuda":
        curr_model = curr_model.cuda()
    return curr_model

def train(
    graph: dgl.DGLHeteroGraph,
    rateBuckets: Dict[str, Dict[str, Tensor]],
    cfg: DictConfig,
    hydra_output_dir: str,
    results_dir: str,
):
    
    # set the seed
    set_torch_seed(cfg.seed)
    device = cfg.device
    perf_stores = [PerformanceStore()]
    perf_store = perf_stores[0]
    
    os.makedirs(os.path.join(hydra_output_dir, "results"), exist_ok=True)
    
    num_users = rateBuckets["user-movie"]["train"].shape[0]
    num_items = rateBuckets["movie-user"]["train"].shape[0]
    
    model = FedGNNModel(num_users, num_items, cfg.model.hidden_dim, 100, cfg.model.dropout)
    opt = SGD(model.parameters(), lr=cfg.learning_rate[0])
    
    curr_model = setup_model(model, 0, cfg.device)
    sync_model(curr_model)
    
    
    # Training Layer 0
    print("Training Layer 0")
    
    
    train_ratings = graph.edges["user-movie"].data["rate"][graph.edges["user-movie"].data["train_mask"]].to(device) 
    
    userBuckets = rateBuckets["user-movie"]["train"].to(device)
    itemBuckets = rateBuckets["movie-user"]["train"].to(device)
    # userBuckets = graph.nodes("user").to(device)#rateBuckets["user-movie"]["train"].to(device)
    # itemBuckets = graph.nodes("movie").to(device)#rateBuckets["movie-user"]["train"].to(device)
    itemFeats = graph.nodes["movie"].data["feat"].to(device)
    
    user_embs = [{} for _ in range(cfg.num_layers + 1)]
    item_embs = [{} for _ in range(cfg.num_layers + 1)]
 
    train_graph = graph.edge_subgraph(graph.edata["train_mask"], relabel_nodes=False)
    train_edges = train_graph.edges["user-movie"].data["train_mask"]
    test_edges = graph.edges["user-movie"].data["test_mask"]
    test_ratings = graph.edges["user-movie"].data["rate"][test_edges].to(device) 
      
    for round in range(cfg.num_rounds[0]):
        curr_model.train()
        
        out, user_embs[0]["train"], item_embs[0]["train"] = curr_model(train_graph, userBuckets, itemBuckets, train_edges, "user-movie")
        
        # clamp/process the output?
        
        loss = mse_loss(out.view(-1), train_ratings)
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        # Test on full graph and test edges  
        curr_model.eval()
        with torch.no_grad():
            test_out, user_embs[0]["test"], item_embs[0]["test"] = curr_model(graph, userBuckets, itemBuckets, test_edges, "user-movie")
        
            test_loss = mse_loss(test_out.view(-1), test_ratings)
        
            print("L0 Round: ", round, "Train Loss: ", torch.sqrt(loss).item(), "Test Loss: ", torch.sqrt(test_loss).item())
        
    
    # Remaining layers
    train_graph = train_graph.to("cuda")
    graph = graph.to("cuda")
    for curr_layer in range(1, cfg.num_layers + 1):
        curr_model = setup_model(model, curr_layer, cfg.device)
        sync_model(curr_model)
        opt = SGD(model.parameters(), lr=cfg.learning_rate[curr_layer])
                
        for round in range(cfg.num_rounds[curr_layer]):
            curr_model.train()
            
            out, user_embs[curr_layer]["train"], item_embs[curr_layer]["train"] = curr_model(train_graph, user_embs[curr_layer-1]["train"], item_embs[curr_layer-1]["train"], train_edges, "user-movie")
            
            # clamp/process the output?
            
            loss = mse_loss(out.view(-1), train_ratings)
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            curr_model.eval()
            with torch.no_grad():
                test_out, user_embs[curr_layer]["test"], item_embs[curr_layer]["test"] = curr_model(graph, user_embs[curr_layer-1]["test"], item_embs[curr_layer-1]["test"], test_edges, "user-movie")
            
                test_loss = mse_loss(test_out.view(-1), test_ratings)
            
                print("L", curr_layer, "Round: ", round, "Train Loss: ", torch.sqrt(loss).item(), "Test Loss: ", torch.sqrt(test_loss).item())
        
    print("TODO")

def init_process(rank, cfg, hydra_output_dir):  
    """Initialize the distributed environment"""

    os.environ["MASTER_ADDR"] = cfg.distributed.master_addr
    os.environ["MASTER_PORT"] = cfg.distributed.master_port
    dist.init_process_group(
        cfg.distributed.backend, rank=rank, world_size=cfg.num_partitions
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
    
    dataset = MovieLensDataset(name="ml-1m", valid_ratio=0.1, test_ratio=0.2, raw_dir=cfg.dataset_dir)
    graph = dataset[0]
    
    # complete processing by creating rate_buckets

    rate_buckets = {}
    for etype in ["user-movie", "movie-user"]:
        edges = graph.edges(etype=etype)
        ids = edges[0]
        ratings = graph.edges[etype].data['rate'].long() 
        rate_buckets[etype] = {}
        for split in ["train", "valid", "test"]:
            mask = graph.edges[etype].data[split + '_mask']
            node_type = etype.split('-')[0]
            rate_buckets[etype][split] = torch.zeros((graph.num_nodes(node_type), 5), dtype=torch.int64)
            
            for node, rating in zip(ids[mask], ratings[mask]):
                rate_buckets[etype][split][node, rating - 1] += 1


    # Add rate_buckets as a node feature for users
    # graph.nodes['user'].data['rate_bucket'] = rate_buckets["user-movie"]
    # graph.nodes['movie'].data['rate_bucket'] = rate_buckets["movie-user"]
    
    start_time = time.time()
    train(
        graph,
        rate_buckets,
        cfg,
        hydra_output_dir,
        results_dir,
    )
    print(f"Rank {rank:2} | Total time taken: {time.time() - start_time:2.4f} s")

    dist.destroy_process_group()
    