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
from torch.utils.data import DataLoader
from dgl.distributed import GraphPartitionBook # type: ignore
from hydra.utils import instantiate
from omegaconf import DictConfig

from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

from performance import PerformanceStore
from data.dataset import load_partition, load_user_item_data
from trainers.worker_trainer import WorkerTrain
from models.pinsage import PinSAGEModel
import trainers.pinsage_eval as evaluation 

from comm_utils import (
    get_boundary_nodes,
    send_and_receive_embeddings,
    aggregate_metrics,
    sync_model,
    MultiThreadReducerCentralized,
)

import models.pinsage_sampler as sampler_module

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
    graph: dgl.DGLGraph,
    train_graph,
    dataset,
    # num_classes: int,
    # num_tr_val_te: Tuple[int, int, int],
    # graph_partition_book: GraphPartitionBook,
    cfg: DictConfig,
    hydra_output_dir: str,
    results_dir: str,
) -> None:
    """Implement end-to-end training process"""

    # set the seed
    set_torch_seed(cfg.seed)
    
    num_tr_val_te = [0,0,0]
    
    val_matrix = dataset["val-matrix"].tocsr()
    test_matrix = dataset["test-matrix"].tocsr()
    item_texts = dataset["item-texts"]
    user_ntype = dataset["user-type"]
    item_ntype = dataset["item-type"]
    user_to_item_etype = dataset["user-to-item-type"]
    timestamp = dataset["timestamp-edge-column"]

    # set up the model
    # model = instantiate(cfg.model, input_dim=num_feat, output_dim=1)
    device = cfg.device
    perf_stores = [PerformanceStore()]
    perf_store = perf_stores[0]

    # rank, _ = dist.get_rank(), dist.get_world_size()

    os.makedirs(os.path.join(hydra_output_dir, "results"), exist_ok=True)
    
    # == Training taking from DGL Pinsage example ==
    # Assign user and movie IDs and use them as features (to learn an individual trainable
    # embedding for each entity)
    
    train_graph.nodes[user_ntype].data["id"] = torch.arange(train_graph.num_nodes(user_ntype))
    train_graph.nodes[item_ntype].data["id"] = torch.arange(train_graph.num_nodes(item_ntype))
    
    # Prepare torchtext dataset and Vocabulary
    textset = {}
    tokenizer = get_tokenizer(None)

    textlist = []
    batch_first = True

    for i in range(train_graph.num_nodes(item_ntype)):
        for key in item_texts.keys():
            l = tokenizer(item_texts[key][i].lower())
            textlist.append(l)
    for key, field in item_texts.items():
        vocab2 = build_vocab_from_iterator(
            textlist, specials=["<unk>", "<pad>"]
        )
        textset[key] = (
            textlist,
            vocab2,
            vocab2.get_stoi()["<pad>"],
            batch_first,
        )


    # Sampler
    batch_sampler = sampler_module.FullBatchItemToItemSampler(
        train_graph, user_ntype, item_ntype
    )
    neighbor_sampler = sampler_module.NeighborSampler(
        train_graph,
        user_ntype,
        item_ntype,
        2,
        0.5,
        10,
        2,
        2,
    )
    collator = sampler_module.PinSAGECollator(
        neighbor_sampler, train_graph, item_ntype, textset
    )
    dataloader = DataLoader(
        batch_sampler,
        collate_fn=collator.collate_train,
    )
    dataloader_test = DataLoader(
        torch.arange(train_graph.num_nodes(item_ntype)),
        batch_size=train_graph.num_nodes(item_ntype),
        collate_fn=collator.collate_test,
    )
    dataloader_it = iter(dataloader)

    # Model
    num_layers = cfg.num_layers
    model = PinSAGEModel(
        train_graph, item_ntype, textset, cfg.hidden_dim, num_layers
    ).to(device)
    
    # Optimizer
    opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate[0])
    
    curr_model = setup_model(model, 0, cfg.device)
    sync_model(curr_model)
    
    # first layer
    pos_graph, neg_graph, train_blocks = next(dataloader_it)
    for i in range(len(train_blocks)):
        train_blocks[i] = train_blocks[i].to(device)
        
    test_blocks = next(iter(dataloader_test))
    for i in range(len(test_blocks)):
        test_blocks[i] = test_blocks[i].to(device)
        
        
    print("Starting training... Layer 0")
    for i in range(cfg.num_rounds[0]):
        curr_model.train()
        # Copy to GPU
       
        pos_graph = pos_graph.to(device)
        neg_graph = neg_graph.to(device)

        loss, train_h_item, train_h_item_dst = curr_model(pos_graph, neg_graph, train_blocks)
        loss = loss.mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        # Evaluate
        curr_model.eval()
        with torch.no_grad():
            item_batches = torch.arange(train_graph.num_nodes(item_ntype))
            h_item_batches = []
            repr, val_h_item, val_h_item_dst = curr_model.get_repr(test_blocks)
            h_item_batches.append(repr)
            h_items = torch.cat(h_item_batches, 0)

            print(
                evaluation.evaluate_nn(dataset, h_items, 10, train_graph.num_nodes(item_ntype))
            )
    
    # reminaings
    
    for i in range(1, num_layers + 1):
        print("Layer " + str(i))
        curr_model = setup_model(model, i, cfg.device)
        sync_model(curr_model)
        
        opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate[i])
        
        for j in range(cfg.num_rounds[0]):
            curr_model.train()

            # TODO copy / detach graph / blocks?
            
            loss, train_h_item = curr_model(pos_graph, neg_graph, train_h_item, train_h_item_dst, train_blocks)
            loss = loss.mean()
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            # Evaluate
            curr_model.eval()
            with torch.no_grad():
                item_batches = torch.arange(train_graph.num_nodes(item_ntype))
                h_item_batches = []

                repr, val_h_item = curr_model.get_repr(test_blocks, val_h_item, val_h_item_dst)
                h_item_batches.append(repr)
                h_items = torch.cat(h_item_batches, 0)

                print(
                    evaluation.evaluate_nn(dataset, h_items, 10, train_graph.num_nodes(item_ntype))
                )
    
    # if rank == 0:
    #     os.makedirs(os.path.join(hydra_output_dir, "checkpoint"), exist_ok=True)

    # get the in and out graph
    # in_graph, _ = get_in_out_graph(graph, node_dict)

    # logger.info(
    #     "Process %d has %d nodes and %d edges, with fractions %f inner nodes and %f inner edges",
    #     rank,
    #     graph.num_nodes(),
    #     graph.num_edges(),
    #     in_graph.num_nodes() / graph.num_nodes(),
    #     in_graph.num_edges() / graph.num_edges(),
    # )

    # # get the boundary node lists
    # boundary_nodes = get_boundary_nodes(node_dict, graph_partition_book)

    # # get the inner node indices
    # inner_node_indices = torch.arange(node_dict["inner_node"].int().sum())

    # # get the number of local samples for train val and test
    # num_local_train = node_dict["train_mask"].int().sum()
    # num_local_val = node_dict["val_mask"].int().sum()
    # num_local_test = node_dict["test_mask"].int().sum()

    # # get the dataloaders and model for the first layer
    # # create zeroth embedding for all nodes
    # node_dict["feat_0"] = torch.zeros((len(node_dict[dgl.NID]), num_feat))
    # node_dict["feat_0"][inner_node_indices] = node_dict["feat"]

    # # send and receive the zeroth embedding of all nodes to their graph neighbors
    # emb_share_time_s = time.time()
    # send_and_receive_embeddings(
    #     boundary_nodes, node_dict, "feat_0", comm_volume_perf_store, cfg.measure_dv
    # )
    # perf_store.set_emb_broadcast_time(time.time() - emb_share_time_s)

    # # set up the task
    # task = instantiate(cfg.task)
    
    # # get the induced train, val, and test subgraphs on the training nodes
    # traindata, valdata, testdata = task.get_tr_val_te_data(
    #     graph, node_dict, "feat_0"
    # )
    
    # # == Creating a full batch for training ==
    # # Sampler
    # batch_sampler = sampler_module.ItemToItemBatchSampler(
    #     graph, user_type, item_type, batch_size=graph.num_nodes() 
    # )
    
    # neighbor_sampler = sampler_module.NeighborSampler(
    #     graph,
    #     user_type,
    #     item_type,
    #     2,
    #     0.5,
    #     20, # default 10
    #     5, # default 3
    #     1,
    # )
    # collator = sampler_module.PinSAGECollator(
    #     # neighbor_sampler, graph, item_type, textset
    # )
    # dataloader = DataLoader(
    #     batch_sampler,
    #     collate_fn=collator.collate_train,
    # )
    
    
    # # ===================
    # # set up the model for the first layer
    # curr_model = setup_model(model, 0, cfg.device)
    # sync_model(curr_model)

    # if isinstance(cfg.learning_rate, float):
    #     cfg.learning_rate = [cfg.learning_rate]
    # # set up the optimizer
    # optimizer = instantiate(
    #     cfg.optimizer, lr=cfg.learning_rate[0], params=curr_model.parameters()
    # )
    # # set up the loss function
    # loss_function = task.get_loss_function()
    # # set up the metrics
    # metrics = task.get_evaluation_metrics()
    # # set up the worker trainer
    # worker_trainer = WorkerTrain()
    # if isinstance(cfg.num_rounds, int):
    #     cfg.num_rounds = [cfg.num_rounds]

    # # get the data ready to compute embeddings for next layer in parallel
    # prev_feat_tag = "feat_0"
    # emb_data_dict: Dict[str, Tuple[Any, Tensor, Tensor]] = {}
    # emb_data_thread = threading.Thread(
    #     target=construct_graph_and_features_to_compute_next_embedding,
    #     args=(emb_data_dict, task, graph, node_dict, prev_feat_tag, inner_node_indices),
    # )
    # emb_data_thread.start()

    # # train the first layer
    # os.makedirs(results_dir + f"layer_0/", exist_ok=True)
    # train_for_one_layer(
    #     curr_model,
    #     traindata,
    #     valdata,
    #     testdata,
    #     optimizer,
    #     metrics,
    #     loss_function,
    #     cfg,
    #     worker_trainer,
    #     num_tr_val_te,
    #     tuple([num_local_train, num_local_val, num_local_test]),
    #     cfg.num_rounds[0],
    #     perf_store,
    #     results_dir + f"layer_0/",
    # )

    # for curr_layer in range(1, cfg.model.n_layers):
    #     perf_stores.append(PerformanceStore())
    #     perf_store = perf_stores[curr_layer]

    #     # get the feat_1 of the nodes
    #     curr_shape_feat = (
    #         node_dict["inner_node"].int().sum(),
    #         curr_model.agg_layer._out_feats,
    #     )
    #     # get the embeddings (features) to train the current layer
    #     compute_embs_time_s = time.time()
    #     emb_data_thread.join()
    #     emb_data = emb_data_dict["emb_data"]
    #     if device == "cuda":
    #         emb_data = [data.to(torch.device("cuda")) for data in emb_data] # type: ignore
    #     curr_feats = worker_trainer.get_embeddings(
    #         curr_model, emb_data, curr_shape_feat, hidden_layer=True
    #     )
    #     curr_feat_tag = "feat_" + str(curr_layer)
    #     node_dict[curr_feat_tag] = torch.zeros(
    #         (len(node_dict[dgl.NID]), curr_feats.shape[1])
    #     )
    #     node_dict[curr_feat_tag][inner_node_indices] = curr_feats
    #     perf_store.add_compute_local_embs_time(time.time() - compute_embs_time_s)

    #     # send and receive the embeddings of all nodes to their graph neighbors
    #     emb_share_time_s = time.time()
    #     send_and_receive_embeddings(
    #         boundary_nodes,
    #         node_dict,
    #         curr_feat_tag,
    #         comm_volume_perf_store,
    #         cfg.measure_dv,
    #     )
    #     perf_store.set_emb_broadcast_time(time.time() - emb_share_time_s)

    #     # set up the dataloaders and model for the current layer
    #     traindata, valdata, testdata = task.get_tr_val_te_data(
    #         graph, node_dict, curr_feat_tag
    #     )
    #     curr_model = setup_model(model, curr_layer, cfg.device)
    #     sync_model(curr_model)

    #     # reset the optimizer
    #     optimizer = instantiate(
    #         cfg.optimizer,
    #         lr=cfg.learning_rate[curr_layer]
    #         if curr_layer < len(cfg.learning_rate)
    #         else cfg.learning_rate[0],
    #         params=curr_model.parameters(),
    #     )
    #     # prepare the data to compute embeddings for next layer in parallel to training
    #     prev_feat_tag = curr_feat_tag
    #     if curr_layer < cfg.model.n_layers - 1:
    #         emb_data_dict.clear()
    #         emb_data_thread = threading.Thread(
    #             target=construct_graph_and_features_to_compute_next_embedding,
    #             args=(
    #                 emb_data_dict,
    #                 task,
    #                 graph,
    #                 node_dict,
    #                 prev_feat_tag,
    #                 inner_node_indices,
    #             ),
    #         )
    #         emb_data_thread.start()

    #     # train the current layer
    #     os.makedirs(results_dir + f"layer_{curr_layer}/", exist_ok=True)
    #     train_for_one_layer(
    #         curr_model,
    #         traindata,
    #         valdata,
    #         testdata,
    #         optimizer,
    #         metrics,
    #         loss_function,
    #         cfg,
    #         worker_trainer,
    #         num_tr_val_te,
    #         tuple([num_local_train, num_local_val, num_local_test]),
    #         cfg.num_rounds[curr_layer]
    #         if curr_layer < len(cfg.num_rounds)
    #         else cfg.num_rounds[0],
    #         perf_store,
    #         results_dir + f"layer_{curr_layer}/",
    #     )

    # per_metrics = []
    # total_comp_comm_time = 0.0
    # total_emb_broadcast_time = 0.0
    # avg_comp_comm_time = 0.0
    # for i, perf_store in enumerate(perf_stores):
    #     per_metrics.append(perf_store.get_necessary_time_metrics())
    #     total_comp_comm_time += per_metrics[-1]["total_local_train_time"]
    #     total_comp_comm_time += per_metrics[-1]["total_grad_reduce_time"]
    #     total_comp_comm_time += per_metrics[-1]["total_compute_local_embs_time"]
    #     total_comp_comm_time += per_metrics[-1]["total_emb_broadcast_time"]
    #     total_emb_broadcast_time += per_metrics[-1]["total_emb_broadcast_time"]
    #     avg_comp_comm_time += per_metrics[-1]["avg_local_train_time"]
    #     avg_comp_comm_time += per_metrics[-1]["avg_grad_reduce_time"]
    #     print(f"\nRank {rank:2} | Training time metrics for layer {i}")
    #     for k, v in per_metrics[-1].items():
    #         print(f"Rank {rank:2} | {k:2}: {v:2.4f}")
    # print(
    #     f"\nRank {rank:2} | total_comp_comm_time: {total_comp_comm_time:2.4f} | avg_comp_comm_time: {avg_comp_comm_time:2.4f} | total_emb_broadcast_time: {total_emb_broadcast_time:2.4f}"
    # )

    # per_metrics.append(
    #     {
    #         "total_comp_comm_time": total_comp_comm_time,
    #         "avg_comp_comm_time": avg_comp_comm_time,
    #         "total_emb_broadcast_time": total_emb_broadcast_time,
    #     }
    # )

    # if cfg.measure_dv:
    #     print(f"\nRank {rank:2} | Communication volume metrics")
    #     cv = comm_volume_perf_store.get_communication_volume()
    #     cv_message_passing_t = comm_volume_perf_store.get_cv_message_passing_t()
    #     cv_grad_reduce_t = comm_volume_perf_store.get_cv_grad_reduce_t()
    #     per_metrics.append(comm_volume_perf_store.get_necessary_cv_metrics())
    #     print(f"Rank {rank:2} | total communication_volume: {cv:2d}")
    #     print(f"Rank {rank:2} | cv_message_passing_t: {cv_message_passing_t:2d}")
    #     print(f"Rank {rank:2} | cv_grad_reduce_t: {cv_grad_reduce_t:2d}")

    # # save the performance metrics
    # with open(os.path.join(hydra_output_dir, "results", "perf_metrics.pkl"), "wb") as f:
    #     pickle.dump(per_metrics, f)

    # for i in range(len(per_metrics) - 2):
    #     with open(results_dir + f"layer_{i}/perf_metrics.txt", "a+") as f:
    #         # total_local_train_time, total_grad_reduce_time, avg_local_train_time, avg_grad_reduce_time
    #         f.write(
    #             f"{per_metrics[i]['total_local_train_time']}, {per_metrics[i]['total_grad_reduce_time']}, {per_metrics[i]['avg_local_train_time']}, {per_metrics[i]['avg_grad_reduce_time']}\n"
    #         )

    # if cfg.measure_dv:
    #     with open(results_dir + "total_perf_metrics.txt", "a+") as f:
    #         # 'total_comp_comm_time' 'avg_comp_comm_time' 'total_cv_message_passing_t' 'total_cv_grad_reduce_t', 'total_communication_volume'
    #         f.write(
    #             f"{per_metrics[-2]['total_comp_comm_time']}, {per_metrics[-2]['avg_comp_comm_time']}, {per_metrics[-2]['total_emb_broadcast_time']}, {per_metrics[-1]['total_cv_message_passing_t']}, {per_metrics[-1]['total_cv_grad_reduce_t']}, {per_metrics[-1]['total_communication_volume']}\n"
    #         )

    # # save the model
    # if rank == 0:
    #     torch.save(
    #         model.state_dict(), os.path.join(hydra_output_dir, "checkpoint", "model.pt")
    #     )
    #     print(
    #         f"\nModel saved at {os.path.join(hydra_output_dir, 'checkpoint', 'model.pt')}"
    #     )

def init_process(rank, cfg, hydra_output_dir):
    """Initialize the distributed environment"""

    os.environ["MASTER_ADDR"] = cfg.distributed.master_addr
    os.environ["MASTER_PORT"] = cfg.distributed.master_port
    dist.init_process_group(
        cfg.distributed.backend, rank=rank, world_size=cfg.num_partitions
    )

    # central version for now
    graph, dataset, train_graph = load_user_item_data(**cfg.dataset.download)

    # # load the partition (for edge_prediction)
    # (
    #     sub_graph,
    #     node_dict,
    #     edge_feat,
    #     num_feat,
    #     user_type,
    #     item_type,
    #     graph_partition_book,
    # ) = load_partition(
    #     cfg.dataset.partition.partition_dir, cfg.dataset.partition.dataset_name, rank, task="edge_prediction"
    # )


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
        train_graph,
        dataset,
        # tuple([num_train, num_val, num_test]),
        # graph_partition_book,
        cfg,
        hydra_output_dir,
        results_dir,
    )
    print(f"Rank {rank:2} | Total time taken: {time.time() - start_time:2.4f} s")

    dist.destroy_process_group()
