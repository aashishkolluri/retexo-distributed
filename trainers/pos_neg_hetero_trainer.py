"""Implement trainer that oversees end-to-end training process"""

import pickle
import os
import logging
from copy import deepcopy
import time
import threading
import copy
from typing import Any, Dict, List, Tuple
import dgl # type: ignore
import torch
from tqdm import tqdm
from torch.optim import lr_scheduler
import torch.distributed as dist
from torch import Tensor
from torch.utils.data import DataLoader
from dgl.distributed import GraphPartitionBook # type: ignore
from hydra.utils import instantiate
from omegaconf import DictConfig
from data.dataset import load_data
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

from models.newsSAGE import NewsSAGEModel
from models.utils.layers import seed_everything
from performance import PerformanceStore
from data.dataset import load_partition, load_user_item_data
from trainers.metrics import auc, mrr, nDCG
from trainers.worker_trainer import WorkerTrain
from models.pinsage import PinSAGEModel
import trainers.pinsage_eval as evaluation 
import torch.nn.functional as F


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
    dataset,
    cfg: DictConfig,
    hydra_output_dir: str,
    results_dir: str,
) -> None:
    """Implement end-to-end training process"""
    # set the seed
    set_torch_seed(cfg.seed)
    base_etypes = ['history', 'history_r']
    node_emb_meta = {
        'user': {
            'Category': 768,
            'SubCategory': 768,
            # 'Node2Vec': 128,
        },
        'news': {
            'News_Title_Embedding': 768,
            'News_Abstract_Embedding': 768,
            'Category': 768,
            'SubCategory': 768,
            # 'Node2Vec': 128,
        },
    }
    
    for ntype in dataset.num_node:
        dataset.graph.nodes[ntype].data['GNN_Emb'] = torch.zeros([dataset.num_node[ntype], cfg['hidden_dim'] * 2]).float()
    for etype in dataset.num_relation:
        dataset.graph.edges[etype].data['Sampling_Weight'] = torch.ones([dataset.num_relation[etype]]).float() * 0.5

    base_canonical_etypes = sorted([canonical_etype for canonical_etype in dataset.graph.canonical_etypes if canonical_etype[1] in base_etypes])


    # set up the model
    device = cfg.device
    perf_stores = [PerformanceStore()]
    perf_store = perf_stores[0]

    # rank, _ = dist.get_rank(), dist.get_world_size()

    log_dir = os.path.join(hydra_output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    best_acc = 0.0
    best_metrics = []
    best_epoch = 0
    best_model = None
    global_step = 0
    not_improved_count = 0
    acc_training_time = 0
    global_ct = []


    # Model
    num_layers = cfg.num_layers
    model = NewsSAGEModel(
        cfg.feat_hidden, cfg.hidden_dim, cfg.hidden_dim, cfg.hidden_dim, node_emb_meta, device, num_layers, cfg.cross_score, cfg.dropout
    ).to(device)

    
    # Optimizer
    opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate[0])
    
    pos_dataloader, neg_dataloader = dataset.get_gnn_train_loader(base_etypes, cfg["num_layers"])
    trainloader = enumerate(zip(pos_dataloader, neg_dataloader))
    
    for i, ((pos_input_nodes, pos_sample_graph, pos_blocks), (neg_input_nodes, neg_sample_graph, neg_blocks)) in trainloader:
        assert i == 0
        pos_sample_graph = pos_sample_graph
        pos_blocks = pos_blocks
        neg_sample_graph = neg_sample_graph
        neg_blocks = neg_blocks
    
    # first layer
    print("Starting training... Layer 0")
    for i in range(cfg.num_rounds[0]):
        model.train()

        iter_start_time = time.time()
        # TODO for first layer will need the sample graph to be without message passing (ie. only link prediction on src/dst nodes)
        pos_sample_graph = pos_sample_graph.to(device)
        pos_blocks = [b.to(device) for b in pos_blocks]
        pos_scores, pos_output_features, pos_gnn_kls = model(pos_sample_graph, pos_blocks, ('user', 'pos_train', 'news'))
        neg_sample_graph = neg_sample_graph.to(device)
        neg_blocks = [b.to(device) for b in neg_blocks]
        neg_scores, neg_output_features, neg_gnn_kls = model(neg_sample_graph, neg_blocks, ('user', 'neg_train', 'news'))

        pred = torch.cat([pos_scores.unsqueeze(1), neg_scores.reshape(-1, cfg['gnn_neg_ratio'])], dim=1)
        score_diff = (F.sigmoid(pred)[:, 0] - F.sigmoid(pred)[:, 0:].mean(dim=1)).mean()
        
        if cfg['loss_func'] == 'log_sofmax':
            pred_loss = (-torch.log_softmax(pred, dim=1).select(1, 0)).mean()
        elif cfg['loss_func'] == 'cross_entropy':
            label = torch.cat([torch.ones([pred.shape[0], 1]), torch.zeros([pred.shape[0], cfg['gnn_neg_ratio']])], dim=1).to(device)
            pred_loss = F.binary_cross_entropy(F.sigmoid(pred), label)
        else:
            raise Exception('Unexpected Loss Function')
        
        gnn_kl = (sum(pos_gnn_kls) / len(pos_gnn_kls) + sum(neg_gnn_kls) / len(neg_gnn_kls)).mean()
        
        loss = pred_loss + cfg['gnn_kl_weight'] * gnn_kl

        opt.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1) # try other values?
        # if cfg['pruning']:
        #     model.collecting_metapath_utility()
        opt.step()

        iter_end_time = time.time()
        iter_elapsed_time = iter_end_time - iter_start_time
        epoch_time = iter_elapsed_time * len(pos_dataloader)
        acc_training_time += iter_elapsed_time


        if i >= cfg.eval_after and (i + 1) % cfg.log_every == 0:
            print('\nTrain Result @ Iter = {}\n- Training Loss = {}\n- Predict Loss = {}\n- KL = {}\n- Score Diff = {}\n'.format(
                i, loss.item(), pred_loss.item(), gnn_kl.item(), score_diff.item()
            ))
            result = eval(base_etypes, dataset,  cfg.hidden_dim, device, model, i, cfg)
            this_acc = result[0]
            with open(log_dir + "/accuracy.txt", "a+") as f:
                f.write(
                    f'Epoch {i}, AUC: {this_acc} - MRR = {result[1]} - nDCG@5 = {result[2]} - nDCG@10 = {result[3]} - ILAD@5 = {result[4]} - ILAD@10 = {result[5]}\n'
                )
            if this_acc > best_acc:
                best_acc = this_acc
                best_metrics = result
                best_epoch = i
                best_model = deepcopy(model.state_dict())
                torch.save(best_model, '{}/{}_{}_seed={}.pth'.format(
                    hydra_output_dir, 
                    "mind", 
                    "0", 
                    cfg.seed
                ))
                not_improved_count = 0
            else:
                not_improved_count += 1
                if not_improved_count >= cfg.early_stop:
                    break
    fstr = '\nDONE after {} iterations\nBest AUC: {} at epoch {}. All metrics: {}'.format(i, best_acc, best_epoch, best_metrics)
    print(fstr)
    with open(log_dir + "/accuracy.txt", "a+") as f:
        f.write(fstr)
        

def eval(etypes, mind_dgl, out_dim, device, model, epoch, cfg):
    for ntype in mind_dgl.num_node:
        mind_dgl.graph.nodes[ntype].data['GNN_Emb'] = torch.zeros([mind_dgl.num_node[ntype], out_dim * 2]).float()
    for etype in mind_dgl.num_relation:
        mind_dgl.graph.edges[etype].data['Sampling_Weight'] = torch.ones([mind_dgl.num_relation[etype]]).float() * 0.5

    encode_all_graph(model, mind_dgl, device, etypes)
    
    if cfg["quick_eval"]:
        result = quick_rec(model, mind_dgl, epoch, cfg)
    else:
        result = full_rec(model, mind_dgl, epoch, cfg)
    return result


def init_process(rank, cfg, hydra_output_dir):
    """Initialize the distributed environment"""

    os.environ["MASTER_ADDR"] = cfg.distributed.master_addr
    os.environ["MASTER_PORT"] = cfg.distributed.master_port
    dist.init_process_group(
        cfg.distributed.backend, rank=rank, world_size=cfg.num_partitions
    )

    seed_everything(cfg.seed)
    
    # central version for now
    graph, dataset = load_data(**cfg.dataset.download, cfg=cfg)


    os.makedirs("results/", exist_ok=True)
    os.makedirs(
        f"results/{cfg.dataset.partition.dataset_name}_{cfg.model.conv_layer._target_}_{cfg.dataset.partition.num_parts}/",
        exist_ok=True,
    )
    results_dir = f"results/{cfg.dataset.partition.dataset_name}_{cfg.model.conv_layer._target_}_{cfg.dataset.partition.num_parts}/rank_{rank}/"
    os.makedirs(
        results_dir,
        exist_ok=True,
    )

    
    
    start_time = time.time()
    train(
        graph,
        dataset,
        cfg,
        hydra_output_dir,
        results_dir,
    )
    print(f"Rank {rank:2} | Total time taken: {time.time() - start_time:2.4f} s")

    dist.destroy_process_group()



def encode_all_graph(model, mind_dgl, device, etypes, attention_head=4):
    print('Generating GNN Representation')
    model.eval()
    with torch.no_grad():
        user_dataloader, news_dataloader = mind_dgl.get_gnn_dev_node_loader(etypes, model.n_layers)

        user_dataloader = enumerate(user_dataloader)
        news_dataloader = enumerate(news_dataloader)

        for i, (user_input_nodes, user_sample_graph, user_blocks) in user_dataloader:
            user_blocks = [b.to(device) for b in user_blocks]
            user_output_features = model.encode(user_blocks)

            mind_dgl.graph.nodes['user'].data['GNN_Emb'][user_blocks[-1].dstdata['_ID']['user'].long()] = user_output_features['user'].cpu()
        for i, (news_input_nodes, news_sample_graph, news_blocks) in news_dataloader:
            news_blocks = [b.to(device) for b in news_blocks]
            news_output_features = model.encode(news_blocks)

            mind_dgl.graph.nodes['news'].data['GNN_Emb'][news_blocks[-1].dstdata['_ID']['news'].long()] = news_output_features['news'].cpu()
    print('Generating GNN Representation Finished')
    
    
def full_rec(model, mind_dgl, epoch, cfg):
    # testing performance w/o EDC using randomly sampled users
    
    dev_session_loader = mind_dgl.get_dev_session_loader(shuffle=False)
    cache_size = cfg.cache_size 
    
    
    mind_dgl.graph.nodes['user'].data['News_Pref'] = mind_dgl.graph.nodes['user'].data['GNN_Emb'].unsqueeze(1).unsqueeze(1).repeat(1, mind_dgl.graph.nodes['news'].data['CateID'].max()+1, cache_size, 1)
    mind_dgl.graph.nodes['user'].data['Last_Update_Time'] = torch.zeros([mind_dgl.num_node['user'], mind_dgl.graph.nodes['news'].data['CateID'].max()+1, cache_size])
   
    
    epoch_auc_score = 0
    epoch_mrr = 0
    epoch_ndcg_5 = 0
    epoch_ndcg_10 = 0
    epoch_ilad_5 = 0
    epoch_ilad_10 = 0


    devloader = enumerate(dev_session_loader)

    print("Evaluating...\n")
    for i, (pos_links, neg_links) in devloader:
        if cfg['gnn_quick_dev_reco'] and i >= cfg['gnn_quick_dev_reco_size']:
            break
        sub_g = dgl.edge_subgraph(mind_dgl.graph, {('news', 'pos_dev_r', 'user'): pos_links, ('news', 'neg_dev_r', 'user'): neg_links})
        # sub_g.apply_nodes(model.scorer.get_representation, ntype='user')
        sub_g.apply_nodes(model.scorer.get_representation, ntype='news')
        sub_g.update_all(model.scorer.msgfunc_score_neg_edc, model.scorer.reduce_score_neg_edc, etype=('news', 'neg_dev_r', 'user'))
        sub_g.update_all(model.scorer.msgfunc_score_pos_edc, model.scorer.reduce_score_pos_edc, etype=('news', 'pos_dev_r', 'user'))
        mind_dgl.graph.nodes['user'].data['News_Pref'][sub_g.dstdata['_ID']['user'].long()] = sub_g.dstdata['pref']['user']  # write back to mind.graph
        mind_dgl.graph.nodes['user'].data['Last_Update_Time'][sub_g.dstdata['_ID']['user'].long()] = sub_g.dstdata['lut']['user']

        labels = torch.cat([torch.ones(sub_g.nodes['user'].data['pos_score'].shape), torch.zeros(sub_g.nodes['user'].data['neg_score'].shape)], dim=1).type(torch.int32).squeeze(0)
        scores = torch.cat([sub_g.nodes['user'].data['pos_score'], sub_g.nodes['user'].data['neg_score']], dim=1).squeeze(0)
        news_representation = torch.cat([sub_g.nodes['user'].data['pos_news_representation'], sub_g.nodes['user'].data['neg_news_representation']], dim=1).squeeze(0)
        
        auc_score = auc(labels.numpy(), scores.numpy())
        mrr_score = mrr(labels.numpy(), scores.numpy())
        ndcg_5 = nDCG(labels.unsqueeze(0).numpy(), scores.unsqueeze(0).numpy(), k=5)
        ndcg_10 = nDCG(labels.unsqueeze(0).numpy(), scores.unsqueeze(0).numpy(), k=10)
        
        epoch_auc_score += auc_score
        epoch_mrr += mrr_score
        epoch_ndcg_5 += ndcg_5
        epoch_ndcg_10 += ndcg_10

        if news_representation.shape[0] >= 5:
            top_5_news_representation = news_representation[torch.topk(scores, k=5).indices]
        else:
            top_5_news_representation = news_representation
        top_5_news_representation = (top_5_news_representation.T / top_5_news_representation.norm(dim=1)).T
        
        if news_representation.shape[0] >= 10:
            top_10_news_representation = news_representation[torch.topk(scores, k=10).indices]
        else:
            top_10_news_representation = news_representation
        top_10_news_representation = (top_10_news_representation.T / top_10_news_representation.norm(dim=1)).T

    if cfg['gnn_quick_dev_reco']:
        epoch_auc_score /= cfg['gnn_quick_dev_reco_size']
        epoch_mrr /= cfg['gnn_quick_dev_reco_size']
        epoch_ndcg_5 /= cfg['gnn_quick_dev_reco_size']
        epoch_ndcg_10 /= cfg['gnn_quick_dev_reco_size']
        epoch_ilad_5 /= cfg['gnn_quick_dev_reco_size']
        epoch_ilad_10 /= cfg['gnn_quick_dev_reco_size']
    else:
        epoch_auc_score /= len(dev_session_loader)
        epoch_mrr /= len(dev_session_loader)
        epoch_ndcg_5 /= len(dev_session_loader)
        epoch_ndcg_10 /= len(dev_session_loader)
        epoch_ilad_5 /= len(dev_session_loader)
        epoch_ilad_10 /= len(dev_session_loader)

    print('Testing Result @ Epoch = {}\n- AUC = {}\n- MRR = {}\n- nDCG@5 = {}\n- nDCG@10 = {}\n- ILAD@5 = {}\n- ILAD@10 = {}\n'.format(epoch, epoch_auc_score, epoch_mrr, epoch_ndcg_5, epoch_ndcg_10, epoch_ilad_5, epoch_ilad_10))

    return [epoch_auc_score, epoch_mrr, epoch_ndcg_5, epoch_ndcg_10, epoch_ilad_5, epoch_ilad_10]


def quick_rec(model, mind_dgl, epoch, cfg):
    # testing performance w/o EDC using randomly sampled users
    
    dev_session_loader = mind_dgl.get_dev_session_loader(shuffle=True)
    cache_size = cfg.cache_size 

    epoch_auc_score = 0
    epoch_mrr = 0
    epoch_ndcg_5 = 0
    epoch_ndcg_10 = 0
    epoch_ilad_5 = 0
    epoch_ilad_10 = 0


    devloader = enumerate(dev_session_loader)
    # pos_links = mind_dgl._dev_session_positive
    # neg_links = mind_dgl._dev_session_negative

    print("Evaluating...\n")
    for i, (pos_links, neg_links) in devloader:
        if cfg['gnn_quick_dev_reco'] and i >= cfg['gnn_quick_dev_reco_size']:
            break
        sub_g = dgl.edge_subgraph(mind_dgl.graph, {('news', 'pos_dev_r', 'user'): pos_links, ('news', 'neg_dev_r', 'user'): neg_links})
        sub_g.apply_nodes(model.scorer.get_representation, ntype='user')
        sub_g.apply_nodes(model.scorer.get_representation, ntype='news')
        sub_g.update_all(model.scorer.msgfunc_score_neg, model.scorer.reduce_score_neg, etype=('news', 'neg_dev_r', 'user'))
        sub_g.update_all(model.scorer.msgfunc_score_pos, model.scorer.reduce_score_pos, etype=('news', 'pos_dev_r', 'user'))

        labels = torch.cat([torch.ones(sub_g.nodes['user'].data['pos_score'].shape), torch.zeros(sub_g.nodes['user'].data['neg_score'].shape)], dim=1).type(torch.int32).squeeze(0)
        scores = torch.cat([sub_g.nodes['user'].data['pos_score'], sub_g.nodes['user'].data['neg_score']], dim=1).squeeze(0)
        news_representation = torch.cat([sub_g.nodes['user'].data['pos_news_representation'], sub_g.nodes['user'].data['neg_news_representation']], dim=1).squeeze(0)
        
        auc_score = auc(labels.numpy(), scores.numpy())
        mrr_score = mrr(labels.numpy(), scores.numpy())
        ndcg_5 = nDCG(labels.unsqueeze(0).numpy(), scores.unsqueeze(0).numpy(), k=5)
        ndcg_10 = nDCG(labels.unsqueeze(0).numpy(), scores.unsqueeze(0).numpy(), k=10)
        
        epoch_auc_score += auc_score
        epoch_mrr += mrr_score
        epoch_ndcg_5 += ndcg_5
        epoch_ndcg_10 += ndcg_10

        if news_representation.shape[0] >= 5:
            top_5_news_representation = news_representation[torch.topk(scores, k=5).indices]
        else:
            top_5_news_representation = news_representation
        top_5_news_representation = (top_5_news_representation.T / top_5_news_representation.norm(dim=1)).T
        
        if news_representation.shape[0] >= 10:
            top_10_news_representation = news_representation[torch.topk(scores, k=10).indices]
        else:
            top_10_news_representation = news_representation
        top_10_news_representation = (top_10_news_representation.T / top_10_news_representation.norm(dim=1)).T

    if cfg['gnn_quick_dev_reco']:
        epoch_auc_score /= cfg['gnn_quick_dev_reco_size']
        epoch_mrr /= cfg['gnn_quick_dev_reco_size']
        epoch_ndcg_5 /= cfg['gnn_quick_dev_reco_size']
        epoch_ndcg_10 /= cfg['gnn_quick_dev_reco_size']
        epoch_ilad_5 /= cfg['gnn_quick_dev_reco_size']
        epoch_ilad_10 /= cfg['gnn_quick_dev_reco_size']
    else:
        epoch_auc_score /= len(dev_session_loader)
        epoch_mrr /= len(dev_session_loader)
        epoch_ndcg_5 /= len(dev_session_loader)
        epoch_ndcg_10 /= len(dev_session_loader)
        epoch_ilad_5 /= len(dev_session_loader)
        epoch_ilad_10 /= len(dev_session_loader)
            

    print('Testing Result @ Epoch = {}\n- AUC = {}\n- MRR = {}\n- nDCG@5 = {}\n- nDCG@10 = {}\n- ILAD@5 = {}\n- ILAD@10 = {}\n'.format(epoch, epoch_auc_score, epoch_mrr, epoch_ndcg_5, epoch_ndcg_10, epoch_ilad_5, epoch_ilad_10))

    return [epoch_auc_score, epoch_mrr, epoch_ndcg_5, epoch_ndcg_10, epoch_ilad_5, epoch_ilad_10]


