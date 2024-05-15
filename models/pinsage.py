from torch_geometric.nn import MLP 
import torch
from torch import Tensor
from torch.nn import Embedding, ModuleDict

import models.pinsage_layers as layers


class PinSAGEModel(torch.nn.Module):
    def __init__(self, full_graph, ntype, textsets, hidden_dims, n_layers):
        super().__init__()
        
        self.num_layers = n_layers
        self.full_graph = full_graph
        self.n_type = ntype

        self.proj = layers.LinearProjector(
            full_graph, ntype, textsets, hidden_dims
        )
        self.sage = layers.SAGENet(hidden_dims, n_layers)
        self.scorer = layers.ItemToItemScorer(full_graph, ntype)

    def forward(self, pos_graph, neg_graph, blocks):
        h_item = self.get_repr(blocks)
        pos_score = self.scorer(pos_graph, h_item)
        neg_score = self.scorer(neg_graph, h_item)
        return (neg_score - pos_score + 1).clamp(min=0)

    def get_repr(self, blocks):
        h_item = self.proj(blocks[0].srcdata)
        h_item_dst = self.proj(blocks[-1].dstdata)
        return h_item_dst + self.sage(blocks, h_item)
    
    def get_nth_layer(self, n: int):
        
        scorer = layers.ItemToItemScorer(self.full_graph, self.n_type)
        if n == 0:
            return FirstLayerModel(self.proj, scorer)
        
        sage = self.sage.get_nth_layer(n-1) #1st layer is projector
        
        if n == self.num_layers:
            scorer = self.scorer
            
        return IntermediateModel(sage, scorer)
    
class FirstLayerModel(torch.nn.Module):
    def __init__(self, proj, scorer) -> None:
        super().__init__()
        self.proj = proj
        self.scorer = scorer
    
    def forward(self, pos_graph, neg_graph, blocks):
        #project
        h_item = self.proj(blocks[0].srcdata)
        h_item_dst = self.proj(blocks[-1].dstdata)
        h_item = h_item + h_item_dst
        # scorer
        pos_score = self.scorer(pos_graph, h_item)
        neg_score = self.scorer(neg_graph, h_item)
        
        h_item_copy = h_item.detach()
        h_item_dst_copy = h_item.detach()
        
        return (neg_score - pos_score + 1).clamp(min=0), h_item_copy, h_item_dst_copy
    
    def get_repr(self, blocks):
        h_item = self.proj(blocks[0].srcdata)
        h_item_dst = self.proj(blocks[-1].dstdata)
        
        return h_item_dst + h_item, h_item, h_item_dst
    
class IntermediateModel(torch.nn.Module):
    def __init__(self, sage, scorer) -> None:
        super().__init__()
        self.sage = sage
        self.scorer = scorer
        
    def forward(self, pos_graph, neg_graph, h_item, h_item_dst, blocks):
        #project
        h_item = h_item_dst + self.sage(blocks, h_item)
        # scorer
        pos_score = self.scorer(pos_graph, h_item)
        neg_score = self.scorer(neg_graph, h_item)
        
        h_item_copy = h_item.detach()
        return (neg_score - pos_score + 1).clamp(min=0), h_item_copy
    
    def get_repr(self, blocks, h_item, h_item_dst):
        # h_item = self.proj(blocks[0].srcdata)
        # h_item_dst = self.proj(blocks[-1].dstdata)
        next_item = self.sage(blocks, h_item)
        next_item_copy = next_item.detach()
        return h_item_dst + next_item, next_item_copy
    
    