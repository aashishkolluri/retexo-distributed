"""Implement simple gnn model"""

from typing import Optional
from hydra.utils import instantiate
import torch
from torch import nn
from dgl.nn import GraphConv # type: ignore
from models.base_model import BaseGNN
from dgl.nn.pytorch import HeteroGraphConv, SAGEConv, GATv2Conv
import torch.nn.functional as F


from models.utils.layers import MultiLayerProcessorAdaptor, ScorePredictor, attention, kl


class NewsSAGEModel(BaseGNN):
    """Implement a GNN model

    Parameters
    ----------
    input_dim : int
        Input feature dimension
    hidden_dim : int
        Hidden layer dimension
    output_dim : int
        Output feature dimension
    conv_layer : Optional[nn.Module], optional
        Convolution layer, by default GCNConv
    n_layers : int, optional
        Number of layers, by default 2
    activation : nn.Module, optional
        Activation function, by default nn.ReLU()
    dropout : float, optional
        Dropout rate, by default 0.0
    """

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        feat_hidden: int,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        emb_data: None,
        device,
        n_layers: int,
        cross_score: bool,
        dropout: float = 0.0,
        activation: nn.Module = nn.ReLU(),
    ) -> None:
        super().__init__()
        self.n_layers = n_layers
        self.device = device
        self.node_emb_data = emb_data
        self.hetero_convs = nn.ModuleList()
        for _ in range(n_layers):
            self.hetero_convs.append(HeteroGraphConv({
            'history': SAGEConv(hidden_dim, hidden_dim, 'mean'),
            'history_r': SAGEConv(hidden_dim, hidden_dim, 'mean'),#GATv2Conv(hidden_dim, hidden_dim, 4)#
        }))
        
        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        self.output_dim = output_dim
        
        self.attr_set = {}
        self.adaptor_align = nn.ModuleDict()
        for node_type in emb_data:
            for emb_type in emb_data[node_type]:
                if emb_type in self.adaptor_align:
                    continue
                else:
                    self.adaptor_align[emb_type] = MultiLayerProcessorAdaptor([emb_data[node_type][emb_type], feat_hidden, input_dim])
                    self.attr_set[emb_type] = len(self.attr_set)
                    
        self.fusioner = nn.Sequential(
            nn.Linear(len(self.attr_set) * input_dim, 2 * input_dim),
            nn.Linear(2 * input_dim, input_dim),
        )
        fusioner_router = {}
        for node_type in self.node_emb_data:
            fusioner_router[node_type] = torch.zeros([len(self.attr_set), len(self.node_emb_data[node_type])])  # ALL_ATTR_NUM * CUR_NODE_ATTR_NUM
            for i, emb_type in enumerate(self.node_emb_data[node_type]):
                fusioner_router[node_type][self.attr_set[emb_type]][i] = 1
        self.fusioner_router = nn.ParameterDict({
            node_type: nn.Parameter(fusioner_router[node_type])
            for node_type in fusioner_router
        }).to(device)
        
        self.denser = nn.Linear(input_dim, output_dim * 2)
        self.scorer = ScorePredictor(output_dim, device, cross_score=cross_score)
        
    def adapt(self, blocks):
        input_features = {}
        for node_type in self.node_emb_data:
            node_attr = []
            for emb_type in self.node_emb_data[node_type]:
                # Directly fetch features from blocks[0].srcdata without using embeddings
                node_attr.append(self.adaptor_align[emb_type](
                    blocks[0].srcdata[emb_type][node_type].to(self.device)
                ).unsqueeze(1))
            node_attr = torch.cat(node_attr, dim=1)
            node_attr, _ = attention(node_attr, node_attr, node_attr, self.device)
            input_features[node_type] = node_attr
        return input_features
        
    def fusion(self, adapted_features):
        input_features = {}
        for node_type in adapted_features:
            if adapted_features[node_type].shape[0] == 0:
                continue
            else:
                input_features[node_type] = self.fusioner(
                    torch.matmul(self.fusioner_router[node_type], adapted_features[node_type]).reshape(adapted_features[node_type].shape[0], -1)
                )
        return input_features
    
    def forward(self, edge_subgraph, blocks, scoring_edge):
        adapted_features = self.adapt(blocks)
        input_features = self.fusion(adapted_features)
        
        for i in range(len(self.hetero_convs)):
            conv = self.hetero_convs[i]
            input_features = conv(blocks[i], input_features)
            # average GAT heads:
            # input_features["user"] = input_features["user"].mean(dim=1)
            input_features = {k: F.relu(v) for k, v in input_features.items()}
            if i != len(self.hetero_convs) - 1:
                input_features = {k: self.dropout(v) for k, v in input_features.items()}
            
            
        # output_features, _ = self.rgcn(blocks, input_features)
        output_features = input_features
        for node_type in output_features:
            output_features[node_type] = self.denser(output_features[node_type])
        kls = []
        for node_type in output_features:
            kls.append(kl(
                output_features[node_type][:, :self.output_dim], 
                output_features[node_type][:, self.output_dim:]
            ))
        return self.scorer(edge_subgraph, output_features, scoring_edge), output_features, kls
    
    def encode(self, blocks):
        adapted_features = self.adapt(blocks)
        input_features = self.fusion(adapted_features)
        
        for i in range(len(self.hetero_convs)):
            conv = self.hetero_convs[i]
            input_features = conv(blocks[i], input_features)
            # Average GAT heads
            # input_features["user"] = input_features["user"].mean(dim=1)
            input_features = {k: F.relu(v) for k, v in input_features.items()}
            
        output_features = input_features
            
        for node_type in output_features:
            output_features[node_type] = self.denser(output_features[node_type])
        kls = []
        for node_type in output_features:
            kls.append(kl(
                output_features[node_type][:, :self.output_dim], 
                output_features[node_type][:, self.output_dim:]
            ))
        return output_features
    
    def get_nth_layer(self, n: int) -> nn.Module:
        """Get the nth layer of the model

        Parameters
        ----------
        n : int
            Index of the layer

        Returns
        -------
        nn.Module
            The nth layer, object with forward method
        """
        conv_layer = self.convs[n]
        layer_norm = nn.LayerNorm(conv_layer._out_feats)
        clf_layer = nn.Linear(conv_layer._out_feats, self.output_dim)

        if n == len(self.convs) - 1:
            return IntermediateModel(conv_layer, nn.Sequential(), self.dropout)

        if self.use_layer_norm:
            intermediate_model = IntermediateModel(
                conv_layer, nn.Sequential(layer_norm, nn.ReLU(), clf_layer), self.dropout
            )
        else:
            intermediate_model = IntermediateModel(
                conv_layer, nn.Sequential(nn.ReLU(), clf_layer), self.dropout
            )
        return intermediate_model

class IntermediateModel(nn.Module):
    """Model with one aggregation layer and multiple following layers"""

    def __init__(self, agg_layer, following_layers, dropout) -> None:
        super().__init__()
        self.agg_layer = agg_layer
        self.following_layers = following_layers
        self.dropout = dropout

    def forward(
        self, graph, input_features: torch.Tensor, *args, **kwargs
    ) -> torch.Tensor:
        """Forward pass"""
        x = input_features
        x = self.dropout(x)
        x = self.agg_layer(graph, x)
        # for gatconv
        if len(x.shape) == 3:
            # take average along second dimension and keep the first dimension
            x = x.mean(dim=1)

        if "hidden_layer" in kwargs and kwargs["hidden_layer"]:
            if len(self.following_layers) > 2:
                for layer in self.following_layers[:-1]:
                    x = layer(x)
            if len(self.following_layers) == 2:
                # x = F.layer_norm(x, (x.shape[1],))
                x = nn.ReLU()(x)
            return x
        for layer in self.following_layers:
            x = layer(x)
        return x
