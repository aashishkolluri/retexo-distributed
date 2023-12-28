"""Implement simple gnn model"""

from typing import Optional
from hydra.utils import instantiate
import torch
from torch import nn
from dgl.nn import GraphConv # type: ignore
from models.base_model import BaseGNN


class SimpleGNN(BaseGNN):
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
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        conv_layer: Optional[nn.Module] = GraphConv,
        n_layers: int = 2,
        activation: nn.Module = nn.ReLU(),
        dropout: float = 0.0,
        use_layer_norm: bool = True,
    ) -> None:
        super().__init__()
        self.n_layers = n_layers
        if n_layers == 1:
            self.convs = nn.ModuleList(
                [instantiate(conv_layer, input_dim, output_dim)]
            )
        else:
            self.convs = nn.ModuleList(
                [
                    instantiate(conv_layer, input_dim, hidden_dim)
                    if i == 0
                    else instantiate(conv_layer, hidden_dim, hidden_dim)
                    for i in range(n_layers - 1)
                ]
            )
            self.convs.append(instantiate(conv_layer, hidden_dim, output_dim))
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.output_dim = output_dim
        self.use_layer_norm = use_layer_norm

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
