"""Implement simple gnn model"""

import torch
from torch import nn
from dgl.nn import GraphConv # type: ignore
from dgl.base import DGLError # type: ignore
import dgl.function as fn # type: ignore
from dgl.utils import expand_as_pair # type: ignore

class CustomGraphConv(GraphConv):
    """Custom GraphConv layer"""
    def __init__(self,
        in_feats,
        out_feats,
        norm="both",
        weight=True,
        bias=True,
        activation=None,
        allow_zero_in_degree=False,
        precomputed_feat=None,
        ):
        super().__init__(in_feats, out_feats, norm, weight, bias, activation, allow_zero_in_degree)
        self.precomputed_feat = precomputed_feat
        if not precomputed_feat:
            self.precomputed_feat = {}

    def forward(self, graph, feat, weight=None, edge_weight=None):
        """Custom Forward computation with precomputed features"""
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError(
                        "There are 0-in-degree nodes in the graph, "
                        "output for those nodes will be invalid. "
                        "This is harmful for some applications, "
                        "causing silent performance regression. "
                        "Adding self-loop on the input graph by "
                        "calling `g = dgl.add_self_loop(g)` will resolve "
                        "the issue. Setting ``allow_zero_in_degree`` "
                        "to be `True` when constructing this module will "
                        "suppress the check and let the code run."
                    )
            aggregate_fn = fn.copy_u("h", "m")
            if edge_weight is not None:
                assert edge_weight.shape[0] == graph.num_edges()
                graph.edata["_edge_weight"] = edge_weight
                aggregate_fn = fn.u_mul_e("h", "_edge_weight", "m")

            # (BarclayII) For RGCN on heterogeneous graphs we need to support GCN on bipartite.
            feat_src, feat_dst = expand_as_pair(feat, graph)
            if self._norm in ["left", "both"]:
                degs = graph.out_degrees().to(feat_src).clamp(min=1)
                if self._norm == "both":
                    norm = torch.pow(degs, -0.5)
                else:
                    norm = 1.0 / degs
                shp = norm.shape + (1,) * (feat_src.dim() - 1)
                norm = torch.reshape(norm, shp)
                feat_src = feat_src * norm

            if weight is not None:
                if self.weight is not None:
                    raise DGLError(
                        "External weight is provided while at the same time the"
                        " module has defined its own weight parameter. Please"
                        " create the module with flag weight=False."
                    )
            else:
                weight = self.weight

            if "feat" in self.precomputed_feat and self.training:
                rst = self.precomputed_feat["feat"]
            else:
                # aggregate first then mult W
                graph.srcdata["h"] = feat_src
                graph.dstdata["h"] = graph.srcdata["h"]
                graph.update_all(aggregate_fn, fn.sum(msg="m", out="neigh"))
                rst = graph.dstdata["h"] + graph.dstdata["neigh"]

                if self._norm in ["right", "both"]:
                    degs = graph.in_degrees().to(feat_dst).clamp(min=1)
                    if self._norm == "both":
                        norm = torch.pow(degs, -0.5)
                    else:
                        norm = 1.0 / (degs + 1)
                    shp = norm.shape + (1,) * (feat_dst.dim() - 1)
                    norm = torch.reshape(norm, shp)
                    rst = rst * norm
                if self.training:
                    self.precomputed_feat["feat"] = rst

            if weight is not None:
                    rst = torch.matmul(rst, weight)

            if self.bias is not None:
                rst = rst + self.bias

            if self._activation is not None:
                rst = self._activation(rst)

            return rst
