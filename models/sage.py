"""Custom SAGE layer"""

import math
import dgl # type: ignore
import dgl.function as fn # type: ignore
import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomSAGEConv(dgl.nn.SAGEConv):
    """Custom SAGEConv layer"""
    def __init__(
        self,
        in_feats,
        out_feats,
        aggregator_type,
        feat_drop=0.0,
        bias=True,
        norm=None,
        activation=None,
        pooling_size=512,
    ):
        super().__init__(in_feats, out_feats, aggregator_type, feat_drop, bias, norm, activation)
        if self._aggre_type == "pool":
            self.fc_self = None
            self.fc_neigh = None
            self.fc_pool = nn.Linear(in_feats, pooling_size, bias=True)
            self.fc_int = nn.Linear(pooling_size+in_feats, out_feats, bias=True)
        elif self._aggre_type == "mean":
            self.precomputed_feat = {}
            self.fc_int = nn.Linear(2*in_feats, out_feats, bias=True)
            # initialize fc_int
            # nn.init.xavier_uniform_(self.fc_int.weight, gain=nn.init.calculate_gain("relu"))
            self.fc_self = None
            self.fc_neigh = None
            self._reset_parameters(self.fc_int)


    def _reset_parameters(self, fc_int):
        stdv = 1. / math.sqrt(fc_int.weight.size(1))
        fc_int.weight.data.uniform_(-stdv, stdv)
        if fc_int.bias is not None:
            fc_int.bias.data.uniform_(-stdv, stdv)


    def forward(self, graph, feat, edge_weight=None):
        """Change how pooling works"""
        if not self._aggre_type in ["pool", "mean"]:
            return super().forward(graph, feat, edge_weight)
        elif self._aggre_type == "pool":
            with graph.local_scope():
                if isinstance(feat, tuple):
                    feat_src = self.feat_drop(feat[0])
                    feat_dst = self.feat_drop(feat[1])
                else:
                    feat_src = feat_dst = self.feat_drop(feat)
                    if graph.is_block:
                        feat_dst = feat_src[: graph.number_of_dst_nodes()]
                msg_fn = fn.copy_u("h", "m")
                if edge_weight is not None:
                    assert edge_weight.shape[0] == graph.num_edges()
                    graph.edata["_edge_weight"] = edge_weight
                    msg_fn = fn.u_mul_e("h", "_edge_weight", "m")

                h_self = feat_dst

                # Handle the case of graphs without edges
                if graph.num_edges() == 0:
                    graph.dstdata["neigh"] = torch.zeros(
                        feat_dst.shape[0], self._in_src_feats
                    ).to(feat_dst)

                graph.srcdata["h"] = F.relu(self.fc_pool(feat_src))
                graph.update_all(msg_fn, fn.max("m", "neigh"))
                h_neigh = graph.dstdata["neigh"]

                rst = self.fc_int(torch.cat([h_self, h_neigh], dim=1))

                # activation
                if self.activation is not None:
                    rst = self.activation(rst)
                # normalization
                if self.norm is not None:
                    rst = self.norm(rst)
                return rst
        elif self._aggre_type == "mean":
             with graph.local_scope():
                if isinstance(feat, tuple):
                    feat_src = self.feat_drop(feat[0])
                    feat_dst = self.feat_drop(feat[1])
                else:
                    feat_src = feat_dst = self.feat_drop(feat)
                    if graph.is_block:
                        feat_dst = feat_src[: graph.number_of_dst_nodes()]
                msg_fn = fn.copy_u("h", "m")
                if edge_weight is not None:
                    assert edge_weight.shape[0] == graph.num_edges()
                    graph.edata["_edge_weight"] = edge_weight
                    msg_fn = fn.u_mul_e("h", "_edge_weight", "m")

                h_self = feat_dst

                # Handle the case of graphs without edges
                if graph.num_edges() == 0:
                    graph.dstdata["neigh"] = torch.zeros(
                        feat_dst.shape[0], self._in_src_feats
                    ).to(feat_dst)

                if "feat" in self.precomputed_feat and self.training:
                    h_neigh = self.precomputed_feat["feat"]
                    # h_neigh = self.fc_neigh(saved_nei_h)
                else:
                    graph.srcdata["h"] = feat_src
                    graph.update_all(msg_fn, fn.mean("m", "neigh"))
                    h_neigh = graph.dstdata["neigh"]
                    if self.training:
                        self.precomputed_feat["feat"] = h_neigh
                    # h_neigh = self.fc_neigh(h_neigh)

                rst = self.fc_int(torch.cat([h_self, h_neigh], dim=1))

                # activation
                if self.activation is not None:
                    rst = self.activation(rst)
                # normalization
                if self.norm is not None:
                    rst = self.norm(rst)
                return rst