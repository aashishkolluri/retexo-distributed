""""Model from Relbench"""


from typing import Any, Dict, List

import torch
from torch import Tensor
from torch.nn import Embedding, ModuleDict
from torch_frame.data.stats import StatType
from torch_geometric.data import HeteroData
from torch_geometric.nn import MLP 
from torch_geometric.typing import NodeType

from relbench.external.nn import HeteroEncoder, HeteroTemporalEncoder

from models.rel_sage import HeteroGraphSAGE


class RelModel(torch.nn.Module):
    def __init__(
        self,
        data: HeteroData,
        col_stats_dict: Dict[str, dict[str, dict[StatType, Any]]],
        num_layers: int,
        channels: int,
        out_channels: int,
        aggr: str,
        norm: str,
        # List of node types to add shallow embeddings to input
        shallow_list: List[NodeType] = [],
    ):
        super().__init__()
        
        self.chan = channels
        self.out_chan = out_channels
        self.m_norm = norm
        self.num_layers = num_layers
        
        self.encoder = HeteroEncoder(
            channels=channels,
            node_to_col_names_dict={
                node_type: data[node_type].tf.col_names_dict
                for node_type in data.node_types
            },
            node_to_col_stats=col_stats_dict,
        )
        
        self.temporal_encoder = HeteroTemporalEncoder(
            node_types=[
                node_type for node_type in data.node_types if "time" in data[node_type]
            ],
            channels=channels,
        )
        
        # Our very own retexo_x_relbench SAGE
        self.gnn = HeteroGraphSAGE(
            node_types=data.node_types,
            edge_types=data.edge_types,
            channels=channels,
            aggr=aggr,
            num_layers=num_layers,
        )
        
        self.head = MLP(
            channels,
            out_channels=out_channels,
            norm=norm,
            num_layers=1,
        )
        self.embedding_dict = ModuleDict(
            {
                node: Embedding(data.num_nodes_dict[node], channels)
                for node in shallow_list
            }
        )
        for embedding in self.embedding_dict.values():
            torch.nn.init.normal_(embedding.weight, std=0.1)
            

    def forward(
        self,
        batch: HeteroData,
        entity_table: NodeType,
    ) -> Tensor:
        seed_time = batch[entity_table].seed_time
        x_dict = self.encoder(batch.tf_dict)

        rel_time_dict = self.temporal_encoder(
            seed_time, batch.time_dict, batch.batch_dict
        )
        
        for node_type, rel_time in rel_time_dict.items():
            x_dict[node_type] = x_dict[node_type] + rel_time


        # shallow embeddings. Ignore for now
        # for node_type, embedding in self.embedding_dict.items():
        #     x_dict[node_type] = x_dict[node_type] + embedding(batch[node_type].n_id)
            
        x_dict = self.gnn(
            x_dict,
            batch.edge_index_dict,
            batch.num_sampled_nodes_dict,
            batch.num_sampled_edges_dict,
        )
        
        output = self.head(x_dict[entity_table][: seed_time.size(0)])        
        x_copy = {key: value.detach() for key, value in x_dict.items()}
        return output, x_copy
    
    # def get_emebeddings(
    #     self,
    #     batch: HeteroData,
    #     entity_table: NodeType,
    # ) -> Dict[NodeType, Tensor]:
    #     seed_time = batch[entity_table].seed_time
    #     x_dict = self.encoder(batch.tf_dict)

    #     rel_time_dict = self.temporal_encoder(
    #         seed_time, batch.time_dict, batch.batch_dict
    #     )
        
    #     for node_type, rel_time in rel_time_dict.items():
    #         x_dict[node_type] = x_dict[node_type] + rel_time

    #     for node_type, embedding in self.embedding_dict.items():
    #         x_dict[node_type] = x_dict[node_type] + embedding(batch[node_type].n_id)
         
    #     return x_dict

    def get_nth_layer(
        self,
        n: int,
    ):
        mlp = MLP(
            self.chan,
            out_channels=self.out_chan,
            norm=self.m_norm,
            num_layers=1,
        )
                
        if n == 0:
            return FirstLayerRelModel(
                self.temporal_encoder,
                self.encoder,
                mlp,
                self.embedding_dict
            )
        conv, norm = self.gnn.get_nth_layer(n-1)

        if n == self.num_layers:
            mlp = self.head
         
        return IntermediateRelModel(conv, norm, mlp)
    
    
class FirstLayerRelModel(torch.nn.Module):
    """Model with encoders, one convolution layer, and MLP"""

    def __init__(self, temporal_encoder, encoder, mlp, embedding_dict) -> None:
        super().__init__()
        self.temporal_encoder = temporal_encoder
        self.encoder = encoder
        self.mlp = mlp
        self.embedding_dict = embedding_dict

    def forward(
        self, 
        batch: HeteroData,
        entity_table: NodeType,
    ) -> torch.Tensor:
        """Forward pass"""
        seed_time = batch[entity_table].seed_time
        x_dict = self.encoder(batch.tf_dict)

        rel_time_dict = self.temporal_encoder(
            seed_time, batch.time_dict, batch.batch_dict
        )
        
        for node_type, rel_time in rel_time_dict.items():
            x_dict[node_type] = x_dict[node_type] + rel_time

        for node_type, embedding in self.embedding_dict.items():
            x_dict[node_type] = x_dict[node_type] + embedding(batch[node_type].n_id)
        
        output = self.mlp(x_dict[entity_table][: seed_time.size(0)])    
        x_copy = {key: value.detach() for key, value in x_dict.items()}    
        return output, x_copy
    
class IntermediateRelModel(torch.nn.Module):
    """Model with one convolution layer and one MLP layer"""

    def __init__(self, conv, norm, mlp) -> None:
        super().__init__()
        self.conv = conv
        self.norm = norm
        self.mlp = mlp

    def forward(
        self, 
        seed_time,
        x_dict: Dict[NodeType, Tensor],
        edge_index_dict: Dict[NodeType, Tensor],
        entity_table: NodeType,
    ) -> torch.Tensor:
        """Forward pass"""
        x_dict = self.conv(x_dict, edge_index_dict)
        x_dict = {key: self.norm[key](x) for key, x in x_dict.items()}
        x_dict = {key: x.relu() for key, x in x_dict.items()}
        
        output = self.mlp(x_dict[entity_table][: seed_time.size(0)])  
        x_copy = {key: value.detach() for key, value in x_dict.items()}
        return output, x_copy
