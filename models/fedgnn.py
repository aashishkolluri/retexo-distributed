from torch_geometric.nn import MLP 
import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import Embedding, ModuleDict
from dgl.nn.pytorch import HeteroGraphConv, GraphConv, SAGEConv
import torch.nn.functional as F


class FedGNNModel(nn.Module):
    
    def __init__(self, num_users, num_items, hidden_dim, history_len=100, dropout=0.5):
        super().__init__()
        
        self.num_users = num_users
        self.num_items = num_items
        
        self.user_embedding = nn.Embedding(num_users, hidden_dim)
        self.item_embedding = nn.Embedding(num_items, hidden_dim)
        
        self.history_len = history_len
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)

        # Define dense layers before GCN
        self.user_dense = nn.Sequential(nn.Linear(23, hidden_dim), nn.ReLU())
        self.item_dense = nn.Sequential(nn.Linear(319, hidden_dim), nn.ReLU())

        # Define a hetero graph conv with two types of edges: 'user-item' and 'item-user'
        self.hetero_convs = nn.ModuleList()
        self.hetero_convs.append(HeteroGraphConv({
            'user-movie': SAGEConv(hidden_dim, hidden_dim, 'mean'),
            'movie-user': SAGEConv(hidden_dim, hidden_dim, 'mean')
        }))
        self.hetero_convs.append(HeteroGraphConv({
            'movie-user': SAGEConv(hidden_dim, hidden_dim, 'mean'),
            'user-movie': SAGEConv(hidden_dim, hidden_dim, 'mean')
        }))

        # self.history_dense = nn.Linear(hidden_dim, hidden_dim)
        self.output_layer = MLP(
            2 * hidden_dim,
            # hidden_channels=32,
            out_channels=1,
            num_layers=1,
            # dropout=0.5,
        )
        # torch.nn.Sequential(
        #     nn.Linear(2 * hidden_dim, 1),
        #     nn.ReLU(),
        #     # nn.Linear(hidden_dim // 2, 1),
        # )
        # 

    def forward(self, g, user_feats, item_feats, edge_mask, etype):
        # Get user and item embeddings
        user_emb = self.user_dense(user_feats)#.mean(1)
        item_emb = self.item_dense(item_feats)


        # Apply the heterogeneous GCN
        h =  {'user': user_emb, 'movie': item_emb}
        for conv in self.hetero_convs:
            h = conv(g, h)
            h = {k: F.relu(v) for k, v in h.items()}
        
        user_emb = h["user"]
        item_emb = h["movie"]
        
        edge_ids = torch.nonzero(edge_mask, as_tuple=True)[0].to("cuda")
        src_ids, dst_ids = g.find_edges(edge_ids, etype=etype)

        final_emb = torch.cat([user_emb[src_ids] , item_emb[dst_ids] ], dim=-1)
        out = self.output_layer(final_emb)
        # out = torch.relu(out)

        u_emb_copy = user_emb.detach()
        i_emb_copy = item_emb.detach()
        return out, u_emb_copy, i_emb_copy
    
    def get_nth_layer(self, n):
        output_layer = MLP(
            2 * self.hidden_dim,
            # hidden_channels=32,
            out_channels=1,
            num_layers=1,
        )
        # torch.nn.Sequential(
        #     nn.Linear(2 * self.hidden_dim, 1),
        #     nn.ReLU(),
        #     # nn.Linear(self.hidden_dim // 2, 1),
        # )


        
        if n == 0:
            return FirstFedModel(self.user_embedding, self.item_dense, output_layer)
        if n == 1:
            return IntermediateFedModel(self.hetero_convs[0], output_layer)
        if n == 2:
            return IntermediateFedModel(self.hetero_convs[1], self.output_layer)
        
        raise IndexError("unkown layer index")
    
class FirstFedModel(nn.Module):
    def __init__(self, u_embedding, i_embedding, output_layer):
        super().__init__()
        self.user_embedding = u_embedding
        self.item_embedding = i_embedding
        self.output_layer = output_layer
    
    def forward(self, g, user_buckets, item_buckets, edge_mask, etype): 
        user_emb = self.user_embedding(user_buckets).mean(dim=1)#.squeeze(1)
        item_emb = self.item_embedding(item_buckets)#.mean(dim=1)#.squeeze(1)
    
        edge_ids = torch.nonzero(edge_mask, as_tuple=True)[0]
        src_ids, dst_ids = g.find_edges(edge_ids, etype=etype)
        
        final_emb = torch.cat([user_emb[src_ids], item_emb[dst_ids]], dim=1)
    
        out = self.output_layer(final_emb)
        # out = torch.relu(out)

        u_emb_copy = user_emb.detach()
        i_emb_copy = item_emb.detach()
        return out, u_emb_copy, i_emb_copy
    
class IntermediateFedModel(nn.Module):
    
    def __init__(self, conv, output_layer):
        super().__init__()
    
        self.hetero_gcn = conv
        self.output_layer = output_layer
        
    def forward(self, g, user_emb, item_emb, edge_mask, etype):
        # g.nodes['user'].data['h'] = user_emb
        # g.nodes['item'].data['h'] = item_emb
        
        h = self.hetero_gcn(g, {'user': user_emb, 'movie': item_emb})
        # g.ndata.update(h)
        # g.ndata['h'] = h
        user_emb = h["user"]
        item_emb = h["movie"]
        
        edge_ids = torch.nonzero(edge_mask, as_tuple=True)[0].to("cuda")
        src_ids, dst_ids = g.find_edges(edge_ids, etype=etype)

        final_emb = torch.cat([user_emb[src_ids] , item_emb[dst_ids] ], dim=-1)
        out = self.output_layer(final_emb)
        # out = torch.relu(out)

        u_emb_copy = user_emb.detach()
        i_emb_copy = item_emb.detach()
        return out, u_emb_copy, i_emb_copy