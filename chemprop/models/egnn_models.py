# Fast protein structure searching using structure graph embeddings

# biopython imported in a function
import torch
from torch.nn import Dropout, Identity, Linear, Sequential, SiLU
from torch.nn.functional import normalize
from torch.utils.data import Dataset
from torch_scatter import scatter
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from einops import rearrange
from math import ceil
import os
import pkg_resources
import sys
from urllib import request


def batched_index_select(values, indices, dim=1):
    value_dims = values.shape[(dim + 1):]
    values_shape, indices_shape = map(lambda t: list(t.shape), (values, indices))
    indices = indices[(..., *((None,) * len(value_dims)))]
    indices = indices.expand(*((-1,) * len(indices_shape)), *value_dims)
    value_expand_len = len(indices_shape) - (dim + 1)
    values = values[(*((slice(None),) * dim), *((None,) * value_expand_len), ...)]

    value_expand_shape = [-1] * len(values.shape)
    expand_slice = slice(dim, (dim + value_expand_len))
    value_expand_shape[expand_slice] = indices.shape[expand_slice]
    values = values.expand(*value_expand_shape)

    dim += value_expand_len
    return values.gather(dim, indices)


# From https://github.com/lucidrains/egnn-pytorch
class EGNN(torch.nn.Module):
    def __init__(
        self,
        dim,
        hidden_edge_dim,
        m_dim=16,
        dropout=0.0,
        init_eps=1e-3,
    ):
        super().__init__()
        edge_input_dim = (dim * 2) + 1
        dropout = Dropout(dropout) if dropout > 0 else Identity()

        self.edge_mlp = Sequential(
            Linear(edge_input_dim, hidden_edge_dim),
            dropout,
            SiLU(),
            Linear(hidden_edge_dim, m_dim),
            SiLU(),
        )

        self.node_mlp = Sequential(
            Linear(dim + m_dim, dim * 2),
            dropout,
            SiLU(),
            Linear(dim * 2, dim),
        )

        self.init_eps = init_eps
        self.apply(self.init_)

    def init_(self, module):
        if type(module) in {Linear}:
            torch.nn.init.normal_(module.weight, std=self.init_eps)

    def forward(self, feats, coors, mask, adj_mat):
        b, n, d, device = *feats.shape, feats.device

        rel_coors = rearrange(coors, "b i d -> b i () d") - rearrange(coors, "b j d -> b () j d")
        rel_dist = (rel_coors ** 2).sum(dim=-1, keepdim=True)

        i = j = n
        ranking = rel_dist[..., 0].clone()
        rank_mask = mask[:, :, None] * mask[:, None, :]
        ranking.masked_fill_(~rank_mask, 1e5)

        num_nearest = int(adj_mat.float().sum(dim=-1).max().item())
        valid_radius = 0

        self_mask = rearrange(torch.eye(n, device=device, dtype=torch.bool), "i j -> () i j")

        adj_mat = adj_mat.masked_fill(self_mask, False)
        ranking.masked_fill_(self_mask, -1.)
        ranking.masked_fill_(adj_mat, 0.)

        nbhd_ranking, nbhd_indices = ranking.topk(num_nearest, dim=-1, largest=False)
        nbhd_mask = nbhd_ranking <= valid_radius

        rel_coors = batched_index_select(rel_coors, nbhd_indices, dim=2)
        rel_dist = batched_index_select(rel_dist, nbhd_indices, dim=2)

        j = num_nearest
        feats_j = batched_index_select(feats, nbhd_indices, dim=1)
        feats_i = rearrange(feats, "b i d -> b i () d")
        feats_i, feats_j = torch.broadcast_tensors(feats_i, feats_j)

        edge_input = torch.cat((feats_i, feats_j, rel_dist), dim=-1)
        m_ij = self.edge_mlp(edge_input)

        mask_i = rearrange(mask, "b i -> b i ()")
        mask_j = batched_index_select(mask, nbhd_indices, dim = 1)
        mask = (mask_i * mask_j) & nbhd_mask

        m_ij_mask = rearrange(mask, "... -> ... ()")
        m_ij = m_ij.masked_fill(~m_ij_mask, 0.)
        m_i = m_ij.sum(dim=-2)

        node_mlp_input = torch.cat((feats, m_i), dim = -1)
        node_out = self.node_mlp(node_mlp_input) + feats

        return node_out, coors

# Based on https://github.com/vgsatorras/egnn/blob/main/qm9/models.py
class EGNN_Model(torch.nn.Module):
    def __init__(self, 
                 n_layers = 4, 
                 hidden_dim = 128, 
                 hidden_egnn_dim = 128, 
                 embedding_size = 128, 
                 hidden_edge_dim = 128, 
                 dropout = 0.0, 
                 dropout_final = 0.0):
        super().__init__()
        n_features = 68 + 3 + 20 + 3
        self.node_enc = Linear(n_features, hidden_dim)
        self.layers = torch.nn.ModuleList()
        for i in range(n_layers):
            self.layers.append(EGNN(
                dim=hidden_dim,
                hidden_edge_dim = hidden_edge_dim,
                m_dim=hidden_egnn_dim,
                dropout=dropout,
            ))
        self.node_dec = torch.nn.Sequential(
            Linear(hidden_dim, hidden_dim),
            Dropout(dropout) if dropout > 0 else Identity(),
            SiLU(),
            Linear(hidden_dim, hidden_dim),
        )
        self.graph_dec = torch.nn.Sequential(
            Linear(hidden_dim, hidden_dim),
            Dropout(dropout) if dropout > 0 else Identity(),
            SiLU(),
            Dropout(dropout_final) if dropout_final > 0 else Identity(),
            Linear(hidden_dim, embedding_size),
        )

    def forward(self, data):
        device = data.x.device
        feats, coords = data.x.unsqueeze(0), data.coords.unsqueeze(0)
        adj_mat = torch.sparse_coo_tensor(
            indices=data.edge_index,
            values=torch.tensor([1] * data.edge_index.size(1), device=device),
            size=(data.num_nodes, data.num_nodes),
        ).to_dense().bool().unsqueeze(0)
        mask = torch.ones(1, data.num_nodes, dtype=torch.bool, device=device)
        feats = self.node_enc(feats)
        for layer in self.layers:
            feats, coords = layer(feats, coords, mask, adj_mat)
        feats = self.node_dec(feats)
        batch = torch.tensor([0] * data.num_nodes, device=device) if data.batch is None else data.batch
        graph_feats = scatter(feats.squeeze(0), batch, dim=0, reduce="sum")
        out = self.graph_dec(graph_feats)
        
        return normalize(out, dim=1)
    