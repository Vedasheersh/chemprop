import json
import numpy as np
import tqdm, random
import torch, math
import torch.utils.data as data
import torch.nn.functional as F
import torch_geometric
import torch_cluster
import ipdb

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

class SinusoidalPositionalEncoding(torch.nn.Module):
    def __init__(self, channels, pos_embed_freq_inv = 2000):
        super().__init__()
        channels = int(ceil(channels / 2) * 2)
        inv_freq = 1.0 / (pos_embed_freq_inv ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, tensor):
        sin_inp_x = torch.einsum("...i,j->...ij", tensor, self.inv_freq)
        emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1)
        return emb_x


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

class EGNN_Dataset(data.Dataset):
    '''
    A map-syle `torch.utils.data.Dataset` which transforms JSON/dictionary-style
    protein structures into featurized protein graphs 
    
    Adapted mainly from https://github.com/jgreener64/progres/blob/main/progres/progres.py
    
    :param data_list: JSON/dictionary-style protein dataset as described in README.md.
    :param contact_dist: distance cutoff for contacts in angstrom
    :param device: if "cuda", will do preprocessing on the GPU
    '''
    def __init__(self, data_list, pos_embed_dim = 64, contact_dist = 10.0, device="cpu"):
        
        super(EGNN_Dataset, self).__init__()
        
        self.data_list = data_list
        self.device = device
        self.contact_dist = contact_dist
        self.protein_max_length = 1024
        self.node_counts = [len(e['seq']) for e in data_list]
        self.pos_embedder = SinusoidalPositionalEncoding(pos_embed_dim)

        
    def __len__(self): return len(self.data_list)
    
    def __getitem__(self, i): return self._featurize_as_graph(self.data_list[i])
    
    def _featurize_as_graph(self, protein):
        name = protein['name']
        with torch.no_grad():
            coords = torch.as_tensor(protein['coords'], 
                                     device=self.device, dtype=torch.float32)   
            
            coords = coords[:self.protein_max_length, 1, :] #CA only
            n_res = len(coords)
            
            dmap = torch.cdist(coords.unsqueeze(0), coords.unsqueeze(0),
                               compute_mode="donot_use_mm_for_euclid_dist")
            contacts = (dmap <= self.contact_dist).squeeze(0)
            edge_index = contacts.to_sparse().indices()

            degrees = contacts.sum(dim=0)
            norm_degrees = (degrees / degrees.max()).unsqueeze(1)
            term_features = [[0.0, 0.0] for _ in range(n_res)]
            term_features[ 0][0] = 1.0
            term_features[-1][1] = 1.0
            term_features = torch.tensor(term_features)

            # The tau torsion angle is between 4 consecutive Cα atoms, we assign it to the second Cα
            # This feature breaks mirror invariance
            vec_ab = coords[1:-2] - coords[ :-3]
            vec_bc = coords[2:-1] - coords[1:-2]
            vec_cd = coords[3:  ] - coords[2:-1]
            cross_ab_bc = torch.cross(vec_ab, vec_bc, dim=1)
            cross_bc_cd = torch.cross(vec_bc, vec_cd, dim=1)
            taus = torch.atan2(
                (torch.cross(cross_ab_bc, cross_bc_cd, dim=1) * normalize(vec_bc, dim=1)).sum(dim=1),
                (cross_ab_bc * cross_bc_cd).sum(dim=1),
            )
            taus_pad = torch.cat((
                torch.tensor([0.0]),
                taus / torch.pi, # Convert to range -1 -> 1
                torch.tensor([0.0, 0.0]),
            )).unsqueeze(1)

            pos_embed = self.pos_embedder(torch.arange(1, n_res + 1))
            x = torch.cat((norm_degrees, term_features, taus_pad, pos_embed), dim=1)
            
        data = torch_geometric.data.Data(x=x, edge_index=edge_index, coords=coords)
        
        return data