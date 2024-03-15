import ipdb
import torch, functools
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter_add

def tuple_sum(*args):
    '''
    Sums any number of tuples (s, V) elementwise.
    '''
    return tuple(map(sum, zip(*args)))

def tuple_cat(*args, dim=-1):
    '''
    Concatenates any number of tuples (s, V) elementwise.
    
    :param dim: dimension along which to concatenate when viewed
                as the `dim` index for the scalar-channel tensors.
                This means that `dim=-1` will be applied as
                `dim=-2` for the vector-channel tensors.
    '''
    dim %= len(args[0][0].shape)
    s_args, v_args = list(zip(*args))
    return torch.cat(s_args, dim=dim), torch.cat(v_args, dim=dim)

def tuple_index(x, idx):
    '''
    Indexes into a tuple (s, V) along the first dimension.
    
    :param idx: any object which can be used to index into a `torch.Tensor`
    '''
    return x[0][idx], x[1][idx]

def randn(n, dims, device="cpu"):
    '''
    Returns random tuples (s, V) drawn elementwise from a normal distribution.
    
    :param n: number of data points
    :param dims: tuple of dimensions (n_scalar, n_vector)
    
    :return: (s, V) with s.shape = (n, n_scalar) and
             V.shape = (n, n_vector, 3)
    '''
    return torch.randn(n, dims[0], device=device), \
            torch.randn(n, dims[1], 3, device=device)

def _norm_no_nan(x, axis=-1, keepdims=False, eps=1e-8, sqrt=True):
    '''
    L2 norm of tensor clamped above a minimum value `eps`.
    
    :param sqrt: if `False`, returns the square of the L2 norm
    '''
    out = torch.clamp(torch.sum(torch.square(x), axis, keepdims), min=eps)
    return torch.sqrt(out) if sqrt else out

def _split(x, nv):
    '''
    Splits a merged representation of (s, V) back into a tuple. 
    Should be used only with `_merge(s, V)` and only if the tuple 
    representation cannot be used.
    
    :param x: the `torch.Tensor` returned from `_merge`
    :param nv: the number of vector channels in the input to `_merge`
    '''
    v = torch.reshape(x[..., -3*nv:], x.shape[:-1] + (nv, 3))
    s = x[..., :-3*nv]
    return s, v

def _merge(s, v):
    '''
    Merges a tuple (s, V) into a single `torch.Tensor`, where the
    vector channels are flattened and appended to the scalar channels.
    Should be used only if the tuple representation cannot be used.
    Use `_split(x, nv)` to reverse.
    '''
    v = torch.reshape(v, v.shape[:-2] + (3*v.shape[-2],))
    return torch.cat([s, v], -1)

class GVP(nn.Module):
    '''
    Geometric Vector Perceptron. See manuscript and README.md
    for more details.
    
    :param in_dims: tuple (n_scalar, n_vector)
    :param out_dims: tuple (n_scalar, n_vector)
    :param h_dim: intermediate number of vector channels, optional
    :param activations: tuple of functions (scalar_act, vector_act)
    :param vector_gate: whether to use vector gating.
                        (vector_act will be used as sigma^+ in vector gating if `True`)
    '''
    def __init__(self, in_dims, out_dims, h_dim=None,
                 activations=(F.relu, torch.sigmoid), vector_gate=False):
        super(GVP, self).__init__()
        self.si, self.vi = in_dims
        self.so, self.vo = out_dims
        self.vector_gate = vector_gate
        if self.vi: 
            self.h_dim = h_dim or max(self.vi, self.vo) 
            self.wh = nn.Linear(self.vi, self.h_dim, bias=False)
            self.ws = nn.Linear(self.h_dim + self.si, self.so)
            if self.vo:
                self.wv = nn.Linear(self.h_dim, self.vo, bias=False)
                if self.vector_gate: self.wsv = nn.Linear(self.so, self.vo)
        else:
            self.ws = nn.Linear(self.si, self.so)
        
        self.scalar_act, self.vector_act = activations
        self.dummy_param = nn.Parameter(torch.empty(0))
        
    def forward(self, x):
        '''
        :param x: tuple (s, V) of `torch.Tensor`, 
                  or (if vectors_in is 0), a single `torch.Tensor`
        :return: tuple (s, V) of `torch.Tensor`,
                 or (if vectors_out is 0), a single `torch.Tensor`
        '''
        if self.vi:
            s, v = x
            v = torch.transpose(v, -1, -2)
            vh = self.wh(v)    
            vn = _norm_no_nan(vh, axis=-2)
            s = self.ws(torch.cat([s, vn], -1))
            if self.vo: 
                v = self.wv(vh) 
                v = torch.transpose(v, -1, -2)
                if self.vector_gate: 
                    if self.vector_act:
                        gate = self.wsv(self.vector_act(s))
                    else:
                        gate = self.wsv(s)
                    v = v * torch.sigmoid(gate).unsqueeze(-1)
                elif self.vector_act:
                    v = v * self.vector_act(
                        _norm_no_nan(v, axis=-1, keepdims=True))
        else:
            s = self.ws(x)
            if self.vo:
                v = torch.zeros(s.shape[0], self.vo, 3,
                                device=self.dummy_param.device)
        if self.scalar_act:
            s = self.scalar_act(s)
        
        return (s, v) if self.vo else s

class _VDropout(nn.Module):
    '''
    Vector channel dropout where the elements of each
    vector channel are dropped together.
    '''
    def __init__(self, drop_rate):
        super(_VDropout, self).__init__()
        self.drop_rate = drop_rate
        self.dummy_param = nn.Parameter(torch.empty(0))

    def forward(self, x):
        '''
        :param x: `torch.Tensor` corresponding to vector channels
        '''
        device = self.dummy_param.device
        if not self.training:
            return x
        mask = torch.bernoulli(
            (1 - self.drop_rate) * torch.ones(x.shape[:-1], device=device)
        ).unsqueeze(-1)
        x = mask * x / (1 - self.drop_rate)
        return x

class Dropout(nn.Module):
    '''
    Combined dropout for tuples (s, V).
    Takes tuples (s, V) as input and as output.
    '''
    def __init__(self, drop_rate):
        super(Dropout, self).__init__()
        self.sdropout = nn.Dropout(drop_rate)
        self.vdropout = _VDropout(drop_rate)

    def forward(self, x):
        '''
        :param x: tuple (s, V) of `torch.Tensor`,
                  or single `torch.Tensor` 
                  (will be assumed to be scalar channels)
        '''
        if type(x) is torch.Tensor:
            return self.sdropout(x)
        s, v = x
        return self.sdropout(s), self.vdropout(v)

class LayerNorm(nn.Module):
    '''
    Combined LayerNorm for tuples (s, V).
    Takes tuples (s, V) as input and as output.
    '''
    def __init__(self, dims):
        super(LayerNorm, self).__init__()
        self.s, self.v = dims
        self.scalar_norm = nn.LayerNorm(self.s)
        
    def forward(self, x):
        '''
        :param x: tuple (s, V) of `torch.Tensor`,
                  or single `torch.Tensor` 
                  (will be assumed to be scalar channels)
        '''
        if not self.v:
            return self.scalar_norm(x)
        s, v = x
        vn = _norm_no_nan(v, axis=-1, keepdims=True, sqrt=False)
        vn = torch.sqrt(torch.mean(vn, dim=-2, keepdim=True))
        return self.scalar_norm(s), v / vn

class GVPConv(MessagePassing):
    '''
    Graph convolution / message passing with Geometric Vector Perceptrons.
    Takes in a graph with node and edge embeddings,
    and returns new node embeddings.
    
    This does NOT do residual updates and pointwise feedforward layers
    ---see `GVPConvLayer`.
    
    :param in_dims: input node embedding dimensions (n_scalar, n_vector)
    :param out_dims: output node embedding dimensions (n_scalar, n_vector)
    :param edge_dims: input edge embedding dimensions (n_scalar, n_vector)
    :param n_layers: number of GVPs in the message function
    :param module_list: preconstructed message function, overrides n_layers
    :param aggr: should be "add" if some incoming edges are masked, as in
                 a masked autoregressive decoder architecture, otherwise "mean"
    :param activations: tuple of functions (scalar_act, vector_act) to use in GVPs
    :param vector_gate: whether to use vector gating.
                        (vector_act will be used as sigma^+ in vector gating if `True`)
    '''
    def __init__(self, in_dims, out_dims, edge_dims,
                 n_layers=3, module_list=None, aggr="mean", 
                 activations=(F.relu, torch.sigmoid), vector_gate=False):
        super(GVPConv, self).__init__(aggr=aggr)
        self.si, self.vi = in_dims
        self.so, self.vo = out_dims
        self.se, self.ve = edge_dims
        
        GVP_ = functools.partial(GVP, 
                activations=activations, vector_gate=vector_gate)
        
        module_list = module_list or []
        if not module_list:
            if n_layers == 1:
                module_list.append(
                    GVP_((2*self.si + self.se, 2*self.vi + self.ve), 
                        (self.so, self.vo), activations=(None, None)))
            else:
                module_list.append(
                    GVP_((2*self.si + self.se, 2*self.vi + self.ve), out_dims)
                )
                for i in range(n_layers - 2):
                    module_list.append(GVP_(out_dims, out_dims))
                module_list.append(GVP_(out_dims, out_dims,
                                       activations=(None, None)))
        self.message_func = nn.Sequential(*module_list)

    def forward(self, x, edge_index, edge_attr):
        '''
        :param x: tuple (s, V) of `torch.Tensor`
        :param edge_index: array of shape [2, n_edges]
        :param edge_attr: tuple (s, V) of `torch.Tensor`
        '''
        x_s, x_v = x
        message = self.propagate(edge_index, 
                    s=x_s, v=x_v.reshape(x_v.shape[0], 3*x_v.shape[1]),
                    edge_attr=edge_attr)
        return _split(message, self.vo) 

    def message(self, s_i, v_i, s_j, v_j, edge_attr):
        v_j = v_j.view(v_j.shape[0], v_j.shape[1]//3, 3)
        v_i = v_i.view(v_i.shape[0], v_i.shape[1]//3, 3)
        message = tuple_cat((s_j, v_j), edge_attr, (s_i, v_i))
        message = self.message_func(message)
        return _merge(*message)


class GVPConvLayer(nn.Module):
    '''
    Full graph convolution / message passing layer with 
    Geometric Vector Perceptrons. Residually updates node embeddings with
    aggregated incoming messages, applies a pointwise feedforward 
    network to node embeddings, and returns updated node embeddings.
    
    To only compute the aggregated messages, see `GVPConv`.
    
    :param node_dims: node embedding dimensions (n_scalar, n_vector)
    :param edge_dims: input edge embedding dimensions (n_scalar, n_vector)
    :param n_message: number of GVPs to use in message function
    :param n_feedforward: number of GVPs to use in feedforward function
    :param drop_rate: drop probability in all dropout layers
    :param autoregressive: if `True`, this `GVPConvLayer` will be used
           with a different set of input node embeddings for messages
           where src >= dst
    :param activations: tuple of functions (scalar_act, vector_act) to use in GVPs
    :param vector_gate: whether to use vector gating.
                        (vector_act will be used as sigma^+ in vector gating if `True`)
    '''
    def __init__(self, node_dims, edge_dims,
                 n_message=3, n_feedforward=2, drop_rate=.1,
                 autoregressive=False, 
                 activations=(F.relu, torch.sigmoid), vector_gate=False):
        
        super(GVPConvLayer, self).__init__()
        self.conv = GVPConv(node_dims, node_dims, edge_dims, n_message,
                           aggr="add" if autoregressive else "mean",
                           activations=activations, vector_gate=vector_gate)
        GVP_ = functools.partial(GVP, 
                activations=activations, vector_gate=vector_gate)
        self.norm = nn.ModuleList([LayerNorm(node_dims) for _ in range(2)])
        self.dropout = nn.ModuleList([Dropout(drop_rate) for _ in range(2)])

        ff_func = []
        if n_feedforward == 1:
            ff_func.append(GVP_(node_dims, node_dims, activations=(None, None)))
        else:
            hid_dims = 4*node_dims[0], 2*node_dims[1]
            ff_func.append(GVP_(node_dims, hid_dims))
            for i in range(n_feedforward-2):
                ff_func.append(GVP_(hid_dims, hid_dims))
            ff_func.append(GVP_(hid_dims, node_dims, activations=(None, None)))
        self.ff_func = nn.Sequential(*ff_func)

    def forward(self, x, edge_index, edge_attr,
                autoregressive_x=None, node_mask=None):
        '''
        :param x: tuple (s, V) of `torch.Tensor`
        :param edge_index: array of shape [2, n_edges]
        :param edge_attr: tuple (s, V) of `torch.Tensor`
        :param autoregressive_x: tuple (s, V) of `torch.Tensor`. 
                If not `None`, will be used as src node embeddings
                for forming messages where src >= dst. The corrent node 
                embeddings `x` will still be the base of the update and the 
                pointwise feedforward.
        :param node_mask: array of type `bool` to index into the first
                dim of node embeddings (s, V). If not `None`, only
                these nodes will be updated.
        '''
        
        if autoregressive_x is not None:
            src, dst = edge_index
            mask = src < dst
            edge_index_forward = edge_index[:, mask]
            edge_index_backward = edge_index[:, ~mask]
            edge_attr_forward = tuple_index(edge_attr, mask)
            edge_attr_backward = tuple_index(edge_attr, ~mask)
            
            dh = tuple_sum(
                self.conv(x, edge_index_forward, edge_attr_forward),
                self.conv(autoregressive_x, edge_index_backward, edge_attr_backward)
            )
            
            count = scatter_add(torch.ones_like(dst), dst,
                        dim_size=dh[0].size(0)).clamp(min=1).unsqueeze(-1)
            
            dh = dh[0] / count, dh[1] / count.unsqueeze(-1)

        else:
            dh = self.conv(x, edge_index, edge_attr)
        
        if node_mask is not None:
            x_ = x
            x, dh = tuple_index(x, node_mask), tuple_index(dh, node_mask)
            
        x = self.norm[0](tuple_sum(x, self.dropout[0](dh)))
        
        dh = self.ff_func(x)
        x = self.norm[1](tuple_sum(x, self.dropout[1](dh)))
        
        if node_mask is not None:
            x_[0][node_mask], x_[1][node_mask] = x[0], x[1]
            x = x_
        return x

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch_scatter import scatter_mean
from typing import Tuple

class GVPEmbedderModel(nn.Module):
    """Bert + GVP-GNN head (LM-GVP).

    Takes in protein structure graphs of type `torch_geometric.data.Data`
    or `torch_geometric.data.Batch` and returns a scalar score for
    each graph in the batch in a `torch.Tensor` of shape [n_nodes]

    Should be used with `data.ProteinGraphDataset`, or with generators
    of `torch_geometric.data.Batch` objects with the same attributes.
    """

    def __init__(
        self,
        node_in_dim: Tuple[int, int] = (6, 3),
        node_h_dim: Tuple[int, int] = (200, 20),
        edge_in_dim: Tuple[int, int] = (32, 1),
        edge_h_dim: Tuple[int, int] = (100, 10),
        seq_embed_dim: int = 100,
        num_layers=3,
        drop_rate=0.1,
        esm_dim = 1280,
        residual=True,
        include_esm_feats = False,
        **kwargs,
    ):
        """

        Args:
            node_in_dim: node dimensions (s, V) in input graph
            node_h_dim: node dimensions to use in GVP-GNN layers
            edge_in_dim: edge dimensions (s, V) in input graph
            edge_h_dim: edge dimensions to embed to before use in GVP-GNN layers
            weights: a tensor of class weights
            num_layers: number of GVP-GNN layers
            drop_rate: rate to use in all dropout layers
            residual: whether to have residual connections among GNN layers

        Returns:
            None
        """

        super(GVPEmbedderModel, self).__init__(**kwargs)
        self.identity = nn.Identity()
        self.esm_dim = esm_dim
        self.residual = residual
        self.include_esm_feats = include_esm_feats
        
        if include_esm_feats:
            node_in_dim = (node_in_dim[0] + seq_embed_dim + self.esm_dim, node_in_dim[1])
        else:
            node_in_dim = (node_in_dim[0] + seq_embed_dim, node_in_dim[1])
        
        if seq_embed_dim>0: self.W_s = nn.Embedding(20, seq_embed_dim)
        else: self.W_s = None

        self.W_v = nn.Sequential(
            LayerNorm(node_in_dim),
            GVP(node_in_dim, node_h_dim, activations=(None, None)),
        )
        self.W_e = nn.Sequential(
            LayerNorm(edge_in_dim),
            GVP(edge_in_dim, edge_h_dim, activations=(None, None)),
        )

        self.layers = nn.ModuleList(
            GVPConvLayer(node_h_dim, edge_h_dim, drop_rate=drop_rate)
            for _ in range(num_layers)
        )

        if self.residual:
            # concat outputs from GVPConvLayer(s)
            node_h_dim = (
                node_h_dim[0] * num_layers,
                node_h_dim[1] * num_layers,
            )
        ns, _ = node_h_dim
        self.W_out = nn.Sequential(
            LayerNorm(node_h_dim), GVP(node_h_dim, (ns, 0))
        )

        self.dense = nn.Sequential(
            nn.Linear(ns, 2 * ns),
            nn.ReLU(inplace=True),
            nn.Dropout(p=drop_rate),
            nn.Linear(2 * ns, ns),
        )

    def forward(self, batch):
        """Perform the forward pass.

        Args:
            batch: (torch_geometric.data.Data, targets)
            
        Returns:
            logits
        """
        logits = self._forward(batch)
        return logits

    def _forward(self, batch):
        """
        Helper function to perform the forward pass.

        Args:
            batch: torch_geometric.data.Data
        Returns:
            logits
        """
        # ipdb.set_trace()
        if not self.W_s is None:
            seq = self.W_s(batch.seq)
        else:
            seq = batch.seq_feats
            
        h_V = (batch.node_s, batch.node_v)
        h_V = (torch.cat([h_V[0], seq], dim=-1), h_V[1])
        
        h_E = (batch.edge_s, batch.edge_v)
        edge_index = batch.edge_index

        batch_size = batch.num_graphs

        h_V = self.W_v(h_V)
        h_E = self.W_e(h_E)
        if not self.residual:
            for layer in self.layers:
                h_V = layer(h_V, edge_index, h_E)
            out = self.W_out(h_V)
        else:
            h_V_out = []  # collect outputs from GVPConvLayers
            h_V_in = h_V
            for layer in self.layers:
                h_V_out.append(layer(h_V_in, edge_index, h_E))
                h_V_in = h_V_out[-1]
            # concat outputs from GVPConvLayers (separately for s and V)
            h_V_out = (
                torch.cat([h_V[0] for h_V in h_V_out], dim=-1),
                torch.cat([h_V[1] for h_V in h_V_out], dim=-2),
            )
            out = self.W_out(h_V_out)

        out = scatter_mean(out, batch.batch, dim=0)
        
        return out#self.dense(out)

import ipdb
import torch, functools
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter_add

def tuple_sum(*args):
    '''
    Sums any number of tuples (s, V) elementwise.
    '''
    return tuple(map(sum, zip(*args)))

def tuple_cat(*args, dim=-1):
    '''
    Concatenates any number of tuples (s, V) elementwise.
    
    :param dim: dimension along which to concatenate when viewed
                as the `dim` index for the scalar-channel tensors.
                This means that `dim=-1` will be applied as
                `dim=-2` for the vector-channel tensors.
    '''
    dim %= len(args[0][0].shape)
    s_args, v_args = list(zip(*args))
    return torch.cat(s_args, dim=dim), torch.cat(v_args, dim=dim)

def tuple_index(x, idx):
    '''
    Indexes into a tuple (s, V) along the first dimension.
    
    :param idx: any object which can be used to index into a `torch.Tensor`
    '''
    return x[0][idx], x[1][idx]

def randn(n, dims, device="cpu"):
    '''
    Returns random tuples (s, V) drawn elementwise from a normal distribution.
    
    :param n: number of data points
    :param dims: tuple of dimensions (n_scalar, n_vector)
    
    :return: (s, V) with s.shape = (n, n_scalar) and
             V.shape = (n, n_vector, 3)
    '''
    return torch.randn(n, dims[0], device=device), \
            torch.randn(n, dims[1], 3, device=device)

def _norm_no_nan(x, axis=-1, keepdims=False, eps=1e-8, sqrt=True):
    '''
    L2 norm of tensor clamped above a minimum value `eps`.
    
    :param sqrt: if `False`, returns the square of the L2 norm
    '''
    out = torch.clamp(torch.sum(torch.square(x), axis, keepdims), min=eps)
    return torch.sqrt(out) if sqrt else out

def _split(x, nv):
    '''
    Splits a merged representation of (s, V) back into a tuple. 
    Should be used only with `_merge(s, V)` and only if the tuple 
    representation cannot be used.
    
    :param x: the `torch.Tensor` returned from `_merge`
    :param nv: the number of vector channels in the input to `_merge`
    '''
    v = torch.reshape(x[..., -3*nv:], x.shape[:-1] + (nv, 3))
    s = x[..., :-3*nv]
    return s, v

def _merge(s, v):
    '''
    Merges a tuple (s, V) into a single `torch.Tensor`, where the
    vector channels are flattened and appended to the scalar channels.
    Should be used only if the tuple representation cannot be used.
    Use `_split(x, nv)` to reverse.
    '''
    v = torch.reshape(v, v.shape[:-2] + (3*v.shape[-2],))
    return torch.cat([s, v], -1)

class GVP(nn.Module):
    '''
    Geometric Vector Perceptron. See manuscript and README.md
    for more details.
    
    :param in_dims: tuple (n_scalar, n_vector)
    :param out_dims: tuple (n_scalar, n_vector)
    :param h_dim: intermediate number of vector channels, optional
    :param activations: tuple of functions (scalar_act, vector_act)
    :param vector_gate: whether to use vector gating.
                        (vector_act will be used as sigma^+ in vector gating if `True`)
    '''
    def __init__(self, in_dims, out_dims, h_dim=None,
                 activations=(F.relu, torch.sigmoid), vector_gate=False):
        super(GVP, self).__init__()
        self.si, self.vi = in_dims
        self.so, self.vo = out_dims
        self.vector_gate = vector_gate
        if self.vi: 
            self.h_dim = h_dim or max(self.vi, self.vo) 
            self.wh = nn.Linear(self.vi, self.h_dim, bias=False)
            self.ws = nn.Linear(self.h_dim + self.si, self.so)
            if self.vo:
                self.wv = nn.Linear(self.h_dim, self.vo, bias=False)
                if self.vector_gate: self.wsv = nn.Linear(self.so, self.vo)
        else:
            self.ws = nn.Linear(self.si, self.so)
        
        self.scalar_act, self.vector_act = activations
        self.dummy_param = nn.Parameter(torch.empty(0))
        
    def forward(self, x):
        '''
        :param x: tuple (s, V) of `torch.Tensor`, 
                  or (if vectors_in is 0), a single `torch.Tensor`
        :return: tuple (s, V) of `torch.Tensor`,
                 or (if vectors_out is 0), a single `torch.Tensor`
        '''
        if self.vi:
            s, v = x
            v = torch.transpose(v, -1, -2)
            vh = self.wh(v)    
            vn = _norm_no_nan(vh, axis=-2)
            s = self.ws(torch.cat([s, vn], -1))
            if self.vo: 
                v = self.wv(vh) 
                v = torch.transpose(v, -1, -2)
                if self.vector_gate: 
                    if self.vector_act:
                        gate = self.wsv(self.vector_act(s))
                    else:
                        gate = self.wsv(s)
                    v = v * torch.sigmoid(gate).unsqueeze(-1)
                elif self.vector_act:
                    v = v * self.vector_act(
                        _norm_no_nan(v, axis=-1, keepdims=True))
        else:
            s = self.ws(x)
            if self.vo:
                v = torch.zeros(s.shape[0], self.vo, 3,
                                device=self.dummy_param.device)
        if self.scalar_act:
            s = self.scalar_act(s)
        
        return (s, v) if self.vo else s

class _VDropout(nn.Module):
    '''
    Vector channel dropout where the elements of each
    vector channel are dropped together.
    '''
    def __init__(self, drop_rate):
        super(_VDropout, self).__init__()
        self.drop_rate = drop_rate
        self.dummy_param = nn.Parameter(torch.empty(0))

    def forward(self, x):
        '''
        :param x: `torch.Tensor` corresponding to vector channels
        '''
        device = self.dummy_param.device
        if not self.training:
            return x
        mask = torch.bernoulli(
            (1 - self.drop_rate) * torch.ones(x.shape[:-1], device=device)
        ).unsqueeze(-1)
        x = mask * x / (1 - self.drop_rate)
        return x

class Dropout(nn.Module):
    '''
    Combined dropout for tuples (s, V).
    Takes tuples (s, V) as input and as output.
    '''
    def __init__(self, drop_rate):
        super(Dropout, self).__init__()
        self.sdropout = nn.Dropout(drop_rate)
        self.vdropout = _VDropout(drop_rate)

    def forward(self, x):
        '''
        :param x: tuple (s, V) of `torch.Tensor`,
                  or single `torch.Tensor` 
                  (will be assumed to be scalar channels)
        '''
        if type(x) is torch.Tensor:
            return self.sdropout(x)
        s, v = x
        return self.sdropout(s), self.vdropout(v)

class LayerNorm(nn.Module):
    '''
    Combined LayerNorm for tuples (s, V).
    Takes tuples (s, V) as input and as output.
    '''
    def __init__(self, dims):
        super(LayerNorm, self).__init__()
        self.s, self.v = dims
        self.scalar_norm = nn.LayerNorm(self.s)
        
    def forward(self, x):
        '''
        :param x: tuple (s, V) of `torch.Tensor`,
                  or single `torch.Tensor` 
                  (will be assumed to be scalar channels)
        '''
        if not self.v:
            return self.scalar_norm(x)
        s, v = x
        vn = _norm_no_nan(v, axis=-1, keepdims=True, sqrt=False)
        vn = torch.sqrt(torch.mean(vn, dim=-2, keepdim=True))
        return self.scalar_norm(s), v / vn

class GVPConv(MessagePassing):
    '''
    Graph convolution / message passing with Geometric Vector Perceptrons.
    Takes in a graph with node and edge embeddings,
    and returns new node embeddings.
    
    This does NOT do residual updates and pointwise feedforward layers
    ---see `GVPConvLayer`.
    
    :param in_dims: input node embedding dimensions (n_scalar, n_vector)
    :param out_dims: output node embedding dimensions (n_scalar, n_vector)
    :param edge_dims: input edge embedding dimensions (n_scalar, n_vector)
    :param n_layers: number of GVPs in the message function
    :param module_list: preconstructed message function, overrides n_layers
    :param aggr: should be "add" if some incoming edges are masked, as in
                 a masked autoregressive decoder architecture, otherwise "mean"
    :param activations: tuple of functions (scalar_act, vector_act) to use in GVPs
    :param vector_gate: whether to use vector gating.
                        (vector_act will be used as sigma^+ in vector gating if `True`)
    '''
    def __init__(self, in_dims, out_dims, edge_dims,
                 n_layers=3, module_list=None, aggr="mean", 
                 activations=(F.relu, torch.sigmoid), vector_gate=False):
        super(GVPConv, self).__init__(aggr=aggr)
        self.si, self.vi = in_dims
        self.so, self.vo = out_dims
        self.se, self.ve = edge_dims
        
        GVP_ = functools.partial(GVP, 
                activations=activations, vector_gate=vector_gate)
        
        module_list = module_list or []
        if not module_list:
            if n_layers == 1:
                module_list.append(
                    GVP_((2*self.si + self.se, 2*self.vi + self.ve), 
                        (self.so, self.vo), activations=(None, None)))
            else:
                module_list.append(
                    GVP_((2*self.si + self.se, 2*self.vi + self.ve), out_dims)
                )
                for i in range(n_layers - 2):
                    module_list.append(GVP_(out_dims, out_dims))
                module_list.append(GVP_(out_dims, out_dims,
                                       activations=(None, None)))
        self.message_func = nn.Sequential(*module_list)

    def forward(self, x, edge_index, edge_attr):
        '''
        :param x: tuple (s, V) of `torch.Tensor`
        :param edge_index: array of shape [2, n_edges]
        :param edge_attr: tuple (s, V) of `torch.Tensor`
        '''
        x_s, x_v = x
        message = self.propagate(edge_index, 
                    s=x_s, v=x_v.reshape(x_v.shape[0], 3*x_v.shape[1]),
                    edge_attr=edge_attr)
        return _split(message, self.vo) 

    def message(self, s_i, v_i, s_j, v_j, edge_attr):
        v_j = v_j.view(v_j.shape[0], v_j.shape[1]//3, 3)
        v_i = v_i.view(v_i.shape[0], v_i.shape[1]//3, 3)
        message = tuple_cat((s_j, v_j), edge_attr, (s_i, v_i))
        message = self.message_func(message)
        return _merge(*message)


class GVPConvLayer(nn.Module):
    '''
    Full graph convolution / message passing layer with 
    Geometric Vector Perceptrons. Residually updates node embeddings with
    aggregated incoming messages, applies a pointwise feedforward 
    network to node embeddings, and returns updated node embeddings.
    
    To only compute the aggregated messages, see `GVPConv`.
    
    :param node_dims: node embedding dimensions (n_scalar, n_vector)
    :param edge_dims: input edge embedding dimensions (n_scalar, n_vector)
    :param n_message: number of GVPs to use in message function
    :param n_feedforward: number of GVPs to use in feedforward function
    :param drop_rate: drop probability in all dropout layers
    :param autoregressive: if `True`, this `GVPConvLayer` will be used
           with a different set of input node embeddings for messages
           where src >= dst
    :param activations: tuple of functions (scalar_act, vector_act) to use in GVPs
    :param vector_gate: whether to use vector gating.
                        (vector_act will be used as sigma^+ in vector gating if `True`)
    '''
    def __init__(self, node_dims, edge_dims,
                 n_message=3, n_feedforward=2, drop_rate=.1,
                 autoregressive=False, 
                 activations=(F.relu, torch.sigmoid), vector_gate=False):
        
        super(GVPConvLayer, self).__init__()
        self.conv = GVPConv(node_dims, node_dims, edge_dims, n_message,
                           aggr="add" if autoregressive else "mean",
                           activations=activations, vector_gate=vector_gate)
        GVP_ = functools.partial(GVP, 
                activations=activations, vector_gate=vector_gate)
        self.norm = nn.ModuleList([LayerNorm(node_dims) for _ in range(2)])
        self.dropout = nn.ModuleList([Dropout(drop_rate) for _ in range(2)])

        ff_func = []
        if n_feedforward == 1:
            ff_func.append(GVP_(node_dims, node_dims, activations=(None, None)))
        else:
            hid_dims = 4*node_dims[0], 2*node_dims[1]
            ff_func.append(GVP_(node_dims, hid_dims))
            for i in range(n_feedforward-2):
                ff_func.append(GVP_(hid_dims, hid_dims))
            ff_func.append(GVP_(hid_dims, node_dims, activations=(None, None)))
        self.ff_func = nn.Sequential(*ff_func)

    def forward(self, x, edge_index, edge_attr,
                autoregressive_x=None, node_mask=None):
        '''
        :param x: tuple (s, V) of `torch.Tensor`
        :param edge_index: array of shape [2, n_edges]
        :param edge_attr: tuple (s, V) of `torch.Tensor`
        :param autoregressive_x: tuple (s, V) of `torch.Tensor`. 
                If not `None`, will be used as src node embeddings
                for forming messages where src >= dst. The corrent node 
                embeddings `x` will still be the base of the update and the 
                pointwise feedforward.
        :param node_mask: array of type `bool` to index into the first
                dim of node embeddings (s, V). If not `None`, only
                these nodes will be updated.
        '''
        
        if autoregressive_x is not None:
            src, dst = edge_index
            mask = src < dst
            edge_index_forward = edge_index[:, mask]
            edge_index_backward = edge_index[:, ~mask]
            edge_attr_forward = tuple_index(edge_attr, mask)
            edge_attr_backward = tuple_index(edge_attr, ~mask)
            
            dh = tuple_sum(
                self.conv(x, edge_index_forward, edge_attr_forward),
                self.conv(autoregressive_x, edge_index_backward, edge_attr_backward)
            )
            
            count = scatter_add(torch.ones_like(dst), dst,
                        dim_size=dh[0].size(0)).clamp(min=1).unsqueeze(-1)
            
            dh = dh[0] / count, dh[1] / count.unsqueeze(-1)

        else:
            dh = self.conv(x, edge_index, edge_attr)
        
        if node_mask is not None:
            x_ = x
            x, dh = tuple_index(x, node_mask), tuple_index(dh, node_mask)
            
        x = self.norm[0](tuple_sum(x, self.dropout[0](dh)))
        
        dh = self.ff_func(x)
        x = self.norm[1](tuple_sum(x, self.dropout[1](dh)))
        
        if node_mask is not None:
            x_[0][node_mask], x_[1][node_mask] = x[0], x[1]
            x = x_
        return x

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, MessagePassing
from torch_geometric.utils import degree
    
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch_scatter import scatter_mean
from typing import Tuple

import torch
import torch.nn.functional as F
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU, Dropout
from torch_geometric.nn import GCNConv, GINConv
from torch_geometric.nn import global_mean_pool, global_add_pool

class GCN(torch.nn.Module):
    """GCN"""
    def __init__(self, dim_in, dim_h):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(dim_in, dim_h)
        self.conv2 = GCNConv(dim_h, dim_h)
        self.conv3 = GCNConv(dim_h, dim_h)

    def forward(self, x, edge_index, batch):
        # Node embeddings 
        h = self.conv1(x, edge_index)
        h = h.relu()
        h = self.conv2(h, edge_index)
        h = h.relu()
        h = self.conv3(h, edge_index)

        # Graph-level readout
        hG = global_mean_pool(h, batch)

        return hG

class GIN(torch.nn.Module):
    """GIN"""
    def __init__(self, dim_in, dim_h):
        super(GIN, self).__init__()
        self.conv1 = GINConv(
            Sequential(Linear(dim_in, dim_h),
                       BatchNorm1d(dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU()))
        self.conv2 = GINConv(
            Sequential(Linear(dim_h, dim_h), BatchNorm1d(dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU()))
        self.conv3 = GINConv(
            Sequential(Linear(dim_h, dim_h), BatchNorm1d(dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU()))

    def forward(self, x, edge_index, batch):
        # Node embeddings 
        h1 = self.conv1(x, edge_index)
        h2 = self.conv2(h1, edge_index)
        h3 = self.conv3(h2, edge_index)

        # Graph-level readout
        # ipdb.set_trace()
        h1 = global_add_pool(h1, batch)
        h2 = global_add_pool(h2, batch)
        h3 = global_add_pool(h3, batch)

        # Concatenate graph embeddings
        h = torch.cat((h1, h2, h3), dim=1)
        
        return h

class GVPEmbedderModel(nn.Module):
    """Bert + GVP-GNN head (LM-GVP).

    Takes in protein structure graphs of type `torch_geometric.data.Data`
    or `torch_geometric.data.Batch` and returns a scalar score for
    each graph in the batch in a `torch.Tensor` of shape [n_nodes]

    Should be used with `data.ProteinGraphDataset`, or with generators
    of `torch_geometric.data.Batch` objects with the same attributes.
    """

    def __init__(
        self,
        node_in_dim: Tuple[int, int] = (6, 3),
        node_h_dim: Tuple[int, int] = (200, 20),
        edge_in_dim: Tuple[int, int] = (32, 1),
        edge_h_dim: Tuple[int, int] = (100, 10),
        seq_embed_dim: int = 100,
        num_layers=3,
        drop_rate=0.1,
        esm_dim = 1280,
        residual=True,
        include_esm_feats = False,
        use_gin = False,
        device = 'cpu',
        **kwargs,
    ):
        """

        Args:
            node_in_dim: node dimensions (s, V) in input graph
            node_h_dim: node dimensions to use in GVP-GNN layers
            edge_in_dim: edge dimensions (s, V) in input graph
            edge_h_dim: edge dimensions to embed to before use in GVP-GNN layers
            weights: a tensor of class weights
            num_layers: number of GVP-GNN layers
            drop_rate: rate to use in all dropout layers
            residual: whether to have residual connections among GNN layers

        Returns:
            None
        """

        super(GVPEmbedderModel, self).__init__(**kwargs)
        self.identity = nn.Identity()
        self.esm_dim = esm_dim
        self.residual = residual
        self.include_esm_feats = include_esm_feats
        self.use_gin = use_gin
        
        if include_esm_feats:
            node_in_dim = (node_in_dim[0] + seq_embed_dim + self.esm_dim, node_in_dim[1])
        else:
            node_in_dim = (node_in_dim[0] + seq_embed_dim, node_in_dim[1])
        
        if seq_embed_dim>0: self.W_s = nn.Embedding(20, seq_embed_dim)
        else: self.W_s = None

        self.W_v = nn.Sequential(
            LayerNorm(node_in_dim),
            GVP(node_in_dim, node_h_dim, activations=(None, None)),
        )
        self.W_e = nn.Sequential(
            LayerNorm(edge_in_dim),
            GVP(edge_in_dim, edge_h_dim, activations=(None, None)),
        )

        self.layers = nn.ModuleList(
            GVPConvLayer(node_h_dim, edge_h_dim, drop_rate=drop_rate)
            for _ in range(num_layers)
        )

        if self.residual:
            # concat outputs from GVPConvLayer(s)
            node_h_dim = (
                node_h_dim[0] * num_layers,
                node_h_dim[1] * num_layers,
            )
        ns, _ = node_h_dim
        self.W_out = nn.Sequential(
            LayerNorm(node_h_dim), GVP(node_h_dim, (ns, 0))
        )

        self.dense = nn.Sequential(
            nn.Linear(ns, 2 * ns),
            nn.ReLU(inplace=True),
            nn.Dropout(p=drop_rate),
            nn.Linear(2 * ns, ns),
        )
        
        self.device = device
        # ipdb.set_trace()
        if self.use_gin:
            self.gin = GIN(dim_in = node_in_dim[0], dim_h = node_h_dim[0])

    def forward(self, batch):
        """Perform the forward pass.

        Args:
            batch: (torch_geometric.data.Data, targets)
            
        Returns:
            logits
        """
        if self.use_gin:
            if not self.W_s is None:
                seq = self.W_s(batch.seq)
            else:
                seq = batch.seq_feats

            x = batch.node_s
            x = torch.cat([x, seq], dim=-1).to(self.device)

            edge_index = batch.edge_index.to(self.device)
            logits = self.gin(x, edge_index, batch.batch)
            
        else:
            logits = self._forward(batch)
            
        return logits

    def _forward(self, batch):
        """
        Helper function to perform the forward pass.

        Args:
            batch: torch_geometric.data.Data
        Returns:
            logits
        """
        # ipdb.set_trace()
        if not self.W_s is None:
            seq = self.W_s(batch.seq)
        else:
            seq = batch.seq_feats
            
        h_V = (batch.node_s, batch.node_v)
        h_V = (torch.cat([h_V[0], seq], dim=-1), h_V[1])
        
        h_E = (batch.edge_s, batch.edge_v)
        edge_index = batch.edge_index

        batch_size = batch.num_graphs

        h_V = self.W_v(h_V)
        h_E = self.W_e(h_E)
        if not self.residual:
            for layer in self.layers:
                h_V = layer(h_V, edge_index, h_E)
            out = self.W_out(h_V)
        else:
            h_V_out = []  # collect outputs from GVPConvLayers
            h_V_in = h_V
            for layer in self.layers:
                h_V_out.append(layer(h_V_in, edge_index, h_E))
                h_V_in = h_V_out[-1]
            # concat outputs from GVPConvLayers (separately for s and V)
            h_V_out = (
                torch.cat([h_V[0] for h_V in h_V_out], dim=-1),
                torch.cat([h_V[1] for h_V in h_V_out], dim=-2),
            )
            out = self.W_out(h_V_out)

        out = scatter_mean(out, batch.batch, dim=0)
        
        return out#self.dense(out)