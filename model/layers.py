import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from dgl.nn.pytorch import GATConv, HeteroGraphConv
import numpy as np

def weight_variable_glorot(input_dim, output_dim):
    """Create a weight variable with Glorot & Bengio (AISTATS 2010)
    initialization.
    """
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = torch.FloatTensor(input_dim, output_dim).uniform_(-init_range, init_range)
    return nn.Parameter(initial, requires_grad=True)

class Dense(nn.Module):
    """Dense layer."""

    def __init__(self, input_dim, output_dim, dropout=0., bias=True, sparse_inputs=False, act=F.relu):
        super(Dense, self).__init__()
        self.dropout = dropout
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.bias = bias
        # helper variable for sparse dropout
        self.sparse_inputs = sparse_inputs
        self.act = act
        # Parameter with requires_grad=True(default) mean nead to compute gradient
        self.weights = weight_variable_glorot(input_dim, output_dim)
        if self.bias:
            self.bias = Parameter(torch.zeros([output_dim], dtype=torch.float32))

    def forward(self, inputs):
        inputs = F.dropout(inputs, self.dropout, self.training)
        if self.sparse_inputs:
            output = torch.stack([torch.sparse.mm(inp, self.weights) for inp in torch.unbind(inputs, dim=0)], dim=0)
        else:
            output = torch.stack([torch.mm(inp, self.weights) for inp in torch.unbind(inputs, dim=0)], dim=0)

        # bias
        output += self.bias
        output = self.act(output)
        return output

class RGATLayer(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 num_heads,
                 rel_names,
                 act=lambda x: x,
                 ):
        super().__init__()
        self.layer = HeteroGraphConv(
            {rel: GATConv(in_feats, out_feats // num_heads, num_heads, feat_drop=0.2, attn_drop=0.2) for rel in
             rel_names},
            aggregate='mean')
        self.out_feats = out_feats
        self.act = act

    def forward(self, dataset, graph, inputs):
        outputs = []
        for input in inputs:
            output = []
            h = self.layer(graph, input)
            if dataset == 'ACM':
                key = ['paper', 'author', 'subject']
            elif dataset == 'DBLP':
                key = ['author', 'paper', 'term', 'venue']
            elif dataset == 'YELP':
                key = ['business', 'user', 'service', 'level']
            for k in key:
                out = self.act(h[k]).reshape(-1, self.out_feats)
                output.append(out)
            output = torch.cat(output, dim=0)
            outputs.append(output)
        outputs = torch.stack(outputs, dim=0)
        return outputs

