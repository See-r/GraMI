import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
from dgl.nn.pytorch import GATConv, HeteroGraphConv
import numpy as np
import dgl.function as fn
from dgl.nn.pytorch import edge_softmax
from dgl.utils import expand_as_pair


# from torch_geometric.nn import RGATConv

def weight_variable_glorot(input_dim, output_dim):
    """Create a weight variable with Glorot & Bengio (AISTATS 2010)
    initialization.
    """
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = torch.FloatTensor(input_dim, output_dim).uniform_(-init_range, init_range)
    # tf.random_uniform([input_dim, output_dim], minval=-init_range,maxval=init_range, dtype=tf.float32)
    return nn.Parameter(initial, requires_grad=True)


'''Dense接收的就是x的转置矩阵，运算过程很简单'''


class Dense(nn.Module):
    """Dense layer.3层"""

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
        '''
        if self.sparse_inputs:
            output = torch.sparse.mm(x, self.weights)
        else:
            output = torch.mm(x, self.weights)

        # bias
        output += self.bias

        return output          
        '''


'''实现GCN'''


class GraphConvolution(Module):
    """
    GCN layer, based on
    that allows MIMO
    """

    def __init__(self, in_features, out_features, dropout=0., act=F.relu):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.act = act
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, input, adj):
        input = F.dropout(input, self.dropout, self.training)
        # input（特征矩阵）是三维的
        """
        if the input features are a matrix -- excute regular GCN,
        if the input features are of shape [K, N, Z] -- excute MIMO GCN with shared weights.
        """
        # An alternative to derive XW (line 32 to 35)
        # W = self.weight.view(
        #         [1, self.in_features, self.out_features]
        #         ).expand([input.shape[0], -1, -1])
        # support = torch.bmm(input, W)

        support = torch.stack(
            [torch.mm(inp, self.weight) for inp in torch.unbind(input, dim=0)],
            dim=0)  # XW
        output = torch.stack(
            [torch.spmm(adj, sup) for sup in torch.unbind(support, dim=0)],
            dim=0)  # AXW
        output = self.act(output)
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'


'''实现HAN'''


class SemanticAttention(nn.Module):  # 语义级注意力
    def __init__(self, in_size, hidden_size=64):
        super(SemanticAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, Z):  # z为3维
        output = []
        for z in Z:
            w = self.project(z).mean(0)
            beta = torch.softmax(w, dim=0)
            beta = beta.expand((z.shape[0],) + beta.shape)
            output.append((beta * z).sum(1))
        return output


class HANLayer(nn.Module):  # 一次的HAN基本层（实际有K次即K头）
    def __init__(self, num_meta_paths, in_size, out_size, layer_num_heads, dropout, noise):
        super(HANLayer, self).__init__()
        self.noise = noise  # 判断加不加噪声
        # 基于每条元路径邻接矩阵的图注意力层
        self.gat_layers = nn.ModuleList()
        for i in range(num_meta_paths):
            self.gat_layers.append(
                GATConv(
                    in_size,
                    out_size,
                    layer_num_heads,
                    dropout,
                    dropout,
                    # activation=torch.tanh,
                    activation=F.elu,
                )
            )
        self.semantic_attention = SemanticAttention(
            in_size=out_size * layer_num_heads
        )
        self.num_meta_paths = num_meta_paths

    def forward(self, gs, H, e=[]):  # g : list[DGLGraph],List of graphs基于元路径的同构图,e为噪声矩阵
        semantic_embeddings = []
        # single_embedding = [],e维数不对
        # H为三维
        '''
        for i, g in enumerate(gs):
            semantic_embeddings.append(self.gat_layers[i](g, h).flatten(1))
        semantic_embeddings = torch.stack(semantic_embeddings, dim=1)  # 将node_level的embedding堆叠起来        
        '''
        if self.noise == 1:  # H第一维为1
            for noise in e:  # 加入j个噪声矩阵
                single_embedding = []
                for i, g in enumerate(gs):
                    single_embedding.append(self.gat_layers[i](g, H[0]).flatten(1) + noise)
                single_embedding = torch.stack(single_embedding, dim=1)
                semantic_embeddings.append(single_embedding)
        elif self.noise == 0:
            for j, h in enumerate(torch.unbind(H, dim=0)):
                single_embedding = []
                for i, g in enumerate(gs):
                    single_embedding.append(self.gat_layers[i](g, h).flatten(1))
                single_embedding = torch.stack(single_embedding, dim=1)
                semantic_embeddings.append(single_embedding)
        '''
        for j, h in enumerate(torch.unbind(H, dim=0)):  # 取一个特征矩阵
            single_embedding = []
            for i, g in enumerate(gs):
                if self.noise == 1:
                    embedding = self.gat_layers[i](g, h)
                    print("Adding noise......")
                    print(e[j].shape)  # 噪声矩阵
                    print(embedding.shape)  # GAT嵌入矩阵
                    single_embedding.append(embedding.flatten(1) + e[j])
                else:
                    single_embedding.append(self.gat_layers[i](g, h).flatten(1))
            single_embedding = torch.stack(single_embedding, dim=1)
            semantic_embeddings.append(single_embedding)  # n个特征矩阵        
        '''

        return self.semantic_attention(semantic_embeddings)  # 生成semantic_level embedding


class ProjectLayer(nn.Module):
    def __init__(self, hidden_size, out_size, num_heads):
        super(ProjectLayer, self).__init__()
        self.predict = nn.Linear(hidden_size * num_heads[-1], out_size)

    def forward(self, H):  # H为feature
        '''h为3维'''
        output = []
        for h in H:
            output.append(self.predict(h))
        output = torch.stack(output, dim=0)
        return output


class HAN(nn.Module):
    def __init__(self, num_meta_paths, in_size, hidden_size, out_size, num_heads, dropout, noise=0):
        super(HAN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(
            HANLayer(
                num_meta_paths, in_size, hidden_size, num_heads[0], dropout, noise
            )
        )
        for l in range(1, len(num_heads)):
            self.layers.append(
                HANLayer(
                    num_meta_paths,
                    hidden_size * num_heads[l - 1],
                    hidden_size,
                    num_heads[l],
                    dropout,
                    noise,
                )
            )
        self.predict = nn.Linear(hidden_size * num_heads[-1], out_size)

    def forward(self, g, H, e):  # g为dgl graph的list,H为feature,e为噪声矩阵
        '''h为3维'''
        output = []
        for gnn in self.layers:
            H = gnn(g, H, e)
        for h in H:
            output.append(self.predict(h))
        output = torch.stack(output, dim=0)
        return output


'''实现MAGNN'''

fc_switch = False


class MAGNN_metapath_specific(nn.Module):
    def __init__(self,
                 etypes,  # 边类型
                 out_dim,
                 num_heads,
                 rnn_type='gru',
                 r_vec=None,
                 attn_drop=0.5,
                 alpha=0.01,
                 use_minibatch=False,
                 attn_switch=False):
        super(MAGNN_metapath_specific, self).__init__()
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.rnn_type = rnn_type
        self.etypes = etypes
        self.r_vec = r_vec
        self.use_minibatch = use_minibatch
        self.attn_switch = attn_switch

        # rnn-like metapath instance aggregator
        # consider multiple attention heads
        if rnn_type == 'gru':
            self.rnn = nn.GRU(out_dim, num_heads * out_dim)
        elif rnn_type == 'lstm':
            self.rnn = nn.LSTM(out_dim, num_heads * out_dim)
        elif rnn_type == 'bi-gru':
            self.rnn = nn.GRU(out_dim, num_heads * out_dim // 2, bidirectional=True)
        elif rnn_type == 'bi-lstm':
            self.rnn = nn.LSTM(out_dim, num_heads * out_dim // 2, bidirectional=True)
        elif rnn_type == 'linear':
            self.rnn = nn.Linear(out_dim, num_heads * out_dim)
        elif rnn_type == 'max-pooling':
            self.rnn = nn.Linear(out_dim, num_heads * out_dim)
        elif rnn_type == 'neighbor-linear':
            self.rnn = nn.Linear(out_dim, num_heads * out_dim)

        # node-level attention
        # attention considers the center node embedding or not
        if self.attn_switch:
            self.attn1 = nn.Linear(out_dim, num_heads, bias=False)
            self.attn2 = nn.Parameter(torch.empty(size=(1, num_heads, out_dim)))
        else:
            self.attn = nn.Parameter(torch.empty(size=(1, num_heads, out_dim)))
        self.leaky_relu = nn.LeakyReLU(alpha)
        self.softmax = edge_softmax
        if attn_drop:
            self.attn_drop = nn.Dropout(attn_drop)
        else:
            self.attn_drop = lambda x: x

        # weight initialization
        if self.attn_switch:
            nn.init.xavier_normal_(self.attn1.weight, gain=1.414)
            nn.init.xavier_normal_(self.attn2.data, gain=1.414)
        else:
            nn.init.xavier_normal_(self.attn.data, gain=1.414)

    def edge_softmax(self, g):
        attention = self.softmax(g, g.edata.pop('a'))
        # Dropout attention scores and save them
        g.edata['a_drop'] = self.attn_drop(attention)

    def message_passing(self, edges):
        ft = edges.data['eft'] * edges.data['a_drop']
        return {'ft': ft}

    def forward(self, inputs):
        # features: num_all_nodes x out_dim
        if self.use_minibatch:
            g, features, type_mask, edge_metapath_indices, target_idx = inputs
        else:
            g, features, type_mask, edge_metapath_indices = inputs

        # Embedding layer
        # use torch.nn.functional.embedding or torch.embedding here
        # do not use torch.nn.embedding
        # edata: E x Seq x out_dim
        edata = F.embedding(edge_metapath_indices, features)

        # apply rnn to metapath-based feature sequence
        if self.rnn_type == 'gru':
            _, hidden = self.rnn(edata.permute(1, 0, 2))
        elif self.rnn_type == 'lstm':
            _, (hidden, _) = self.rnn(edata.permute(1, 0, 2))
        elif self.rnn_type == 'bi-gru':
            _, hidden = self.rnn(edata.permute(1, 0, 2))
            hidden = hidden.permute(1, 0, 2).reshape(-1, self.out_dim, self.num_heads).permute(0, 2, 1).reshape(
                -1, self.num_heads * self.out_dim).unsqueeze(dim=0)
        elif self.rnn_type == 'bi-lstm':
            _, (hidden, _) = self.rnn(edata.permute(1, 0, 2))
            hidden = hidden.permute(1, 0, 2).reshape(-1, self.out_dim, self.num_heads).permute(0, 2, 1).reshape(
                -1, self.num_heads * self.out_dim).unsqueeze(dim=0)
        elif self.rnn_type == 'average':
            hidden = torch.mean(edata, dim=1)
            hidden = torch.cat([hidden] * self.num_heads, dim=1)
            hidden = hidden.unsqueeze(dim=0)
        elif self.rnn_type == 'linear':
            hidden = self.rnn(torch.mean(edata, dim=1))
            hidden = hidden.unsqueeze(dim=0)
        elif self.rnn_type == 'max-pooling':
            hidden, _ = torch.max(self.rnn(edata), dim=1)
            hidden = hidden.unsqueeze(dim=0)
        elif self.rnn_type == 'TransE0' or self.rnn_type == 'TransE1':
            r_vec = self.r_vec
            if self.rnn_type == 'TransE0':
                r_vec = torch.stack((r_vec, -r_vec), dim=1)
                r_vec = r_vec.reshape(self.r_vec.shape[0] * 2, self.r_vec.shape[1])  # etypes x out_dim
            edata = F.normalize(edata, p=2, dim=2)
            for i in range(edata.shape[1] - 1):
                # consider None edge (symmetric relation)
                temp_etypes = [etype for etype in self.etypes[i:] if etype is not None]
                edata[:, i] = edata[:, i] + r_vec[temp_etypes].sum(dim=0)
            hidden = torch.mean(edata, dim=1)
            hidden = torch.cat([hidden] * self.num_heads, dim=1)
            hidden = hidden.unsqueeze(dim=0)
        elif self.rnn_type == 'RotatE0' or self.rnn_type == 'RotatE1':
            r_vec = F.normalize(self.r_vec, p=2, dim=2)
            if self.rnn_type == 'RotatE0':
                r_vec = torch.stack((r_vec, r_vec), dim=1)
                r_vec[:, 1, :, 1] = -r_vec[:, 1, :, 1]
                r_vec = r_vec.reshape(self.r_vec.shape[0] * 2, self.r_vec.shape[1], 2)  # etypes x out_dim/2 x 2
            edata = edata.reshape(edata.shape[0], edata.shape[1], edata.shape[2] // 2, 2)
            final_r_vec = torch.zeros([edata.shape[1], self.out_dim // 2, 2], device=edata.device)
            final_r_vec[-1, :, 0] = 1
            for i in range(final_r_vec.shape[0] - 2, -1, -1):
                # consider None edge (symmetric relation)
                if self.etypes[i] is not None:
                    final_r_vec[i, :, 0] = final_r_vec[i + 1, :, 0].clone() * r_vec[self.etypes[i], :, 0] - \
                                           final_r_vec[i + 1, :, 1].clone() * r_vec[self.etypes[i], :, 1]
                    final_r_vec[i, :, 1] = final_r_vec[i + 1, :, 0].clone() * r_vec[self.etypes[i], :, 1] + \
                                           final_r_vec[i + 1, :, 1].clone() * r_vec[self.etypes[i], :, 0]
                else:
                    final_r_vec[i, :, 0] = final_r_vec[i + 1, :, 0].clone()
                    final_r_vec[i, :, 1] = final_r_vec[i + 1, :, 1].clone()
            for i in range(edata.shape[1] - 1):
                temp1 = edata[:, i, :, 0].clone() * final_r_vec[i, :, 0] - \
                        edata[:, i, :, 1].clone() * final_r_vec[i, :, 1]
                temp2 = edata[:, i, :, 0].clone() * final_r_vec[i, :, 1] + \
                        edata[:, i, :, 1].clone() * final_r_vec[i, :, 0]
                edata[:, i, :, 0] = temp1
                edata[:, i, :, 1] = temp2
            edata = edata.reshape(edata.shape[0], edata.shape[1], -1)
            hidden = torch.mean(edata, dim=1)
            hidden = torch.cat([hidden] * self.num_heads, dim=1)
            hidden = hidden.unsqueeze(dim=0)
        elif self.rnn_type == 'neighbor':
            hidden = edata[:, 0]
            hidden = torch.cat([hidden] * self.num_heads, dim=1)
            hidden = hidden.unsqueeze(dim=0)
        elif self.rnn_type == 'neighbor-linear':
            hidden = self.rnn(edata[:, 0])
            hidden = hidden.unsqueeze(dim=0)

        eft = hidden.permute(1, 0, 2).view(-1, self.num_heads, self.out_dim)  # E x num_heads x out_dim
        if self.attn_switch:
            center_node_feat = F.embedding(edge_metapath_indices[:, -1], features)  # E x out_dim
            a1 = self.attn1(center_node_feat)  # E x num_heads
            a2 = (eft * self.attn2).sum(dim=-1)  # E x num_heads
            a = (a1 + a2).unsqueeze(dim=-1)  # E x num_heads x 1
        else:
            a = (eft * self.attn).sum(dim=-1).unsqueeze(dim=-1)  # E x num_heads x 1
        a = self.leaky_relu(a)
        g.edata.update({'eft': eft, 'a': a})
        # compute softmax normalized attention values
        self.edge_softmax(g)
        # compute the aggregated node features scaled by the dropped,
        # unnormalized attention values.
        g.update_all(self.message_passing, fn.sum('ft', 'ft'))
        ret = g.ndata['ft']  # E x num_heads x out_dim

        if self.use_minibatch:
            return ret[target_idx]
        else:
            return ret


class MAGNN_ctr_ntype_specific(nn.Module):
    def __init__(self,
                 num_metapaths,
                 etypes_list,
                 out_dim,
                 num_heads,
                 attn_vec_dim,
                 rnn_type='gru',
                 r_vec=None,  # 特定边类型的参数
                 attn_drop=0.5,
                 use_minibatch=False):
        super(MAGNN_ctr_ntype_specific, self).__init__()
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.use_minibatch = use_minibatch

        # metapath-specific layers特定metapath的网络层
        self.metapath_layers = nn.ModuleList()
        for i in range(num_metapaths):  # 根据metapath的数量增加相应网络层
            self.metapath_layers.append(MAGNN_metapath_specific(etypes_list[i],
                                                                out_dim,
                                                                num_heads,
                                                                rnn_type,
                                                                r_vec,
                                                                attn_drop=attn_drop,
                                                                use_minibatch=use_minibatch))

        # metapath-level attention
        # note that the acutal input dimension should consider the number of heads
        # as multiple head outputs are concatenated together
        self.fc1 = nn.Linear(out_dim * num_heads, attn_vec_dim, bias=True)
        self.fc2 = nn.Linear(attn_vec_dim, 1, bias=False)

        # weight initialization
        nn.init.xavier_normal_(self.fc1.weight, gain=1.414)
        nn.init.xavier_normal_(self.fc2.weight, gain=1.414)

    def forward(self, inputs):
        if self.use_minibatch:
            g_list, features, type_mask, edge_metapath_indices_list, target_idx_list = inputs

            # metapath-specific layers
            metapath_outs = [F.elu(metapath_layer((g, features, type_mask, edge_metapath_indices, target_idx)).view(-1,
                                                                                                                    self.num_heads * self.out_dim))
                             for g, edge_metapath_indices, target_idx, metapath_layer in
                             zip(g_list, edge_metapath_indices_list, target_idx_list, self.metapath_layers)]
        else:
            g_list, features, type_mask, edge_metapath_indices_list = inputs

            # metapath-specific layers
            metapath_outs = [F.elu(
                metapath_layer((g, features, type_mask, edge_metapath_indices)).view(-1, self.num_heads * self.out_dim))
                for g, edge_metapath_indices, metapath_layer in
                zip(g_list, edge_metapath_indices_list, self.metapath_layers)]

        beta = []
        for metapath_out in metapath_outs:
            fc1 = torch.tanh(self.fc1(metapath_out))
            fc1_mean = torch.mean(fc1, dim=0)
            fc2 = self.fc2(fc1_mean)
            beta.append(fc2)
        beta = torch.cat(beta, dim=0)
        beta = F.softmax(beta, dim=0)
        beta = torch.unsqueeze(beta, dim=-1)
        beta = torch.unsqueeze(beta, dim=-1)
        metapath_outs = [torch.unsqueeze(metapath_out, dim=0) for metapath_out in metapath_outs]
        metapath_outs = torch.cat(metapath_outs, dim=0)
        h = torch.sum(beta * metapath_outs, dim=0)
        return h


# MAGNN_nc

class MAGNN_nc_layer(nn.Module):
    def __init__(self,
                 num_metapaths_list,
                 num_edge_type,
                 etypes_lists,
                 in_dim,
                 out_dim,
                 num_heads,
                 attn_vec_dim,
                 rnn_type='gru',
                 attn_drop=0.5):
        super(MAGNN_nc_layer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads

        # 特定类型边的参数
        r_vec = None
        if rnn_type == 'TransE0':
            r_vec = nn.parameter(torch.empty(size=(num_edge_type // 2, in_dim)))
        elif rnn_type == 'TransE1':
            r_vec = nn.Parameter(torch.empty(size=(num_edge_type, in_dim)))
        elif rnn_type == 'RotatE0':
            r_vec = nn.Parameter(torch.empty(size=(num_edge_type // 2, in_dim // 2, 2)))
        elif rnn_type == 'RotatE1':
            r_vec = nn.Parameter(torch.empty(size=(num_edge_type, in_dim // 2, 2)))
        if r_vec is not None:
            nn.init.xavier_normal_(r_vec.data, gain=1.414)
        # 特定目标节点的网络层
        self.ctr_ntype_layers = nn.ModuleList()
        for i in range(len(num_metapaths_list)):
            self.ctr_ntype_layers.append(MAGNN_ctr_ntype_specific(num_metapaths_list[i],
                                                                  etypes_lists[i],
                                                                  in_dim,
                                                                  num_heads,
                                                                  attn_vec_dim,
                                                                  rnn_type,
                                                                  r_vec,
                                                                  attn_drop,
                                                                  use_minibatch=False))
        # 实际的输入维度需要考虑注意力头的个数
        if fc_switch:
            self.fc1 = nn.Linear(in_dim, out_dim, bias=False)
            self.fc2 = nn.Linear(in_dim * num_heads, out_dim, bias=True)
            nn.init.xavier_normal_(self.fc1.weight, gain=1.414)
            nn.init.xavier_normal_(self.fc2.weight, gain=1.414)
        else:
            self.fc = nn.Linear(in_dim * num_heads, out_dim, bias=True)
            nn.init.xavier_normal_(self.fc.weight, gain=1.414)

    def forward(self, inputs):
        g_lists, features, type_mask, edge_metapath_indices_lists = inputs

        # 为不同的目标节点设置网络层
        h = torch.zeros(type_mask.shape[0], self.in_dim * self.num_heads, device=features.device)
        for i, (g_list, edge_metapath_indices_list, ctr_ntype_layer) in enumerate(
                zip(g_lists, edge_metapath_indices_lists, self.ctr_ntype_layers)):
            h[np.where(type_mask == i)[0]] = ctr_ntype_layer((g_list, features, type_mask, edge_metapath_indices_list))

        if fc_switch:
            h_fc = self.fc1(features) + self.fc2(h)
        else:
            h_fc = self.fc(h)

        return h_fc, h


class MAGNN_nc(nn.Module):
    def __init__(self,
                 num_layers,
                 num_metapaths_list,
                 num_edge_type,
                 etypes_lists,
                 feats_dim_list,
                 hidden_dim,
                 out_dim,
                 num_heads,
                 attn_vec_dim,
                 rnn_type='gru',
                 dropout_rate=0.5):
        super(MAGNN_nc, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        # 特定节点类型的特征转换（转换到同一个空间中）
        self.fc_list = nn.ModuleList([nn.Linear(feats_dim, hidden_dim, bias=True) for feats_dim in feats_dim_list])
        if dropout_rate > 0:
            self.feat_drop = nn.Dropout(dropout_rate)
        else:
            self.feat_drop = lambda x: x
        # 初始化fc
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)

        # MAGNN_nc_layer
        self.layers = nn.ModuleList()
        # hidden layers
        for l in range(num_layers - 1):
            self.layers.append(
                MAGNN_nc_layer(num_metapaths_list, num_edge_type, etypes_lists, hidden_dim, hidden_dim,
                               num_heads, attn_vec_dim, rnn_type, attn_drop=dropout_rate))
        # output projection layer
        self.layers.append(
            MAGNN_nc_layer(num_metapaths_list, num_edge_type, etypes_lists, hidden_dim, out_dim,
                           num_heads, attn_vec_dim, rnn_type, attn_drop=dropout_rate))

    def forward(self, inputs, target_node_indices):
        g_lists, features_list, type_mask, edge_metapath_indices_lists = inputs

        # 特定类型节点转换
        transformed_features = torch.zeros(type_mask.shape[0], self.hidden_dim, device=features_list[0].device)
        for i, fc in enumerate(self.fc_list):
            node_indices = np.where(type_mask == i)[0]
            transformed_features[node_indices] = fc(features_list[i])
        h = self.feat_drop(transformed_features)

        # hidden layers
        for l in range(self.num_layers - 1):
            h, _ = self.layers[l]((g_lists, h, type_mask, edge_metapath_indices_lists))
            h = F.elu(h)
        # output projection layer
        logits, h = self.layers[-1]((g_lists, h, type_mask, edge_metapath_indices_lists))

        # 返回目标节点的logits和embedding
        return logits[target_node_indices], h[target_node_indices]


'''实现RGAT:不需要预定义元路径'''


class RGATConv(nn.Module):
    def __init__(
            self,
            in_feats,  # 输入的节点特征维度
            out_feats,  # 输出的节点特征维度
            edge_feats,  # 输入边的特征维度
            num_heads=1,  # 注意力头数
            feat_drop=0.1,  # 节点特征dropout
            attn_drop=0.1,  # 注意力dropout
            edge_drop=0.1,  # 边特征dropout
            negative_slope=0.2,
            activation=None,
            allow_zero_in_degree=False,
            use_symmetric_norm=False,
    ):
        super(RGATConv, self).__init__()
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        self._use_symmetric_norm = use_symmetric_norm
        if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(self._in_src_feats, out_feats * num_heads, bias=False)
            self.fc_dst = nn.Linear(self._in_dst_feats, out_feats * num_heads, bias=False)
        else:
            self.fc = nn.Linear(self._in_src_feats, out_feats * num_heads, bias=False)

        self.fc_edge = nn.Linear(edge_feats, out_feats * num_heads, bias=False)

        self.attn_l = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_edge = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_r = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))

        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.edge_drop = edge_drop
        self.leaky_relu = nn.LeakyReLU(negative_slope)

        self.reset_parameters()
        self._activation = activation

    # 初始化参数
    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        # gain = 1.414
        if hasattr(self, "fc"):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.fc_edge.weight, gain=gain)

        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        nn.init.xavier_normal_(self.attn_edge, gain=gain)

        '''
        if hasattr(self, "fc"):
            nn.init.kaiming_normal_(self.fc.weight)
        else:
            nn.init.kaiming_normal_(self.fc_src.weight)
            nn.init.kaiming_normal_(self.fc_dst.weight)
        nn.init.kaiming_normal_(self.fc_edge.weight)

        nn.init.kaiming_normal_(self.attn_l)
        nn.init.kaiming_normal_(self.attn_r)
        nn.init.kaiming_normal_(self.attn_edge)        
        '''

    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat):
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    assert False

            # feat[0]源节点的特征
            # feat[1]目标节点的特征
            # h_edge 边的特征
            h_src = self.feat_drop(feat[0])
            h_dst = self.feat_drop(feat[1])
            h_edge = self.feat_drop(graph.edata['feature'])

            if not hasattr(self, "fc_src"):
                self.fc_src, self.fc_dst = self.fc, self.fc

            # 特征赋值
            feat_src, feat_dst, feat_edge = h_src, h_dst, h_edge
            # 转换成多头注意力的形状
            feat_src = self.fc_src(h_src).view(-1, self._num_heads, self._out_feats)
            feat_dst = self.fc_dst(h_dst).view(-1, self._num_heads, self._out_feats)
            feat_edge = self.fc_edge(h_edge).view(-1, self._num_heads, self._out_feats)

            # NOTE: GAT paper uses "first concatenation then linear projection"
            # to compute attention scores, while ours is "first projection then
            # addition", the two approaches are mathematically equivalent:
            # We decompose the weight vector a mentioned in the paper into
            # [a_l || a_r], then
            # a^T [Wh_i || Wh_j] = a_l Wh_i + a_r Wh_j
            # Our implementation is much efficient because we do not need to
            # save [Wh_i || Wh_j] on edges, which is not memory-efficient. Plus,
            # addition could be optimized with DGL's built-in function u_add_v,
            # which further speeds up computation and saves memory footprint.
            # 简单来说就是拼接矩阵相乘和拆开分别矩阵相乘再相加的效果是一样的
            # 但是前者更加高效

            # 左节点的注意力权重
            el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
            graph.srcdata.update({"ft": feat_src, "el": el})
            # 右节点的注意力权重
            er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
            graph.dstdata.update({"er": er})
            # 左节点权重+右节点权重 = 节点计算出的注意力权重（e）
            graph.apply_edges(fn.u_add_v("el", "er", "e"))

            # 边计算出来的注意力权重
            ee = (feat_edge * self.attn_edge).sum(dim=-1).unsqueeze(-1)
            # 边注意力权重加上节点注意力权重得到最终的注意力权重
            # 这里可能应该也是和那个拼接操作等价吧
            graph.edata.update({"e": graph.edata["e"] + ee})
            # 经过激活函数，一起激活和分别激活可能也是等价吧
            e = self.leaky_relu(graph.edata["e"])

            # 注意力权重的正则化
            if self.training and self.edge_drop > 0:
                perm = torch.randperm(graph.number_of_edges(), device=graph.device)
                bound = int(graph.number_of_edges() * self.edge_drop)
                eids = perm[bound:]
                a = torch.zeros_like(e)
                a[eids] = self.attn_drop(edge_softmax(graph, e[eids], eids=eids))
                graph.edata.update({"a": a})
            else:
                graph.edata["a"] = self.attn_drop(edge_softmax(graph, e))

            # 消息传递
            graph.update_all(fn.u_mul_e("ft", "a", "m"), fn.sum("m", "ft"))
            rst = graph.dstdata["ft"]

            # 标准化
            degs = graph.in_degrees().float().clamp(min=1)
            norm = torch.pow(degs, -1)
            shp = norm.shape + (1,) * (feat_dst.dim() - 1)
            norm = torch.reshape(norm, shp)
            rst = rst * norm

        return rst


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
            # {rel: RGATConv(in_feats, out_feats // num_heads, in_feats, num_heads, feat_drop=0.2, attn_drop=0.2) for rel in rel_names},
            {rel: GATConv(in_feats, out_feats // num_heads, num_heads, feat_drop=0.2, attn_drop=0.2) for rel in
             rel_names},
            aggregate='mean')  # 一个关系对应一个RGATConv层
        self.out_feats = out_feats
        self.act = act

    def forward(self, dataset, graph, inputs):  # inputs是三维
        outputs = []
        for input in inputs:
            output = []
            h = self.layer(graph, input)
            # h = {k: self.act(v).view(-1, self.out_feats) for k, v in h.items}
            # print("hhhhh")
            # print(h['paper'].shape)
            # key =['paper','author']
            if dataset == 'ACM':
                key = ['paper', 'author', 'subject']
            elif dataset == 'DBLP':
                key = ['author', 'paper', 'term', 'venue']
            elif dataset == 'IMDB':
                key = ['movie', 'director', 'actor']
            elif dataset == 'YELP':
                key = ['business', 'user', 'service', 'level']
            elif dataset == 'LastFM':
                key = ['user', 'artist', 'tag']
            for k in key:
                #out = self.act(h[k]).view(-1, self.out_feats)
                #print("))))))))))))))))))))")
                #print(self.act(h[k]).shape)
                out = self.act(h[k]).reshape(-1, self.out_feats)
                #out = nn.Flatten()(self.act(h[k]))
                # print(k)
                output.append(out)
                # print(out.shape)
            output = torch.cat(output, dim=0)
            # print(output.shape)
            outputs.append(output)
        outputs = torch.stack(outputs, dim=0)
        # print(outputs.shape)
        '''
            for k, v in h.items():
                out = self.act(v).view(-1, self.out_feats)
                print(k)
                output.append(out)
                print(out.shape)
            output = torch.cat(output, dim=0)
            print(output.shape)
            outputs.append(output)
        outputs = torch.stack(outputs, dim=0)
        print(outputs.shape)        
        '''

        return outputs  # 返回所有类型节点的embedding


class RGAT(nn.Module):
    def __init__(self,
                 in_feats,  # 输入维度
                 hid_feats,  # 隐藏层维度
                 out_feats,  # 输出维度
                 num_heads,  # 注意力头
                 rel_names,  # 用于异构图卷积，关系名称
                 ):
        super().__init__()
        self.conv1 = HeteroGraphConv(
            {rel: RGATConv(in_feats, hid_feats // num_heads, in_feats, num_heads) for rel in rel_names},
            aggregate='sum')
        self.conv2 = HeteroGraphConv({rel: RGATConv(hid_feats, out_feats, in_feats, num_heads) for rel in rel_names},
                                     aggregate='mean')
        self.hid_feats = hid_feats

    def forward(self, graph, inputs):  # inputs是三维的
        h = self.conv1(graph, inputs)  # 第一层异构卷积
        h = {k: F.relu(v).view(-1, self.hid_feats) for k, v in h.items()}
        h = self.conv2(graph, h)  # 第二层异构卷积
        return h
