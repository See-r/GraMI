import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as tdist
import numpy as np
from model.layers import GraphConvolution, Dense, ProjectLayer, RGATLayer
from utils.tools import evaluate_results_nc
from torch.nn.parameter import Parameter
import warnings
from utils.preprocessing import feature_in_graph
import numpy as np
# np.set_printoptions(threshold=np.inf)
torch.set_printoptions(profile="full")
warnings.filterwarnings("ignore")


def MinMaxScalar(x):
    # 沿着行的方向计算最小值和最大值
    min_vals, _ = torch.min(x, dim=1, keepdim=True)
    max_vals, _ = torch.max(x, dim=1, keepdim=True)

    scaled_x = (x - min_vals) / (max_vals - min_vals)
    return scaled_x


def sigmoid(x):
    if x >= 0:
        return 1 / (1 + np.exp(-x))
    else:
        return np.exp(x) / (1 + np.exp(x))


####### 带特征补全的SIG_VAE


# 使用RGAT的final Model
class final_model_RGAT(nn.Module):
    def __init__(self, dataset, graph, src_node, feats_dim_list, num_nodes, ndim, input_feat_dim, hidden_dim,
                 hidden_dim1,
                 hidden_dim2,
                 num_heads,
                 dropout,
                 encsto='semi', gdc='ip',
                 ndist='Bernoulli', copyK=1, copyJ=1, device='cuda'):
        super(final_model_RGAT, self).__init__()
        self.dataset = dataset
        # 添加线性层，对特征进行映射,feats_dim_list为所有特征维度,input_feat_dim为原始特征维度4000,hidden_dim为共同隐空间维度
        self.fc_list = nn.ModuleList([nn.Linear(feats_dim, hidden_dim, bias=True) for feats_dim in feats_dim_list])
        if dropout > 0:
            self.feat_drop = nn.Dropout(dropout)
        else:
            self.feat_drop = lambda x: x
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)
        ndim = hidden_dim1
        # sigvae进行链路预测+特征重构
        self.ac = SIG_VAE_AC_Independent_RGAT(dataset, graph, num_nodes, ndim, input_feat_dim, hidden_dim, hidden_dim1,
                                              hidden_dim2,
                                              num_heads,
                                              dropout,
                                              src_node,
                                              encsto, gdc, ndist, copyK, copyJ, device)

    # x为两种类型节点的共同特征矩阵
    def forward(self, features_list, graph):  # features_list为所有类型特征的list,graph为异构图，relation为选择预测的关系
        '''
        print("final_model:x2.shape")
        print(x1.shape)
        print(x2.shape)
        print(len(x2))#1
        '''
        # 将所有特征变为同一空间的
        x_all = []  # 所有特征
        for i in range(len(features_list)):
            x_all.append(torch.tanh(self.feat_drop(self.fc_list[i](features_list[i]))))
            # x_all.append(F.elu(self.feat_drop(self.fc_list[i](features_list[i]))))za
        for i in range(len(x_all)):
            x_all[i] = x_all[i].view([1, x_all[i].shape[0], x_all[i].shape[1]])
        # 进行特征重构的x,torch.Size([1, 11246, 4000])全部特征进行重构
        x = torch.cat(x_all, dim=1)
        label_a = x.squeeze(dim=0)  # torch.Size([11246, 4000])
        x_all = torch.cat(x_all, dim=1)  # torch.Size([1, 11246, 4000]),type=torch.Tensor
        # graph = feature_in_graph_ACM(graph, x_all)  # 为graph节点添加特征
        ac = self.ac(x, graph, x_all)

        return label_a, ac


# 采用RGAT作为encoder,特征使用MLP
class SIG_VAE_AC_Independent_RGAT(nn.Module):
    def __init__(self, dataset, graph, num_nodes, ndim, input_feat_dim, hidden_dim, hidden_dim1, hidden_dim2, num_heads,
                 dropout,
                 src_node,
                 encsto='semi',
                 gdc='ip',
                 ndist='Bernoulli', copyK=1, copyJ=1, device='cuda'):
        # hidden_dim可以看作input_feat_dim
        super(SIG_VAE_AC_Independent_RGAT, self).__init__()
        self.dataset = dataset
        self.n_samples = num_nodes
        self.graph = graph
        self.rel_names = graph.etypes
        # node embedding
        # 使用RGAT进行embedding，同时对所有节点进行
        ndim = hidden_dim
        print("------hidden_dim,hidden_dim1,hidden_dim2------")
        print(hidden_dim, hidden_dim1, hidden_dim2)
        print("------num_heads,dropout,K,J-------")
        print(num_heads, dropout, copyK, copyJ)
        self.gat_e = RGATLayer(in_feats=ndim, out_feats=hidden_dim1, num_heads=num_heads, rel_names=self.rel_names,
                               act=F.relu)  # 噪声层
        self.gat_1 = RGATLayer(in_feats=hidden_dim, out_feats=hidden_dim1, num_heads=num_heads,
                               rel_names=self.rel_names,
                               act=F.relu)  # 共享层
        self.gat_2 = RGATLayer(in_feats=hidden_dim1, out_feats=hidden_dim2, num_heads=num_heads,
                               rel_names=self.rel_names,
                               act=lambda x: x)  # 均值
        self.gat_3 = RGATLayer(in_feats=hidden_dim1, out_feats=hidden_dim2, num_heads=num_heads,
                               rel_names=self.rel_names,
                               act=lambda x: x)  # 方差
        # feature embedding
        self.mlpe = Dense(input_dim=ndim, output_dim=hidden_dim1, dropout=dropout, act=torch.tanh)  # 特征噪声层
        self.mlp1 = Dense(input_dim=self.n_samples, output_dim=hidden_dim1, act=torch.tanh)
        self.mlp2 = Dense(input_dim=hidden_dim1, output_dim=hidden_dim2, dropout=dropout, act=lambda x: x)  # 均值
        self.mlp3 = Dense(input_dim=hidden_dim1, output_dim=hidden_dim2, dropout=dropout, act=lambda x: x)  # 方差
        self.mlp_recover = Dense(input_dim=hidden_dim, output_dim=input_feat_dim, dropout=dropout,
                                 act=torch.sigmoid)  # 复原paper节点的线性层
        if dataset == 'ACM':
            self.adj_weight_0 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
            self.adj_weight_1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
            self.adj_weight_0.data.fill_(1.0)
            self.adj_weight_1.data.fill_(1.0)
        else:
            self.adj_weight_0 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
            self.adj_weight_1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
            self.adj_weight_2 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
            self.adj_weight_0.data.fill_(1.0)
            self.adj_weight_1.data.fill_(1.0)
            self.adj_weight_2.data.fill_(1.0)
        self.encsto = encsto
        self.dc = GraphDecoder(hidden_dim2, dropout, gdc=gdc)
        self.dc2 = GraphDecoder2(hidden_dim2, dropout, gdc=gdc)
        self.device = device

        # 噪声分布
        if ndist == 'Bernoulli':
            self.ndist = tdist.Bernoulli(torch.tensor([.5], device=self.device))
        elif ndist == 'Normal':
            self.ndist = tdist.Normal(
                torch.tensor([0.], device=self.device),
                torch.tensor([1.], device=self.device)
            )
        elif ndist == 'Exponential':
            self.ndist = tdist.Exponential(torch.tensor([1.], device=self.device))

        self.K = copyK
        self.J = copyJ
        self.ndim = ndim
        self.src_node = src_node
        self.reweight = ((self.ndim + hidden_dim1) / (hidden_dim + hidden_dim1)) ** (.5)

    def node_encode(self, graph, x_all):
        # inputs为图graph的特征矩阵，再加一个维度
        # x_all为构造的全部类型节点的features_list
        graph_noise = graph
        inputs = []
        graph_noise = feature_in_graph(self.dataset, graph_noise, x_all, device=self.device)
        # graph_noise = feature_in_graph_ACM(graph_noise, x_all, device=self.device)
        inputs.append(graph_noise.ndata['feature'])  # len(inputs)=1
        hiddenx = self.gat_1(self.dataset, graph_noise, inputs)

        noise = []
        if self.ndim >= 1:
            # 生成噪声，要求跟特征维度一样
            for i in range(len(x_all)):
                e = self.ndist.sample(torch.Size([self.K + self.J, x_all[i].shape[0], self.ndim]))
                e = torch.squeeze(e, -1)
                e = e.mul(self.reweight)
                noise.append(e)
            noise = torch.cat(noise, dim=1)  # torch.Size([8, 11246, 64])
            inputs = []
            for h in noise:  # h:torch.Size([11246, 64])
                h = torch.unsqueeze(h, 0)  # h:torch.Size([1, 11246, 64])
                graph_noise = feature_in_graph(self.dataset, graph_noise, h, device=self.device)
                # graph_noise = feature_in_graph_ACM(graph_noise, h, device=self.device)
                inputs.append(graph_noise.ndata['feature'])
            hiddene = self.gat_e(self.dataset, graph_noise, inputs)
        else:
            print("no randomness.")
            for x in hiddenx:
                e = torch.zeros(self.K + self.J, device=self.device)
        # 在特征中加入噪声，使之满足随机性
        hidden1 = hiddenx + hiddene  # torch.Size([K+J, 11246, 64])对于ACM顺序是paper,author,subject
        # DBLP顺序是author,paper,term,venue
        # hidden1 = hidden1.to(self.device)
        # 计算噪声特征的均值
        inputs = []
        for h in hidden1:
            h = torch.unsqueeze(h, 0)
            graph_noise = feature_in_graph(self.dataset, graph_noise, h, device=self.device)
            # graph_noise = feature_in_graph_ACM(graph_noise, h, device=self.device)
            inputs.append(graph_noise.ndata['feature'])
        mu = self.gat_2(self.dataset, graph_noise, inputs)

        EncSto = (self.encsto == 'full')  # 方差是否随机
        hidden_std1 = EncSto * hidden1 + (1 - EncSto) * hiddenx
        # 计算噪声特征的方差
        inputs = []
        for h in hidden_std1:
            h = torch.unsqueeze(h, 0)
            graph_noise = feature_in_graph(self.dataset, graph_noise, h, device=self.device)
            # graph_noise = feature_in_graph_ACM(graph_noise, h, device=self.device)
            inputs.append(graph_noise.ndata['feature'])
        logvar = self.gat_3(self.dataset, graph_noise, inputs)

        # ACM
        '''
        mu = torch.split(mu, [graph.number_of_nodes('paper'), graph.number_of_nodes('author')], dim=1)
        logvar = torch.split(logvar, [graph.number_of_nodes('paper'), graph.number_of_nodes('author')], dim=1)                                      
        '''
        if self.dataset == 'ACM':
            mu = torch.split(mu, [graph.number_of_nodes('paper'), graph.number_of_nodes('author'),
                                  graph.number_of_nodes('subject')], dim=1)
            logvar = torch.split(logvar, [graph.number_of_nodes('paper'), graph.number_of_nodes('author'),
                                          graph.number_of_nodes('subject')], dim=1)
        elif self.dataset == 'DBLP':
            mu = torch.split(mu, [graph.number_of_nodes('author'), graph.number_of_nodes('paper'),
                                  graph.number_of_nodes('term'), graph.number_of_nodes('venue')], dim=1)
            logvar = torch.split(logvar, [graph.number_of_nodes('author'), graph.number_of_nodes('paper'),
                                          graph.number_of_nodes('term'), graph.number_of_nodes('venue')], dim=1)
        elif self.dataset == 'YELP':
            mu = torch.split(mu, [graph.number_of_nodes('business'), graph.number_of_nodes('user'),
                                  graph.number_of_nodes('service'), graph.number_of_nodes('level')], dim=1)
            logvar = torch.split(logvar, [graph.number_of_nodes('business'), graph.number_of_nodes('user'),
                                          graph.number_of_nodes('service'), graph.number_of_nodes('level')], dim=1)


        return mu, logvar

    def feature_encode(self, x):  # x是全部的节点特征
        assert len(x.shape) == 3, 'The input tensor dimension is not 3!'
        f = torch.transpose(x, 1, 2)  # 维度一二转置
        hiddenf = self.mlp1(f)

        if self.ndim >= 1:
            e = self.ndist.sample(torch.Size([self.K + self.J, f.shape[1], self.ndim]))
            e = torch.squeeze(e, -1)
            e = e.mul(self.reweight)
            hiddene = self.mlpe(e)
        else:
            print("no randomness.")
            hiddene = torch.zeros(self.K + self.J, hiddenf.shape[1], hiddenf.shape[2], device=self.device)

        hidden1 = hiddenf + hiddene

        muf = self.mlp2(hidden1)

        EncSto = (self.encsto == 'full')  # encsto="semi"
        hidden_sd = EncSto * hidden1 + (1 - EncSto) * hiddenf
        logvarf = self.mlp3(hidden_sd)

        return muf, logvarf

    def reparameterize(self, mu, logvar):
        std = torch.exp(logvar / 2.)
        eps = torch.randn_like(std)
        return eps.mul(std).add(mu), eps

    def forward(self, x, graph, x_all):  # x为需要重构的特征，graph为异构图(有特征）,node_num_list是所有节点个数的list
        src_node = self.src_node
        mu_list, logvar_list = self.node_encode(graph, x_all)

        muf, logvarf = self.feature_encode(x)

        emb_mu_list, emb_logvar_list = [], []
        for mu in mu_list:
            emb_mu_list.append(mu[self.K:, :])  # J个似然样本
        for logvar in logvar_list:
            emb_logvar_list.append(logvar[self.K:, :])
        '''
        print("------------------emb_mu1-------------------")
        print("emb_mu1")  # torch.Size([5, 4019, 32])    
        '''

        emb_muf = muf[self.K:, :]
        emb_logvarf = logvarf[self.K:, :]
        '''
        print("------------------emb_muf-------------------")
        print("emb_muf")  # torch.Size([5, 4000, 32])  
        '''

        assert len(emb_mu_list[1].shape) == len(emb_logvar_list[1].shape), 'mu and logvar are not equi-dimension.'
        z_all, eps_all = [], []  # 分别存储重参数化的所有类型节点的z和psi列表
        # 重参数化
        for emb_mu, emb_logvar in zip(emb_mu_list, emb_logvar_list):
            z, eps = self.reparameterize(emb_mu, emb_logvar)
            z_all.append(z)
            eps_all.append(eps)

        zf, epsf = self.reparameterize(emb_muf, emb_logvarf)
        pred_adj_all, z_scaled1_all, z_scaled2_all, rk_all = [], [], [], []  # 预测所有关系的邻接矩阵,对于每种关系left节点的隐变量，对于每种关系right节点的隐变量

        for i in range(len(z_all)):
            if i == src_node:
                continue
            else:
                adj_, z_scaled1, z_scaled2, rk = self.dc(z_all[src_node], z_all[i])  # node embedding解码
                pred_adj_all.append(adj_)
                z_scaled1_all.append(z_scaled1)
                z_scaled2_all.append(z_scaled2)
                rk_all.append(rk)
        Za = torch.cat(z_all, dim=1)
        pred_a, z_scaleda, z_scaledf, _ = self.dc2(Za, zf)  # feature embedding 解码
        fea_recover = pred_a[:, :mu_list[src_node].shape[1], :] if src_node == 0 else pred_a[:,
                                                                                      mu_list[src_node - 1].shape[1]:
                                                                                      mu_list[src_node - 1].shape[1] +
                                                                                      mu_list[src_node].shape[1], :]
        fea_recover = self.mlp_recover(fea_recover)  # 用loss_function_a_mse计算pred_recover和fea_orig的值

        return pred_adj_all, pred_a, fea_recover, mu_list, muf, logvar_list, logvarf, z_all, zf, Za, z_scaled1_all, z_scaled2_all, z_scaledf, z_scaleda, eps_all, epsf, rk_all, adj_weight


# 解码器
class GraphDecoder(nn.Module):
    """Decoder for using inner product for prediction."""

    def __init__(self, zdim, dropout, gdc='ip'):
        super(GraphDecoder, self).__init__()
        self.dropout = dropout
        self.gdc = gdc
        self.zdim = zdim
        self.rk_lgt = Parameter(torch.FloatTensor(torch.Size([1, zdim])))
        self.reset_parameters()
        self.SMALL = 1e-16

    def reset_parameters(self):
        torch.nn.init.uniform_(self.rk_lgt, a=-6., b=0.)

    def forward(self, z1, z2):
        z1 = F.dropout(z1, self.dropout, training=self.training)
        z2 = F.dropout(z2, self.dropout, training=self.training)
        assert self.zdim == z1.shape[2], 'zdim not compatible!'

        # The variable 'rk' in the code is the square root of the same notation in
        # http://proceedings.mlr.press/v80/yin18b/yin18b-supp.pdf
        # i.e., instead of do Z*diag(rk)*Z', we perform [Z*diag(rk)] * [Z*diag(rk)]'.
        rk = torch.sigmoid(self.rk_lgt).pow(.5)

        # Z shape: [J, N, zdim]
        # Z' shape: [J, zdim, N]

        '''
        if self.gdc == 'bp':
            z = z.mul(rk.view(1, 1, self.zdim))        
        '''

        adj_lgt = torch.bmm(z1, torch.transpose(z2, 1, 2))

        if self.gdc == 'ip':
            adj = torch.sigmoid(adj_lgt)
        elif self.gdc == 'bp':
            # 1 - exp( - exp(ZZ'))
            adj_lgt = torch.clamp(adj_lgt, min=-np.Inf, max=25)
            adj = 1 - torch.exp(-adj_lgt.exp())

        # if self.training:
        #     adj_lgt = - torch.log(1 / (adj + self.SMALL) - 1 + self.SMALL)
        # else:
        #     adj_mean = torch.mean(adj, dim=0, keepdim=True)
        #     adj_lgt = - torch.log(1 / (adj_mean + self.SMALL) - 1 + self.SMALL)

        if not self.training:
            adj = torch.mean(adj, dim=0, keepdim=True)

        return adj, z1, z2, rk.pow(2)


# feature decoder
class GraphDecoder2(nn.Module):
    """Decoder for using inner product for prediction."""

    def __init__(self, zdim, dropout, gdc='ip'):
        super(GraphDecoder2, self).__init__()
        self.dropout = dropout
        self.gdc = gdc
        self.zdim = zdim
        self.rk_lgt = Parameter(torch.FloatTensor(torch.Size([1, zdim])))
        self.reset_parameters()
        self.SMALL = 1e-16

    def reset_parameters(self):
        torch.nn.init.uniform_(self.rk_lgt, a=-6., b=0.)

    def forward(self, z1, z2):
        z1 = F.dropout(z1, self.dropout, training=self.training)
        z2 = F.dropout(z2, self.dropout, training=self.training)
        assert self.zdim == z1.shape[2], 'zdim not compatible!'

        # The variable 'rk' in the code is the square root of the same notation in
        # http://proceedings.mlr.press/v80/yin18b/yin18b-supp.pdf
        # i.e., instead of do Z*diag(rk)*Z', we perform [Z*diag(rk)] * [Z*diag(rk)]'.
        rk = torch.sigmoid(self.rk_lgt).pow(.5)

        # Z shape: [J, N, zdim]
        # Z' shape: [J, zdim, N]

        '''
        if self.gdc == 'bp':
            z = z.mul(rk.view(1, 1, self.zdim))        
        '''

        adj_lgt = torch.bmm(z1, torch.transpose(z2, 1, 2))

        if self.gdc == 'ip':
            adj = torch.tanh(adj_lgt)
            # adj = torch.sigmoid(adj_lgt)
        elif self.gdc == 'bp':
            # 1 - exp( - exp(ZZ'))
            adj_lgt = torch.clamp(adj_lgt, min=-np.Inf, max=25)
            adj = 1 - torch.exp(-adj_lgt.exp())

        # if self.training:
        #     adj_lgt = - torch.log(1 / (adj + self.SMALL) - 1 + self.SMALL)
        # else:
        #     adj_mean = torch.mean(adj, dim=0, keepdim=True)
        #     adj_lgt = - torch.log(1 / (adj_mean + self.SMALL) - 1 + self.SMALL)

        if not self.training:
            adj = torch.mean(adj, dim=0, keepdim=True)

        return adj, z1, z2, rk.pow(2)
