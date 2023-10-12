import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as tdist
import numpy as np
from model.layers import Dense,  RGATLayer
from utils.tools import evaluate_results_nc
from torch.nn.parameter import Parameter
import warnings
from utils.preprocessing import feature_in_graph
import numpy as np
torch.set_printoptions(profile="full")
warnings.filterwarnings("ignore")


def MinMaxScalar(x):
    min_vals, _ = torch.min(x, dim=1, keepdim=True)
    max_vals, _ = torch.max(x, dim=1, keepdim=True)

    scaled_x = (x - min_vals) / (max_vals - min_vals)
    return scaled_x


def sigmoid(x):
    if x >= 0:
        return 1 / (1 + np.exp(-x))
    else:
        return np.exp(x) / (1 + np.exp(x))


class GraMI(nn.Module):
    def __init__(self, dataset, graph, src_node, feats_dim_list, num_nodes, ndim, input_feat_dim, hidden_dim,
                 hidden_dim1,
                 hidden_dim2,
                 num_heads,
                 dropout,
                 encsto='semi', gdc='ip',
                 ndist='Bernoulli', copyK=1, copyJ=1, device='cuda'):
        super(GraMI, self).__init__()
        self.dataset = dataset
        self.fc_list = nn.ModuleList([nn.Linear(feats_dim, hidden_dim, bias=True) for feats_dim in feats_dim_list])
        if dropout > 0:
            self.feat_drop = nn.Dropout(dropout)
        else:
            self.feat_drop = lambda x: x
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)
        ndim = hidden_dim1
        self.ac = base_HGNN(dataset, graph, num_nodes, ndim, input_feat_dim, hidden_dim, hidden_dim1,
                            hidden_dim2,
                            num_heads,
                            dropout,
                            src_node,
                            encsto, gdc, ndist, copyK, copyJ, device)

    def forward(self, features_list, graph):
        x_all = []
        for i in range(len(features_list)):
            x_all.append(torch.tanh(self.feat_drop(self.fc_list[i](features_list[i]))))
        for i in range(len(x_all)):
            x_all[i] = x_all[i].view([1, x_all[i].shape[0], x_all[i].shape[1]])
        x = torch.cat(x_all, dim=1)
        label_a = x.squeeze(dim=0)
        x_all = torch.cat(x_all, dim=1)
        ac = self.ac(x, graph, x_all)

        return label_a, ac


class base_HGNN(nn.Module):
    def __init__(self, dataset, graph, num_nodes, ndim, input_feat_dim, hidden_dim, hidden_dim1, hidden_dim2, num_heads,
                 dropout,
                 src_node,
                 encsto='semi',
                 gdc='ip',
                 ndist='Bernoulli', copyK=1, copyJ=1, device='cuda'):
        super(base_HGNN, self).__init__()
        self.dataset = dataset
        self.n_samples = num_nodes
        self.graph = graph
        self.rel_names = graph.etypes
        # node embedding
        ndim = hidden_dim
        self.gat_e = RGATLayer(in_feats=ndim, out_feats=hidden_dim1, num_heads=num_heads, rel_names=self.rel_names,
                               act=F.relu)
        self.gat_1 = RGATLayer(in_feats=hidden_dim, out_feats=hidden_dim1, num_heads=num_heads,
                               rel_names=self.rel_names,
                               act=F.relu)
        self.gat_2 = RGATLayer(in_feats=hidden_dim1, out_feats=hidden_dim2, num_heads=num_heads,
                               rel_names=self.rel_names,
                               act=lambda x: x)
        self.gat_3 = RGATLayer(in_feats=hidden_dim1, out_feats=hidden_dim2, num_heads=num_heads,
                               rel_names=self.rel_names,
                               act=lambda x: x)
        # feature embedding
        self.mlpe = Dense(input_dim=ndim, output_dim=hidden_dim1, dropout=dropout, act=torch.tanh)
        self.mlp1 = Dense(input_dim=self.n_samples, output_dim=hidden_dim1, act=torch.tanh)
        self.mlp2 = Dense(input_dim=hidden_dim1, output_dim=hidden_dim2, dropout=dropout, act=lambda x: x)
        self.mlp3 = Dense(input_dim=hidden_dim1, output_dim=hidden_dim2, dropout=dropout, act=lambda x: x)
        self.mlp_recover = Dense(input_dim=hidden_dim, output_dim=input_feat_dim, dropout=dropout,
                                 act=torch.sigmoid)
        self.encsto = encsto
        self.dc = GraphDecoder(hidden_dim2, dropout, gdc=gdc)
        self.dc2 = GraphDecoder2(hidden_dim2, dropout, gdc=gdc)
        self.device = device

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
        graph_noise = graph
        inputs = []
        graph_noise = feature_in_graph(self.dataset, graph_noise, x_all, device=self.device)
        inputs.append(graph_noise.ndata['feature'])
        hiddenx = self.gat_1(self.dataset, graph_noise, inputs)

        noise = []
        if self.ndim >= 1:
            for i in range(len(x_all)):
                e = self.ndist.sample(torch.Size([self.K + self.J, x_all[i].shape[0], self.ndim]))
                e = torch.squeeze(e, -1)
                e = e.mul(self.reweight)
                noise.append(e)
            noise = torch.cat(noise, dim=1)
            inputs = []
            for h in noise:
                h = torch.unsqueeze(h, 0)
                graph_noise = feature_in_graph(self.dataset, graph_noise, h, device=self.device)
                inputs.append(graph_noise.ndata['feature'])
            hiddene = self.gat_e(self.dataset, graph_noise, inputs)
        else:
            print("no randomness.")
            for x in hiddenx:
                e = torch.zeros(self.K + self.J, device=self.device)
        hidden1 = hiddenx + hiddene
        inputs = []
        for h in hidden1:
            h = torch.unsqueeze(h, 0)
            graph_noise = feature_in_graph(self.dataset, graph_noise, h, device=self.device)
            inputs.append(graph_noise.ndata['feature'])
        mu = self.gat_2(self.dataset, graph_noise, inputs)

        EncSto = (self.encsto == 'full')
        hidden_std1 = EncSto * hidden1 + (1 - EncSto) * hiddenx
        inputs = []
        for h in hidden_std1:
            h = torch.unsqueeze(h, 0)
            graph_noise = feature_in_graph(self.dataset, graph_noise, h, device=self.device)
            inputs.append(graph_noise.ndata['feature'])
        logvar = self.gat_3(self.dataset, graph_noise, inputs)

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

    def feature_encode(self, x):
        assert len(x.shape) == 3, 'The input tensor dimension is not 3!'
        f = torch.transpose(x, 1, 2)
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

        EncSto = (self.encsto == 'full')
        hidden_sd = EncSto * hidden1 + (1 - EncSto) * hiddenf
        logvarf = self.mlp3(hidden_sd)

        return muf, logvarf

    def reparameterize(self, mu, logvar):
        std = torch.exp(logvar / 2.)
        eps = torch.randn_like(std)
        return eps.mul(std).add(mu), eps

    def forward(self, x, graph, x_all):
        src_node = self.src_node
        mu_list, logvar_list = self.node_encode(graph, x_all)

        muf, logvarf = self.feature_encode(x)

        emb_mu_list, emb_logvar_list = [], []
        for mu in mu_list:
            emb_mu_list.append(mu[self.K:, :])
        for logvar in logvar_list:
            emb_logvar_list.append(logvar[self.K:, :])

        emb_muf = muf[self.K:, :]
        emb_logvarf = logvarf[self.K:, :]

        assert len(emb_mu_list[1].shape) == len(emb_logvar_list[1].shape), 'mu and logvar are not equi-dimension.'
        z_all, eps_all = [], []
        for emb_mu, emb_logvar in zip(emb_mu_list, emb_logvar_list):
            z, eps = self.reparameterize(emb_mu, emb_logvar)
            z_all.append(z)
            eps_all.append(eps)

        zf, epsf = self.reparameterize(emb_muf, emb_logvarf)
        pred_adj_all, z_scaled1_all, z_scaled2_all, rk_all = [], [], [], []

        for i in range(len(z_all)):
            if i == src_node:
                continue
            else:
                adj_, z_scaled1, z_scaled2, rk = self.dc(z_all[src_node], z_all[i])  # node embedding
                pred_adj_all.append(adj_)
                z_scaled1_all.append(z_scaled1)
                z_scaled2_all.append(z_scaled2)
                rk_all.append(rk)
        Za = torch.cat(z_all, dim=1)
        pred_a, z_scaleda, z_scaledf, _ = self.dc2(Za, zf)  # feature embedding
        fea_recover = pred_a[:, :mu_list[src_node].shape[1], :] if src_node == 0 else pred_a[:,
                                                                                      mu_list[src_node - 1].shape[1]:
                                                                                      mu_list[src_node - 1].shape[1] +
                                                                                      mu_list[src_node].shape[1], :]
        fea_recover = self.mlp_recover(fea_recover)

        return pred_adj_all, pred_a, fea_recover, mu_list, muf, logvar_list, logvarf, z_all, zf, Za, z_scaled1_all, z_scaled2_all, z_scaledf, z_scaleda, eps_all, epsf, rk_all


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

        rk = torch.sigmoid(self.rk_lgt).pow(.5)

        adj_lgt = torch.bmm(z1, torch.transpose(z2, 1, 2))

        if self.gdc == 'ip':
            adj = torch.sigmoid(adj_lgt)
        elif self.gdc == 'bp':
            adj_lgt = torch.clamp(adj_lgt, min=-np.Inf, max=25)
            adj = 1 - torch.exp(-adj_lgt.exp())

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
        rk = torch.sigmoid(self.rk_lgt).pow(.5)

        adj_lgt = torch.bmm(z1, torch.transpose(z2, 1, 2))

        if self.gdc == 'ip':
            adj = torch.tanh(adj_lgt)
        elif self.gdc == 'bp':
            adj_lgt = torch.clamp(adj_lgt, min=-np.Inf, max=25)
            adj = 1 - torch.exp(-adj_lgt.exp())
        if not self.training:
            adj = torch.mean(adj, dim=0, keepdim=True)

        return adj, z1, z2, rk.pow(2)
