import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as tdist
import numpy as np
from others import args
from model.layers import GraphConvolution, Dense, HANLayer, ProjectLayer, RGATLayer
from torch.nn.parameter import Parameter
import warnings
from utils.preprocessing import feature_in_graph

# np.set_printoptions(threshold=np.inf)
torch.set_printoptions(profile="full")
warnings.filterwarnings("ignore")


####### 带特征补全的SIG_VAE

# 使用GCN的final model,总model对特征进行线性映射+SIG_VAE_AC
class final_model(nn.Module):
    def __init__(self, x_feats_dim, num_nodes, ndim, input_feat_dim, hidden_dim1, hidden_dim2, dropout, method='In',
                 encsto='semi', gdc='ip',
                 ndist='Bernoulli', copyK=1, copyJ=1, device='cuda'):
        super(final_model, self).__init__()
        # 添加线性层，对特征进行映射,x_feats_dim表示另一个one-hot节点的特征维度
        self.fc = nn.Linear(x_feats_dim, input_feat_dim, bias=True)
        nn.init.xavier_uniform_(self.fc.weight, gain=1.414)

        if dropout > 0:
            self.feat_drop = nn.Dropout(dropout)
        else:
            self.feat_drop = lambda x: x
        # sigvae进行链路预测+特征重构
        if method == 'In':
            self.ac = SIG_VAE_AC_Independent(num_nodes, ndim, input_feat_dim, hidden_dim1, hidden_dim2, dropout, encsto,
                                             gdc, ndist, copyK, copyJ, device)
        elif method == 'Un':
            self.ac = SIG_VAE_AC_Independent(num_nodes, ndim, input_feat_dim, hidden_dim1, hidden_dim2, dropout, encsto,
                                             gdc, ndist, copyK, copyJ, device)
        elif method == 'MLP':
            self.ac = SIG_VAE_AC_Independent_MLP(num_nodes, ndim, input_feat_dim, hidden_dim1, hidden_dim2, dropout,
                                                 encsto,
                                                 gdc, ndist, copyK, copyJ, device)

    # x为两种类型节点的共同特征矩阵
    def forward(self, x1, x2, adj1, adj2):  # x2为one-hot
        '''
        print("final_model:x2.shape")
        print(x1.shape)
        print(x2.shape)
        print(len(x2))#1
        '''
        # x2变成了自己构造的
        for i in range(len(x2)):
            x2 = torch.tanh(self.feat_drop(self.fc(x2)))
        x = torch.cat((x1, x2), dim=1)
        # adj_, pred_a, mu1, mu2, muf, logvar1, logvar2, logvarf, z1, z2, zf, z_scaled1, z_scaled2, z_scaledf, eps1, eps2, epsf, rk, snr1, snr2=self.ac(x,x1,x2,adj1,adj2)
        return self.ac(x, x1, x2, adj1, adj2)


# 使用RGAT的final Model
class final_model_RGAT(nn.Module):
    def __init__(self, dataset, graph, relation, feats_dim_list, num_nodes, ndim, input_feat_dim, hidden_dim,
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
        # sigvae进行链路预测+特征重构
        self.ac = SIG_VAE_AC_Independent_RGAT(dataset, graph, num_nodes, ndim, input_feat_dim, hidden_dim, hidden_dim1,
                                              hidden_dim2,
                                              num_heads,
                                              dropout,
                                              relation,
                                              encsto, gdc, ndist, copyK, copyJ, device)

    # x为两种类型节点的共同特征矩阵
    def forward(self, features_list, graph, relation):  # features_list为所有类型特征的list,graph为异构图，relation为选择预测的关系
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
            # x_all.append(F.elu(self.feat_drop(self.fc_list[i](features_list[i]))))
        for i in range(len(x_all)):
            x_all[i] = x_all[i].view([1, x_all[i].shape[0], x_all[i].shape[1]])
        if self.dataset == 'ACM':
            x = torch.cat((x_all[0], x_all[relation]), dim=1)  # 进行特征重构的x,torch.Size([1, 11186, 4000])
        # x:torch.Size([1, 11186, 4000])
        elif self.dataset == 'DBLP':
            x = torch.cat((x_all[1], x_all[relation]), dim=1)  # 进行特征重构的x
        elif self.dataset == 'IMDB':
            x = torch.cat((x_all[0], x_all[relation]), dim=1)
        elif self.dataset == 'YELP':
            x = torch.cat((x_all[0], x_all[relation]), dim=1)
        elif self.dataset == 'LastFM':
            x = torch.cat((x_all[1], x_all[relation]), dim=1)
        label_a = x.squeeze(dim=0)  # torch.Size([11186, 4000])
        x_all = torch.cat(x_all, dim=1)  # torch.Size([1, 11246, 4000]),type=torch.Tensor
        # graph = feature_in_graph_ACM(graph, x_all)  # 为graph节点添加特征
        ac = self.ac(x, graph, x_all)

        return label_a, ac


# 使用HAN的final Model
class final_model_HAN(nn.Module):
    def __init__(self, x_feats_dim, num_nodes, ndim, input_feat_dim, hidden_dim1, hidden_dim2, dropout, method='In',
                 encsto='semi', gdc='ip',
                 ndist='Bernoulli', copyK=1, copyJ=1, device='cuda'):
        super(final_model_HAN, self).__init__()
        # 添加线性层，对特征进行映射,x_feats_dim表示另一个one-hot节点的特征维度
        self.fc = nn.Linear(x_feats_dim, input_feat_dim, bias=True)
        nn.init.xavier_uniform_(self.fc.weight, gain=1.414)

        if dropout > 0:
            self.feat_drop = nn.Dropout(dropout)
        else:
            self.feat_drop = lambda x: x
        # sigvae进行链路预测+特征重构
        self.ac = SIG_VAE_AC_Independent_HAN(num_nodes, ndim, input_feat_dim, hidden_dim1, hidden_dim2, dropout,
                                             encsto,
                                             gdc, ndist, copyK, copyJ, device)

    # x为两种类型节点的共同特征矩阵
    def forward(self, x1, x2, g1, g2):  # x2为one-hot
        '''
        print("final_model:x2.shape")
        print(x1.shape)
        print(x2.shape)
        print(len(x2))#1
        '''
        # x2变成了自己构造的
        for i in range(len(x2)):
            x2 = torch.tanh(self.feat_drop(self.fc(x2)))
        x = torch.cat((x1, x2), dim=1)
        return self.ac(x, x1, x2, g1, g2)


# 不采用图网络，全部使用线性层
class SIG_VAE_AC_Independent_MLP(nn.Module):
    def __init__(self, num_nodes, ndim, input_feat_dim, hidden_dim1, hidden_dim2, dropout, encsto='semi', gdc='ip',
                 ndist='Bernoulli', copyK=1, copyJ=1, device='cuda'):
        super(SIG_VAE_AC_Independent_MLP, self).__init__()
        self.n_samples = num_nodes
        # node embedding
        self.gce = Dense(ndim, hidden_dim1, dropout, act=F.relu)  # 噪声层
        self.gc1 = Dense(input_feat_dim, hidden_dim1, dropout, act=F.relu)
        self.gc2 = Dense(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        self.gc3 = Dense(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)

        # feature embedding
        # 需要把dense改成3层的
        self.mlpe = Dense(input_dim=ndim, output_dim=hidden_dim1, dropout=dropout)  # 特征噪声层
        self.mlp1 = Dense(input_dim=self.n_samples, output_dim=hidden_dim1)
        self.mlp2 = Dense(input_dim=hidden_dim1, output_dim=hidden_dim2, dropout=dropout)  # 均值
        self.mlp3 = Dense(input_dim=hidden_dim1, output_dim=hidden_dim2, dropout=dropout)  # 方差
        self.encsto = encsto
        self.dc = GraphDecoder(hidden_dim2, dropout, gdc=gdc)
        self.device = device

        # 噪声分布
        if ndist == 'Bernoulli':  # 伯努利分布
            self.ndist = tdist.Bernoulli(torch.tensor([.5], device=self.device))
        elif ndist == 'Normal':  # 正态分布
            self.ndist == tdist.Normal(
                torch.tensor([0.], device=self.device),
                torch.tensor([1.], device=self.device))
        elif ndist == 'Exponential':  # 指数分布
            self.ndist = tdist.Exponential(torch.tensor([1.], device=self.device))

        self.K = copyK
        self.J = copyJ
        self.ndim = ndim
        # parameters in network gc1 and gce are NOT identically distributed, so we need to reweight the output
        # 网络 gc1和gce中的参数不是同分布的，所以我们需要重新加权输出。
        # of gce() so that the effect of hiddenx + hiddene is equivalent to gc(x || e).   gce() 使得 hiddenx + hiddene 的效果等同于 gc(x || e)。
        self.reweight = ((self.ndim + hidden_dim1) / (input_feat_dim + hidden_dim1)) ** (.5)

    def node_encode(self, x1, x2):
        # 对node进行encode
        assert len(x1.shape) == 3, 'The input tensor dimension is not 3!'
        # Without torch.Size(), an error would occur while resampling.
        hiddenx1 = self.gc1(x1)
        hiddenx2 = self.gc1(x2)

        if self.ndim >= 1:
            e1 = self.ndist.sample(torch.Size([self.K + self.J, x1.shape[1], self.ndim]))
            e1 = torch.squeeze(e1, -1)
            e1 = e1.mul(self.reweight)
            hiddene1 = self.gce(e1)

            e2 = self.ndist.sample(torch.Size([self.K + self.J, x2.shape[1], self.ndim]))
            e2 = torch.squeeze(e2, -1)
            e2 = e2.mul(self.reweight)
            hiddene2 = self.gce(e2)
        else:
            print("no randomness.")
            hiddene1 = torch.zeros(self.K + self.J, hiddenx1.shape[1], hiddenx1.shape[2], device=self.device)
            hiddene2 = torch.zeros(self.K + self.J, hiddenx2.shape[1], hiddenx2.shape[2], device=self.device)

        hidden1 = hiddenx1 + hiddene1
        hidden2 = hiddenx2 + hiddene2

        # hiddens = self.gc0(x, adj)

        p_signal1 = hiddenx1.pow(2.).mean()
        p_noise1 = hiddene1.pow(2.).mean([-2, -1])
        snr1 = (p_signal1 / p_noise1)  # 信噪比

        p_signal2 = hiddenx2.pow(2.).mean()
        p_noise2 = hiddene2.pow(2.).mean([-2, -1])
        snr2 = (p_signal2 / p_noise2)  # 信噪比
        # below are 3 options for producing logvar
        # 1. stochastic logvar (more instinctive)
        #    where logvar = self.gc3(hidden1, adj)
        #    set args.encsto to 'full'.
        # 2. deterministic logvar, shared by all K+J samples, and share a previous hidden layer with mu
        #    where logvar = self.gc3(hiddenx, adj)
        #    set args.encsto to 'semi'.
        # 3. deterministic logvar, shared by all K+J samples, and produced by another branch of network
        #    (the one applied by A. Hasanzadeh et al.)

        mu1 = self.gc2(hidden1)
        mu2 = self.gc2(hidden2)

        EncSto = (self.encsto == 'full')  # encsto="semi"

        hidden_sd1 = EncSto * hidden1 + (1 - EncSto) * hiddenx1
        hidden_sd2 = EncSto * hidden2 + (1 - EncSto) * hiddenx2

        logvar1 = self.gc3(hidden_sd1)
        logvar2 = self.gc3(hidden_sd2)

        return mu1, mu2, logvar1, logvar2, snr1, snr2

        # 利用dense对特征进行编码,为两种类型节点全部的特征

    def feature_encode(self, x):
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

    # 重参数化
    def reparameterize(self, mu, logvar):
        std = torch.exp(logvar / 2.)
        eps = torch.randn_like(std)
        return eps.mul(std).add(mu), eps
        # 返回z,eps
        # return mu, eps

    def forward(self, x, x1, x2, adj1, adj2):
        mu1, mu2, logvar1, logvar2, snr1, snr2 = self.node_encode(x1, x2)
        muf, logvarf = self.feature_encode(x)

        emb_mu1 = mu1[self.K:, :]
        emb_logvar1 = logvar1[self.K:, :]

        emb_mu2 = mu2[self.K:, :]
        emb_logvar2 = logvar2[self.K:, :]

        emb_muf = muf[self.K:, :]
        emb_logvarf = logvarf[self.K:, :]
        # check tensor size compatibility
        assert len(emb_mu1.shape) == len(emb_logvar1.shape), 'mu and logvar are not equi-dimension.'

        z1, eps1 = self.reparameterize(emb_mu1, emb_logvar1)  # 重参数化
        z2, eps2 = self.reparameterize(emb_mu2, emb_logvar2)
        zf, epsf = self.reparameterize(emb_muf, emb_logvarf)

        adj_, z_scaled1, z_scaled2, rk = self.dc(z1, z2)  # node embedding解码
        Za = torch.cat((z1, z2), dim=1)
        pred_a, z_scaleda, z_scaledf, _ = self.dc(Za, zf)
        # print("***SIG_VAE_AC_IN:pred_a**")
        # print(pred_a.shape)

        return adj_, pred_a, mu1, mu2, muf, logvar1, logvar2, logvarf, z1, z2, zf, Za, z_scaled1, z_scaled2, z_scaledf, z_scaleda, eps1, eps2, epsf, rk, snr1, snr2


# 采用HAN作为encoder,特征依然使用MLP
class SIG_VAE_AC_Independent_HAN(nn.Module):
    def __init__(self, num_nodes, ndim, input_feat_dim, hidden_dim1, hidden_dim2, dropout, encsto='semi', gdc='ip',
                 ndist='Bernoulli', copyK=1, copyJ=1, device='cuda'):
        super(SIG_VAE_AC_Independent_HAN, self).__init__()
        self.n_samples = num_nodes
        if ndim >= 1:
            noise = 1
        else:
            print("no noise....")
            noise = 0
        # node embedding
        # paper,输入为PP同构图
        '''
        #method3: 用相同的HAN网络训练两个不同类型的节点
        self.han_e = HANLayer(num_meta_paths=1, in_size=ndim, out_size=hidden_dim1, layer_num_heads=8, dropout=dropout,
                              noise=0)
        self.han_1 = HANLayer(num_meta_paths=1, in_size=input_feat_dim, out_size=hidden_dim1, layer_num_heads=8,
                              dropout=dropout, noise=0)
        self.han_2 = ProjectLayer(hidden_size=hidden_dim1, out_size=hidden_dim2, num_heads=[8])  # mu
        self.han_3 = ProjectLayer(hidden_size=hidden_dim1, out_size=hidden_dim2, num_heads=[8])  # std        
        '''

        '''
        #method1:使用四个HAN分别计算paper_mu,paper_std,x_mu,x_std
        self.han_pe = HANLayer(num_meta_paths=2, in_size=ndim, out_size=hidden_dim1, layer_num_heads=8, dropout=dropout,
                               noise=0)  # 噪声矩阵只需要一个注意力层就行了#2
        self.han_xe = HANLayer(num_meta_paths=1, in_size=ndim, out_size=hidden_dim1, layer_num_heads=8, dropout=dropout,
                               noise=0)  # 噪声矩阵只需要一个注意力层就行了
        self.han_p1 = HAN(num_meta_paths=2, in_size=input_feat_dim, hidden_size=hidden_dim1, out_size=hidden_dim2,
                          num_heads=[8], dropout=dropout, noise=noise)
        self.han_p2 = HAN(num_meta_paths=2, in_size=input_feat_dim, hidden_size=hidden_dim1, out_size=hidden_dim2,
                          num_heads=[8], dropout=dropout, noise=noise)   
        self.han_x1 = HAN(num_meta_paths=1, in_size=input_feat_dim, hidden_size=hidden_dim1, out_size=hidden_dim2,
                          num_heads=[8], dropout=dropout, noise=noise)
        self.han_x2 = HAN(num_meta_paths=1, in_size=input_feat_dim, hidden_size=hidden_dim1, out_size=hidden_dim2,
                          num_heads=[8], dropout=dropout, noise=noise)              
        '''

        # method2:用两个不同的HAN训练paper和x节点
        self.han_pe = HANLayer(num_meta_paths=2, in_size=ndim, out_size=hidden_dim1, layer_num_heads=8, dropout=dropout,
                               noise=0)  # 噪声矩阵只需要一个注意力层就行了#2

        self.han_xe = HANLayer(num_meta_paths=1, in_size=ndim, out_size=hidden_dim1, layer_num_heads=8, dropout=dropout,
                               noise=0)  # 噪声矩阵只需要一个注意力层就行了
        self.han_p0 = HANLayer(num_meta_paths=2, in_size=input_feat_dim, out_size=hidden_dim1, layer_num_heads=8,
                               dropout=dropout,
                               noise=0)  # 2
        self.han_p1 = ProjectLayer(hidden_size=hidden_dim1, out_size=hidden_dim2, num_heads=[8])
        self.han_p2 = ProjectLayer(hidden_size=hidden_dim1, out_size=hidden_dim2, num_heads=[8])
        # other，输入为XX同构图

        self.han_x0 = HANLayer(num_meta_paths=1, in_size=input_feat_dim, out_size=hidden_dim1, layer_num_heads=8,
                               dropout=dropout,
                               noise=0)
        self.han_x1 = ProjectLayer(hidden_size=hidden_dim1, out_size=hidden_dim2, num_heads=[8])
        self.han_x2 = ProjectLayer(hidden_size=hidden_dim1, out_size=hidden_dim2, num_heads=[8])

        # feature embedding
        # 需要把dense改成3层的
        self.mlpe = Dense(input_dim=ndim, output_dim=hidden_dim1, dropout=dropout, act=torch.tanh)  # 特征噪声层
        self.mlp1 = Dense(input_dim=self.n_samples, output_dim=hidden_dim1, act=torch.tanh)
        self.mlp2 = Dense(input_dim=hidden_dim1, output_dim=hidden_dim2, dropout=dropout, act=lambda x: x)  # 均值
        self.mlp3 = Dense(input_dim=hidden_dim1, output_dim=hidden_dim2, dropout=dropout, act=lambda x: x)  # 方差
        self.encsto = encsto
        self.dc = GraphDecoder(hidden_dim2, dropout, gdc=gdc)
        self.device = device

        # 噪声分布
        if ndist == 'Bernoulli':  # 伯努利分布
            self.ndist = tdist.Bernoulli(torch.tensor([.5], device=self.device))
        elif ndist == 'Normal':  # 正态分布
            self.ndist == tdist.Normal(
                torch.tensor([0.], device=self.device),
                torch.tensor([1.], device=self.device))
        elif ndist == 'Exponential':  # 指数分布
            self.ndist = tdist.Exponential(torch.tensor([1.], device=self.device))

        self.K = copyK
        self.J = copyJ
        self.ndim = ndim
        # parameters in network gc1 and gce are NOT identically distributed, so we need to reweight the output
        # 网络 gc1和gce中的参数不是同分布的，所以我们需要重新加权输出。
        # of gce() so that the effect of hiddenx + hiddene is equivalent to gc(x || e).   gce() 使得 hiddenx + hiddene 的效果等同于 gc(x || e)。
        self.reweight = ((self.ndim + hidden_dim1) / (input_feat_dim + hidden_dim1)) ** (.5)

    '''
    #method3
    def node_encode(self, x1, x2, g1, g2):
        # 对node进行encode
        assert len(x1.shape) == 3, 'The input tensor dimension is not 3!'
        # Without torch.Size(), an error would occur while resampling.
        hiddenx1 = self.han_1(g1, x1)
        hiddenx2 = self.han_1(g2, x2)

        if self.ndim >= 1:
            e1 = self.ndist.sample(torch.Size([self.K + self.J, x1.shape[1], self.ndim]))
            e1 = torch.squeeze(e1, -1)
            e1 = e1.mul(self.reweight)
            hiddene1 = self.han_e(g1, e1)

            e2 = self.ndist.sample(torch.Size([self.K + self.J, x2.shape[1], self.ndim]))
            e2 = torch.squeeze(e2, -1)
            e2 = e2.mul(self.reweight)
            hiddene2 = self.han_e(g2, e2)
        else:
            print("no randomness.")
            hiddene1 = torch.zeros(self.K + self.J, hiddenx1.shape[1], hiddenx1.shape[2], device=self.device)
            hiddene2 = torch.zeros(self.K + self.J, hiddenx2.shape[1], hiddenx2.shape[2], device=self.device)

        hidden1 = torch.Tensor([item.cpu().detach().numpy() for item in hiddenx1]) + torch.Tensor(
            [item.cpu().detach().numpy() for item in hiddene1])
        hidden2 = torch.Tensor([item.cpu().detach().numpy() for item in hiddenx2]) + torch.Tensor(
            [item.cpu().detach().numpy() for item in hiddene2])
        # print(hidden1.shape, hiddenx1.shape, hiddene1.shape)
        # result = np.sum([a, b], axis=0).tolist()
        # print(len(hidden1),len(hiddenx1),len(hiddene1))
        hidden1 = hidden1.to(self.device)
        hidden2 = hidden2.to(self.device)
        mu1 = self.han_2(hidden1)

        mu2 = self.han_2(hidden2)

        EncSto = (self.encsto == 'full')  # encsto="semi"

        hidden_sd1 = EncSto * hidden1  # + (1 - EncSto) * torch.Tensor([item.cpu().detach().numpy() for item in hiddenx1])
        hidden_sd2 = EncSto * hidden2  # + (1 - EncSto) * torch.Tensor([item.cpu().detach().numpy() for item in hiddenx2])

        logvar1 = self.han_3(hidden_sd1)
        logvar2 = self.han_3(hidden_sd2)

        return mu1, mu2, logvar1, logvar2    
    '''

    # method2
    def node_encode(self, x1, x2, g1, g2):
        # 对node进行encode
        assert len(x1.shape) == 3, 'The input tensor dimension is not 3!'
        # Without torch.Size(), an error would occur while resampling.
        hiddenx1 = self.han_p0(g1, x1)
        hiddenx2 = self.han_x0(g2, x2)

        if self.ndim >= 1:
            e1 = self.ndist.sample(torch.Size([self.K + self.J, x1.shape[1], self.ndim]))
            e1 = torch.squeeze(e1, -1)
            e1 = e1.mul(self.reweight)
            hiddene1 = self.han_pe(g1, e1)

            e2 = self.ndist.sample(torch.Size([self.K + self.J, x2.shape[1], self.ndim]))
            e2 = torch.squeeze(e2, -1)
            e2 = e2.mul(self.reweight)
            hiddene2 = self.han_xe(g2, e2)
        else:
            print("no randomness.")
            hiddene1 = torch.zeros(self.K + self.J, hiddenx1.shape[1], hiddenx1.shape[2], device=self.device)
            hiddene2 = torch.zeros(self.K + self.J, hiddenx2.shape[1], hiddenx2.shape[2], device=self.device)

        hidden1 = torch.Tensor([item.cpu().detach().numpy() for item in hiddenx1]) + torch.Tensor(
            [item.cpu().detach().numpy() for item in hiddene1])
        hidden2 = torch.Tensor([item.cpu().detach().numpy() for item in hiddenx2]) + torch.Tensor(
            [item.cpu().detach().numpy() for item in hiddene2])
        # print(hidden1.shape, hiddenx1.shape, hiddene1.shape)
        # result = np.sum([a, b], axis=0).tolist()
        # print(len(hidden1),len(hiddenx1),len(hiddene1))
        hidden1 = hidden1.to(self.device)
        hidden2 = hidden2.to(self.device)
        mu1 = self.han_p1(hidden1)

        mu2 = self.han_x1(hidden2)

        EncSto = (self.encsto == 'full')  # encsto="semi"

        hidden_sd1 = EncSto * hidden1  # + (1 - EncSto) * torch.Tensor([item.cpu().detach().numpy() for item in hiddenx1])
        hidden_sd2 = EncSto * hidden2  # + (1 - EncSto) * torch.Tensor([item.cpu().detach().numpy() for item in hiddenx2])

        logvar1 = self.han_p2(hidden_sd1)
        logvar2 = self.han_x2(hidden_sd2)

        return mu1, mu2, logvar1, logvar2

    '''
    #method 1
    def node_encode(self, x1, x2, g1, g2):
        # 对node进行encode
        assert len(x1.shape) == 3, 'The input tensor dimension is not 3!'
        # Without torch.Size(), an error would occur while resampling.
        if self.ndim >= 1:
            e1 = self.ndist.sample(torch.Size([self.K + self.J, x1.shape[1], self.ndim]))
            e1 = torch.squeeze(e1, -1)
            e1 = e1.mul(self.reweight)
            hiddene1 = self.han_pe(g1, e1)
            e2 = self.ndist.sample(torch.Size([self.K + self.J, x2.shape[1], self.ndim]))
            e2 = torch.squeeze(e2, -1)
            e2 = e2.mul(self.reweight)
            hiddene2 = self.han_xe(g2, e2)
            mu1 = self.han_p1(g1, x1, hiddene1)
            mu2 = self.han_x1(g2, x2, hiddene2)
            logvar1 = self.han_p2(g1, x1, hiddene1)
            logvar2 = self.han_x2(g2, x2, hiddene2)
        else:
            print("no randomness.")
            mu1 = self.han_p1(g1, x1, e=[])
            mu2 = self.han_x1(g2, x2, e=[])
            logvar1 = self.han_p2(g1, x1, e=[])
            logvar2 = self.han_x2(g2, x2, e=[])

        return mu1, mu2, logvar1, logvar2

    
    
    '''

    # 利用dense对特征进行编码,为两种类型节点全部的特征

    def feature_encode(self, x):
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

    # 重参数化
    def reparameterize(self, mu, logvar):
        std = torch.exp(logvar / 2.)
        eps = torch.randn_like(std)
        return eps.mul(std).add(mu), eps
        # 返回z,eps
        # return mu, eps

    def forward(self, x, x1, x2, g1, g2):
        mu1, mu2, logvar1, logvar2 = self.node_encode(x1, x2, g1, g2)
        muf, logvarf = self.feature_encode(x)
        emb_mu1 = mu1[self.K:, :]
        emb_logvar1 = logvar1[self.K:, :]

        emb_mu2 = mu2[self.K:, :]
        emb_logvar2 = logvar2[self.K:, :]

        emb_muf = muf[self.K:, :]
        emb_logvarf = logvarf[self.K:, :]
        # check tensor size compatibility
        assert len(emb_mu1.shape) == len(emb_logvar1.shape), 'mu and logvar are not equi-dimension.'

        z1, eps1 = self.reparameterize(emb_mu1, emb_logvar1)  # 重参数化
        z2, eps2 = self.reparameterize(emb_mu2, emb_logvar2)
        zf, epsf = self.reparameterize(emb_muf, emb_logvarf)

        adj_, z_scaled1, z_scaled2, rk = self.dc(z1, z2)  # node embedding解码
        Za = torch.cat((z1, z2), dim=1)
        pred_a, z_scaleda, z_scaledf, _ = self.dc(Za, zf)
        # print("***SIG_VAE_AC_IN:pred_a**")
        # print(pred_a.shape)

        return adj_, pred_a, mu1, mu2, muf, logvar1, logvar2, logvarf, z1, z2, zf, Za, z_scaled1, z_scaled2, z_scaledf, z_scaleda, eps1, eps2, epsf, rk


# 采用RGAT作为encoder,特征使用MLP
class SIG_VAE_AC_Independent_RGAT(nn.Module):
    def __init__(self, dataset, graph, num_nodes, ndim, input_feat_dim, hidden_dim, hidden_dim1, hidden_dim2, num_heads, dropout,
                 relation=1,
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
        self.gat_e = RGATLayer(in_feats=ndim, out_feats=hidden_dim1, num_heads=num_heads, rel_names=self.rel_names,
                               act=F.relu)  # 噪声层
        self.gat_1 = RGATLayer(in_feats=hidden_dim, out_feats=hidden_dim1, num_heads=num_heads, rel_names=self.rel_names,
                               act=F.relu)  # 共享层
        self.gat_2 = RGATLayer(in_feats=hidden_dim1, out_feats=hidden_dim2, num_heads=int(num_heads/2), rel_names=self.rel_names,
                               act=lambda x: x)  # 均值
        self.gat_3 = RGATLayer(in_feats=hidden_dim1, out_feats=hidden_dim2, num_heads=int(num_heads/2), rel_names=self.rel_names,
                               act=lambda x: x)  # 方差
        # feature embedding
        self.mlpe = Dense(input_dim=ndim, output_dim=hidden_dim1, dropout=dropout, act=torch.tanh)  # 特征噪声层
        self.mlp1 = Dense(input_dim=self.n_samples, output_dim=hidden_dim1, act=torch.tanh)
        self.mlp2 = Dense(input_dim=hidden_dim1, output_dim=hidden_dim2, dropout=dropout, act=lambda x: x)  # 均值
        self.mlp3 = Dense(input_dim=hidden_dim1, output_dim=hidden_dim2, dropout=dropout, act=lambda x: x)  # 方差
        self.mlp_recover = Dense(input_dim=hidden_dim, output_dim=input_feat_dim, dropout=dropout,
                                 act=torch.sigmoid)  # 复原paper节点的线性层
        self.encsto = encsto
        self.dc = GraphDecoder(hidden_dim2, dropout, gdc=gdc)
        self.dc2 = GraphDecoder2(hidden_dim2, dropout, gdc=gdc)
        self.device = device

        # 噪声分布
        if ndist == 'Bernoulli':
            self.ndist = tdist.Bernoulli(torch.tensor([.5], device=self.device))
        elif ndist == 'Normal':
            self.ndist == tdist.Normal(
                torch.tensor([0.], device=self.device),
                torch.tensor([1.], device=self.device)
            )
        elif ndist == 'Exponential':
            self.ndist = tdist.Exponential(torch.tensor([1.], device=self.device))

        self.K = copyK
        self.J = copyJ
        self.ndim = ndim
        self.relation = relation
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
            mu1, mu2 = mu[0], mu[self.relation]
            logvar1, logvar2 = logvar[0], logvar[self.relation]
        elif self.dataset == 'DBLP':
            mu = torch.split(mu, [graph.number_of_nodes('author'), graph.number_of_nodes('paper'),
                                  graph.number_of_nodes('term'), graph.number_of_nodes('venue')], dim=1)
            logvar = torch.split(logvar, [graph.number_of_nodes('author'), graph.number_of_nodes('paper'),
                                          graph.number_of_nodes('term'), graph.number_of_nodes('venue')], dim=1)
            mu1, mu2 = mu[1], mu[self.relation]
            logvar1, logvar2 = logvar[1], logvar[self.relation]
        elif self.dataset == 'IMDB':
            mu = torch.split(mu, [graph.number_of_nodes('movie'), graph.number_of_nodes('director'),
                                  graph.number_of_nodes('actor')], dim=1)
            logvar = torch.split(logvar, [graph.number_of_nodes('movie'), graph.number_of_nodes('director'),
                                          graph.number_of_nodes('actor')], dim=1)
            mu1, mu2 = mu[0], mu[self.relation]
            logvar1, logvar2 = logvar[0], logvar[self.relation]
        elif self.dataset == 'YELP':
            mu = torch.split(mu, [graph.number_of_nodes('business'), graph.number_of_nodes('user'),
                                  graph.number_of_nodes('service'), graph.number_of_nodes('level')], dim=1)
            logvar = torch.split(logvar, [graph.number_of_nodes('business'), graph.number_of_nodes('user'),
                                          graph.number_of_nodes('service'), graph.number_of_nodes('level')], dim=1)
            mu1, mu2 = mu[0], mu[self.relation]
            logvar1, logvar2 = logvar[0], logvar[self.relation]
        elif self.dataset == 'LastFM':
            mu = torch.split(mu, [graph.number_of_nodes('user'), graph.number_of_nodes('artist'),
                                  graph.number_of_nodes('tag')], dim=1)
            logvar = torch.split(logvar, [graph.number_of_nodes('user'), graph.number_of_nodes('artist'),
                                          graph.number_of_nodes('tag')], dim=1)
            mu1, mu2 = mu[1], mu[self.relation]
            logvar1, logvar2 = logvar[1], logvar[self.relation]
        return mu1, mu2, logvar1, logvar2

    def feature_encode(self, x):
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

    def forward(self, x, graph, x_all):  # x为需要重构的特征，graph为异构图(有特征）
        mu1, mu2, logvar1, logvar2 = self.node_encode(graph, x_all)

        muf, logvarf = self.feature_encode(x)
        emb_mu1 = mu1[self.K:, :]  # J个似然样本
        emb_logvar1 = logvar1[self.K:, :]
        emb_mu2 = mu2[self.K:, :]
        emb_logvar2 = logvar2[self.K:, :]
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

        assert len(emb_mu1.shape) == len(emb_logvar1.shape), 'mu and logvar are not equi-dimension.'

        z1, eps1 = self.reparameterize(emb_mu1, emb_logvar1)  # 重参数化paper J个
        z2, eps2 = self.reparameterize(emb_mu2, emb_logvar2)  # x
        zf, epsf = self.reparameterize(emb_muf, emb_logvarf)
        adj_, z_scaled1, z_scaled2, rk = self.dc(z1, z2)  # node embedding解码
        Za = torch.cat((z1, z2), dim=1)
        pred_a, z_scaleda, z_scaledf, _ = self.dc(Za, zf)  # feature embedding 解码
        fea_recover = pred_a[:, :mu1.shape[1], :]
        fea_recover = self.mlp_recover(fea_recover)  # 用loss_function_a_mse计算pred_recover和fea_orig的值
        return adj_, pred_a, fea_recover, mu1, mu2, muf, logvar1, logvar2, logvarf, z1, z2, zf, Za, z_scaled1, z_scaled2, z_scaledf, z_scaleda, eps1, eps2, epsf, rk


# 假设特征和节点独立，GNN作为encoder,特征使用MLP
class SIG_VAE_AC_Independent(nn.Module):
    def __init__(self, num_nodes, ndim, input_feat_dim, hidden_dim1, hidden_dim2, dropout, encsto='semi', gdc='ip',
                 ndist='Bernoulli', copyK=1, copyJ=1, device='cuda'):
        super(SIG_VAE_AC_Independent, self).__init__()
        self.n_samples = num_nodes
        # node embedding
        # 对不同类型的节点使用相同的网络参数，生成noise embedding,mu,std
        self.gce = GraphConvolution(ndim, hidden_dim1, dropout, act=F.relu)  # 噪声层
        self.gc1 = GraphConvolution(input_feat_dim, hidden_dim1, dropout, act=F.relu)
        self.gc2 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        self.gc3 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)

        # feature embedding
        # 需要把dense改成3层的
        self.mlpe = Dense(input_dim=ndim, output_dim=hidden_dim1, dropout=dropout, act=torch.tanh)  # 特征噪声层
        self.mlp1 = Dense(input_dim=self.n_samples, output_dim=hidden_dim1, act=torch.tanh)
        self.mlp2 = Dense(input_dim=hidden_dim1, output_dim=hidden_dim2, dropout=dropout, act=lambda x: x)  # 均值
        self.mlp3 = Dense(input_dim=hidden_dim1, output_dim=hidden_dim2, dropout=dropout, act=lambda x: x)  # 方差
        self.encsto = encsto
        self.dc = GraphDecoder(hidden_dim2, dropout, gdc=gdc)
        self.dc2 = GraphDecoder2(hidden_dim2, dropout, gdc=gdc)
        self.device = device

        # 噪声分布
        if ndist == 'Bernoulli':  # 伯努利分布
            self.ndist = tdist.Bernoulli(torch.tensor([.5], device=self.device))
        elif ndist == 'Normal':  # 正态分布
            self.ndist == tdist.Normal(
                torch.tensor([0.], device=self.device),
                torch.tensor([1.], device=self.device))
        elif ndist == 'Exponential':  # 指数分布
            self.ndist = tdist.Exponential(torch.tensor([1.], device=self.device))

        self.K = copyK
        self.J = copyJ
        self.ndim = ndim
        # parameters in network gc1 and gce are NOT identically distributed, so we need to reweight the output
        # 网络 gc1和gce中的参数不是同分布的，所以我们需要重新加权输出。
        # of gce() so that the effect of hiddenx + hiddene is equivalent to gc(x || e).   gce() 使得 hiddenx + hiddene 的效果等同于 gc(x || e)。
        self.reweight = ((self.ndim + hidden_dim1) / (input_feat_dim + hidden_dim1)) ** (.5)

    def node_encode(self, x1, x2, adj1, adj2):
        # 对node进行encode
        assert len(x1.shape) == 3, 'The input tensor dimension is not 3!'
        # Without torch.Size(), an error would occur while resampling.
        hiddenx1 = self.gc1(x1, adj1)  # 两个不同类型的节点用一个GCN
        hiddenx2 = self.gc1(x2, adj2)

        if self.ndim >= 1:
            e1 = self.ndist.sample(torch.Size([self.K + self.J, x1.shape[1], self.ndim]))
            e1 = torch.squeeze(e1, -1)
            e1 = e1.mul(self.reweight)
            hiddene1 = self.gce(e1, adj1)

            e2 = self.ndist.sample(torch.Size([self.K + self.J, x2.shape[1], self.ndim]))
            e2 = torch.squeeze(e2, -1)
            e2 = e2.mul(self.reweight)
            hiddene2 = self.gce(e2, adj2)
        else:
            print("no randomness.")
            hiddene1 = torch.zeros(self.K + self.J, hiddenx1.shape[1], hiddenx1.shape[2], device=self.device)
            hiddene2 = torch.zeros(self.K + self.J, hiddenx2.shape[1], hiddenx2.shape[2], device=self.device)

        hidden1 = hiddenx1 + hiddene1
        hidden2 = hiddenx2 + hiddene2

        # hiddens = self.gc0(x, adj)

        p_signal1 = hiddenx1.pow(2.).mean()
        p_noise1 = hiddene1.pow(2.).mean([-2, -1])
        snr1 = (p_signal1 / p_noise1)  # 信噪比

        p_signal2 = hiddenx2.pow(2.).mean()
        p_noise2 = hiddene2.pow(2.).mean([-2, -1])
        snr2 = (p_signal2 / p_noise2)  # 信噪比
        # below are 3 options for producing logvar
        # 1. stochastic logvar (more instinctive)
        #    where logvar = self.gc3(hidden1, adj)
        #    set args.encsto to 'full'.
        # 2. deterministic logvar, shared by all K+J samples, and share a previous hidden layer with mu
        #    where logvar = self.gc3(hiddenx, adj)
        #    set args.encsto to 'semi'.
        # 3. deterministic logvar, shared by all K+J samples, and produced by another branch of network
        #    (the one applied by A. Hasanzadeh et al.)

        mu1 = self.gc2(hidden1, adj1)
        mu2 = self.gc2(hidden2, adj2)

        EncSto = (self.encsto == 'full')  # encsto="semi"

        hidden_sd1 = EncSto * hidden1 + (1 - EncSto) * hiddenx1
        hidden_sd2 = EncSto * hidden2 + (1 - EncSto) * hiddenx2

        logvar1 = self.gc3(hidden_sd1, adj1)
        logvar2 = self.gc3(hidden_sd2, adj2)

        return mu1, mu2, logvar1, logvar2, snr1, snr2

        # 利用dense对特征进行编码,为两种类型节点全部的特征

    def feature_encode(self, x):
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

    # 重参数化
    def reparameterize(self, mu, logvar):
        std = torch.exp(logvar / 2.)
        eps = torch.randn_like(std)
        return eps.mul(std).add(mu), eps
        # 返回z,eps
        # return mu, eps

    def forward(self, x, x1, x2, adj1, adj2):
        mu1, mu2, logvar1, logvar2, snr1, snr2 = self.node_encode(x1, x2, adj1, adj2)
        muf, logvarf = self.feature_encode(x)

        emb_mu1 = mu1[self.K:, :]
        emb_logvar1 = logvar1[self.K:, :]

        emb_mu2 = mu2[self.K:, :]
        emb_logvar2 = logvar2[self.K:, :]

        emb_muf = muf[self.K:, :]
        emb_logvarf = logvarf[self.K:, :]
        # check tensor size compatibility
        assert len(emb_mu1.shape) == len(emb_logvar1.shape), 'mu and logvar are not equi-dimension.'

        z1, eps1 = self.reparameterize(emb_mu1, emb_logvar1)  # 重参数化
        z2, eps2 = self.reparameterize(emb_mu2, emb_logvar2)
        zf, epsf = self.reparameterize(emb_muf, emb_logvarf)

        adj_, z_scaled1, z_scaled2, rk = self.dc(z1, z2)  # node embedding解码
        Za = torch.cat((z1, z2), dim=1)
        pred_a, z_scaleda, z_scaledf, _ = self.dc2(Za, zf)
        # print("***SIG_VAE_AC_IN:pred_a**")
        # print(pred_a.shape)

        return adj_, pred_a, mu1, mu2, muf, logvar1, logvar2, logvarf, z1, z2, zf, Za, z_scaled1, z_scaled2, z_scaledf, z_scaleda, eps1, eps2, epsf, rk, snr1, snr2


class GCNModelSIGVAE(nn.Module):
    def __init__(self, ndim, input_feat_dim, hidden_dim1, hidden_dim2, dropout, encsto='semi', gdc='ip',
                 ndist='Bernoulli', copyK=1, copyJ=1, device='cuda'):
        super(GCNModelSIGVAE, self).__init__()

        self.gce = GraphConvolution(ndim, hidden_dim1, dropout, act=F.relu)  # 噪声层
        # self.gc0 = GraphConvolution(input_feat_dim, hidden_dim1, dropout, act=F.relu)
        self.gc1 = GraphConvolution(input_feat_dim, hidden_dim1, dropout, act=F.relu)
        self.gc2 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        self.gc3 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        self.encsto = encsto
        self.dc = GraphDecoder(hidden_dim2, dropout, gdc=gdc)
        self.device = device

        if ndist == 'Bernoulli':  # 伯努利分布
            self.ndist = tdist.Bernoulli(torch.tensor([.5], device=self.device))
        elif ndist == 'Normal':  # 正态分布
            self.ndist == tdist.Normal(
                torch.tensor([0.], device=self.device),
                torch.tensor([1.], device=self.device))
        elif ndist == 'Exponential':  # 指数分布
            self.ndist = tdist.Exponential(torch.tensor([1.], device=self.device))

        # K and J are defined in http://proceedings.mlr.press/v80/yin18b/yin18b-supp.pdf
        # Algorthm 1.
        self.K = copyK
        self.J = copyJ
        self.ndim = ndim

        # parameters in network gc1 and gce are NOT identically distributed, so we need to reweight the output
        # 网络 gc1和gce中的参数不是同分布的，所以我们需要重新加权输出。
        # of gce() so that the effect of hiddenx + hiddene is equivalent to gc(x || e).   gce() 使得 hiddenx + hiddene 的效果等同于 gc(x || e)。
        self.reweight = ((self.ndim + hidden_dim1) / (input_feat_dim + hidden_dim1)) ** (.5)

    # 改进部分
    def encode(self, x1, x2, adj1, adj2):
        assert len(x1.shape) == 3, 'The input tensor dimension is not 3!'
        # Without torch.Size(), an error would occur while resampling.
        hiddenx1 = self.gc1(x1, adj1)
        hiddenx2 = self.gc1(x2, adj2)
        if self.ndim >= 1:
            e1 = self.ndist.sample(torch.Size([self.K + self.J, x1.shape[1], self.ndim]))
            e1 = torch.squeeze(e1, -1)
            e1 = e1.mul(self.reweight)
            hiddene1 = self.gce(e1, adj1)

            e2 = self.ndist.sample(torch.Size([self.K + self.J, x2.shape[1], self.ndim]))
            e2 = torch.squeeze(e2, -1)
            e2 = e2.mul(self.reweight)
            hiddene2 = self.gce(e2, adj2)
        else:
            print("no randomness.")
            hiddene1 = torch.zeros(self.K + self.J, hiddenx1.shape[1], hiddenx1.shape[2], device=self.device)
            hiddene2 = torch.zeros(self.K + self.J, hiddenx2.shape[1], hiddenx2.shape[2], device=self.device)

        hidden1 = hiddenx1 + hiddene1
        hidden2 = hiddenx2 + hiddene2

        # hiddens = self.gc0(x, adj)

        p_signal1 = hiddenx1.pow(2.).mean()
        p_noise1 = hiddene1.pow(2.).mean([-2, -1])
        snr1 = (p_signal1 / p_noise1)  # 信噪比

        p_signal2 = hiddenx2.pow(2.).mean()
        p_noise2 = hiddene2.pow(2.).mean([-2, -1])
        snr2 = (p_signal2 / p_noise2)  # 信噪比
        # below are 3 options for producing logvar
        # 1. stochastic logvar (more instinctive)
        #    where logvar = self.gc3(hidden1, adj)
        #    set args.encsto to 'full'.
        # 2. deterministic logvar, shared by all K+J samples, and share a previous hidden layer with mu
        #    where logvar = self.gc3(hiddenx, adj)
        #    set args.encsto to 'semi'.
        # 3. deterministic logvar, shared by all K+J samples, and produced by another branch of network
        #    (the one applied by A. Hasanzadeh et al.)

        mu1 = self.gc2(hidden1, adj1)
        mu2 = self.gc2(hidden2, adj2)

        EncSto = (self.encsto == 'full')  # encsto="semi"

        hidden_sd1 = EncSto * hidden1 + (1 - EncSto) * hiddenx1
        hidden_sd2 = EncSto * hidden2 + (1 - EncSto) * hiddenx2

        logvar1 = self.gc3(hidden_sd1, adj1)
        logvar2 = self.gc3(hidden_sd2, adj2)

        return mu1, mu2, logvar1, logvar2, snr1, snr2

    # 重参数化
    def reparameterize(self, mu, logvar):
        std = torch.exp(logvar / 2.)
        eps = torch.randn_like(std)
        return eps.mul(std).add(mu), eps
        # 返回z,eps
        # return mu, eps

    def forward(self, x1, x2, adj1, adj2):
        mu1, mu2, logvar1, logvar2, snr1, snr2 = self.encode(x1, x2, adj1, adj2)

        emb_mu1 = mu1[self.K:, :]
        emb_logvar1 = logvar1[self.K:, :]

        emb_mu2 = mu2[self.K:, :]
        emb_logvar2 = logvar2[self.K:, :]
        # check tensor size compatibility
        assert len(emb_mu1.shape) == len(emb_logvar1.shape), 'mu and logvar are not equi-dimension.'

        z1, eps1 = self.reparameterize(emb_mu1, emb_logvar1)  # 重参数化
        z2, eps2 = self.reparameterize(emb_mu2, emb_logvar2)

        adj_, z_scaled1, z_scaled2, rk = self.dc(z1, z2)  # 解码

        return adj_, mu1, mu2, logvar1, logvar2, z1, z2, z_scaled1, z_scaled2, eps1, eps2, rk, snr1, snr2


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
            adj = adj_lgt
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


#################################################################################

class SIG_VAE(nn.Module):
    def __init__(self, adj1, adj2):
        super(SIG_VAE, self).__init__()

        self.gce = GraphConvolution(args.edim, args.hidden1_dim, adj1, adj2, act=F.relu)  # 噪声层
        # self.gc0 = GraphConvolution(input_feat_dim, hidden_dim1, dropout, act=F.relu)
        self.gc1 = GraphConvSparse(args.input_dim, args.hidden1_dim, adj1, adj2)
        self.gc2 = GraphConvSparse(args.hidden1_dim, args.hidden2_dim, adj1, adj2, activation=lambda x: x)
        self.gc3 = GraphConvSparse(args.hidden1_dim, args.hidden2_dim, adj1, adj2, activation=lambda x: x)
        self.encsto = args.encsto
        self.dc = args.gdc

        # 噪声服从的分布
        if args.noise_dist == 'Bernoulli':  # 伯努利分布
            self.ndist = tdist.Bernoulli(torch.tensor([.5]))
        elif args.noise_dist == 'Normal':  # 正态分布
            self.ndist == tdist.Normal(
                torch.tensor([0.], device=self.device),
                torch.tensor([1.], device=self.device))
        elif args.noise_dist == 'Exponential':  # 指数分布
            self.ndist = tdist.Exponential(torch.tensor([1.]))

        # K and J are defined in http://proceedings.mlr.press/v80/yin18b/yin18b-supp.pdf
        # Algorthm 1.
        self.K = args.K
        self.J = args.J
        self.ndim = args.edim

        # parameters in network gc1 and gce are NOT identically distributed, so we need to reweight the output
        # 网络 gc1和gce中的参数不是同分布的，所以我们需要重新加权输出。
        # of gce() so that the effect of hiddenx + hiddene is equivalent to gc(x || e).   gce() 使得 hiddenx + hiddene 的效果等同于 gc(x || e)。
        self.reweight = ((self.ndim + args.hidden1_dim) / (args.input_dim + args.hidden1_dim)) ** (.5)

    # 重参数化
    def reparameterize(self, mu, logvar):
        std = torch.exp(logvar / 2.)
        eps = torch.randn_like(std)
        return eps.mul(std).add(mu), eps
        # 返回z,eps
        # return mu, eps

    def encode(self, X1, X2):  # 输出两个隐变量z
        hiddenx1, hiddenx2 = self.gc1(X1, X2)  # 生成特征矩阵的隐藏层
        print("++++++++++++hiddenx1++++++++++++")
        print(hiddenx1.shape)
        if len(hiddenx1.shape) == 2:
            features1 = hiddenx1.view([1, hiddenx1.shape[0], hiddenx1.shape[1]])
        if len(hiddenx2.shape) == 2:
            features2 = hiddenx2.view([1, hiddenx2.shape[0], hiddenx2.shape[1]])

        if self.ndim >= 1:
            # 分别为两种类型，采样两个噪声
            e1 = self.ndist.sample(torch.Size([self.K + self.J, X1.shape[0], self.ndim]))
            e1 = torch.squeeze(e1, -1)
            e1 = e1.mul(self.reweight)
            print("+++++++++e1++++++++++")
            print(e1.shape)
            e2 = self.ndist.sample(torch.Size([self.K + self.J, X2.shape[0], self.ndim]))
            e2 = torch.squeeze(e2, -1)
            e2 = e2.mul(self.reweight)
            hiddene1, hiddene2 = self.gce(e1, e2)  # 传递两个特征矩阵，生成两个隐变量
            print("++++++++噪声矩阵+++++++++")
            print(hiddene1.shape)
        else:
            print("no randomness.")
            hiddene1 = torch.zeros(self.K + self.J, hiddenx1.shape[1], hiddenx1.shape[2], device=self.device)
            hiddene2 = torch.zeros(self.K + self.J, hiddenx2.shape[1], hiddenx2.shape[2], device=self.device)

        hidden1 = hiddenx1 + hiddene1  # 添加噪声之后的隐变量
        hidden2 = hiddenx2 + hiddene2
        # 信噪比
        p_signal1 = hiddenx1.pow(2.).mean()
        p_noise1 = hiddene1.pow(2.).mean([-2, -1])
        snr1 = (p_signal1 / p_noise1)

        p_signal2 = hiddenx2.pow(2.).mean()
        p_noise2 = hiddene2.pow(2.).mean([-2, -1])
        snr2 = (p_signal2 / p_noise2)

        # below are 3 options for producing logvar三种方式计算方差
        # 1. stochastic logvar (more instinctive)
        #    where logvar = self.gc3(hidden1, adj)
        #    set args.encsto to 'full'.
        # 2. deterministic logvar, shared by all K+J samples, and share a previous hidden layer with mu
        #    where logvar = self.gc3(hiddenx, adj)
        #    set args.encsto to 'semi'.
        # 3. deterministic logvar, shared by all K+J samples, and produced by another branch of network
        #    (the one applied by A. Hasanzadeh et al.)

        mu1, mu2 = self.gc2(hidden1, hidden2)  # 均值

        EncSto = (self.encsto == 'full')  # encsto="semi"
        hidden_sd1 = EncSto * hidden1 + (1 - EncSto) * hiddenx1
        hidden_sd2 = EncSto * hidden2 + (1 - EncSto) * hiddenx2

        logstd1, logstd2 = self.gc3(hidden_sd1, hidden_sd2)  # 方差

        # 重参数化
        emb_mu1 = mu1[self.K:, :]
        emb_logvar1 = logstd1[self.K:, :]

        emb_mu2 = mu2[self.K:, :]
        emb_logvar2 = logstd2[self.K:, :]

        # check tensor size compatibility
        assert len(emb_mu1.shape) == len(emb_logvar1.shape), 'mu1 and logvar1 are not equi-dimension.'
        assert len(emb_mu2.shape) == len(emb_logvar2.shape), 'mu2 and logvar2 are not equi-dimension.'

        sampled_z1, eps1 = self.reparameterize(emb_mu1, emb_logvar1)  # 重参数化
        sampled_z2, eps2 = self.reparameterize(emb_mu2, emb_logvar2)  # 重参数化
        return sampled_z1, sampled_z2, eps1, eps2, mu1, mu2, logstd1, logstd2, snr1, snr2

    def forward(self, X1, X2):
        Z1, Z2, eps1, eps2, mu1, mu2, logstd1, logstd2, snr1, snr2 = self.encode(X1, X2)
        # 将两个类堆叠起来
        mu = torch.cat((mu1, mu2), 0)
        print("********mu.shape********")
        print(mu.shape)
        logvar = torch.cat((logstd1, logstd2), 0)
        Z = torch.cat((Z1, Z2), 0)
        eps = torch.cat((eps1, eps2), 0)
        snr = torch.cat((snr1, snr2), 0)
        '''
        print("-----------Z1-----------")
        print(Z1.max())
        print("-----------Z2-----------")
        print(Z2.max())       
        '''
        A_pred = dot_product_decode(Z1, Z2)
        return A_pred, mu, logvar, Z, eps, snr


class VGAE(nn.Module):
    def __init__(self, adj1, adj2):
        super(VGAE, self).__init__()
        self.base_gcn = GraphConvSparse(args.input_dim, args.hidden1_dim, adj1, adj2)
        self.gcn_mean = GraphConvSparse(args.hidden1_dim, args.hidden2_dim, adj1, adj2, activation=lambda x: x)
        self.gcn_logstddev = GraphConvSparse(args.hidden1_dim, args.hidden2_dim, adj1, adj2, activation=lambda x: x)

    def encode(self, X1, X2):
        hidden1, hidden2 = self.base_gcn(X1, X2)
        self.mean1, self.mean2 = self.gcn_mean(hidden1, hidden2)  # p均值，a均值
        self.logstd1, self.logstd2 = self.gcn_logstddev(hidden1, hidden2)  # p方差，a方差

        gaussian_noise1 = torch.randn(X1.size(0), args.hidden2_dim)  # 重参数化的高斯噪声
        '''
        print("******gaussian_noise******")
        print(gaussian_noise1.shape)
        print(self.mean1.shape)
        print(self.logstd1.shape)        
        '''

        sampled_z1 = gaussian_noise1 * torch.exp(self.logstd1) + self.mean1
        gaussian_noise2 = torch.randn(X2.size(0), args.hidden2_dim)  # 重参数化的高斯噪声
        sampled_z2 = gaussian_noise2 * torch.exp(self.logstd2) + self.mean2
        return sampled_z1, sampled_z2

    def forward(self, X1, X2):
        Z1, Z2 = self.encode(X1, X2)
        '''
        print("-----------Z1-----------")
        print(Z1.max())
        print("-----------Z2-----------")
        print(Z2.max())       
        '''
        A_pred = dot_product_decode(Z1, Z2)
        return A_pred


class GraphConvSparse(nn.Module):
    def __init__(self, input_dim, output_dim, adj1, adj2, activation=F.relu, **kwargs):
        super(GraphConvSparse, self).__init__(**kwargs)
        self.weight = glorot_init(input_dim, output_dim)
        self.adj1 = adj1
        self.adj2 = adj2
        self.activation = activation

    def forward(self, inputs1, inputs2):  # input为特征矩阵
        # paper
        x1 = inputs1
        x1 = torch.mm(x1, self.weight)
        x1 = torch.mm(self.adj1, x1)
        outputs1 = self.activation(x1)
        # other
        x2 = inputs2
        x2 = torch.mm(x2, self.weight)
        x2 = torch.mm(self.adj2, x2)
        outputs2 = self.activation(x2)
        return outputs1, outputs2


def dot_product_decode(Z1, Z2):
    A_pred = torch.sigmoid(torch.matmul(Z1, Z2.t()))
    return A_pred


def glorot_init(input_dim, output_dim):
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = torch.rand(input_dim, output_dim) * 2 * init_range - init_range
    return nn.Parameter(initial)


class GAE(nn.Module):
    def __init__(self, adj1, adj2):
        super(GAE, self).__init__()
        self.base_gcn = GraphConvSparse(args.input_dim, args.hidden1_dim, adj1, adj2)
        self.gcn_mean = GraphConvSparse(args.hidden1_dim, args.hidden2_dim, adj1, adj2, activation=lambda x: x)

    def encode(self, X1, X2):
        hidden1, hidden2 = self.base_gcn(X1, X2)
        z1, z2 = self.mean1, self.mean2 = self.gcn_mean(hidden1, hidden2)
        return z1, z2

    def forward(self, X1, X2):
        Z1, Z2 = self.encode(X1, X2)
        A_pred = dot_product_decode(Z1, Z2)
        return A_pred

# class GraphConv(nn.Module):
# 	def __init__(self, input_dim, hidden_dim, output_dim):
# 		super(VGAE,self).__init__()
# 		self.base_gcn = GraphConvSparse(args.input_dim, args.hidden1_dim, adj)
# 		self.gcn_mean = GraphConvSparse(args.hidden1_dim, args.hidden2_dim, adj, activation=lambda x:x)
# 		self.gcn_logstddev = GraphConvSparse(args.hidden1_dim, args.hidden2_dim, adj, activation=lambda x:x)

# 	def forward(self, X, A):
# 		out = A*X*self.w0
# 		out = F.relu(out)
# 		out = A*X*self.w0
# 		return out
