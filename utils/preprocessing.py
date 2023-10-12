
import time
import random
import numpy as np
import scipy.sparse as sp
import torch
import dgl
import warnings
import torch.nn.functional as F

np.set_printoptions(threshold=np.inf)
torch.set_printoptions(threshold=np.inf)

warnings.filterwarnings("ignore")
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve

np.random.seed(0)


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

def preprocess_graph(dataset, adj, device):
    if dataset == 'ACM':
        num_p = 4019
        adj_p = sp.csr_matrix(np.eye(num_p))
        adj.append(adj_p)
        C = []
        for i, adjx in enumerate(adj):
            rows, cols = adjx.nonzero()
            c = list(zip(rows, cols))
            C.append(c)

        graph_data = {
            ('paper', 'written-by', 'author'): C[0],
            ('author', 'write', 'paper'): C[1],
            ('paper', 'in', 'subject'): C[2],
            ('subject', 'out', 'paper'): C[3]
        }
        g = dgl.heterograph(graph_data)
        g = g.to(device)
        print(g)
    elif dataset == 'DBLP':
        num_p = 14328
        adj_p = sp.csr_matrix(np.eye(num_p))
        adj.append(adj_p)
        C = []
        for i, adjx in enumerate(adj):
            rows, cols = adjx.nonzero()
            c = list(zip(rows, cols))
            C.append(c)

        graph_data = {
            ('author', 'write', 'paper'): C[0],
            ('paper', 'written-by', 'author'): C[1],
            ('paper', 'in1', 'term'): C[2],
            ('term', 'out1', 'paper'): C[3],
            ('paper', 'in2', 'venue'): C[4],
            ('venue', 'out2', 'paper'): C[5],
        }
        g = dgl.heterograph(graph_data)
        g = g.to(device)
        print(g)
    elif dataset == 'YELP':
        C = []
        for i, adjx in enumerate(adj):
            rows, cols = adjx.nonzero()
            c = list(zip(rows, cols))
            C.append(c)
        graph_data = {
            ('business', 'used', 'user'): C[0],
            ('user', 'use', 'business'): C[1],
            ('business', 'include', 'service'): C[2],
            ('service', 'included', 'business'): C[3],
            ('business', 'in', 'level'): C[4],
            ('level', 'out', 'business'): C[5],
        }
        g = dgl.heterograph(graph_data)
        g = g.to(device)
        print(g)
    return g


def preprocess_add_node(dataset, graph):
    if dataset == 'ACM':
        num_p = 4019
        num_a = 7167
        num_s = 60
        if graph.number_of_nodes('paper') < num_p:
            graph.add_nodes(num=num_p - graph.number_of_nodes('paper'), ntype='paper')
        if graph.number_of_nodes('author') < num_a:
            graph.add_nodes(num=num_a - graph.number_of_nodes('author'), ntype='author')
        if graph.number_of_nodes('subject') < num_s:
            graph.add_nodes(num=num_s - graph.number_of_nodes('subject'), ntype='subject')
    elif dataset == 'DBLP':
        num_a = 4057
        num_p = 14328
        num_t = 7723
        num_v = 20
        if graph.number_of_nodes('author') < num_a:
            graph.add_nodes(num=num_a - graph.number_of_nodes('author'), ntype='author')
        if graph.number_of_nodes('paper') < num_p:
            graph.add_nodes(num=num_p - graph.number_of_nodes('paper'), ntype='paper')
        if graph.number_of_nodes('term') < num_t:
            graph.add_nodes(num=num_t - graph.number_of_nodes('term'), ntype='term')
        if graph.number_of_nodes('venue') < num_v:
            graph.add_nodes(num=num_v - graph.number_of_nodes('venue'), ntype='venue')
    elif dataset == 'YELP':
        num_b = 2614
        num_u = 1286
        num_s = 4
        num_l = 9
        if graph.number_of_nodes('business') < num_b:
            graph.add_nodes(num=num_b - graph.number_of_nodes('business'), ntype='business')
        if graph.number_of_nodes('user') < num_u:
            graph.add_nodes(num=num_u - graph.number_of_nodes('user'), ntype='user')
        if graph.number_of_nodes('service') < num_s:
            graph.add_nodes(num=num_s - graph.number_of_nodes('service'), ntype='service')
        if graph.number_of_nodes('level') < num_l:
            graph.add_nodes(num=num_l - graph.number_of_nodes('level'), ntype='level')
    return graph

def feature_in_graph(dataset, graph, features_list_tensor, device):
    if dataset == 'ACM':
        num_p = 4019
        num_a = 7167
        num_s = 60
        features_list = torch.split(features_list_tensor, [num_p, num_a, num_s], dim=1)
        graph.nodes['paper'].data['feature'] = torch.squeeze(features_list[0], 0).to(device)
        graph.nodes['author'].data['feature'] = torch.squeeze(features_list[1], 0).to(device)
        graph.nodes['subject'].data['feature'] = torch.squeeze(features_list[2], 0).to(device)

    elif dataset == 'DBLP':
        num_a = 4057
        num_p = 14328
        num_t = 7723
        num_v = 20
        features_list = torch.split(features_list_tensor, [num_a, num_p, num_t, num_v], dim=1)

        graph.nodes['author'].data['feature'] = torch.squeeze(features_list[0], 0).to(device)
        graph.nodes['paper'].data['feature'] = torch.squeeze(features_list[1], 0).to(device)
        graph.nodes['term'].data['feature'] = torch.squeeze(features_list[2], 0).to(device)
        graph.nodes['venue'].data['feature'] = torch.squeeze(features_list[3], 0).to(device)

    elif dataset == 'YELP':
        num_b = 2614
        num_u = 1286
        num_s = 4
        num_l = 9
        features_list = torch.split(features_list_tensor, [num_b, num_u, num_s, num_l], dim=1)

        graph.nodes['business'].data['feature'] = torch.squeeze(features_list[0], 0).to(device)
        graph.nodes['user'].data['feature'] = torch.squeeze(features_list[1], 0).to(device)
        graph.nodes['service'].data['feature'] = torch.squeeze(features_list[2], 0).to(device)
        graph.nodes['level'].data['feature'] = torch.squeeze(features_list[3], 0).to(device)

    return graph

def each_mask_test_edges(adj, rate):
    # Function to build test set with 10% positive links
    # NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.
    # TODO: Clean up.

    adj.eliminate_zeros()

    adj_tuple = sparse_to_tuple(adj)
    edges = adj_tuple[0]

    edges_all = sparse_to_tuple(adj)[0]
    num_test = int(np.floor(edges.shape[0] * rate))
    num_val = int(np.floor(edges.shape[0] * rate * 0.5))

    all_edge_idx = list(range(edges.shape[0]))
    # print("Random number with seed 0 : ", random.random())

    np.random.shuffle(all_edge_idx)

    if rate == 0:
        train_edges = edges
        data = np.ones(train_edges.shape[0])
        adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
        val_edges = 0
        val_edges_false = 0
        test_edges = 0
        test_edges_false = 0
        adj_val = 0
        return adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false, adj_val

    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    test_edges = edges[test_edge_idx]
    val_edges = edges[val_edge_idx]
    train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)

    def ismember(a, b, tol=5):
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return np.any(rows_close)

    test_edges_false = []

    while len(test_edges_false) < len(test_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[1])
        if ismember([idx_i, idx_j], edges_all):
            continue
        if test_edges_false:
            if ismember([idx_i, idx_j], np.array(test_edges_false)):
                continue
        test_edges_false.append([idx_i, idx_j])

    val_edges_false = []
    while len(val_edges_false) < len(val_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[1])
        if ismember([idx_i, idx_j], edges_all):
            continue
        if ismember([idx_i, idx_j], train_edges):
            continue
        if ismember([idx_i, idx_j], val_edges):
            continue
        if val_edges_false:
            if ismember([idx_i, idx_j], np.array(val_edges_false)):
                continue
        val_edges_false.append([idx_i, idx_j])

    assert ~ismember(test_edges_false, edges_all)
    assert ~ismember(val_edges_false, edges_all)
    assert ~ismember(val_edges, train_edges)
    assert ~ismember(test_edges, train_edges)
    assert ~ismember(val_edges, test_edges)

    data = np.ones(train_edges.shape[0])


    # Re-build adj matrix
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_val = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)

    # NOTE: these edge lists only contain single direction of edge!
    return adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false, adj_val


def mask_test_edges(all_adj, mask_rate):
    # Function to build test set with 10% positive links
    # NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.
    # TODO: Clean up.
    all_adj_train, all_train_edges, all_val_edges, all_val_edges_false, all_test_edges, all_test_edges_false, all_adj_val = [], [], [], [], [], [], []
    for (adj, rate) in list(zip(all_adj, mask_rate)):
        adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false, adj_val = each_mask_test_edges(
            adj,
            rate)
        all_adj_train.append(adj_train)
        all_train_edges.append(train_edges)
        all_val_edges.append(val_edges)
        all_val_edges_false.append(val_edges_false)
        all_test_edges.append(test_edges)
        all_test_edges_false.append(test_edges_false)
        all_adj_val.append(adj_val)
    return all_adj_train, all_train_edges, all_val_edges, all_val_edges_false, all_test_edges, all_test_edges_false, all_adj_val


def mask_test_feas(features, rate):
    feature_norm = features
    features = sp.csc_matrix(features)
    fea_row = features.nonzero()[0]
    fea_col = features.nonzero()[1]
    feas = []
    feas_dic = {}
    for i in range(len(fea_row)):
        feas.append([fea_row[i], fea_col[i]])
        feas_dic[(fea_row[i], fea_col[i])] = feature_norm[fea_row[i]][fea_col[i]]
    num_test = int(np.floor(len(feas) * rate))
    num_val = int(np.floor(len(feas) * rate * 0.5))
    all_fea_idx = np.arange(len(feas))
    np.random.shuffle(all_fea_idx)
    if rate == 0:
        feas = np.array(feas)
        train_feas = feas
        fea_train = np.zeros((feature_norm.shape[0], feature_norm.shape[1]), dtype=float)
        for idx in train_feas:
            fea_train[idx[0]][idx[1]] = feas_dic[(idx[0], idx[1])]
        fea_train = sp.csr_matrix(fea_train)
        val_feas = 0
        fea_val = 0
        test_feas = 0
        fea_test = 0
        return fea_train, train_feas, fea_val, val_feas, fea_test, test_feas
    val_fea_idx = all_fea_idx[:num_val]
    test_fea_idx = all_fea_idx[num_val:(num_val + num_test)]
    feas = np.array(feas)
    test_feas = feas[test_fea_idx]
    val_feas = feas[val_fea_idx]
    train_feas = np.delete(feas, np.hstack([test_fea_idx, val_fea_idx]), axis=0)
    fea_train = np.zeros((feature_norm.shape[0], feature_norm.shape[1]), dtype=float)
    fea_val = np.zeros((feature_norm.shape[0], feature_norm.shape[1]), dtype=float)
    fea_test = np.zeros((feature_norm.shape[0], feature_norm.shape[1]), dtype=float)
    for idx in train_feas:
        fea_train[idx[0]][idx[1]] = feas_dic[(idx[0], idx[1])]

    for idx in val_feas:
        fea_val[idx[0]][idx[1]] = feas_dic[(idx[0], idx[1])]

    for idx in test_feas:
        fea_test[idx[0]][idx[1]] = feas_dic[(idx[0], idx[1])]

    fea_train = sp.csr_matrix(fea_train)
    fea_val = sp.csr_matrix(fea_val)
    fea_test = sp.csr_matrix(fea_test)

    return fea_train, train_feas, fea_val, val_feas, fea_test, test_feas


def get_roc_score(emb1, emb2, edges_pos, edges_neg, gdc):
    def GraphDC(x):
        if gdc == 'ip':
            return 1 / (1 + np.exp(-x))
        elif gdc == 'bp':
            return 1 - np.exp(- np.exp(x))

    if type(edges_pos) == int:
        roc_score = 0.
        ap_score = 0.
        return roc_score, ap_score

    J = emb1.shape[0]

    # Predict on test set of edges
    edges_pos = np.array(edges_pos).transpose((1, 0))
    emb_pos_sp = emb1[:, edges_pos[0], :]
    emb_pos_ep = emb2[:, edges_pos[1], :]
    # preds_pos is torch.Tensor with shape [J, #pos_edges]
    preds_pos = GraphDC(
        np.einsum('ijk,ijk->ij', emb_pos_sp, emb_pos_ep)
    )

    edges_neg = np.array(edges_neg).transpose((1, 0))
    emb_neg_sp = emb1[:, edges_neg[0], :]
    emb_neg_ep = emb2[:, edges_neg[1], :]
    preds_neg = GraphDC(
        np.einsum('ijk,ijk->ij', emb_neg_sp, emb_neg_ep)
    )

    preds_all = np.hstack([preds_pos, preds_neg])

    labels_all = np.hstack([np.ones(preds_pos.shape[-1]), np.zeros(preds_neg.shape[-1])])

    where_are_nan = np.isnan(preds_all)
    where_are_inf = np.isinf(preds_all)
    preds_all[where_are_nan] = 0.
    preds_all[where_are_inf] = 0.

    roc_score = np.array(
        [roc_auc_score(labels_all, pred_all.flatten()) \
         for pred_all in np.vsplit(preds_all, J)]
    ).mean()
    ap_score = np.array(
        [average_precision_score(labels_all, pred_all.flatten()) \
         for pred_all in np.vsplit(preds_all, J)]
    ).mean()

    return roc_score, ap_score

def get_mseloss_score(mask, fea_orig, fea_pred):
    SMALL = 1e-6
    fea_pred = torch.clamp(fea_pred, min=SMALL, max=1 - SMALL)
    sum = torch.count_nonzero(mask).item()
    def get_rec(pred):
        loss_ac = F.mse_loss(pred, fea_orig, reduction="sum")
        loss_ac = torch.sqrt(loss_ac)
        return loss_ac

    rec_costs = torch.stack(
        # [get_rec(pred) for pred in torch.unbind(fea_pred, dim=0)]
        [get_rec(mask * pred) for pred in torch.unbind(fea_pred, dim=0)]
    )
    rec_cost = rec_costs.mean()
    return rec_cost

def get_mseloss_score1(dic, fea_orig, fea_pred):
    if type(dic) == int:
        maeloss_score = 100.
        return maeloss_score
    orig = torch.stack([fea_orig[idx[0]][idx[1]] for idx in dic], dim=-1)
    mseloss_score = []
    for pred in torch.unbind(fea_pred, dim=0):
        pred_m = torch.stack([pred[idx[0]][idx[1]] for idx in dic], dim=-1)
        loss_ac = F.mse_loss(pred_m, orig, reduction="mean")

        mseloss_score.append(loss_ac)
    mseloss_score = torch.stack(mseloss_score).mean()
    return mseloss_score

def get_maeloss_score(dic, fea_orig, fea_pred):
    if type(dic) == int:
        maeloss_score = 100.
        return maeloss_score
    orig = torch.stack([fea_orig[idx[0]][idx[1]] for idx in dic], dim=-1)
    maeloss_score = []
    for pred in torch.unbind(fea_pred, dim=0):
        pred_m = torch.stack([pred[idx[0]][idx[1]] for idx in dic], dim=-1)
        loss_ac = F.l1_loss(pred_m, orig, reduction="mean")
        maeloss_score.append(loss_ac)
    maeloss_score = torch.stack(maeloss_score).mean()
    return maeloss_score