import argparse
import time
import random
from torch import optim
from utils.tools import EarlyStopping
from model.model import GraMI
from utils.input_data import load_DBLP_data
from utils.preprocessing import *
from utils.optimizer import  loss_function_a_mse, loss_get_rec, loss_distribution, \
    get_pos_norm
import torch, gc
import warnings
import torch.backends.cudnn as cudnn

cudnn.deterministic = True
cudnn.benchmark = True
torch.backends.cudnn.enabled = True

warnings.filterwarnings("ignore")
gc.collect()
torch.cuda.empty_cache()

parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--model', type=str, default='gcn_vae', help="models used.")
parser.add_argument('--seed', type=int, default=8, help='Random seed.')
parser.add_argument('--epochs', type=int, default=3500, help='Number of epochs to train.')
parser.add_argument('--patience', type=int, default=100, help='Patience. Default is 100.')
parser.add_argument('--weight-decay', type=float, default=0.001, help='Weight decay.Default is 0.001')
parser.add_argument('--hidden-dim', type=int, default=256, help='Number of units in fc_layer.')
parser.add_argument('--hidden1', type=int, default=256, help='Number of units in hidden layer 1.')
parser.add_argument('--hidden2', type=int, default=128, help='Number of units in hidden layer 2.')
parser.add_argument('--num-heads', type=int, default=4, help='Number of the attention heads. Default is 4.')
parser.add_argument('--lr', type=float, default=0.005, help='Initial learning rate.')
parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset', type=str, default='DBLP', help='type of dataset.')
parser.add_argument('--monit', type=int, default=50,
                    help='Number of epochs to train before a test')
parser.add_argument('--edim', type=int, default=64,
                    help='Number of units in noise epsilon.')
parser.add_argument('--encsto', type=str, default='full', help='encoder stochasticity.')
parser.add_argument('--gdc', type=str, default='ip', help='type of graph decoder')
parser.add_argument('--noise-dist', type=str, default='Bernoulli',
                    help='Distriubtion of random noise in generating psi.')
parser.add_argument('--K', type=int, default=1,
                    help='number of samples to draw for MC estimation of h(psi).')
parser.add_argument('--J', type=int, default=1,
                    help='Number of samples to draw for MC estimation of log-likelihood.')
parser.add_argument('--task', type=str, default='pv',
                    help='the choise of tasks,pa/pt/pv is prediction of links, fea is prediction of feature.')
parser.add_argument('--mask-rate-link', type=list, default=[0., 0., 0.],
                    help='The rate of mask each relation .')
parser.add_argument('--mask-rate-fea', type=float, default=0., help='The rate of mask orig feature')
parser.add_argument('--src-node', type=int, default=1,
                    help='the node type of source node which has attribute.')
parser.add_argument('--tar-node', type=int, default=0,
                    help='the node type of target node which is used to classify')
parser.add_argument('--loss-lambda1', type=float, default=0.2, help='Coefficient lambda to balance loss.')
parser.add_argument('--loss-lambda2', type=float, default=0.2, help='Coefficient lambda to balance loss.')
parser.add_argument('--method', type=str, default='In', help='The choise of method')
parser.add_argument('--gpu-id', type=int, default=2, help='GPU ID.')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
args.device = f'cuda:{args.gpu_id}' if args.cuda else 'cpu'

seed = args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if args.cuda:
    torch.cuda.manual_seed(seed)

n_classes = 4
n_relations = 4

save_model_path = 'checkpoint/checkpoint_DBLP.pt'

if args.task == 'pa':
    args.mask_rate_link = [0.1, 0., 0.]
elif args.task == 'pt':
    args.mask_rate_link = [0., 0.1, 0.]
elif args.task == 'pv':
    args.mask_rate_link = [0., 0., 0.1]
elif args.task == 'fea':
    args.mask_rate_fea = 0.1


def main_DBLP(args):
    adjlists, edge_metapath_indices_list, features_list, adjM, type_mask, labels, train_val_test_idx = load_DBLP_data()
    test_idx = train_val_test_idx['test_idx']
    src_node = args.src_node
    tar_node = args.tar_node
    num_src_node = features_list[src_node].shape[0]
    num_tar_node = features_list[tar_node].shape[0]
    n_nodes = 0
    for i in range(len(features_list)):
        n_nodes += features_list[i].shape[0]
    features_src = features_list[src_node]

    all_mask = []
    for i in range(n_relations):
        all_mask.append(np.where(type_mask == i)[0])
        if i == src_node:
            features_list[i] = torch.FloatTensor(np.array(features_list[i]))
        else:
            features_list[i] = torch.FloatTensor(np.array(features_list[i]))

    feats_dim_list = [features.shape[1] for features in features_list]
    all_adj = []
    for i in range(len(all_mask)):
        if i == src_node:
            continue
        else:
            all_adj.append(sp.csr_matrix(adjM[all_mask[src_node], :][:, all_mask[i]]))

    all_adj_train, all_train_edges, all_val_edges, all_val_edges_false, all_test_edges, all_test_edges_false, all_adj_val = mask_test_edges(
        all_adj,
        args.mask_rate_link)
    fea_train, train_feas, fea_val, val_feas, fea_test, test_feas = mask_test_feas(features_src,
                                                                                   args.mask_rate_fea)
    mask_fea = np.zeros((fea_train.shape[0], fea_train.shape[1]), dtype=float)
    mask_fea[fea_train.nonzero()] = 1
    mask_fea = sp.csr_matrix(mask_fea)
    all_adj = all_adj_train

    adj = []
    for i in range(len(all_adj)):
        if i == 0:  # ap pa
            adj.append(all_adj[i].T)
            adj.append(all_adj[i])
        else:
            adj.append(all_adj[i])
            adj.append(all_adj[i].T)

    graph = preprocess_graph(args.dataset, adj, device=args.device)
    graph = preprocess_add_node(args.dataset, graph)

    features_orig = features_src
    features_orig = sp.csr_matrix(features_orig)
    features_src = sparse_to_tuple(sp.csr_matrix(features_src))
    num_features_src = features_src[2][1]

    # Creat Model
    all_pos_weight = []
    all_norm = []
    for i in range(len(all_adj)):
        pos_weight, norm = get_pos_norm(all_adj[i])
        all_pos_weight.append(pos_weight)
        all_norm.append(norm)

    all_adj_label = all_adj_train
    for i in range(len(all_adj_label)):
        all_adj_label[i] = torch.FloatTensor(all_adj_label[i].toarray())
    for i in range(len(all_adj_val)):
        if type(all_adj_val[i]) != int:
            all_adj_val[i] = torch.FloatTensor(all_adj_val[i].toarray())
    features_src_label = torch.FloatTensor(fea_train.toarray())

    model = GraMI(args.dataset, graph, src_node, feats_dim_list, n_nodes, args.edim, num_features_src,
                  args.hidden_dim,
                  args.hidden1, args.hidden2, args.num_heads, args.dropout, encsto=args.encsto,
                  gdc=args.gdc,
                  ndist=args.noise_dist, copyK=args.K, copyJ=args.J, device=args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr,weight_decay=args.weight_decay)
    for i in range(len(features_list)):
        features_list[i] = features_list[i].to(args.device)

    model.to(args.device)
    for i in range(len(all_adj_label)):
        all_adj_label[i] = all_adj_label[i].to(args.device)
    for i in range(len(all_adj_val)):
        if type(all_adj_val[i]) != int:
            all_adj_val[i] = all_adj_val[i].to(args.device)
    features_src_label = features_src_label.to(args.device)
    for i in range(len(all_pos_weight)):
        all_pos_weight[i] = all_pos_weight[i].to(args.device)
    mask_fea = torch.FloatTensor(mask_fea.toarray()).to(args.device)
    features_orig = torch.FloatTensor(features_orig.toarray()).to(args.device)
    fea_train = torch.FloatTensor(fea_train.toarray()).to(args.device)
    if type(fea_val) != int:
        fea_val = torch.FloatTensor(fea_val.toarray()).to(args.device)
    if type(fea_test) != int:
        fea_test = torch.FloatTensor(fea_test.toarray()).to(args.device)

    early_stopping = EarlyStopping(patience=args.patience, verbose=True, save_path=save_model_path)
    for epoch in range(args.epochs):
        t = time.time()
        model.train()
        optimizer.zero_grad()
        label_a, (
            recovered_all, pred_f, fea_recover, mu_list, muf, logvar_list, logvar_f, z_all, zf, za, z_scaled1_all,
            z_scaled2_all,
            z_scaledf,
            z_scaled_a, eps_all, epsf, rk_all) = model(
            features_list, graph)
        label_a = label_a.to(args.device)

        loss_rec_all, loss_prior_all, loss_post_all = [], [], []
        for (recovered, adj_label, norm, pos_weight) in list(
                zip(recovered_all, all_adj_label, all_norm, all_pos_weight)):
            loss_rec_all.append(loss_get_rec(recovered, adj_label, norm, pos_weight))
        for (mu, logvar, emb, eps) in list(zip(mu_list, logvar_list, z_all, eps_all)):
            loss_prior, loss_post = loss_distribution(mu, logvar, emb, eps)
            loss_prior_all.append(loss_prior)
            loss_post_all.append(loss_post)

        loss_rec_f, loss_prior_f, loss_post_f = loss_function_a_mse(
            preds=pred_f,
            labels=label_a,
            mu=muf,
            logvar=logvar_f,
            emb=zf,
            eps=epsf
        )
        loss_post_all.append(loss_post_f)
        loss_prior_all.append(loss_prior_f)

        loss_recover = get_mseloss_score(mask=mask_fea, fea_orig=fea_train, fea_pred=fea_recover)

        loss_rec = sum(loss_rec_all)
        loss_post = sum(loss_post_all)
        loss_prior = sum(loss_prior_all)

        WU = np.min([epoch / 300., 1.])
        reg = (loss_post - loss_prior) * WU / (n_nodes ** 2)
        loss_train = loss_rec + WU * reg + args.loss_lambda1 * loss_rec_f + args.loss_lambda2 * loss_recover
        loss_train.backward()

        cur_loss = loss_train.item()
        cur_link_rec = loss_rec.item()
        cur_fea_rec = loss_rec_f.item()
        cur_loss_recover = loss_recover.item()
        optimizer.step()

        loss_rec_all_val = []
        for (recovered, adj_val, norm, pos_weight) in list(
                zip(recovered_all, all_adj_val, all_norm, all_pos_weight)):
            if type(adj_val) == int:
                loss_rec_all_val.append(0)
            else:
                loss_rec_all_val.append(loss_get_rec(recovered, adj_val, norm, pos_weight))
        val_fea_loss = get_mseloss_score1(dic=val_feas, fea_orig=fea_val,
                                          fea_pred=fea_recover)
        loss_val = sum(
            loss_rec_all_val) + WU * reg + args.loss_lambda1 * loss_rec_f + args.loss_lambda2 * val_fea_loss
        ap_val_all = []
        roc_val_all = []
        for (hidden_emb1, hidden_emb2, val_edges, val_edges_false) in list(
                zip(z_scaled1_all, z_scaled2_all, all_val_edges,
                    all_val_edges_false)):
            hidden_emb1 = hidden_emb1.detach().cpu().numpy()
            hidden_emb2 = hidden_emb2.detach().cpu().numpy()
            roc_cur, ap_cur = get_roc_score(hidden_emb1, hidden_emb2, val_edges, val_edges_false, args.gdc)
            roc_val_all.append(roc_cur)
            ap_val_all.append(ap_cur)

        print(
            "Epoch:", '%04d' % (epoch + 1),
            "train_loss=", "{:.5f}".format(cur_loss),
            "link_rec_loss=", "{:.5f}".format(cur_link_rec),
            "fea_rec_loss=", "{:.5f}".format(cur_fea_rec),
            "fea_orig_loss=", "{:.5f}".format(cur_loss_recover),
            end=' '
        )
        if args.task == 'pa':
            print("link_val_ap=", "{:.5f}".format(ap_val_all[0]),end=' ')
        elif args.task == 'pt':
            print("link_val_ap=", "{:.5f}".format(ap_val_all[1]),end=' ')
        elif args.task == 'pv':
            print("link_val_ap=", "{:.5f}".format(ap_val_all[2]),end=' ')
        elif args.task == 'fea':
            print("fea_val_ap=", "{:.5f}".format(val_fea_loss),end=' ')
        print(
            "time=", "{:.5f}".format(time.time() - t)
        )
        early_stopping(loss_val, model, np.sum(roc_val_all), np.sum(ap_val_all))
        if early_stopping.early_stop:
            print('Early stopping!')
            break
    model.eval()
    with torch.no_grad():
        label_a, (
            recovered_all, pred_f, fea_recover, mu_list, muf, logvar_list, logvar_f, z_all, zf, za, z_scaled1_all,
            z_scaled2_all,
            z_scaledf,
            z_scaled_a, eps_all, epsf, rk_all) = model(
            features_list, graph)
        roc_score_all, ap_score_all = [], []
        for (hidden_emb1, hidden_emb2, test_edges, test_edges_false) in list(
                zip(z_scaled1_all, z_scaled2_all,
                    all_test_edges,
                    all_test_edges_false)):
            hidden_emb1 = hidden_emb1.detach().cpu().numpy()
            hidden_emb2 = hidden_emb2.detach().cpu().numpy()
            roc_score, ap_score = get_roc_score(hidden_emb1, hidden_emb2, test_edges, test_edges_false,
                                                args.gdc)
            roc_score_all.append(roc_score)
            ap_score_all.append(ap_score)
        test_fea_loss = get_mseloss_score1(dic=test_feas, fea_orig=fea_test,
                                           fea_pred=fea_recover)

        if args.task == 'pa':
            rslt = "Test ROC score: {:.4f},Test AP score: {:.4f}\n".format(roc_score_all[0], ap_score_all[0])
        elif args.task == 'pt':
            rslt = "Test ROC score: {:.4f}, Test AP score: {:.4f}\n".format(roc_score_all[1], ap_score_all[1])
        elif args.task == 'pv':
            rslt = "Test ROC score: {:.4f}, Test AP score: {:.4f}\n".format(roc_score_all[2], ap_score_all[2])
        elif args.task == 'fea':
            rslt = "Test Loss score:{:.4f}\n".format(test_fea_loss)
        print("\n", rslt, "\n")
    print("Optimization Finished!")


if __name__ == '__main__':
    print('--------args----------')
    for k in list(vars(args).keys()):
        print('%s: %s' % (k, vars(args)[k]))
    print('--------args----------\n')
    main_DBLP(args)
