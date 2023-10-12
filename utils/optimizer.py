import torch
import torch.nn.modules.loss
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp

def loss_function_a_mse(preds, labels, mu, logvar, emb, eps):
    def get_rec(pred):
        loss_ac = F.mse_loss(pred, labels, reduction="mean")
        loss_ac = torch.sqrt(loss_ac)
        return loss_ac

    SMALL = 1e-6
    std = torch.exp(0.5 * logvar)
    J, N, zdim = emb.shape
    K = mu.shape[0] - J

    mu_mix, mu_emb = mu[:K, :], mu[K:, :]
    std_mix, std_emb = std[:K, :], std[K:, :]

    preds = torch.clamp(preds, min=SMALL, max=1 - SMALL)
    rec_costs = torch.stack(
        [get_rec(pred) for pred in torch.unbind(preds, dim=0)]
    )

    # average over J * N * N items
    rec_cost = rec_costs.mean()

    # compute log_prior_ker, the constant 1/sqrt(2*pi) is cancelled out.
    # average over J items
    log_prior_ker = torch.sum(- 0.5 * emb.pow(2), dim=[1, 2]).mean()

    # compute log_posterior
    # Z.shape = [J, 1, N, zdim]
    Z = emb.view(J, 1, N, zdim)

    # mu_mix.shape = std_mix.shape = [1, K, N, zdim]
    mu_mix = mu_mix.view(1, K, N, zdim)
    std_mix = std_mix.view(1, K, N, zdim)

    # compute -log std[k] - (Z[j] - mu[k])^2 / 2*std[k]^2 for all (j,k)
    # the shape of result tensor log_post_ker_JK is [J,K]
    log_post_ker_JK = - torch.sum(
        0.5 * ((Z - mu_mix) / (std_mix + SMALL)).pow(2), dim=[-2, -1]
    )

    log_post_ker_JK += - torch.sum(
        (std_mix + SMALL).log(), dim=[-2, -1]
    )

    # compute -log std[j] - (Z[j] - mu[j])^2 / 2*std[j]^2 for j = 1,2,...,J
    # the shape of result tensor log_post_ker_J is [J, 1]
    log_post_ker_J = - torch.sum(
        0.5 * eps.pow(2), dim=[-2, -1]
    )
    log_post_ker_J += - torch.sum(
        (std_emb + SMALL).log(), dim=[-2, -1]
    )
    log_post_ker_J = log_post_ker_J.view(-1, 1)

    # bind up log_post_ker_JK and log_post_ker_J into log_post_ker, the shape of result tensor is [J, K+1].
    log_post_ker = torch.cat([log_post_ker_JK, log_post_ker_J], dim=-1)

    # apply "log-mean-exp" to the above tensor
    log_post_ker -= np.log(K + 1.) / J
    # average over J items.
    log_posterior_ker = torch.logsumexp(log_post_ker, dim=-1).mean()

    return rec_cost, log_prior_ker, log_posterior_ker


def loss_function_a(preds, labels, mu, logvar, emb, eps, norm, pos_weight):
    def get_rec(pred):
        log_lik = norm * (pos_weight * labels * torch.log(pred) + (1 - labels) * torch.log(1 - pred))
        rec = -log_lik.mean()
        return rec

    SMALL = 1e-6
    std = torch.exp(0.5 * logvar)
    J, N, zdim = emb.shape
    K = mu.shape[0] - J

    mu_mix, mu_emb = mu[:K, :], mu[K:, :]
    std_mix, std_emb = std[:K, :], std[K:, :]

    preds = torch.clamp(preds, min=SMALL, max=1 - SMALL)

    # compute rec_cost
    rec_costs = torch.stack(
        [get_rec(pred) for pred in torch.unbind(preds, dim=0)],
        dim=0)
    # average over J * N * N items
    rec_cost = rec_costs.mean()

    # compute log_prior_ker, the constant 1/sqrt(2*pi) is cancelled out.
    # average over J items
    log_prior_ker = torch.sum(- 0.5 * emb.pow(2), dim=[1, 2]).mean()
    # compute log_posterior
    # Z.shape = [J, 1, N, zdim]
    Z = emb.view(J, 1, N, zdim)

    # mu_mix.shape = std_mix.shape = [1, K, N, zdim]
    mu_mix = mu_mix.view(1, K, N, zdim)
    std_mix = std_mix.view(1, K, N, zdim)

    # compute -log std[k] - (Z[j] - mu[k])^2 / 2*std[k]^2 for all (j,k)
    # the shape of result tensor log_post_ker_JK is [J,K]
    log_post_ker_JK = - torch.sum(
        0.5 * ((Z - mu_mix) / (std_mix + SMALL)).pow(2), dim=[-2, -1]
    )

    log_post_ker_JK += - torch.sum(
        (std_mix + SMALL).log(), dim=[-2, -1]
    )

    # compute -log std[j] - (Z[j] - mu[j])^2 / 2*std[j]^2 for j = 1,2,...,J
    # the shape of result tensor log_post_ker_J is [J, 1]
    log_post_ker_J = - torch.sum(
        0.5 * eps.pow(2), dim=[-2, -1]
    )
    log_post_ker_J += - torch.sum(
        (std_emb + SMALL).log(), dim=[-2, -1]
    )
    log_post_ker_J = log_post_ker_J.view(-1, 1)

    # bind up log_post_ker_JK and log_post_ker_J into log_post_ker, the shape of result tensor is [J, K+1].
    log_post_ker = torch.cat([log_post_ker_JK, log_post_ker_J], dim=-1)

    # apply "log-mean-exp" to the above tensor
    log_post_ker -= np.log(K + 1.) / J
    # average over J items.
    log_posterior_ker = torch.logsumexp(log_post_ker, dim=-1).mean()

    return rec_cost, log_prior_ker, log_posterior_ker


def loss_function(preds, labels, mu1, mu2, logvar1, logvar2, emb1, emb2, eps1, eps2, norm, pos_weight):
    def get_rec(pred):
        log_lik = norm * (pos_weight * labels * torch.log(pred) + (1 - labels) * torch.log(1 - pred))  # N * N
        rec = -log_lik.mean()
        return rec

    SMALL = 1e-6
    std1 = torch.exp(0.5 * logvar1)
    J1, N1, zdim1 = emb1.shape
    K1 = mu1.shape[0] - J1
    std2 = torch.exp(0.5 * logvar2)
    J2, N2, zdim2 = emb2.shape
    K2 = mu2.shape[0] - J2

    mu_mix1, mu_emb1 = mu1[:K1, :], mu1[K1:, :]
    std_mix1, std_emb1 = std1[:K1, :], std1[K1:, :]
    mu_mix2, mu_emb2 = mu2[:K2, :], mu2[K2:, :]
    std_mix2, std_emb2 = std2[:K2, :], std2[K2:, :]

    preds = torch.clamp(preds, min=SMALL, max=1 - SMALL)

    # compute rec_cost
    rec_costs = torch.stack(
        [get_rec(pred) for pred in torch.unbind(preds, dim=0)],
        dim=0)
    # average over J * N * N items
    rec_cost = rec_costs.mean()

    # compute log_prior_ker, the constant 1/sqrt(2*pi) is cancelled out.
    # average over J items
    log_prior_ker1 = torch.sum(- 0.5 * emb1.pow(2), dim=[1, 2]).mean()
    log_prior_ker2 = torch.sum(- 0.5 * emb2.pow(2), dim=[1, 2]).mean()

    # compute log_posterior
    # Z.shape = [J, 1, N, zdim]
    Z1 = emb1.view(J1, 1, N1, zdim1)
    Z2 = emb2.view(J2, 1, N2, zdim2)

    # mu_mix.shape = std_mix.shape = [1, K, N, zdim]
    mu_mix1 = mu_mix1.view(1, K1, N1, zdim1)
    std_mix1 = std_mix1.view(1, K1, N1, zdim1)
    mu_mix2 = mu_mix2.view(1, K2, N2, zdim2)
    std_mix2 = std_mix2.view(1, K2, N2, zdim2)

    # compute -log std[k] - (Z[j] - mu[k])^2 / 2*std[k]^2 for all (j,k)
    # the shape of result tensor log_post_ker_JK is [J,K]
    log_post_ker_JK1 = - torch.sum(
        0.5 * ((Z1 - mu_mix1) / (std_mix1 + SMALL)).pow(2), dim=[-2, -1]
    )
    log_post_ker_JK1 += - torch.sum(
        (std_mix1 + SMALL).log(), dim=[-2, -1]
    )

    log_post_ker_JK2 = - torch.sum(
        0.5 * ((Z2 - mu_mix2) / (std_mix2 + SMALL)).pow(2), dim=[-2, -1]
    )

    log_post_ker_JK2 += - torch.sum(
        (std_mix2 + SMALL).log(), dim=[-2, -1]
    )

    # compute -log std[j] - (Z[j] - mu[j])^2 / 2*std[j]^2 for j = 1,2,...,J
    # the shape of result tensor log_post_ker_J is [J, 1]
    log_post_ker_J1 = - torch.sum(
        0.5 * eps1.pow(2), dim=[-2, -1]
    )
    log_post_ker_J1 += - torch.sum(
        (std_emb1 + SMALL).log(), dim=[-2, -1]
    )
    log_post_ker_J1 = log_post_ker_J1.view(-1, 1)

    log_post_ker_J2 = - torch.sum(
        0.5 * eps2.pow(2), dim=[-2, -1]
    )
    log_post_ker_J2 += - torch.sum(
        (std_emb2 + SMALL).log(), dim=[-2, -1]
    )
    log_post_ker_J2 = log_post_ker_J2.view(-1, 1)

    # bind up log_post_ker_JK and log_post_ker_J into log_post_ker, the shape of result tensor is [J, K+1].
    log_post_ker1 = torch.cat([log_post_ker_JK1, log_post_ker_J1], dim=-1)
    log_post_ker2 = torch.cat([log_post_ker_JK2, log_post_ker_J2], dim=-1)

    # apply "log-mean-exp" to the above tensor
    log_post_ker1 -= np.log(K1 + 1.) / J1
    log_post_ker2 -= np.log(K2 + 1.) / J2
    # average over J items.
    log_posterior_ker1 = torch.logsumexp(log_post_ker1, dim=-1).mean()
    log_posterior_ker2 = torch.logsumexp(log_post_ker2, dim=-1).mean()

    return rec_cost, log_prior_ker1, log_prior_ker2, log_posterior_ker1, log_posterior_ker2

def get_pos_norm(adj):
    pos_weight = torch.tensor(adj.shape[0] * adj.shape[1] - adj.sum()) / adj.sum()
    norm = adj.shape[0] * adj.shape[1] / float(
        (adj.shape[0] * adj.shape[1] - adj.sum()) * 2)
    return pos_weight, norm

def loss_get_rec(preds, labels, norm, pos_weight):
    def get_rec(pred):
        log_lik = norm * (pos_weight * labels * torch.log(pred) + (1 - labels) * torch.log(1 - pred))
        rec = -log_lik.mean()
        return rec

    SMALL = 1e-6
    preds = torch.clamp(preds, min=SMALL, max=1 - SMALL)

    # compute rec_cost
    rec_costs = torch.stack(
        [get_rec(pred) for pred in torch.unbind(preds, dim=0)],
        dim=0)
    # average over J * N * N items
    rec_cost = rec_costs.mean()
    return rec_cost

def loss_distribution(mu, logvar, emb, eps):
    SMALL = 1e-6
    std = torch.exp(0.5 * logvar)
    J, N, zdim = emb.shape
    K = mu.shape[0] - J

    mu_mix, mu_emb = mu[:K, :], mu[K:, :]
    std_mix, std_emb = std[:K, :], std[K:, :]

    # compute log_prior_ker, the constant 1/sqrt(2*pi) is cancelled out.
    # average over J items
    log_prior_ker = torch.sum(- 0.5 * emb.pow(2), dim=[1, 2]).mean()
    # compute log_posterior
    # Z.shape = [J, 1, N, zdim]
    Z = emb.view(J, 1, N, zdim)

    # mu_mix.shape = std_mix.shape = [1, K, N, zdim]
    mu_mix = mu_mix.view(1, K, N, zdim)
    std_mix = std_mix.view(1, K, N, zdim)

    # compute -log std[k] - (Z[j] - mu[k])^2 / 2*std[k]^2 for all (j,k)
    # the shape of result tensor log_post_ker_JK is [J,K]
    log_post_ker_JK = - torch.sum(
        0.5 * ((Z - mu_mix) / (std_mix + SMALL)).pow(2), dim=[-2, -1]
    )

    log_post_ker_JK += - torch.sum(
        (std_mix + SMALL).log(), dim=[-2, -1]
    )

    # compute -log std[j] - (Z[j] - mu[j])^2 / 2*std[j]^2 for j = 1,2,...,J
    # the shape of result tensor log_post_ker_J is [J, 1]
    log_post_ker_J = - torch.sum(
        0.5 * eps.pow(2), dim=[-2, -1]
    )
    log_post_ker_J += - torch.sum(
        (std_emb + SMALL).log(), dim=[-2, -1]
    )
    log_post_ker_J = log_post_ker_J.view(-1, 1)

    # bind up log_post_ker_JK and log_post_ker_J into log_post_ker, the shape of result tensor is [J, K+1].
    log_post_ker = torch.cat([log_post_ker_JK, log_post_ker_J], dim=-1)

    # apply "log-mean-exp" to the above tensor
    log_post_ker -= np.log(K + 1.) / J
    # average over J items.
    log_posterior_ker = torch.logsumexp(log_post_ker, dim=-1).mean()
    return log_prior_ker, log_posterior_ker
