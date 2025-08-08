import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import wandb
import random
import collections
from .layers import MLPLayers
from .rq import ResidualVectorQuantizer



class MoE_RQVAE_no_cf(nn.Module):
    def __init__(self,
                 in_dim=768,
                 num_emb_list=None,
                 e_dim=64,
                 layers=None,
                 dropout_prob=0.0,
                 bn=False,
                 loss_type="mse",
                 quant_loss_weight=1.0,
                 kmeans_init=False,
                 kmeans_iters=100,
                 sk_epsilons= None,
                 sk_iters=100,
                 alpha = 1.0,
                 beta = 0.001,
                 n_clusters = 10,
                 sample_strategy = 'all',
                 cf_embedding = 0,
                 num_experts = 10
        ):
        super(MoE_RQVAE_no_cf, self).__init__()

        self.in_dim = in_dim
        self.num_emb_list = num_emb_list
        self.e_dim = e_dim
        self.layers = layers
        self.dropout_prob = dropout_prob
        self.bn = bn
        self.loss_type = loss_type
        self.quant_loss_weight=quant_loss_weight
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_epsilons = sk_epsilons
        self.sk_iters = sk_iters
        self.cf_embedding = cf_embedding
        self.alpha = alpha
        self.beta = beta
        self.n_clusters = n_clusters
        self.sample_strategy = sample_strategy
        self.num_experts = num_experts


        self.encode_layer_dims = [self.in_dim] + self.layers + [self.e_dim]
        print("encoder layers:", self.encode_layer_dims)
        self.encoder = MLPLayers(layers=self.encode_layer_dims,
                                 dropout=self.dropout_prob,bn=self.bn)

        self.rq = ResidualVectorQuantizer(num_emb_list, e_dim, beta=self.beta,
                                          kmeans_init = self.kmeans_init,
                                          kmeans_iters = self.kmeans_iters,
                                          sk_epsilons=self.sk_epsilons,
                                          sk_iters=self.sk_iters, num_experts=self.num_experts)

        self.decode_layer_dims = self.encode_layer_dims[::-1]
        print("decoder layers:", self.decode_layer_dims)
        self.decoder = MLPLayers(layers=self.decode_layer_dims,
                                       dropout=self.dropout_prob,bn=self.bn)
        print("In MoE_rqvae_no_cf, num of experts:", self.num_experts)

    def forward(self, x, labels, use_sk=True):
        x = self.encoder(x)
        x_q, rq_loss, indices = self.rq(x,labels, use_sk=use_sk)
        out = self.decoder(x_q)

        return out, rq_loss, indices, x_q
    
    def CF_loss(self, quantized_rep, encoded_rep):
        batch_size = quantized_rep.size(0)
        labels = torch.arange(batch_size, dtype=torch.long, device=quantized_rep.device)
        similarities = torch.matmul(quantized_rep, encoded_rep.transpose(0, 1))
        cf_loss = F.cross_entropy(similarities, labels)
        return cf_loss
    
    def vq_initialization(self,x, data_id, num_experts, use_sk=True):
        print("data size in MoE rqvae：", x.size())
        self.rq.vq_ini(self.encoder(x), data_id, num_experts)

    @torch.no_grad()
    def get_indices(self, xs, labels, use_sk=False):
        x_e = self.encoder(xs)
        _, _, indices = self.rq(x_e, labels, use_sk=use_sk)
        return indices

    def compute_loss(self, out, quant_loss, emb_idx, dense_out, xs=None):

        if self.loss_type == 'mse':
            loss_recon = F.mse_loss(out, xs, reduction='mean')
        elif self.loss_type == 'l1':
            loss_recon = F.l1_loss(out, xs, reduction='mean')
        else:
            raise ValueError('incompatible loss type')

        rqvae_n_diversity_loss = loss_recon + self.quant_loss_weight * quant_loss

        # CF_Loss
        # cf_embedding_in_batch = self.cf_embedding[emb_idx]
        # cf_embedding_in_batch = torch.from_numpy(cf_embedding_in_batch).to(dense_out.device)
        # cf_loss = self.CF_loss(dense_out, cf_embedding_in_batch)
        cf_loss = 0

        total_loss = rqvae_n_diversity_loss + self.alpha * 0*cf_loss

        return total_loss, cf_loss, loss_recon, quant_loss
    

    # ######## HSIC loss  ########
    def rbf_kernel(self, x, sigma=1.0):
        if x.dim() == 1:
            x = x.unsqueeze(1)
        x_norm = (x ** 2).sum(dim=1).unsqueeze(1)
        dist = x_norm + x_norm.t() - 2 * x @ x.t()
        return torch.exp(-dist / (2 * sigma ** 2))


    # def rbf_kernel(self, x, sigma=1.0):
    #     x_norm = (x ** 2).sum(dim=1).unsqueeze(1)
    #     dist = x_norm + x_norm.t() - 2 * x @ x.t()
    #     return torch.exp(-dist / (2 * sigma ** 2))

    def center_kernel(self, K):
        n = K.size(0)
        H = torch.eye(n, device=K.device) - torch.ones(n, n, device=K.device) / n
        return H @ K @ H

    def hsic(self, x, y, sigma=1.0):
        #print("x shape:", x.shape)  # should be [B, 32]
        #print("y shape:", y.shape)  # should be [B, 768]
        x = x.unsqueeze(0)
        y = y.unsqueeze(0)
        K = self.rbf_kernel(x, sigma)
        L = self.rbf_kernel(y, sigma)
        Kc = self.center_kernel(K)
        Lc = self.center_kernel(L)
        n = x.size(0)
        hsic_score = torch.trace(Kc @ Lc) / ((n - 1) ** 2)

        if torch.isnan(hsic_score):
            return torch.tensor(0.0, device=x.device)
        
        return hsic_score

    def get_HSIC(self, x):
        x_e = self.encoder(x)

        return self.hsic(x, x_e)

    def mi_calibration_loss(self, X_list, Z_list, alpha=1.0, beta=1.0, sigma=1.0):
        hsic_vals = [self.hsic(X, Z, sigma) for X, Z in zip(X_list, Z_list)]
        #hsic_vals = self.hsic(X_list, Z_list, sigma)
        hsic_vals = torch.tensor(hsic_vals)
        print("shape:", hsic_vals.shape)
        #hsic_tensor = torch.stack(hsic_vals)
        #hsic_tensor = torch.nan_to_num(hsic_tensor, nan=0.0)
        hsic_tensor = hsic_vals
        
        mean_hsic = hsic_tensor.mean()
        var_hsic = hsic_tensor.var(unbiased=True)
        loss = alpha * var_hsic - beta * mean_hsic
        #loss = torch.nan_to_num(loss, nan=0.0)
        return loss, hsic_vals, mean_hsic, var_hsic
