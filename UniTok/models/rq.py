import torch
import torch.nn as nn

from .vq import VectorQuantizer
from .MoE import MoEGate


class ResidualVectorQuantizer(nn.Module):

    def __init__(self, n_e_list, e_dim, sk_epsilons, beta = 1,
                 kmeans_init = False, kmeans_iters = 100, sk_iters=100, num_experts=10):
        super().__init__()
        self.n_e_list = n_e_list
        self.e_dim = e_dim
        self.num_quantizers = len(n_e_list)
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_epsilons = sk_epsilons
        self.sk_iters = sk_iters
        self.vq_layers = nn.ModuleList([VectorQuantizer(n_e, e_dim, beta=beta,
                                                        kmeans_init = self.kmeans_init,
                                                        kmeans_iters = self.kmeans_iters,
                                                        sk_epsilon=sk_epsilon,
                                                        sk_iters=sk_iters, num_experts=num_experts)
                                        for n_e, sk_epsilon in zip(n_e_list,sk_epsilons) ])
        self.MoEGate = MoEGate(e_dim, num_experts)
        self.num_experts = num_experts
        print("In rq, num of experts:", self.num_experts)

    def get_codebook(self):
        all_codebook = []
        for quantizer in self.vq_layers:
            codebook = quantizer.get_codebook()
            all_codebook.append(codebook)
        return torch.stack(all_codebook)
    
    def vq_ini(self, x, data_id, num_experts):
        x_q = 0
        residual = x
        for idx, quantizer in enumerate(self.vq_layers):

            x_res = quantizer.vq_init(residual, data_id, num_experts, use_sk=True)
            print("level idx initialized:", idx)
            residual = residual - x_res
            x_q = x_q + x_res

    def forward(self, x, labels, use_sk=True):       
        all_losses = []
        all_indices = []

        x_q = 0    # hidden embedding from encoder
        residual = x
        gate_probs = self.MoEGate(residual)

        for idx, quantizer in enumerate(self.vq_layers):
            label = labels[str(idx)]
            
            x_res, loss, indices = quantizer(residual, label, idx, gate_probs, use_sk=use_sk)
            #print("indices shape:", indices.size())
            residual = residual - x_res
            x_q = x_q + x_res    # sum of all level of codebook

            all_losses.append(loss)
            all_indices.append(indices)

        expert_id = torch.argmax(gate_probs, dim=-1)
        all_indices.append(expert_id)
        #print("all_indices:", all_indices)
        
        #print("all_indices size in rq (before stack):", len(all_indices), len(all_indices[0]))
        mean_losses = torch.stack(all_losses).mean() 
        all_indices = torch.stack(all_indices, dim=-1)     # all_indices: [4, 64] ==> [64, 4]
        #print("all_indices size in rq:", all_indices.size())

        return x_q, mean_losses, all_indices
    