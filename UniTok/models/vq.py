import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import kmeans, sinkhorn_algorithm
import random
import wandb


class VectorQuantizer(nn.Module):

    def __init__(self, n_e, e_dim, mu = 0.25,
                 beta = 1, kmeans_init = False, kmeans_iters = 10,
                 sk_epsilon=0.01, sk_iters=100, num_experts=10, n_e1=12101, n_e2=9922, n_e3=20033):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.mu = mu
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_epsilon = sk_epsilon
        self.sk_iters = sk_iters
        self.num_experts = num_experts

        # self.n_e1 = n_e1
        # self.n_e2 = n_e2
        # self.n_e3 = n_e3

        # self.embedding = nn.Embedding(self.n_e, self.e_dim)
        # print("self.embedding size in vq_init:", self.embedding.weight.data.size(), "n_e:", self.n_e, "e_dim:", self.e_dim)
        # self.embedding2 = nn.Embedding(self.n_e, self.e_dim)
        # self.embedding3 = nn.Embedding(self.n_e, self.e_dim)
        # if not kmeans_init:
        #     self.initted = True
        #     self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

        #     self.embedding2.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
        #     self.initted = True
        #     self.embedding3.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
        # else:
        #     self.initted = False
        #     self.embedding.weight.data.zero_()
        #     self.initted2 = False
        #     self.embedding2.weight.data.zero_()
        #     self.initted3 = False
        #     self.embedding3.weight.data.zero_()

        self.embeddings = nn.ModuleList([
            nn.Embedding(self.n_e, self.e_dim) for _ in range(self.num_experts+1)
        ])

        for idx, embedding in enumerate(self.embeddings, 0):
            print(f"self.embedding{idx} size in vq_init:", embedding.weight.data.size(), "n_e:", self.n_e, "e_dim:", self.e_dim)

        if not kmeans_init:
            for idx, embedding in enumerate(self.embeddings, 0):
                setattr(self, f"initted{idx}", True)
                embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
        else:
            for idx, embedding in enumerate(self.embeddings, 0):
                setattr(self, f"initted{idx}", False)
                embedding.weight.data.zero_()


    def get_codebook(self):
        return self.embedding.weight

    def get_codebook_entry(self, indices, shape=None):
        # get quantized latent vectors
        z_q = self.embedding(indices)
        if shape is not None:
            z_q = z_q.view(shape)

        return z_q

    def init_emb(self, data, data_id):

        # centers = kmeans(
        #     data,
        #     self.n_e,
        #     self.kmeans_iters,
        # )

        if data_id == 10:
            centers, _ = self.constrained_km2(data, 256)
            print("data_id", data_id, "center size in vq：", centers.size())
        else:
            print("data size in vq init_emb：", data.size())
            centers, _ = self.constrained_km(data, 256)
            print("center size in vq：", centers.size())
        # print("self embedding size in vq:", self.embedding.weight.data.size())
        # if data_id == 0:
        #     self.embedding.weight.data.copy_(centers)
        #     self.initted = True
        #     print("data 0: initialized!")
        # elif data_id == 1:
        #     self.embedding2.weight.data.copy_(centers)
        #     self.initted2 = True
        #     print("data 1: initialized!")
        # elif data_id == 2:
        #     self.embedding3.weight.data.copy_(centers)
        #     self.initted3 = True
        #     print("data 2: initialized!")

        self.embeddings[data_id].weight.data.copy_(centers)
        setattr(self, f"initted{data_id}", True)
        print(f"data {data_id}: initialized with kmeans!")
        
    
    def constrained_km(self, data, n_clusters=10):
        from k_means_constrained import KMeansConstrained 
        x = data.cpu().detach().numpy()

        size_min = min(len(data) // (n_clusters * 2), 50) # 50 for the very first time, 10 the latter

        clf = KMeansConstrained(n_clusters=n_clusters, size_min=size_min, size_max=size_min * 4, max_iter=10, n_init=10,
                                n_jobs=10, verbose=False) # 'size_min * 4' for the very first time, 'n_clusters * 4' for the latter
        clf.fit(x)
        t_centers = torch.from_numpy(clf.cluster_centers_)
        t_labels = torch.from_numpy(clf.labels_).tolist()
        value_counts = {}
        return t_centers, t_labels
    
    def constrained_km2(self, data, n_clusters=10):
        from k_means_constrained import KMeansConstrained 
        x = data.cpu().detach().numpy()

        size_min = min(len(data) // (n_clusters * 2), 50) # 50 for the very first time, 10 the latter

        clf = KMeansConstrained(n_clusters=n_clusters, size_min=size_min, size_max=size_min * 20, max_iter=10, n_init=10,
                                n_jobs=10, verbose=False) # 'size_min * 4' for the very first time, 'n_clusters * 4' for the latter
        clf.fit(x)
        t_centers = torch.from_numpy(clf.cluster_centers_)
        t_labels = torch.from_numpy(clf.labels_).tolist()
        value_counts = {}
        return t_centers, t_labels


    def diversity_loss(self, x_q, indices, indices_cluster, indices_list):
        emb = self.embeddings[0].weight
        temp = 1

        pos_list = [indices_list[i] for i in indices_cluster]
        pos_sample = []
        for idx, pos in enumerate(pos_list):
            random_element = random.choice(pos)

            while random_element == indices[idx]:
                random_element = random.choice(pos)
            pos_sample.append(random_element)

        y_true = torch.tensor(pos_sample, device=x_q.device)

        # sim = F.cosine_similarity(x_q, emb, dim=-1)
        sim = torch.matmul(x_q, emb.t())

        # sampled_ids = torch.multinomial(best_scores, num_samples=1)
        sim_self = torch.zeros_like(sim)
        for idx, row in enumerate(sim_self):
            sim_self[idx, indices[idx]] = 1e12
        sim = sim - sim_self
        sim = sim / temp
        loss = F.cross_entropy(sim, y_true)

        return loss

    def diversity_loss_main_entry(self, x, x_q, indices, labels):

        indices_cluster = [labels[idx.item()] for idx in indices]
        target_numbers = list(range(10)) 
        indices_list = {}
        for target_number in target_numbers:
            indices_list[target_number] = [index for index, num in enumerate(labels) if num == target_number]

        diversity_loss = self.diversity_loss(x_q, indices, indices_cluster, indices_list)

        return diversity_loss
                    
    
    @staticmethod
    def center_distance_for_constraint(distances):
        # distances: B, K
        max_distance = distances.max()
        min_distance = distances.min()

        middle = (max_distance + min_distance) / 2
        amplitude = max_distance - middle + 1e-5
        assert amplitude > 0
        centered_distances = (distances - middle) / amplitude
        return centered_distances
    
    def vq_init(self, x, data_id, num_experts, use_sk=True):
        latent = x.view(-1, self.e_dim)
        print("latent size in vq_init:", latent.size())
        if not getattr(self, f"initted{data_id}"):
            self.init_emb(latent, data_id)

        # if not self.initted and data_id == 0:
        #     self.init_emb(latent, 0)
        # if not self.initted2 and data_id == 1:
        #     self.init_emb(latent, 1)
        # if not self.initted3 and data_id == 2:
        #     self.init_emb(latent, 2)

        _distance_flag = 'distance'    
        
        # if _distance_flag == 'distance':
        #     if data_id == 0:
        #         d = torch.sum(latent**2, dim=1, keepdim=True) + \
        #             torch.sum(self.embedding.weight**2, dim=1, keepdim=True).t()- \
        #             2 * torch.matmul(latent, self.embedding.weight.t())
        #     elif data_id == 1: 
        #         d = torch.sum(latent**2, dim=1, keepdim=True) + \
        #             torch.sum(self.embedding2.weight**2, dim=1, keepdim=True).t()- \
        #             2 * torch.matmul(latent, self.embedding2.weight.t())
        #     elif data_id == 2:
        #         d = torch.sum(latent**2, dim=1, keepdim=True) + \
        #             torch.sum(self.embedding3.weight**2, dim=1, keepdim=True).t()- \
        #             2 * torch.matmul(latent, self.embedding3.weight.t())
                
        if _distance_flag == 'distance':
            embedding_weight = self.embeddings[data_id].weight
            d = torch.sum(latent**2, dim=1, keepdim=True) + \
                torch.sum(embedding_weight**2, dim=1, keepdim=True).t() - \
                2 * torch.matmul(latent, embedding_weight.t())        
        
        else:    
        # Calculate Cosine Similarity 
            d = latent@self.embedding.weight.t()


        if not use_sk or self.sk_epsilon <= 0:
            if _distance_flag == 'distance':
                indices = torch.argmin(d, dim=-1)
            else:    
                indices = torch.argmax(d, dim=-1)
        else:
            d = self.center_distance_for_constraint(d)
            d = d.double()

            Q = sinkhorn_algorithm(d,self.sk_epsilon,self.sk_iters)
            if torch.isnan(Q).any() or torch.isinf(Q).any():
                print(f"Sinkhorn Algorithm returns nan/inf values.")
            indices = torch.argmax(Q, dim=-1)

        # if data_id == 0:
        #     x_q = self.embedding(indices).view(x.shape)
        # elif data_id == 1:
        #     x_q = self.embedding2(indices).view(x.shape)
        # elif data_id == 2:
        #     x_q = self.embedding3(indices).view(x.shape)
        x_q = self.embeddings[data_id](indices).view(x.shape)

        return x_q
    
    # def forward(self,  x, label, idx, gate_probs, use_sk=True):
    #     # Flatten input
    #     latent = x.view(-1, self.e_dim)

    #     if not self.initted and self.training:
    #         self.init_emb(latent)

    #     # Calculate the L2 Norm between latent and Embedded weights
    #     _distance_flag = 'distance'    
        
    #     if _distance_flag == 'distance':
    #         d1 = torch.sum(latent**2, dim=1, keepdim=True) + \
    #             torch.sum(self.embedding.weight**2, dim=1, keepdim=True).t()- \
    #             2 * torch.matmul(latent, self.embedding.weight.t())

    #         d2 = torch.sum(latent**2, dim=1, keepdim=True) + \
    #             torch.sum(self.embedding2.weight**2, dim=1, keepdim=True).t()- \
    #             2 * torch.matmul(latent, self.embedding2.weight.t())

    #         d3 = torch.sum(latent**2, dim=1, keepdim=True) + \
    #             torch.sum(self.embedding3.weight**2, dim=1, keepdim=True).t()- \
    #             2 * torch.matmul(latent, self.embedding3.weight.t()) 
    #     else:    
    #     # Calculate Cosine Similarity 
    #         d = latent@self.embedding.weight.t()
    #     if not use_sk or self.sk_epsilon <= 0:
    #         #print("use_sk:", use_sk, "self.sk_epsilon:", self.sk_epsilon)
    #         if _distance_flag == 'distance':
    #             if idx != -1:
    #                 indices_1 = torch.argmin(d1, dim=-1)
    #                 indices_2 = torch.argmin(d2, dim=-1)
    #                 indices_3 = torch.argmin(d3, dim=-1)
    #                 #print("indices finished!")
    #             else:
    #                 temp = 1.0
    #                 prob_dist = F.softmax(-d/temp, dim=1)  
    #                 indices = torch.multinomial(prob_dist, 1).squeeze()
    #         else:    
    #             indices = torch.argmax(d, dim=-1)
    #     else:
    #         d1 = self.center_distance_for_constraint(d1)
    #         d1 = d1.double()

    #         Q1 = sinkhorn_algorithm(d1,self.sk_epsilon,self.sk_iters)
    #         # print(Q.sum(0)[:10])
    #         if torch.isnan(Q1).any() or torch.isinf(Q1).any():
    #             print(f"Sinkhorn Algorithm returns nan/inf values.")
    #         indices_1 = torch.argmax(Q1, dim=-1)

    #         d2 = self.center_distance_for_constraint(d2)
    #         d2 = d2.double()

    #         Q2 = sinkhorn_algorithm(d2,self.sk_epsilon,self.sk_iters)
    #         # print(Q.sum(0)[:10])
    #         if torch.isnan(Q2).any() or torch.isinf(Q2).any():
    #             print(f"Sinkhorn Algorithm returns nan/inf values.")
    #         indices_2 = torch.argmax(Q2, dim=-1)

    #         d3 = self.center_distance_for_constraint(d3)
    #         d3 = d3.double()

    #         Q3 = sinkhorn_algorithm(d3,self.sk_epsilon,self.sk_iters)
    #         # print(Q.sum(0)[:10])
    #         if torch.isnan(Q3).any() or torch.isinf(Q3).any():
    #             print(f"Sinkhorn Algorithm returns nan/inf values.")
    #         indices_3 = torch.argmax(Q3, dim=-1)

    #     # indices = torch.argmin(d, dim=-1)

    #     #x_q = self.embedding(indices).view(x.shape)

    #     x_q1 = self.embedding(indices_1).view(x.shape)
    #     x_q2 = self.embedding2(indices_2).view(x.shape)
    #     x_q3 = self.embedding3(indices_3).view(x.shape)
        
    #     #print("indices1 size:", indices_1.size(), "indices2 size:", indices_2.size(), "indices3 size:", indices_3.size())
    #     #print("gate_probs size:", gate_probs.size())
    #     #print("x_q1 size:", x_q1.size(), "x_q2 size:", x_q2.size(), "x_q3:", x_q3.size())

    #     #x_q = x_q1 * gate_probs[0] + x_q2 * gate_probs[1] + x_q3 * gate_probs[2]
    #     x_q = gate_probs[:, 0].unsqueeze(1) * x_q1 + \
    #             gate_probs[:, 1].unsqueeze(1) * x_q2 + \
    #             gate_probs[:, 2].unsqueeze(1) * x_q3
        
    #     # Diversity
    #     diversity_loss = self.diversity_loss_main_entry(x, x_q, indices_1, label)
    #     # wandb.log({'diversity_loss': diversity_loss})

    #     # compute loss for embedding
    #     commitment_loss = F.mse_loss(x_q.detach(), x)
    #     codebook_loss = F.mse_loss(x_q, x.detach())

    #     loss = codebook_loss + self.mu * commitment_loss + self.beta * diversity_loss


    #     batch_size = gate_probs.shape[0]
    #     # Combine into a list
    #     all_indices = torch.stack([indices_1, indices_2, indices_3], dim=1)  # Shape: (1024, 3)

    #     # Get the index of the maximum value in each row
    #     expert_id = torch.argmax(gate_probs, dim=1)  # Shape: (1024,)
    #     #print("expert_id size:", expert_id.size())
    #     # Select values based on max indices
    #     indices = all_indices[torch.arange(batch_size), expert_id]  # Shape: (1024,)
    #     #print("indices output size:", indices.size())
    #     # preserve gradients
    #     x_q = x + (x_q - x).detach()

    #     indices = indices.view(x.shape[:-1])

    #     return x_q, loss, indices
    
    
    # def forward(self,  x, label, idx, gate_probs, use_sk=True):
    #     # Flatten input
    #     latent = x.view(-1, self.e_dim)

    #     # if not self.initted and self.training:
    #     #     self.init_emb(latent)

    #     for data_id in range(self.num_experts):
    #         if not getattr(self, f"initted{data_id}") and self.training:
    #             self.init_emb(latent, data_id)

    #     # Calculate the L2 Norm between latent and Embedded weights
    #     _distance_flag = 'distance'    

    #     x_q = 0
    #     all_xq_id = []
    #     indices_list = []
    #     for i, embedding in enumerate(self.embeddings):
    #         if _distance_flag == 'distance':
    #             embedding_weight = embedding.weight
    #             d = torch.sum(latent**2, dim=1, keepdim=True) + \
    #                 torch.sum(embedding_weight**2, dim=1, keepdim=True).t() - \
    #                 2 * torch.matmul(latent, embedding_weight.t())  
                 
    #             if not use_sk or self.sk_epsilon <= 0:
    #                 if idx != -1:
    #                     indices = torch.argmin(d, dim=-1)
    #                 else:
    #                     temp = 1.0
    #                     prob_dist = F.softmax(-d/temp, dim=1)  
    #                     indices = torch.multinomial(prob_dist, 1).squeeze()
    #             else:    
    #                 d = self.center_distance_for_constraint(d)
    #                 d = d.double()

    #                 Q = sinkhorn_algorithm(d,self.sk_epsilon,self.sk_iters)
    #                 # print(Q.sum(0)[:10])
    #                 if torch.isnan(Q).any() or torch.isinf(Q).any():
    #                     print(f"Sinkhorn Algorithm returns nan/inf values.")
    #                 indices = torch.argmax(Q, dim=-1)

    #         xq_id = embedding(indices).view(x.shape)
    #         all_xq_id.append(xq_id)
    #         indices_list.append(indices)

    #     for i in range(self.num_experts):
    #         x_q += gate_probs[:, i].unsqueeze(1) * all_xq_id[i]

    #     #print("x_q size:", x_q.size())

    #     # Diversity
    #     diversity_loss = self.diversity_loss_main_entry(x, x_q, indices_list[0], label)
    #     # wandb.log({'diversity_loss': diversity_loss})

    #     # compute loss for embedding
    #     commitment_loss = F.mse_loss(x_q.detach(), x)
    #     codebook_loss = F.mse_loss(x_q, x.detach())

    #     loss = codebook_loss + self.mu * commitment_loss + self.beta * diversity_loss

    #     batch_size = gate_probs.shape[0]
    #     # Combine into a list
    #     all_indices = torch.stack(indices_list, dim=1)  # Shape: (1024, 3)

    #     # Get the index of the maximum value in each row
    #     expert_id = torch.argmax(gate_probs, dim=1)  # Shape: (1024,)
    #     #print("expert_id size:", expert_id.size())
    #     # Select values based on max indices
    #     indices = all_indices[torch.arange(batch_size), expert_id]  # Shape: (1024,)
    #     #print("indices output size:", indices.size())
    #     # preserve gradients
    #     x_q = x + (x_q - x).detach()

    #     indices = indices.view(x.shape[:-1])

    #     return x_q, loss, indices
    


    # Seperate shared expert
    def forward(self,  x, label, idx, gate_probs, use_sk=True):
        # Flatten input
        latent = x.view(-1, self.e_dim)

        # if not self.initted and self.training:
        #     self.init_emb(latent)

        for data_id in range(self.num_experts+1):
            if not getattr(self, f"initted{data_id}") and self.training:
                self.init_emb(latent, data_id)

        # Calculate the L2 Norm between latent and Embedded weights
        _distance_flag = 'distance'    

        x_q = 0
        all_xq_id = []
        indices_list = []
        for i, embedding in enumerate(self.embeddings):
            if _distance_flag == 'distance':
                embedding_weight = embedding.weight
                d = torch.sum(latent**2, dim=1, keepdim=True) + \
                    torch.sum(embedding_weight**2, dim=1, keepdim=True).t() - \
                    2 * torch.matmul(latent, embedding_weight.t())  
                 
                if not use_sk or self.sk_epsilon <= 0:
                    if idx != -1:
                        indices = torch.argmin(d, dim=-1)
                    else:
                        temp = 1.0
                        prob_dist = F.softmax(-d/temp, dim=1)  
                        indices = torch.multinomial(prob_dist, 1).squeeze()
                else:    
                    d = self.center_distance_for_constraint(d)
                    d = d.double()

                    Q = sinkhorn_algorithm(d,self.sk_epsilon,self.sk_iters)
                    # print(Q.sum(0)[:10])
                    if torch.isnan(Q).any() or torch.isinf(Q).any():
                        print(f"Sinkhorn Algorithm returns nan/inf values.")
                    indices = torch.argmax(Q, dim=-1)

            xq_id = embedding(indices).view(x.shape)
            all_xq_id.append(xq_id)
            indices_list.append(indices)

        for i in range(self.num_experts):
            x_q += gate_probs[:, i].unsqueeze(1) * all_xq_id[i]

        # Shared expertss
        x_q = 0.95 * x_q + 0.05 * all_xq_id[10]
        #print("x_q size:", x_q.size())

        # Diversity
        diversity_loss = self.diversity_loss_main_entry(x, x_q, indices_list[0], label)
        # wandb.log({'diversity_loss': diversity_loss})

        # compute loss for embedding
        commitment_loss = F.mse_loss(x_q.detach(), x)
        codebook_loss = F.mse_loss(x_q, x.detach())

        loss = codebook_loss + self.mu * commitment_loss + self.beta * diversity_loss

        batch_size = gate_probs.shape[0]
        # Combine into a list
        all_indices = torch.stack(indices_list, dim=1)  # Shape: (1024, 3)

        # Get the index of the maximum value in each row
        expert_id = torch.argmax(gate_probs, dim=1)  # Shape: (1024,)
        #print("expert_id size:", expert_id.size())
        # Select values based on max indices
        indices = all_indices[torch.arange(batch_size), expert_id]  # Shape: (1024,)
        #print("indices output size:", indices.size())
        # preserve gradients
        x_q = x + (x_q - x).detach()

        indices = indices.view(x.shape[:-1])

        return x_q, loss, indices



    # def forward(self,  x, label, idx, gate_probs, use_sk=True):
    #     # Flatten input
    #     latent = x.view(-1, self.e_dim)

    #     # if not self.initted and self.training:
    #     #     self.init_emb(latent)

    #     for data_id in range(self.num_experts):
    #         if not getattr(self, f"initted{data_id}") and self.training:
    #             self.init_emb(latent, data_id)

    #     # Calculate the L2 Norm between latent and Embedded weights
    #     _distance_flag = 'distance'    

    #     x_q = 0
    #     all_xq_id = []
    #     indices_list = []

    #     # topk = 2
    #     # print("gate_probs shape:", gate_probs.shape)
    #     # topk_values, topk_indices = torch.topk(gate_probs, topk, dim=1)
    #     # print("topk indices shape:", topk_indices.shape)
    #     # print(topk_indices[0])
    #     # print(topk_indices[1])
    #     for i, embedding in enumerate(self.embeddings):
    #     #for i in topk_indices:
    #         embedding = self.embeddings[i]
    #         if _distance_flag == 'distance':
    #             embedding_weight = embedding.weight
    #             d = torch.sum(latent**2, dim=1, keepdim=True) + \
    #                 torch.sum(embedding_weight**2, dim=1, keepdim=True).t() - \
    #                 2 * torch.matmul(latent, embedding_weight.t())  
                 
    #             if not use_sk or self.sk_epsilon <= 0:
    #                 if idx != -1:
    #                     indices = torch.argmin(d, dim=-1)
    #                 else:
    #                     temp = 1.0
    #                     prob_dist = F.softmax(-d/temp, dim=1)  
    #                     indices = torch.multinomial(prob_dist, 1).squeeze()
    #             else:    
    #                 d = self.center_distance_for_constraint(d)
    #                 d = d.double()

    #                 Q = sinkhorn_algorithm(d,self.sk_epsilon,self.sk_iters)
    #                 # print(Q.sum(0)[:10])
    #                 if torch.isnan(Q).any() or torch.isinf(Q).any():
    #                     print(f"Sinkhorn Algorithm returns nan/inf values.")
    #                 indices = torch.argmax(Q, dim=-1)

    #         xq_id = embedding(indices).view(x.shape)
    #         all_xq_id.append(xq_id) # [num_experts, batch_size, feature_dim]
    #         indices_list.append(indices)

    #     # for i in range(self.num_experts):
    #     # #for i, e_id in enumerate(topk_indices):
    #     #     x_q += gate_probs[:, i].unsqueeze(1) * all_xq_id[i]
                
    #     ################### MoE version 1
    #     # topK = 1  # or whatever top-k you want

    #     # # gate_probs: [batch_size, num_experts]
    #     # #print("gate_probs shape:", gate_probs.shape)
    #     # topk_probs, topk_indices = torch.topk(gate_probs, topK, dim=1)  # [batch_size, K]
    #     # #print("topk indices shape:", topk_indices.shape)

    #     # # Convert all_xq_id to a tensor if it's a list
    #     # # all_xq_id: [num_experts, batch_size, feature_dim]
    #     # if isinstance(all_xq_id, list):
    #     #     all_xq_id = torch.stack(all_xq_id, dim=0)

    #     # # Transpose to [batch_size, num_experts, feature_dim] for indexing
    #     # all_xq_id = all_xq_id.permute(1, 0, 2)

    #     # # Gather top-K expert outputs: [batch_size, K, feature_dim]
    #     # topk_xq = torch.gather(
    #     #     all_xq_id,
    #     #     dim=1,
    #     #     index=topk_indices.unsqueeze(-1).expand(-1, -1, all_xq_id.size(-1))
    #     # )
    #     # #print("topk xq shape:", topk_xq.shape)
    #     # # Weight each expert’s output by its gate probability
    #     # x_q = (topk_probs.unsqueeze(-1) * topk_xq).sum(dim=1)  # [batch_size, feature_dim]
    #     # #print("topk xq shape:", topk_xq.shape)
    #     # #print("x_q size:", x_q.size())

    #     ########### version 2
    #     topK = 1

    #     topk_probs, topk_indices = torch.topk(gate_probs, topK, dim=1)  # [num_experts, 1]

    #     # Flatten for easier handling
    #     topk_probs = topk_probs.squeeze(1)    # [num_experts]
    #     topk_indices = topk_indices.squeeze(1)  # [num_experts]

    #     feature_dim = all_xq_id[0].size(-1)
    #     batch_size = all_xq_id[0].size(0)
    #     x_q = torch.zeros(batch_size, feature_dim, device=gate_probs.device)

    #     for expert_id in range(self.num_experts):
    #         mask = (topk_indices == expert_id)  # shape: [batch_size]

    #         if mask.any():
    #             expert_output = all_xq_id[expert_id][mask]         # [N_i, feature_dim]
    #             gate_weight = topk_probs[mask].unsqueeze(1)        # [N_i, 1]
    #             weighted_output = expert_output * gate_weight      # [N_i, feature_dim]
    #             x_q[mask] = weighted_output


    #     # Diversity
    #     diversity_loss = self.diversity_loss_main_entry(x, x_q, indices_list[0], label)
    #     # wandb.log({'diversity_loss': diversity_loss})

    #     # compute loss for embedding
    #     commitment_loss = F.mse_loss(x_q.detach(), x)
    #     codebook_loss = F.mse_loss(x_q, x.detach())

    #     loss = codebook_loss + self.mu * commitment_loss + self.beta * diversity_loss

    #     batch_size = gate_probs.shape[0]
    #     # Combine into a list
    #     all_indices = torch.stack(indices_list, dim=1)  # Shape: (1024, 3)

    #     # Get the index of the maximum value in each row
    #     expert_id = torch.argmax(gate_probs, dim=1)  # Shape: (1024,)
    #     #print("expert_id size:", expert_id.size())
    #     # Select values based on max indices
    #     indices = all_indices[torch.arange(batch_size), expert_id]  # Shape: (1024,)
    #     #print("indices output size:", indices.size())
    #     # preserve gradients
    #     x_q = x + (x_q - x).detach()

    #     indices = indices.view(x.shape[:-1])

    #     return x_q, loss, indices



    # Top 1 trick
    # def forward(self, x, label, idx, gate_probs, use_sk=True):
    #     # Flatten input
    #     latent = x.view(-1, self.e_dim)

    #     batch_size = gate_probs.size(0)
    #     feature_dim = latent.size(-1)

    #     topk_probs, topk_indices = torch.topk(gate_probs, 1, dim=1)  # [batch_size, 1]
    #     topk_probs = topk_probs.squeeze(1)      # [batch_size]
    #     topk_indices = topk_indices.squeeze(1)  # [batch_size]

    #     x_q = torch.zeros_like(latent)
    #     indices = torch.zeros(batch_size, dtype=torch.long, device=x.device)

    #     # Ensure embeddings are initialized
    #     for expert_id in range(self.num_experts):
    #         if not getattr(self, f"initted{expert_id}") and self.training:
    #             self.init_emb(latent, expert_id)

    #     indices_list = [torch.zeros(batch_size, dtype=torch.long, device=x.device)
    #                     for _ in range(self.num_experts)]

    #     # Process data grouped by assigned expert
    #     for expert_id in range(self.num_experts):
    #         mask = (topk_indices == expert_id)  # [batch_size]
    #         if mask.any():
    #             latent_expert = latent[mask]  # Select only data assigned to current expert
    #             embedding = self.embeddings[expert_id]
    #             embedding_weight = embedding.weight

    #             # Compute distances only for data assigned to this expert
    #             d = torch.sum(latent_expert ** 2, dim=1, keepdim=True) + \
    #                 torch.sum(embedding_weight ** 2, dim=1).unsqueeze(0) - \
    #                 2 * torch.matmul(latent_expert, embedding_weight.t())

    #             if use_sk and self.sk_epsilon > 0:
    #                 d = self.center_distance_for_constraint(d)
    #                 d = d.double()
    #                 Q = sinkhorn_algorithm(d, self.sk_epsilon, self.sk_iters)
    #                 if torch.isnan(Q).any() or torch.isinf(Q).any():
    #                     print("Sinkhorn Algorithm returns nan/inf values.")
    #                 indices_expert = torch.argmax(Q, dim=-1)
    #             else:
    #                 if idx != -1:
    #                     indices_expert = torch.argmin(d, dim=-1)
    #                 else:
    #                     temp = 1.0
    #                     prob_dist = F.softmax(-d / temp, dim=1)
    #                     indices_expert = torch.multinomial(prob_dist, 1).squeeze()

    #             # Store quantized embeddings
    #             quantized = embedding(indices_expert)
    #             x_q[mask] = quantized
    #             indices[mask] = indices_expert
    #             indices_list[expert_id][mask] = indices_expert

    #     # Reshape quantized outputs
    #     x_q = x_q.view(x.shape)

    #     # Diversity loss (considering only selected expert embeddings)
    #     diversity_loss = self.diversity_loss_main_entry(x, x_q, indices, label)

    #     # Compute codebook and commitment loss
    #     commitment_loss = F.mse_loss(x_q.detach(), x)
    #     codebook_loss = F.mse_loss(x_q, x.detach())
    #     loss = codebook_loss + self.mu * commitment_loss + self.beta * diversity_loss

    #     # Preserve gradients (Straight-Through Estimator)
    #     x_q = x + (x_q - x).detach()
    #     indices = indices.view(x.shape[:-1])

    #     return x_q, loss, indices
