import collections
import json
import logging

import numpy as np
import torch
from time import time
from torch import optim
from tqdm import tqdm

from torch.utils.data import DataLoader

from datasets import EmbDataset
from models.rqvae import RQVAE
from models.MoE_rqvae_no_cf import MoE_RQVAE_no_cf
import argparse
import os


def check_collision(all_indices_str):
    tot_item = len(all_indices_str)
    tot_indice = len(set(all_indices_str.tolist()))
    return tot_item==tot_indice

def get_indices_count(all_indices_str):
    indices_count = collections.defaultdict(int)
    for index in all_indices_str:
        indices_count[index] += 1
    return indices_count

def get_collision_item(all_indices_str):
    index2id = {}
    for i, index in enumerate(all_indices_str):
        if index not in index2id:
            index2id[index] = []
        index2id[index].append(i)

    collision_item_groups = []

    for index in index2id:
        if len(index2id[index]) > 1:
            collision_item_groups.append(index2id[index])

    return collision_item_groups

def parse_args():
    parser = argparse.ArgumentParser(description="RQ-VAE")
    parser.add_argument("--dataset", type=str,default="Instruments", help='dataset')
    parser.add_argument("--root_path", type=str,default="checkpoint/", help='root path')
    parser.add_argument('--alpha', type=str, default='1e-1', help='cf loss weight')
    parser.add_argument('--epoch', type=int, default='10000', help='epoch')
    parser.add_argument('--checkpoint', type=str, default='epoch_9999_collision_0.0012_model.pth', help='checkpoint name')
    parser.add_argument('--beta', type=str, default='1e-4', help='div loss weight')


    return parser.parse_args()

args_setting = parse_args()

dataset = args_setting.dataset
print("Dataset:", args_setting.dataset)
#ckpt_path = args_setting.root_path + f'alpha{args_setting.alpha}-beta{args_setting.beta}/'+args_setting.checkpoint
ckpt_path = args_setting.root_path + args_setting.checkpoint
print("ckpt path:", ckpt_path)
output_dir = f"./data/{dataset}/"
output_file = f"{dataset}.index.epoch{args_setting.epoch}.alpha{args_setting.alpha}-beta{args_setting.beta}.json"
output_file = os.path.join(output_dir,output_file)
device = torch.device("cuda:0")

ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'), weights_only=False)
args = ckpt["args"]
state_dict = ckpt["state_dict"]

print("Data path:", args.data_path)
data = EmbDataset(args.data_path)
data = data.subset(0, 12101) # Beauty

model = MoE_RQVAE_no_cf(in_dim=data.dim,
                  num_emb_list=args.num_emb_list,
                  e_dim=args.e_dim,
                  layers=args.layers,
                  dropout_prob=args.dropout_prob,
                  bn=args.bn,
                  loss_type=args.loss_type,
                  quant_loss_weight=args.quant_loss_weight,
                  kmeans_init=args.kmeans_init,
                  kmeans_iters=args.kmeans_iters,
                  sk_epsilons=args.sk_epsilons,
                  sk_iters=args.sk_iters,
                  num_experts = 10,
                  )

model.load_state_dict(state_dict,strict=False)
model = model.to(device)
model.eval()
print(model)


print("#######################################################")

data_loader = DataLoader(data,num_workers=args.num_workers,
                             batch_size=64, shuffle=False,
                             pin_memory=True)

all_indices = []
all_indices_str = []
prefix = ["<a_{}>","<b_{}>","<c_{}>","<d_{}>","<e_{}>","<f_{}>"]

def constrained_km(data, n_clusters=10):
    from k_means_constrained import KMeansConstrained 
    x = data
    size_min = min(len(data) // (n_clusters * 2), 10)
    clf = KMeansConstrained(n_clusters=n_clusters, size_min=size_min, size_max=n_clusters * 6, max_iter=10, n_init=10,
                            n_jobs=10, verbose=False)
    clf.fit(x)
    t_centers = torch.from_numpy(clf.cluster_centers_)
    t_labels = torch.from_numpy(clf.labels_).tolist()
    return t_centers, t_labels

labels = {"0":[],"1":[],"2":[], "3":[]}
embs  = [layer.embeddings[0].weight.cpu().detach().numpy() for layer in model.rq.vq_layers]


for idx, emb in enumerate(embs):
    centers, label = constrained_km(emb)
    labels[str(idx)] = label



######## HSIC loss  ########

def rbf_kernel(x, sigma=1.0):
    x_norm = (x ** 2).sum(dim=1).unsqueeze(1)
    dist = x_norm + x_norm.t() - 2 * x @ x.t()
    return torch.exp(-dist / (2 * sigma ** 2))

def center_kernel(K):
    n = K.size(0)
    H = torch.eye(n, device=K.device) - torch.ones(n, n, device=K.device) / n
    return H @ K @ H

def hsic(x, y, sigma=1.0):
    K = rbf_kernel(x, sigma)
    L = rbf_kernel(y, sigma)
    Kc = center_kernel(K)
    Lc = center_kernel(L)
    n = x.size(0)
    return torch.trace(Kc @ Lc) / ((n - 1) ** 2)

def get_HSIC(x):
    x_e = model.encoder(x)

    return hsic(x, x_e)


total_hsic = 0
i = 0
for d in tqdm(data_loader):
    d, emb_idx = d[0], d[1]
    d = d.to(device)
    
    hsic_value = get_HSIC(d)
    if torch.isnan(hsic_value).any():
        print("NaN detected in x:", hsic_value, emb_idx)
        continue

        
    total_hsic += hsic_value
    i += 1

average_hsic = total_hsic / i

print("Average HSIC:", total_hsic, average_hsic)
