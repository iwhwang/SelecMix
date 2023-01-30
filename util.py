import torch
import numpy as np
import torch.nn as nn

'''Modified from https://github.com/alinlab/LfF and https://github.com/kakaoenterprise/Learning-Debiased-Disentangled'''
class EMA:
    def __init__(self, label, num_classes=None, alpha=0.9):
        # self.label = label.cuda()
        self.label = label
        self.alpha = alpha
        self.parameter = torch.zeros(label.size(0))
        self.updated = torch.zeros(label.size(0))
        self.num_classes = num_classes
        # self.max = torch.zeros(self.num_classes).cuda()
        self.max = torch.zeros(self.num_classes)

    def update(self, data, index, curve=None, iter_range=None, step=None):
        self.parameter = self.parameter.to(data.device)
        self.updated = self.updated.to(data.device)
        index = index.to(data.device)

        if curve is None:
            self.parameter[index] = self.alpha * self.parameter[index] + (1 - self.alpha * self.updated[index]) * data
        else:
            alpha = curve ** -(step / iter_range)
            self.parameter[index] = alpha * self.parameter[index] + (1 - alpha * self.updated[index]) * data
        self.updated[index] = 1

    def max_loss(self, label):
        label_index = torch.where(self.label == label)[0]
        return self.parameter[label_index].max()


def setup_step_num(args, epch):
    total_epoch = 0
    if args.dataset == 'cifar10c': total_epoch = 300
    elif args.dataset == 'bffhq': total_epoch = 200
    elif args.dataset == 'cmnist': total_epoch = 200
    args.num_steps = epch * total_epoch
    args.valid_freq = epch
    args.log_freq = epch

def mixup_naive(batch_data, batch_label):
    alpha = 1.0
    beta = torch.distributions.beta.Beta(alpha, alpha)
    mix_idx = torch.randperm(batch_data.shape[0], device=batch_data.device)
    lam = beta.sample([batch_data.shape[0]]).to(device=batch_data.device)
    lam_expanded = lam.view([-1] + [1]*(batch_data.dim()-1))
    mixed_batch_data = lam_expanded * batch_data + (1. - lam_expanded) * batch_data[mix_idx]
    mix_batch_label = batch_label[mix_idx]
    return mixed_batch_data, mix_batch_label, lam, (1. - lam)

def inter_mask_with_sim_matrix(sim, indexes):
    diff_label = (indexes.t() != indexes)
    inter_mask = diff_label.to(int)
    masked_sim = sim * inter_mask
    mix_idx = torch.argmax(masked_sim, 1, keepdim=True).view(-1)
    return mix_idx

def intra_mask_with_sim_matrix(sim, indexes):
    same_label = (indexes.t() == indexes)
    intra_mask = same_label.to(int)
    masked_sim = sim * intra_mask + torch.ones_like(sim) * (1 - intra_mask)
    mix_idx = torch.argmin(masked_sim, 1, keepdim=True).view(-1)
    return mix_idx

def get_mixup_index_with_similarity_matrix(args, sim, batch_label):
    indexes = batch_label.unsqueeze(0)
    if args.intra: mix_idx = intra_mask_with_sim_matrix(sim, indexes)
    elif args.inter: mix_idx = inter_mask_with_sim_matrix(sim, indexes)
    elif args.rand: 
        a = torch.rand(1)
        if a>0.5: mix_idx = intra_mask_with_sim_matrix(sim, indexes)
        else: mix_idx = inter_mask_with_sim_matrix(sim, indexes)
    will_be_mixed = torch.zeros_like(mix_idx).to(float)
    return mix_idx, will_be_mixed


def mixup_with_similarity_matrix(args, batch_data, batch_label, sim2):
    sim = sim2.clone().detach()
    mix_idx, will_be_mixed = get_mixup_index_with_similarity_matrix(args, sim, batch_label)

    return mixup_with_index(args, batch_data, batch_label, mix_idx, will_be_mixed)

def mixup_with_index(args, batch_data, batch_label, mix_idx, will_be_mixed):
    alpha = 1.0
    beta = torch.distributions.beta.Beta(alpha, alpha)
    lam = beta.sample([batch_data.shape[0]]).to(float).to(device=batch_data.device)
    if args.ours:
        lam = lam * 0.5
    lam_adj = torch.where(lam>will_be_mixed, lam, will_be_mixed)
    lam_expanded = lam_adj.view([-1] + [1]*(batch_data.dim()-1))
    mixed_batch_data = lam_expanded * batch_data + (1. - lam_expanded) * batch_data[mix_idx]
    mix_batch_label = batch_label[mix_idx].clone().detach()
    lam_adj_b = 1. - lam_adj.clone().detach()
    lam_adj = torch.where(lam>will_be_mixed, lam, will_be_mixed)
    if args.ours:
        lam_adj = torch.where(lam>will_be_mixed, torch.zeros_like(will_be_mixed).to(float), will_be_mixed)
        lam_adj_b = 1. - lam_adj.clone().detach()
    return mixed_batch_data.to(torch.float32), mix_batch_label, lam_adj.to(torch.float32), lam_adj_b.to(torch.float32)

def get_mixup_index_with_gt_bias_label(args, batch_attr):
    indexes = batch_attr[:,args.target_attr_idx].unsqueeze(0)
    indexes_bias = batch_attr[:,args.bias_attr_idx].unsqueeze(0)
    if args.intra: mask = intra_mask_with_gt_bias_label(indexes, indexes_bias)
    elif args.inter: mask = inter_mask_with_gt_bias_label(indexes, indexes_bias)
    elif args.rand: 
        a = torch.rand(1)
        if a>0.5: mask = intra_mask_with_gt_bias_label(indexes, indexes_bias)
        else: mask = inter_mask_with_gt_bias_label(indexes, indexes_bias)
    mask = mask.clone().detach()
    d = mask.sum(1)
    will_be_mixed = (d==0).to(float)
    idx = torch.arange(mask.shape[1], 0, -1, device=batch_attr.device)
    mix_idx = torch.argmax(mask * idx, 1, keepdim=True).view(-1)
    return mix_idx, will_be_mixed

def mixup_with_gt_bias_label(args, batch_data, batch_attr, batch_label):
    mix_idx, will_be_mixed = get_mixup_index_with_gt_bias_label(args, batch_attr)
    return mixup_with_index(args, batch_data, batch_label, mix_idx, will_be_mixed)

def inter_mask_with_gt_bias_label(indexes, indexes_bias):
    diff_label = (indexes.t() != indexes)
    same_bias = (indexes_bias.t() == indexes_bias)
    inter_mask = (diff_label * same_bias).to(int)
    return inter_mask

def intra_mask_with_gt_bias_label(indexes, indexes_bias):
    same_label = (indexes.t() == indexes)
    diff_bias = (indexes_bias.t() != indexes_bias)
    intra_mask = (same_label * diff_bias).to(int)
    return intra_mask
