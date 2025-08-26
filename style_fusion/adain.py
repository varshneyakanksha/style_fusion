# Placeholder AdaIN-style fusion utilities
# Implement your encoder/decoder elsewhere; this file focuses on feature stats fusion

import torch

def calc_mean_std(feat, eps=1e-5):
    # feat: [N, C, H, W]
    size = feat.size()
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

def adaptive_instance_normalization(content_feat, style_feat, eps=1e-5):
    # Matches content features to style feature channel-wise mean/std
    assert content_feat.size()[:2] == style_feat.size()[:2]
    c_mean, c_std = calc_mean_std(content_feat, eps)
    s_mean, s_std = calc_mean_std(style_feat, eps)
    normalized = (content_feat - c_mean) / c_std
    return normalized * s_std + s_mean

def fuse_style_stats(style_feats, weights):
    # style_feats: list of feature maps [N,C,H,W]
    # weights: list of floats summing to 1
    weights = torch.as_tensor(weights, device=style_feats[0].device, dtype=style_feats[0].dtype)
    weights = weights / weights.sum()
    means, stds = [], []
    for sf in style_feats:
        m, s = calc_mean_std(sf)
        means.append(m); stds.append(s)
    mean_stack = torch.stack(means)     # [K, N, C, 1, 1]
    std_stack  = torch.stack(stds)      # [K, N, C, 1, 1]
    # Weighted sum over K
    w = weights.view(-1, 1, 1, 1, 1)
    fused_mean = (w * mean_stack).sum(dim=0)
    fused_std  = (w * std_stack).sum(dim=0)
    return fused_mean, fused_std
