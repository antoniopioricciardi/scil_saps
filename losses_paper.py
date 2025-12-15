# losses_paper.py
# Direct implementation from SCIL paper Appendix A (page 5)
# Adapted for discrete actions only

import torch
from torch import nn as nn
from torch.nn import functional as F

class SupConLoss(nn.Module):
    """
    SupCon loss definition from SCIL paper:
    https://arxiv.org/pdf/2509.11880

    Adapted from the paper's implementation for discrete actions
    """
    def __init__(self, device, temperature=0.07, base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.device = device

    def forward(self, features, labels):
        """
        Args:
            features: [batch_size, feature_dim] - embeddings (z)
            labels: [batch_size] - discrete action labels
        """
        if len(features.shape) < 2:
            raise ValueError("'features' needs to be [bsz, ...],"
                             'at least 2 dimensions are required')

        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)

        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')

        # Create mask for positive pairs (same action label)
        mask = torch.eq(labels, labels.T).float().to(self.device)

        # Normalize features
        features = F.normalize(features, p=2, dim=1)

        # Compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature)

        # For numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # Mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(self.device),
            0
        )
        mask = mask * logits_mask

        # Compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        eps = 1e-12
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + eps)

        # Compute mean of log-likelihood over positive pairs
        # Modified to handle edge cases when there is no positive pair
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # Loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(1, batch_size).mean()

        return loss
