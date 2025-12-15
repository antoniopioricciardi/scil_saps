# losses.py
import torch
import torch.nn.functional as F

def supervised_contrastive_loss(features, labels, temperature=0.07):
    """
    Args:
        features: [batch_size, z_dim]
        labels: [batch_size]
    """
    device = features.device
    
    # 1. Normalize features
    features = F.normalize(features, dim=1)
    
    # 2. Similarity Matrix
    similarity_matrix = torch.matmul(features, features.T)
    
    # 3. Create Masks
    labels = labels.contiguous().view(-1, 1)
    mask = torch.eq(labels, labels.T).float().to(device)
    
    # Remove self-contrast (diagonal)
    logits_mask = torch.scatter(
        torch.ones_like(mask), 
        1, 
        torch.arange(features.shape[0]).view(-1, 1).to(device), 
        0
    )
    mask = mask * logits_mask

    # 4. Compute Logits and Log Prob (with numerical stability)
    # Subtract max for numerical stability (log-sum-exp trick)
    logits_max, _ = torch.max(similarity_matrix / temperature, dim=1, keepdim=True)
    logits = similarity_matrix / temperature - logits_max.detach()

    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)
    
    # 5. Compute Mean Loss (Handling missing positives safely)
    positives_count = mask.sum(1)
    valid_anchors_mask = (positives_count > 0).float()
    
    mean_log_prob_pos = (mask * log_prob).sum(1) / (positives_count + 1e-6)
    
    # Average only over valid anchors
    loss = - (mean_log_prob_pos * valid_anchors_mask).sum() / (valid_anchors_mask.sum() + 1e-6)
    
    return loss