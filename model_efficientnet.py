# model_efficientnet.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class SCILEncoderEfficientNet(nn.Module):
    def __init__(self, num_actions=7, freeze_backbone=False, variant='b0'):
        """
        SCIL Model with pretrained EfficientNet backbone

        Args:
            num_actions: Number of discrete actions
            freeze_backbone: If True, freeze EfficientNet weights (faster training)
            variant: EfficientNet variant ('b0', 'b1', 'b2', etc.)
        """
        super(SCILEncoderEfficientNet, self).__init__()

        # Load pretrained EfficientNet (trained on ImageNet)
        if variant == 'b0':
            efficientnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
            self.repr_dim = 1280  # EfficientNet-B0 output channels
        elif variant == 'b1':
            efficientnet = models.efficientnet_b1(weights=models.EfficientNet_B1_Weights.DEFAULT)
            self.repr_dim = 1280  # EfficientNet-B1 output channels
        elif variant == 'b2':
            efficientnet = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.DEFAULT)
            self.repr_dim = 1408  # EfficientNet-B2 output channels
        else:
            raise ValueError(f"Unsupported EfficientNet variant: {variant}")

        # Remove the final classifier layer
        # EfficientNet structure: features -> avgpool -> classifier
        self.backbone = nn.Sequential(
            efficientnet.features,
            efficientnet.avgpool
        )

        # Optionally freeze backbone for faster training
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print(f"EfficientNet-{variant.upper()} backbone frozen - only training heads")
        else:
            print(f"EfficientNet-{variant.upper()} backbone unfrozen - fine-tuning all layers")

        # Policy Head (for action prediction - L_pred)
        # As per paper Figure 1: policy head operates on embeddings e_i directly
        self.policy_head = nn.Linear(self.repr_dim, num_actions)

    def forward(self, x):
        # Get features from pretrained backbone
        # Input: (B, 3, H, W) - typically (B, 3, 224, 224)
        h = self.backbone(x)  # Output: (B, repr_dim, 1, 1)
        h = h.view(h.size(0), -1)  # Flatten to (B, repr_dim)

        # SCIL Architecture (as per paper Figure 1):
        # Sequential flow: backbone → embeddings e_i (h) → BOTH losses use e_i
        # 1. SupCon loss uses e_i (h) - normalized internally by SupConLoss
        # 2. Policy head uses e_i (h) → action predictions → L_pred
        action_logits = self.policy_head(h)

        return action_logits, h


# Convenience function to create the right model
def create_scil_efficientnet(num_actions=7, freeze_backbone=False, variant='b0'):
    """
    Factory function to create SCIL EfficientNet model

    Variants:
        'b0': ~5.3M params, fastest, good for quick experiments
        'b1': ~7.8M params, slightly better accuracy
        'b2': ~9.2M params, better accuracy, slower
    """
    return SCILEncoderEfficientNet(
        num_actions=num_actions,
        freeze_backbone=freeze_backbone,
        variant=variant
    )
