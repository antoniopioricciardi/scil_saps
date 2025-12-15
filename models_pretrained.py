# models_pretrained.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class SCILEncoderPretrained(nn.Module):
    def __init__(self, num_actions=7, freeze_backbone=False):
        """
        SCIL Model with pretrained ResNet18 backbone

        Args:
            num_actions: Number of discrete actions
            freeze_backbone: If True, freeze ResNet weights (faster training)
        """
        super(SCILEncoderPretrained, self).__init__()

        # Load pretrained ResNet18 (trained on ImageNet)
        resnet = models.resnet18(pretrained=True)

        # Remove the final FC layer (we'll add our own heads)
        # ResNet18 outputs 512-dim features before the final FC
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        # Optionally freeze backbone for faster training
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print("Backbone frozen - only training heads")
        else:
            print("Backbone unfrozen - fine-tuning all layers")

        # Representation dimension from ResNet18
        self.repr_dim = 512

        # Policy Head (for action prediction - L_pred)
        # As per paper Figure 1: policy head operates on embeddings e_i directly
        self.policy_head = nn.Linear(self.repr_dim, num_actions)

    def forward(self, x):
        # Get features from pretrained backbone
        # Input: (B, 3, 84, 84)
        h = self.backbone(x)  # Output: (B, 512, 1, 1)
        h = h.view(h.size(0), -1)  # Flatten to (B, 512)

        # SCIL Architecture (as per paper Figure 1):
        # Sequential flow: backbone → embeddings e_i (h) → BOTH losses use e_i
        # 1. SupCon loss uses e_i (h) - normalized internally by SupConLoss
        # 2. Policy head uses e_i (h) → action predictions → L_pred
        action_logits = self.policy_head(h)

        return action_logits, h
