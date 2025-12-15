# models.py
import torch.nn as nn
import torch.nn.functional as F

class SCILEncoder(nn.Module):
    def __init__(self, num_actions=7):
        """
        SCIL Model - Nature CNN backbone with policy head

        Args:
            num_actions: Number of discrete actions (7 for Mario SIMPLE_MOVEMENT)
        """
        super(SCILEncoder, self).__init__()

        # Nature CNN Backbone (adapted for RGB)
        # Input: 3 channels (RGB), 84x84
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Flatten size: 64 * 7 * 7 = 3136
        self.fc = nn.Linear(3136, 512)

        # Policy Head (for action prediction - L_pred)
        # As per paper Figure 1: policy head operates on embeddings e_i directly
        self.policy_head = nn.Linear(512, num_actions)

    def forward(self, x):
        # Representation (h) - This is what we keep for RL/SAPS
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        h = F.relu(self.fc(x))

        # SCIL Architecture (as per paper Figure 1):
        # Sequential flow: backbone → embeddings e_i (h) → BOTH losses use e_i
        # 1. SupCon loss uses e_i (h) - normalized internally by SupConLoss
        # 2. Policy head uses e_i (h) → action predictions → L_pred
        action_logits = self.policy_head(h)

        return action_logits, h