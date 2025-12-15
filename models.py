# models.py
import torch.nn as nn
import torch.nn.functional as F

class SCILEncoder(nn.Module):
    def __init__(self, num_actions=7, projection_dim=128):
        """
        SCIL Model with both policy head and projection head

        Args:
            num_actions: Number of discrete actions (7 for Mario SIMPLE_MOVEMENT)
            projection_dim: Dimension of projection head output (z)
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
        self.policy_head = nn.Linear(512, num_actions)

        # Projection Head (for SCIL contrastive loss - L_SupCon)
        # Simpler 2-layer MLP without batch norm
        self.projection_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, projection_dim)
        )

    def forward(self, x):
        # Representation (h) - This is what we keep for RL/SAPS
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        h = F.relu(self.fc(x))

        # Action logits (for L_pred)
        action_logits = self.policy_head(h)

        # Projection (z) - For L_SupCon
        z = self.projection_head(h)

        return action_logits, h, z