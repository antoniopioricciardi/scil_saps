# Test SupCon loss in isolation
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from losses import supervised_contrastive_loss

# Create a simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 128)
        self.fc2 = nn.Linear(128, 64)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = SimpleModel()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Create simple data
torch.manual_seed(42)
X = torch.randn(32, 10)
y = torch.randint(0, 5, (32,))  # 5 classes

print("Testing SupCon loss learning:")
for epoch in range(20):
    optimizer.zero_grad()

    embeddings = model(X)
    loss = supervised_contrastive_loss(embeddings, y, temperature=0.07)

    loss.backward()
    optimizer.step()

    if epoch % 5 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

        # Check if embeddings are diverse
        z_norm = F.normalize(embeddings, dim=1)
        sim = torch.matmul(z_norm, z_norm.T)
        print(f"  Similarity: mean={sim.mean():.4f}, std={sim.std():.4f}")
