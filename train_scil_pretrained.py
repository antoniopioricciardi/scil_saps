# train_scil_pretrained.py
# SCIL training with pretrained ResNet18 encoder

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

# Import our custom modules
from dataset import MarioSCILDataset
from models_pretrained import SCILEncoderPretrained
from losses_paper import SupConLoss

# --- CONFIG ---
DATA_FILES = "mario_*_expert.pkl"  # Will load all expert data files

SAVE_PATH = "scil_encoder_mario_pretrained.pth"
BATCH_SIZE = 64
LR = 3e-4
EPOCHS = 50
TEMPERATURE = 0.07
LAMBDA_SUPCON = 0.5  # Weight for SupCon loss
FREEZE_BACKBONE = False  # Set True for faster training, False for better performance
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def validate(model, val_loader):
    """Validation function to track action prediction accuracy"""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for obs, actions in val_loader:
            obs, actions = obs.to(DEVICE), actions.to(DEVICE)

            action_logits, _, _ = model(obs)
            predictions = torch.argmax(action_logits, dim=1)

            correct += (predictions == actions).sum().item()
            total += actions.size(0)

    accuracy = 100.0 * correct / total
    return accuracy


def main():
    # 1. Setup Data
    dataset = MarioSCILDataset(DATA_FILES)

    # Split into train/val (90/10)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # 2. Setup Model with Pretrained Encoder
    model = SCILEncoderPretrained(
        num_actions=7,
        projection_dim=128,
        freeze_backbone=FREEZE_BACKBONE
    ).to(DEVICE)

    # Use different learning rates for pretrained backbone vs heads
    if FREEZE_BACKBONE:
        # Only train heads
        optimizer = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad],
            lr=LR
        )
    else:
        # Fine-tune backbone with lower LR, train heads with normal LR
        optimizer = torch.optim.Adam([
            {'params': model.backbone.parameters(), 'lr': LR / 10},  # Lower LR for pretrained
            {'params': model.policy_head.parameters()},
            {'params': model.projection_head.parameters()}
        ], lr=LR)

    # Setup SupCon loss (from paper)
    supcon_criterion = SupConLoss(device=DEVICE, temperature=TEMPERATURE)

    print(f"Starting training on {DEVICE}...")
    print(f"Lambda SupCon: {LAMBDA_SUPCON}")
    print(f"Using pretrained ResNet18 backbone")

    # 3. Training Loop
    best_val_acc = 0.0
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        total_pred_loss = 0
        total_supcon_loss = 0
        valid_batches = 0

        for batch_idx, (obs, actions) in enumerate(train_loader):
            obs, actions = obs.to(DEVICE), actions.to(DEVICE)

            # Forward pass
            action_logits, h, z = model(obs)

            # L_pred: Cross-Entropy Loss for action prediction
            pred_loss = F.cross_entropy(action_logits, actions)

            # L_SupCon: Supervised Contrastive Loss (from paper implementation)
            supcon_loss = supcon_criterion(z, actions)

            # Combined Loss (as per SCIL paper)
            loss = pred_loss + LAMBDA_SUPCON * supcon_loss

            if not torch.isnan(loss):
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                total_pred_loss += pred_loss.item()
                total_supcon_loss += supcon_loss.item()
                valid_batches += 1

        # Compute averages
        avg_loss = total_loss / (valid_batches + 1e-8)
        avg_pred = total_pred_loss / (valid_batches + 1e-8)
        avg_supcon = total_supcon_loss / (valid_batches + 1e-8)

        # Validation
        val_acc = validate(model, val_loader)

        print(f"Epoch {epoch+1}/{EPOCHS} | "
              f"Loss: {avg_loss:.4f} (Pred: {avg_pred:.4f}, SupCon: {avg_supcon:.4f}) | "
              f"Val Acc: {val_acc:.2f}%")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"  → New best! Saved to {SAVE_PATH}")

    print(f"\nTraining complete! Best Val Acc: {best_val_acc:.2f}%")


if __name__ == "__main__":
    main()
