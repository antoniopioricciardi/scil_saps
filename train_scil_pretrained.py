# train_scil_pretrained.py
# SCIL training with pretrained encoder (ResNet18 or EfficientNet)

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

# Import our custom modules
from dataset import MarioSCILDataset
from models_pretrained import SCILEncoderPretrained
from model_efficientnet import SCILEncoderEfficientNet
from losses_paper import SupConLoss

# --- CONFIG ---
# Data selection - examples:
#   "mario_*_expert.pkl"           → all_levels
#   "mario_1_1_expert.pkl"         → 1_1
#   "mario_1_[12]_expert.pkl"      → 1_1_1_2
#   ["mario_1_1.pkl", "mario_1_2.pkl"] → 1_1_1_2
# DATA_FILES = "data/mario_*_expert.pkl"
DATA_FILES = "data/mario_1_2_expert.pkl"

# Model selection: 'resnet18' or 'efficientnet-b0' or 'efficientnet-b1' or 'efficientnet-b2'
BACKBONE = "efficientnet-b1"  # Options: 'resnet18', 'efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2'

# Auto-generate data tag for filename
def get_data_tag(data_files):
    """Generate a descriptive tag from DATA_FILES pattern"""
    if isinstance(data_files, str):
        if '*' in data_files or 'all' in data_files.lower():
            return "all_levels"
        # Extract level info (e.g., "mario_1_1_expert.pkl" → "1_1")
        import re
        match = re.search(r'mario_(\d+_\d+)', data_files)
        if match:
            return match.group(1)
        return "custom"
    elif isinstance(data_files, list):
        # Multiple specific files
        import re
        levels = []
        for f in data_files:
            match = re.search(r'mario_(\d+_\d+)', f)
            if match:
                levels.append(match.group(1))
        return "_".join(levels) if levels else "multi"
    return "unknown"

DATA_TAG = get_data_tag(DATA_FILES)

BATCH_SIZE = 32  # Reduced from 64 for 8GB GPU with EfficientNet-B1
LR = 1e-3 #3e-4
EPOCHS = 80
TEMPERATURE = 0.05
LAMBDA_SUPCON = 2 #0.5  # Weight for SupCon loss
FREEZE_BACKBONE = False  # Set True for faster training, False for better performance

SAVE_PATH = f"checkpoints/scil_encoder_mario_{DATA_TAG}_{BACKBONE.replace('-', '_')}_lam{LAMBDA_SUPCON}.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def validate(model, val_loader):
    """Validation function to track action prediction accuracy"""
    model.eval()
    correct = 0
    total = 0
    all_predictions = []
    all_actions = []
    all_logits = []
    all_features = []

    with torch.no_grad():
        for obs, actions in val_loader:
            obs, actions = obs.to(DEVICE), actions.to(DEVICE)

            action_logits, h = model(obs)
            predictions = torch.argmax(action_logits, dim=1)

            correct += (predictions == actions).sum().item()
            total += actions.size(0)

            all_predictions.extend(predictions.cpu().tolist())
            all_actions.extend(actions.cpu().tolist())
            all_logits.append(action_logits.cpu())
            all_features.append(h.cpu())

    accuracy = 100.0 * correct / total

    # Debug: Check prediction diversity
    unique_preds = len(set(all_predictions))
    most_common_pred = max(set(all_predictions), key=all_predictions.count)
    most_common_count = all_predictions.count(most_common_pred)

    # Debug: Check logits and features
    all_logits = torch.cat(all_logits, dim=0)
    all_features = torch.cat(all_features, dim=0)

    # Check feature diversity (std across samples)
    feature_std = all_features.std(dim=0).mean().item()

    # Check logit statistics
    logit_mean = all_logits.mean(dim=0)
    logit_std = all_logits.std(dim=0)

    return accuracy, unique_preds, most_common_pred, most_common_count, total, feature_std, logit_mean, logit_std


def main():
    # Clear GPU cache before training
    torch.cuda.empty_cache()

    # 1. Setup Data
    dataset = MarioSCILDataset(DATA_FILES, img_size=224, use_imagenet_norm=True)

    # Split into train/val (90/10)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # Check action distribution in training data and compute class weights
    print("\nChecking action distribution in training data...")
    action_counts = {}
    for _, action in train_loader:
        for a in action.tolist():
            action_counts[a] = action_counts.get(a, 0) + 1
    print("Action distribution in training set:")
    total_actions = sum(action_counts.values())
    for action in sorted(action_counts.keys()):
        count = action_counts[action]
        print(f"  Action {action}: {count} ({100*count/total_actions:.1f}%)")

    # Compute class weights (inverse frequency)
    num_actions = 7
    class_weights = torch.zeros(num_actions, device=DEVICE)
    for action in range(num_actions):
        count = action_counts.get(action, 1)  # Avoid division by zero
        class_weights[action] = total_actions / (num_actions * count)
    print(f"\nClass weights: {class_weights.cpu().numpy()}")

    # 2. Setup Model with Pretrained Encoder
    if BACKBONE == 'resnet18':
        model = SCILEncoderPretrained(
            num_actions=7,
            freeze_backbone=FREEZE_BACKBONE
        ).to(DEVICE)
    elif BACKBONE.startswith('efficientnet'):
        # Extract variant (b0, b1, b2)
        variant = BACKBONE.split('-')[1]  # 'efficientnet-b0' -> 'b0'
        model = SCILEncoderEfficientNet(
            num_actions=7,
            freeze_backbone=FREEZE_BACKBONE,
            variant=variant
        ).to(DEVICE)
    else:
        raise ValueError(f"Unknown backbone: {BACKBONE}. Choose 'resnet18' or 'efficientnet-b0/b1/b2'")

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
            {'params': model.policy_head.parameters()}
        ], lr=LR)

    # Setup SupCon loss (from paper)
    supcon_criterion = SupConLoss(device=DEVICE, temperature=TEMPERATURE)

    print(f"Starting training on {DEVICE}...")
    print(f"Lambda SupCon: {LAMBDA_SUPCON}")
    print(f"Using pretrained {BACKBONE.upper()} backbone")

    # Debug: Store initial policy head weights to verify they change
    initial_policy_weight = model.policy_head.weight.data.clone()

    # 3. Training Loop
    best_val_acc = 0.0
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        total_pred_loss = 0
        total_supcon_loss = 0
        valid_batches = 0
        total_grad_norm = 0

        for batch_idx, (obs, actions) in enumerate(train_loader):
            obs, actions = obs.to(DEVICE), actions.to(DEVICE)

            # Forward pass
            action_logits, h = model(obs)

            # L_pred: Cross-Entropy Loss for action prediction (with class weighting)
            pred_loss = F.cross_entropy(action_logits, actions, weight=class_weights)

            # L_SupCon: Supervised Contrastive Loss (from paper implementation)
            # As per paper Figure 1: SupCon operates on embeddings e_i (h) directly
            supcon_loss = supcon_criterion(h, actions)

            # Combined Loss (as per SCIL paper)
            loss = pred_loss + LAMBDA_SUPCON * supcon_loss

            if not torch.isnan(loss):
                optimizer.zero_grad()
                loss.backward()

                # Check gradient norm
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))
                total_grad_norm += grad_norm.item()

                optimizer.step()

                total_loss += loss.item()
                total_pred_loss += pred_loss.item()
                total_supcon_loss += supcon_loss.item()
                valid_batches += 1

        # Compute averages
        avg_loss = total_loss / (valid_batches + 1e-8)
        avg_pred = total_pred_loss / (valid_batches + 1e-8)
        avg_supcon = total_supcon_loss / (valid_batches + 1e-8)
        avg_grad_norm = total_grad_norm / (valid_batches + 1e-8)

        # Check if weights are changing
        weight_change = (model.policy_head.weight.data - initial_policy_weight).abs().max().item()

        # Validation
        val_acc, unique_preds, most_common_pred, most_common_count, total_val, feature_std, logit_mean, logit_std = validate(model, val_loader)

        print(f"Epoch {epoch+1}/{EPOCHS} | "
              f"Loss: {avg_loss:.4f} (Pred: {avg_pred:.4f}, SupCon: {avg_supcon:.4f}) | "
              f"Grad: {avg_grad_norm:.4f} | "
              f"Val Acc: {val_acc:.2f}%")
        # print(f"  Val Stats: {unique_preds} unique predictions, "
        #       f"most common: class {most_common_pred} ({most_common_count}/{total_val} = {100*most_common_count/total_val:.1f}%)")
        # print(f"  Weight change: {weight_change:.6f}, Feature std: {feature_std:.4f}")
        # print(f"  Logit means: {logit_mean.numpy()}")
        # print(f"  Logit stds:  {logit_std.numpy()}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"  → New best! Saved to {SAVE_PATH}")

    print(f"\nTraining complete! Best Val Acc: {best_val_acc:.2f}%")


if __name__ == "__main__":
    main()
