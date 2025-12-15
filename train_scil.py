# train_scil.py
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

# Import our custom modules
from dataset import MarioSCILDataset
from models import SCILEncoder
from losses_paper import SupConLoss

# --- CONFIG ---
# DATA_FILES can be:
#   - Single file: "mario_1_1_expert.pkl"
#   - List of files: ["mario_1_1.pkl", "mario_1_2.pkl", "mario_1_3.pkl"]
#   - Glob pattern: "mario_*_expert.pkl" (loads ALL matching files)
DATA_FILES = "mario_*_expert.pkl"  # Will load all expert data files

SAVE_PATH = "scil_encoder_mario.pth"
BATCH_SIZE = 64
LR = 3e-4
EPOCHS = 50
TEMPERATURE = 0.07
LAMBDA_SUPCON = 0.0  # Weight for SupCon loss (set to 0 for basic IL, 0.5 for SCIL)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def validate(model, val_loader):
    """Validation function to track action prediction accuracy"""
    model.eval()
    correct = 0
    total = 0
    all_predictions = []

    with torch.no_grad():
        for obs, actions in val_loader:
            obs, actions = obs.to(DEVICE), actions.to(DEVICE)

            action_logits, _, _ = model(obs)
            predictions = torch.argmax(action_logits, dim=1)

            correct += (predictions == actions).sum().item()
            total += actions.size(0)
            all_predictions.extend(predictions.cpu().tolist())

    accuracy = 100.0 * correct / total

    # Check prediction diversity
    unique_preds = len(set(all_predictions))
    most_common_pred = max(set(all_predictions), key=all_predictions.count)
    most_common_count = all_predictions.count(most_common_pred)

    return accuracy, unique_preds, most_common_pred, most_common_count, total


def main():
    # 1. Setup Data
    dataset = MarioSCILDataset(DATA_FILES, img_size=84, use_imagenet_norm=False)

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

    # 2. Setup Model
    model = SCILEncoder(num_actions=7, projection_dim=128).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # Setup SupCon loss (from paper)
    supcon_criterion = SupConLoss(device=DEVICE, temperature=TEMPERATURE)

    print(f"Starting training on {DEVICE}...")
    print(f"Lambda SupCon: {LAMBDA_SUPCON}")
    print(f"Training Nature CNN from scratch (no pretrained weights)")

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
            action_logits, h, z = model(obs)

            # L_pred: Cross-Entropy Loss for action prediction
            # NOTE: Extreme class weights (35x) may cause instability - trying without first
            pred_loss = F.cross_entropy(action_logits, actions)  # , weight=class_weights)

            # L_SupCon: Supervised Contrastive Loss (from paper implementation)
            supcon_loss = supcon_criterion(z, actions)

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

        # Validation
        val_acc, unique_preds, most_common_pred, most_common_count, total_val = validate(model, val_loader)

        print(f"Epoch {epoch+1}/{EPOCHS} | "
              f"Loss: {avg_loss:.4f} (Pred: {avg_pred:.4f}, SupCon: {avg_supcon:.4f}) | "
              f"Grad: {avg_grad_norm:.4f} | "
              f"Val Acc: {val_acc:.2f}%")
        print(f"  Val Stats: {unique_preds} unique predictions, "
              f"most common: class {most_common_pred} ({most_common_count}/{total_val} = {100*most_common_count/total_val:.1f}%)")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"  → New best! Saved to {SAVE_PATH}")

    print(f"\nTraining complete! Best Val Acc: {best_val_acc:.2f}%")

if __name__ == "__main__":
    main()