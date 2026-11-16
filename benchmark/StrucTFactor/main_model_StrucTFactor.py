import json, os, random, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score, average_precision_score, matthews_corrcoef, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, Dataset, Subset

warnings.filterwarnings("ignore")

# ========== 0. Basic Configuration ==========
CSV_FILE = "13_multimodal_dataset.csv"
MAX_LEN = 1000
N_SPLITS = 5
BATCH_SIZE = 128
EPOCHS = 50
LEARNING_RATE = 0.001
MODEL_SAVE_TEMPLATE = "best_structfactor_13_model_fold_{}.pth"
THRESHOLD_SAVE_FILE = "structfactor_13_fold_thresholds.json"


# ========== 1. Fix Random Seed ==========
def seed_everything(seed=1):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(1)


# ========== 2. StrucTFactor Input Encoding ==========
# Sequence + SS3 -> 1000 x 63 one-hot
AA_LIST = list("ACDEFGHIKLMNPQRSTVWY") + ["X"]
SS_LIST = ["a", "b", "N"]
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_LIST)}
SS_TO_IDX = {ss: i for i, ss in enumerate(SS_LIST)}
N_FEATURES = 63


def normalize_ss3_char(ch: str) -> str:
    """Map several SS3 notations to a/b/N (helix / beta / coil)."""
    if ch in ["a", "H", "G", "I"]:
        return "a"
    if ch in ["b", "E", "B"]:
        return "b"
    return "N"


def encode_sequence_ss3(sequence: str, ss3: str) -> np.ndarray:
    """Encode Sequence + SS3 into a (MAX_LEN, 63) one-hot matrix."""
    sequence, ss3 = str(sequence).strip(), str(ss3).strip()
    x = np.zeros((MAX_LEN, N_FEATURES), dtype=np.float32)
    usable_len = min(len(sequence), len(ss3), MAX_LEN)
    for i in range(usable_len):
        aa = sequence[i] if sequence[i] in AA_TO_IDX else "X"
        ss = normalize_ss3_char(ss3[i])
        x[i, AA_TO_IDX[aa] * 3 + SS_TO_IDX[ss]] = 1.0
    return x


# ========== 3. StrucTFactor Dataset ==========
class StrucTFactorDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        missing = {"Sequence", "SS3", "Label"} - set(self.data.columns)
        if missing:
            raise ValueError(f"CSV file is missing required columns: {missing}")
        if not self.data["Label"].isin([0, 1]).all():
            raise ValueError("Label must be 0/1.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        x = encode_sequence_ss3(row["Sequence"], row["SS3"])
        # Conv2d input layout (C, H, W): single sample -> (1, 1000, 63)
        return (
            torch.tensor(x, dtype=torch.float32).unsqueeze(0),
            torch.tensor(float(row["Label"]), dtype=torch.float32),
        )


# ========== 4. StrucTFactor Model ==========
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dropout=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=0),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True), nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.block(x)


class SubNetwork(nn.Module):
    def __init__(self, kernels):
        super().__init__()
        self.net = nn.Sequential(
            ConvBlock(1, 128, kernel_size=kernels[0]),
            ConvBlock(128, 128, kernel_size=kernels[1]),
            ConvBlock(128, 128, kernel_size=kernels[2]),
        )

    def forward(self, x):
        return self.net(x)


class StrucTFactorModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.sub_a = SubNetwork(kernels=[(4, 63), (4, 1), (16, 1)])
        self.sub_b = SubNetwork(kernels=[(12, 63), (8, 1), (4, 1)])
        self.sub_c = SubNetwork(kernels=[(16, 63), (4, 1), (4, 1)])
        self.after_concat = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=(1, 1), stride=1, padding=0),
            nn.BatchNorm2d(384), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(MAX_LEN - 21, 1), stride=1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(384, 512), nn.BatchNorm1d(512), nn.ReLU(inplace=True),
            nn.Linear(512, 1), nn.BatchNorm1d(1),
        )
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x):
        x = torch.cat([self.sub_a(x), self.sub_b(x), self.sub_c(x)], dim=1)
        x = self.after_concat(x)
        return self.classifier(x).squeeze(1)


# ========== 5. Threshold Selection ==========
def find_best_threshold(labels, probs):
    labels = np.asarray(labels).reshape(-1)
    probs = np.asarray(probs).reshape(-1)
    best_threshold, best_mcc = 0.5, -1.0
    for threshold in np.arange(0.1, 0.91, 0.01):
        mcc = matthews_corrcoef(labels, (probs > threshold).astype(int))
        if mcc > best_mcc:
            best_mcc, best_threshold = mcc, float(threshold)
    return best_threshold, best_mcc


# ========== 6. 5-Fold Training + Independent Test Ensemble Evaluation ==========
def train_and_evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    full_dataset = StrucTFactorDataset(CSV_FILE)
    labels = full_dataset.data["Label"].astype(int).tolist()
    indices = list(range(len(full_dataset)))

    # Step 1: 10% independent blind test set
    print("\n[Step 1] Splitting 10% as the final blind test set...")
    cv_idx, test_idx, cv_labels, _ = train_test_split(
        indices, labels, test_size=0.1, stratify=labels, random_state=42)
    test_dataset = Subset(full_dataset, test_idx)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Step 2: 5-fold cross-validation
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    fold_best_thresholds, model_save_paths = [], []

    print("\n[Step 2] Starting StrucTFactor 5-fold cross-validation...")

    for fold, (train_v_idx, val_v_idx) in enumerate(skf.split(cv_idx, cv_labels), start=1):
        print(f"\n========== Fold {fold}/{N_SPLITS} ==========")
        train_dataset = Subset(full_dataset, [cv_idx[i] for i in train_v_idx])
        val_dataset = Subset(full_dataset, [cv_idx[i] for i in val_v_idx])

        if len(train_dataset) < BATCH_SIZE:
            raise ValueError(f"Fold {fold} training samples {len(train_dataset)} < BATCH_SIZE={BATCH_SIZE}. Please reduce BATCH_SIZE.")

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

        model = StrucTFactorModel().to(device)
        # BCEWithLogitsLoss + Adam(lr=0.001, weight_decay=0) per original StrucTFactor config
        criterion = nn.BCEWithLogitsLoss().to(device)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=0.0)

        best_val_loss, fold_best_thresh = float("inf"), 0.5
        save_path = MODEL_SAVE_TEMPLATE.format(fold)
        model_save_paths.append(save_path)

        for epoch in range(EPOCHS):
            # Train
            model.train()
            train_loss_sum, train_n = 0.0, 0
            for x, batch_labels in train_loader:
                x, batch_labels = x.to(device), batch_labels.to(device)
                optimizer.zero_grad()
                loss = criterion(model(x), batch_labels)
                loss.backward()
                optimizer.step()
                train_loss_sum += loss.item() * len(batch_labels)
                train_n += len(batch_labels)

            if train_n == 0:
                raise RuntimeError("Training DataLoader produced no batches. Please reduce BATCH_SIZE.")
            avg_train_loss = train_loss_sum / train_n

            # Validation
            model.eval()
            val_loss_sum, val_n, all_labels, all_probs = 0.0, 0, [], []
            with torch.no_grad():
                for x, batch_labels in val_loader:
                    x, batch_labels = x.to(device), batch_labels.to(device)
                    outputs = model(x)
                    val_loss_sum += criterion(outputs, batch_labels).item() * len(batch_labels)
                    val_n += len(batch_labels)
                    all_probs.extend(torch.sigmoid(outputs).detach().cpu().numpy().reshape(-1).tolist())
                    all_labels.extend(batch_labels.detach().cpu().numpy().reshape(-1).tolist())

            avg_val_loss = val_loss_sum / val_n
            labels_np = np.asarray(all_labels, dtype=int)
            probs_np = np.asarray(all_probs, dtype=float)
            best_epoch_thresh, best_epoch_mcc = find_best_threshold(labels_np, probs_np)
            auc = roc_auc_score(labels_np, probs_np)

            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"Epoch [{epoch + 1}/{EPOCHS}] | Train Loss: {avg_train_loss:.4f} | "
                      f"Val Loss: {avg_val_loss:.4f} | AUC: {auc:.4f} | "
                      f"MCC: {best_epoch_mcc:.4f} @ {best_epoch_thresh:.2f}")

            # Save best model per fold by lowest validation loss (same as original)
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                fold_best_thresh = best_epoch_thresh
                torch.save(model.state_dict(), save_path)

        print(f"✅ Fold {fold} done | Best Val Loss: {best_val_loss:.4f} | "
              f"Threshold: {fold_best_thresh:.2f} | Weights: {save_path}")
        fold_best_thresholds.append(float(fold_best_thresh))
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Save 5-fold thresholds
    avg_best_thresh = float(np.mean(fold_best_thresholds))
    with open(THRESHOLD_SAVE_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "fold_thresholds": fold_best_thresholds,
            "mean_threshold": avg_best_thresh,
            "rule": "validation MCC; thresholds 0.10-0.90 step 0.01; strict >",
        }, f, indent=2, ensure_ascii=False)
    print(f"\n5-fold thresholds saved to: {THRESHOLD_SAVE_FILE}")
    print(f"5-fold mean threshold: {avg_best_thresh:.3f}")

    # Step 3: 5-model ensemble on independent test set
    print("\n[Step 3] ================ StrucTFactor Final Ensemble Test ================")
    ensemble_test_probs = np.zeros(len(test_dataset), dtype=np.float64)
    test_labels_final = None

    for fold, path in enumerate(model_save_paths, start=1):
        print(f"Loading Fold {fold}: {path}")
        model = StrucTFactorModel().to(device)
        model.load_state_dict(torch.load(path, map_location=device), strict=True)
        model.eval()

        fold_probs, fold_labels = [], []
        with torch.no_grad():
            for x, batch_labels in test_loader:
                x = x.to(device)
                fold_probs.extend(torch.sigmoid(model(x)).cpu().numpy().reshape(-1).tolist())
                if fold == 1:
                    fold_labels.extend(batch_labels.numpy().reshape(-1).tolist())

        if fold == 1:
            test_labels_final = np.asarray(fold_labels, dtype=int)
        ensemble_test_probs += np.asarray(fold_probs, dtype=np.float64)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    ensemble_test_probs /= N_SPLITS
    test_preds = (ensemble_test_probs > avg_best_thresh).astype(int)

    print("\n🎉 Final Independent Test Results (StrucTFactor 5-Fold Ensemble):")
    print(f"   - Test Accuracy: {accuracy_score(test_labels_final, test_preds) * 100:.2f}%")
    print(f"   - Test AUC:      {roc_auc_score(test_labels_final, ensemble_test_probs):.4f}")
    print(f"   - Test AU-PRC:   {average_precision_score(test_labels_final, ensemble_test_probs):.4f}")
    print(f"   - Test MCC:      {matthews_corrcoef(test_labels_final, test_preds):.4f}")

    # Save independent test predictions
    result_df = full_dataset.data.iloc[test_idx].copy()
    result_df["Ensemble_Probability"] = ensemble_test_probs
    result_df["Predicted_Label"] = test_preds
    result_df.to_csv("structfactor_ensemble_test_predictions.csv", index=False)
    print("\nIndependent test predictions saved to: structfactor_ensemble_test_predictions.csv")


if __name__ == "__main__":
    train_and_evaluate()