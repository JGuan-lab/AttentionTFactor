import json, math, os, random, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, average_precision_score, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, Dataset, Subset

warnings.filterwarnings("ignore")

# ========== 0. Basic Configuration ==========
CSV_FILE = "13_multimodal_dataset.csv"
MAX_LEN = 1000
N_SPLITS = 5
BATCH_SIZE = 32
EPOCHS = 40
LEARNING_RATE = 0.0003
MODEL_SAVE_TEMPLATE = "best_attentiontfactor_13_model_fold_{}.pth"
THRESHOLD_SAVE_FILE = "attentiontfactor_13_fold_thresholds.json"


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


seed_everything(42)


# ========== 2. Dataset ==========
class TFDatasetFromScratch(Dataset):
    def __init__(self, csv_file, max_len=1000):
        self.data = pd.read_csv(csv_file)
        self.max_len = max_len
        missing = {"Sequence", "SS3", "Label"} - set(self.data.columns)
        if missing:
            raise ValueError(f"CSV file is missing required columns: {missing}")
        if not self.data["Label"].isin([0, 1]).all():
            raise ValueError("Label must be 0/1.")
        amino_acids = "ACDEFGHIKLMNPQRSTVWY"
        self.seq_vocab = {aa: idx + 1 for idx, aa in enumerate(amino_acids)}
        self.ss3_vocab = {"H": 1, "E": 2, "C": 3, "N": 3, "a": 1, "b": 2}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        seq_ids = [self.seq_vocab.get(aa, 0) for aa in str(row["Sequence"])[:self.max_len]]
        ss3_ids = [self.ss3_vocab.get(s, 0) for s in str(row["SS3"])[:self.max_len]]
        seq_padded = seq_ids + [0] * (self.max_len - len(seq_ids))
        ss3_padded = ss3_ids + [0] * (self.max_len - len(ss3_ids))
        label = float(row["Label"])
        return (torch.tensor(seq_padded, dtype=torch.long),
                torch.tensor(ss3_padded, dtype=torch.long),
                torch.tensor(label, dtype=torch.float32))


# ========== 3. Basic Model Modules ==========
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class SEBlock(nn.Module):
    def __init__(self, channel, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.fc(self.avg_pool(x).view(b, c)).view(b, c, 1)
        return x * y.expand_as(x)


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha, self.gamma = alpha, gamma

    def forward(self, inputs, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        pt = torch.exp(-bce)
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        return (alpha_t * (1 - pt) ** self.gamma * bce).mean()


# ========== 4. AttentionTFactor Model ==========
class AttentionTFactorCNN(nn.Module):
    def __init__(self, seq_vocab_size=21, ss3_vocab_size=4, embed_dim=256):
        super().__init__()
        self.seq_embedding = nn.Embedding(seq_vocab_size, embed_dim, padding_idx=0)
        self.ss3_embedding = nn.Embedding(ss3_vocab_size, embed_dim, padding_idx=0)
        in_ch = embed_dim * 2
        out_ch = 64
        self.branch_a = self._make_branch(in_ch, out_ch, [4, 4, 16])
        self.branch_b = self._make_branch(in_ch, out_ch, [12, 8, 4])
        self.branch_c = self._make_branch(in_ch, out_ch, [16, 4, 4])
        cnn_out = out_ch * 3
        self.pos_encoder = PositionalEncoding(d_model=cnn_out, max_len=MAX_LEN)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=cnn_out, nhead=8, dim_feedforward=512,
                                       dropout=0.1, batch_first=True),
            num_layers=1)
        self.fc = nn.Sequential(
            nn.Linear(cnn_out * 2, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Dropout(0.5), nn.Linear(128, 1))
        self._init_weights()

    def _make_branch(self, in_c, out_c, kernels):
        return nn.Sequential(
            nn.Conv1d(in_c, out_c, kernel_size=kernels[0], padding="same"), nn.BatchNorm1d(out_c), nn.ReLU(),
            nn.Conv1d(out_c, out_c, kernel_size=kernels[1], padding="same"), nn.BatchNorm1d(out_c), nn.ReLU(),
            nn.Conv1d(out_c, out_c, kernel_size=kernels[2], padding="same"), nn.BatchNorm1d(out_c), nn.ReLU(),
            SEBlock(out_c), nn.Dropout(0.1))

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")

    def forward(self, x_seq, x_ss3):
        padding_mask = (x_seq == 0)
        z = torch.cat([self.seq_embedding(x_seq), self.ss3_embedding(x_ss3)], dim=-1).transpose(1, 2)
        feats = torch.cat([self.branch_a(z), self.branch_b(z), self.branch_c(z)], dim=1)
        trans_in = self.pos_encoder(feats.transpose(1, 2))
        gf = self.transformer(trans_in, src_key_padding_mask=padding_mask)
        mask = padding_mask.unsqueeze(-1).expand_as(gf)
        p1 = gf.masked_fill(mask, -1e9).max(dim=1)[0]
        valid_lens = torch.clamp((~padding_mask).sum(dim=1, keepdim=True).float(), min=1.0)
        p2 = gf.masked_fill(mask, 0.0).sum(dim=1) / valid_lens
        return self.fc(torch.cat([p1, p2], dim=1)).squeeze(1)


# ========== 5. Best Threshold Search ==========
def find_best_threshold(labels, probs):
    labels = np.asarray(labels).reshape(-1)
    probs = np.asarray(probs).reshape(-1)
    best_threshold, best_mcc = 0.5, -1.0
    for t in np.arange(0.1, 0.91, 0.01):
        mcc = matthews_corrcoef(labels, (probs > t).astype(int))
        if mcc > best_mcc:
            best_mcc, best_threshold = mcc, float(t)
    return best_threshold, best_mcc


# ========== 6. Training + Evaluation ==========
def train_and_evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    full_dataset = TFDatasetFromScratch(CSV_FILE, max_len=MAX_LEN)
    labels = full_dataset.data["Label"].astype(int).tolist()
    indices = list(range(len(full_dataset)))

    # Step 1: 10% independent test set
    print("\n[Step 1] Splitting 10% as the final blind test set...")
    cv_idx, test_idx, cv_labels, _ = train_test_split(
        indices, labels, test_size=0.1, stratify=labels, random_state=42)
    test_loader = DataLoader(Subset(full_dataset, test_idx), batch_size=BATCH_SIZE, shuffle=False)

    # Step 2: 5-fold cross-validation
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    fold_best_thresholds, model_save_paths = [], []
    print("\n[Step 2] Starting AttentionTFactor 5-fold cross-validation...")

    for fold, (train_v_idx, val_v_idx) in enumerate(skf.split(cv_idx, cv_labels), start=1):
        print(f"\n========== Fold {fold}/{N_SPLITS} ==========")
        train_loader = DataLoader(
            Subset(full_dataset, [cv_idx[i] for i in train_v_idx]),
            batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
        val_loader = DataLoader(
            Subset(full_dataset, [cv_idx[i] for i in val_v_idx]),
            batch_size=BATCH_SIZE, shuffle=False)

        model = AttentionTFactorCNN().to(device)
        criterion = FocalLoss(alpha=0.75, gamma=2.0).to(device)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4)

        best_val_loss, fold_best_thresh = float("inf"), 0.5
        patience_counter, early_stopping = 0, 10
        save_path = MODEL_SAVE_TEMPLATE.format(fold)
        model_save_paths.append(save_path)

        for epoch in range(EPOCHS):
            # Train
            model.train()
            train_loss = 0.0
            for seq, ss3, bl in train_loader:
                seq, ss3, bl = seq.to(device), ss3.to(device), bl.to(device)
                optimizer.zero_grad()
                loss = criterion(model(seq, ss3), bl)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item()
            avg_train_loss = train_loss / len(train_loader)

            # Validation
            model.eval()
            val_loss, all_labels, all_probs = 0.0, [], []
            with torch.no_grad():
                for seq, ss3, bl in val_loader:
                    seq, ss3, bl = seq.to(device), ss3.to(device), bl.to(device)
                    outputs = model(seq, ss3)
                    val_loss += criterion(outputs, bl).item()
                    all_probs.extend(torch.sigmoid(outputs).cpu().numpy().reshape(-1).tolist())
                    all_labels.extend(bl.cpu().numpy().reshape(-1).tolist())
            avg_val_loss = val_loss / len(val_loader)

            labels_np = np.asarray(all_labels, dtype=int)
            probs_np = np.asarray(all_probs, dtype=float)
            best_epoch_thresh, best_epoch_mcc = find_best_threshold(labels_np, probs_np)
            auc = roc_auc_score(labels_np, probs_np)
            scheduler.step(avg_val_loss)

            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"Epoch [{epoch + 1}/{EPOCHS}] | Train Loss: {avg_train_loss:.4f} | "
                      f"Val Loss: {avg_val_loss:.4f} | AUC: {auc:.4f} | "
                      f"MCC: {best_epoch_mcc:.4f} @ {best_epoch_thresh:.2f}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                fold_best_thresh = best_epoch_thresh
                torch.save(model.state_dict(), save_path)
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping:
                    print(f" Fold {fold} triggered early stopping at epoch {epoch + 1}.")
                    break

        print(f" Fold {fold} done | Best Val Loss: {best_val_loss:.4f} | "
              f"Threshold: {fold_best_thresh:.2f} | Weights: {save_path}")
        fold_best_thresholds.append(float(fold_best_thresh))
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Save per-fold thresholds
    avg_best_thresh = float(np.mean(fold_best_thresholds))
    with open(THRESHOLD_SAVE_FILE, "w", encoding="utf-8") as f:
        json.dump({"fold_thresholds": fold_best_thresholds, "mean_threshold": avg_best_thresh,
                   "rule": "validation MCC; thresholds 0.10-0.90 step 0.01; strict >"},
                  f, indent=2, ensure_ascii=False)
    print(f"\n5-fold thresholds saved to: {THRESHOLD_SAVE_FILE}")
    print(f"5-fold mean threshold: {avg_best_thresh:.3f}")

    # Step 3: 5-model ensemble
    print("\n[Step 3] ================ AttentionTFactor Final Ensemble Test ================")
    ensemble_test_probs = np.zeros(len(test_idx), dtype=np.float64)
    test_labels_final = None

    for fold, path in enumerate(model_save_paths, start=1):
        print(f"Loading Fold {fold}: {path}")
        model = AttentionTFactorCNN().to(device)
        model.load_state_dict(torch.load(path, map_location=device), strict=True)
        model.eval()
        fold_probs, fold_labels = [], []
        with torch.no_grad():
            for seq, ss3, bl in test_loader:
                seq, ss3 = seq.to(device), ss3.to(device)
                fold_probs.extend(torch.sigmoid(model(seq, ss3)).cpu().numpy().reshape(-1).tolist())
                if fold == 1:
                    fold_labels.extend(bl.numpy().reshape(-1).tolist())
        if fold == 1:
            test_labels_final = np.asarray(fold_labels, dtype=int)
        ensemble_test_probs += np.asarray(fold_probs, dtype=np.float64)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    ensemble_test_probs /= N_SPLITS
    test_preds = (ensemble_test_probs > avg_best_thresh).astype(int)

    print("\n Final Independent Test Results (AttentionTFactor 5-Fold Ensemble):")
    print(f"   - Test Accuracy: {accuracy_score(test_labels_final, test_preds) * 100:.2f}%")
    print(f"   - Test AUC:      {roc_auc_score(test_labels_final, ensemble_test_probs):.4f}")
    print(f"   - Test AU-PRC:   {average_precision_score(test_labels_final, ensemble_test_probs):.4f}")
    print(f"   - Test MCC:      {matthews_corrcoef(test_labels_final, test_preds):.4f}")

    result_df = full_dataset.data.iloc[test_idx].copy()
    result_df["Ensemble_Probability"] = ensemble_test_probs
    result_df["Predicted_Label"] = test_preds
    result_df.to_csv("attentiontfactor_ensemble_test_predictions.csv", index=False)
    print("\nIndependent test predictions saved to: attentiontfactor_ensemble_test_predictions.csv")


if __name__ == "__main__":
    train_and_evaluate()