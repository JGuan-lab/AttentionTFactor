import json, math, os, random, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, average_precision_score, matthews_corrcoef, roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore")

# ========== 0. Basic Configuration ==========
CSV_FILE = "bacillus_subtilis_multimodal_dataset.csv"
MODEL_PATHS = [f"best_attentiontfactor_bacteria_domain_model_fold_{i}.pth" for i in range(1, 6)]
THRESHOLD_FILE = "attentiontfactor_bacteria_domain_fold_thresholds.json"
BATCH_SIZE = 32
MAX_LEN = 1000
FOLD_BEST_THRESHOLDS = None
DEFAULT_THRESHOLD = 0.5


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


# ========== 5. Utility Functions ==========
def load_model_weight(model, weight_path, device):
    state_dict = torch.load(weight_path, map_location=device)
    if any(key.startswith("module.") for key in state_dict.keys()):
        state_dict = {key.replace("module.", "", 1): value for key, value in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    return model


def get_final_threshold():
    if os.path.exists(THRESHOLD_FILE):
        with open(THRESHOLD_FILE, "r", encoding="utf-8") as f:
            info = json.load(f)
        fold_thresholds = info.get("fold_thresholds", None)
        if fold_thresholds:
            return float(np.mean(fold_thresholds))
        if "mean_threshold" in info:
            return float(info["mean_threshold"])
    if FOLD_BEST_THRESHOLDS is not None:
        return float(np.mean(np.asarray(FOLD_BEST_THRESHOLDS, dtype=float)))
    return float(DEFAULT_THRESHOLD)


# ========== 6. Single-Species Cross-Species Generalization Test ==========
def single_species_generalization_test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset = TFDatasetFromScratch(CSV_FILE, max_len=MAX_LEN)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    final_threshold = get_final_threshold()

    print("\n[Step 1] Target species data loaded")
    print(f"Number of samples: {len(dataset)}")
    labels_array = dataset.data["Label"].values
    print(f"Positive samples: {int(np.sum(labels_array == 1))}")
    print(f"Negative samples: {int(np.sum(labels_array == 0))}")

    print("\n[Step 2] Loading AttentionTFactor 5-fold models for ensemble prediction")
    print(f"Using threshold: {final_threshold:.3f}")

    all_fold_probs = []
    true_labels = None

    for fold_idx, weight_path in enumerate(MODEL_PATHS, start=1):
        if not os.path.exists(weight_path):
            raise FileNotFoundError(f"Weight file not found: {weight_path}")
        print(f"Loading model Fold {fold_idx}: {weight_path}")
        model = AttentionTFactorCNN().to(device)
        model = load_model_weight(model, weight_path, device)
        model.eval()
        fold_probs, fold_labels = [], []
        with torch.no_grad():
            for seq, ss3, labels in loader:
                seq, ss3 = seq.to(device), ss3.to(device)
                probs = torch.sigmoid(model(seq, ss3))
                fold_probs.extend(probs.cpu().numpy().reshape(-1).tolist())
                fold_labels.extend(labels.numpy().reshape(-1).tolist())
        all_fold_probs.append(np.asarray(fold_probs, dtype=np.float64))
        if true_labels is None:
            true_labels = np.asarray(fold_labels, dtype=int)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    all_fold_probs = np.vstack(all_fold_probs)
    ensemble_probs = np.mean(all_fold_probs, axis=0)
    preds = (ensemble_probs > final_threshold).astype(int)

    print("\n[Step 3] AttentionTFactor Single-Species Generalization Test Results")
    print("========================================")
    acc = accuracy_score(true_labels, preds)
    auc_score = roc_auc_score(true_labels, ensemble_probs)
    auprc = average_precision_score(true_labels, ensemble_probs)
    mcc = matthews_corrcoef(true_labels, preds)
    print(f"Test Accuracy: {acc * 100:.2f}%")
    print(f"Test AUC:      {auc_score:.4f}")
    print(f"Test AU-PRC:   {auprc:.4f}")
    print(f"Test MCC:      {mcc:.4f}")
    print("========================================")


if __name__ == "__main__":
    single_species_generalization_test()