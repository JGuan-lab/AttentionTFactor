import json, os, random, warnings
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
CSV_FILE = "bacillus_subtilis_multimodal_dataset.csv"   # change to the target species CSV
MODEL_PATHS = [f"best_structfactor_bacteria_domain_model_fold_{i}.pth" for i in range(1, 6)]
THRESHOLD_FILE = "structfactor_bacteria_domain_fold_thresholds.json"
BATCH_SIZE = 128
MAX_LEN = 1000
FOLD_BEST_THRESHOLDS = None   # optional manual fallback, e.g. [0.43, 0.47, 0.45, 0.41, 0.46]
DEFAULT_THRESHOLD = 0.5


# ========== 1. Fix Random Seed ==========
def seed_everything(seed=42):
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


# ========== 3. Target Species Dataset ==========
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
        return (
            torch.tensor(x, dtype=torch.float32).unsqueeze(0),
            torch.tensor(float(row["Label"]), dtype=torch.float32),
        )


# ========== 4. StrucTFactor Model (must match the training script) ==========
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


# ========== 5. Weight and Threshold Loading ==========
def load_model_weight(model, weight_path, device):
    state_dict = torch.load(weight_path, map_location=device)
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    return model


def get_final_threshold():
    """Priority: JSON fold thresholds -> manual FOLD_BEST_THRESHOLDS -> default 0.5."""
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

    dataset = StrucTFactorDataset(CSV_FILE)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    final_threshold = get_final_threshold()

    print("\n[Step 1] Target species data loaded")
    print(f"Number of samples: {len(dataset)}")
    labels_array = dataset.data["Label"].values
    print(f"Positive samples: {int(np.sum(labels_array == 1))}")
    print(f"Negative samples: {int(np.sum(labels_array == 0))}")

    print("\n[Step 2] Loading StrucTFactor 5-fold models for ensemble prediction")
    print(f"Using threshold: {final_threshold:.3f}")

    all_fold_probs = []
    true_labels = None

    for fold_idx, weight_path in enumerate(MODEL_PATHS, start=1):
        if not os.path.exists(weight_path):
            raise FileNotFoundError(f"Weight file not found: {weight_path}")
        print(f"Loading model Fold {fold_idx}: {weight_path}")

        model = StrucTFactorModel().to(device)
        model = load_model_weight(model, weight_path, device)
        model.eval()

        fold_probs, fold_labels = [], []
        with torch.no_grad():
            for x, labels in loader:
                x = x.to(device)
                probs = torch.sigmoid(model(x)).cpu().numpy().reshape(-1)
                fold_probs.extend(probs.tolist())
                fold_labels.extend(labels.numpy().reshape(-1).tolist())

        all_fold_probs.append(np.asarray(fold_probs, dtype=np.float64))
        if true_labels is None:
            true_labels = np.asarray(fold_labels, dtype=int)

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # shape: (5, N)
    all_fold_probs = np.vstack(all_fold_probs)
    ensemble_probs = np.mean(all_fold_probs, axis=0)
    preds = (ensemble_probs > final_threshold).astype(int)

    print("\n[Step 3] StrucTFactor Single-Species Generalization Test Results")
    print("========================================")
    acc = accuracy_score(true_labels, preds)
    auc_value = roc_auc_score(true_labels, ensemble_probs)
    auprc = average_precision_score(true_labels, ensemble_probs)
    mcc = matthews_corrcoef(true_labels, preds)
    print(f"Test Accuracy: {acc * 100:.2f}%")
    print(f"Test AUC:      {auc_value:.4f}")
    print(f"Test AU-PRC:   {auprc:.4f}")
    print(f"Test MCC:      {mcc:.4f}")
    print("========================================")

    # -------- Save per-sequence predictions --------
    result_df = dataset.data.copy()
    for fold_idx in range(len(MODEL_PATHS)):
        result_df[f"Probability_Fold_{fold_idx + 1}"] = all_fold_probs[fold_idx]
    result_df["Ensemble_Probability"] = ensemble_probs
    result_df["Predicted_Label"] = preds

    output_file = "structfactor_single_species_generalization_predictions.csv"
    result_df.to_csv(output_file, index=False)
    print(f"\nPredictions saved to: {output_file}")


if __name__ == "__main__":
    single_species_generalization_test()