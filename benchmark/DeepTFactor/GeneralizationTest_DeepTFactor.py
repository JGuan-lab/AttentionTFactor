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
CSV_FILE = "bacillus_subtilis_multimodal_dataset.csv"
MODEL_PATHS = [f"best_deeptfactor_bacteria_domain_model_fold_{i}.pth" for i in range(1, 6)]
THRESHOLD_FILE = "deeptfactor_bacteria_domain_fold_thresholds.json"
BATCH_SIZE = 128
MAX_LEN = 1000
FOLD_BEST_THRESHOLDS = None
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


# ========== 2. DeepTFactor Input ==========
DEEP_AA_VOCAB = "ACDEFGHIKLMNPQRSTVWXY"
DEEP_AA_TO_IDX = {aa: i for i, aa in enumerate(DEEP_AA_VOCAB)}


# ========== 3. Dataset ==========
class DeepTFactorDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        missing = {"Sequence", "Label"} - set(self.data.columns)
        if missing:
            raise ValueError(f"CSV file is missing required columns: {missing}")
        if not self.data["Label"].isin([0, 1]).all():
            raise ValueError("Label must be 0/1.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        sequence = str(row["Sequence"]).strip()
        if not 1 <= len(sequence) <= MAX_LEN:
            raise ValueError(f"Row {idx}: sequence length must be 1..{MAX_LEN}, got {len(sequence)}")

        x = np.zeros((1, MAX_LEN, len(DEEP_AA_VOCAB)), dtype=np.float32)
        channels = [DEEP_AA_TO_IDX.get(aa, DEEP_AA_TO_IDX["X"]) for aa in sequence]
        x[0, np.arange(len(sequence)), channels] = 1.0

        return torch.from_numpy(x), torch.tensor(float(row["Label"]), dtype=torch.float32)


# ========== 4. Model ==========
class DeepTFactorCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList()
        for kernels in ((4, 4, 16), (12, 8, 4), (16, 4, 4)):
            blocks = []
            for i, kernel in enumerate(kernels):
                blocks.append(nn.Sequential(
                    nn.Conv2d(1 if i == 0 else 128, 128,
                              kernel_size=(kernel, 21 if i == 0 else 1)),
                    nn.BatchNorm2d(128), nn.ReLU(), nn.Dropout(0.1),
                ))
            self.layers.append(nn.Sequential(*blocks))

        self.conv = nn.Conv2d(384, 384, kernel_size=(1, 1))
        self.batchnorm = nn.BatchNorm2d(384)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=(MAX_LEN - 21, 1), stride=1)

    def forward(self, x):
        x = torch.cat([layer(x) for layer in self.layers], dim=1)
        return self.pool(self.relu(self.batchnorm(self.conv(x))))


class DeepTFactorModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn0 = DeepTFactorCNN()
        self.fc1 = nn.Linear(384, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 1)
        self.bn2 = nn.BatchNorm1d(1)
        self.relu = nn.ReLU()
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(module.weight)

    def forward(self, x):
        x = self.cnn0(x).flatten(1)
        x = self.relu(self.bn1(self.fc1(x)))
        return self.bn2(self.fc2(x)).squeeze(1)


# ========== 5. Utility Functions ==========
def load_model_weight(model, weight_path, device):
    state_dict = torch.load(weight_path, map_location=device)
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
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


# ========== 6. Single-Species Generalization Test ==========
def single_species_generalization_test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset = DeepTFactorDataset(CSV_FILE)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    final_threshold = get_final_threshold()

    print("\n[Step 1] Target species data loaded")
    print(f"Number of samples: {len(dataset)}")
    labels_array = dataset.data["Label"].values
    print(f"Positive samples: {int(np.sum(labels_array == 1))}")
    print(f"Negative samples: {int(np.sum(labels_array == 0))}")

    print("\n[Step 2] Loading DeepTFactor 5-fold models for ensemble prediction")
    print(f"Using threshold: {final_threshold:.3f}")

    all_fold_probs = []
    true_labels = None

    for fold_idx, weight_path in enumerate(MODEL_PATHS, start=1):
        if not os.path.exists(weight_path):
            raise FileNotFoundError(f"Weight file not found: {weight_path}")
        print(f"Loading model Fold {fold_idx}: {weight_path}")

        model = DeepTFactorModel().to(device)
        model = load_model_weight(model, weight_path, device)
        model.eval()

        fold_probs, fold_labels = [], []
        with torch.no_grad():
            for x, labels in loader:
                x = x.to(device)
                probs = torch.sigmoid(model(x))
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

    print("\n[Step 3] DeepTFactor Single-Species Generalization Test Results")
    print("========================================")
    print(f"Test Accuracy: {accuracy_score(true_labels, preds) * 100:.2f}%")
    print(f"Test AUC:      {roc_auc_score(true_labels, ensemble_probs):.4f}")
    print(f"Test AU-PRC:   {average_precision_score(true_labels, ensemble_probs):.4f}")
    print(f"Test MCC:      {matthews_corrcoef(true_labels, preds):.4f}")
    print("========================================")

    result_df = dataset.data.copy()
    for fold_idx in range(len(MODEL_PATHS)):
        result_df[f"Probability_Fold_{fold_idx + 1}"] = all_fold_probs[fold_idx]
    result_df["Ensemble_Probability"] = ensemble_probs
    result_df["Predicted_Label"] = preds

    output_file = "deeptfactor_single_species_generalization_predictions.csv"
    result_df.to_csv(output_file, index=False)
    print(f"\nPredictions saved to: {output_file}")


if __name__ == "__main__":
    single_species_generalization_test()