"""StrucTFactor 5-fold ensemble: 15-mer input-occlusion analysis.

StrucTFactor rewrite of the AttentionTFactorCNN 15-mer analysis script.
Uses exactly the same model architecture and Sequence+SS3 encoding as the
StrucTFactor training script.
Occlusion: zero out all 63 features of the selected window (other positions unchanged).

Usage:
  1. Place this script in the same directory as
     best_structfactor_hxx_model_fold_*.pth, or adjust MODEL_PATTERN.
  2. Adjust CSV_FILE and run python extract_15mers_StrucTFactor.py.

Note: the models come from the hxx training run, but the analysed CSV still
uses the legacy gj dataset. Verify that model and dataset match your task.
Weights are required; the AttentionTFactorCNN weights cannot be reused.
"""

import glob
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# ========== 1. Configuration ==========
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Keep the legacy occlusion dataset; switch to hxx-multimodal_dataset.csv if needed.
CSV_FILE = "human_dataset.csv"

# Same as the supplied StrucTFactor training script.
MODEL_PATTERN = "best_structfactor_human_model_fold_*.pth"

OUTPUT_ALL_WINDOWS = "all_15mer_windows_human_structfactor.csv"
OUTPUT_TOP_WINDOWS = "top_15mers_human_structfactor.csv"
OUTPUT_FASTA = "top_15mers_human_structfactor.fasta"

MAX_LEN = 1000                 # strict match with training (affects pooling kernel)
WINDOW_SIZE = 15
TOP_K = 5
OCCLUSION_BATCH_SIZE = 8       # 3-branch Conv2d is memory-hungry; 4/16/32 also fine
MIN_TF_PROBABILITY = 0.3380    # pre-filter on original TF prob; 0 = no filter


# ========== 2. StrucTFactor training-time encoding (identical to training code) ==========
AA_LIST = list("ACDEFGHIKLMNPQRSTVWY") + ["X"]
SS_LIST = ["a", "b", "N"]
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_LIST)}
SS_TO_IDX = {ss: i for i, ss in enumerate(SS_LIST)}
N_FEATURES = 63


def normalize_ss3_char(ch: str) -> str:
    """Must match the training script so inference does not change SS3 semantics."""
    if ch in ["a", "H", "G", "I"]:
        return "a"
    if ch in ["b", "E", "B"]:
        return "b"
    return "N"


def encode_sequence_ss3(sequence: str, ss3: str) -> np.ndarray:
    """Reuse the training input encoding, returning a (1000, 63) float32 array."""
    sequence, ss3 = str(sequence).strip(), str(ss3).strip()
    x = np.zeros((MAX_LEN, N_FEATURES), dtype=np.float32)
    usable_len = min(len(sequence), len(ss3), MAX_LEN)
    for i in range(usable_len):
        aa = sequence[i] if sequence[i] in AA_TO_IDX else "X"
        x[i, AA_TO_IDX[aa] * 3 + SS_TO_IDX[normalize_ss3_char(ss3[i])]] = 1.0
    return x


def encode_sequence(sequence: str, ss3: str) -> Tuple[torch.Tensor, str, str, int, bool]:
    """Return model input (1, 1000, 63) plus valid sequence/SS3 and length info."""
    sequence, ss3 = str(sequence).strip(), str(ss3).strip()
    original_length = len(sequence)
    was_truncated = original_length > MAX_LEN

    # Strictly follow training encoding: residues beyond SS3 coverage stay all-zero,
    # so only scan the intersection to avoid reporting unencoded windows.
    usable_len = min(len(sequence), len(ss3), MAX_LEN)
    analyzed_sequence, analyzed_ss3 = sequence[:usable_len], ss3[:usable_len]

    input_tensor = torch.from_numpy(encode_sequence_ss3(sequence, ss3)).unsqueeze(0)
    return input_tensor, analyzed_sequence, analyzed_ss3, original_length, was_truncated


# ========== 3. StrucTFactor model (identical to training file) ==========
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


# ========== 4. Strict per-fold StrucTFactor weight loading ==========
def natural_sort_key(path: str):
    return [int(p) if p.isdigit() else p.lower() for p in re.split(r"(\d+)", path)]


def load_model_state(model: nn.Module, model_path: str, device: torch.device):
    try:
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(model_path, map_location=device)

    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    new_state_dict = {
        (k[len("module."):] if k.startswith("module.") else k): v
        for k, v in state_dict.items()
    }
    model.load_state_dict(new_state_dict, strict=True)


def load_all_models(model_pattern: str, device: torch.device) -> List[StrucTFactorModel]:
    model_paths = sorted(glob.glob(model_pattern), key=natural_sort_key)
    if not model_paths:
        raise FileNotFoundError(
            f"No StrucTFactor weights found: {model_pattern}. "
            "Run the training script first, or change MODEL_PATTERN.")

    models = []
    for model_path in model_paths:
        model = StrucTFactorModel().to(device)
        load_model_state(model, model_path, device)
        model.eval()   # BatchNorm/Dropout must stay in inference mode
        models.append(model)
        print(f"Loaded StrucTFactor: {model_path}")
    if len(models) != 5:
        print(f"Note: found {len(models)} fold checkpoints (training script expects 5).")
    return models


# ========== 5. Input-level occlusion of all 63 features in a 15-mer ==========
def predict_with_input_occlusion(
    model: StrucTFactorModel,
    input_batch: torch.Tensor,
    occlusion_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """input_batch: (B,1,1000,63); mask: (B,1000) True = zero all 63 features."""
    if input_batch.ndim != 4 or tuple(input_batch.shape[1:]) != (1, MAX_LEN, N_FEATURES):
        raise ValueError(f"Model input must be (B, 1, {MAX_LEN}, {N_FEATURES})")
    if occlusion_mask is not None:
        if occlusion_mask.shape != (input_batch.shape[0], MAX_LEN):
            raise ValueError("occlusion_mask must have shape (B, 1000)")
        input_batch = input_batch.masked_fill(occlusion_mask[:, None, :, None], 0.0)
    return model(input_batch).reshape(-1)


@torch.no_grad()
def predict_original(
    models: List[StrucTFactorModel], input_tensor: torch.Tensor
) -> Tuple[np.ndarray, np.ndarray]:
    input_batch = input_tensor.unsqueeze(0).to(DEVICE)   # (1,1,1000,63)
    fold_logits, fold_probabilities = [], []
    for model in models:
        logits = predict_with_input_occlusion(model, input_batch)
        fold_logits.append(float(logits.item()))
        fold_probabilities.append(float(torch.sigmoid(logits).item()))
    return (np.asarray(fold_logits, dtype=np.float32),
            np.asarray(fold_probabilities, dtype=np.float32))


# ========== 6. All 15-mer sliding-window scores ==========
@torch.no_grad()
def calculate_15mer_scores(
    models: List[StrucTFactorModel],
    input_tensor: torch.Tensor,
    sequence_length: int,
    window_size: int = 15,
    batch_size: int = 8,
    original_logits: Optional[np.ndarray] = None,
    original_probabilities: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    if sequence_length < window_size:
        raise ValueError(f"Sequence length {sequence_length} < window size {window_size}")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if original_logits is None or original_probabilities is None:
        original_logits, original_probabilities = predict_original(models, input_tensor)

    starts = np.arange(sequence_length - window_size + 1)
    occluded_logits = np.zeros((len(models), len(starts)), dtype=np.float32)
    single_input = input_tensor.unsqueeze(0).to(DEVICE)   # (1,1,1000,63)

    for batch_start in range(0, len(starts), batch_size):
        batch_end = min(batch_start + batch_size, len(starts))
        current_starts = starts[batch_start:batch_end]
        current_batch_size = len(current_starts)

        # expand does not copy the input; masked_fill creates a new tensor.
        input_batch = single_input.expand(current_batch_size, -1, -1, -1)
        occlusion_mask = torch.zeros(
            (current_batch_size, MAX_LEN), dtype=torch.bool, device=DEVICE)
        for row_index, start in enumerate(current_starts):
            occlusion_mask[row_index, int(start):int(start) + window_size] = True

        # Share one occluded batch across all folds to reduce copy overhead.
        masked_batch = input_batch.masked_fill(occlusion_mask[:, None, :, None], 0.0)
        for model_index, model in enumerate(models):
            occluded_logits[model_index, batch_start:batch_end] = (
                predict_with_input_occlusion(model, masked_batch).detach().cpu().numpy())

    occluded_probabilities = torch.sigmoid(torch.from_numpy(occluded_logits)).numpy()
    delta_logits = original_logits[:, None] - occluded_logits
    delta_probabilities = original_probabilities[:, None] - occluded_probabilities
    return {
        "starts": starts,
        "original_logits": original_logits,
        "original_probabilities": original_probabilities,
        "occluded_logits": occluded_logits,
        "occluded_probabilities": occluded_probabilities,
        "delta_logits": delta_logits,
        "delta_probabilities": delta_probabilities,
    }


# ========== 7. Top-K non-overlapping positive windows (legacy logic preserved) ==========
def select_top_non_overlapping_windows(records: List[Dict], top_k: int = 5) -> List[Dict]:
    sorted_records = sorted(records, key=lambda r: r["DeltaLogitMean"], reverse=True)
    selected = []
    for record in sorted_records:
        if record["DeltaLogitMean"] <= 0:
            continue
        start, end = int(record["Start"]), int(record["End"])
        if any(not (end < int(item["Start"]) or start > int(item["End"])) for item in selected):
            continue
        selected.append(record.copy())
        if len(selected) >= top_k:
            break
    for rank, record in enumerate(selected, start=1):
        record["Rank"] = rank
    return selected


# ========== 8. Dataset-wide analysis + CSV/FASTA output ==========
def extract_top_15mers():
    print(f"Device: {DEVICE}")
    models = load_all_models(MODEL_PATTERN, DEVICE)
    dataframe = pd.read_csv(CSV_FILE)

    missing_columns = {"Sequence", "SS3"} - set(dataframe.columns)
    if missing_columns:
        raise ValueError(f"CSV missing required columns: {missing_columns}")

    possible_id_columns = ["Protein_ID", "ProteinID", "Entry", "Accession", "ID", "Name"]
    id_column = next((c for c in possible_id_columns if c in dataframe.columns), None)

    all_window_records, top_window_records = [], []

    for row_index, row in dataframe.iterrows():
        protein_id = str(row[id_column]) if id_column is not None else f"row_{row_index}"
        if pd.isna(row["Sequence"]) or pd.isna(row["SS3"]):
            print(f"Skipping {protein_id}: empty Sequence or SS3")
            continue

        (input_tensor, analyzed_sequence, analyzed_ss3,
         original_length, was_truncated) = encode_sequence(row["Sequence"], row["SS3"])
        analyzed_length = len(analyzed_sequence)

        if analyzed_length < WINDOW_SIZE:
            print(f"Skipping {protein_id}: analyzable length < {WINDOW_SIZE}")
            continue

        original_logits, original_probabilities = predict_original(models, input_tensor)
        ensemble_probability = float(original_probabilities.mean())
        ensemble_logit = float(original_logits.mean())

        print(f"Analysing: {protein_id} | length: {analyzed_length} | "
              f"TF probability: {ensemble_probability:.4f}")
        if ensemble_probability < MIN_TF_PROBABILITY:
            print(f"Skipping {protein_id}: TF probability below {MIN_TF_PROBABILITY}")
            continue

        score_results = calculate_15mer_scores(
            models=models, input_tensor=input_tensor,
            sequence_length=analyzed_length, window_size=WINDOW_SIZE,
            batch_size=OCCLUSION_BATCH_SIZE,
            original_logits=original_logits,
            original_probabilities=original_probabilities)

        protein_records = []
        for window_index, start_zero_based in enumerate(score_results["starts"]):
            start_zero_based = int(start_zero_based)
            end_zero_based = start_zero_based + WINDOW_SIZE

            fold_delta_logits = score_results["delta_logits"][:, window_index]
            fold_delta_probabilities = score_results["delta_probabilities"][:, window_index]
            fold_occluded_logits = score_results["occluded_logits"][:, window_index]
            fold_occluded_probabilities = score_results["occluded_probabilities"][:, window_index]

            record = {
                "ProteinID": protein_id,
                "RowIndex": row_index,
                "TrueLabel": row["Label"] if "Label" in dataframe.columns else np.nan,
                "OriginalLength": original_length,
                "AnalyzedLength": analyzed_length,
                "WasTruncated": was_truncated,
                "Start": start_zero_based + 1,   # 1-based inclusive
                "End": end_zero_based,
                "Peptide15mer": analyzed_sequence[start_zero_based:end_zero_based],
                "SS3Fragment": analyzed_ss3[start_zero_based:end_zero_based],
                "OriginalLogitMean": ensemble_logit,
                "OriginalProbMean": ensemble_probability,
                "OriginalProbSD": float(original_probabilities.std()),
                "OccludedLogitMean": float(fold_occluded_logits.mean()),
                "OccludedProbMean": float(fold_occluded_probabilities.mean()),
                "DeltaLogitMean": float(fold_delta_logits.mean()),
                "DeltaLogitSD": float(fold_delta_logits.std()),
                "DeltaProbMean": float(fold_delta_probabilities.mean()),
                "DeltaProbSD": float(fold_delta_probabilities.std()),
            }
            for fold_index, fold_score in enumerate(fold_delta_logits, start=1):
                record[f"Fold{fold_index}_DeltaLogit"] = float(fold_score)
            protein_records.append(record)

        all_window_records.extend(protein_records)
        top_window_records.extend(select_top_non_overlapping_windows(protein_records, TOP_K))

    all_windows_df = pd.DataFrame(all_window_records)
    if not all_windows_df.empty:
        all_windows_df = all_windows_df.sort_values(
            by=["ProteinID", "DeltaLogitMean"], ascending=[True, False])
    all_windows_df.to_csv(OUTPUT_ALL_WINDOWS, index=False, encoding="utf-8-sig")

    top_windows_df = pd.DataFrame(top_window_records)
    if not top_windows_df.empty:
        top_windows_df = top_windows_df.sort_values(
            by=["ProteinID", "Rank"], ascending=[True, True])
    top_windows_df.to_csv(OUTPUT_TOP_WINDOWS, index=False, encoding="utf-8-sig")

    with open(OUTPUT_FASTA, "w", encoding="utf-8") as fasta_file:
        if not top_windows_df.empty:
            for _, record in top_windows_df.iterrows():
                fasta_header = (
                    f">{record['ProteinID']}"
                    f"|rank={int(record['Rank'])}"
                    f"|position={int(record['Start'])}-{int(record['End'])}"
                    f"|delta_logit={float(record['DeltaLogitMean']):.6f}"
                    f"|tf_probability={float(record['OriginalProbMean']):.6f}"
                )
                fasta_file.write(fasta_header + "\n")
                fasta_file.write(str(record["Peptide15mer"]) + "\n")

    print("\nExtraction complete.")
    print(f"All windows: {OUTPUT_ALL_WINDOWS}")
    print(f"Top 15-mers: {OUTPUT_TOP_WINDOWS}")
    print(f"Top 15-mer FASTA: {OUTPUT_FASTA}")


if __name__ == "__main__":
    extract_top_15mers()