"""DeepTFactor five-fold ensemble 15-mer input-occlusion analysis.

Strictly reuses the DeepTFactor training code:
  - DeepTFactorCNN / DeepTFactorModel structure and state_dict names;
  - DEEP_AA_VOCAB = 'ACDEFGHIKLMNPQRSTVWXY' (case-sensitive) encoding;
  - Single sequence -> (1, 1000, 21); batched -> (B, 1, 1000, 21).

Differences from the AttentionTFactor / StrucTFactor scripts:
  - DeepTFactor does not use SS3; the CSV only needs a Sequence column.
  - Occlusion zeros out all 21 one-hot features of the 15 positions
    (equivalent to a no-AA padding).
  - Original and occluded logits are computed under model.eval(),
    then differences are averaged across the five fold models.

  DeltaLogit = original logit - occluded logit
  Positive-contribution windows are chosen top-K by DeltaLogitMean,
  and overlapping windows are skipped.

Usage:
  1. Place best_deeptfactor_gj_model_fold_1.pth ... fold_5.pth
     in the working directory (or adjust MODEL_PATTERN).
  2. Check CSV_FILE, MIN_TF_PROBABILITY and OCCLUSION_BATCH_SIZE.
  3. python extract_15mers_DeepTFactor.py

Note: the training code rejects sequences > 1000 aa. To match the earlier
occlusion scripts, this script truncates to the first 1000 aa by default and
records WasTruncated in the output. Set LONG_SEQUENCE_POLICY='skip' to strictly
match the training-length restriction.
"""

import glob
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# ========== 1. Basic configuration ==========
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CSV_FILE = "human_dataset.csv"
MODEL_PATTERN = "best_deeptfactor_human_model_fold_*.pth"

OUTPUT_ALL_WINDOWS = "all_15mer_windows_human_deeptfactor.csv"
OUTPUT_TOP_WINDOWS = "top_15mers_human_deeptfactor.csv"
OUTPUT_FASTA = "top_15mers_human_deeptfactor.fasta"

MAX_LEN = 1000
WINDOW_SIZE = 15
TOP_K = 5
OCCLUSION_BATCH_SIZE = 8          # three-branch CNN is memory-hungry; drop to 4 / 2 / 1 on OOM
MIN_TF_PROBABILITY = 0.3580       # only analyse proteins with ensemble TF prob >= this; 0 = all
LONG_SEQUENCE_POLICY = "truncate" # 'truncate' (legacy behaviour) or 'skip' (strict training length)


# ========== 2. DeepTFactor amino-acid one-hot encoding ==========
# Must exactly match the training order: Y is channel 21, X is channel 20.
DEEP_AA_VOCAB = "ACDEFGHIKLMNPQRSTVWXY"
DEEP_AA_TO_IDX = {aa: i for i, aa in enumerate(DEEP_AA_VOCAB)}
N_FEATURES = len(DEEP_AA_VOCAB)   # 21


def encode_sequence(sequence: str) -> Tuple[torch.Tensor, str, int, bool]:
    """Encode a protein into a (1, 1000, 21) float32 Tensor.

    Same as training: case-sensitive, unknown chars -> X, unused positions -> 0.
    Same as legacy 15-mer scripts: by default truncate to first 1000 aa, and
    explicitly record the truncation flag.

    Returns (input_tensor, analyzed_sequence, original_length, was_truncated).
    """
    sequence = str(sequence).strip()
    original_length = len(sequence)
    if original_length < 1:
        raise ValueError("Sequence must not be empty")
    if LONG_SEQUENCE_POLICY not in ("truncate", "skip"):
        raise ValueError("LONG_SEQUENCE_POLICY must be 'truncate' or 'skip'")
    if original_length > MAX_LEN and LONG_SEQUENCE_POLICY == "skip":
        raise ValueError(f"Length {original_length} exceeds training MAX_LEN={MAX_LEN}")

    analyzed_sequence = sequence[:MAX_LEN]
    was_truncated = original_length > MAX_LEN

    # Exact match with DeepTFactorDataset.__getitem__: x has shape (1, 1000, 21).
    x = np.zeros((1, MAX_LEN, N_FEATURES), dtype=np.float32)
    channels = [DEEP_AA_TO_IDX.get(aa, DEEP_AA_TO_IDX["X"]) for aa in analyzed_sequence]
    x[0, np.arange(len(analyzed_sequence)), channels] = 1.0
    return torch.from_numpy(x), analyzed_sequence, original_length, was_truncated


# ========== 3. DeepTFactor model (identical to training) ==========
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


# ========== 4. Weight loading (state_dict / DataParallel prefix) ==========
def natural_sort_key(path: str):
    return [int(p) if p.isdigit() else p.lower() for p in re.split(r"(\d+)", path)]


def load_model_state(model: nn.Module, model_path: str, device: torch.device) -> None:
    try:
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(model_path, map_location=device)

    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    elif isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]

    if not isinstance(state_dict, dict):
        raise TypeError(f"{model_path} does not contain a valid state_dict")

    state_dict = {(n[len("module."):] if n.startswith("module.") else n): v
                  for n, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)


def load_all_models(model_pattern: str, device: torch.device) -> List[DeepTFactorModel]:
    model_paths = sorted(glob.glob(model_pattern), key=natural_sort_key)
    if not model_paths:
        raise FileNotFoundError(
            f"No DeepTFactor weights found: {model_pattern}. "
            "Run the training script first, or change MODEL_PATTERN.")
    if len(model_paths) != 5:
        print(f"Warning: found {len(model_paths)} checkpoint files; training script expects 5 folds.")

    models = []
    for path in model_paths:
        model = DeepTFactorModel().to(device)
        load_model_state(model, path, device)
        model.eval()   # important: Dropout / BatchNorm behaviour must match across original and occluded
        models.append(model)
        print(f"Loaded model: {path}")
    return models


# ========== 5. Input occlusion for DeepTFactor ==========
def predict_with_input_occlusion(
    model: DeepTFactorModel,
    input_batch: torch.Tensor,
    occlusion_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """input_batch: (B,1,1000,21); mask: (B,1000) bool; True positions are zeroed.

    DeepTFactor has no embedding, so we do not use forward hooks; we simply
    modify a copy of the input.
    """
    if input_batch.ndim != 4 or tuple(input_batch.shape[1:]) != (1, MAX_LEN, N_FEATURES):
        raise ValueError(f"DeepTFactor input must be (B, 1, {MAX_LEN}, {N_FEATURES})")
    if occlusion_mask is not None:
        if tuple(occlusion_mask.shape) != (input_batch.shape[0], MAX_LEN):
            raise ValueError(f"occlusion_mask must be (B, {MAX_LEN})")
        input_batch = input_batch.masked_fill(occlusion_mask[:, None, :, None], 0.0)
    return model(input_batch).reshape(-1)


@torch.no_grad()
def predict_original(
    models: List[DeepTFactorModel], input_tensor: torch.Tensor
) -> Tuple[np.ndarray, np.ndarray]:
    """Return per-fold original logits and sigmoid probabilities, both (num_models,) float32."""
    input_batch = input_tensor.unsqueeze(0).to(DEVICE)
    logits_list, probs_list = [], []
    for model in models:
        logits = predict_with_input_occlusion(model, input_batch)
        logits_list.append(float(logits.item()))
        probs_list.append(float(torch.sigmoid(logits).item()))
    return (np.asarray(logits_list, dtype=np.float32),
            np.asarray(probs_list, dtype=np.float32))


# ========== 6. All 15-mer occlusion contributions ==========
@torch.no_grad()
def calculate_15mer_scores(
    models: List[DeepTFactorModel],
    input_tensor: torch.Tensor,
    sequence_length: int,
    window_size: int = 15,
    batch_size: int = 8,
    original_logits: Optional[np.ndarray] = None,
    original_probabilities: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    if sequence_length < window_size:
        raise ValueError(f"Sequence length {sequence_length} < window size {window_size}")
    if not 1 <= sequence_length <= MAX_LEN:
        raise ValueError("sequence_length out of 1..1000")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if not models:
        raise ValueError("models must not be empty")
    if original_logits is None or original_probabilities is None:
        original_logits, original_probabilities = predict_original(models, input_tensor)

    starts = np.arange(0, sequence_length - window_size + 1)
    occluded_logits = np.empty((len(models), len(starts)), dtype=np.float32)
    base = input_tensor.unsqueeze(0).to(DEVICE)   # (1,1,1000,21)

    for batch_start in range(0, len(starts), batch_size):
        batch_end = min(batch_start + batch_size, len(starts))
        batch_starts = starts[batch_start:batch_end]
        current_size = len(batch_starts)

        # Each batch row corresponds to a different occluded 15-mer.
        batch = base.expand(current_size, -1, -1, -1)
        mask = torch.zeros((current_size, MAX_LEN), dtype=torch.bool, device=DEVICE)
        for i, start in enumerate(batch_starts):
            mask[i, int(start):int(start) + window_size] = True
        masked_batch = batch.masked_fill(mask[:, None, :, None], 0.0)

        # All five folds use the exact same occluded input.
        for fold_index, model in enumerate(models):
            logits = predict_with_input_occlusion(model, masked_batch)
            occluded_logits[fold_index, batch_start:batch_end] = logits.cpu().numpy()

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


# ========== 7. Top-K positive, non-overlapping windows ==========
def select_top_non_overlapping_windows(records: List[Dict], top_k: int = 5) -> List[Dict]:
    sorted_records = sorted(records, key=lambda r: r["DeltaLogitMean"], reverse=True)
    selected = []
    for record in sorted_records:
        if record["DeltaLogitMean"] <= 0:
            continue
        start, end = int(record["Start"]), int(record["End"])
        if any(not (end < int(r["Start"]) or start > int(r["End"])) for r in selected):
            continue
        selected.append(record.copy())
        if len(selected) >= top_k:
            break
    for rank, record in enumerate(selected, start=1):
        record["Rank"] = rank
    return selected


# ========== 8. CSV reading, scoring, and export ==========
BASE_COLUMNS = [
    "ProteinID", "RowIndex", "TrueLabel", "OriginalLength", "AnalyzedLength",
    "WasTruncated", "Start", "End", "Peptide15mer", "SS3Fragment",
    "OriginalLogitMean", "OriginalProbMean", "OriginalProbSD",
    "OccludedLogitMean", "OccludedProbMean", "DeltaLogitMean", "DeltaLogitSD",
    "DeltaProbMean", "DeltaProbSD",
]


def extract_top_15mers() -> None:
    print(f"Device: {DEVICE}")
    models = load_all_models(MODEL_PATTERN, DEVICE)
    dataframe = pd.read_csv(CSV_FILE)
    if "Sequence" not in dataframe.columns:
        raise ValueError("CSV missing required column: Sequence; DeepTFactor does not need SS3")

    possible_id_columns = ["Protein_ID", "ProteinID", "Entry", "Accession", "ID", "Name"]
    id_column = next((c for c in possible_id_columns if c in dataframe.columns), None)

    all_window_records, top_window_records = [], []

    for row_index, row in dataframe.iterrows():
        protein_id = str(row[id_column]) if id_column is not None else f"row_{row_index}"
        if pd.isna(row["Sequence"]) or not str(row["Sequence"]).strip():
            print(f"Skipping {protein_id}: empty Sequence")
            continue

        try:
            input_tensor, analyzed_sequence, original_length, was_truncated = encode_sequence(
                row["Sequence"])
        except ValueError as exc:
            print(f"Skipping {protein_id}: {exc}")
            continue

        analyzed_length = len(analyzed_sequence)
        if analyzed_length < WINDOW_SIZE:
            print(f"Skipping {protein_id}: length < {WINDOW_SIZE}")
            continue
        if was_truncated:
            print(f"Note: {protein_id} original length {original_length}, analysing first {MAX_LEN} aa only")

        # SS3 is only kept for backward-compatible output columns; it does not affect DeepTFactor.
        ss3 = ""
        if "SS3" in dataframe.columns and pd.notna(row["SS3"]):
            ss3 = str(row["SS3"]).strip()[:analyzed_length]

        original_logits, original_probabilities = predict_original(models, input_tensor)
        ensemble_probability = float(original_probabilities.mean())
        ensemble_logit = float(original_logits.mean())
        print(f"Analysing: {protein_id} | length: {analyzed_length} | "
              f"TF probability: {ensemble_probability:.4f}")
        if ensemble_probability < MIN_TF_PROBABILITY:
            print(f"Skipping {protein_id}: ensemble probability below {MIN_TF_PROBABILITY}")
            continue

        results = calculate_15mer_scores(
            models=models, input_tensor=input_tensor,
            sequence_length=analyzed_length, window_size=WINDOW_SIZE,
            batch_size=OCCLUSION_BATCH_SIZE,
            original_logits=original_logits, original_probabilities=original_probabilities)

        protein_records = []
        for j, start0 in enumerate(results["starts"]):
            start0 = int(start0)
            end0 = start0 + WINDOW_SIZE
            fold_delta_logits = results["delta_logits"][:, j]
            fold_delta_probs = results["delta_probabilities"][:, j]
            fold_occluded_logits = results["occluded_logits"][:, j]
            fold_occluded_probs = results["occluded_probabilities"][:, j]

            record = {
                "ProteinID": protein_id,
                "RowIndex": row_index,
                "TrueLabel": row["Label"] if "Label" in dataframe.columns else np.nan,
                "OriginalLength": original_length,
                "AnalyzedLength": analyzed_length,
                "WasTruncated": was_truncated,
                "Start": start0 + 1,   # 1-based inclusive
                "End": end0,
                "Peptide15mer": analyzed_sequence[start0:end0],
                "SS3Fragment": ss3[start0:end0] if ss3 else "",
                "OriginalLogitMean": ensemble_logit,
                "OriginalProbMean": ensemble_probability,
                "OriginalProbSD": float(original_probabilities.std()),
                "OccludedLogitMean": float(fold_occluded_logits.mean()),
                "OccludedProbMean": float(fold_occluded_probs.mean()),
                "DeltaLogitMean": float(fold_delta_logits.mean()),
                "DeltaLogitSD": float(fold_delta_logits.std()),
                "DeltaProbMean": float(fold_delta_probs.mean()),
                "DeltaProbSD": float(fold_delta_probs.std()),
            }
            for fold_number, score in enumerate(fold_delta_logits, start=1):
                record[f"Fold{fold_number}_DeltaLogit"] = float(score)
            protein_records.append(record)

        all_window_records.extend(protein_records)
        top_window_records.extend(select_top_non_overlapping_windows(protein_records, TOP_K))

    # Always write a headered CSV, even if all proteins are filtered out.
    fold_columns = [f"Fold{i}_DeltaLogit" for i in range(1, len(models) + 1)]
    columns = BASE_COLUMNS + fold_columns

    all_df = pd.DataFrame(all_window_records, columns=columns)
    if not all_df.empty:
        all_df = all_df.sort_values(by=["ProteinID", "DeltaLogitMean"], ascending=[True, False])
    all_df.to_csv(OUTPUT_ALL_WINDOWS, index=False, encoding="utf-8-sig")

    top_df = pd.DataFrame(top_window_records, columns=columns + ["Rank"])
    if not top_df.empty:
        top_df = top_df.sort_values(by=["ProteinID", "Rank"], ascending=[True, True])
    top_df.to_csv(OUTPUT_TOP_WINDOWS, index=False, encoding="utf-8-sig")

    with open(OUTPUT_FASTA, "w", encoding="utf-8") as fasta_file:
        for _, record in top_df.iterrows():
            header = (
                f">{record['ProteinID']}"
                f"|rank={int(record['Rank'])}"
                f"|position={int(record['Start'])}-{int(record['End'])}"
                f"|delta_logit={float(record['DeltaLogitMean']):.6f}"
                f"|tf_probability={float(record['OriginalProbMean']):.6f}"
            )
            fasta_file.write(header + "\n" + str(record["Peptide15mer"]) + "\n")

    print("\nExtraction complete.")
    print(f"All windows: {OUTPUT_ALL_WINDOWS} ({len(all_df)} windows)")
    print(f"Top 15-mers: {OUTPUT_TOP_WINDOWS} ({len(top_df)} fragments)")
    print(f"Top 15-mer FASTA: {OUTPUT_FASTA}")


if __name__ == "__main__":
    extract_top_15mers()