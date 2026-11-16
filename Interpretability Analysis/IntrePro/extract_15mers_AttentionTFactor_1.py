import glob
import re
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from main_model_AttentionTFactor import AttentionTFactorCNN


# ========== 1. Basic configuration ==========
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CSV_FILE = "human_dataset.csv"
MODEL_PATTERN = "best_attentiontfactor_human_model_fold_*.pth"
OUTPUT_ALL_WINDOWS = "all_15mer_windows_human_attentiontfactor.csv"
OUTPUT_TOP_WINDOWS = "top_15mers_human_attentiontfactor.csv"
OUTPUT_FASTA = "top_15mers_human_attentiontfactor.fasta"

MAX_LEN = 1000
WINDOW_SIZE = 15
TOP_K = 5
OCCLUSION_BATCH_SIZE = 32        # reduce to 8/16/32 if GPU memory is insufficient
MIN_TF_PROBABILITY = 0.550       # 0 = analyse all proteins


# ========== 2. Encoding dictionaries ==========
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
SEQ_VOCAB = {aa: index + 1 for index, aa in enumerate(AMINO_ACIDS)}
SS3_VOCAB = {"H": 1, "E": 2, "C": 3, "N": 3, "a": 1, "b": 2}


# ========== 3. Sequence encoding ==========
def encode_sequence(sequence: str, ss3: str, max_len: int = 1000):
    """Encode (Sequence, SS3) into padded tensors. Returns tensors plus metadata."""
    sequence = str(sequence).strip().upper()
    ss3 = str(ss3).strip()

    original_length = len(sequence)
    was_truncated = original_length > max_len
    sequence, ss3 = sequence[:max_len], ss3[:max_len]

    if len(ss3) < len(sequence):
        ss3 = ss3 + "N" * (len(sequence) - len(ss3))
    ss3 = ss3[:len(sequence)]

    seq_ids = [SEQ_VOCAB.get(aa, 0) for aa in sequence]
    ss3_ids = [SS3_VOCAB.get(s, 0) for s in ss3]
    seq_ids = seq_ids + [0] * (max_len - len(seq_ids))
    ss3_ids = ss3_ids + [0] * (max_len - len(ss3_ids))

    return (
        torch.tensor(seq_ids, dtype=torch.long),
        torch.tensor(ss3_ids, dtype=torch.long),
        sequence, ss3, original_length, was_truncated,
    )


# ========== 4. Model weight loading ==========
def natural_sort_key(path: str):
    """Avoid fold_10 sorting before fold_2."""
    return [int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", path)]


def load_model_state(model, model_path: str, device: torch.device):
    """Compatible with multiple PyTorch versions and DataParallel checkpoints."""
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


def load_all_models(model_pattern: str, device: torch.device):
    """Load all cross-validation fold models (AttentionTFactorCNN)."""
    model_paths = sorted(glob.glob(model_pattern), key=natural_sort_key)
    if not model_paths:
        raise FileNotFoundError(f"No model files found: {model_pattern}")

    models = []
    for model_path in model_paths:
        model = AttentionTFactorCNN().to(device)
        load_model_state(model, model_path, device)
        model.eval()
        models.append(model)
        print(f"Loaded model: {model_path}")
    return models


# ========== 5. Embedding-level occlusion via forward hooks ==========
def predict_with_embedding_occlusion(
    model,
    seq_batch: torch.Tensor,
    ss3_batch: torch.Tensor,
    occlusion_mask: Optional[torch.Tensor] = None,
):
    """
    Zero out embedding outputs via forward hooks (no model.forward change).
    occlusion_mask: (B, L) bool, True = occlude that residue.
    Returns logits of shape (B,).
    """
    handles = []

    if occlusion_mask is not None:
        if occlusion_mask.shape != seq_batch.shape:
            raise ValueError("occlusion_mask must match seq_batch shape")

        def seq_embedding_hook(module, inputs, output):
            return output.masked_fill(occlusion_mask.unsqueeze(-1), 0.0)

        def ss3_embedding_hook(module, inputs, output):
            return output.masked_fill(occlusion_mask.unsqueeze(-1), 0.0)

        handles.append(model.seq_embedding.register_forward_hook(seq_embedding_hook))
        handles.append(model.ss3_embedding.register_forward_hook(ss3_embedding_hook))

    try:
        logits = model(seq_batch, ss3_batch).squeeze(-1)
    finally:
        for handle in handles:   # must remove, otherwise next prediction is affected
            handle.remove()

    return logits


# ========== 6. Original prediction ==========
@torch.no_grad()
def predict_original(models, seq_tensor: torch.Tensor, ss3_tensor: torch.Tensor):
    """Return per-fold logits and probabilities."""
    seq_batch = seq_tensor.unsqueeze(0).to(DEVICE)
    ss3_batch = ss3_tensor.unsqueeze(0).to(DEVICE)

    fold_logits, fold_probabilities = [], []
    for model in models:
        logits = predict_with_embedding_occlusion(
            model, seq_batch, ss3_batch, occlusion_mask=None)
        fold_logits.append(float(logits.item()))
        fold_probabilities.append(float(torch.sigmoid(logits).item()))

    return (np.asarray(fold_logits, dtype=np.float32),
            np.asarray(fold_probabilities, dtype=np.float32))


# ========== 7. All 15-mer occlusion scores ==========
@torch.no_grad()
def calculate_15mer_scores(
    models,
    seq_tensor: torch.Tensor,
    ss3_tensor: torch.Tensor,
    sequence_length: int,
    window_size: int = 15,
    batch_size: int = 32,
):
    """delta_logit = original_logit - occluded_logit; larger = more supportive of TF."""
    if sequence_length < window_size:
        raise ValueError(f"Sequence length {sequence_length} < window size {window_size}")

    original_logits, original_probabilities = predict_original(models, seq_tensor, ss3_tensor)
    starts = np.arange(0, sequence_length - window_size + 1)

    number_of_models, number_of_windows = len(models), len(starts)
    occluded_logits = np.zeros((number_of_models, number_of_windows), dtype=np.float32)

    for batch_start in range(0, number_of_windows, batch_size):
        batch_end = min(batch_start + batch_size, number_of_windows)
        current_starts = starts[batch_start:batch_end]
        current_batch_size = len(current_starts)

        seq_batch = seq_tensor.unsqueeze(0).repeat(current_batch_size, 1).to(DEVICE)
        ss3_batch = ss3_tensor.unsqueeze(0).repeat(current_batch_size, 1).to(DEVICE)

        occlusion_mask = torch.zeros(
            (current_batch_size, seq_tensor.shape[0]), dtype=torch.bool, device=DEVICE)

        for row_index, window_start in enumerate(current_starts):
            ws = int(window_start)
            occlusion_mask[row_index, ws:ws + window_size] = True

        for model_index, model in enumerate(models):
            logits = predict_with_embedding_occlusion(
                model, seq_batch, ss3_batch, occlusion_mask=occlusion_mask)
            occluded_logits[model_index, batch_start:batch_end] = logits.detach().cpu().numpy()

    occluded_probabilities = 1.0 / (1.0 + np.exp(-occluded_logits))
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


# ========== 8. Top-K non-overlapping windows ==========
def select_top_non_overlapping_windows(records: List[Dict], top_k: int = 5):
    """Select highest-scoring, mutually non-overlapping, positive-contribution windows."""
    sorted_records = sorted(records, key=lambda r: r["DeltaLogitMean"], reverse=True)
    selected = []
    for record in sorted_records:
        if record["DeltaLogitMean"] <= 0:
            continue
        start, end = int(record["Start"]), int(record["End"])
        if any(not (end < int(s["Start"]) or start > int(s["End"])) for s in selected):
            continue
        selected.append(record.copy())
        if len(selected) >= top_k:
            break
    for rank, record in enumerate(selected, start=1):
        record["Rank"] = rank
    return selected


# ========== 9. Main extraction ==========
def extract_top_15mers():
    print(f"Device: {DEVICE}")
    models = load_all_models(MODEL_PATTERN, DEVICE)
    dataframe = pd.read_csv(CSV_FILE)

    missing = {"Sequence", "SS3"} - set(dataframe.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    id_column = next(
        (c for c in ("Protein_ID", "ProteinID", "Entry", "Accession", "ID", "Name")
         if c in dataframe.columns), None)

    all_window_records, top_window_records = [], []

    for row_index, row in dataframe.iterrows():
        protein_id = str(row[id_column]) if id_column is not None else f"row_{row_index}"
        sequence_value, ss3_value = row["Sequence"], row["SS3"]

        if pd.isna(sequence_value):
            print(f"Skipping {protein_id}: empty Sequence")
            continue
        if pd.isna(ss3_value):
            print(f"Skipping {protein_id}: empty SS3")
            continue

        (seq_tensor, ss3_tensor, analyzed_sequence, analyzed_ss3,
         original_length, was_truncated) = encode_sequence(
            sequence_value, ss3_value, max_len=MAX_LEN)

        analyzed_length = len(analyzed_sequence)
        if analyzed_length < WINDOW_SIZE:
            print(f"Skipping {protein_id}: length < {WINDOW_SIZE}")
            continue

        original_logits, original_probabilities = predict_original(
            models, seq_tensor, ss3_tensor)
        ensemble_probability = float(original_probabilities.mean())
        ensemble_logit = float(original_logits.mean())

        print(f"Analysing: {protein_id} | length: {analyzed_length} | "
              f"TF probability: {ensemble_probability:.4f}")

        if ensemble_probability < MIN_TF_PROBABILITY:
            print(f"Skipping {protein_id}: TF probability below {MIN_TF_PROBABILITY}")
            continue

        score_results = calculate_15mer_scores(
            models, seq_tensor, ss3_tensor, analyzed_length,
            window_size=WINDOW_SIZE, batch_size=OCCLUSION_BATCH_SIZE)

        protein_records = []
        for window_index, start_zero_based in enumerate(score_results["starts"]):
            start_zero_based = int(start_zero_based)
            end_zero_based = start_zero_based + WINDOW_SIZE
            start_one_based, end_one_based = start_zero_based + 1, end_zero_based

            peptide = analyzed_sequence[start_zero_based:end_zero_based]
            ss3_fragment = analyzed_ss3[start_zero_based:end_zero_based]

            fold_delta_logits = score_results["delta_logits"][:, window_index]
            fold_delta_probabilities = score_results["delta_probabilities"][:, window_index]
            fold_occluded_logits = score_results["occluded_logits"][:, window_index]
            fold_occluded_probabilities = score_results["occluded_probabilities"][:, window_index]

            record = {
                "ProteinID": protein_id,
                "RowIndex": row_index,
                "TrueLabel": (row["Label"] if "Label" in dataframe.columns else np.nan),
                "OriginalLength": original_length,
                "AnalyzedLength": analyzed_length,
                "WasTruncated": was_truncated,
                "Start": start_one_based,
                "End": end_one_based,
                "Peptide15mer": peptide,
                "SS3Fragment": ss3_fragment,
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

        selected_records = select_top_non_overlapping_windows(protein_records, top_k=TOP_K)
        all_window_records.extend(protein_records)
        top_window_records.extend(selected_records)

    # ---- Save all windows ----
    all_windows_df = pd.DataFrame(all_window_records)
    if not all_windows_df.empty:
        all_windows_df = all_windows_df.sort_values(
            by=["ProteinID", "DeltaLogitMean"], ascending=[True, False])
    all_windows_df.to_csv(OUTPUT_ALL_WINDOWS, index=False, encoding="utf-8-sig")

    # ---- Save Top-K windows ----
    top_windows_df = pd.DataFrame(top_window_records)
    if not top_windows_df.empty:
        top_windows_df = top_windows_df.sort_values(
            by=["ProteinID", "Rank"], ascending=[True, True])
    top_windows_df.to_csv(OUTPUT_TOP_WINDOWS, index=False, encoding="utf-8-sig")

    # ---- Save FASTA ----
    with open(OUTPUT_FASTA, "w", encoding="utf-8") as fasta_file:
        if not top_windows_df.empty:
            for _, record in top_windows_df.iterrows():
                header = (
                    f">{record['ProteinID']}"
                    f"|rank={int(record['Rank'])}"
                    f"|position={int(record['Start'])}-{int(record['End'])}"
                    f"|delta_logit={float(record['DeltaLogitMean']):.6f}"
                    f"|tf_probability={float(record['OriginalProbMean']):.6f}"
                )
                fasta_file.write(header + "\n")
                fasta_file.write(str(record["Peptide15mer"]) + "\n")

    print("\nExtraction complete.")
    print(f"All windows: {OUTPUT_ALL_WINDOWS}")
    print(f"Top 15-mers: {OUTPUT_TOP_WINDOWS}")
    print(f"Top 15-mer FASTA: {OUTPUT_FASTA}")


if __name__ == "__main__":
    extract_top_15mers()