#!/usr/bin/env python3
"""
AttentionTFactor Ablation Benchmark
===================================

Variants
--------
A0  CNN_OneHot
A1  CNN_JointEncoding
A4  AttentionTFactor_NoSE
A5  AttentionTFactor_NoTransformer
A6  AttentionTFactor_NoSS3
A7  AttentionTFactor

All variants share the same splits, seeds, optimizer, loss, scheduler,
early stopping, validation-MCC threshold search, five-fold ensemble,
and held-out test set.

Input CSV columns: Protein_ID, Sequence, SS3, Label
"""

from __future__ import annotations

import argparse, hashlib, json, math, os, random
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import sklearn
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    auc, confusion_matrix, matthews_corrcoef,
    precision_recall_curve, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, Dataset, Subset


# ============================================================
# 0. Global constants
# ============================================================
MAX_LEN = 1000
AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i + 1 for i, aa in enumerate(AA_ALPHABET)}   # 0 = padding
SS3_TO_IDX = {"H": 1, "a": 1, "E": 2, "b": 2, "C": 3, "N": 3}
KERNEL_GROUPS = ((4, 4, 16), (12, 8, 4), (16, 4, 4))

INCREMENTAL_ORDER = ["CNN_OneHot", "CNN_JointEncoding", "AttentionTFactor"]
LEAVE_ONE_OUT_MODELS = [
    "AttentionTFactor_NoSE",
    "AttentionTFactor_NoTransformer",
    "AttentionTFactor_NoSS3",
]


# ============================================================
# 1. Utilities
# ============================================================
def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_empty_output_dir(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {path}\n"
            "Choose a new --outdir or remove the old directory first."
        )


# ============================================================
# 2. Data validation and shared split generation
# ============================================================
def read_data(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = ["Protein_ID", "Sequence", "SS3", "Label"]
    if not set(required).issubset(df.columns):
        raise ValueError("CSV must contain columns: Protein_ID, Sequence, SS3, Label")
    if df[required].isna().any().any():
        raise ValueError("Missing Protein_ID, Sequence, SS3, or Label.")
    if df["Protein_ID"].duplicated().any():
        dup = df.loc[df["Protein_ID"].duplicated(), "Protein_ID"].iloc[0]
        raise ValueError(f"Duplicate Protein_ID detected: {dup}")
    if not df["Label"].isin([0, 1]).all() or df["Label"].nunique() != 2:
        raise ValueError("Label must contain both classes 0 and 1 only.")

    for row in df.itertuples():
        seq, ss3 = str(row.Sequence), str(row.SS3)
        if not (1 <= len(seq) <= MAX_LEN):
            raise ValueError(f"{row.Protein_ID}: sequence length must be 1..{MAX_LEN}, got {len(seq)}.")
        if len(seq) != len(ss3):
            raise ValueError(f"{row.Protein_ID}: Sequence/SS3 length mismatch ({len(seq)} vs {len(ss3)}).")
        bad_aa = set(seq) - set(AA_ALPHABET)
        if bad_aa:
            raise ValueError(f"{row.Protein_ID}: unsupported amino acid(s): {sorted(bad_aa)}")
        bad_ss3 = set(ss3) - set(SS3_TO_IDX)
        if bad_ss3:
            raise ValueError(f"{row.Protein_ID}: unsupported SS3 symbol(s): {sorted(bad_ss3)}")
    return df


def make_splits(df: pd.DataFrame, csv_path, split_seed: int = 1) -> dict:
    """10% held-out test + 5-fold stratified CV on the remaining 90%."""
    labels = df["Label"].to_numpy(dtype=int)
    all_idx = np.arange(len(df))

    dev_idx, test_idx = train_test_split(
        all_idx, test_size=0.10, stratify=labels, random_state=split_seed)
    dev_labels = labels[dev_idx]

    folds = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=split_seed)
    for fold_id, (tr, va) in enumerate(skf.split(dev_idx, dev_labels), start=1):
        folds.append({
            "fold": fold_id,
            "train": [int(i) for i in dev_idx[tr]],
            "validation": [int(i) for i in dev_idx[va]],
        })

    result = {
        "protocol": "AttentionTFactor_10pct_holdout_5fold_ablation_v1",
        "csv_sha256": sha256(csv_path),
        "n_samples": int(len(df)),
        "split_seed": int(split_seed),
        "sklearn_version": sklearn.__version__,
        "protein_ids_in_csv_order": df["Protein_ID"].astype(str).tolist(),
        "development": [int(i) for i in dev_idx.tolist()],
        "test": [int(i) for i in test_idx.tolist()],
        "folds": folds,
    }
    validate_splits(df, csv_path, result)
    return result


def validate_splits(df: pd.DataFrame, csv_path, splits: dict) -> None:
    if splits.get("csv_sha256") != sha256(csv_path):
        raise ValueError("CSV fingerprint differs from the saved split file. "
                         "Use exactly the same CSV file/order as the original split.")
    if splits.get("protein_ids_in_csv_order") != df["Protein_ID"].astype(str).tolist():
        raise ValueError("Protein order differs from the saved split file.")

    total = set(range(len(df)))

    def checked(values, name: str) -> set:
        if any(type(i) is not int for i in values):
            raise ValueError(f"{name}: indices must be Python integers.")
        if len(values) != len(set(values)):
            raise ValueError(f"{name}: duplicate indices detected.")
        if not set(values) <= total:
            raise ValueError(f"{name}: out-of-range indices detected.")
        return set(values)

    dev = checked(splits["development"], "development")
    test = checked(splits["test"], "test")
    if dev & test:
        raise ValueError("Development and test sets overlap.")
    if dev | test != total:
        raise ValueError("Development/test partition does not cover all rows.")

    folds = splits["folds"]
    if len(folds) != 5:
        raise ValueError("Exactly five folds are required.")
    if [f["fold"] for f in folds] != [1, 2, 3, 4, 5]:
        raise ValueError("Fold IDs must be exactly [1,2,3,4,5].")

    seen_validation = []
    for fi in folds:
        fold = fi["fold"]
        train = checked(fi["train"], f"fold_{fold}_train")
        val = checked(fi["validation"], f"fold_{fold}_validation")
        if train & val:
            raise ValueError(f"Fold {fold}: train/validation overlap.")
        if train | val != dev:
            raise ValueError(f"Fold {fold}: train + validation does not equal development.")
        if (train | val) & test:
            raise ValueError(f"Fold {fold}: development overlaps test.")
        for ln, l, rn, r in [
            ("train", train, "validation", val),
            ("train", train, "test", test),
            ("validation", val, "test", test),
        ]:
            if set(df.iloc[sorted(l)]["Sequence"]) & set(df.iloc[sorted(r)]["Sequence"]):
                raise ValueError(f"Fold {fold}: identical sequence crosses {ln}/{rn} boundaries.")
        for sname, ids in [("train", train), ("validation", val), ("test", test)]:
            if df.iloc[list(ids)]["Label"].nunique() != 2:
                raise ValueError(f"Fold {fold}: {sname} contains only one class.")
        seen_validation.extend(fi["validation"])

    if sorted(seen_validation) != sorted(dev):
        raise ValueError("Every development sample must appear in validation exactly once.")


def save_split_tables(df: pd.DataFrame, splits: dict, outdir) -> None:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "fixed_splits.json").write_text(
        json.dumps(splits, ensure_ascii=False, indent=2), encoding="utf-8")

    partitions = [("test", splits["test"]), ("development", splits["development"])]
    for fi in splits["folds"]:
        f = fi["fold"]
        partitions.extend([
            (f"fold_{f}_train", fi["train"]),
            (f"fold_{f}_validation", fi["validation"]),
        ])

    summary = []
    for name, idx in partitions:
        part = df.iloc[idx][["Protein_ID", "Label"]].copy()
        part.insert(0, "row_index", idx)
        part.to_csv(out / f"{name}.csv", index=False)
        summary.append({
            "split": name,
            "total": int(len(idx)),
            "TF": int(part["Label"].sum()),
            "Non_TF": int((part["Label"] == 0).sum()),
        })
    pd.DataFrame(summary).to_csv(out / "split_summary.csv", index=False)


# ============================================================
# 3. Dataset
# ============================================================
class AttentionTFactorDataset(Dataset):
    def __init__(self, df: pd.DataFrame, max_len: int = MAX_LEN):
        self.df = df.reset_index(drop=True)
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        sequence = str(row["Sequence"])[: self.max_len]
        ss3 = str(row["SS3"])[: self.max_len]
        seq_ids = [AA_TO_IDX[aa] for aa in sequence]
        ss3_ids = [SS3_TO_IDX[s] for s in ss3]
        pad = self.max_len - len(sequence)
        seq_ids.extend([0] * pad)
        ss3_ids.extend([0] * pad)
        return (
            torch.tensor(seq_ids, dtype=torch.long),
            torch.tensor(ss3_ids, dtype=torch.long),
            torch.tensor(float(row["Label"]), dtype=torch.float32),
        )


# ============================================================
# 4. Model building blocks
# ============================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = MAX_LEN):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class SEBlock(nn.Module):
    def __init__(self, channel: int, reduction: int = 4):
        super().__init__()
        hidden = max(channel // reduction, 1)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _ = x.shape
        w = self.fc(self.avg_pool(x).view(b, c)).view(b, c, 1)
        return x * w.expand_as(x)


class ConvBranch(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 kernels: Tuple[int, int, int], use_se: bool,
                 se_reduction: int = 4, dropout: float = 0.1):
        super().__init__()
        layers: List[nn.Module] = [
            nn.Conv1d(in_channels, out_channels, kernel_size=kernels[0], padding="same"),
            nn.BatchNorm1d(out_channels), nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=kernels[1], padding="same"),
            nn.BatchNorm1d(out_channels), nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=kernels[2], padding="same"),
            nn.BatchNorm1d(out_channels), nn.ReLU(),
        ]
        if use_se:
            layers.append(SEBlock(out_channels, reduction=se_reduction))
        layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ============================================================
# 5. Configurable AttentionTFactor ablation model
# ============================================================
class AttentionTFactorAblation(nn.Module):
    def __init__(
        self,
        *,
        input_mode: str = "joint_embedding",
        use_ss3: bool = True,
        use_se: bool = True,
        use_transformer: bool = True,
        use_masked_pooling: bool = True,
        embed_dim: int = 256,
        branch_channels: int = 64,
        se_reduction: int = 4,
        transformer_heads: int = 8,
        transformer_ff_dim: int = 512,
        transformer_dropout: float = 0.1,
    ):
        super().__init__()
        if input_mode not in {"onehot_seq", "joint_embedding"}:
            raise ValueError(f"Unsupported input_mode: {input_mode}")
        if input_mode == "onehot_seq" and use_ss3:
            raise ValueError("onehot_seq baseline is sequence-only; use_ss3 must be False.")

        self.input_mode = input_mode
        self.use_ss3 = use_ss3
        self.use_se = use_se
        self.use_transformer = use_transformer
        self.use_masked_pooling = use_masked_pooling

        if input_mode == "joint_embedding":
            self.seq_embedding = nn.Embedding(len(AA_ALPHABET) + 1, embed_dim, padding_idx=0)
            self.ss3_embedding = nn.Embedding(4, embed_dim, padding_idx=0)
            in_channels = embed_dim * 2
        else:
            self.seq_embedding = None
            self.ss3_embedding = None
            in_channels = len(AA_ALPHABET)

        self.branch_a = ConvBranch(in_channels, branch_channels, KERNEL_GROUPS[0], use_se, se_reduction)
        self.branch_b = ConvBranch(in_channels, branch_channels, KERNEL_GROUPS[1], use_se, se_reduction)
        self.branch_c = ConvBranch(in_channels, branch_channels, KERNEL_GROUPS[2], use_se, se_reduction)

        cnn_out = branch_channels * 3
        if use_transformer:
            self.pos_encoder = PositionalEncoding(d_model=cnn_out, max_len=MAX_LEN)
            self.transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=cnn_out, nhead=transformer_heads,
                    dim_feedforward=transformer_ff_dim,
                    dropout=transformer_dropout, batch_first=True),
                num_layers=1,
            )
        else:
            self.pos_encoder = None
            self.transformer = None

        self.classifier = nn.Sequential(
            nn.Linear(cnn_out * 2, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Dropout(0.5), nn.Linear(128, 1),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _encode_inputs(self, x_seq: torch.Tensor, x_ss3: torch.Tensor) -> torch.Tensor:
        if self.input_mode == "onehot_seq":
            one_hot = F.one_hot(x_seq, num_classes=len(AA_ALPHABET) + 1).float()
            one_hot = one_hot[..., 1:]  # drop padding channel -> 20 channels
            return one_hot.transpose(1, 2)

        seq_emb = self.seq_embedding(x_seq)
        ss3_emb = self.ss3_embedding(x_ss3) if self.use_ss3 else torch.zeros_like(seq_emb)
        return torch.cat([seq_emb, ss3_emb], dim=-1).transpose(1, 2)

    @staticmethod
    def _masked_max_mean_pool(features: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        expanded = padding_mask.unsqueeze(-1).expand_as(features)
        max_pool = features.masked_fill(expanded, -1e9).max(dim=1).values
        valid_lens = torch.clamp((~padding_mask).sum(dim=1, keepdim=True).float(), min=1.0)
        mean_pool = features.masked_fill(expanded, 0.0).sum(dim=1) / valid_lens
        return torch.cat([max_pool, mean_pool], dim=1)

    @staticmethod
    def _plain_max_mean_pool(features: torch.Tensor) -> torch.Tensor:
        return torch.cat([features.max(dim=1).values, features.mean(dim=1)], dim=1)

    def forward(self, x_seq: torch.Tensor, x_ss3: torch.Tensor) -> torch.Tensor:
        padding_mask = x_seq.eq(0)
        x = self._encode_inputs(x_seq, x_ss3)

        multi = torch.cat([self.branch_a(x), self.branch_b(x), self.branch_c(x)], dim=1)
        features = multi.transpose(1, 2)

        if self.use_transformer:
            features = self.pos_encoder(features)
            features = self.transformer(features, src_key_padding_mask=padding_mask)

        protein_features = (
            self._masked_max_mean_pool(features, padding_mask)
            if self.use_masked_pooling
            else self._plain_max_mean_pool(features)
        )
        return self.classifier(protein_features).squeeze(1)


# ============================================================
# 6. Ablation registry (A0, A1, A4, A5, A6, A7)
# ============================================================
VARIANTS = OrderedDict([
    ("CNN_OneHot", dict(
        input_mode="onehot_seq", use_ss3=False, use_se=False,
        use_transformer=False, use_masked_pooling=False)),
    ("CNN_JointEncoding", dict(
        input_mode="joint_embedding", use_ss3=True, use_se=False,
        use_transformer=False, use_masked_pooling=False)),
    ("AttentionTFactor_NoSE", dict(
        input_mode="joint_embedding", use_ss3=True, use_se=False,
        use_transformer=True, use_masked_pooling=True)),
    ("AttentionTFactor_NoTransformer", dict(
        input_mode="joint_embedding", use_ss3=True, use_se=True,
        use_transformer=False, use_masked_pooling=True)),
    ("AttentionTFactor_NoSS3", dict(
        input_mode="joint_embedding", use_ss3=False, use_se=True,
        use_transformer=True, use_masked_pooling=True)),
    ("AttentionTFactor", dict(
        input_mode="joint_embedding", use_ss3=True, use_se=True,
        use_transformer=True, use_masked_pooling=True)),
])


def build_model(name: str) -> nn.Module:
    if name not in VARIANTS:
        raise ValueError(f"Unknown model: {name}. Available: {list(VARIANTS.keys())}")
    return AttentionTFactorAblation(**VARIANTS[name])


# ============================================================
# 7. Loss and metrics
# ============================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alpha, self.gamma = alpha, gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pt = torch.exp(-bce)
        alpha_t = targets * self.alpha + (1.0 - targets) * (1.0 - self.alpha)
        return (alpha_t * (1.0 - pt) ** self.gamma * bce).mean()


def select_threshold(labels: np.ndarray, probabilities: np.ndarray) -> Tuple[float, float]:
    labels = np.asarray(labels).ravel()
    probabilities = np.asarray(probabilities).ravel()
    if len(labels) == 0 or len(labels) != len(probabilities):
        raise ValueError("Invalid validation predictions.")
    if not np.isfinite(probabilities).all():
        raise ValueError("Non-finite probabilities detected.")
    if np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("Probabilities must be within [0, 1].")
    if not np.array_equal(np.unique(labels), [0, 1]):
        raise ValueError("Validation labels must contain both 0 and 1.")

    best_t, best_mcc = None, float("-inf")
    # Grid 0.10..0.90, step 0.01, strict '>' comparator, first maximum wins.
    for t in np.arange(0.10, 0.91, 0.01):
        mcc = matthews_corrcoef(labels, (probabilities > t).astype(int))
        if mcc > best_mcc:
            best_t, best_mcc = float(t), float(mcc)
    return best_t, best_mcc


def base_metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict:
    labels = np.asarray(labels).ravel().astype(int)
    probabilities = np.asarray(probabilities).ravel()
    if len(labels) != len(probabilities):
        raise ValueError("labels/probabilities length mismatch.")
    if not np.isfinite(probabilities).all():
        raise ValueError("Non-finite probabilities detected.")
    if np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("Probabilities must be within [0,1].")

    pred = (probabilities > threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, pred, labels=[0, 1]).ravel()

    precision_curve, recall_curve, _ = precision_recall_curve(labels, probabilities)
    auprc = auc(recall_curve, precision_curve)

    return {
        "threshold": float(threshold),
        "AUROC": float(roc_auc_score(labels, probabilities)),
        "AU-PRC": float(auprc),
        "MCC": float(matthews_corrcoef(labels, pred)),
        "Accuracy": float(np.mean(pred == labels)),
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
    }


def full_metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict:
    values = base_metrics(labels, probabilities, threshold)
    tp, fp, fn = values["TP"], values["FP"], values["FN"]
    values["Precision"] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    values["Recall"] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    values["n_test"] = int(len(labels))
    return values


# ============================================================
# 8. DataLoader / forward pass helpers
# ============================================================
def make_loader(dataset, indices, batch_size, num_workers, device, *,
                training: bool, generator_seed: int) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(generator_seed)
    return DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=training,
        drop_last=training,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        generator=generator if training else None,
    )


def epoch_pass(model, loader, criterion, device, *,
               optimizer=None, gradient_clip=1.0):
    training = optimizer is not None
    model.train(training)
    loss_sum, n_batches = 0.0, 0
    labels_all, probs_all = [], []

    with torch.set_grad_enabled(training):
        for x_seq, x_ss3, labels in loader:
            x_seq = x_seq.to(device, non_blocking=True)
            x_ss3 = x_ss3.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(x_seq, x_ss3)
            loss = criterion(logits, labels)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss encountered.")
            if training:
                loss.backward()
                if gradient_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                optimizer.step()
            loss_sum += float(loss.item())
            n_batches += 1
            if not training:
                labels_all.extend(labels.detach().cpu().numpy().ravel().tolist())
                probs_all.extend(torch.sigmoid(logits).detach().cpu().numpy().ravel().tolist())

    if n_batches == 0:
        raise ValueError("No batches were produced. Reduce --batch-size.")
    return (loss_sum / n_batches,
            np.asarray(labels_all, dtype=int),
            np.asarray(probs_all, dtype=float))


def prediction_frame(df, indices, probabilities, threshold) -> pd.DataFrame:
    rows = df.iloc[indices][["Protein_ID", "Label"]].copy()
    rows.insert(0, "row_index", indices)
    rows["Pred_Prob"] = probabilities
    rows["Pred_Label_validation_threshold"] = (probabilities > threshold).astype(int)
    return rows


# ============================================================
# 9. Training one ablation variant
# ============================================================
def train_variant(name, dataset, df, splits, args, device, out_root) -> List[dict]:
    dest = out_root / name
    dest.mkdir(parents=True, exist_ok=False)
    criterion = FocalLoss(args.focal_alpha, args.focal_gamma).to(device)

    history_rows, selection_rows = [], []

    for fi in splits["folds"]:
        fold = int(fi["fold"])
        fold_seed = args.seed + fold
        seed_everything(fold_seed)

        train_loader = make_loader(dataset, fi["train"], args.batch_size, args.num_workers,
                                   device, training=True, generator_seed=fold_seed)
        val_loader = make_loader(dataset, fi["validation"], args.batch_size, args.num_workers,
                                 device, training=False, generator_seed=fold_seed)

        model = build_model(name).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                     weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=args.scheduler_factor,
            patience=args.scheduler_patience)

        best_loss, stale, best_record = float("inf"), 0, None
        checkpoint = dest / f"best_model_fold_{fold}.pth"

        for epoch in range(1, args.epochs + 1):
            train_loss, _, _ = epoch_pass(model, train_loader, criterion, device,
                                          optimizer=optimizer,
                                          gradient_clip=args.gradient_clip)
            val_loss, val_labels, val_probs = epoch_pass(model, val_loader, criterion, device,
                                                          optimizer=None, gradient_clip=None)

            expected = df.iloc[fi["validation"]]["Label"].to_numpy(dtype=int)
            if not np.array_equal(val_labels, expected):
                raise RuntimeError(f"{name}, fold {fold}: validation label/order mismatch.")

            threshold, val_mcc = select_threshold(val_labels, val_probs)
            vm = base_metrics(val_labels, val_probs, threshold)

            history_rows.append({
                "model": name, "fold": fold, "epoch": epoch,
                "train_loss": train_loss, "validation_loss": val_loss,
                "validation_AUROC": vm["AUROC"],
                "validation_AU-PRC": vm["AU-PRC"],
                "validation_MCC": val_mcc,
                "validation_threshold": threshold,
                "lr": optimizer.param_groups[0]["lr"],
            })
            scheduler.step(val_loss)

            if val_loss < best_loss:
                best_loss, stale = float(val_loss), 0
                torch.save(model.state_dict(), checkpoint)
                best_record = {
                    "model": name, "fold": fold, "best_epoch": epoch,
                    "best_validation_loss": best_loss,
                    "validation_threshold": threshold,
                    "validation_AUROC": vm["AUROC"],
                    "validation_AU-PRC": vm["AU-PRC"],
                    "validation_MCC": val_mcc,
                    "checkpoint": checkpoint.name,
                }
                prediction_frame(df, fi["validation"], val_probs, threshold).to_csv(
                    dest / f"validation_predictions_fold_{fold}.csv", index=False)
            else:
                stale += 1

            pd.DataFrame(history_rows).to_csv(dest / "training_history.csv", index=False)
            print(f"{name} | fold {fold}/5 | epoch {epoch}/{args.epochs} | "
                  f"train_loss={train_loss:.5f} val_loss={val_loss:.5f} "
                  f"val_AUROC={vm['AUROC']:.4f} val_AU-PRC={vm['AU-PRC']:.4f} "
                  f"val_MCC={val_mcc:.4f} threshold={threshold:.2f}", flush=True)

            if stale >= args.early_stopping:
                print(f"Early stopping: {name}, fold {fold}, epoch {epoch}.", flush=True)
                break

        if best_record is None:
            raise RuntimeError(f"{name}, fold {fold}: no checkpoint was selected.")

        best_record["checkpoint_sha256"] = sha256(checkpoint)
        selection_rows.append(best_record)
        pd.DataFrame(selection_rows).to_csv(dest / "fold_selection.csv", index=False)

        print(f"Selected {name} fold {fold}: "
              f"epoch={best_record['best_epoch']}, "
              f"threshold={best_record['validation_threshold']:.2f}", flush=True)

        del model, optimizer, scheduler
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return selection_rows


# ============================================================
# 10. Evaluate one variant on the held-out test set
# ============================================================
def evaluate_variant(name, selection_rows, dataset, df, splits, args, device, out_root):
    dest = out_root / name
    test_loader = make_loader(dataset, splits["test"], args.batch_size, args.num_workers,
                              device, training=False, generator_seed=args.seed)
    expected_labels = df.iloc[splits["test"]]["Label"].to_numpy(dtype=int)
    criterion = FocalLoss(args.focal_alpha, args.focal_gamma).to(device)

    fold_probs_list, result_rows = [], []

    for selected in selection_rows:
        fold = int(selected["fold"])
        threshold = float(selected["validation_threshold"])
        model = build_model(name).to(device)
        state = torch.load(dest / selected["checkpoint"], map_location=device, weights_only=True)
        model.load_state_dict(state, strict=True)

        _, labels, probabilities = epoch_pass(model, test_loader, criterion, device,
                                              optimizer=None, gradient_clip=None)
        if not np.array_equal(labels, expected_labels):
            raise RuntimeError(f"{name}, fold {fold}: test label/order mismatch.")

        fold_probs_list.append(probabilities)
        prediction_frame(df, splits["test"], probabilities, threshold).to_csv(
            dest / f"test_predictions_fold_{fold}.csv", index=False)
        result_rows.append({
            "model": name, "scope": "single_model", "fold": fold,
            "threshold_rule": "validation_MCC_optimized",
            **full_metrics(labels, probabilities, threshold),
        })
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    stacked = np.stack(fold_probs_list, axis=0)
    ensemble_prob = stacked.mean(axis=0)
    ensemble_threshold = float(np.mean(
        [float(r["validation_threshold"]) for r in selection_rows]))

    ensemble_pred = prediction_frame(df, splits["test"], ensemble_prob, ensemble_threshold)
    for i, p in enumerate(fold_probs_list, start=1):
        ensemble_pred[f"Probability_fold_{i}"] = p
    ensemble_pred.to_csv(dest / "ensemble_predictions.csv", index=False)

    result_rows.append({
        "model": name, "scope": "five_model_ensemble", "fold": "ensemble",
        "threshold_rule": "validation_MCC_optimized",
        **full_metrics(expected_labels, ensemble_prob, ensemble_threshold),
    })
    pd.DataFrame(result_rows).to_csv(dest / "test_metrics.csv", index=False)

    return result_rows, ensemble_prob, ensemble_threshold


# ============================================================
# 11. Summary tables for the paper
# ============================================================
def create_architecture_summary(selected_models, out_root) -> None:
    rows = []
    for name in selected_models:
        cfg = VARIANTS[name]
        rows.append({
            "model": name,
            "sequence_input": ("20-aa one-hot" if cfg["input_mode"] == "onehot_seq"
                               else "256-d embedding"),
            "SS3": bool(cfg["use_ss3"]),
            "joint_encoding": (cfg["input_mode"] == "joint_embedding" and cfg["use_ss3"]),
            "multi_scale_CNN": True,
            "SE_Attention": bool(cfg["use_se"]),
            "Transformer": bool(cfg["use_transformer"]),
            "Masked_MaxMean_Pooling": bool(cfg["use_masked_pooling"]),
            "loss": "Focal Loss",
        })
    pd.DataFrame(rows).to_csv(out_root / "ablation_architecture_summary.csv", index=False)


def create_ablation_delta_tables(ensemble_df: pd.DataFrame, out_root) -> None:
    metrics_cols = ["AUROC", "AU-PRC", "MCC", "Accuracy", "Precision", "Recall"]
    by_model = ensemble_df.set_index("model")

    # Incremental ablation
    inc_available = [n for n in INCREMENTAL_ORDER if n in by_model.index]
    inc_rows = []
    for i, name in enumerate(inc_available):
        row = {"step": i, "model": name}
        for m in metrics_cols:
            row[m] = float(by_model.loc[name, m])
        if i == 0:
            for m in metrics_cols:
                row[f"delta_{m}_vs_previous"] = np.nan
        else:
            prev = inc_available[i - 1]
            for m in metrics_cols:
                row[f"delta_{m}_vs_previous"] = (
                    float(by_model.loc[name, m]) - float(by_model.loc[prev, m]))
        inc_rows.append(row)
    if inc_rows:
        pd.DataFrame(inc_rows).to_csv(out_root / "incremental_ablation_deltas.csv", index=False)

    # Leave-one-out
    if "AttentionTFactor" in by_model.index:
        full = by_model.loc["AttentionTFactor"]
        loo_rows = []
        for name in LEAVE_ONE_OUT_MODELS:
            if name not in by_model.index:
                continue
            row = {"model": name, "reference": "AttentionTFactor"}
            for m in metrics_cols:
                v = float(by_model.loc[name, m])
                f = float(full[m])
                row[m] = v
                row[f"full_{m}"] = f
                row[f"full_minus_variant_{m}"] = f - v
            loo_rows.append(row)
        if loo_rows:
            pd.DataFrame(loo_rows).to_csv(out_root / "leave_one_out_ablation_deltas.csv", index=False)


# ============================================================
# 12. Main experiment
# ============================================================
def resolve_models(model_arg: str) -> List[str]:
    if model_arg.strip().lower() == "all":
        return list(VARIANTS.keys())
    requested = [item.strip() for item in model_arg.split(",") if item.strip()]
    unknown = [m for m in requested if m not in VARIANTS]
    if unknown:
        raise ValueError(f"Unknown model(s): {unknown}\n"
                         f"Available models: {list(VARIANTS.keys())}")
    req_set = set(requested)
    return [n for n in VARIANTS if n in req_set]


def run(args: argparse.Namespace) -> None:
    out_root = Path(args.outdir)
    ensure_empty_output_dir(out_root)

    df = read_data(args.csv)

    if args.splits:
        split_path = Path(args.splits)
        splits = json.loads(split_path.read_text(encoding="utf-8"))
        validate_splits(df, args.csv, splits)
        split_source = str(split_path.resolve())
    else:
        splits = make_splits(df, args.csv, split_seed=args.split_seed)
        split_source = "generated_in_this_run"

    selected_models = resolve_models(args.models)
    min_train_size = min(len(f["train"]) for f in splits["folds"])

    if args.batch_size < 2:
        raise ValueError("--batch-size must be >= 2.")
    if args.batch_size > min_train_size:
        raise ValueError(f"--batch-size={args.batch_size} exceeds the smallest "
                         f"fold train size={min_train_size}.")
    if args.epochs < 1:
        raise ValueError("--epochs must be >= 1.")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be >= 0.")
    if not math.isfinite(args.lr) or args.lr <= 0:
        raise ValueError("--lr must be finite and > 0.")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable. "
                           "Use --device cpu or a CUDA-enabled environment.")

    dataset = AttentionTFactorDataset(df)
    out_root.mkdir(parents=True, exist_ok=True)
    save_split_tables(df, splits, out_root / "shared_splits")
    create_architecture_summary(selected_models, out_root)

    config = {
        "csv_sha256": sha256(args.csv),
        "script_sha256": sha256(__file__),
        "arguments": vars(args),
        "selected_models": selected_models,
        "variant_definitions": {n: VARIANTS[n] for n in selected_models},
        "device": str(device),
        "torch_version": str(torch.__version__),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "sklearn_version": sklearn.__version__,
        "split_source": split_source,
        "actual_split_seed_if_generated": splits.get("split_seed", None),
        "training_seed_policy": "same fold seed across variants: seed + fold_id",
        "model_selection": "minimum validation loss within each fold",
        "threshold_selection": ("validation MCC maximum over "
                                "np.arange(0.10, 0.91, 0.01); strict > comparator; first maximum wins"),
        "ensemble": ("mean of five fold probabilities; "
                     "mean of five validation-selected thresholds"),
        "AU-PRC": "trapezoidal area under precision-recall curve (precision_recall_curve + auc)",
        "fairness_rule": ("all ablation variants share training configuration; "
                          "only architecture/input switches differ"),
    }
    (out_root / "run_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== Shared split summary ===", flush=True)
    print(pd.read_csv(out_root / "shared_splits" / "split_summary.csv").to_string(index=False),
          flush=True)
    print(f"\nUsing device: {device}", flush=True)
    print("Ablation variants:\n  - " + "\n  - ".join(selected_models), flush=True)

    selections: Dict[str, List[dict]] = {}
    for name in selected_models:
        print(f"\n{'=' * 80}\nTRAINING: {name}\n{'=' * 80}", flush=True)
        selections[name] = train_variant(name, dataset, df, splits, args, device, out_root)

    all_results = []
    paired = df.iloc[splits["test"]][["Protein_ID", "Label"]].copy()
    paired.insert(0, "row_index", splits["test"])

    for name in selected_models:
        print(f"\nEvaluating held-out test set: {name}", flush=True)
        results, probabilities, threshold = evaluate_variant(
            name, selections[name], dataset, df, splits, args, device, out_root)
        all_results.extend(results)
        paired[f"{name}_probability"] = probabilities
        paired[f"{name}_pred"] = (probabilities > threshold).astype(int)

    all_results_df = pd.DataFrame(all_results)
    all_results_df.to_csv(out_root / "all_test_metrics.csv", index=False)

    ensemble_df = all_results_df[all_results_df["scope"] == "five_model_ensemble"].copy()
    order_map = {n: i for i, n in enumerate(selected_models)}
    ensemble_df["_order"] = ensemble_df["model"].map(order_map)
    ensemble_df = ensemble_df.sort_values("_order").drop(columns="_order").reset_index(drop=True)
    ensemble_df.to_csv(out_root / "ensemble_comparison.csv", index=False)
    paired.to_csv(out_root / "paired_test_predictions.csv", index=False)

    single_df = all_results_df[all_results_df["scope"] == "single_model"].copy()
    single_df.groupby("model")[["AUROC", "AU-PRC", "MCC", "Accuracy"]].agg(
        ["mean", "std"]).to_csv(out_root / "single_model_summary.csv")

    create_ablation_delta_tables(ensemble_df, out_root)

    columns = ["model", "threshold", "AUROC", "AU-PRC", "MCC", "Accuracy",
               "Precision", "Recall", "TP", "TN", "FP", "FN", "n_test"]
    print("\n=== MAIN ABLATION COMPARISON (five-fold ensemble, held-out test) ===",
          flush=True)
    print(ensemble_df[columns].to_string(index=False), flush=True)
    print(f"\nAll outputs saved to:\n{out_root.resolve()}", flush=True)


# ============================================================
# 13. CLI
# ============================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Controlled AttentionTFactor ablation benchmark "
                     "with shared 10% holdout + 5-fold development splits."))

    parser.add_argument("--csv", default="13_multimodal_dataset.csv",
                        help="Input CSV containing Protein_ID, Sequence, SS3, Label.")
    parser.add_argument("--outdir", default="attentiontfactor_ablation_results",
                        help="New/empty output directory.")
    parser.add_argument("--splits", default=None,
                        help="Optional existing fixed_splits.json.")
    parser.add_argument("--models", default="all",
                        help=("Comma-separated subset of ablation variants, or 'all'. "
                              f"Available: {','.join(VARIANTS.keys())}"))
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base training seed. Fold seed = seed + fold_id.")
    parser.add_argument("--split-seed", type=int, default=1,
                        help="Used only when --splits is omitted.")

    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--focal-alpha", type=float, default=0.75)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--scheduler-patience", type=int, default=4)
    parser.add_argument("--scheduler-factor", type=float, default=0.5)
    parser.add_argument("--early-stopping", type=int, default=10)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())