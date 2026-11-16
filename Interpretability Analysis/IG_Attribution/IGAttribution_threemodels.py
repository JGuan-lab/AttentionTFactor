#!/usr/bin/env python3
"""
Three-model DBD / non-DBD Integrated Gradients interpretability comparison.

Models (kept identical to the supplied training networks):
  1. AttentionTFactorCNN: Sequence + SS3 dual embedding
  2. StrucTFactorModel:   Sequence x SS3, 63-d joint one-hot
  3. DeepTFactorModel:    Sequence only, 21-d one-hot

Principles:
  - Rebuild OOF/TEST five-fold splits from each training CSV's original row order;
  - Only compare TFs present in all three models with consistent OOF/TEST membership;
  - OOF uses each model's own validation-fold checkpoint; TEST averages five folds;
  - IG targets TF sigmoid probability; zero baseline;
  - Per-model attributions saved separately; joint boxplots and residue plots also exported.

Dependencies: pip install torch numpy pandas matplotlib scikit-learn captum
Run example:  python three_model_dbd_ig.py --split oof --steps 32
Test example: python three_model_dbd_ig.py --split test --steps 8 --limit 3
"""

import argparse, csv, hashlib, json, math, re, time, urllib.error, urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.model_selection import StratifiedKFold, train_test_split

try:
    from captum.attr import IntegratedGradients
except ImportError:
    IntegratedGradients = None

BASE = Path(__file__).resolve().parent
MAX_LEN = 1000
N_SPLITS = 5
AA = {c: i + 1 for i, c in enumerate("ACDEFGHIKLMNPQRSTVWY")}
SS = {"H": 1, "E": 2, "C": 3, "N": 3, "a": 1, "b": 2}
STRUCT_AA = list("ACDEFGHIKLMNPQRSTVWY") + ["X"]
STRUCT_AA_TO_IDX = {aa: i for i, aa in enumerate(STRUCT_AA)}
STRUCT_SS_TO_IDX = {"a": 0, "b": 1, "N": 2}
DEEP_AA_VOCAB = "ACDEFGHIKLMNPQRSTVWXY"
DEEP_AA_TO_IDX = {aa: i for i, aa in enumerate(DEEP_AA_VOCAB)}
MODEL_NAMES = ("AttentionTFactorCNN", "StrucTFactor", "DeepTFactor")
MODEL_COLORS = {"AttentionTFactorCNN": "#4472A0", "StrucTFactor": "#D27A52", "DeepTFactor": "#36927E"}
CHECKPOINT_TEMPLATES = {
    "AttentionTFactorCNN": "best_attentiontfactor_13_model_fold_{}.pth",
    "StrucTFactor": "best_structfactor_13_model_fold_{}.pth",
    "DeepTFactor": "best_deeptfactor_13_model_fold_{}.pth",
}


def resolve(path):
    path = Path(path).expanduser()
    return path if path.is_absolute() else BASE / path


# ================= Model architectures (strict-load compatible) =================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class SEBlock(nn.Module):
    def __init__(self, channel, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False), nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False), nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.fc(self.avg_pool(x).view(b, c)).view(b, c, 1)
        return x * y.expand_as(x)


class AttentionTFactorCNN(nn.Module):
    def __init__(self, seq_vocab_size=21, ss3_vocab_size=4, embed_dim=256):
        super().__init__()
        self.seq_embedding = nn.Embedding(seq_vocab_size, embed_dim, padding_idx=0)
        self.ss3_embedding = nn.Embedding(ss3_vocab_size, embed_dim, padding_idx=0)
        ch = embed_dim * 2
        self.branch_a = self._make_branch(ch, 64, [4, 4, 16])
        self.branch_b = self._make_branch(ch, 64, [12, 8, 4])
        self.branch_c = self._make_branch(ch, 64, [16, 4, 4])
        self.pos_encoder = PositionalEncoding(192)
        layer = nn.TransformerEncoderLayer(d_model=192, nhead=8, dim_feedforward=512,
                                           dropout=0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=1)
        self.fc = nn.Sequential(
            nn.Linear(384, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Dropout(0.5), nn.Linear(128, 1),
        )
        self._init_weights()

    def _make_branch(self, in_c, out_c, kernels):
        layers = []
        for i, k in enumerate(kernels):
            layers += [nn.Conv1d(in_c if i == 0 else out_c, out_c, k, padding="same"),
                       nn.BatchNorm1d(out_c), nn.ReLU()]
        layers += [SEBlock(out_c), nn.Dropout(0.1)]
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")

    def forward_embedded(self, z, padding_mask):
        x = z.transpose(1, 2)
        x = torch.cat([self.branch_a(x), self.branch_b(x), self.branch_c(x)], dim=1)
        x = self.pos_encoder(x.transpose(1, 2))
        x = self.transformer(x, src_key_padding_mask=padding_mask)
        expanded = padding_mask.unsqueeze(-1).expand_as(x)
        max_pool = x.masked_fill(expanded, -1e9).max(dim=1)[0]
        avg_pool = (x.masked_fill(expanded, 0.0).sum(dim=1)
                    / (~padding_mask).sum(dim=1, keepdim=True).float().clamp_min(1.0))
        return self.fc(torch.cat([max_pool, avg_pool], dim=1))

    def forward(self, x_seq, x_ss3):
        mask = x_seq == 0
        z = torch.cat([self.seq_embedding(x_seq), self.ss3_embedding(x_ss3)], dim=-1)
        return self.forward_embedded(z, mask)


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
                    nn.BatchNorm2d(128), nn.ReLU(), nn.Dropout(0.1)))
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


# ================= Data loading, split reconstruction, cross-model matching =================
def load_dataset(path, model_name):
    if not path.exists():
        raise FileNotFoundError(f"{model_name} training CSV not found: {path}")
    df = pd.read_csv(path, dtype={"Protein_ID": str})
    required = {"Sequence", "Label"}
    if model_name != "DeepTFactor":
        required.add("SS3")
    if model_name == "AttentionTFactorCNN":
        required.add("Protein_ID")
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{model_name} CSV missing columns: {missing}")
    if df[list(required)].isna().any().any():
        raise ValueError(f"{model_name} CSV has nulls in required columns")
    labels = pd.to_numeric(df["Label"], errors="raise")
    if not labels.isin([0, 1]).all():
        raise ValueError(f"{model_name} CSV Label must be 0 or 1")
    df["Label"] = labels.astype(int)
    # Row order and index must stay intact: original train_test_split/StratifiedKFold replay depends on it.
    df["Sequence"] = df["Sequence"].astype(str).str.strip()
    if "SS3" in df:
        df["SS3"] = df["SS3"].astype(str).str.strip()
    if "Protein_ID" in df:
        df["Protein_ID"] = df["Protein_ID"].astype(str).str.strip()
        if df["Protein_ID"].eq("").any():
            raise ValueError(f"{model_name} CSV has empty Protein_ID")
    if df["Sequence"].eq("").any():
        raise ValueError(f"{model_name} CSV has empty sequence")
    if model_name == "AttentionTFactorCNN":
        if df["Protein_ID"].duplicated().any():
            raise ValueError("Main CSV has duplicate Protein_ID")
        for row_i, row in df.iterrows():
            if len(row.Sequence) != len(row.SS3):
                raise ValueError(f"Main CSV row {row_i}: Sequence/SS3 length mismatch")
            if any(aa not in AA for aa in row.Sequence):
                raise ValueError(f"Main CSV row {row_i}: unknown amino acid")
            if any(ch not in SS for ch in row.SS3):
                raise ValueError(f"Main CSV row {row_i}: unknown SS3 symbol")
    if model_name == "StrucTFactor":
        if (df["Sequence"].str.len() != df["SS3"].str.len()).any():
            raise ValueError("StrucTFactor CSV has Sequence/SS3 length mismatch")
    if model_name == "DeepTFactor" and (df["Sequence"].str.len() > MAX_LEN).any():
        raise ValueError("DeepTFactor original training requires sequence length <= 1000")
    print(f"[{model_name}] CSV={path.name} | total {len(df)} | TF={sum(df.Label == 1)}")
    return df


def compute_split(df):
    """Exactly replay the split used by the three training scripts."""
    indices = list(range(len(df)))
    y = df["Label"].astype(int).tolist()
    cv_idx, test_idx, cv_labels, _ = train_test_split(
        indices, y, test_size=0.1, stratify=y, random_state=42)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    folds = {}
    for fold, (_, val_pos) in enumerate(skf.split(cv_idx, cv_labels), start=1):
        for pos in val_pos:
            folds[cv_idx[pos]] = fold
    group = {i: "OOF" for i in cv_idx}
    group.update({i: "TEST" for i in test_idx})
    return {"group": group, "fold": folds}


def normalize_ss3_char(ch):
    if ch in ("a", "H", "G", "I"):
        return "a"
    if ch in ("b", "E", "B"):
        return "b"
    return "N"


def normalized_ss3(ss3):
    return "".join(normalize_ss3_char(ch) for ch in ss3)


def get_matching_indices(main_df, other_df, model_name):
    """
    If the files are in the same order (allowing duplicate sequences), align by row.
    Otherwise align by Protein_ID or unique sequence.
    Ambiguous duplicate sequences are skipped, not guessed.
    """
    if (len(main_df) == len(other_df)
            and main_df["Sequence"].tolist() == other_df["Sequence"].tolist()
            and main_df["Label"].tolist() == other_df["Label"].tolist()):
        print(f"[{model_name}] Main CSV and training CSV order and sequences match; using raw row indices")
        return {idx: idx for idx in range(len(main_df))}

    id_lookup = defaultdict(list)
    if "Protein_ID" in other_df:
        for idx, pid in enumerate(other_df["Protein_ID"].tolist()):
            id_lookup[pid].append(idx)
    sequence_lookup = defaultdict(list)
    for idx, seq in enumerate(other_df["Sequence"].tolist()):
        sequence_lookup[seq].append(idx)

    mapping, reasons = {}, Counter()
    for idx, row in main_df.iterrows():
        pid = row.Protein_ID
        matched = []
        candidates = id_lookup.get(pid, [])
        if len(candidates) == 1:
            matched = candidates
        elif len(candidates) > 1:
            matched = [j for j in candidates if other_df.at[j, "Sequence"] == row.Sequence]
        if not matched:
            matched = sequence_lookup.get(row.Sequence, [])
        if len(matched) != 1:
            reasons["missing_or_ambiguous_sequence"] += 1
            continue
        other_idx = matched[0]
        if other_df.at[other_idx, "Sequence"] != row.Sequence:
            reasons["Protein_ID_sequence_mismatch"] += 1
            continue
        if int(other_df.at[other_idx, "Label"]) != int(row.Label):
            reasons["label_mismatch"] += 1
            continue
        mapping[idx] = other_idx

    print(f"[{model_name}] Matched {len(mapping)}/{len(main_df)} by ID / unique sequence; skipped: {dict(reasons)}")
    return mapping


def select_common_tf(datasets, splits, split):
    main = datasets["AttentionTFactorCNN"]
    matching = {
        "AttentionTFactorCNN": {i: i for i in range(len(main))},
        "StrucTFactor": get_matching_indices(main, datasets["StrucTFactor"], "StrucTFactor"),
        "DeepTFactor": get_matching_indices(main, datasets["DeepTFactor"], "DeepTFactor"),
    }
    reasons = Counter()
    selected = []
    for idx, row in main.iterrows():
        if int(row.Label) != 1:
            continue
        if len(row.Sequence) > MAX_LEN:
            reasons["length_gt_1000"] += 1
            continue
        if any(idx not in matching[name] for name in MODEL_NAMES):
            reasons["not_in_all_three_datasets"] += 1
            continue
        source_idx = {name: matching[name][idx] for name in MODEL_NAMES}
        groups = {splits[name]["group"][source_idx[name]] for name in MODEL_NAMES}
        if len(groups) != 1:
            reasons["OOF_TEST_membership_differs"] += 1
            continue
        group = groups.pop()
        if split != "both" and group.lower() != split:
            reasons["not_requested_split"] += 1
            continue
        struct_row = datasets["StrucTFactor"].iloc[source_idx["StrucTFactor"]]
        if normalized_ss3(struct_row.SS3) != normalized_ss3(row.SS3):
            reasons["structure_labels_differ"] += 1
            continue
        selected.append({
            "idx": idx, "id": row.Protein_ID, "seq": row.Sequence,
            "ss3": row.SS3, "struct_ss3": struct_row.SS3, "group": group,
            "source_idx": source_idx,
            "fold": {name: splits[name]["fold"].get(source_idx[name]) for name in MODEL_NAMES},
        })
    print("Common TFs across all three models with comparable splits:", len(selected))
    print("Excluded samples:", dict(reasons))
    return selected


# ================= UniProt DNA-binding annotations, caching, parsing =================
def cache_filename(pid):
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", pid)
    if len(safe) > 100:
        safe = safe[:100]
    return f"{safe}_{hashlib.sha1(pid.encode('utf-8')).hexdigest()[:10]}.json"


def fetch_one(pid, sequence, cache_dir, refresh=False):
    path = cache_dir / cache_filename(pid)
    try:
        if path.exists() and not refresh:
            data = json.loads(path.read_text(encoding="utf-8"))
        else:
            url = f"https://rest.uniprot.org/uniprotkb/{pid}.json"
            data = None
            for attempt in range(5):
                try:
                    req = urllib.request.Request(url, headers={
                        "Accept": "application/json", "User-Agent": "TF-IG-Research/1.0"})
                    with urllib.request.urlopen(req, timeout=30) as resp:
                        data = json.load(resp)
                    path.write_text(json.dumps(data), encoding="utf-8")
                    break
                except urllib.error.HTTPError as e:
                    if e.code in (400, 404, 410):
                        return pid, "", f"http_{e.code}"
                    if attempt == 4:
                        raise
                    time.sleep(min(3 * (attempt + 1), 15))
                except (urllib.error.URLError, TimeoutError, OSError):
                    if attempt == 4:
                        raise
                    time.sleep(min(3 * (attempt + 1), 15))
            if data is None:
                raise RuntimeError("UniProt returned no data")
        api_seq = (data.get("sequence") or {}).get("value", "")
        if not api_seq.startswith(sequence):
            return pid, "", "sequence_mismatch"
        ranges = []
        for feature in data.get("features", []):
            if str(feature.get("type", "")).lower().replace("-", " ").strip() != "dna binding":
                continue
            loc = feature.get("location") or {}
            a, b = loc.get("start") or {}, loc.get("end") or {}
            if a.get("modifier") or b.get("modifier"):
                continue
            x, y = a.get("value"), b.get("value")
            if isinstance(x, int) and isinstance(y, int) and 1 <= x <= y:
                ranges.append((x, y))
        if not ranges:
            return pid, "", "no_DBD_annotation"
        ranges = sorted(set(ranges))
        return pid, ";".join(f"{a}-{b}" for a, b in ranges), "ok"
    except Exception as e:
        return pid, "", "request_error: " + str(e)[:150]


LEGACY_OK_STATUSES = {"ok", "ok_from_legacy"}
LEGACY_EMPTY_STATUSES = {"no_DBD_annotation", "no_valid_legacy_interval"}


def normalize_status(status):
    s = str(status).strip()
    if s in LEGACY_OK_STATUSES:
        return "ok"
    if s in LEGACY_EMPTY_STATUSES:
        return "no_DBD_annotation"
    return s


def ensure_annotations(tf_records, annotations, cache_dir, workers=4, refresh=False, offline=False):
    if annotations.exists():
        old = pd.read_csv(annotations, dtype={"Protein_ID": str})
        required = {"Protein_ID", "DBD_Ranges", "Status"}
        if not required.issubset(old.columns) or old.Protein_ID.duplicated().any():
            raise ValueError(f"Existing DBD annotation file has wrong format: {annotations}. "
                             "Must have unique Protein_ID, DBD_Ranges, Status")
    else:
        old = pd.DataFrame(columns=["Protein_ID", "DBD_Ranges", "Status"])

    records = {}
    if not refresh:
        for row in old.itertuples(index=False):
            if str(row.Status).startswith("request_error"):
                continue   # allow retry
            records[str(row.Protein_ID)] = {
                "Protein_ID": str(row.Protein_ID),
                "DBD_Ranges": row.DBD_Ranges if pd.notna(row.DBD_Ranges) else "",
                "Status": normalize_status(row.Status),
            }

    missing = [(p["id"], p["seq"]) for p in tf_records if p["id"] not in records]
    if missing and offline:
        print(f"Offline mode: {len(missing)} TFs unannotated, skipped; existing entries reused")
    elif missing:
        print(f"Querying UniProt DNA-binding ranges for {len(missing)} TFs")
        cache_dir.mkdir(parents=True, exist_ok=True)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(fetch_one, pid, seq, cache_dir, refresh): pid
                       for pid, seq in missing}
            for i, future in enumerate(as_completed(futures), 1):
                pid, ranges, status = future.result()
                records[pid] = {"Protein_ID": pid, "DBD_Ranges": ranges, "Status": status}
                if i % 25 == 0 or i == len(futures):
                    print(f"  UniProt annotation: {i}/{len(futures)}", flush=True)
        annotations.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(records.values(), columns=["Protein_ID", "DBD_Ranges", "Status"]).to_csv(
            annotations, index=False, encoding="utf-8-sig")
        print("Annotation table saved:", annotations)
    return records


def _parse_range_token(token):
    """Parse a single range token, tolerating several legacy formats. Returns (a,b) or None."""
    token = token.strip()
    if not token:
        return None
    token = token.replace("–", "-").replace("—", "-").replace(":", "-").replace("..", "-")
    if re.fullmatch(r"\d+", token):
        v = int(token)
        return (v, v) if v >= 1 else None
    m = re.fullmatch(r"(\d+)\s*-\s*(\d+)", token)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        return (a, b) if a >= 1 and b >= a else None
    m = re.fullmatch(r"(\d+)\s*(?:to|\.\.|~)\s*(\d+)", token, flags=re.IGNORECASE)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        return (a, b) if a >= 1 and b >= a else None
    return None


def parse_ranges(raw):
    if pd.isna(raw) or not str(raw).strip():
        return []
    ranges = []
    for part in re.split(r"[;,|]", str(raw)):
        if not part.strip():
            continue
        parsed = _parse_range_token(part)
        if parsed is not None:
            ranges.append(parsed)
    return sorted(set(ranges))


def apply_dbd_masks(selected, annotations, limit=0):
    valid, reasons = [], Counter()
    for p in selected:
        ann = annotations.get(p["id"])
        if ann is None:
            reasons["missing_annotation"] += 1
            continue
        status = normalize_status(ann["Status"])
        if status != "ok":
            reasons[status] += 1
            continue
        ranges = parse_ranges(ann["DBD_Ranges"])
        if not ranges:
            reasons["DBD_ranges_unparseable"] += 1
            continue
        mask = np.zeros(len(p["seq"]), dtype=bool)
        for a, b in ranges:
            if a <= len(mask):
                mask[a - 1:min(b, len(mask))] = True
        if not mask.any() or mask.all():
            reasons["invalid_or_no_nonDBD"] += 1
            continue
        p["ranges"] = ranges
        p["mask"] = mask
        valid.append(p)
    print("TFs with usable DBD and non-DBD:", len(valid), "; excluded:", dict(reasons))
    if limit:
        valid = valid[:limit]
        print(f"limit={limit}: only explaining first {len(valid)} proteins")
    if not valid:
        raise RuntimeError("No common TF with valid DBD annotation. Check datasets, splits, annotations.")
    return valid


# ================= Strict checkpoint loading (no retraining) =================
def load_model(name, fold, weights_dir, device):
    filename = CHECKPOINT_TEMPLATES[name].format(fold)
    path = weights_dir / filename
    model = {"AttentionTFactorCNN": AttentionTFactorCNN,
             "StrucTFactor": StrucTFactorModel,
             "DeepTFactor": DeepTFactorModel}[name]()
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        ckpt = ckpt["model_state_dict"]
    try:
        model.load_state_dict(ckpt, strict=True)
    except RuntimeError as e:
        raise RuntimeError(f"{name} fold {fold} weights do not match network: {path}\n{e}") from e
    model.to(device).eval()
    for param in model.parameters():
        param.requires_grad_(False)   # only input grads needed -> saves memory
    return model


def validate_checkpoints(weights_dirs):
    missing = []
    for name in MODEL_NAMES:
        for fold in range(1, 6):
            path = weights_dirs[name] / CHECKPOINT_TEMPLATES[name].format(fold)
            if not path.is_file():
                missing.append(str(path))
    if missing:
        raise FileNotFoundError("Missing pretrained checkpoints (15 .pth required):\n" + "\n".join(missing))


# ================= Input encoding + Integrated Gradients =================
def attention_input(model, p, device):
    seq, ss3 = p["seq"], p["ss3"]
    length = len(seq)
    seq_ids = torch.tensor([[AA[c] for c in seq] + [0] * (MAX_LEN - length)],
                           dtype=torch.long, device=device)
    ss_ids = torch.tensor([[SS[c] for c in ss3] + [0] * (MAX_LEN - length)],
                          dtype=torch.long, device=device)
    mask = seq_ids.eq(0)
    with torch.no_grad():
        x = torch.cat([model.seq_embedding(seq_ids), model.ss3_embedding(ss_ids)], dim=-1)
    return x, mask


def struc_input(p, device):
    x = torch.zeros((1, 1, MAX_LEN, 63), dtype=torch.float32, device=device)
    for i, (aa, ch) in enumerate(zip(p["seq"], p["struct_ss3"])):
        aa_idx = STRUCT_AA_TO_IDX.get(aa, STRUCT_AA_TO_IDX["X"])
        ss_idx = STRUCT_SS_TO_IDX[normalize_ss3_char(ch)]
        x[0, 0, i, aa_idx * 3 + ss_idx] = 1.0
    return x


def deep_input(p, device):
    x = torch.zeros((1, 1, MAX_LEN, 21), dtype=torch.float32, device=device)
    for i, aa in enumerate(p["seq"]):
        x[0, 0, i, DEEP_AA_TO_IDX.get(aa, DEEP_AA_TO_IDX["X"])] = 1.0
    return x


def integrated_gradients(forward_fn, inputs, steps, internal_batch_size):
    """Prefer Captum; fall back to Gauss-Legendre quadrature if unavailable."""
    base = torch.zeros_like(inputs)
    if IntegratedGradients is not None:
        ig = IntegratedGradients(forward_fn)
        attribution, delta = ig.attribute(
            inputs, baselines=base, n_steps=steps,
            internal_batch_size=internal_batch_size,
            return_convergence_delta=True)
        delta = abs(float(delta.detach().reshape(-1)[0].item()))
        return attribution.detach(), delta
    nodes, weights = np.polynomial.legendre.leggauss(steps)
    grad_sum = torch.zeros_like(inputs)
    for alpha, weight in zip((nodes + 1) / 2, weights / 2):
        z = (base + (inputs - base) * float(alpha)).detach().requires_grad_(True)
        output = forward_fn(z)
        grad = torch.autograd.grad(output.sum(), z, retain_graph=False)[0]
        grad_sum += grad.detach() * float(weight)
    result = (inputs - base) * grad_sum
    with torch.no_grad():
        difference = (forward_fn(inputs) - forward_fn(base)).sum()
        delta = abs(float((result.sum() - difference).item()))
    return result, delta


def explain_one(name, model, p, steps, device, internal_batch_size):
    """
    All models explain P(TF) = sigmoid(logit), not one logit and one probability.
    Returns per-residue signed IG; Struc joint one-hot cannot be split into Seq/SS3.
    """
    length = len(p["seq"])
    if name == "AttentionTFactorCNN":
        inputs, mask = attention_input(model, p, device)

        def forward_fn(z):
            return torch.sigmoid(model.forward_embedded(z, mask.expand(z.shape[0], -1)).reshape(-1))

        with torch.no_grad():
            probability = float(forward_fn(inputs).item())
        attr, delta = integrated_gradients(forward_fn, inputs, steps, internal_batch_size)
        arr = attr[0, :length, :].cpu().numpy()
        seq_attr = arr[:, :256].sum(axis=1).astype(np.float32)
        ss_attr = arr[:, 256:].sum(axis=1).astype(np.float32)
        total = (seq_attr + ss_attr).astype(np.float32)
    else:
        inputs = struc_input(p, device) if name == "StrucTFactor" else deep_input(p, device)

        def forward_fn(z):
            return torch.sigmoid(model(z).reshape(-1))

        with torch.no_grad():
            probability = float(forward_fn(inputs).item())
        attr, delta = integrated_gradients(forward_fn, inputs, steps, internal_batch_size)
        total = attr[0, 0, :length, :].sum(dim=-1).cpu().numpy().astype(np.float32)
        seq_attr = total.copy() if name == "DeepTFactor" else None
        ss_attr = None
    return {"total": total, "seq": seq_attr, "ss": ss_attr, "prob": probability, "delta": delta}


# ================= Statistics, per-residue output, joint figures =================
PROTEIN_COLUMNS = [
    "Model", "Protein_ID", "Split", "OOF_Fold", "TF_probability", "DBD_Ranges",
    "DBD_length", "nonDBD_length", "DBD_IG_sum", "nonDBD_IG_sum",
    "DBD_IG_mean", "nonDBD_IG_mean", "DBD_minus_nonDBD_mean", "IG_delta",
]
RESIDUE_COLUMNS = [
    "Model", "Protein_ID", "Split", "Position", "Residue", "SS3", "is_DBD",
    "IG_total", "IG_sequence", "IG_SS3",
]


def setup_plot_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9, "axes.labelsize": 9,
        "axes.titlesize": 10, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 8, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "svg.fonttype": "none", "pdf.fonttype": 42, "axes.unicode_minus": False,
    })


def save_figure(fig, out, name):
    fig.savefig(out / (name + ".svg"), bbox_inches="tight")
    fig.savefig(out / (name + ".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def boxplot_comparison(summary, output, mean=False):
    groups = [g for g in ("OOF", "TEST") if (summary["Split"] == g).any()]
    fig, axes = plt.subplots(1, len(groups), figsize=(7.4 * len(groups), 3.7), squeeze=False)
    col_dbd = "DBD_IG_mean" if mean else "DBD_IG_sum"
    col_non = "nonDBD_IG_mean" if mean else "nonDBD_IG_sum"
    rng = np.random.default_rng(42)
    for ax, group in zip(axes[0], groups):
        sub = summary[summary["Split"] == group]
        for j, name in enumerate(MODEL_NAMES):
            data = sub[sub["Model"] == name]
            x = j * 1.5 + 1.0
            for k, col in enumerate((col_dbd, col_non)):
                values = data[col].to_numpy(dtype=float)
                pos = x + (0.28 if k else -0.28)
                bp = ax.boxplot([values], positions=[pos], widths=0.42,
                                patch_artist=True, showfliers=False,
                                boxprops={"edgecolor": "#333333", "linewidth": 0.9},
                                medianprops={"color": "#222222", "linewidth": 1.5},
                                whiskerprops={"color": "#444444"},
                                capprops={"color": "#444444"})
                bp["boxes"][0].set_facecolor(MODEL_COLORS[name] if k == 0 else "white")
                if mean:
                    scatter_x = pos + rng.uniform(-0.18, 0.18, len(values))
                    ax.scatter(scatter_x, values,
                               c=MODEL_COLORS[name] if k == 0 else "#777777",
                               s=5, alpha=0.25, zorder=1, linewidths=0)
        ax.set_xticks([1, 2.5, 4], ["AttentionTFactorCNN", "StrucTFactor", "DeepTFactor"])
        ax.set_xlim(0.3, 4.7)
        ax.axhline(0, linestyle=(0, (4, 3)), color="#999999", linewidth=0.8)
        ax.set_title(f"{group} (paired proteins N={len(sub) // len(MODEL_NAMES)})")
        ax.set_ylabel("Mean signed IG per residue" if mean else "Signed IG sum")
    fig.legend([Patch(facecolor="#999999", edgecolor="#333333"),
                Patch(facecolor="white", edgecolor="#333333")],
               ["DBD (model color)", "non-DBD (white)"], loc="upper center",
               ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout()
    stem = "ThreeModels_length_normalized" if mean else "ThreeModels_IG_sum"
    save_figure(fig, output, stem)


def paired_difference_plot(summary, output):
    groups = [g for g in ("OOF", "TEST") if (summary["Split"] == g).any()]
    fig, axes = plt.subplots(1, len(groups), figsize=(5 * len(groups), 3.6), squeeze=False)
    rng = np.random.default_rng(17)
    for ax, group in zip(axes[0], groups):
        sub = summary[summary["Split"] == group]
        for j, name in enumerate(MODEL_NAMES, start=1):
            values = sub.loc[sub.Model == name, "DBD_minus_nonDBD_mean"].to_numpy(dtype=float)
            bp = ax.boxplot([values], positions=[j], widths=0.45, patch_artist=True,
                            showfliers=False, boxprops={"edgecolor": "#333333"},
                            medianprops={"color": "#222222", "linewidth": 1.4})
            bp["boxes"][0].set_facecolor(MODEL_COLORS[name])
            ax.scatter(j + rng.uniform(-0.18, 0.18, len(values)), values,
                       s=5, alpha=0.32, color="#333333", linewidths=0)
        ax.axhline(0, color="#999999", linestyle=(0, (4, 3)), linewidth=0.8)
        ax.set_xticks([1, 2, 3], ["AttentionTFactorCNN", "StrucTFactor", "DeepTFactor"])
        ax.set_ylabel("DBD mean IG - non-DBD mean IG")
        ax.set_title(f"{group}: within-protein difference")
    fig.tight_layout()
    save_figure(fig, output, "ThreeModels_paired_difference")


def plot_residue_examples(selected, profiles, summary, output, example_count):
    if example_count <= 0:
        return
    contrast = (summary.groupby("Protein_ID")["DBD_minus_nonDBD_mean"]
                .apply(lambda x: float(np.mean(np.abs(x.to_numpy()))))
                .sort_values(ascending=False))
    selected_map = {p["id"]: p for p in selected}
    for pid in contrast.index[:example_count]:
        p = selected_map[pid]
        fig, axes = plt.subplots(3, 1, figsize=(8.0, 5.5), sharex=True)
        x = np.arange(1, len(p["seq"]) + 1)
        for ax, name in zip(axes, MODEL_NAMES):
            for a, b in p["ranges"]:
                ax.axvspan(a - 0.5, min(b, len(x)) + 0.5,
                           facecolor="#F3B6CB", alpha=0.40, linewidth=0)
            ax.plot(x, profiles[name][p["idx"]], linewidth=0.8, color=MODEL_COLORS[name])
            ax.axhline(0, color="#AAAAAA", linestyle="--", linewidth=0.65)
            ax.set_ylabel("Residue IG")
            ax.set_title(name, loc="left", color=MODEL_COLORS[name], fontsize=9)
        axes[-1].set_xlabel("Residue position")
        fig.suptitle(f"{pid}   ({p['group']}; DBD shaded pink)", fontsize=11)
        fig.tight_layout()
        safe_id = re.sub(r"[^A-Za-z0-9_.-]", "_", pid)[:80]
        save_figure(fig, output, f"residue_compare_{safe_id}")


def run_ig(selected, weights_dirs, device, output, steps, internal_batch_size, example_count):
    proteins = []
    profiles = {name: {} for name in MODEL_NAMES}
    residue_file = output / "residue_IG_scores_three_models.csv"
    with residue_file.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(RESIDUE_COLUMNS)
        for name in MODEL_NAMES:
            accumulated = {}
            print(f"\n========== Model: {name} ==========", flush=True)
            for fold in range(1, 6):
                jobs = [p for p in selected if p["group"] == "TEST" or p["fold"][name] == fold]
                if not jobs:
                    continue
                print(f"{name} | Fold {fold} | {len(jobs)} TFs", flush=True)
                model = load_model(name, fold, weights_dirs[name], device)
                for k, p in enumerate(jobs, 1):
                    result = explain_one(name, model, p, steps, device, internal_batch_size)
                    idx = p["idx"]
                    if idx not in accumulated:
                        accumulated[idx] = {
                            "total": np.zeros(len(p["seq"]), dtype=np.float64),
                            "seq": np.zeros(len(p["seq"]), dtype=np.float64) if name == "AttentionTFactorCNN" else None,
                            "ss": np.zeros(len(p["seq"]), dtype=np.float64) if name == "AttentionTFactorCNN" else None,
                            "prob": 0.0, "delta": 0.0, "count": 0,
                        }
                    agg = accumulated[idx]
                    agg["total"] += result["total"]
                    if name == "AttentionTFactorCNN":
                        agg["seq"] += result["seq"]
                        agg["ss"] += result["ss"]
                    agg["prob"] += result["prob"]
                    agg["delta"] += result["delta"]
                    agg["count"] += 1
                    if k % 10 == 0 or k == len(jobs):
                        print(f"  done {k}/{len(jobs)}", flush=True)
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            for p in selected:
                agg = accumulated.get(p["idx"])
                expected = 5 if p["group"] == "TEST" else 1
                if agg is None or agg["count"] != expected:
                    raise RuntimeError(f"{name} {p['id']} fold count mismatch: expected {expected}")
                n = agg["count"]
                score = (agg["total"] / n).astype(np.float32)
                sequence_score = (agg["seq"] / n).astype(np.float32) if agg["seq"] is not None else None
                ss_score = (agg["ss"] / n).astype(np.float32) if agg["ss"] is not None else None
                mask = p["mask"]
                dbd_mean = float(score[mask].mean())
                non_mean = float(score[~mask].mean())
                proteins.append({
                    "Model": name, "Protein_ID": p["id"], "Split": p["group"],
                    "OOF_Fold": p["fold"][name] if p["group"] == "OOF" else "",
                    "TF_probability": agg["prob"] / n,
                    "DBD_Ranges": ";".join(f"{a}-{b}" for a, b in p["ranges"]),
                    "DBD_length": int(mask.sum()), "nonDBD_length": int((~mask).sum()),
                    "DBD_IG_sum": float(score[mask].sum()),
                    "nonDBD_IG_sum": float(score[~mask].sum()),
                    "DBD_IG_mean": dbd_mean, "nonDBD_IG_mean": non_mean,
                    "DBD_minus_nonDBD_mean": dbd_mean - non_mean,
                    "IG_delta": agg["delta"] / n,
                })
                profiles[name][p["idx"]] = score
                for i, ig_value in enumerate(score):
                    seq_ig = (float(sequence_score[i]) if sequence_score is not None
                              else float(ig_value) if name == "DeepTFactor" else "")
                    ss_ig = float(ss_score[i]) if ss_score is not None else ""
                    writer.writerow([name, p["id"], p["group"], i + 1,
                                     p["seq"][i], p["ss3"][i], int(mask[i]),
                                     float(ig_value), seq_ig, ss_ig])
            print(f"{name} done: {len(selected)} proteins", flush=True)
            del accumulated

    summary = pd.DataFrame(proteins, columns=PROTEIN_COLUMNS)
    summary.to_csv(output / "protein_DBD_IG_scores_three_models.csv",
                   index=False, encoding="utf-8-sig")
    wide = summary.pivot(index=["Protein_ID", "Split"], columns="Model",
                         values=["TF_probability", "DBD_IG_mean", "nonDBD_IG_mean",
                                 "DBD_minus_nonDBD_mean", "DBD_IG_sum", "nonDBD_IG_sum"])
    wide.columns = [f"{metric}_{name}" for metric, name in wide.columns]
    wide.reset_index().to_csv(output / "protein_DBD_IG_scores_wide.csv",
                              index=False, encoding="utf-8-sig")
    setup_plot_style()
    boxplot_comparison(summary, output, mean=False)
    boxplot_comparison(summary, output, mean=True)
    paired_difference_plot(summary, output)
    plot_residue_examples(selected, profiles, summary, output, example_count)
    return summary


# ================= Main =================
def main():
    parser = argparse.ArgumentParser(description="Three-model DBD/non-DBD five-fold IG comparison")
    parser.add_argument("--csv", default="13_multimodal_dataset.csv", help="Attention training CSV (main data)")
    parser.add_argument("--struct-csv", default="13_multimodal_dataset.csv")
    parser.add_argument("--deep-csv", default="13_multimodal_dataset.csv")
    parser.add_argument("--attn-weights-dir", default=".")
    parser.add_argument("--struct-weights-dir", default=".")
    parser.add_argument("--deep-weights-dir", default=".")
    parser.add_argument("--annotations", default="dbd_annotations.csv")
    parser.add_argument("--cache", default="uniprot_cache")
    parser.add_argument("--output", default="three_model_dbd_results")
    parser.add_argument("--split", choices=("oof", "test", "both"), default="oof")
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--ig-internal-batch-size", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0, help="Max TFs with DBD to compute; 0 = all")
    parser.add_argument("--example-count", type=int, default=5)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--refresh-dbds", action="store_true")
    parser.add_argument("--offline", action="store_true", help="Use only local annotations, no network")
    args = parser.parse_args()
    if args.steps < 2 or args.limit < 0 or args.example_count < 0:
        parser.error("--steps >= 2, --limit >= 0, --example-count >= 0")
    if args.ig_internal_batch_size < 1 or not 1 <= args.workers <= 10:
        parser.error("--ig-internal-batch-size >= 1, --workers in 1..10")
    if args.offline and args.refresh_dbds:
        parser.error("--offline and --refresh-dbds cannot be combined")

    torch.manual_seed(42)
    if hasattr(torch.backends, "mha"):
        torch.backends.mha.set_fastpath_enabled(False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}; IG={'Captum' if IntegratedGradients else 'Gauss-Legendre fallback'}")

    csv_paths = {
        "AttentionTFactorCNN": resolve(args.csv),
        "StrucTFactor": resolve(args.struct_csv),
        "DeepTFactor": resolve(args.deep_csv),
    }
    weights_dirs = {
        "AttentionTFactorCNN": resolve(args.attn_weights_dir),
        "StrucTFactor": resolve(args.struct_weights_dir),
        "DeepTFactor": resolve(args.deep_weights_dir),
    }
    validate_checkpoints(weights_dirs)

    datasets = {name: load_dataset(csv_paths[name], name) for name in MODEL_NAMES}
    splits = {name: compute_split(datasets[name]) for name in MODEL_NAMES}
    common = select_common_tf(datasets, splits, args.split)
    if not common:
        raise RuntimeError("No common TF matching requirements across three training datasets; check CSVs and splits")

    ann = ensure_annotations(common, resolve(args.annotations), resolve(args.cache),
                             args.workers, args.refresh_dbds, args.offline)
    selected = apply_dbd_masks(common, ann, args.limit)

    output = resolve(args.output)
    output.mkdir(parents=True, exist_ok=True)
    alignment_rows = []
    for p in selected:
        record = {"Protein_ID": p["id"], "Split": p["group"], "Length": len(p["seq"])}
        for name in MODEL_NAMES:
            record[name + "_row"] = p["source_idx"][name]
            record[name + "_OOF_fold"] = p["fold"][name]
        alignment_rows.append(record)
    pd.DataFrame(alignment_rows).to_csv(output / "sample_alignment.csv",
                                        index=False, encoding="utf-8-sig")

    summary = run_ig(selected, weights_dirs, device, output, args.steps,
                     args.ig_internal_batch_size, args.example_count)

    metadata = {
        "split": args.split, "steps": args.steps,
        "method": "Captum IntegratedGradients" if IntegratedGradients else "Gauss-Legendre fallback",
        "target": "TF sigmoid probability", "baseline": "zero at embedding / one-hot input",
        "n_proteins": len(selected), "n_rows": len(summary),
        "datasets": {name: str(csv_paths[name]) for name in MODEL_NAMES},
        "weights_dirs": {name: str(weights_dirs[name]) for name in MODEL_NAMES},
        "note": "StrucTFactor joint one-hot cannot split Sequence/SS3 attributions. Signed IG may cancel.",
    }
    (output / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n========== Three-model DBD IG analysis complete ==========")
    print("Compared proteins:", len(selected))
    print("Output directory:", output)
    for filename in ("ThreeModels_IG_sum.svg",
                     "ThreeModels_length_normalized.svg",
                     "ThreeModels_paired_difference.svg",
                     "protein_DBD_IG_scores_three_models.csv",
                     "protein_DBD_IG_scores_wide.csv",
                     "residue_IG_scores_three_models.csv",
                     "sample_alignment.csv"):
        print("  -", filename)


if __name__ == "__main__":
    main()