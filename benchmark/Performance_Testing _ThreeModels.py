# AU-PRC is computed as the trapezoidal area under the precision-recall curve
# using sklearn.metrics.precision_recall_curve + auc, not average_precision_score.

import argparse, hashlib, json, math, os, random, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, matthews_corrcoef, confusion_matrix,
    precision_recall_curve, auc,
)
from torch.utils.data import Dataset, DataLoader, Subset

warnings.filterwarnings("ignore")


# ========== Utilities ==========
def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


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


# ========== Data validation / shared splits ==========
def read_data(path):
    df = pd.read_csv(path)
    required = ["Protein_ID", "Sequence", "SS3", "Label"]
    if not set(required).issubset(df.columns):
        raise ValueError("CSV requires Protein_ID, Sequence, SS3, Label")
    if df[required].isna().any().any():
        raise ValueError("Missing ID, sequence, SS3, or label")
    if df.Protein_ID.duplicated().any():
        raise ValueError("Duplicate Protein_ID; resolve before splitting")
    if not df.Label.isin([0, 1]).all() or df.Label.nunique() != 2:
        raise ValueError("Labels must contain both 0 and 1")
    # Match the AttentionTFactor vocabulary; fail rather than silently discard residues.
    for row in df.itertuples():
        if not (1 <= len(row.Sequence) <= 1000):
            raise ValueError(f"{row.Protein_ID}: sequence length outside 1..1000")
        if len(row.Sequence) != len(row.SS3):
            raise ValueError(f"{row.Protein_ID}: sequence/SS3 length mismatch")
        if set(row.Sequence) - set("ACDEFGHIKLMNPQRSTVWY"):
            raise ValueError(f"{row.Protein_ID}: unsupported amino acid")
        if set(row.SS3) - set("HECNab"):
            raise ValueError(f"{row.Protein_ID}: unsupported SS3 symbol")
    return df


def make_splits(df, csv_path):
    labels = df.Label.tolist()
    cv_idx, test_idx, cv_labels, _ = train_test_split(
        list(range(len(df))), labels, test_size=0.1, stratify=labels, random_state=1)
    folds = []
    for fold, (tr, va) in enumerate(StratifiedKFold(
            n_splits=5, shuffle=True, random_state=1).split(cv_idx, cv_labels), 1):
        folds.append({"fold": fold, "train": [cv_idx[i] for i in tr],
                      "validation": [cv_idx[i] for i in va]})
    result = {
        "protocol": "AttentionTFactor_10pct_holdout_5fold_v1",
        "csv_sha256": sha256(csv_path), "n_samples": len(df),
        "split_seed": 42, "sklearn_version": sklearn.__version__,
        "protein_ids_in_csv_order": df.Protein_ID.astype(str).tolist(),
        "development": cv_idx, "test": test_idx, "folds": folds,
    }
    validate_splits(df, csv_path, result)
    return result


def validate_splits(df, csv_path, splits):
    if splits["csv_sha256"] != sha256(csv_path):
        raise ValueError("CSV fingerprint differs from saved split file. Use the original CSV; do not reorder it.")
    if splits["protein_ids_in_csv_order"] != df.Protein_ID.astype(str).tolist():
        raise ValueError("Protein order differs from saved splits")
    total = set(range(len(df)))

    def checked(values):
        if any(type(i) is not int for i in values) or len(values) != len(set(values)) or not set(values) <= total:
            raise ValueError("Invalid or duplicate row indices in split file")
        return set(values)

    dev, test = checked(splits["development"]), checked(splits["test"])
    if dev & test or dev | test != total:
        raise ValueError("Development/test partition is invalid")
    if len(splits["folds"]) != 5 or [f["fold"] for f in splits["folds"]] != [1, 2, 3, 4, 5]:
        raise ValueError("Exactly five ordered folds required")
    seen_val = []
    for f in splits["folds"]:
        tr, va = checked(f["train"]), checked(f["validation"])
        if tr & va or tr | va != dev or (tr | va) & test:
            raise ValueError("Train/validation/test overlap or missing samples")
        for a, b in [(tr, va), (tr, test), (va, test)]:
            if set(df.iloc[sorted(a)].Sequence) & set(df.iloc[sorted(b)].Sequence):
                raise ValueError("Identical sequence crosses split boundaries")
        seen_val.extend(f["validation"])
        if any(df.iloc[list(ids)].Label.nunique() != 2 for ids in [tr, va, test]):
            raise ValueError("A split lacks one class")
    if sorted(seen_val) != sorted(dev):
        raise ValueError("Each development protein must appear in validation exactly once")


def save_split_tables(df, splits, outdir):
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "fixed_splits.json").write_text(json.dumps(splits, indent=2), encoding="utf-8")
    counts = []
    parts = [("test", splits["test"]), ("development", splits["development"])]
    for f in splits["folds"]:
        parts += [(f"fold_{f['fold']}_train", f["train"]),
                  (f"fold_{f['fold']}_validation", f["validation"])]
    for name, idx in parts:
        rows = df.iloc[idx][["Protein_ID", "Label"]].copy()
        rows.insert(0, "row_index", idx)
        rows.to_csv(out / f"{name}.csv", index=False)
        counts.append({"split": name, "total": len(idx),
                       "TF": int(rows.Label.sum()), "Non_TF": int((rows.Label == 0).sum())})
    pd.DataFrame(counts).to_csv(out / "split_summary.csv", index=False)


# ========== Metrics / threshold ==========
def select_threshold(labels, probs):
    labels, probs = np.asarray(labels).ravel(), np.asarray(probs).ravel()
    if (len(labels) == 0 or len(labels) != len(probs)
            or not np.isfinite(probs).all()
            or np.any((probs < 0) | (probs > 1))
            or not np.array_equal(np.unique(labels), [0, 1])):
        raise ValueError("Invalid validation predictions")
    best_threshold, best_mcc = None, float("-inf")
    # Exactly the grid, > comparator and first-maximum tie rule in the supplied AttentionTFactor.
    for threshold in np.arange(0.1, 0.91, 0.01):
        score = matthews_corrcoef(labels, (probs > threshold).astype(int))
        if score > best_mcc:
            best_threshold, best_mcc = float(threshold), float(score)
    return best_threshold, best_mcc


def metrics(labels, probs, threshold):
    labels, probs = np.asarray(labels).ravel(), np.asarray(probs).ravel()
    if len(labels) != len(probs) or not np.isfinite(probs).all() or np.any((probs < 0) | (probs > 1)):
        raise ValueError("Invalid probabilities")
    pred = (probs > threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, pred, labels=[0, 1]).ravel()
    precision, recall, _ = precision_recall_curve(labels, probs)
    auprc = auc(recall, precision)   # trapezoidal AU-PRC
    return {
        "threshold": float(threshold), "AUROC": float(roc_auc_score(labels, probs)),
        "AU-PRC": float(auprc), "MCC": float(matthews_corrcoef(labels, pred)),
        "Accuracy": float(np.mean(pred == labels)),
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
    }


def full_metrics(labels, probabilities, threshold):
    values = metrics(labels, probabilities, threshold)
    tp, fp, fn = values["TP"], values["FP"], values["FN"]
    values["Precision"] = tp / (tp + fp) if tp + fp else 0.0
    values["Recall"] = tp / (tp + fn) if tp + fn else 0.0
    values["n_test"] = len(labels)
    return values


# ========== AttentionTFactor ==========
class TFDatasetFromScratch(Dataset):
    def __init__(self, csv_file, max_len=1000):
        self.data = pd.read_csv(csv_file)
        self.max_len = max_len
        amino_acids = "ACDEFGHIKLMNPQRSTVWY"
        self.seq_vocab = {aa: idx + 1 for idx, aa in enumerate(amino_acids)}
        self.ss3_vocab = {"H": 1, "E": 2, "C": 3, "N": 3, "a": 1, "b": 2}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        raw_seq = str(row["Sequence"])[: self.max_len]
        raw_ss3 = str(row["SS3"])[: self.max_len]
        seq_ids = [self.seq_vocab.get(aa, 0) for aa in raw_seq]
        ss3_ids = [self.ss3_vocab.get(s, 0) for s in raw_ss3]
        seq_padded = seq_ids + [0] * (self.max_len - len(seq_ids))
        ss3_padded = ss3_ids + [0] * (self.max_len - len(ss3_ids))
        return (
            torch.tensor(seq_padded, dtype=torch.long),
            torch.tensor(ss3_padded, dtype=torch.long),
            torch.tensor([row["Label"]], dtype=torch.float32),
        )


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
        return x + self.pe[:, : x.size(1), :]


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


class AttentionTFactorCNN(nn.Module):
    def __init__(self, seq_vocab_size=21, ss3_vocab_size=4, embed_dim=256):
        super().__init__()
        self.seq_embedding = nn.Embedding(seq_vocab_size, embed_dim, padding_idx=0)
        self.ss3_embedding = nn.Embedding(ss3_vocab_size, embed_dim, padding_idx=0)
        in_ch = embed_dim * 2
        out_ch = 64
        self.branch_a = self._make_branch(in_ch, out_ch, kernels=[4, 4, 16])
        self.branch_b = self._make_branch(in_ch, out_ch, kernels=[12, 8, 4])
        self.branch_c = self._make_branch(in_ch, out_ch, kernels=[16, 4, 4])
        cnn_out = out_ch * 3
        self.pos_encoder = PositionalEncoding(d_model=cnn_out)
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
        padding_mask = x_seq.eq(0)   # True = padding
        z = torch.cat([self.seq_embedding(x_seq), self.ss3_embedding(x_ss3)], dim=-1).transpose(1, 2)
        feats = torch.cat([self.branch_a(z), self.branch_b(z), self.branch_c(z)], dim=1)
        trans_in = self.pos_encoder(feats.transpose(1, 2))
        gf = self.transformer(trans_in, src_key_padding_mask=padding_mask)

        # Masked max/average pooling
        mask = padding_mask.unsqueeze(-1).expand_as(gf)
        p1 = gf.masked_fill(mask, -1e9).max(dim=1)[0]
        valid_lens = torch.clamp((~padding_mask).sum(dim=1, keepdim=True).float(), min=1.0)
        p2 = gf.masked_fill(mask, 0.0).sum(dim=1) / valid_lens

        return self.fc(torch.cat([p1, p2], dim=1))


# ========== StrucTFactor ==========
AA_LIST = list("ACDEFGHIKLMNPQRSTVWY") + ["X"]
SS_LIST = ["a", "b", "N"]
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_LIST)}
SS_TO_IDX = {ss: i for i, ss in enumerate(SS_LIST)}
MAX_LEN = 1000
N_FEATURES = 63


def normalize_ss3_char(ch):
    """Map several SS3 notations to a/b/N (helix / beta / coil)."""
    if ch in ["a", "H", "G", "I"]:
        return "a"
    if ch in ["b", "E", "B"]:
        return "b"
    return "N"


def encode_sequence_ss3(sequence, ss3):
    """Encode Sequence + SS3 into a (MAX_LEN, 63) one-hot matrix."""
    sequence, ss3 = str(sequence).strip(), str(ss3).strip()
    x = np.zeros((MAX_LEN, N_FEATURES), dtype=np.float32)
    for i in range(min(len(sequence), len(ss3), MAX_LEN)):
        aa = sequence[i] if sequence[i] in AA_TO_IDX else "X"
        x[i, AA_TO_IDX[aa] * 3 + SS_TO_IDX[normalize_ss3_char(ss3[i])]] = 1.0
    return x


class StrucTFactorDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x = encode_sequence_ss3(row["Sequence"], row["SS3"])
        return (
            torch.tensor(x, dtype=torch.float32).unsqueeze(0),
            torch.tensor(float(row["Label"]), dtype=torch.float32),
        )


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


# ========== DeepTFactor ==========
DEEP_AA_VOCAB = "ACDEFGHIKLMNPQRSTVWXY"
DEEP_AA_TO_IDX = {aa: i for i, aa in enumerate(DEEP_AA_VOCAB)}


class DeepTFactorDataset(Dataset):
    """Official channel order (X before Y); all-zero right padding."""

    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sequence = row["Sequence"]
        if not 1 <= len(sequence) <= MAX_LEN:
            raise ValueError("DeepTFactor sequence length outside 1..1000")
        x = np.zeros((1, MAX_LEN, len(DEEP_AA_VOCAB)), dtype=np.float32)
        channels = [DEEP_AA_TO_IDX[aa] for aa in sequence]
        x[0, np.arange(len(sequence)), channels] = 1.0
        return torch.from_numpy(x), torch.tensor(float(row["Label"]), dtype=torch.float32)


class DeepTFactorCNN(nn.Module):
    """Official CNN topology, with the original state_dict layer names."""

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
    """DeepTFactor with output BatchNorm, returning logits for shared training."""

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


# ========== Method configuration / building ==========
def method_settings(args):
    return {
        "AttentionTFactor": {
            "batch_size": args.att_batch_size, "epochs": args.att_epochs,
            "lr": 3e-4, "weight_decay": 1e-3, "loss": "focal",
            "loss_aggregation": "batch_mean", "gradient_clip": 1.0,
            "scheduler_patience": 4, "scheduler_factor": 0.5, "early_stopping": 10,
        },
        "StrucTFactor": {
            "batch_size": args.struc_batch_size, "epochs": args.struc_epochs,
            "lr": 0.001, "weight_decay": 0.0, "loss": "bce",
            "loss_aggregation": "sample_mean", "gradient_clip": None,
            "scheduler_patience": None, "scheduler_factor": None, "early_stopping": None,
        },
        "DeepTFactor": {
            "batch_size": args.deep_batch_size, "epochs": args.deep_epochs,
            "lr": args.deep_lr, "weight_decay": 0.0, "loss": "bce",
            "loss_aggregation": "sample_mean", "gradient_clip": None,
            "scheduler_patience": None, "scheduler_factor": None, "early_stopping": None,
        },
    }


def build_model(name):
    models = {
        "AttentionTFactor": AttentionTFactorCNN,
        "StrucTFactor": StrucTFactorModel,
        "DeepTFactor": DeepTFactorModel,
    }
    if name not in models:
        raise ValueError(f"Unknown model: {name}")
    return models[name]()


def build_criterion(cfg):
    return FocalLoss(alpha=0.90, gamma=2.0) if cfg["loss"] == "focal" else nn.BCEWithLogitsLoss()


def batch_forward(model, batch, device):
    inputs = [item.to(device) for item in batch[:-1]]
    labels = batch[-1].to(device)
    return model(*inputs), labels


def epoch_pass(model, loader, criterion, device, cfg, optimizer=None):
    training = optimizer is not None
    model.train(training)
    loss_sum, denominator = 0.0, 0
    labels, probabilities = [], []
    with torch.set_grad_enabled(training):
        for batch in loader:
            if training:
                optimizer.zero_grad()
            logits, y = batch_forward(model, batch, device)
            loss = criterion(logits, y)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss; no results will be substituted")
            if training:
                loss.backward()
                if cfg["gradient_clip"] is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip"])
                optimizer.step()
            weight = 1 if cfg["loss_aggregation"] == "batch_mean" else len(y)
            loss_sum += loss.item() * weight
            denominator += weight
            if not training:
                labels.extend(y.detach().cpu().numpy().ravel().tolist())
                probabilities.extend(torch.sigmoid(logits).detach().cpu().numpy().ravel().tolist())
    if denominator == 0:
        raise ValueError("No batches; reduce training batch size")
    return loss_sum / denominator, np.asarray(labels, dtype=int), np.asarray(probabilities)


def make_loader(dataset, indices, cfg, args, device, training=False):
    return DataLoader(Subset(dataset, indices), batch_size=cfg["batch_size"],
                      shuffle=training, drop_last=training,
                      num_workers=args.num_workers, pin_memory=(device.type == "cuda"))


def prediction_frame(df, indices, probabilities, threshold):
    rows = df.iloc[indices][["Protein_ID", "Label"]].copy()
    rows.insert(0, "row_index", indices)
    rows["Pred_Prob"] = probabilities
    rows["Pred_Label_validation_threshold"] = (probabilities > threshold).astype(int)
    return rows


# ========== Training / evaluation ==========
def train_method(name, cfg, dataset, df, splits, args, device, out):
    seed_everything(args.seed)   # independent RNG stream per method
    dest = out / name
    dest.mkdir()
    history, selection = [], []
    criterion = build_criterion(cfg).to(device)

    for fold_data in splits["folds"]:
        fold = fold_data["fold"]
        tr = make_loader(dataset, fold_data["train"], cfg, args, device, True)
        va = make_loader(dataset, fold_data["validation"], cfg, args, device)
        model = build_model(name).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        scheduler = (torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=cfg["scheduler_factor"],
            patience=cfg["scheduler_patience"]) if cfg["scheduler_patience"] is not None else None)

        best_loss, stale, best_record = float("inf"), 0, None
        checkpoint = dest / f"best_model_fold_{fold}.pth"

        for epoch in range(1, cfg["epochs"] + 1):
            train_loss, _, _ = epoch_pass(model, tr, criterion, device, cfg, optimizer)
            val_loss, vy, vp = epoch_pass(model, va, criterion, device, cfg)

            if not np.array_equal(vy, df.iloc[fold_data["validation"]].Label.to_numpy(dtype=int)):
                raise RuntimeError("Validation label/order mismatch")

            threshold, vmcc = select_threshold(vy, vp)
            val_values = metrics(vy, vp, threshold)
            history.append({
                "fold": fold, "epoch": epoch, "train_loss": train_loss,
                "validation_loss": val_loss, "validation_AUROC": val_values["AUROC"],
                "validation_AU-PRC": val_values["AU-PRC"], "validation_MCC": vmcc,
                "validation_threshold": threshold, "lr": optimizer.param_groups[0]["lr"],
            })
            if scheduler is not None:
                scheduler.step(val_loss)

            if val_loss < best_loss:
                best_loss, stale = val_loss, 0
                torch.save(model.state_dict(), checkpoint)
                best_record = {
                    "fold": fold, "best_epoch": epoch, "best_validation_loss": best_loss,
                    "validation_threshold": threshold, "validation_MCC": vmcc,
                    "checkpoint": checkpoint.name,
                }
                prediction_frame(df, fold_data["validation"], vp, threshold).to_csv(
                    dest / f"validation_predictions_fold_{fold}.csv", index=False)
            else:
                stale += 1

            pd.DataFrame(history).to_csv(dest / "training_history.csv", index=False)
            print(f'{name} | fold {fold}/5 | epoch {epoch}/{cfg["epochs"]} | '
                  f'train_loss={train_loss:.5f} val_loss={val_loss:.5f} '
                  f'val_AU-PRC={val_values["AU-PRC"]:.4f} val_MCC={vmcc:.4f} '
                  f'threshold={threshold:.2f}', flush=True)

            if cfg["early_stopping"] is not None and stale >= cfg["early_stopping"]:
                print(f"Early stopping: {name}, fold {fold}", flush=True)
                break

        if best_record is None:
            raise RuntimeError("No checkpoint selected")
        best_record["checkpoint_sha256"] = sha256(checkpoint)
        selection.append(best_record)
        pd.DataFrame(selection).to_csv(dest / "fold_selection.csv", index=False)
        print(f'Selected {name} fold {fold}: epoch {best_record["best_epoch"]}, '
              f'threshold {best_record["validation_threshold"]:.2f}', flush=True)
        del model, optimizer, scheduler
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return selection


def evaluate_method(name, cfg, selection, dataset, df, splits, args, device, out):
    dest = out / name
    loader = make_loader(dataset, splits["test"], cfg, args, device)
    expected_labels = df.iloc[splits["test"]].Label.to_numpy(dtype=int)
    criterion = build_criterion(cfg).to(device)
    probability_columns, results = [], []

    for selected in selection:
        fold, threshold = selected["fold"], selected["validation_threshold"]
        model = build_model(name).to(device)
        model.load_state_dict(torch.load(dest / selected["checkpoint"], map_location=device,
                                         weights_only=True), strict=True)
        _, labels, probabilities = epoch_pass(model, loader, criterion, device, cfg)
        if not np.array_equal(labels, expected_labels):
            raise RuntimeError("Test label/order mismatch")
        probability_columns.append(probabilities)
        prediction_frame(df, splits["test"], probabilities, threshold).to_csv(
            dest / f"test_predictions_fold_{fold}.csv", index=False)
        results.append({
            "model": name, "scope": "single_model", "fold": fold,
            "threshold_rule": "validation_optimized",
            **full_metrics(labels, probabilities, threshold),
        })
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    avg_probs = np.mean(np.stack(probability_columns), axis=0)
    avg_threshold = float(np.mean([f["validation_threshold"] for f in selection]))
    predictions = prediction_frame(df, splits["test"], avg_probs, avg_threshold)
    for fold, p in enumerate(probability_columns, 1):
        predictions[f"Probability_fold_{fold}"] = p
    predictions.to_csv(dest / "ensemble_predictions.csv", index=False)
    results.append({
        "model": name, "scope": "five_model_ensemble", "fold": "ensemble",
        "threshold_rule": "validation_optimized",
        **full_metrics(expected_labels, avg_probs, avg_threshold),
    })
    pd.DataFrame(results).to_csv(dest / "test_metrics.csv", index=False)
    return results, avg_probs, avg_threshold


# ========== Joint run ==========
def run_joint(args):
    out = Path(args.outdir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Output directory is not empty. Choose another --outdir.")
    df = read_data(args.csv)

    if args.splits:
        splits = json.loads(Path(args.splits).read_text(encoding="utf-8"))
        validate_splits(df, args.csv, splits)
    else:
        splits = make_splits(df, args.csv)

    settings = method_settings(args)
    min_train = min(len(f["train"]) for f in splits["folds"])
    for cfg in settings.values():
        if cfg["batch_size"] < 2 or cfg["batch_size"] > min_train or cfg["epochs"] < 1:
            raise ValueError("Invalid epoch count or batch size")
        if not math.isfinite(cfg["lr"]) or cfg["lr"] <= 0:
            raise ValueError("Learning rate must be finite and positive")
    if args.num_workers < 0:
        raise ValueError("num_workers must be nonnegative")

    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu")
                          if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Use a CUDA-enabled environment or --device cpu.")

    att_dataset = TFDatasetFromScratch(args.csv)
    if not att_dataset.data.equals(df):
        raise RuntimeError("AttentionTFactor input differs from shared input")
    datasets = {
        "AttentionTFactor": att_dataset,
        "StrucTFactor": StrucTFactorDataset(df),
        "DeepTFactor": DeepTFactorDataset(df),
    }

    out.mkdir(parents=True, exist_ok=True)
    save_split_tables(df, splits, out / "shared_splits")

    config = {
        "csv_sha256": sha256(args.csv), "script_sha256": sha256(__file__),
        "arguments": vars(args), "model_settings": settings,
        "device": str(device), "torch_version": str(torch.__version__),
        "numpy_version": np.__version__, "pandas_version": pd.__version__,
        "sklearn_version": sklearn.__version__,
        "training_seed_reset_per_method": args.seed,
        "split_seed": splits["split_seed"],
        "model_selection": "minimum own validation loss",
        "threshold_selection": "validation MCC, np.arange(0.1,0.91,0.01), first maximum, strict >",
        "ensemble": "mean five probabilities; mean five validation thresholds",
        "deeptfactor_reference": {
            "repository": "https://bitbucket.org/kaistsystemsbiology/deeptfactor/",
            "commit": "7f1bcb451a38102ad1768be39666508f3288acb5",
            "input": "sequence only, 1000x21 one-hot, ACDEFGHIKLMNPQRSTVWXY",
            "output_adaptation": "retain output BatchNorm; move sigmoid to evaluation",
            "training": "from scratch; benchmark hyperparameters, not original paper protocol",
        },
        "AU-PRC": "trapezoidal area under precision-recall curve (precision_recall_curve + auc)",
        "train_drop_last": True,
        "comparison_scope": "full methods with shared splits and evaluation; training configurations differ",
    }
    (out / "run_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    print(pd.read_csv(out / "shared_splits" / "split_summary.csv").to_string(index=False), flush=True)
    print(f"Using {device}. Training {len(settings)} independent methods, five models each.", flush=True)

    selections = {}
    for name, cfg in settings.items():
        selections[name] = train_method(name, cfg, datasets[name], df, splits, args, device, out)

    # No test predictions were computed before this point.
    all_results = []
    paired = df.iloc[splits["test"]][["Protein_ID", "Label"]].copy()
    paired.insert(0, "row_index", splits["test"])

    for name, cfg in settings.items():
        results, probs, threshold = evaluate_method(
            name, cfg, selections[name], datasets[name], df, splits, args, device, out)
        all_results.extend(results)
        paired[f"{name}_probability"] = probs
        paired[f"{name}_pred_validation_threshold"] = (probs > threshold).astype(int)

    results = pd.DataFrame(all_results)
    results.to_csv(out / "all_test_metrics.csv", index=False)
    ensemble = results[results.scope == "five_model_ensemble"].copy()
    ensemble.to_csv(out / "ensemble_comparison.csv", index=False)
    main = ensemble[ensemble.threshold_rule == "validation_optimized"]
    main.to_csv(out / "main_comparison.csv", index=False)
    paired.to_csv(out / "paired_test_predictions.csv", index=False)
    results[results.scope == "single_model"].groupby(["model", "threshold_rule"])[
        ["AUROC", "AU-PRC", "MCC", "Accuracy"]].agg(["mean", "std"]).to_csv(
        out / "single_model_summary.csv")

    columns = ["model", "threshold", "AUROC", "AU-PRC", "MCC", "Accuracy",
               "Precision", "Recall", "TP", "TN", "FP", "FN", "n_test"]
    print("\nMAIN COMPARISON: each method uses its own validation-selected mean threshold")
    print(main[columns].to_string(index=False))
    print(f"All outputs saved to {out.resolve()}")


# ========== CLI ==========
def parse_args():
    p = argparse.ArgumentParser(
        description="Train three models with shared AttentionTFactor splits and validation-selected thresholds")
    p.add_argument("--csv", default="13_multimodal_dataset.csv")
    p.add_argument("--outdir", default="three_model_benchmark_results-1:3")
    p.add_argument("--splits", default=None, help="Optional existing fixed_splits.json")
    p.add_argument("--device", default="auto", help="auto, cuda, cuda:0, cpu")
    p.add_argument("--seed", type=int, default=42,
                   help="Training seed for each method; split seed stays 42")
    p.add_argument("--att-epochs", type=int, default=40)
    p.add_argument("--struc-epochs", type=int, default=50)
    p.add_argument("--deep-epochs", type=int, default=50)
    p.add_argument("--deep-lr", type=float, default=0.001)
    p.add_argument("--att-batch-size", type=int, default=32)
    p.add_argument("--struc-batch-size", type=int, default=128)
    p.add_argument("--deep-batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    run_joint(parse_args())