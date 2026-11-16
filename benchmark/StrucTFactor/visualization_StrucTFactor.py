# ============================================================
# StrucTFactor
# Five-fold ensemble prediction + sliding-window joint occlusion
# + Top 15-mer candidate motif
# + Signed contribution CSV + publication-ready SVG
# ============================================================

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import csv, json, warnings
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

warnings.filterwarnings("ignore")

# ========== 0. Configuration ==========
MAX_LEN = 1000
N_FEATURES = 63
MODEL_FILES = [f"best_structfactor_bacillus_subtilis_model_fold_{i}.pth" for i in range(1, 6)]
THRESHOLD_FILE = "structfactor_bacillus_subtilis_fold_thresholds.json"
FOLD_BEST_THRESHOLDS = None   # or [0.43, 0.47, 0.45, 0.41, 0.46]
DEFAULT_THRESHOLD = 0.50
WINDOW_SIZE = 5
MOTIF_LEN = 15
OCCLUSION_BATCH_SIZE = 8      # reduce to 1-2 if GPU memory is limited

try:
    font_manager.findfont("Arial", fallback_to_default=False)
    FONT = "Arial"
except Exception:
    FONT = "Liberation Sans"

plt.rcParams.update({
    "font.family": FONT, "font.size": 11,
    "axes.titlesize": 13, "axes.labelsize": 11,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 9.5,
    "svg.fonttype": "none", "axes.linewidth": 1.0,
})


# ========== 1. StrucTFactor input: Sequence + SS3 -> (1000, 63) ==========
AA_LIST = list("ACDEFGHIKLMNPQRSTVWY") + ["X"]
SS_LIST = ["a", "b", "N"]
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_LIST)}
SS_TO_IDX = {ss: i for i, ss in enumerate(SS_LIST)}


def normalize_ss3_char(ch: str) -> str:
    if ch in ["a", "H", "G", "I"]:
        return "a"
    if ch in ["b", "E", "B"]:
        return "b"
    return "N"


def encode_sequence_ss3(sequence: str, ss3: str) -> np.ndarray:
    """Identical encoding semantics to the StrucTFactor training script."""
    sequence, ss3 = str(sequence).strip(), str(ss3).strip()
    x = np.zeros((MAX_LEN, N_FEATURES), dtype=np.float32)
    for i in range(min(len(sequence), len(ss3), MAX_LEN)):
        aa = sequence[i] if sequence[i] in AA_TO_IDX else "X"
        x[i, AA_TO_IDX[aa] * 3 + SS_TO_IDX[normalize_ss3_char(ss3[i])]] = 1.0
    return x


# ========== 2. StrucTFactor model (layer names/parameters identical) ==========
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


# ========== 3. Load five folds / threshold ==========
def load_model_weight(model, weight_path, device):
    try:
        state_dict = torch.load(weight_path, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(weight_path, map_location=device)

    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    if any(k.startswith("module.") for k in state_dict):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    return model


def get_final_threshold():
    if os.path.exists(THRESHOLD_FILE):
        with open(THRESHOLD_FILE, "r", encoding="utf-8") as f:
            info = json.load(f)
        fold_thresholds = info.get("fold_thresholds")
        if fold_thresholds:
            return float(np.mean(fold_thresholds))
        if "mean_threshold" in info:
            return float(info["mean_threshold"])
    if FOLD_BEST_THRESHOLDS is not None:
        return float(np.mean(np.asarray(FOLD_BEST_THRESHOLDS, dtype=float)))
    print("Warning: threshold file not found; using DEFAULT_THRESHOLD.")
    return float(DEFAULT_THRESHOLD)


# ========== 4. Predictor + signed occlusion interpretability ==========
class StrucTFactorPredictor:
    def __init__(self, model_paths, avg_threshold=0.5, device="cpu"):
        self.device = torch.device(device)
        self.avg_threshold = float(avg_threshold)
        self.models = []
        for fold, path in enumerate(model_paths, start=1):
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing fold {fold} weight: {path}")
            model = StrucTFactorModel().to(self.device)
            load_model_weight(model, path, self.device)
            model.eval()
            self.models.append(model)
            print(f"Loaded fold {fold}: {path}")
        if not self.models:
            raise ValueError("model_paths is empty")

    @staticmethod
    def encode_input(raw_seq, raw_ss3):
        seq, ss3 = str(raw_seq).strip(), str(raw_ss3).strip()
        actual_len = min(len(seq), len(ss3), MAX_LEN)
        if actual_len <= 0:
            raise ValueError("Sequence and SS3 must both contain residues")
        if len(seq) != len(ss3):
            warnings.warn("Sequence/SS3 length mismatch; common part is used")
        return encode_sequence_ss3(seq, ss3), actual_len

    @torch.inference_mode()
    def predict_probs(self, encoded_batch):
        """Return ensemble P(TF) for (1000,63) or (B,1000,63)."""
        arr = np.asarray(encoded_batch, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr[None, ...]
        if arr.ndim != 3 or arr.shape[1:] != (MAX_LEN, N_FEATURES):
            raise ValueError("Expected (1000,63) or (B,1000,63)")
        x = torch.from_numpy(np.ascontiguousarray(arr)).unsqueeze(1).to(self.device)
        prob_sum = np.zeros(x.shape[0], dtype=np.float64)
        for model in self.models:
            prob_sum += torch.sigmoid(model(x)).cpu().numpy().reshape(-1)
        return prob_sum / len(self.models)

    def analyze_sequence(self, raw_seq, raw_ss3, window_size=5, occlusion_batch_size=8):
        """
        Joint sequence+SS3 occlusion.

        Masked residues: all 63 one-hot features -> 0 (no padding-mask override
        is needed since StrucTFactor is a CNN without a transformer mask).
        All 1000 positions are kept, no positional shifting.

        signed Delta P = P_original(TF) - P_masked(TF)
            > 0: supports TF-class prediction
            < 0: suppresses TF-class prediction
        """
        if not isinstance(window_size, int) or window_size < 1 or window_size % 2 == 0:
            raise ValueError("window_size must be a positive odd integer")
        if not isinstance(occlusion_batch_size, int) or occlusion_batch_size < 1:
            raise ValueError("occlusion_batch_size must be a positive integer")

        encoded, actual_len = self.encode_input(raw_seq, raw_ss3)
        base_prob = float(self.predict_probs(encoded)[0])
        signed_scores = np.zeros(actual_len, dtype=np.float64)
        half_window = window_size // 2

        for batch_start in range(0, actual_len, occlusion_batch_size):
            batch_end = min(batch_start + occlusion_batch_size, actual_len)
            batch = np.repeat(encoded[None, ...], batch_end - batch_start, axis=0)
            for batch_idx, residue_idx in enumerate(range(batch_start, batch_end)):
                start = max(0, residue_idx - half_window)
                end = min(actual_len, residue_idx + half_window + 1)
                # Joint removal of AA identity AND SS3 in this window.
                batch[batch_idx, start:end, :] = 0.0
            masked_probs = self.predict_probs(batch)
            signed_scores[batch_start:batch_end] = base_prob - masked_probs
            print(f"\rOcclusion progress: {batch_end}/{actual_len}", end="", flush=True)
        print()

        positive_scores = np.clip(signed_scores, 0.0, None)
        normalized_positive = positive_scores.copy()
        if normalized_positive.max() > 0:
            normalized_positive /= normalized_positive.max()

        return {
            "base_prob": base_prob,
            "signed_scores": signed_scores,
            "positive_scores": positive_scores,
            "normalized_positive": normalized_positive,
            "actual_len": actual_len,
        }

    @staticmethod
    def find_top_motif(raw_seq, positive_scores, motif_len=15):
        """Choose the contiguous motif with the largest mean *raw* positive Delta P."""
        if not isinstance(motif_len, int) or motif_len < 1:
            raise ValueError("motif_len must be a positive integer")
        if len(positive_scores) < motif_len:
            raise ValueError("Sequence shorter than motif_len")
        scores = np.convolve(
            positive_scores, np.ones(motif_len, dtype=float) / motif_len, mode="valid")
        best_start_0 = int(np.argmax(scores))
        best_end_0 = best_start_0 + motif_len
        return {
            "sequence": raw_seq[best_start_0:best_end_0],
            "start": best_start_0 + 1,
            "end": best_end_0,
            "score": float(scores[best_start_0]),
        }

    @staticmethod
    def export_scores_csv(raw_seq, raw_ss3, signed_scores, normalized_positive, output_file):
        with open(output_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Position", "AminoAcid", "SS3",
                "Signed_Delta_Probability", "Normalized_Positive_Importance",
            ])
            for i, signed in enumerate(signed_scores):
                writer.writerow([
                    i + 1, raw_seq[i], raw_ss3[i],
                    float(signed), float(normalized_positive[i]),
                ])
        print(f"CSV saved to: {output_file}")

    def plot_interpretability(
        self,
        raw_seq,
        raw_ss3,
        protein_id="Protein",
        known_label=None,
        window_size=5,
        motif_len=15,
        output_file="StrucTFactor_Interpretability.svg",
        csv_file=None,
        occlusion_batch_size=8,
    ):
        result = self.analyze_sequence(
            raw_seq, raw_ss3, window_size=window_size,
            occlusion_batch_size=occlusion_batch_size)
        base_prob = result["base_prob"]
        signed_scores = result["signed_scores"]
        positive_scores = result["positive_scores"]
        normalized_positive = result["normalized_positive"]
        actual_len = result["actual_len"]
        seq = str(raw_seq).strip()[:actual_len]
        ss3 = str(raw_ss3).strip()[:actual_len]

        motif = self.find_top_motif(seq, positive_scores, motif_len=motif_len)
        prediction_label = "TF" if base_prob > self.avg_threshold else "Non-TF"

        if csv_file is not None:
            self.export_scores_csv(seq, ss3, signed_scores, normalized_positive, csv_file)

        x = np.arange(1, actual_len + 1)
        fig = plt.figure(figsize=(13.0, 6.0))
        gs = fig.add_gridspec(
            2, 1, height_ratios=[0.25, 3.75], hspace=0.05,
            top=0.88, bottom=0.15, left=0.085, right=0.975)
        ax_domain = fig.add_subplot(gs[0, 0])
        ax = fig.add_subplot(gs[1, 0])
        xmin, xmax = 1, actual_len

        # ================= Panel A: Top motif only =================
        ax_domain.set_xlim(xmin - 0.5, xmax + 0.5)
        ax_domain.set_ylim(0, 1)

        # Top 15-mer indicator block
        ax_domain.axvspan(
            motif["start"] - 0.5, motif["end"] + 0.5,
            ymin=0.6, ymax=0.9, color="#E15759", zorder=3)
        ax_domain.text(
            motif["start"] - 0.5, 1.05,
            f"Top {motif_len}-mer ({motif['start']}-{motif['end']})",
            ha="left", va="bottom", fontsize=10,
            color="#E15759", fontweight="bold", clip_on=False)

        ax_domain.set_yticks([])
        ax_domain.set_xticks([])
        for spine in ax_domain.spines.values():
            spine.set_visible(False)

        # ================= Panel B: occlusion contribution curve =================
        ax.axvspan(motif["start"] - 0.5, motif["end"] + 0.5,
                   color="#E15759", alpha=0.15, zorder=0)
        ax.axhline(0, color="#666666", linewidth=0.8, linestyle="--", zorder=1)

        ax.fill_between(x, 0, signed_scores, where=(signed_scores >= 0),
                        interpolate=True, alpha=0.55, color="#E15759",
                        label="Positive contribution")
        ax.fill_between(x, 0, signed_scores, where=(signed_scores < 0),
                        interpolate=True, alpha=0.30, color="#4C78A8",
                        label="Negative contribution")
        ax.plot(x, signed_scores, linewidth=1.0, color="#333333", zorder=3)

        # Axis tick formatting (match reference figure)
        ax.yaxis.set_major_locator(MultipleLocator(0.025))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        ax.xaxis.set_major_locator(MultipleLocator(50))

        if positive_scores.max() > 0:
            peak_idx = int(np.argmax(positive_scores))
            ax.scatter(peak_idx + 1, signed_scores[peak_idx],
                       s=30, color="#A61B1B", zorder=5)

        ax.set_xlim(xmin - 0.5, xmax + 0.5)
        ax.set_ylim(min(-0.08, signed_scores.min() * 1.1),
                    max(0.08, signed_scores.max() * 1.1))
        ax.set_xlabel("Amino acid position")
        ax.set_ylabel("Occlusion contribution\n" + r"$\Delta P(\mathrm{TF})$")
        ax.grid(axis="x", linestyle=":", linewidth=0.5, alpha=0.25)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # ================= Figure header =================
        header_center = (
            f'PROTEIN_ID = "{protein_id}"' if known_label is None
            else f'PROTEIN_ID = "{protein_id}({known_label})"')
        fig.text(0.5, 0.96, header_center, ha="center", va="top",
                 fontsize=11, fontweight="normal")
        fig.text(0.975, 0.96,
                 f"P(TF) = {base_prob:.3f} | "
                 f"Threshold = {self.avg_threshold:.3f} | "
                 f"Predicted: {prediction_label}",
                 ha="right", va="top", fontsize=11)

        # ================= Shared legend =================
        legend_handles = [
            Patch(facecolor="#E15759", alpha=0.55, label="Positive contribution"),
            Patch(facecolor="#4C78A8", alpha=0.30, label="Negative contribution"),
            Patch(facecolor="#E15759", alpha=0.90, label=f"Top {motif_len}-mer"),
        ]
        fig.legend(handles=legend_handles, loc="lower center",
                   bbox_to_anchor=(0.5, 0.02), ncol=len(legend_handles),
                   frameon=False, fontsize=10, columnspacing=2.5, handletextpad=0.6)

        plt.savefig(output_file, format="svg", bbox_inches="tight", facecolor="white")
        plt.show()
        plt.close(fig)
        print(f"\nSVG saved to: {output_file}")

        return {
            "base_prob": base_prob,
            "prediction": prediction_label,
            "motif": motif,
            "signed_scores": signed_scores,
            "normalized_positive": normalized_positive,
        }


# ========== 5. Main ==========
if __name__ == "__main__":
    PROTEIN_ID = "O34558"
    KNOWN_LABEL = "Non-TF"

    unknown_seq = (
        "MFNSEIKEKYLDTLSEGMVMQMRPIFAKAEITETLYNKDIYDFTSMQILELIRSFDQTTIGSVRRTLALLSLYIDWAISYKLSKGLTNLARTISEEELYECLGDKKLYITYSELEEMENQLVNYQSKAVLRLLFEGVSGLAHSELLSLTKKQVEDAMLNGNVLTLYDSKHGERKLKVSSECLVIALNAAQETKYKLKNGKAKGQTKEVFLVENDYVVKTKRTSNKGDGQASKFVITNLITDISEFFKINFLTPNTIVRSGHLYRAYQLYKEKGVIDNSVRYQIIDDFNLRVKSKYRAVYSMQDYINEEEVNKYYAEELGLKETTI"
    )
    unknown_ss3 = (
        "NNNaaaaaaaaNNNNaaaaaaaaaaaaaaaaaaaaaNNNaaaNNaaaaaaaaaaNNNNNaaaaaaaaaaaaaaaaaaaaNNNNNNNNNaaaaNNaaaaaNNNNNNNNNNNaaaaaaaaaaNNNaaaaaaaaaaaNNNNNaaaaaaaaNbaaaaaaaaNNNNbbbbbbNNNbbbbbbNNaaaaaaaaaaaaNNbbbaaaaNNNNNNNbbbbNNNNbNNNNbNNNNNNNNNNNNaaaaaaaaaaaaaaNNNNNNaaaaaaaaaaaaaaaaaaaaNNNNaaaaaaaaaaNNNNNNNaaaaaaaaaNNNNaaaaaaaNNNNNNNNNNNN"
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    threshold = get_final_threshold()
    print(f"Device: {device}")
    print(f"Five-fold mean threshold: {threshold:.4f}")

    predictor = StrucTFactorPredictor(
        model_paths=MODEL_FILES,
        avg_threshold=threshold,
        device=device,
    )

    encoded, length = predictor.encode_input(unknown_seq, unknown_ss3)
    final_prob = float(predictor.predict_probs(encoded)[0])
    prediction = "TF" if final_prob > threshold else "Non-TF"

    print("\n" + "=" * 60)
    print("StrucTFactor Prediction")
    print("=" * 60)
    print(f"Protein: {PROTEIN_ID}")
    print(f"Sequence length: {length} aa")
    print(f"Ensemble P(TF): {final_prob:.4f}")
    print(f"Threshold: {threshold:.4f}")
    print(f"Prediction: {prediction}")
    print("=" * 60)

    predictor.plot_interpretability(
        raw_seq=unknown_seq,
        raw_ss3=unknown_ss3,
        protein_id=PROTEIN_ID,
        known_label=KNOWN_LABEL,
        window_size=WINDOW_SIZE,
        motif_len=MOTIF_LEN,
        output_file="StrucTFactor_Occlusion_Interpretability.svg",
        csv_file="StrucTFactor_Occlusion_Scores.csv",
        occlusion_batch_size=OCCLUSION_BATCH_SIZE,
    )