# ============================================================
# AttentionTFactor
# Five-fold ensemble prediction
# + Sliding-window occlusion interpretability
# + Top 15-mer motif extraction
# + Publication-ready SVG output
# ============================================================

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import math, csv, warnings
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch

warnings.filterwarnings("ignore")


# ============================================================
# 0. Figure / font settings
# ============================================================
try:
    font_manager.findfont("Arial", fallback_to_default=False)
    FONT = "Arial"
except Exception:
    FONT = "Liberation Sans"
    print("Warning: Arial not found. Using Liberation Sans instead.")

plt.rcParams.update({
    "font.family": FONT,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9.5,
    "svg.fonttype": "none",   # keep SVG text editable
    "axes.linewidth": 1.0,
})


# ============================================================
# 1. Model (must stay consistent with training)
# ============================================================
class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention."""

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
        b, c, _ = x.shape
        y = self.fc(self.avg_pool(x).view(b, c)).view(b, c, 1)
        return x * y.expand_as(x)


class AttentionTFactorCNN(nn.Module):
    """
    Sequence + SS3 -> Embedding -> Multi-scale CNN + SE
    -> Transformer -> Masked Max/Avg Pooling -> Binary classification
    """

    def __init__(self, seq_vocab_size=21, ss3_vocab_size=4, embed_dim=256):
        super().__init__()
        self.seq_embedding = nn.Embedding(seq_vocab_size, embed_dim, padding_idx=0)
        self.ss3_embedding = nn.Embedding(ss3_vocab_size, embed_dim, padding_idx=0)

        in_channels = embed_dim * 2
        branch_channels = 64
        self.branch_a = self._make_branch(in_channels, branch_channels, [4, 4, 16])
        self.branch_b = self._make_branch(in_channels, branch_channels, [12, 8, 4])
        self.branch_c = self._make_branch(in_channels, branch_channels, [16, 4, 4])

        cnn_channels = branch_channels * 3
        self.pos_encoder = PositionalEncoding(cnn_channels)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=cnn_channels, nhead=8, dim_feedforward=512,
                dropout=0.1, batch_first=True),
            num_layers=1,
        )
        self.fc = nn.Sequential(
            nn.Linear(cnn_channels * 2, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Dropout(0.5), nn.Linear(128, 1),
        )

    @staticmethod
    def _make_branch(in_c, out_c, kernels):
        return nn.Sequential(
            nn.Conv1d(in_c, out_c, kernel_size=kernels[0], padding="same"), nn.BatchNorm1d(out_c), nn.ReLU(),
            nn.Conv1d(out_c, out_c, kernel_size=kernels[1], padding="same"), nn.BatchNorm1d(out_c), nn.ReLU(),
            nn.Conv1d(out_c, out_c, kernel_size=kernels[2], padding="same"), nn.BatchNorm1d(out_c), nn.ReLU(),
            SEBlock(out_c), nn.Dropout(0.1),
        )

    def forward(self, x_seq, x_ss3, padding_mask_override=None):
        """
        padding_mask_override:
            Normal prediction -> None (mask derived from x_seq == 0)
            Occlusion analysis -> use the ORIGINAL padding mask
            (internally occluded residues must NOT become true padding)
        """
        if padding_mask_override is None:
            padding_mask = x_seq.eq(0)
        else:
            padding_mask = padding_mask_override.to(x_seq.device).bool()

        # Embedding: [B, L] -> [B, L, 256] each -> concat [B, L, 512]
        x = torch.cat([self.seq_embedding(x_seq), self.ss3_embedding(x_ss3)], dim=-1)

        # CNN: [B, L, 512] -> [B, 512, L] -> [B, 192, L] -> [B, L, 192]
        x = x.transpose(1, 2)
        x = torch.cat([self.branch_a(x), self.branch_b(x), self.branch_c(x)], dim=1)
        x = x.transpose(1, 2)

        # Transformer
        x = self.pos_encoder(x)
        x = self.transformer(x, src_key_padding_mask=padding_mask)

        # Masked pooling
        mask = padding_mask.unsqueeze(-1).expand_as(x)
        p_max = x.masked_fill(mask, -1e9).max(dim=1)[0]
        sum_features = x.masked_fill(mask, 0.0).sum(dim=1)
        valid_lens = (~padding_mask).sum(dim=1, keepdim=True).float().clamp(min=1.0)
        p_avg = sum_features / valid_lens

        return self.fc(torch.cat([p_max, p_avg], dim=1))


# ============================================================
# 2. Predictor + interpretation
# ============================================================
class TFPredictor:

    def __init__(self, model_paths, avg_threshold=0.5, device="cpu"):
        self.device = torch.device(device)
        self.avg_threshold = avg_threshold
        self.max_len = 1000

        self.seq_vocab = {aa: i + 1 for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}
        self.ss3_vocab = {"H": 1, "E": 2, "C": 3, "N": 3, "a": 1, "b": 2}

        self.models = []
        for path in model_paths:
            model = AttentionTFactorCNN().to(self.device)
            checkpoint = torch.load(path, map_location=self.device)
            # Compatible with either raw state_dict or {"state_dict": ...}
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint
            model.load_state_dict(state_dict)
            model.eval()
            self.models.append(model)

        print(f"Loaded {len(self.models)} models.")

    # --------------------------------------------------------
    # Encode input
    # --------------------------------------------------------
    def encode_input(self, seq, ss3):
        seq, ss3 = str(seq)[:self.max_len], str(ss3)[:self.max_len]
        actual_len = min(len(seq), len(ss3), self.max_len)
        seq, ss3 = seq[:actual_len], ss3[:actual_len]

        seq_ids = [self.seq_vocab.get(aa, 0) for aa in seq]
        ss3_ids = [self.ss3_vocab.get(s, 0) for s in ss3]
        seq_padded = seq_ids + [0] * (self.max_len - actual_len)
        ss3_padded = ss3_ids + [0] * (self.max_len - actual_len)

        return (
            torch.tensor([seq_padded], dtype=torch.long),
            torch.tensor([ss3_padded], dtype=torch.long),
            actual_len,
        )

    # --------------------------------------------------------
    # Ensemble probability
    # --------------------------------------------------------
    def predict_prob(self, seq_tensor, ss3_tensor, padding_mask_override=None):
        seq_tensor = seq_tensor.to(self.device)
        ss3_tensor = ss3_tensor.to(self.device)
        if padding_mask_override is not None:
            padding_mask_override = padding_mask_override.to(self.device)

        probs = []
        with torch.no_grad():
            for model in self.models:
                logits = model(seq_tensor, ss3_tensor,
                               padding_mask_override=padding_mask_override)
                probs.append(torch.sigmoid(logits).item())
        return float(np.mean(probs))

    # --------------------------------------------------------
    # Sliding-window occlusion
    # --------------------------------------------------------
    def analyze_sequence(self, raw_seq, raw_ss3, window_size=5):
        """
        Joint sequence + SS3 occlusion.
        Importance = base_probability - masked_probability
            positive -> region supports TF prediction
            negative -> region suppresses TF prediction
        The original padding mask is preserved during internal occlusion.
        """
        if window_size < 1:
            raise ValueError("window_size must be >= 1")
        if window_size % 2 == 0:
            raise ValueError("Please use an odd window_size, e.g. 5, 7, 9, 15.")

        seq_tensor, ss3_tensor, actual_len = self.encode_input(raw_seq, raw_ss3)
        original_padding_mask = seq_tensor.eq(0)

        base_prob = self.predict_prob(seq_tensor, ss3_tensor,
                                      padding_mask_override=original_padding_mask)

        signed_scores = np.zeros(actual_len, dtype=float)
        half_window = window_size // 2

        for i in range(actual_len):
            start = max(0, i - half_window)
            end = min(actual_len, i + half_window + 1)

            masked_seq = seq_tensor.clone()
            masked_ss3 = ss3_tensor.clone()
            # Joint occlusion: remove both sequence and structural info,
            # but keep the original padding mask unchanged.
            masked_seq[0, start:end] = 0
            masked_ss3[0, start:end] = 0

            masked_prob = self.predict_prob(masked_seq, masked_ss3,
                                            padding_mask_override=original_padding_mask)
            signed_scores[i] = base_prob - masked_prob

        positive_scores = np.clip(signed_scores, 0, None)

        # 0-1 normalization FOR PLOTTING ONLY (not for cross-protein ranking)
        normalized_positive = positive_scores.copy()
        max_score = normalized_positive.max()
        if max_score > 0:
            normalized_positive /= max_score

        return {
            "base_prob": base_prob,
            "signed_scores": signed_scores,
            "positive_scores": positive_scores,
            "normalized_positive": normalized_positive,
            "actual_len": actual_len,
        }

    # --------------------------------------------------------
    # Top motif
    # --------------------------------------------------------
    @staticmethod
    def find_top_motif(raw_seq, positive_scores, motif_len=15):
        """
        Find the highest-scoring contiguous motif.
        Uses RAW positive Δprob scores, NOT per-protein normalized scores.
        """
        if motif_len < 1:
            raise ValueError("motif_len must be >= 1")
        if len(positive_scores) < motif_len:
            raise ValueError("Sequence shorter than motif_len.")

        kernel = np.ones(motif_len, dtype=float) / motif_len
        motif_scores = np.convolve(positive_scores, kernel, mode="valid")
        best_start_0 = int(np.argmax(motif_scores))
        best_end_0 = best_start_0 + motif_len

        return {
            "sequence": raw_seq[best_start_0:best_end_0],
            "start": best_start_0 + 1,   # 1-based
            "end": best_end_0,
            "score": float(motif_scores[best_start_0]),
        }

    # --------------------------------------------------------
    # Export residue scores
    # --------------------------------------------------------
    @staticmethod
    def export_scores_csv(raw_seq, raw_ss3, signed_scores, normalized_positive, output_file):
        with open(output_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Position", "AminoAcid", "SS3",
                "Signed_Delta_Probability", "Normalized_Positive_Importance",
            ])
            for i in range(len(signed_scores)):
                writer.writerow([
                    i + 1, raw_seq[i], raw_ss3[i],
                    signed_scores[i], normalized_positive[i],
                ])
        print(f"CSV saved to: {output_file}")

    # --------------------------------------------------------
    # Publication figure (optimized layout)
    # --------------------------------------------------------
    def plot_interpretability(
        self,
        raw_seq,
        raw_ss3,
        protein_id="Protein",
        known_label=None,
        window_size=5,
        motif_len=15,
        output_file="Interpretability.svg",
        csv_file=None,
    ):

        # ---------------- Analysis ----------------
        result = self.analyze_sequence(raw_seq, raw_ss3, window_size=window_size)
        base_prob = result["base_prob"]
        signed_scores = result["signed_scores"]
        positive_scores = result["positive_scores"]
        normalized_positive = result["normalized_positive"]
        actual_len = result["actual_len"]

        seq = raw_seq[:actual_len]
        ss3 = raw_ss3[:actual_len]

        motif = self.find_top_motif(seq, positive_scores, motif_len=motif_len)

        is_tf = base_prob > self.avg_threshold
        prediction_label = "TF" if is_tf else "Non-TF"

        # ---------------- Console report ----------------
        print("\n" + "=" * 60)
        print("AttentionTFactor Interpretability Report")
        print("=" * 60)
        print(f"Protein ID: {protein_id}")
        if known_label is not None:
            print(f"Known label: {known_label}")
        print(f"Sequence length: {actual_len} aa")
        print(f"P(TF): {base_prob:.4f}")
        print(f"Threshold: {self.avg_threshold:.4f}")
        print(f"Prediction: {prediction_label}")
        print(f"Top {motif_len}-mer: {motif['sequence']}")
        print(f"Position: {motif['start']}-{motif['end']}")
        print(f"Raw motif score: {motif['score']:.6f}")
        print("=" * 60)

        # ---------------- Optional CSV ----------------
        if csv_file is not None:
            self.export_scores_csv(seq, ss3, signed_scores, normalized_positive, csv_file)

        # ================= Figure =================
        x = np.arange(1, actual_len + 1)
        fig = plt.figure(figsize=(13.0, 6.0))
        gs = fig.add_gridspec(
            2, 1, height_ratios=[0.45, 3.55], hspace=0.05,
            top=0.84, bottom=0.16, left=0.085, right=0.975,
        )
        ax_domain = fig.add_subplot(gs[0, 0])
        ax = fig.add_subplot(gs[1, 0])
        xmin, xmax = 1, actual_len

        # ================= TOP TRACK: motif only =================
        ax_domain.set_xlim(xmin, xmax)
        ax_domain.set_ylim(0, 1)

        # Protein backbone
        ax_domain.hlines(y=0.5, xmin=xmin, xmax=xmax,
                         linewidth=3.2, color="#C8C8C8", zorder=1)

        # Top motif
        ax_domain.axvspan(motif["start"], motif["end"], ymin=0.05, ymax=0.95,
                          color="#E15759", alpha=0.90, zorder=3)
        ax_domain.text((motif["start"] + motif["end"]) / 2, 1.08,
                       f"Top {motif_len}-mer ({motif['start']}-{motif['end']})",
                       ha="center", va="bottom", fontsize=9,
                       color="#9C2F31", fontweight="bold", clip_on=False)

        ax_domain.set_yticks([])
        ax_domain.set_xticks([])
        for spine in ax_domain.spines.values():
            spine.set_visible(False)

        ax_domain.text(-0.045, 0.5, "A", transform=ax_domain.transAxes,
                       ha="right", va="center", fontsize=12, fontweight="bold")

        # ================= BOTTOM: signed occlusion curve =================
        # Motif background band
        ax.axvspan(motif["start"], motif["end"],
                   color="#E15759", alpha=0.22, zorder=0)

        # Zero reference line
        ax.axhline(0, color="#666666", linewidth=0.9, linestyle="--", zorder=1)

        # Positive area
        ax.fill_between(x, 0, signed_scores, where=(signed_scores >= 0),
                        interpolate=True, alpha=0.55, color="#E15759",
                        label="Supports TF prediction")
        # Negative area
        ax.fill_between(x, 0, signed_scores, where=(signed_scores < 0),
                        interpolate=True, alpha=0.30, color="#4C78A8",
                        label="Suppresses TF prediction")

        # Main signed curve
        ax.plot(x, signed_scores, linewidth=1.25, color="#333333", zorder=3)

        # Peak marker
        if positive_scores.max() > 0:
            peak_idx = int(np.argmax(positive_scores))
            peak_pos = peak_idx + 1
            peak_score = signed_scores[peak_idx]
            ax.scatter(peak_pos, peak_score, s=45, color="#A61B1B", zorder=5)
            y_range = signed_scores.max() - signed_scores.min()
            ax.annotate(
                f"Peak: {peak_pos} aa",
                xy=(peak_pos, peak_score),
                xytext=(peak_pos, peak_score + y_range * 0.15),
                ha="center", fontsize=8.5,
                arrowprops=dict(arrowstyle="-", linewidth=0.8, color="#666666"),
            )

        ax.set_xlim(xmin, xmax)
        ax.set_xlabel("Amino acid position")
        ax.set_ylabel("Occlusion contribution\n" r"$\Delta P(\mathrm{TF})$")
        ax.grid(axis="x", linestyle=":", linewidth=0.5, alpha=0.25)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.text(-0.045, 1.02, "B", transform=ax.transAxes,
                ha="right", va="top", fontsize=12, fontweight="bold")

        # ================= Figure-level header =================
        header_left = f"{protein_id}" if known_label is None else f"{protein_id} ({known_label})"
        fig.text(0.085, 0.955, header_left,
                 ha="left", va="top", fontsize=13, fontweight="bold")
        fig.text(0.975, 0.955,
                 f"P(TF) = {base_prob:.3f}   |   "
                 f"Threshold = {self.avg_threshold:.3f}   |   "
                 f"Predicted: {prediction_label}",
                 ha="right", va="top", fontsize=10)

        # ================= Shared legend =================
        legend_handles = [
            Patch(facecolor="#E15759", alpha=0.55, label="Positive contribution"),
            Patch(facecolor="#4C78A8", alpha=0.30, label="Negative contribution"),
            Patch(facecolor="#E15759", alpha=0.90, label=f"Top {motif_len}-mer"),
        ]

        fig.legend(handles=legend_handles, loc="lower center",
                   bbox_to_anchor=(0.5, 0.01), ncol=len(legend_handles), frameon=False,
                   fontsize=10, columnspacing=1.8, handletextpad=0.6)

        # ================= Save =================
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


# ============================================================
# 3. Main
# ============================================================
if __name__ == "__main__":

    MODEL_FILES = [
        "best_attentiontfactor_bacillus_subtilis_model_fold_1.pth",
        "best_attentiontfactor_bacillus_subtilis_model_fold_2.pth",
        "best_attentiontfactor_bacillus_subtilis_model_fold_3.pth",
        "best_attentiontfactor_bacillus_subtilis_model_fold_4.pth",
        "best_attentiontfactor_bacillus_subtilis_model_fold_5.pth",
    ]

    # Replace with your REAL mean optimal threshold from the five validation folds.
    AVG_THRESHOLD = 0.550

    # Example protein: replace these with the protein you want to analyse.
    PROTEIN_ID = "O34558"
    KNOWN_LABEL = "non-TF"

    unknown_seq = (
        "MFNSEIKEKYLDTLSEGMVMQMRPIFAKAEITETLYNKDIYDFTSMQILELIRSFDQTTIGSVRRTLALLSLYIDWAISYKLSKGLTNLARTISEEELYECLGDKKLYITYSELEEMENQLVNYQSKAVLRLLFEGVSGLAHSELLSLTKKQVEDAMLNGNVLTLYDSKHGERKLKVSSECLVIALNAAQETKYKLKNGKAKGQTKEVFLVENDYVVKTKRTSNKGDGQASKFVITNLITDISEFFKINFLTPNTIVRSGHLYRAYQLYKEKGVIDNSVRYQIIDDFNLRVKSKYRAVYSMQDYINEEEVNKYYAEELGLKETTI"
    )
    unknown_ss3 = (
        "NNNaaaaaaaaNNNNaaaaaaaaaaaaaaaaaaaaaNNNaaaNNaaaaaaaaaaNNNNNaaaaaaaaaaaaaaaaaaaaNNNNNNNNNaaaaNNaaaaaNNNNNNNNNNNaaaaaaaaaaNNNaaaaaaaaaaaNNNNNaaaaaaaaNbaaaaaaaaNNNNbbbbbbNNNbbbbbbNNaaaaaaaaaaaaNNbbbaaaaNNNNNNNbbbbNNNNbNNNNbNNNNNNNNNNNNaaaaaaaaaaaaaaNNNNNNaaaaaaaaaaaaaaaaaaaaNNNNaaaaaaaaaaNNNNNNNaaaaaaaaaNNNNaaaaaaaNNNNNNNNNNNN"
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading models to {device}...")

    # ---------------- Predictor ----------------
    predictor = TFPredictor(
        model_paths=MODEL_FILES,
        avg_threshold=AVG_THRESHOLD,
        device=device,
    )

    # ---------------- Standard ensemble prediction ----------------
    seq_tensor, ss3_tensor, length = predictor.encode_input(unknown_seq, unknown_ss3)
    original_mask = seq_tensor.eq(0)
    final_prob = predictor.predict_prob(seq_tensor, ss3_tensor,
                                        padding_mask_override=original_mask)
    prediction = "TF" if final_prob > AVG_THRESHOLD else "Non-TF"

    print("\n" + "=" * 60)
    print("AttentionTFactor Prediction")
    print("=" * 60)
    print(f"Protein: {PROTEIN_ID}")
    print(f"Sequence length: {length} aa")
    print(f"Ensemble P(TF): {final_prob:.4f}")
    print(f"Threshold: {AVG_THRESHOLD:.4f}")
    print(f"Prediction: {prediction}")
    print("=" * 60)

    # ---------------- Sliding-window interpretation (no domain annotation) ----------------
    predictor.plot_interpretability(
        raw_seq=unknown_seq,
        raw_ss3=unknown_ss3,
        protein_id=PROTEIN_ID,
        known_label=KNOWN_LABEL,
        window_size=5,          # recommended starting point
        motif_len=15,           # manuscript uses 15-aa candidate motifs
        output_file="AttentionTFactor_Occlusion_Interpretability.svg",
        csv_file="AttentionTFactor_Occlusion_Scores.csv",
    )