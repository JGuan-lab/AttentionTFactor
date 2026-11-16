# AttentionTFactor: A Transcription Factor Prediction Framework Integrating Joint Sequence-Secondary Structure Representations and Contextual Modeling

Code and datasets for transcription factor (TF) prediction using **AttentionTFactor**, with comparative experiments involving **DeepTFactor** and **StrucTFactor**, ablation studies, cross-dataset generalization tests, and interpretability analyses.

![AttentionTFactor model framework and analysis workflow](assets/attentiontfactor_framework.png)

*Figure 1. Overview of the AttentionTFactor architecture, including joint sequence–secondary-structure representations, multi-scale convolutional branches, SE attention, Transformer encoding, and the single-species, multi-species, generalization, and interpretability analyses.*

> **Repository status:** This repository contains source code and ten CSV datasets from the supplied project archive. It does **not** include pretrained `.pth` checkpoints, pretrained model download links, completed experiment outputs, or a locked software environment. Some scripts require path edits or extra inputs before they can run. No performance results are asserted in this README.

## Repository structure

```text
.
├── AttentionTFactor/
│   ├── main_model_AttentionTFactor.py   # Main model training
│   ├── Ablation_experiment.py          # Ablation experiments (command-line arguments)
│   ├── GeneralizationTest.py          # Generalization evaluation; needs checkpoints
│   └── visualization.py               # Model attribution/occlusion plots; needs checkpoints
├── benchmark/
│   ├── DeepTFactor/
│   │   ├── main_model_DeepTFactor.py
│   │   ├── GeneralizationTest_DeepTFactor.py
│   │   └── visualization_DeepTFactor.py
│   ├── StrucTFactor/
│   │   ├── main_model_StrucTFactor.py
│   │   ├── GeneralizationTest_StrucTFactor.py
│   │   └── visualization_StrucTFactor.py
│   └── Performance_Testing _ThreeModels.py  # Shared-split, three-model benchmark
├── Datasets/                        # Ten provided CSV datasets
├── Interpretability Analysis/
│   ├── IG_Attribution/              # Integrated Gradients and summary statistics
│   ├── IntrePro/                    # InterPro annotation / 15-mer workflows
│   ├── Rate4Site/                   # Conservation analysis + Colab notebook
│   └── main_model_AttentionTFactor.py
├── assets/
│   └── attentiontfactor_framework.png    # Model framework diagram
├── .gitignore
├── requirements.txt
└── README.md
```

The folder spelling `IntrePro` and the space in `Performance_Testing _ThreeModels.py` are retained to match the existing files.

## Requirements

- **Python:** 3.10 or 3.11 is a reasonable starting point; the original archive did not specify a tested version. Compatibility has not been verified for every dependency combination.
- **Python dependencies:** PyTorch, NumPy, pandas, scikit-learn, SciPy, Matplotlib, Requests, and Captum (see `requirements.txt`).
- **Hardware:** CPU operation is supported by scripts that offer a CPU/auto device option; CUDA-compatible PyTorch/GPU can be used where available. Training may be computationally expensive.
- **For the Rate4Site Google Colab notebook only:** NCBI BLAST+, MAFFT, Rate4Site, and `wget` are system-level tools, installed in the notebook using Ubuntu `apt-get`; they are **not** installed by `pip`.

### Install Python dependencies

From the **repository root**:

```bash
python -m venv .venv
# macOS / Linux:
source .venv/bin/activate
# Windows PowerShell alternative:
# .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For NVIDIA GPU support, select the appropriate PyTorch build for your CUDA environment rather than assuming that a default `pip install torch` will match your system.

## Data

The `Datasets/` directory includes:

```text
110_multimodal_dataset.csv
13_multimodal_dataset.csv
15_multimodal_dataset.csv
arabidopsis_thaliana_dataset.csv
bacillus_subtilis_dataset.csv
bacteria_domain_dataset.csv
chimpanzee_dataset.csv
eukaryotic_domain_dataset.csv
human_dataset.csv
mouse_dataset.csv
```

The included CSVs have the columns `Protein_ID`, `Sequence`, `SS3`, `Label`, and `Length`. The model scripts use protein sequences and, depending on the model, secondary-structure (`SS3`) information to predict a binary `Label`.

**Important path detail:** The original scripts frequently use relative filenames such as `CSV_FILE = "13_multimodal_dataset.csv"`. Python resolves these relative to the **current working directory**, not the script's folder. If you run a script from the repository root, change that setting to:

```python
CSV_FILE = "Datasets/13_multimodal_dataset.csv"
```

The two command-line experiment scripts accept a `--csv` flag and can use the path without editing their source, as shown below. Please keep the original CSV file content and row order unchanged when reproducing saved dataset splits.

## Run experiments

All examples below assume you are in the **repository root**.

### 1. Train AttentionTFactor

In `AttentionTFactor/main_model_AttentionTFactor.py`, first set `CSV_FILE` to `"Datasets/13_multimodal_dataset.csv"`, then run:

```bash
python AttentionTFactor/main_model_AttentionTFactor.py
```

The script uses a five-fold training procedure and writes `best_attentiontfactor_13_model_fold_*.pth` checkpoints and fold-threshold metadata to the current working directory. Checkpoints are excluded by `.gitignore` by default.

### 2. Train baseline models

Set `CSV_FILE = "Datasets/13_multimodal_dataset.csv"` in **each** baseline training file, then run:

```bash
python benchmark/DeepTFactor/main_model_DeepTFactor.py
python benchmark/StrucTFactor/main_model_StrucTFactor.py
```

These scripts write their own `best_deeptfactor_13_model_fold_*.pth` and `best_structfactor_13_model_fold_*.pth` checkpoints.

### 3. AttentionTFactor ablation study

This script accepts the CSV path and a new/empty results directory via command-line flags:

```bash
python AttentionTFactor/Ablation_experiment.py \
  --csv Datasets/13_multimodal_dataset.csv \
  --outdir attentiontfactor_ablation_results \
  --device auto
```

It supports additional flags such as `--models`, `--epochs`, `--batch-size`, and `--splits`. Run `python AttentionTFactor/Ablation_experiment.py --help` to see the current options.

### 4. Compare all three models with shared splits

**Note the space before the underscore** in the supplied benchmark filename:

```bash
python "benchmark/Performance_Testing _ThreeModels.py" \
  --csv Datasets/13_multimodal_dataset.csv \
  --outdir three_model_benchmark_results \
  --device auto
```

This comparison script uses shared development/test splitting and a five-fold protocol. The output directory should be new or empty; choose a different name to rerun without overwriting prior results.

### 5. Generalization tests and interpretability

- **Generalization:** `AttentionTFactor/GeneralizationTest.py` and the baseline `GeneralizationTest_*.py` scripts load model checkpoints specified in `MODEL_PATHS`. Those weights are not in this repository. Their default CSV name, `bacillus_subtilis_multimodal_dataset.csv`, also differs from the included file `Datasets/bacillus_subtilis_dataset.csv`. Update **both** the input CSV path and checkpoint locations for your experiment.
- **Visualizations:** The `visualization*.py` scripts require the trained checkpoints named in their `MODEL_FILES` settings. Configure the checkpoint paths and sample inputs before running.
- **Integrated Gradients (IG):** `Interpretability Analysis/IG_Attribution/IGAttribution_threemodels.py` compares the three trained models using saved fold checkpoints. Its `--csv` and `--*-weights-dir` paths are interpreted relative to **the IG script's directory**, not the shell's working directory. From this repository layout, the training CSV can be passed as `../../Datasets/13_multimodal_dataset.csv` and model weights in the repository root as `../../`. The IG script may obtain domain annotations online unless run in offline mode with local annotations. Captum is used when installed; a numerical fallback is implemented in the script.
- **InterPro / 15-mer analysis:** The directory is named `Interpretability Analysis/IntrePro/` in the original archive. These scripts depend on input tables, trained checkpoints and/or online InterPro/UniProt requests; some generated tables are not included.
- **Rate4Site:** `Interpretability Analysis/Rate4Site/Rate4Site_SwissProt.ipynb` is a Google Colab pipeline requiring additional input CSVs named `human-multimodal_dataset.csv` and `interpro_dbd_paper_table_human_deeptfactor.csv`. **Neither is included under those names in the archive.** The notebook installs BLAST+, MAFFT and Rate4Site, downloads Swiss-Prot reference data and performs conservation analysis. `Conservative_Analysis.py` expects the notebook's extracted analysis results.

The additional `Interpretability Analysis/main_model_AttentionTFactor.py` currently points to `gj-multimodal_dataset.csv`, which is **not present** in this archive; select an appropriate input dataset before running it.

## Reproducibility and repository notes

- Paths, original dataset ordering, training seeds, and saved fold checkpoints can affect reproduction. Read each experiment script's settings before running it.
- Please verify dataset sharing permissions and add an explicit `LICENSE` file if you intend to grant others reuse rights. No license or data provenance document was supplied in the archive.
- If this repository accompanies a paper, add the paper citation, exact experimental environment/package versions, pretrained checkpoint sources, and dataset attribution once available.
