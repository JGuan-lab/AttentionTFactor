import json, os, random, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, average_precision_score, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, Dataset, Subset

warnings.filterwarnings('ignore')

CSV_FILE = '13_multimodal_dataset.csv'
MAX_LEN = 1000
N_SPLITS = 5
BATCH_SIZE = 128
EPOCHS = 50
LEARNING_RATE = 0.001
MODEL_SAVE_TEMPLATE = 'best_deeptfactor_13_model_fold_{}.pth'
THRESHOLD_SAVE_FILE = 'deeptfactor_13_fold_thresholds.json'


def seed_everything(seed=1):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)

DEEP_AA_VOCAB = 'ACDEFGHIKLMNPQRSTVWXY'
DEEP_AA_TO_IDX = {aa: i for i, aa in enumerate(DEEP_AA_VOCAB)}


class DeepTFactorDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        missing = {'Sequence', 'Label'} - set(self.data.columns)
        if missing:
            raise ValueError(f'CSV file is missing required columns: {missing}')
        if not self.data['Label'].isin([0, 1]).all():
            raise ValueError('Label must be 0/1.')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        sequence = str(row['Sequence']).strip()
        if not 1 <= len(sequence) <= MAX_LEN:
            raise ValueError(f'Row {idx}: sequence length must be 1..{MAX_LEN}, got {len(sequence)}')

        x = np.zeros((1, MAX_LEN, len(DEEP_AA_VOCAB)), dtype=np.float32)
        channels = [DEEP_AA_TO_IDX.get(aa, DEEP_AA_TO_IDX['X']) for aa in sequence]
        x[0, np.arange(len(sequence)), channels] = 1.0

        return torch.from_numpy(x), torch.tensor(float(row['Label']), dtype=torch.float32)


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


def find_best_threshold(labels, probs):
    labels = np.asarray(labels).ravel()
    probs = np.asarray(probs).ravel()
    best_threshold, best_mcc = 0.5, -1.0
    for threshold in np.arange(0.1, 0.91, 0.01):
        mcc = matthews_corrcoef(labels, (probs > threshold).astype(int))
        if mcc > best_mcc:
            best_threshold, best_mcc = float(threshold), float(mcc)
    return best_threshold, best_mcc


def train_and_evaluate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    full_dataset = DeepTFactorDataset(CSV_FILE)
    labels = full_dataset.data['Label'].astype(int).tolist()
    indices = list(range(len(full_dataset)))

    print('\n[Step 1] Splitting 10% as the final blind test set...')
    cv_idx, test_idx, cv_labels, _ = train_test_split(
        indices, labels, test_size=0.1, stratify=labels, random_state=42)
    test_dataset = Subset(full_dataset, test_idx)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    fold_best_thresholds, model_save_paths = [], []

    print('\n[Step 2] Starting DeepTFactor 5-fold cross-validation...')

    for fold, (train_v_idx, val_v_idx) in enumerate(skf.split(cv_idx, cv_labels), start=1):
        print(f'\n========== Fold {fold}/{N_SPLITS} ==========')
        train_dataset = Subset(full_dataset, [cv_idx[i] for i in train_v_idx])
        val_dataset = Subset(full_dataset, [cv_idx[i] for i in val_v_idx])
        if len(train_dataset) < BATCH_SIZE:
            raise ValueError(f'Fold {fold} training samples {len(train_dataset)} < BATCH_SIZE={BATCH_SIZE}')

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

        model = DeepTFactorModel().to(device)
        criterion = nn.BCEWithLogitsLoss().to(device)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=0.0)

        best_val_loss, fold_best_thresh = float('inf'), 0.5
        save_path = MODEL_SAVE_TEMPLATE.format(fold)
        model_save_paths.append(save_path)

        for epoch in range(EPOCHS):
            model.train()
            train_loss_sum, train_n = 0.0, 0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                loss = criterion(model(x), y)
                loss.backward()
                optimizer.step()
                train_loss_sum += loss.item() * len(y)
                train_n += len(y)

            model.eval()
            val_loss_sum, val_n, all_labels, all_probs = 0.0, 0, [], []
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    logits = model(x)
                    val_loss_sum += criterion(logits, y).item() * len(y)
                    val_n += len(y)
                    all_probs.extend(torch.sigmoid(logits).cpu().numpy().ravel().tolist())
                    all_labels.extend(y.cpu().numpy().ravel().tolist())

            avg_train_loss = train_loss_sum / train_n
            avg_val_loss = val_loss_sum / val_n
            labels_np = np.asarray(all_labels, dtype=int)
            probs_np = np.asarray(all_probs, dtype=float)
            best_epoch_thresh, best_epoch_mcc = find_best_threshold(labels_np, probs_np)
            auc = roc_auc_score(labels_np, probs_np)

            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f'Epoch [{epoch + 1}/{EPOCHS}] | Train Loss: {avg_train_loss:.4f} | '
                      f'Val Loss: {avg_val_loss:.4f} | AUC: {auc:.4f} | '
                      f'MCC: {best_epoch_mcc:.4f} @ {best_epoch_thresh:.2f}')

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                fold_best_thresh = best_epoch_thresh
                torch.save(model.state_dict(), save_path)

        fold_best_thresholds.append(float(fold_best_thresh))
        print(f'✅ Fold {fold} done | Best Val Loss: {best_val_loss:.4f} | '
              f'Threshold: {fold_best_thresh:.2f} | Weights: {save_path}')
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    avg_best_thresh = float(np.mean(fold_best_thresholds))
    with open(THRESHOLD_SAVE_FILE, 'w', encoding='utf-8') as f:
        json.dump({
            'fold_thresholds': fold_best_thresholds,
            'mean_threshold': avg_best_thresh,
            'rule': 'validation MCC; thresholds 0.10-0.90 step 0.01; strict >',
        }, f, indent=2, ensure_ascii=False)

    print(f'\n5-fold thresholds saved to: {THRESHOLD_SAVE_FILE}')
    print(f'5-fold mean threshold: {avg_best_thresh:.3f}')

    print('\n[Step 3] ================ DeepTFactor Final Ensemble Test ================')
    ensemble_test_probs = np.zeros(len(test_dataset), dtype=np.float64)
    test_labels_final = None

    for fold, path in enumerate(model_save_paths, start=1):
        print(f'Loading Fold {fold}: {path}')
        model = DeepTFactorModel().to(device)
        model.load_state_dict(torch.load(path, map_location=device), strict=True)
        model.eval()

        fold_probs, fold_labels = [], []
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device)
                fold_probs.extend(torch.sigmoid(model(x)).cpu().numpy().ravel().tolist())
                if fold == 1:
                    fold_labels.extend(y.numpy().ravel().tolist())

        if fold == 1:
            test_labels_final = np.asarray(fold_labels, dtype=int)
        ensemble_test_probs += np.asarray(fold_probs, dtype=np.float64)
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    ensemble_test_probs /= N_SPLITS
    test_preds = (ensemble_test_probs > avg_best_thresh).astype(int)

    print('\n🎉 Final Independent Test Results (DeepTFactor 5-Fold Ensemble):')
    print(f'   - Test Accuracy: {accuracy_score(test_labels_final, test_preds) * 100:.2f}%')
    print(f'   - Test AUC:      {roc_auc_score(test_labels_final, ensemble_test_probs):.4f}')
    print(f'   - Test AU-PRC:   {average_precision_score(test_labels_final, ensemble_test_probs):.4f}')
    print(f'   - Test MCC:      {matthews_corrcoef(test_labels_final, test_preds):.4f}')

    result_df = full_dataset.data.iloc[test_idx].copy()
    result_df['Ensemble_Probability'] = ensemble_test_probs
    result_df['Predicted_Label'] = test_preds
    result_df.to_csv('deeptfactor_ensemble_test_predictions.csv', index=False)
    print('\nIndependent test predictions saved to: deeptfactor_ensemble_test_predictions.csv')


if __name__ == '__main__':
    train_and_evaluate()