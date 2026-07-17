import torch
import numpy as np
import matplotlib.pyplot as plt
from dataset import DATASET_PATH, concatenate_h5_files, ModelNet40Dataset
from torch.utils.data import DataLoader
from baseline import BaselineModel
from tokenizer import PointCloudTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_test, label_test = concatenate_h5_files(DATASET_PATH, 2, 'test')
dataset_test = ModelNet40Dataset(data_test, label_test, 1024, augment=False)
dataset_test_loader = DataLoader(dataset_test, batch_size=64, shuffle=False)


tokenizer = PointCloudTokenizer(token_dim=128, num_tokens=64, num_points=1024)
model = BaselineModel(tokenizer).to(device)
model.load_state_dict(torch.load('best_baseline_model.pth', map_location=device))

model.eval()
all_probs, all_preds, all_labels = [], [], []

with torch.no_grad():
    for points, labels in dataset_test_loader:
        points, labels = points.to(device), labels.to(device)
        logits = model(points)
        probs = torch.softmax(logits, dim=1)
        confidences, preds = probs.max(dim=1)

        all_probs.append(probs.cpu())
        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())

all_probs = torch.cat(all_probs)
all_preds = torch.cat(all_preds)
all_labels = torch.cat(all_labels)

print(f"Test accuracy check: {(all_preds == all_labels).float().mean().item():.2%}")

correct_mask = (all_preds == all_labels)
conf_correct = all_probs.max(dim=1).values[correct_mask]
conf_incorrect = all_probs.max(dim=1).values[~correct_mask]

fig, ax = plt.subplots(figsize=(7, 5))
ax.hist(conf_correct.numpy(), bins=20, alpha=0.6, label='Correct', color='green')
ax.hist(conf_incorrect.numpy(), bins=20, alpha=0.6, label='Incorrect', color='red')
ax.set_xlabel('Predicted confidence (max softmax prob)')
ax.set_ylabel('Count')
ax.legend()
ax.set_title('Confidence Distribution: Correct vs Incorrect')
plt.tight_layout()
plt.savefig('confidence_histogram.png', dpi=150)
plt.show()

def reliability_diagram(confidences, correct, n_bins=10):
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_accs, bin_confs, bin_counts = [], [], []

    for i in range(n_bins):
        mask = (confidences >= bin_edges[i]) & (confidences < bin_edges[i + 1])
        if mask.sum() > 0:
            bin_accs.append(correct[mask].float().mean().item())
            bin_confs.append(confidences[mask].mean().item())
            bin_counts.append(mask.sum().item())
        else:
            bin_accs.append(0)
            bin_confs.append((bin_edges[i] + bin_edges[i + 1]) / 2)
            bin_counts.append(0)

    return np.array(bin_confs), np.array(bin_accs), np.array(bin_counts)

confidences = all_probs.max(dim=1).values
correct = (all_preds == all_labels)

bin_confs, bin_accs, bin_counts = reliability_diagram(confidences, correct)

# Expected Calibration Error: weighted average gap between confidence and accuracy per bin
total = bin_counts.sum()
ece = np.sum((bin_counts / total) * np.abs(bin_confs - bin_accs))
print(f"Expected Calibration Error (ECE): {ece:.4f}")

fig, ax = plt.subplots(figsize=(6, 6))
ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect calibration')
ax.bar(bin_confs, bin_accs, width=0.08, alpha=0.7, edgecolor='black', label='Model')
ax.set_xlabel('Mean predicted confidence (per bin)')
ax.set_ylabel('Actual accuracy (per bin)')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.legend()
ax.set_title(f'Reliability Diagram (ECE = {ece:.4f})')
plt.tight_layout()
plt.savefig('reliability_diagram.png', dpi=150)
plt.show()

from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(all_labels.numpy(), all_preds.numpy())

fig, ax = plt.subplots(figsize=(14, 12))
sns.heatmap(cm, ax=ax, cmap='Blues', square=True, cbar=True)
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
ax.set_title('Confusion Matrix — ModelNet40 Baseline')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
plt.show()

# ModelNet40 class names, in the standard label-index order
class_names = [
    'airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car',
    'chair', 'cone', 'cup', 'curtain', 'desk', 'door', 'dresser', 'flower_pot',
    'glass_box', 'guitar', 'keyboard', 'lamp', 'laptop', 'mantel', 'monitor',
    'night_stand', 'person', 'piano', 'plant', 'radio', 'range_hood', 'sink',
    'sofa', 'stairs', 'stool', 'table', 'tent', 'toilet', 'tv_stand', 'vase',
    'wardrobe', 'xbox'
]

cm_offdiag = cm.copy()
np.fill_diagonal(cm_offdiag, 0)

pairs = []
for i in range(cm_offdiag.shape[0]):
    for j in range(cm_offdiag.shape[1]):
        if cm_offdiag[i, j] > 0:
            pairs.append((cm_offdiag[i, j], class_names[i], class_names[j]))

pairs.sort(reverse=True)

print("Top 10 confused pairs (true → predicted, count):")
for count, true_cls, pred_cls in pairs[:10]:
    print(f"  {true_cls:15s} → {pred_cls:15s} : {count}")

    
