import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, matthews_corrcoef
import pandas as pd
from model import ProteinDataset, stage_2model
# Load data
X = np.load('stage-2-features.npy')
y = np.load('stage-2-labels.npy')
# Define category name
target_names = [
    'antibacterial', 'antibiofilm', 'anticancer', 'anticandida', 'antifungal', 'antigram-negative',
    'antigram-positive', 'antihiv', 'antimalarial', 'antimrsa', 'antiparasitic', 'antiplasmodial',
    'antiprotozoal', 'antitb', 'antiviral', 'anti_mammalian_cells', 'anurandefense', 'chemotactic',
    'cytotoxic', 'endotoxin', 'hemolytic', 'insecticidal'
]


def compute_metrics(y_true, y_pred, y_pred_prob):
    metrics = {}
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics['Accuracy'] = accuracy_score(y_true, y_pred)
    metrics['Sensitivity (Recall)'] = recall_score(y_true, y_pred)
    metrics['Precision'] = precision_score(y_true, y_pred)
    metrics['Specificity'] = tn / (tn + fp) if (tn + fp) != 0 else 0
    metrics['MCC'] = matthews_corrcoef(y_true, y_pred)
    metrics['F1 Score'] = f1_score(y_true, y_pred)
    metrics['AUC-ROC'] = roc_auc_score(y_true, y_pred_prob)
    metrics['GMean'] = np.sqrt(metrics['Sensitivity (Recall)'] * metrics['Specificity'])
    return metrics
# Five-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
all_metrics = []
all_results = []
for fold, (train_index, test_index) in enumerate(kf.split(X)):
    print(f"Fold {fold + 1}")
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    train_dataset = ProteinDataset(X_train, y_train)
    test_dataset = ProteinDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = stage_2model(input_size=X.shape[1], num_classes=y.shape[1]).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 40
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            features = batch['features'].to(device)
            labels = batch['labels'].to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}')
    # Save model
    model_save_path = f'stage-2Model_fold_{fold + 1}.pth'
    torch.save(model.state_dict(), model_save_path)
    print(f"Model for fold {fold + 1} saved to {model_save_path}")
    # Evaluation model
    model.eval()
    y_true = []
    y_pred = []
    y_pred_prob = []
    with torch.no_grad():
        for batch in test_loader:
            features = batch['features'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(features)
            y_true.append(labels.cpu().numpy())
            y_pred.append((outputs.cpu().numpy() > 0.5).astype(int))
            y_pred_prob.append(outputs.cpu().numpy())
    y_true = np.vstack(y_true)
    y_pred = np.vstack(y_pred)
    y_pred_prob = np.vstack(y_pred_prob)
    for i, name in enumerate(target_names):
        metrics = compute_metrics(y_true[:, i], y_pred[:, i], y_pred_prob[:, i])
        metrics['Category'] = name
        metrics['Fold'] = fold + 1
        all_metrics.append(metrics)
    results = pd.DataFrame({
        'True Labels': [list(labels) for labels in y_true],
        'Predicted Labels': [list(preds) for preds in y_pred],
        'Predicted Probabilities': [list(probs) for probs in y_pred_prob]
    })
    results['Fold'] = fold + 1
    all_results.append(results)
# Save results
all_results_df = pd.concat(all_results, axis=0)

new_columns = []
for col in ['True Labels', 'Predicted Labels', 'Predicted Probabilities']:
    for name in target_names:
        new_columns.append(f"{name} - {col}")
new_columns.append("Fold")


flattened_results_df = []
for index, row in all_results_df.iterrows():
    flattened_row = []
    for col in ['True Labels', 'Predicted Labels', 'Predicted Probabilities']:
        flattened_row.extend(row[col])
    flattened_row.append(row["Fold"])
    flattened_results_df.append(flattened_row)
flattened_results_df = pd.DataFrame(flattened_results_df, columns=new_columns)
# flattened_results_df.to_csv('all_results.csv', index=False)
metrics_df = pd.DataFrame(all_metrics)
# metrics_df.to_csv('metrics.csv', index=False)

average_metrics_df = metrics_df.groupby('Category').mean().reset_index()
average_metrics_df.to_csv('average_metrics.csv', index=False)
print("Metrics and results saved.")
