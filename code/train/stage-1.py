import numpy as np
import random
import torch
from transformers import AutoTokenizer, AutoModel
import pickle
from sklearn.model_selection import KFold
import pandas as pd
from torch.utils.data import Dataset, DataLoader, Subset
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix, matthews_corrcoef, roc_curve
from model import stage_1model, ProteinDataset

random.seed(1)
np.random.seed(17)
torch.manual_seed(153)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
#Load data from .pkl file
file_path = 'features.pkl'
with open(file_path, 'rb') as f:
    features, labels = pickle.load(f)

scaler = StandardScaler()
features = scaler.fit_transform(features)

def train_model(model, train_loader, criterion, optimizer, scheduler, device, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
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
        scheduler.step(running_loss / len(train_loader))
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

def evaluate_model(model, test_loader, device):
    model.eval()
    all_labels = []
    all_probs = []
    all_preds = []
    with torch.no_grad():
        for batch in test_loader:
            features = batch['features'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(features)
            probs = nn.Softmax(dim=1)(outputs)[:, 1]
            preds = (probs >= 0.5).long()
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
    mcc = matthews_corrcoef(all_labels, all_preds)
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    print(f'Optimal Threshold: {optimal_threshold}')
    auc_roc = roc_auc_score(all_labels, all_probs)
    metrics = {
        'Accuracy': accuracy,
        'Sensitivity': sensitivity,
        'Precision': precision,
        'Specificity': specificity,
        'MCC': mcc,
        'F1 Score': f1,
        'AUC-ROC': auc_roc,
        'Recall': recall,
    }

    return metrics, all_probs, all_preds, all_labels

max_time_steps = features.shape[1]
input_size = features.shape[1]
print(f"Max time steps: {max_time_steps}, Input size: {input_size}")
#Five-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=0)
fold_results = []
all_fold_probs = []
all_fold_labels = []
all_fold_preds = []
for fold, (train_idx, test_idx) in enumerate(kf.split(features)):
    print(f'Fold {fold+1}')
    # Divide training set and test set
    X_train, X_test = features[train_idx], features[test_idx]
    y_train, y_test = labels[train_idx], labels[test_idx]
    train_dataset = ProteinDataset(X_train, y_train)
    test_dataset = ProteinDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    # Initialize model, loss function and optimizer
    model = stage_1model(max_time_steps, input_size).to(device)
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=0.001)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
    # train model
    train_model(model, train_loader, criterion, optimizer, scheduler, device, num_epochs=15)
    # 评估模型
    fold_metrics, fold_probs, fold_preds, fold_labels = evaluate_model(model, test_loader, device)
    fold_results.append(fold_metrics)
    all_fold_probs.extend(fold_probs)
    all_fold_preds.extend(fold_preds)
    all_fold_labels.extend(fold_labels)
    # Evaluation model
    model_path = f'stage-1model_fold_{fold+1}.pth'
    torch.save(model.state_dict(), model_path)
    print(f'Model for fold {fold+1} saved to {model_path}')
# Summarize results across all folds
all_metrics = pd.DataFrame(fold_results)
mean_metrics = all_metrics.mean()
std_metrics = all_metrics.std()
print("Mean metrics:")
print(mean_metrics)
print("\nStandard deviation of metrics:")
print(std_metrics)

all_metrics.to_csv('kfold_results.csv', index=False)
mean_metrics.to_csv('mean_metrics.csv', index=True)

predictions_df = pd.DataFrame({
    'True Label': all_fold_labels,
    'Predicted Probability': all_fold_probs,
    'Predicted Label': all_fold_preds
})
predictions_df.to_csv('all_predictions.csv', index=False)

