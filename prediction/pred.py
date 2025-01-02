import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import csv
import os

class AntimicrobialPeptideDataset(Dataset):
    def __init__(self, input_ids, attention_mask, labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'labels': self.labels[idx]
        }

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.Conv1d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv1d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv1d(2, 1, kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_planes, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)
    def forward(self, x):
        out = x * self.channel_attention(x)
        out = out * self.spatial_attention(out)
        return out

class CNNBiLSTM_CBAM_Model(nn.Module):
    def __init__(self, max_time_steps, input_size=480, num_classes=13):
        super(CNNBiLSTM_CBAM_Model, self).__init__()
        self.conv1d_1 = nn.Conv1d(in_channels=input_size, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv1d_2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.6)
        self.bilstm1 = nn.LSTM(input_size=128, hidden_size=128, num_layers=2, batch_first=True, bidirectional=True)
        self.cbam = CBAM(in_planes=256)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear((max_time_steps // 4) * 256, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x, attention_mask=None):
        x = x.permute(0, 2, 1)
        x = self.conv1d_1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.conv1d_2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = x.permute(0, 2, 1)
        x, _ = self.bilstm1(x)
        x = x.permute(0, 2, 1)
        x = self.cbam(x)
        x = x.permute(0, 2, 1)
        x = self.flatten(x)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x


device = torch.device('cpu')
print(f"Using device: {device}")


data_file = r"/content/deep-AMPpred/prediction/external_data/anti_mammalian_cells_data.csv"
data = pd.read_csv(data_file)
sequences = data.iloc[:, 0].tolist()
labels = data.iloc[:, 1].values



# tokenizer = AutoTokenizer.from_pretrained('/content/drive/MyDrive/predict/Rostlab/esm2_t12_35M_UR50D')
# pretrained_model = AutoModel.from_pretrained('/content/drive/MyDrive/predict/Rostlab/esm2_t12_35M_UR50D')
# Load the ESM-2 model and tokenizer directly from Hugging Face's online model repository
tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t12_35M_UR50D")
pretrained_model = AutoModel.from_pretrained("facebook/esm2_t12_35M_UR50D")
pretrained_model.to(device)


max_length = 100
encodings = tokenizer.batch_encode_plus(
    sequences, add_special_tokens=True, padding="longest", truncation=True, max_length=max_length
)
input_ids = torch.tensor(encodings['input_ids'])
attention_mask = torch.tensor(encodings['attention_mask'])
labels = torch.tensor(labels, dtype=torch.float32)


dataset = AntimicrobialPeptideDataset(input_ids, attention_mask, labels)
data_loader = DataLoader(dataset, batch_size=64, shuffle=False)


output_dir = "/content/deep-AMPpred/prediction/models"
model_save_path = os.path.join(output_dir, "anti_mammalian_cells_model.pth")
model = CNNBiLSTM_CBAM_Model(
    max_time_steps=input_ids.size(1), input_size=480, num_classes=1
).to(device)


model.load_state_dict(torch.load(model_save_path, map_location=device))
model.eval()


y_true, y_pred, y_prob = [], [], []
with torch.no_grad():
    for batch in data_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = pretrained_model(input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        logits = model(sequence_output)
        probs = torch.sigmoid(logits).squeeze()
        preds = (probs > 0.5).long()
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())
        y_prob.extend(probs.cpu().numpy())

accuracy = accuracy_score(y_true, y_pred)
auc = roc_auc_score(y_true, y_prob)
print(f"Accuracy: {accuracy}")
print(f"AUC: {auc}")
