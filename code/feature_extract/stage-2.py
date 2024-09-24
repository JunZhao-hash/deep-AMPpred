import pandas as pd
import torch
from transformers import EsmTokenizer, EsmModel
import numpy as np
from sklearn.preprocessing import StandardScaler

# Read the CSV file containing protein sequences
file_path = 'data/benchmark/stage-2.csv'
df = pd.read_csv(file_path)

# Load the ESM tokenizer and model for feature extraction
tokenizer = EsmTokenizer.from_pretrained("Rostlab/esm2_t12_35M_UR50D")
model = EsmModel.from_pretrained("Rostlab/esm2_t12_35M_UR50D", output_hidden_states=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = model.to(device)


def extract_features(sequence):
    # Tokenize the input sequence with padding, truncation, and a max length of 700
    inputs = tokenizer(sequence, return_tensors='pt', truncation=True, padding='max_length', max_length=700).to(device)


    with torch.no_grad():
        outputs = model(**inputs)

    hidden_states = outputs.hidden_states

    last_4_layers = torch.stack(hidden_states[-4:]).mean(dim=0)

    avg_pool = last_4_layers.mean(dim=1)
    max_pool = last_4_layers.max(dim=1).values

    pooled_output = torch.cat((avg_pool, max_pool), dim=1).squeeze().cpu().numpy()
    return pooled_output

X = df['Sequence'].apply(extract_features).tolist()

X = np.array(X)

scaler = StandardScaler()
X = scaler.fit_transform(X)

# Extract labels from the dataframe, excluding the 'Sequence' column
y = df.drop(columns=['Sequence']).values

# Save the standardized features and corresponding labels as .npy files
np.save('stage-2-features.npy', X)
np.save('stage-2-labels.npy', y)


print("Feature extraction and saving complete.")
