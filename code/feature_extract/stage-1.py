import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig
import pandas as pd
import numpy as np
import pickle
from sklearn.decomposition import PCA


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the ESM model and tokenizer
tokenizer_name = 'Rostlab/esm2_t12_35M_UR50D'
model_name = 'Rostlab/esm2_t12_35M_UR50D'

tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, do_lower_case=False)

config = AutoConfig.from_pretrained(model_name, output_hidden_states=True)

model = AutoModel.from_pretrained(model_name, config=config).to(device)

# Define a function to extract features from a protein sequence
def extract_features(sequence, tokenizer, model, device, max_len=700):
    # Tokenize the input sequence, with padding and truncation to a maximum length
    inputs = tokenizer(sequence, return_tensors='pt', padding='max_length', truncation=True, max_length=max_len)

    inputs = {key: val.to(device) for key, val in inputs.items()}
    # Disable gradient calculations as we're in inference mode
    with torch.no_grad():
        outputs = model(**inputs)

    hidden_states = outputs.hidden_states

    feature = torch.stack(hidden_states[-4:]).mean(0).squeeze(0).mean(dim=0).cpu().numpy()
    return feature

# Load the dataset from a CSV file
df = pd.read_csv('data/benchmark/stage-1.csv', index_col=0)

df = df.sample(frac=1, random_state=0)

# Apply the feature extraction function to each sequence in the dataset
features = df['aa_seq'].apply(lambda seq: extract_features(seq, tokenizer, model, device))

features = np.vstack(features)


pca = PCA(n_components=100)
features_reduced = pca.fit_transform(features)

# Extract labels from the dataframe and convert them to integer values
labels = df['AMP'].astype(int).values

# Save the reduced features and labels as a pickle file for later use
with open('features.pkl', 'wb') as f:
    pickle.dump((features_reduced, labels), f)
