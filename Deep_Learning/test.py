import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import os
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from torch.utils.data import DataLoader, TensorDataset

# Set the visible GPU device (0 for the first GPU)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Set random seed for reproducibility
seed_value = 42
np.random.seed(seed_value)
torch.manual_seed(seed_value)
torch.cuda.manual_seed(seed_value)

# Load test dataset
shuffled_train_path = './data/trainset.txt'
shuffled_train_df = pd.read_csv(shuffled_train_path, delimiter='\t', header=None)
shuffled_test_path = './data/testset.txt'
shuffled_test_df = pd.read_csv(shuffled_test_path, delimiter='\t', header=None)

# Load compound and gene expression profiles
compound_expression_path = './data/CMap2020-chemical-profile.txt'
gene_expression_path = './data/CMap2020-genetic-profiles.txt'
compound_expression_df = pd.read_csv(compound_expression_path, delimiter='\t', header=None)
gene_expression_df = pd.read_csv(gene_expression_path, delimiter='\t', header=None)

# Create dictionaries for quick access to expression profiles
compound_expression_dict = dict(zip(compound_expression_df.iloc[:, 0], compound_expression_df.iloc[:, 1:].astype(np.float32).values))
gene_expression_dict = dict(zip(gene_expression_df.iloc[:, 0], gene_expression_df.iloc[:, 1:].astype(np.float32).values))

# Initialize StandardScaler for standardizing the continuous features
scaler = StandardScaler()

# Function to extract features and labels from the dataset
def extract_features_and_labels(df, scaler=None):
    compound_features = []
    protein_features = []
    other_features = []
    labels = []

    for _, row in df.iterrows():
        compound_name = row[0]
        gene_name = row[1]

        # Extract compound expression
        if compound_name in compound_expression_dict:
            compound_features.append(compound_expression_dict[compound_name])
        else:
            raise ValueError(f"Compound {compound_name} not found in expression data.")

        # Extract gene expression
        if gene_name in gene_expression_dict:
            protein_features.append(gene_expression_dict[gene_name])
        else:
            raise ValueError(f"Gene {gene_name} not found in expression data.")

        # Extract other features (9 cell lines + 4 continuous features)
        row_other_features = row[2:15].astype(np.float32).values

        # Separate continuous features (time, dose, protein_time, binding_score)
        continuous_features = row_other_features[9:]  # Columns from index 9 to 14

        # Apply standardization to continuous features
        if scaler:
            continuous_features = scaler.transform([continuous_features])[0]

        # Combine the cell lines (no scaling) with the standardized continuous features
        other_features.append(np.concatenate([row_other_features[:9], continuous_features]))

        # Labels
        labels.append(int(row[15]))

    return (np.array(compound_features, dtype=np.float32),
            np.array(protein_features, dtype=np.float32),
            np.array(other_features, dtype=np.float32),
            np.array(labels, dtype=np.int64))

# Extract features and labels for train and validation sets
# Fit the scaler on the training set first
train_continuous_features = shuffled_train_df.iloc[:, 2:15].values[:, 9:]  # Extract continuous features
scaler.fit(train_continuous_features)  # Fit scaler to training continuous features

compound_features_train, protein_features_train, other_features_train, labels_train = extract_features_and_labels(shuffled_train_df, scaler)
compound_features_test, protein_features_test, other_features_test, labels_test = extract_features_and_labels(shuffled_test_df, scaler)

# Device configuration for GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define model classes (must match the training model structure)
class MLP(nn.Module):
    """Multi-Layer Perceptron (MLP) Model."""
    def __init__(self, n_features, n_hidden_1, n_hidden_2, n_embedding, dropout_rate):
        super(MLP, self).__init__()
        self.dropout_rate = dropout_rate
        self.fc1 = nn.Linear(n_features, n_hidden_1)
        self.fc2 = nn.Linear(n_hidden_1, n_hidden_2)
        self.fc3 = nn.Linear(n_hidden_2, n_embedding)

    def forward(self, x):
        layer_1 = self.fc1(x)
        layer_1 = F.relu(layer_1)
        layer_1 = F.dropout(layer_1, p=self.dropout_rate, training=self.training)

        layer_2 = self.fc2(layer_1)
        layer_2 = F.relu(layer_2)
        layer_2 = F.dropout(layer_2, p=self.dropout_rate, training=self.training)
        return self.fc3(layer_2)

class SiameseNetwork(nn.Module):
    """Siamese Network that uses the MLP."""
    def __init__(self, n_features, n_hidden_1, n_hidden_2, n_embedding, dropout_rate):
        super(SiameseNetwork, self).__init__()
        self.mlp = MLP(n_features, n_hidden_1, n_hidden_2, n_embedding, dropout_rate)

    def forward(self, x1, x2):
        out1 = self.mlp(x1)
        out2 = self.mlp(x2)

        # Normalize the outputs
        pred1_norm = F.normalize(out1, dim=1)
        pred2_norm = F.normalize(out2, dim=1)

        # Compute the distance between two outputs
        distance = torch.norm(pred1_norm - pred2_norm, p=2, dim=1, keepdim=True)
        return distance, out1, out2

class OutputModel(nn.Module):
    """Output Model for final classification."""
    def __init__(self, n_hidden_3, n_classes):
        super(OutputModel, self).__init__()
        self.fc1 = nn.Linear(14, n_hidden_3)
        self.fc2 = nn.Linear(n_hidden_3, n_classes)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))

# Load the saved model
model_path = './data/SNGene-lr0.001_2048_512_256_dr0_l2_0.0001.pth'  # Replace with the path to your saved model
checkpoint = torch.load(model_path, map_location=device)

# Initialize models with the same architecture as during training
siamese_model = SiameseNetwork(
    n_features=978,
    n_hidden_1=2048,
    n_hidden_2=512,
    n_embedding=100,
    dropout_rate=0
).to(device)

output_model = OutputModel(n_hidden_3=4, n_classes=2).to(device)

# Load the model weights
siamese_model.load_state_dict(checkpoint['siamese_model_state_dict'])
output_model.load_state_dict(checkpoint['output_model_state_dict'])

# Set models to evaluation mode
siamese_model.eval()
output_model.eval()

# Create TensorDataset for test data
test_dataset = TensorDataset(torch.FloatTensor(compound_features_test), 
                             torch.FloatTensor(protein_features_test), 
                             torch.FloatTensor(other_features_test), 
                             torch.LongTensor(labels_test))

# Create DataLoader for test data
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=16, pin_memory=True)

# Test the model
test_preds_probs, test_labels = [], []
test_preds = []  # Store predicted classes

with torch.no_grad():  # Disable gradients for testing
    for x1_test_batch, x2_test_batch, other_features_test_batch, y_test_batch in test_loader:
        x1_test_batch = x1_test_batch.to(device)
        x2_test_batch = x2_test_batch.to(device)
        other_features_test_batch = other_features_test_batch.to(device)
        y_test_batch = y_test_batch.to(device)

        distance, out1, out2 = siamese_model(x1_test_batch, x2_test_batch)
        combined_input = torch.cat([distance, other_features_test_batch], dim=1)
        test_pred = output_model(combined_input)

        test_pred_prob = F.softmax(test_pred, dim=1)  # Get predicted probabilities
        test_preds_probs.extend(test_pred_prob.cpu().numpy()[:, 1])  # Store probabilities for the positive class
        _, test_predicted = torch.max(test_pred.data, 1)  # Get predicted classes
        test_preds.extend(test_predicted.cpu().numpy())
        test_labels.extend(y_test_batch.cpu().numpy())  # Store true labels

# Compute test metrics
test_accuracy = (np.array(test_preds) == np.array(test_labels)).mean()
test_precision = precision_score(test_labels, test_preds, average='binary')
test_recall = recall_score(test_labels, test_preds, average='binary')
test_f1 = f1_score(test_labels, test_preds, average='binary')
test_auc = roc_auc_score(test_labels, test_preds_probs)
test_auprc = average_precision_score(test_labels, test_preds_probs)

# Print test results
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Precision: {test_precision:.4f}")
print(f"Test Recall: {test_recall:.4f}")
print(f"Test F1 Score: {test_f1:.4f}")
print(f"Test AUC: {test_auc:.4f}")
print(f"Test AUPRC: {test_auprc:.4f}")