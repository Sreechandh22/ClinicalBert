import torch
import spacy
import re
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from tqdm import tqdm
import numpy as np
import random
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from torch.nn import CrossEntropyLoss
from sklearn.metrics import classification_report

nlp = spacy.load("en_core_web_sm")

# Full dictionary
SYNONYM_DICT = {
    "doctor": ["physician", "medic", "MD", "surgeon", "oncologist", "specialist"],
    "patient": ["client", "case", "individual", "subject", "sufferer"],
    "hospital": ["medical center", "clinic", "healthcare facility", "infirmary", "hospital ward"],
    "diagnosis": ["assessment", "evaluation", "finding", "diagnostic conclusion", "medical opinion"],
    "cancer": ["malignancy", "neoplasm", "tumor", "carcinoma", "cancerous growth"],
    "colorectal": ["colon", "rectal", "bowel", "intestinal", "colonic", "rectum"],
    "polyp": ["adenoma", "growth", "mass", "tumorous growth", "colonic polyp"],
    "chemotherapy": ["chemo", "antineoplastic therapy", "cancer treatment", "oncology treatment", "cytotoxic therapy"],
    "radiation": ["radiotherapy", "radiation therapy", "irradiation", "radiation oncology", "x-ray therapy"],
    "surgery": ["operation", "surgical procedure", "resection", "surgical intervention", "surgical removal"],
    "symptom": ["sign", "indication", "manifestation", "clinical sign", "physical sign"],
    "bleeding": ["hemorrhage", "blood loss", "spotting", "rectal bleeding", "bloody stool"],
    "pain": ["discomfort", "ache", "soreness", "abdominal pain", "cramping"],
    "treatment": ["therapy", "management", "care", "intervention", "medical care"],
    "biopsy": ["tissue sampling", "histologic examination", "pathology test", "tissue analysis", "cellular analysis"],
    "metastasis": ["spread", "secondary growth", "distant spread", "cancer spreading", "tumor migration"],
    "remission": ["abatement", "subsidence", "regression", "disease control", "symptom relief"],
    "screening": ["early detection", "preventive testing", "cancer screening", "colonoscopy", "medical screening"],
    "colonoscopy": ["endoscopic examination", "bowel scope", "colonic examination", "rectal examination", "colon examination"],
    "risk factor": ["predisposition", "contributing factor", "health risk", "causative factor", "influencing factor"],
    "lifestyle": ["diet", "exercise", "habits", "health behavior", "personal habits"],
    "genetics": ["heredity", "genetic makeup", "DNA", "family history", "inheritance"],
    "staging": ["cancer staging", "disease progression", "tumor stage", "progression assessment", "TNM staging"]
}

def augment_text(text):
    words = text.split()
    for i in range(len(words)):
        if words[i] in SYNONYM_DICT and random.random() < 0.2:  # 20% chance of word being replaced
            words[i] = random.choice(SYNONYM_DICT[words[i]])  # Replace with a synonym
    return ' '.join(words)

# Set device for model computations
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def extract_lab_results(text):
    lab_results = {}
    # Regular expressions for lab tests
    patterns = {
    "white blood cell count": r"white blood cell count\s*:\s*(\d+\.?\d*)",
    "hemoglobin": r"hemoglobin\s*:\s*(\d+\.?\d*)",
    "hematocrit": r"hematocrit\s*:\s*(\d+\.?\d*)",
    "platelets": r"platelets\s*:\s*(\d+)",
    "bilirubin": r"bilirubin\s*:\s*(\d+\.?\d*)",
    "AST": r"AST\s*:\s*(\d+\.?\d*)",
    "ALT": r"ALT\s*:\s*(\d+\.?\d*)",
    "alkaline phosphatase": r"alkaline phosphatase\s*:\s*(\d+\.?\d*)",
    "CEA": r"CEA\s*:\s*(\d+\.?\d*)",  # Carcinoembryonic antigen, a tumor marker
    # More patterns as relevant to your dataset
}

    for test, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            lab_results[test] = float(match.group(1))
        else:
            lab_results[test] = None  # or a default value

    return lab_results

# Function to load and augment data
def load_clinical_data(data_file, sample_size=1.0):
    df = pd.read_csv(data_file, encoding='ISO-8859-1')
    if sample_size < 1.0:
        df = df.sample(frac=sample_size)
    # Apply text augmentation and lab results extraction
    df['text'] = df['text'].astype(str).apply(augment_text)
    df['lab_results'] = df['text'].apply(extract_lab_results)
    # Combine text and lab results for feature extraction
    df['combined_features'] = df.apply(lambda row: row['text'] + ' ' + ' '.join([f'{k}:{v}' for k, v in row['lab_results'].items()]), axis=1)
    texts = df['combined_features'].tolist()
    labels = df['label'].tolist()
    return texts, labels

def extract_medical_features(text):
    # Enhanced to extract more relevant features for diagnosis
    doc = nlp(text)
    features = {
        "symptoms": [],
        "medical_conditions": [],
        "family_history": [],
        " ": []
    }
    for ent in doc.ents:
        if ent.label_ in ["SYMPTOM", "DISEASE"]:
            features["symptoms"].append(ent.text)
        elif ent.label_ == "FAMILY_MEMBER":
            features["family_history"].append(ent.text)
        # Add more conditions as needed
    return features


# Placeholder functions for additional feature extraction
def extract_family_history(text):
    # Implement logic to extract family history
    return "family_history_feature"

def extract_lifestyle_factors(text):
    # Implement logic to extract lifestyle factors
    return "lifestyle_factors_feature"

def extract_clinical_assessments(text):
    # Implement logic to extract clinical assessments
    return "clinical_assessments_feature"


# Dataset class for feature extraction
class FeatureExtractionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text, 
            add_special_tokens=True, 
            max_length=self.max_length, 
            padding='max_length', 
            truncation=True, 
            return_attention_mask=True
        )
        return {
            'input_ids': torch.tensor(encoding['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(encoding['attention_mask'], dtype=torch.long),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Configuration variables
bert_model_name = 'emilyalsentzer/Bio_ClinicalBERT'
max_length = 128  # Adjusted max length for BERT
batch_size = 16  # Adjusted batch size for feature extraction

# Instantiate tokenizer
tokenizer = BertTokenizer.from_pretrained(bert_model_name)

# Load text data
train_texts, train_labels = load_clinical_data("C:\\Users\\sreec\\OneDrive\\Desktop\\archive\\training_data_processed.csv")
print(len(train_texts), len(train_labels))  # Check the size of the loaded dataset


val_texts, val_labels = load_clinical_data("C:\\Users\\sreec\\OneDrive\\Desktop\\archive\\validation_data_processed.csv")
if any(pd.isnull(val_labels)):
    print("NaN values found in validation labels")
# Convert to a DataFrame for easier handling
val_df = pd.DataFrame({
    'text': val_texts,
    'label': val_labels
})

# Drop rows where label is NaN
val_df = val_df.dropna(subset=['label'])

# Extract texts and labels again
val_texts = val_df['text'].tolist()
val_labels = val_df['label'].tolist()


# Create datasets
train_dataset = FeatureExtractionDataset(train_texts, train_labels, tokenizer, max_length)
val_dataset = FeatureExtractionDataset(val_texts, val_labels, tokenizer, max_length)

# Create dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Load BERT model
num_diagnosis_labels = len(set(train_labels))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_model_name = 'emilyalsentzer/Bio_ClinicalBERT'

# Instantiate tokenizer
tokenizer = BertTokenizer.from_pretrained(bert_model_name)

# Load BERT model for sequence classification
model = BertForSequenceClassification.from_pretrained(bert_model_name, num_labels=num_diagnosis_labels).to(device)

def predict_diagnosis(text, model, tokenizer, max_length):
    """
    Predict the diagnosis for a given clinical text.

    Parameters:
    text (str): A string containing the clinical text.
    model (BertForSequenceClassification): The trained BERT model.
    tokenizer (BertTokenizer): The tokenizer used with the BERT model.
    max_length (int): Maximum length of the tokenized input.

    Returns:
    int: The predicted label for the diagnosis.
    """

    # Preprocess and tokenize the text
    encoded_text = tokenizer.encode_plus(
        text, 
        add_special_tokens=True, 
        max_length=max_length, 
        padding='max_length', 
        truncation=True, 
        return_attention_mask=True,
        return_tensors='pt'
    )

    input_ids = encoded_text['input_ids'].to(device)
    attention_mask = encoded_text['attention_mask'].to(device)

    # Predict using the model
    with torch.no_grad():
        model.eval()  # Evaluation mode
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predicted_label = torch.argmax(logits, dim=1).item()

    return predicted_label

# Example usage
example_text = "Patient presents with a cough and high fever for the last three days. History of asthma."
predicted_label = predict_diagnosis(example_text, model, tokenizer, max_length)
print("Predicted Diagnosis Label:", predicted_label)

# Function to extract features using the model
def extract_features(dataloader, model, device):
    model.eval()  # Put model in evaluation mode
    features = []
    for batch in tqdm(dataloader, desc="Extracting features"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        features.append(logits.detach().cpu().numpy())
    return np.vstack(features)  # Stack to create a single array for all features

# Extract features for train and validation sets
train_features = extract_features(train_dataloader, model, device)
val_features = extract_features(val_dataloader, model, device)

# Set up optimizer and loss function
optimizer = AdamW(model.parameters(), lr=2e-5)
loss_function = CrossEntropyLoss()

# Number of training epochs (authors recommend between 2 and 4)
epochs = 3

# Training loop
model.train()
for epoch in range(epochs):
    total_loss = 0
    for batch in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{epochs}"):
        # Clear any previously calculated gradients
        model.zero_grad()

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # Forward pass
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        # Compute loss
        loss = loss_function(logits.view(-1, num_diagnosis_labels), labels.view(-1))
        total_loss += loss.item()

        # Backward pass
        loss.backward()
        optimizer.step()

    # Print average loss for the epoch
    print(f"Average loss in epoch {epoch + 1}: {total_loss / len(train_dataloader)}")

# Save the model
torch.save(model.state_dict(), 'bert_model_for_diagnosis.pth')

# Evaluate the model
def evaluate_model(model, dataloader):
    model.eval()
    predictions, true_labels = [], []
    for batch in tqdm(dataloader, desc="Evaluating"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        logits = outputs.logits
        predictions.extend(torch.argmax(logits, dim=1).tolist())
        true_labels.extend(labels.tolist())
    return true_labels, predictions

true_labels, predictions = evaluate_model(model, val_dataloader)
print("Validation Accuracy:", accuracy_score(true_labels, predictions))
print(classification_report(true_labels, predictions))
