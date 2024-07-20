# ClinicalBert: Assessment of Colon Cancer Risk Using AI-based NLP Models

Welcome to the ClinicalBert project repository. This project focuses on assessing colon cancer risk using advanced AI-based NLP models. Developed as part of the Barrett Research Fellowship, the project leverages state-of-the-art techniques in natural language processing and deep learning.

## Table of Contents

- [Introduction](#introduction)
- [Project Objectives](#project-objectives)
- [Technology and Tools](#technology-and-tools)
- [Setup](#setup)
- [Usage](#usage)
  - [Training the Model](#training-the-model)
  - [Using the Model for Prediction](#using-the-model-for-prediction)
- [File Structure](#file-structure)
- [License](#license)
- [Contact](#contact)

## Introduction

ClinicalBert is a project aimed at assessing the risk of colon cancer using AI-based Natural Language Processing (NLP) models. The project employs advanced techniques such as Convolutional Neural Networks (CNN) and BERT for sequence classification, achieving significant performance optimizations and data processing efficiencies.

## Project Objectives

1. Implement NLP techniques for colon cancer risk assessment.
2. Utilize CNNs and BERT models to analyze patient data.
3. Optimize model performance and data processing efficiency.

## Technology and Tools

- **Frameworks**: TensorFlow, PyTorch
- **NLP**: Spacy, BERT (Bio_ClinicalBERT)
- **Languages**: Python
- **Libraries**: Pandas, Scikit-learn, Matplotlib, Transformers

## Setup

1. **Clone the repository**:

    ```sh
    git clone https://github.com/Sreechandh22/ClinicalBert.git
    cd ClinicalBert
    ```

2. **Create a virtual environment and activate it**:

    ```sh
    python3 -m venv venv
    source venv/bin/activate
    ```

3. **Install the required packages**:

    ```sh
    pip install -r requirements.txt
    ```

4. **Download the pre-trained Bio_ClinicalBERT model**:

    The pre-trained model can be downloaded from [Hugging Face](https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT) and placed in the appropriate directory.

## Usage

### Training the Model

1. **Prepare the data**:
   - Ensure the training and validation data files (`train_preprocessed.dat`, `val_preprocessed.dat`) are in the correct location.

2. **Run the training script**:
   - Execute the `app.py` script to train the model:

    ```sh
    python app.py
    ```

### Using the Model for Prediction

1. **Predict Diagnosis**:
   - Use the trained BERT model to predict the diagnosis for a given clinical text.

    ```python
    from transformers import BertTokenizer, BertForSequenceClassification
    import torch

    def predict_diagnosis(text, model, tokenizer, max_length=128):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

        with torch.no_grad():
            model.eval()
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predicted_label = torch.argmax(logits, dim=1).item()

        return predicted_label

    # Example usage
    example_text = "Patient presents with a cough and high fever for the last three days. History of asthma."
    model = BertForSequenceClassification.from_pretrained('emilyalsentzer/Bio_ClinicalBERT').to(device)
    tokenizer = BertTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
    predicted_label = predict_diagnosis(example_text, model, tokenizer)
    print("Predicted Diagnosis Label:", predicted_label)
    ```

## File Structure

    ClinicalBert/
    ├── .vscode/
    │ ├── settings.json
    ├── README.md
    ├── app.py
    ├── letscheck.xlsx
    ├── train_preprocessed.dat
    ├── val_preprocessed.dat
    ├── requirements.txt


- **.vscode/**: Contains settings for Visual Studio Code.
- **README.md**: Project overview and setup guide.
- **app.py**: Main application file for training and evaluating the model.
- **letscheck.xlsx**: Data file in Excel format.
- **train_preprocessed.dat**: Preprocessed training data.
- **val_preprocessed.dat**: Preprocessed validation data.
- **requirements.txt**: List of dependencies required to run the project.

## License

This project is licensed under the MIT License.

---

## Contact

For any inquiries or collaboration opportunities, please contact sreechandh2204@gmail.com
