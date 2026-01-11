import pandas as pd
import numpy as np
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from transformers import BertTokenizer, BertModel
import torch
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Download NLTK data if not already present
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Initialize NLTK components (moved outside the function to avoid re-initialization on each call)
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Define the clean_text function (moved outside the main function for clarity and reusability)
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z ]', '', text)
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

# Load pre-trained BERT tokenizer and model (moved outside the function for efficiency)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model_bert = BertModel.from_pretrained('bert-base-uncased') # Renamed to avoid conflict with LogisticRegression 'model'

# Define the get_bert_embeddings function (moved outside the main function for clarity and reusability)
def get_bert_embeddings(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model_bert(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

def analyze_personality_data(dataset_path):
    # The rest of the workflow will go here
    print(f"Starting analysis for dataset: {dataset_path}")
