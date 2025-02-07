import os
import torch
import pickle
import pandas as pd
from fuzzywuzzy import process
from app.model import TransactionClassifier  # Importing the model class

# Base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Paths
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'trained_model.pth')
VOCAB_PATH = os.path.join(BASE_DIR, 'model', 'tokenizer.pkl')
DATA_PATH = os.path.join(BASE_DIR, 'data', 'synthetic_transactions.csv')

def load_model_and_vocab():

    # Load dataset to extract unique categories
    df = pd.read_csv(DATA_PATH)
    category_list = sorted(df['merchant_category'].unique().tolist())
    known_merchants = df['merchant_name'].unique().tolist()  # For fuzzy matching

    # Load vocabulary
    with open(VOCAB_PATH, 'rb') as f:
        vocab = pickle.load(f)

    # Define model with the same architecture used during training
    vocab_size = len(vocab) + 1  # Account for padding_idx=0
    embed_dim = 32  
    hidden_size = 64  
    output_size = len(df["merchant_category"].unique())  

    # Initialize the model architecture and load model
    model = TransactionClassifier(vocab_size, embed_dim, hidden_size, output_size)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()

    return model, vocab, category_list, known_merchants

# Tokenize merchant names
def tokenize(merchant_name, vocab):
    merchant_name = merchant_name.lower().replace("'", "").replace("-", "").replace("&", "")  # Remove apostrophes, dashes, and ampersands
    return [vocab.get(word, 0) for word in merchant_name.split()]

# Fuzzy match with known merchants
def fuzzy_match_merchant(merchant_name, known_merchants, threshold=85):
    match, score = process.extractOne(merchant_name, known_merchants)
    return match if score >= threshold else None

def get_closest_match(merchant_name, known_merchants, threshold=80):
    """
    Finds the closest merchant name from the known merchants list using fuzzy matching.
    Args:
        merchant_name (str): The input merchant name to match.
        known_merchants (list): List of known merchants from the dataset.
        threshold (int): The minimum match score required to consider it a valid match.

    Returns:
        str: The closest matching merchant name, or 'Unknown Merchant' if no close match is found.
    """
    match, score = process.extractOne(merchant_name, known_merchants)
    if score >= threshold:
        return match
    return "Unknown Merchant"

def predict_category(merchant_name, model, vocab, category_list, known_merchants):

    tokens = tokenize(merchant_name, vocab)
    tokens_tensor = torch.tensor([tokens], dtype=torch.long)

    # Fuzzy Matching Trigger
    if all(token == 0 for token in tokens):
        closest_match = get_closest_match(merchant_name, known_merchants)

        # Tokenize the fuzzy matched name
        tokens = tokenize(closest_match, vocab)

        if all(token == 0 for token in tokens):
            return "Unknown"

    with torch.no_grad():
        output = model(tokens_tensor)
        probabilities = torch.softmax(output, dim=1)  # Softmax for probabilities
        confidence, predicted = torch.max(probabilities, 1)

    return {"category": category_list[predicted.item()], "confidence": confidence.item()}