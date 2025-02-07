# Imports for dataset and model
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# UI imports
from colorama import Fore, Style

# Other imports
import sys
import os

# BUILD VOCAB - assigns id for each word.
def build_vocab(descriptions):
    vocab = {}
    for _ in descriptions:
        for word in _.lower().split():
            if word not in vocab:
                vocab[word] = len(vocab) + 1
    return vocab

# TOKENIZE - assigns a token for each word.
def tokenize(description, vocab):
    return [vocab.get(word, 0) for word in description.lower().split()]

# PREPARE DATA - converts tokens into tensors.
def prepare_data(df, vocab, category_to_idx):
    descriptions = [torch.tensor(tokenize(desc, vocab), dtype=torch.long) for desc in df['merchant']]
    amounts = torch.tensor(df['amount'].values, dtype=torch.float32).unsqueeze(1)
    labels = torch.tensor([category_to_idx[cat] for cat in df['merchant_category']], dtype=torch.long)
    return descriptions, amounts, labels

# FINANCEDATASET - Class for dataset.
class FinanceDataset(Dataset):
    def __init__(self, descriptions, amounts, labels, max_len=10):
        self.descriptions = descriptions
        self.amounts = amounts
        self.labels = labels
        self.max_len = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        padded_desc = torch.zeros(self.max_len, dtype=torch.long)
        desc = self.descriptions[idx]
        padded_desc[:min(len(desc), self.max_len)] = desc[:self.max_len]

        return padded_desc, self.amounts[idx], self.labels[idx]

# FINANCECATEGORIZER - Class for the model.
class FinanceCategorizer(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_categories):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc = nn.Linear(embed_dim + 1, num_categories)

    def forward(self, descriptions, amounts):
        embedded = self.embedding(descriptions).mean(dim=1)
        x = torch.cat((embedded, amounts), dim=1)
        return self.fc(x)

# TRAIN MODEL - Trains model if pth file doesn't exist.
def train_model(model, train_loader, epochs=10, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for desc_batch, amt_batch, label_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(desc_batch, amt_batch)
            loss = criterion(outputs, label_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")

# INITIALIZE MODEL AND DATA - loads the dataset, build vocab and categories, and establish the model.
def initialize_model_and_data(data_path):
    # Load dataset and build vocab and categories dynamically
    df = pd.read_csv(data_path)
    vocab = build_vocab(df['merchant'])
    category_list = df['merchant_category'].unique().tolist()
    category_to_idx = {cat: idx for idx, cat in enumerate(category_list)}

    # Define model parameters
    vocab_size = len(vocab) + 1
    embed_dim = 50
    num_categories = len(category_list)

    # Initialize FinanceCategorizer model
    model = FinanceCategorizer(vocab_size, embed_dim, num_categories)

    return model, df, vocab, category_list, category_to_idx

# LOAD OR TRAIN MODEL - Loads pre-existing model or trains a brand new one.
def load_or_train_model(data_path="synthetic_fraud_data.csv", model_path="finance_model.pth"):
    # Initialize model and data
    model, df, vocab, category_list, category_to_idx = initialize_model_and_data(data_path)

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.eval()
        print(Fore.GREEN + "Model loaded successfully! Loading program...\n" + Style.RESET_ALL)

    else:
        print(Fore.YELLOW + "No model found. Training a new model...\n" + Style.RESET_ALL)

        # Split data and prepare
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
        train_desc, train_amt, train_labels = prepare_data(train_df, vocab, category_to_idx)
        train_loader = DataLoader(FinanceDataset(train_desc, train_amt, train_labels), batch_size=64, shuffle=True)

        # Train model
        train_model(model, train_loader, epochs=10)

        # Save trained model
        torch.save(model.state_dict(), model_path)
        print(Fore.GREEN + f"\nModel trained successfully and saved as {model_path}. Loading program...\n" + Style.RESET_ALL)

    return model, vocab, category_list

# MAIN
def main():
    # Load welcome message and instructions
    welcome_message()

    # Load or train data
    model, vocab, category_list = load_or_train_model()

    while True:
        try:
            description = get_transaction_description()
            amount = get_transaction_amount()
            category = categorize_transaction(model, description, amount, vocab, category_list)
            print(Fore.CYAN + f"Predicted category: {category}\n" + Style.RESET_ALL)
        except EOFError:
            sys.exit(Fore.GREEN + "\n\nThank you for using Simple Transaction Categorizer!\n" + Style.RESET_ALL)

# WELCOME MESSAGE - Shows welcome message when running the program.
def welcome_message():
    print(Fore.GREEN + "\n------------------------------")
    print("      SIMPLE TRANSACTION")
    print("         CATEGORIZER")
    print("------------------------------\n" + Style.RESET_ALL)
    print("INSTRUCTIONS")
    print("1. Enter the transaction description.")
    print("2. Enter the transaction amount without the dollar ($) sign.")
    print("3. The model will then automatically categorize the transaction.")
    print("4. To exit the program, press CTRL+D at any time.\n")

# GET TRANSACTION DESCRIPTION - Gets and validates transaction description.
def get_transaction_description():
    while True:
        description = input("Enter transaction description: ").strip()
        if description:
            return description
        print(Fore.RED + "Description cannot be empty." + Style.RESET_ALL)

# GET TRANSACTION AMOUNT - Gets and validates transaction amount.
def get_transaction_amount():
    while True:
        try:
            amount = float(input("Enter transaction amount: ").strip())
            return amount
        except ValueError:
            print(Fore.RED + "Amount is not a number." + Style.RESET_ALL)

# CATEGORIZE TRANSACTION - Predicts category for a given transaction.
def categorize_transaction(model, description, amount, vocab, category_list):
    tokens = torch.tensor([tokenize(description, vocab)], dtype=torch.long)
    padded_tokens = torch.zeros(1, 10, dtype=torch.long)
    padded_tokens[:, :len(tokens[0])] = tokens[0][:10]
    amt_tensor = torch.tensor([[amount]], dtype=torch.float32)

    with torch.no_grad():
        output = model(padded_tokens, amt_tensor)
        _, predicted = torch.max(output, 1)
    return category_list[predicted.item()]

if __name__ == "__main__":
    main()

