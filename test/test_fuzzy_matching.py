import sys
import os
import pytest
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # Add the parent directory to sys.path

from app.utils import load_model_and_vocab, predict_category

# Load the model, vocab, and category list
model, vocab, category_list, known_merchants = load_model_and_vocab()

# Fuzzy matching test data with intentional typos
test_data = [
    ("Delat Airlines", "Airlines"),
    ("Starbuks", "Fast Food"),
    ("Amazn", "Retail"),
    ("Mcdonlds", "Fast Food"),
    ("Walmrt", "Groceries"),
    ("Neflix", "Entertainment"),
    ("Kaizer", "Health"),
    ("Chvron", "Gas"),
    ("Lyfft", "Transportation"),
    ("Havard", "Education")
]

@pytest.mark.parametrize("merchant_name, expected_category", test_data)
def test_fuzzy_predict_category(merchant_name, expected_category):
    predicted_category = predict_category(merchant_name, model, vocab, category_list, known_merchants)
    assert predicted_category == expected_category
