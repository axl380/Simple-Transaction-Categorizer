import sys
import os
import pytest
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # Add the parent directory to sys.path

from app.utils import load_model_and_vocab, predict_category

# Load the model, vocab, category list, and known merchants
model, vocab, category_list, known_merchants = load_model_and_vocab()

# Sample test data: (merchant name, expected category)
test_data = [
    ("Delta Airlines", "Airlines"),
    ("Walmart", "Groceries"),
    ("McDonald's", "Fast Food"),
    ("Harvard", "Education"),
    ("Chevron", "Gas"),
    ("Netflix", "Entertainment"),
    ("Starbucks", "Fast Food"),
    ("Kaiser", "Health"),
    ("Amazon", "Retail"),
    ("Lyft", "Transportation")
]

@pytest.mark.parametrize("merchant_name, expected_category", test_data)
def test_predict_category(merchant_name, expected_category):
    # Call predict_category with the updated signature
    predicted_category = predict_category(merchant_name, model, vocab, category_list, known_merchants)
    
    # Assert if the predicted category matches the expected category
    assert predicted_category == expected_category