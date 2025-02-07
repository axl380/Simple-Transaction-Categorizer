from flask import Blueprint, render_template, request, jsonify
from app.utils import load_model_and_vocab, predict_category

# Blueprint setup
main = Blueprint('main', __name__)

# Load model, vocab, category_list, and known_merchants
model, vocab, category_list, known_merchants = load_model_and_vocab()

@main.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        merchant_name = request.form.get('merchant')
        if merchant_name:
            prediction = predict_category(merchant_name, model, vocab, category_list, known_merchants)
            return jsonify(prediction)  # Ensure you're returning JSON
    return render_template('index.html')
