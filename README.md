# Simple Transaction Categorizer
A simple transaction categorizer powered by PyTorch and Flask, designed to automatically categorize and save your transactions.

## Application Structure
```
project/
├── app/                      
│   ├── __init__.py
│   ├── routes.py
│   ├── model.py
│   ├── utils.py
│   └── static/
│       ├── css/
│       │   └── styles.css
│       └── js/
│           └── main.js
│   └── templates/
│       └── index.html
├── data/
│   └── synthetic_transactions.csv
├── model/
│   ├── tokenizer.pkl
│   └── trained_model.pth
├── notebooks/
│   ├── dataset.ipynb
│   └── model.ipynb
├── tests/
│   └── test_project.py
├── README.md
├── run.py
├── requirements.txt
└── .gitignore
```

## Requirements
The application uses the following Python libraries:
- `torch`

- `pandas`

- `scikit-learn`

- `faker`

- `flask`

- `pytest`

- `fuzzywuzzy`

- `python-Levenshtein`


You can install all the required libraries by running:
```bash
pip install -r requirements.txt
```

## Instructions
1. **Run the application**: Open your terminal and run the following command:

    ```
    python run.py
    ```

2. **Access the Web Interface**: Open your web browser and go to http://localhost:5000.

3. **Input Transaction Details**: Enter the **Transaction Description**, **Amount**, and optional **Notes** in the provided fields.

4. **Categorize Transactions**: Click the **"Categorize"** button. Your transaction will appear in the transaction history table with its predicted category and confidence score.

5. **Export Transactions**: To export your transaction history, click on **"Export to CSV"**. This generates a CSV file compatible with budgeting tools and spreadsheets.

6. **Reset Transactions**: Click the **"Reset"** button to clear the transaction history table. You'll receive a confirmation prompt to prevent accidental data loss.

## Features
- **Merchant Name Categorization**: Utilizes a trained neural network model built with PyTorch to accurately predict transaction categories based on merchant names.

- **Fuzzy Matching**: Handles out-of-vocabulary merchant names with fuzzy matching.

- **Confidence Score**: Displays the model's confidence percentage for each prediction.

- **Transaction History**: View all categorized transactions instantly, complete with timestamps and optional notes.

- **CSV Export**: Export transaction data in CSV format, ideal for personal budgeting or integration with financial applications.

- **Dark Mode**: Switch to dark mode for a comfortable viewing experience, day or night.

## Technical Details
- **Model Architecture**: The model is built using PyTorch. More information about the model can be found in `model.ipynb`.

- **Dataset**: Custom-built synthetic dataset simulating U.S. credit card transactions. More information about the dataset can be foudn in `dataset.ipynb`.

- **Training Details**:

    - Epochs: 20

    - Learning Rate: 0.001

    - Loss Function: CrossEntropyLoss

    - Optimizer: Adam optimizer

    - Loss Values:

        ```
        Epochs
        1/20:  0.8841
        2/20:  0.0447
        3/20:  0.0210
        4/20:  0.0165
        5/20:  0.0152
        6/20:  0.0143
        7/20:  0.0138
        8/20:  0.0139
        9/20:  0.0136
        10/20: 0.0136
        11/20: 0.0135
        12/20: 0.0134
        13/20: 0.0134
        14/20: 0.0133
        15/20: 0.0135
        16/20: 0.0133
        17/20: 0.0134
        18/20: 0.0133
        19/20: 0.0133
        20/20: 0.0132
        ```

## Version History
### V0.1.0 (January 6, 2025):
- Initial CLI-based project for CS50P Final Project Submission.

### V1.0.0 (February 4, 2025):
- Flask app with fuzzy matching, custom dataset, improved model accuracy, and CSV export.

## Roadmap
### V2.0.0 (mid-February 2025):
- Transition the program to use a database instead of a CSV file.
- Enhanced querying and faster merchant lookups.

### V3.0.0 (March 2025):
- Incremental learning to allow model to improve with new transaction data over time.

## License
This project is licensed under the [GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0.en.html).

## Acknowledgements
Special thanks to:
- [David J. Malan's CS50P Course](https://cs50.harvard.edu/python/2022/) for the foundation of this project.
- [Google Material Icons](https://fonts.google.com/icons) for the dark mode toggle icon.