# SMS Spam Detector

## Overview
This project implements an SMS and email spam classifier using machine learning techniques. The model is designed to classify messages as either spam or ham (non-spam) based on natural language processing (NLP) and machine learning algorithms. The repository contains end-to-end code for data preprocessing, model training, and evaluation.

## Features
- **Data Preprocessing**: Cleaning and preparing text data using techniques like tokenization, stemming, and TF-IDF vectorization.
- **Machine Learning Models**: Implementation of algorithms such as Naive Bayes, Logistic Regression, or Support Vector Machines (SVM) for classification.
- **Evaluation**: Metrics like accuracy, precision, recall, and F1-score to assess model performance.
- **Dataset**: Utilizes a dataset of labeled SMS/email messages (spam and ham).

## Requirements
To run this project, you need the following dependencies:
- Python 3.8+
- Libraries: 
  - `numpy`
  - `pandas`
  - `scikit-learn`
  - `nltk`
  - `matplotlib` (for visualization)

You can install the dependencies using:
```bash
pip install -r requirements.txt
```

## Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/pallavisuthar03-coder/Sms-spam-detector.git
   cd Sms-spam-detector
   ```

2. **Install Dependencies**:
   ```bash
   pip install numpy pandas scikit-learn nltk matplotlib
   ```

3. **Download NLTK Data**:
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   ```

## Usage
1. **Prepare the Dataset**:
   - Place your dataset (e.g., `spam.csv`) in the project directory.
   - The dataset should have columns for the message text and labels (spam/ham).

2. **Run the Model**:
   - Execute the main script (e.g., `sms-spam-detector.py`) to preprocess data, train the model, and evaluate performance:
     ```bash
     python sms-spam-detector.py
     ```

3. **Example**:
   - Input a message: "Win a free iPhone now!"
   - Output: Predicted label (e.g., "Spam")

## Project Structure
- `sms-spam-detector.py`: Main script for data preprocessing, model training, and evaluation.
- `requirements.txt`: List of required Python libraries.
- `data/`: Directory for storing the dataset (e.g., `spam.csv`).
- `models/`: Directory for saving trained models (if applicable).

## Results
- The model achieves high accuracy on the test set (e.g., ~95% with Naive Bayes, based on typical spam detection performance).
- Detailed classification reports (precision, recall, F1-score) are generated during evaluation.



