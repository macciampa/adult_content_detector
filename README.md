# Adult Content Detection Model Comparison

This project implements and compares four different machine learning models for adult content detection using text descriptions. The models are trained on a dataset of 850 samples and evaluated using comprehensive metrics.

## Features

- **4 Machine Learning Models**:
  - Random Forest
  - XGBoost (Gradient Boosting)
  - LightGBM (Gradient Boosting)
  - Logistic Regression
  - Neural Network (Bidirectional LSTM)

- **Comprehensive Evaluation**:
  - Accuracy, Precision, Recall, F1-Score
  - AUC-ROC and AUC-PR curves
  - Confusion matrices
  - Feature importance analysis

- **Advanced Feature Engineering**:
  - TF-IDF vectorization for traditional ML models
  - Text tokenization and embedding for neural networks
  - N-gram features (unigrams and bigrams)

## Installation

1. **Clone or download the project files**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure you have the data file**:
   - `adult_content.xlsx` should be in the project directory

## Usage

### Quick Start

Run the complete pipeline:
```bash
python adult_content_detector.py
```

This will:
1. Load and preprocess the data
2. Split into train/validation/test sets
3. Train all 4 models
4. Evaluate performance on test set
5. Generate comprehensive reports and visualizations

### Expected Output

The script will produce:
- **Console output**: Detailed performance metrics for each model
- **Visualization**: `model_comparison_results.png` with plots
- **Report**: `model_comparison_report.md` with detailed analysis

## Project Structure

```
ACDetector/
├── adult_content_detector.py      # Main pipeline script
├── adult_content.xlsx             # Input data file
├── requirements.txt               # Python dependencies
├── README.md                      # This file
├── model_comparison_report.md     # Detailed analysis report
└── model_comparison_results.png   # Generated visualizations
```

### Feature Engineering

Modify the TF-IDF parameters in `create_tfidf_features()`:

```python
self.tfidf_vectorizer = TfidfVectorizer(
    max_features=10000,  # Increase from 5000
    ngram_range=(1, 3),  # Add trigrams
    stop_words='english',
    min_df=1,           # Decrease from 2
    max_df=0.9          # Decrease from 0.95
)
```
