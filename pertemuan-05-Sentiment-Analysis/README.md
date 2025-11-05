# Sentiment Analysis on Twitter Data

This project performs sentiment analysis on Twitter data using machine learning models. The analysis classifies tweets into four sentiment categories: Positive, Negative, Neutral, and Irrelevant.

## Dataset

The dataset consists of two CSV files:

- `twitter_training.csv`: Training data
- `twitter_validation.csv`: Validation/test data

Each file contains columns for ID, platform, sentiment label, and tweet text.

## Project Structure

- `Sentiment-Analysis.ipynb`: Jupyter notebook containing the full analysis pipeline
- `twitter_training.csv`: Training dataset
- `twitter_validation.csv`: Validation dataset
- `README.md`: This file

## Requirements

- Python 3.7+
- Jupyter Notebook
- Required libraries:
  - numpy
  - pandas
  - seaborn
  - matplotlib
  - scikit-learn
  - xgboost
  - nltk
  - emoji
  - wordcloud

## Installation

1. Clone or download this repository.

2. Install the required packages:

   ```bash
   pip install numpy pandas seaborn matplotlib scikit-learn xgboost nltk emoji wordcloud
   ```

3. Download NLTK data:
   ```python
   import nltk
   nltk.download('stopwords')
   nltk.download('wordnet')
   nltk.download('averaged_perceptron_tagger')
   nltk.download('omw-1.4')
   nltk.download('punkt')
   ```

## Usage

1. Open the `Sentiment-Analysis.ipynb` notebook in Jupyter.

2. Run the cells in order. The notebook includes:

   - Data loading and exploration
   - Data preprocessing (cleaning, tokenization, lemmatization)
   - Feature extraction using TF-IDF
   - Model training and evaluation for:
     - Logistic Regression
     - Random Forest
     - SVM (LinearSVC)
     - XGBoost
   - Model comparison with metrics and visualizations

3. The notebook will generate:
   - Word clouds for training and validation data
   - Confusion matrices for each model
   - ROC curves
   - Comparative bar plots for accuracy, F1-score, and AUC

## Models and Evaluation

### Models Used:

- **Logistic Regression**: Baseline linear model
- **Random Forest**: Ensemble method with 100 trees
- **SVM**: Linear Support Vector Machine (using LinearSVC for efficiency)
- **XGBoost**: Gradient boosting with multi-class softmax objective

### Evaluation Metrics:

- Accuracy
- F1-score (macro average)
- AUC (macro average for multi-class)
- Confusion Matrix
- ROC Curves

## Results

The notebook compares all models and provides visualizations. XGBoost typically performs best on this dataset, followed by SVM and Random Forest.

## Preprocessing Steps

1. Load and rename columns
2. Remove unnecessary columns (ID, platform)
3. Handle missing values and duplicates
4. Text preprocessing:
   - Lowercasing
   - Emoji handling
   - URL and number removal
   - Punctuation removal
   - Abbreviation expansion
   - Stopword removal
   - Lemmatization

## Feature Extraction

- TF-IDF vectorization with max_features=17000 and ngram_range=(1,2)

## Notes

- The SVM model uses LinearSVC for faster training on large datasets.
- All models are evaluated on the validation set.
- The notebook includes detailed comments and explanations.

## License

This project is for educational purposes. Feel free to use and modify as needed.

## Contact

For questions or improvements, please refer to the notebook comments or create an issue.
