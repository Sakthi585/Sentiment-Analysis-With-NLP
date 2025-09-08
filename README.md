Sentiment Analysis on IMDB Reviews
This project implements Sentiment Analysis on the IMDB movie reviews dataset.
The goal is to classify reviews as positive or negative using TF-IDF vectorization for text representation and Logistic Regression for classification.

Dataset
Included in this repository under the folder dataset.
File: imdb_reviews.csv
Classes:
Positive (1)
Negative (0)
Features: Review text

Steps Performed
Load dataset from dataset/imdb_reviews.csv.
Preprocess text data:
Convert to lowercase
Remove stopwords & punctuation
Tokenization
Transform text into numerical features using TF-IDF Vectorizer.
Train a Logistic Regression classifier.
Evaluate model using:
Accuracy Score
Confusion Matrix
Classification Report

Results
The model achieved strong performance in classifying reviews.
TF-IDF effectively captured important words from reviews.
Metrics like precision, recall, and F1-score show balanced results.

Technologies Used
Python
Scikit-learn
Pandas
Matplotlib / Seaborn
NLTK (for preprocessing)
