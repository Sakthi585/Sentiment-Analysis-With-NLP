Sentiment Analysis on IMDB Reviews
This project implements Sentiment Analysis on the IMDB movie reviews dataset.
The goal is to classify reviews as positive or negative using TF-IDF vectorization for text representation and Logistic Regression for classification.

Dataset
This project uses the **IMDB Movie Reviews Dataset**, a large dataset for binary sentiment classification (positive/negative reviews).  
Due to file size limitations, the dataset is **not included in this repository**.  
You can download it directly from the [Kaggle IMDB Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews).
After downloading, place the dataset file (e.g., `IMDB Dataset.csv`) inside the `Task2_Sentiment_Analysis/` directory, so the structure looks like this:

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
