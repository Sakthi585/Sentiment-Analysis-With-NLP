#Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

#Load Dataset
a = pd.read_csv("D:\Document\Machine learning Intern\Task 2\IMDB Dataset.csv")
print("First 5 rows:")
print(a.head())
print("\nClass distribution:")
print(a['sentiment'].value_counts())

#Preprocessing
a['sentiment'] = a['sentiment'].map({'positive': 1, 'negative': 0})
X = a['review']
y = a['sentiment']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# TF-IDF Vectorization
tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

#Build Model 
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

#Predictions
y_pred = model.predict(X_test_tfidf)

#Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=['Negative','Positive']))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Negative','Positive'], yticklabels=['Negative','Positive'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

#Test on New Reviews
sample_reviews = [
    "The product quality is excellent, I am very happy!",
    "Terrible experience, I will never buy this again.",
    "Average service, nothing special."
]

sample_tfidf = tfidf.transform(sample_reviews)
preds = model.predict(sample_tfidf)

for review, pred in zip(sample_reviews, preds):
    print(f"{review} → {'Positive' if pred==1 else 'Negative'}")
