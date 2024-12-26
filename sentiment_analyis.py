import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# Load the data
data = pd.read_csv('review.csv')

df = data[['reviewText', 'rating']]

# Handling missing values
df.isnull().sum()

# Converting rating into binary (positive: 1, negative: 0)
df['rating'] = df['rating'].apply(lambda x: 0 if x < 3 else 1)

# Lowercase all reviewText
df['reviewText'] = df['reviewText'].str.lower()


nltk.download('stopwords')

df['reviewText'] = df['reviewText'].apply(lambda x: re.sub('[^a-z A-z 0-9-]+', '', x))
df['reviewText'] = df['reviewText'].apply(lambda x: " ".join([y for y in x.split() if y not in stopwords.words('english')]))
df['reviewText'] = df['reviewText'].apply(lambda x: re.sub(r'(http|https|ftp|ssh)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?', '', str(x)))
df['reviewText'] = df['reviewText'].apply(lambda x: BeautifulSoup(x, 'lxml').get_text())
df['reviewText'] = df['reviewText'].apply(lambda x: " ".join(x.split()))

# Lemmatization
lemmatizer = WordNetLemmatizer()
def lemmatize_words(text):
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])

df['reviewText'] = df['reviewText'].apply(lambda x: lemmatize_words(x))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(df['reviewText'], df['rating'], test_size=0.20)


bow = CountVectorizer()
X_train_bow = bow.fit_transform(X_train).toarray()
X_test_bow = bow.transform(X_test).toarray()

tfidf = TfidfVectorizer()
X_train_tfidf = tfidf.fit_transform(X_train).toarray()
X_test_tfidf = tfidf.transform(X_test).toarray()

# Random Forest Classifier
rf_model_bow = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model_tfidf = RandomForestClassifier(n_estimators=100, random_state=42)

# Fit the models
rf_model_bow.fit(X_train_bow, y_train)
rf_model_tfidf.fit(X_train_tfidf, y_train)

# Predictions
y_pred_bow = rf_model_bow.predict(X_test_bow)
y_pred_tfidf = rf_model_tfidf.predict(X_test_tfidf)

# Evaluation
print("BOW Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_bow))
print("BOW Accuracy: ", accuracy_score(y_test, y_pred_bow))

print("TF-IDF Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_tfidf))
print("TF-IDF Accuracy: ", accuracy_score(y_test, y_pred_tfidf))

print("BOW Classification Report:")
print(classification_report(y_test, y_pred_bow))

print("TF-IDF Classification Report:")
print(classification_report(y_test, y_pred_tfidf))
