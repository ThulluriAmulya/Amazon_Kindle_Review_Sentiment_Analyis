# Amazon Kindle Reviews Sentiment Analysis

This project focuses on classifying Amazon Kindle reviews as **positive** or **negative** using text preprocessing, feature extraction (Bag of Words and TF-IDF), and a Random Forest Classifier.

## Features

- **Text Preprocessing**
  - **Text Cleaning**: Removes unwanted characters, punctuation, and numbers.
  - **Stopword Removal**: Excludes common words that do not contribute to sentiment.
  - **Lemmatization**: Reduces words to their base forms for better consistency.
  - **URL Removal**: Eliminates URLs from the text.
  - **HTML Parsing**: Strips HTML tags using BeautifulSoup.

- **Classification**
  - Converts text into numerical features using:
    - **Bag of Words (BOW)**
    - **TF-IDF Vectorization**
  - Uses a **Random Forest Classifier** to predict the sentiment.

- **Evaluation Metrics**
  - **Confusion Matrix**
  - **Accuracy Score**
  - **Classification Report** (Precision, Recall, F1-Score)

## Prerequisites

Ensure the following are installed on your system:

- Python 3.x
- Required libraries:
  - `pandas`
  - `numpy`
  - `nltk`
  - `bs4`
  - `scikit-learn`

Install dependencies using:
```bash
pip install -r req.txt
```

Clone the repository
```bash
git clone https://github.com/your-repo/kindle-reviews-sentiment.git
```

Run the script
```bash
python sentiment_analysis.py
```




