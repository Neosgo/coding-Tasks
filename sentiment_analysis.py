# sentiment_analysis.py

import spacy
import pandas as pd
from textblob import TextBlob

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

# Load the dataset
df = pd.read_csv('amazon_product_reviews.csv', low_memory=False)

# Select the 'review.text' column and drop missing values
reviews_data = df['reviews.text']
clean_data = reviews_data.dropna()

# Function to preprocess text
def preprocess_text(text):
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return ' '.join(tokens)

# Preprocess the review texts
clean_data = clean_data.apply(preprocess_text)

# Define a function for sentiment analysis using TextBlob
def analyze_sentiment(review):
    blob = TextBlob(review)
    sentiment_polarity = blob.sentiment.polarity
    if sentiment_polarity > 0:
        return 'positive'
    elif sentiment_polarity < 0:
        return 'negative'
    else:
        return 'neutral'

# Select a few sample reviews from the dataset for testing
sample_reviews = clean_data.sample(3).tolist()

print("Sample Reviews and Their Sentiments:")
for review in sample_reviews:
    sentiment = analyze_sentiment(review)
    print(f"Review: {review}\nSentiment: {sentiment}\n")

# Apply sentiment analysis to the entire dataset and save results
clean_data_sentiments = clean_data.apply(analyze_sentiment)
df['sentiment'] = clean_data_sentiments
df.to_csv('amazon_product_reviews_with_sentiments.csv', index=False)

print("Sentiment analysis completed and results saved to 'amazon_product_reviews_with_sentiments.csv'.")
