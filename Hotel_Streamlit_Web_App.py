#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[ ]:





# In[3]:


from nltk import pos_tag, word_tokenize
from nltk.corpus import wordnet
import streamlit as st
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the data and preprocess it
reviews = pd.read_excel(r'C:\Users\pv437\Desktop\Data Scince Folder\Projects\Project 2\hotel_reviews.xlsx')
df = pd.read_csv(r'C:\Users\pv437\Desktop\Data Scince Folder\Projects\Project 2\Hotel_Noun.csv')

# Use TF-IDF to vectorize the reviews
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(reviews['Review'])

# Function to get the top N most similar reviews based on user input
def get_top_similar_reviews(user_input, top_n):
    user_vector = vectorizer.transform([user_input])
    similarity = cosine_similarity(user_vector, tfidf_matrix).flatten()

    # Get the indices of the top reviews
    top_indices = similarity.argsort()[-top_n:][::-1]

    # Get the actual reviews based on the indices
    top_reviews = reviews['Review'].iloc[top_indices].tolist()

    return top_reviews

# Function to get sentiment
def get_sentiment(user_input):
    # Initialize the VADER sentiment analyzer
    sia = SentimentIntensityAnalyzer()

    # Get sentiment polarity scores
    sentiment_scores = sia.polarity_scores(user_input)

    # Determine sentiment based on the compound score
    compound_score = sentiment_scores['compound']

    if compound_score >= 0.05:
        sentiment = 'Positive'
    elif compound_score <= -0.05:
        sentiment = 'Negative'
    else:
        sentiment = 'Neutral'

    return sentiment, sentiment_scores

# Function to filter non-English words
def is_english_word(word):
    return word.lower() in wordnet.words()

# Function to extract nouns from a sentence
def extract_nouns_from_sentence(sentence):
    words = word_tokenize(sentence)
    tagged_words = pos_tag(words)
    nouns = [word for word, pos in tagged_words if pos.startswith('N')]
    return set(nouns)

# Function to print min and max counts
def print_min_max_counts(df):
    min_count = df['Count'].min()
    max_count = df['Count'].max()

    st.write(f"Minimum Count/Frequency of a Keyword: {min_count}")
    st.write(f"Maximum Count/Frequency of a Keyword: {max_count}")

# Function to print top N nouns by count
def print_top_n_nouns_by_count(df, threshold_count, top_n):
    # Filter DataFrame based on the specified threshold count
    filtered_df = df[df['Count'] <= threshold_count]

    # Sort the filtered DataFrame by the 'Count' column in descending order
    df_sorted = filtered_df.sort_values(by='Count', ascending=False)

    # Select the top 'n' rows
    top_n_nouns = df_sorted.head(top_n)

    # Print the results
    st.write(f"Top {top_n} KeyWords with Count/Frequency Close To {threshold_count}:")
    for i, (index, row) in enumerate(top_n_nouns.iterrows(), 1):
        st.write(f"Top {i} KeyWord close to {threshold_count} is : {row['Noun']}, With Count/Frequency is : {row['Count']}")      

# Streamlit App
def main():
    st.title("Sentiment Analysis and Review Similarity App")

    # Sidebar navigation
    page = st.sidebar.radio("Navigation", ["Sentiment Analysis", "Top KeyWords", "Review Similarity"])

    # Main content based on selected page
    if page == "Sentiment Analysis":
        st.header("Sentiment Analysis")
        user_input_sentiment = st.text_area("Enter your text for sentiment analysis:", value="")
        
        if st.button("Submit") and user_input_sentiment.strip() != "":
            # Extract nouns from each sentence in the input
            sentences = user_input_sentiment.split('.')
            all_nouns = set()
            words = word_tokenize(user_input_sentiment)

            for sentence in sentences:
                nouns_in_sentence = extract_nouns_from_sentence(sentence)
                all_nouns.update(nouns_in_sentence)
            legit_nouns = [noun for noun in all_nouns if is_english_word(noun)]

            st.write(f"Unique Keywords from Review: {legit_nouns}")

            # Continue with sentiment analysis
            sentiment, scores = get_sentiment(user_input_sentiment)
            st.write(f"Sentiment: {sentiment}")
            st.write(f"Sentiment Scores: {scores}")

    elif page == "Top KeyWords":
        st.header("Top KeyWords Based On Counts/Frequency")
        print_min_max_counts(df)
        user_input_threshold = int(st.number_input("Enter the minimum count or Frequency for KeyWords:", min_value=0, value=0))
        user_input_top_n = int(st.number_input("Enter the number for KeyWords you want to display:", min_value=1, value=1))
        
        if st.button("Submit") and user_input_threshold != 0 and user_input_top_n != 0:
            print_top_n_nouns_by_count(df, user_input_threshold, user_input_top_n)

    elif page == "Review Similarity":
        st.header("Review Similarity Based On Keywords")
        user_input_similarity = st.text_area("Enter your text for finding similar reviews:", value="")
        num_reviews_similarity = int(st.number_input("Enter the number for Similar reviews you want to see:", min_value=1, value=10))
        
        if st.button("Submit") and user_input_similarity.strip() != "":
            # Extract nouns from each sentence in the input
            sentences = user_input_similarity.split('.')
            all_nouns = set()

            for sentence in sentences:
                nouns_in_sentence = extract_nouns_from_sentence(sentence)
                all_nouns.update(nouns_in_sentence)

            st.write(f"Top {num_reviews_similarity} Similar Reviews:")
            top_reviews = get_top_similar_reviews(user_input_similarity, top_n=num_reviews_similarity)
            for i, review in enumerate(top_reviews, 1):
                st.write(f"Top {i} Review:  {review} \n \n ")

if __name__ == "__main__":
    main()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




