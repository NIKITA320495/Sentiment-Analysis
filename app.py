import nltk
import os
import streamlit as st
from joblib import load
import re
import pandas as pd
import plotly.express as px
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

import nltk
import os
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
# Define your NLTK data path explicitly
custom_nltk_data_path = r'C:\Users\Nikita\nltk_data'
# Ensure the directory exists, create it if necessary
if not os.path.exists(custom_nltk_data_path):
    os.makedirs(custom_nltk_data_path)

# Set NLTK data path
nltk.data.path.append(custom_nltk_data_path)

# Download specific NLTK resources if not already available
resources = ['punkt', 'stopwords', 'wordnet']

for resource in resources:
    try:
        nltk.data.find(f'tokenizers/{resource}')
        print(f"{resource} is already available.")
    except LookupError:
        print(f"{resource} not found. Downloading...")
        nltk.download(resource, download_dir=custom_nltk_data_path)


# Initialize components
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags if any
    text = re.sub(r'[^a-zA-Z]', ' ', text)  # Remove non-alphabetic characters
    text = text.lower()  # Convert to lowercase
    tokens = word_tokenize(text)  # Tokenize the text into words
    stop_words = set(nltk.corpus.stopwords.words('english'))
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Load your data and model
df = pd.read_csv('linkedin-reviews.csv')
rf, tfidf = load('model_and_tfidf.joblib')

def plot_rating_distribution(df, column_name):
    value_counts = df[column_name].value_counts()
    colors = px.colors.qualitative.Plotly
    fig = px.bar(x=value_counts.index, y=value_counts.values, labels={'x': column_name, 'y': 'Count'}, title=f'{column_name} Distribution', color=value_counts.index, color_discrete_sequence=colors)
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig)

def predict_sentiment(comment):
    preprocessed_comment = preprocess_text(comment)
    tfidf_vector = tfidf.transform([preprocessed_comment]).toarray()
    predicted_numerical_sentiment = rf.predict(tfidf_vector)[0]
    predicted_sentiment = 'positive' if predicted_numerical_sentiment == 1 else 'negative'
    return predicted_sentiment

def plot_rating_percentage_pie(df, column_name):
    percentage = df[column_name].value_counts(normalize=True) * 100
    fig = px.pie(values=percentage.values, names=percentage.index, title='Percentage Distribution of Ratings', labels={column_name: 'Ratings'})
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(showlegend=True, height=600, width=800)
    st.plotly_chart(fig)

def generate_wordcloud(text):
    cv = CountVectorizer()
    words = cv.fit_transform(text)
    word_list = cv.get_feature_names_out()
    word_freq = dict(zip(word_list, words.sum(axis=0).tolist()[0]))
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)
    plt.figure(figsize=(10, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)

def analyze_sentiment(df):
    positive_comments = df[df['Rating'] >= 3]['Review'].apply(preprocess_text)
    negative_comments = df[df['Rating'] < 3]['Review'].apply(preprocess_text)
    st.subheader('Top 10 Words in Positive Comments')
    generate_wordcloud(positive_comments)
    st.subheader('Top 10 Words in Negative Comments')
    generate_wordcloud(negative_comments)

def main():
    st.title('Sentiment Analysis of Linkedin Reviews')
    st.subheader('Preview of the Dataset')
    st.dataframe(df.head(10))
    st.subheader('Some analysis on the dataset')
    plot_rating_distribution(df, 'Rating')
    plot_rating_percentage_pie(df, 'Rating')
    analyze_sentiment(df)
    st.subheader('Try your own comment:')
    comment = st.text_area('Enter your comment:')
    if st.button('Predict'):
        if comment.strip() == '':
            st.warning('Please enter a comment.')
        else:
            prediction = predict_sentiment(comment)
            if prediction == 'positive':
                st.success(f'The sentiment of the comment is: {prediction}')
            else:
                st.error(f'The sentiment of the comment is: {prediction}')

if __name__ == '__main__':
    main()
