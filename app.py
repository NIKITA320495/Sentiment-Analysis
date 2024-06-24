import os

# Set NLTK_DATA environment variable
os.environ['NLTK_DATA'] = 'C:\\Users\\Nikita\\AppData\\Roaming\\nltk_data'


import nltk
nltk.download('punkt')
import streamlit as st
from joblib import load
import re
import pandas as pd
import nltk 
from nltk.corpus import stopwords
import plotly.express as px
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
df=pd.read_csv('linkedin-reviews.csv')
# Initialize components
lemmatizer = WordNetLemmatizer()
def preprocess_text(text):
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags if any
    text = re.sub(r'[^a-zA-Z]', ' ', text)  # Remove non-alphabetic characters
    text = text.lower()  # Convert to lowercase
    tokens = word_tokenize(text)  # Tokenize the text into words
    stop_words = set(stopwords.words('english'))
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

rf, tfidf = load('model_and_tfidf.joblib')
def plot_rating_distribution(df, column_name):
    value_counts = df[column_name].value_counts()
    
    # Custom color theme
    colors = px.colors.qualitative.Plotly
    
    fig = px.bar(
        x=value_counts.index, 
        y=value_counts.values, 
        labels={'x': column_name, 'y': 'Count'}, 
        title=f'{column_name} Distribution',
        color=value_counts.index,  # Use the column values for coloring
        color_discrete_sequence=colors  # Set the color theme
    )
    
    fig.update_layout(showlegend=False)
    
    # Display plot using Streamlit
    st.plotly_chart(fig)
def predict_sentiment(comment):
    # Preprocess the comment
    preprocessed_comment = preprocess_text(comment)
    
    # Transform the preprocessed comment using tfidf vectorizer
    tfidf_vector = tfidf.transform([preprocessed_comment]).toarray()
    
    # Predict the sentiment using the classifier
    predicted_numerical_sentiment = rf.predict(tfidf_vector)[0]  # Assuming clf.predict returns a single prediction
    
    # Map prediction to sentiment label
    predicted_sentiment = 'positive' if predicted_numerical_sentiment == 1 else 'negative'
    
    return predicted_sentiment
# Main function to run the Streamlit app

def plot_rating_percentage_pie(df, column_name):
    percentage = df[column_name].value_counts(normalize=True) * 100
    
    fig = px.pie(
        values=percentage.values, 
        names=percentage.index, 
        title='Percentage Distribution of Ratings',
        labels={column_name: 'Ratings'}
    )
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(showlegend=True, height=600, width=800)
    
    # Display plot using Streamlit
    st.plotly_chart(fig)
def generate_wordcloud(text):
    # Initialize CountVectorizer
    cv = CountVectorizer()
    words = cv.fit_transform(text)
    
    # Get list of words
    word_list = cv.get_feature_names_out()
    
    # Calculate word frequency
    word_freq = dict(zip(word_list, words.sum(axis=0).tolist()[0]))
    
    # Generate word cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)
    
    # Display the word cloud using Matplotlib in Streamlit
    plt.figure(figsize=(10, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)
def analyze_sentiment(df):
    # Separate comments based on sentiment (assuming 'Rating' column for this example)
    positive_comments = df[df['Rating'] >= 3]['Review'].apply(preprocess_text)
    negative_comments = df[df['Rating'] < 3]['Review'].apply(preprocess_text)
    
    # Generate word cloud for positive comments
    st.subheader('Top 10 Words in Positive Comments')
    generate_wordcloud(positive_comments)
    
    # Generate word cloud for negative comments
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
    # Input box for user to enter text
    st.subheader('Try your own comment:')
    comment = st.text_area('Enter your comment:')
    
    if st.button('Predict'):
        # Ensure comment is not empty
        if comment.strip() == '':
            st.warning('Please enter a comment.')
        else:
            # Perform prediction
            prediction = predict_sentiment(comment)
            if prediction=='positive':
                st.success(f'The sentiment of the comment is: {prediction}')
            else:
                st.error(f'The sentiment of the comment is: {prediction}')

if __name__ == '__main__':
    main()
