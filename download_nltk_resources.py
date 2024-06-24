
import os
import nltk

# Set NLTK_DATA environment variable
os.environ['NLTK_DATA'] = 'C:\\Users\\Nikita\\AppData\\Roaming\\nltk_data'

# Download NLTK resources
nltk.download('stopwords', download_dir=os.environ['NLTK_DATA'])
nltk.download('punkt', download_dir=os.environ['NLTK_DATA'])
