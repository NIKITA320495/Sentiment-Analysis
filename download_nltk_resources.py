import os
import nltk

# Set NLTK_DATA environment variable
os.environ['NLTK_DATA'] = 'C:\\Users\\Nikita\\AppData\\Roaming\\nltk_data'

# Ensure the NLTK_DATA directory exists
if not os.path.exists(os.environ['NLTK_DATA']):
    os.makedirs(os.environ['NLTK_DATA'])

# Download NLTK resources
try:
    nltk.download('stopwords', download_dir=os.environ['NLTK_DATA'])
    nltk.download('punkt', download_dir=os.environ['NLTK_DATA'])
except Exception as e:
    print("Error downloading NLTK resources:", e)
