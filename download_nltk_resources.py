import nltk

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("NLTK resource not found. Attempting to download...")
    nltk.download('punkt')
