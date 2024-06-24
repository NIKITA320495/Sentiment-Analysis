import nltk
import os

# Define your NLTK data path explicitly
custom_nltk_data_path = r'C:\Users\Nikita\nltk_data'

# Ensure the directory exists, create it if necessary
if not os.path.exists(custom_nltk_data_path):
    os.makedirs(custom_nltk_data_path)

# Set NLTK data path
nltk.data.path.append(custom_nltk_data_path)

# Download specific NLTK resources
resources = ['punkt', 'stopwords']

for resource in resources:
    try:
        nltk.data.find(f'{resource}')
        print(f"{resource} is already available.")
    except LookupError:
        print(f"{resource} not found. Downloading...")
        nltk.download(resource, download_dir=custom_nltk_data_path)

print("NLTK Data Path:", nltk.data.path)
