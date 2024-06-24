import nltk

# Define your NLTK data path explicitly
custom_nltk_data_path = [r'C:\Users\Nikita\nltk_data']  # Adjust this path to your chosen directory

# Set NLTK data path
nltk.data.path.extend(custom_nltk_data_path)

# Check if NLTK can find its resources now
print(nltk.data.path)
