# NLP Stemming/Lemmatization (Part 2)
from nltk.stem import PorterStemmer, WordNetLemmatizer
import nltk

# Load tokens saved from Part 1
with open('filtered_tokens.txt', 'r') as f:
    filtered_tokens = f.read().splitlines()

# Ensure NLTK data is downloaded
nltk.download('wordnet', quiet=True)

# 6. Stemming
stemmer = PorterStemmer()
stemmed_words = [stemmer.stem(word) for word in filtered_tokens]
print("\nStemmed Words:\n", stemmed_words)

# 7. Lemmatization
lemmatizer = WordNetLemmatizer()
lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_tokens]
print("\nLemmatized Words:\n", lemmatized_words)