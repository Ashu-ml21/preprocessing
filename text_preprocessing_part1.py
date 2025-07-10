# NLP Preprocessing (Part 1)
import nltk
import re

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

input = "Barack Obama went as a prime minister of USA in the year of 2015 . PM MODI is the prime minister of INDIA."
print("Original Input:\n", input)

# 1. Lowercase Conversion
lowercase = input.lower()
print("\nLowercase Output:\n", lowercase)

# 2. Regular Expressions
lowercase_re = re.sub('2015', '2025', lowercase)  # Replace year
print("\nRegex (Year Replacement):\n", lowercase_re)
lowercase_re = re.sub('[a-m]', '*', lowercase)     # Replace letters a-m
print("\nRegex (Letter Replacement):\n", lowercase_re)
lowercase_re = re.sub('\d', '-', lowercase)       # Replace digits
print("\nRegex (Digit Replacement):\n", lowercase_re)

# 3. Tokenization
word_tokens = nltk.word_tokenize(input)
print("\nWord Tokens:\n", word_tokens)
print("Number of Word Tokens:", len(word_tokens))

sent_tokens = nltk.sent_tokenize(input)
print("\nSentence Tokens:\n", sent_tokens)
print("Number of Sentences:", len(sent_tokens))

# 4. Stopwords Removal
stop_words = set(nltk.corpus.stopwords.words('english'))
filtered_tokens = [word for word in word_tokens if word.lower() not in stop_words]
print("\nAfter Stopwords Removal:\n", ' '.join(filtered_tokens))

# 5. Part-of-Speech (POS) Tagging
pos_tags = nltk.pos_tag(word_tokens)
print("\nPOS Tags:\n", pos_tags)

# Save filtered tokens for Part 2
with open('filtered_tokens.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(filtered_tokens))