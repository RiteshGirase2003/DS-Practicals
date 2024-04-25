import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize

text="Every one is same in sky,so make your wigns strong and take a flight."
words = nltk.word_tokenize(text)
print(words)

# POS tagging 
# it is for part of speech

nltk.download('averaged_perceptron_tagger')
# words is coming from above where it is tokenize
# to pos_tag we have pass tokenized words

pos_tags = nltk.pos_tag(words)
print(pos_tags)


nltk.download('averaged_perceptron_tagger')

# pos_tag is used to apply pos to each word
#  it helps to analyze grammatical role of words in text
# eg . She sings beautifully
#  she = pronoun
#  sings = verd
# beautifully = adverb
pos_tags = nltk.pos_tag(words)
print(pos_tags)


from nltk.stem import PorterStemmer, WordNetLemmatizer
# it downloads wordnet database
#  it has semantic relationship such as synonyms, antonyms, hypernyms

nltk.download('wordnet')

# stemming process is done here 
#  Stemming is a process in which words in converted to its base form
#  playing, played, plays => play

#  create an object
stemmer = PorterStemmer()     # Stemming

# stemming of words

stemmed_words = [stemmer.stem(word) for word in words]

# Lemmatization helps in normalizing words so that different variations of the same word are treated equivalently

lemmatizer = WordNetLemmatizer()  # Lemmatization
lemmatized_words = [lemmatizer.lemmatize(word) for word in words]


# Lemmatization transforms words to their base form (lemma), considering context and part of speech. It gives valid dictionary words.
# Stemming chops off prefixes and suffixes to get to the root form (stem), without considering context or part of speech. It may not always result in valid words.
print("Stemming words:", stemmed_words)  #output
print("Lemmatizing words:", lemmatized_words)



# -------------------------------
# Create representation of documents by calculating Term Frequency and Inverse DocumentFrequency.
from sklearn.feature_extraction.text import TfidfVectorizer
documents = [
    "This is the first document.",
    "This document is the second document.",
    "And this is the third one.",
    "Is this the first document?",
]

# converts text data into numerical vectors based on the importance of words in documents (TF-IDF),
#  which is useful for machine learning tasks like text classification and clustering.
tfidf_vectorizer = TfidfVectorizer()


#  Fits the TF-IDF vectorizer to the input documents and transforms them into a TF-IDF matrix.
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
print(" tfidf_matrix ", tfidf_matrix)

# Retrieves the feature (word) names from the TF-IDF vectorizer.
feature_names = tfidf_vectorizer.get_feature_names_out()
print(" feature_names ", feature_names)

# Converts a sparse matrix (like TF-IDF matrix) to a dense matrix.
dense_tfidf_matrix = tfidf_matrix.todense()
print(" dense_tfidf_matrix ", dense_tfidf_matrix)

for i in range(len(documents)):
    print(f"Document {i + 1}:")
    # enumerate pairs each word in the feature_names list with its corresponding index, 
    # making it useful for looping through a list while keeping track of the position of each element
    for j, word in enumerate(feature_names):
        tfidf_score = dense_tfidf_matrix[i, j]
        if tfidf_score > 0:
            print(f"{word}: {tfidf_score:.2f}")
    print()



# this code optional other way of above code in  the form of matrix
from sklearn.feature_extraction.text import TfidfVectorizer
documents = [
    "This is the first document.",
    "This document is the second document.",
    "And this is the third one.",
    "Is this the first document?",
]
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
print(tfidf_matrix.toarray())
