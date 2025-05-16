import re
import unicodedata
import sys
from bs4 import BeautifulSoup
# tokenizing
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt_tab')
# sentence tokenizer
from nltk.tokenize import sent_tokenize
# remove stop words
from nltk.corpus import stopwords
nltk.download('stopwords')
# stemming words
from nltk.stem.porter import  PorterStemmer
# tagging parts of speech
from nltk import pos_tag
nltk.download('averaged_perceptron_tagger_eng')
# extract features from sentences
from sklearn.preprocessing import MultiLabelBinarizer
# perform named entity recognition
import spacy
# encoding text as a bag of words
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
# weighting word importance
from sklearn.feature_extraction.text import TfidfVectorizer
# calculate text similarity
from sklearn.metrics.pairwise import linear_kernel
# sentiment analysis classifier
from transformers import pipeline


# 6.1 clean test with strip, replace and split
text_data = [" Interrobang. By Aishwarya Henriette ",
"Parking And Going. By Karl Gautier",
" Today Is The night. By Jarek Prakash "]
# strip whitespaces
strip_whitespace = [s.strip() for s in text_data]
strip_whitespace
# remove periods
remove_periods = [s.replace(".", "") for s in strip_whitespace]
remove_periods
# you can create and apply custom functions
def capitalizer (string: str) -> str:
    return string.upper()

[capitalizer(s) for s in remove_periods]

# use regular expressions

def replace_letters_with_X(string: str) -> str:
    return re.sub(r"[a-zA-Z]", "X", string)

[replace_letters_with_X(s) for s in remove_periods]

# string methods

s = "machine learning in python cookbook"
# find first index of letter n
find_n = s.find("n")
# whether or not the string starts with m
starts_with_m = s.startswith("m")
# whether or not the string ends with 'python'
ends_with_python = s.endswith("python")

# is the string alphanumeric
is_alnum = s.isalnum()
# is it composed of only alphabetical characters (excluding space)
is_alpha = s.isalpha()
# encode as utf-8
encode_as_utf8 = s.encode("utf-8")
# decode the same utf-8
decode = encode_as_utf8.decode("utf-8")

print(
find_n,
starts_with_m,
ends_with_python,
is_alnum,
is_alpha,
encode_as_utf8,
decode,
sep = "|"
)
# 6.2: Parse html

# Create some HTML code
html = "<div class='full_name'>"\
"<span style='font-weight:bold'>Masego"\
"</span> Azra</div>"
parsed_html = BeautifulSoup(html, "lxml")
parsed_html.find("div", {"class": "full_name"}).text
# 6.3 Remove punctuation

# Create text
text_data = ['Hi!!!! I. Love. This. Song....',
'10000% Agree!!!! #LoveIT',
'Right?!?!']

# Create a dictionary of punctuation characters
punctuation = dict.fromkeys(
    (i for i in range(sys.maxunicode)
     if unicodedata.category(chr(i)).startswith("P")),
    None
)

# for each string remove any punctuation characters
[x.translate(punctuation) for x in text_data]

# 6.4 tokenizing text- break into individual words

s = "The science of today is the technology of tomorrow"
s.split(" ")
word_tokenize(s)


# you can also tokenize into sentences
ss = "The science of today is the technology of tomorrow. Tomorrow is today."
sent_tokenize(ss)

# 6.5 Removing stop words: e.g., a, is, of, on

tokenized_words = ['i',
'am',
'going',
'to',
'go',
'to',
'the',
'store',
'and',
'park']
# load stop words
stop_words = stopwords.words('english')
# remove stop words
[word for word in tokenized_words if word not in stop_words]

# 6.6 Stemming words

# Create word tokens
tokenized_words = ['i', 'am', 'humbled', 'by', 'this', 'traditional', 'meeting']
# create stemmer
porter = PorterStemmer()
# apply stemmer
[porter.stem(word) for word in tokenized_words]

# 6.7 Tagging parts of speech
# create text
text_data = "Chris loved outdoor running"
# use pretrained part of speech tagger
text_tagged = pos_tag(word_tokenize(text_data))
# show parts of speech
text_tagged
# find certain parts of speech - here nouns
[w for w, tag in text_tagged if tag in ['NN', 'NNS', 'NNP', 'NNPS']]

# convert sentences to features, taking 1 if noun and 0 otherwise
# Create text
tweets = ["I am eating a burrito for breakfast",
"Political science is an amazing field",
"San Francisco is an awesome city"]
# create list
tagged_tweets = []
# tag each word and each tweet
for t in tweets:
    tweet_tag = nltk.pos_tag(word_tokenize(t))
    tagged_tweets.append([tag for word, tag in tweet_tag])

# use one-hot encoding to convert tags into features
one_hot_multi = MultiLabelBinarizer()
one_hot_multi.fit_transform(tagged_tweets)
# show feature names using classes_
one_hot_multi.classes_
# 6.8 Performing named entity recognition
nlp = spacy.load("en_core_web_sm")
doc = nlp("Elon Musk offered to buy Twitter using $21B of his own money.")
# print each entity
print(doc.ents)
# for each entity print the text and entity label
for entity in doc.ents:
    print(entity.text, entity.label_, sep=",")

# 6.9 Encoding text as a Bag of Words
# Create text
text_data = np.array(['I love Brazil. Brazil!',
'Sweden is best',
'Germany beats both'])
# create bag of words feature matrix
count = CountVectorizer()
bag_of_words = count.fit_transform(text_data)
# show feature matrix
bag_of_words
bag_of_words.toarray()
# show feature names
count.get_feature_names_out()

# 6.10 Weighting word importance

# Create text
text_data = np.array(['I love Brazil. Brazil!',
'Sweden is best',
'Germany beats both'])
# create tf-idf feature matrix
tfidf = TfidfVectorizer()
feature_matrix = tfidf.fit_transform(text_data)
# show tf-idf feature matrix
feature_matrix
# show tf-idf feature matrix as a dense matrix
feature_matrix.toarray()
# show feature names
tfidf.vocabulary_

# 6.11 Using Text Vectors to Calculate Text Similarity in a
# Search Query

# Create searchable text data
text_data = np.array(['I love Brazil. Brazil!',
'Sweden is best',
'Germany beats both'])

# Create the tf-idf feature matrix
tfidf = TfidfVectorizer()
feature_matrix = tfidf.fit_transform(text_data)
# Create a search query and transform it into a tf-idf vector
text = "Brazil is the best"
vector = tfidf.transform([text])
# Calculate the cosine similarities between the input vector and all other vectors
cosine_similarities = linear_kernel(vector, feature_matrix).flatten()
# Get the index of the most relevent items in order
related_doc_indicies = cosine_similarities.argsort()[:-10:-1]
# Print the most similar texts to the search query along with the cosine
print([(text_data[i], cosine_similarities[i]) for i in related_doc_indicies])

