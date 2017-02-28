#################################
## Imports
#################################
import numpy as np
import os
import json
import pandas as pd
import re
from collections import Counter
from nltk.tokenize import WordPunctTokenizer
from nltk.tokenize.punkt import PunktSentenceTokenizer
from datetime import datetime

#################################
## Globals
#################################
word_frequency_cutoff = 280
file_name = "reviews_Cell_Phones_and_Accessories_5.json"

#################################
## Functions
#################################
def split_into_sentences(input_file_name, output_file_name=None):
    """ split each review into separate sentences """
    tokenizer = PunktSentenceTokenizer()
    df = pd.read_json(file_name, 'r', lines=True)
    labelled_review = df[['reviewText', 'overall']]
    tokenized_text = labelled_review['reviewText'].apply(lambda x: tokenizer.tokenize(x)) 

    tokenized_df = labelled_review.drop('reviewText', 1)
    tokenized_df['tokenized_text'] = tokenized_text
    return tokenized_df

def clean_data(tokenized_df):
    """ use regex to clean words """
    def clean_word(word):
        word = word.lower()
        word = word.replace('&amp;','&').replace('&lt;','<').replace('&gt;','>').replace('&quot;','"').replace('&#39;',"'")
        word = re.sub(r'(\S)\1+', r'\1\1', word)  # normalize repeated characters to two
        word = re.sub(r'(\S\S)\1+', r'\1\1', word)

        word = word.encode('ascii', 'ignore')

        if re.search(r'((([A-Za-z]{3,9}:(?:\/\/)?)(?:[-;:&=\+\$,\w]+@)?[A-Za-z0-9.-]+|(?:www.|[-;:&=\+\$,\w]+@)[A-Za-z0-9.-]+)((?:\/[\+~%\/.\w-]*)?\??(?:[-\+=&;%@.\w]*)#?(?:[\w]*))?)',word) is not None:
            word = 'GENERIC_HTTP'

        return word.encode('ascii', 'ignore')

    tokenizer = WordPunctTokenizer()

    """ breaks up each sentence in a list of sentences into clean words """
    cleaned_words = tokenized_df['tokenized_text'].apply(lambda sentences: [tokenizer.tokenize(" ".join(map(clean_word, sentence.split()))) for sentence in sentences])

    cleaned_words_df = tokenized_df.drop('tokenized_text', 1)
    cleaned_words_df['cleaned_words'] = cleaned_words
    return cleaned_words_df

# build words2vec
def build_w2v(cleaned_words_df):
    """ merge words into one sentence each on a separate line """
    w2v = []
    cleaned_words_df['cleaned_words'].apply(lambda sentences: [w2v.append(" ".join(sentence)) for sentence in sentences])
    return w2v

# build word dictionary
def build_word_dictionary(cleaned_words_df):
    """ dictionary counts the number of occurences of each word """ 
    dictionary = Counter()
    cleaned_words_df['cleaned_words'].apply(lambda sentences: [dictionary.update(sentence) for sentence in sentences])
    
    with_counts = dictionary

    # removes words below a threshold count
    dictionary = list(sorted(w for w in dictionary if dictionary[w] >= word_frequency_cutoff)) + ['PADDING', 'UNKNOWN']
    return with_counts, dictionary

# project sentences
def project_sentences(cleaned_words_df, dictionary):
    """ keeps words in reviews that appear in dictionary """
    def project_sentence(s):
        return [w if w in dictionary else "UNKNOWN" for w in s]
    
    projected_df =  cleaned_words_df['cleaned_words'].apply(lambda sentences: map(project_sentence, sentences))
    return projected_df

# encode dictionary
def encode_dictionary(dictionary):
    """ """
    encoding = {}
    for index, word in enumerate(dictionary):
        encoding[word] = index

    return encoding

#################################
## Running code
#################################

start_time = datetime.now()
print('Start time: {}'.format(start_time))

#print("split_into_sentences...")
#tokenized_df = split_into_sentences(file_name)
#
#print("clean_data...")
#cleaned_words_df = clean_data(tokenized_df)

#print("build_w2v...")
#w2v = build_w2v(cleaned_words_df)

#print("build_word_dictionary...")
#with_counts, dictionary = build_word_dictionary(cleaned_words_df)

#print("project sentences...")
#projected_df = project_sentences(cleaned_words_df, dictionary)

#print("encode_dictionary")
#encoding = encode_dictionary(dictionary)

print('Duration: {}\n'.format(datetime.now() - start_time))
