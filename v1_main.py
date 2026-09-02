#Bare Bones -> static image first

import easyocr
import cv2

import numpy as np , pandas as pd

import re , nltk

from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from scipy.sparse import triu
from gensim.models import Word2Vec
from scipy import spatial
import networkx as nx


#Download Stop words
nltk.download('punkt_tab')
nltk.download('stopwords')


#Load Image
image = cv2.imread("subject.png")


#Load Reader 
reader = easyocr.Reader(['en'])

#Read Text
result = reader.readtext(image)

#Loop over Result to get text
whole_text = ""
for i in result:
    whole_text = whole_text + i[1].lower()

#Tokenize words
sentences = sent_tokenize(whole_text)

#Clean text
sentences_clean = [re.sub(r'[^\w\s]', '' , sentence.lower()) for sentence in sentences]

#Remove Stop Words
stop_words = stopwords.words("english")
sentence_tokens = [[word for word in sentence.split(' ') if word not in stop_words] for sentence in sentences_clean]

#Embedding
word_2_vec = Word2Vec(sentence_tokens , vector_size=1, min_count=1 , epochs=1000)

sentence_embeddings = [[word_2_vec.wv[word][0] for word in words] for words in sentence_tokens]

max_len = max([len(tokens) for tokens in sentence_tokens])

sentence_embeddings = [np.pad(embedding , (0 , max_len - len(embedding)) , 'constant') for embedding in sentence_tokens]

#Similarity Matrix
similarity_matrix = np.zeros([len(sentence_tokens) , len(sentence_tokens)])

for i , row in enumerate(sentence_embeddings):
    for j , col in enumerate(sentence_embeddings):
        similarity_matrix[i][j] = 1 - spatial.distance.cosine(row , col)


#Convert sim matrix into graph
nx_graph = nx.from_numpy_array(similarity_matrix)

#Use PageRank to rank importance of each scentence
scores = nx.pagerank(nx_graph)


#Mapping our ranked sentences
top_sentence = {scentence:scores [index] for index , scentence in enumerate(sentences)}

#Sort ranking
top = dict(sorted(top_sentence.items() , key = lambda x : x[1], reverse=True) [:4])

#Print top ranked scentences[in order]
for i in sentences:
    if i in top.keys():
        print(i)
