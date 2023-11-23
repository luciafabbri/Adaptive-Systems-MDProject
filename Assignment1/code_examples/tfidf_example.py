import datetime

from gensim import corpora
from gensim import models
from pprint import pprint  # pretty-printer
from gensim import similarities

import re

from nltk.corpus import stopwords
from nltk import PorterStemmer

init_t: datetime = datetime.datetime.now()  # init the time for the execution time calculation

documents = [
    "Human machine survey computer interface interface eps time for lab abc computer applications user",
    "A survey of user opinion of computer system user response time computer user interface interface",
    "The EPS user users interfaces interface human interface computer human management system user",
    "System and human interface interface engineering testing of EPS computer user",
    "Relation of users perceived response time to error measurement trees",
    "The generation of random binary unordered paths minors user user computer",
    "The intersection graph of paths in trees paths trees",
    "Graph minors IV Widths of trees and well quasi ordering graph paths",
    "Graph minors A tree paths binary trees graphs",
]

porter = PorterStemmer()

# remove common words and tokenize
stoplist = stopwords.words('english')
texts = [
    [porter.stem(word) for word in document.lower().split() if word not in stoplist]
    for document in documents
]

print("Tokens of each document:")
pprint(texts)

# create mapping keyword-id
dictionary = corpora.Dictionary(texts)

print()
print("Mapping keyword-id:")
pprint(dictionary.token2id)

# create the vector for each doc
model_bow = [dictionary.doc2bow(text) for text in texts]

# create tfidf model
tfidf = models.TfidfModel(model_bow)
tfidf_vectors = tfidf[model_bow]

id2token = dict(dictionary.items())


def convert(match):
    return dictionary.id2token[int(match.group(0)[0:-1])]


print()
print("Vectors for documents (the positions with zeros are not shown):")
for doc in tfidf_vectors:
    print(re.sub("[0-9]+,", convert, str(doc)))

matrix_tfidf = similarities.MatrixSimilarity(tfidf_vectors)  # this matrix will be necessary to calculate similarity between documents

end_creation_model_t: datetime = datetime.datetime.now()  # just after the calculation of the matrix similarity -> time function

print()
print("Matrix similarities")
print(matrix_tfidf)

# obtain tfidf vector for the following doc
doc = "trees graph human"
doc_s = [porter.stem(word) for word in doc.lower().split() if word not in stoplist]

vec_bow = dictionary.doc2bow(doc_s)
vec_tfidf = tfidf[vec_bow]

# calculate similarities between doc and each doc of texts using tfidf vectors and cosine
sims = matrix_tfidf[vec_tfidf]  # sims is a list a similarities

# sort similarities in descending order
sims = sorted(enumerate(sims), key=lambda item: -item[1])

print()
print("Given the doc: " + doc)
print("whose tfidf vector is: " + str(vec_tfidf))
print()
print("The Similarities between this doc and the documents of the corpus are:")
for doc_position, doc_score in sims:
    print(doc_score, documents[doc_position])

end_t: datetime = datetime.datetime.now()  # to mark the end of the program

# get execution time
elapsed_time_model_creation: datetime = end_creation_model_t - init_t
elapsed_time_comparison: datetime = end_t - end_creation_model_t
print()
print('Execution time model:', elapsed_time_model_creation, 'seconds')
print('Execution time comparison:', elapsed_time_comparison, 'seconds')
