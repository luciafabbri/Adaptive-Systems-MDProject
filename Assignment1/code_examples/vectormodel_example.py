from gensim import corpora
from pprint import pprint  # pretty-printer
import re

from nltk.corpus import stopwords

# a corpus of 9 short documents
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

# remove common words and tokenize
stoplist = stopwords.words('english')
texts = [ # we filter all the words considered as stopwords
    [word for word in document.lower().split() if word not in stoplist]
    for document in documents
]

print("Tokens of each document:")
pprint(texts)

# create mapping keyword-id
dictionary = corpora.Dictionary(texts)

print()
print("Mapping keyword-id:")
pprint(dictionary.token2id)

# create a list of 9 vectors
model_bow = [dictionary.doc2bow(text) for text in texts]  # for each document in the corpus, we create a vector

id2token = dict(dictionary.items())  # only to create a visualisation of the dictionary


def convert(match):
    return dictionary.id2token[int(match.group(0)[0:-1])]


print()
print("Vectors for documents (the positions with zeros are not shown):")
for doc in model_bow:
    print(re.sub("[0-9]+,", convert, str(doc)))
