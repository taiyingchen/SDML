import argparse
import logging
import os
import re
from collections import defaultdict

import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models.word2vec import Word2Vec
from gensim.parsing.preprocessing import (preprocess_documents,
                                          preprocess_string, remove_stopwords)
# from gensim.test.test_doc2vec import ConcatenatedDoc2Vec


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('doc_dir')
    parser.add_argument('model_file')
    return parser.parse_args()


def parse_xml(file):
    with open(file, 'r') as f:
        title = []
        abstract = []
        title_start = False
        abstract_start = False

        for line in f.readlines():
            line = line.strip()
            if line == '<title>':
                title_start = True
            elif line == '</title>':
                title_start = False
            elif line == '<abstract>':
                abstract_start = True
            elif line == '</abstract>':
                abstract_start = False
            elif title_start:
                title.append(line)
            elif abstract_start:
                abstract.append(line)
    
    return ' '.join(title), ' '.join(abstract)  


def get_similarity(test_file):
    title_model = Doc2Vec.load('./title_model')
    abstract_model = Doc2Vec.load('./abstract_model')
    # model = ConcatenatedDoc2Vec([title_model, abstract_model])

    title_docvecs = title_model.docvecs
    abstract_docvecs = abstract_model.docvecs
    test_graph = np.loadtxt(test_file).astype(np.int32)
    sim = []
    for u, v in test_graph:
        # sim.append(np.mean([title_docvecs.similarity(u, v), abstract_docvecs.similarity(u, v)]))
        sim.append(np.mean([abstract_docvecs.similarity(u, v)]))
    sim = np.array(sim)
    return sim
        

def predict_by_similarity(sim):
    median = np.median(sim)
    pred = (sim >= median).astype(np.int8)
    print(pred[pred == 0].shape)
    print(pred[pred == 1].shape)
    return pred


def train_doc2vec(docs):
    titles = []
    abstracts = []
    for key, [title, abstract] in docs.items():
        titles.append(TaggedDocument(title, [key]))
        abstracts.append(TaggedDocument(abstract, [key]))

    title_model = Doc2Vec(dm=0, vector_size=128, min_count=5, workers=4)
    abstract_model = Doc2Vec(dm=0, vector_size=128, min_count=5, workers=4)
    
    title_model.build_vocab(titles)
    abstract_model.build_vocab(abstracts)

    title_model.train(titles, total_examples=len(titles), epochs=100)
    abstract_model.train(abstracts, total_examples=len(titles), epochs=100)
    
    title_model.save('title_model')
    abstract_model.save('abstract_model')


def train_word2vec(docs, model_file):
    sentences = []
    for key, [title, abstract] in docs.items():
        sentences.append(title)
        sentences.append(abstract)

    model = Word2Vec(sentences, size=256, min_count=10, workers=8)
    model.wv.save(model_file)
    print(model.wv.similar_by_word('gamma'))


def main(args):
    doc_dir = args.doc_dir
    docs = defaultdict(list)
    for file in os.listdir(doc_dir):
        title, abstract = parse_xml(os.path.join(doc_dir, file))
        key = int(re.search(r'\d+', file).group())
        title = preprocess_string(title)
        abstract = preprocess_string(abstract)
        docs[key] = [title, abstract]

    train_word2vec(docs, args.model_file)

    
if __name__ == '__main__':
    args = parse_args()
    main(args)
