import argparse
import datetime
import logging
import os
import re
import time
from collections import defaultdict

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.callbacks import EarlyStopping
import scipy
from dateutil.parser import parse
from scipy.sparse import csr_matrix
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.svm import SVC
from xgboost import XGBClassifier, XGBRegressor

from graph import Graph
from utils import *


def parse_args():
    parser = argparse.ArgumentParser(description='Link prediction by similarity of graph embedding')
    parser.add_argument('--doc_dir', type=str, required=True, help='Document directory which place XML files')
    parser.add_argument('--train_file', type=str, required=True, help='t1-train.txt file')
    parser.add_argument('--test_file', type=str, required=True, help='t1-test.txt file')
    parser.add_argument('--phraser', type=str, required=True, help='N-gram phraser model sfile')
    parser.add_argument('--w2v_model', type=str, required=True, help='Word2vec model file')
    parser.add_argument('--output_file', type=str, required=True, help='Output file, ex. pred.csv')
    parser.add_argument('--test_size', type=float, default=0.1)
    return parser.parse_args()


def parse_xml(file):
    with open(file, 'r') as f:
        title = []
        abstract = []
        date_start = False
        title_start = False
        abstract_start = False

        for line in f.readlines():
            line = line.strip()
            if line == '<date>':
                date_start = True
            elif line == '</date>':
                date_start = False
            elif line == '<title>':
                title_start = True
            elif line == '</title>':
                title_start = False
            elif line == '<abstract>':
                abstract_start = True
            elif line == '</abstract>':
                abstract_start = False
            elif date_start:
                date = line
            elif title_start:
                title.append(line)
            elif abstract_start:
                abstract.append(line)

    date = parse_date(date)
    title = parse_title(' '.join(title))
    abstract = parse_title(' '.join(abstract))
    return date, title, abstract


def parse_date(date):
    date = ' '.join(date.split()[:4])
    date = parse(date).replace(tzinfo=None)
    return date


def date2vec(date):
    return [date.year, date.month, date.day, date.weekday()]


def negative_sampling(test_graph, adj_matrix_org, topo_sort, docs, n_sample, encoder):
    logging.info('Negative sampling')
    hop_matrix = csr_matrix(adj_matrix_org)
    hop_matrix = hop_matrix * hop_matrix
    row, col = hop_matrix.nonzero()
    X = []

    adj_matrix = adj_matrix_org.copy()
    for u, v in test_graph:
        adj_matrix[u][v] = 1

    topo_dict = defaultdict(int)
    for index, node_id in enumerate(topo_sort):
        topo_dict[node_id] = index

    while len(X) < int(n_sample):
        index = np.random.randint(0, len(row))
        u, v = row[index], col[index]
        if topo_dict[u] < topo_dict[v] and adj_matrix[u][v] == 0:
            timedelta = get_timedelta(docs[u], docs[v])
            if timedelta > 0:
                adj_matrix[u][v] = 1
                x = get_features(docs[u], docs[v], encoder, adj_matrix)
                X.append(x)

    X = np.array(X)
    y = np.zeros((X.shape[0]))
    return X, y


def get_features(doc_1, doc_2, encoder, adj_matrix):
    X = np.array([])
    # date_vec = [date2vec(doc_1[0]), date2vec(doc_2[0])]
    # date_vec_one_hot = encoder.transform(date_vec).flatten()
    # X = np.concatenate((X, date_vec_one_hot)) # Date vector
    # X = np.concatenate((X, doc_1[1], doc_2[1])) # Title vector
    # X = np.concatenate((X, doc_1[2], doc_2[2])) # Abstract vector
    # Other features
    # Cosine similarity on title and abstract
    title_sim = cosine_similarity([doc_1[1]], [doc_2[1]])[0]
    abstract_sim = cosine_similarity([doc_1[2]], [doc_2[2]])[0]
    X = np.concatenate((X, title_sim, abstract_sim))
    # Date difference
    t1, t2 = get_timestamp(doc_1), get_timestamp(doc_2)
    date_diff = get_timedelta(doc_1, doc_2)
    X = np.append(X, [t1, t2, date_diff])
    # Graph informations
    # In-degree of cited paper
    in_degree = len(adj_matrix[:][doc_2[3]].nonzero())
    X = np.append(X, in_degree)
    return X


def get_timedelta(doc_1, doc_2):
    try:
        timedelta = doc_1[0] - doc_2[0]
    except BaseException as err:
        logging.error(err)
        timedelta = doc_1[0].replace(tzinfo=None) - doc_2[0].replace(tzinfo=None)
    return timedelta.days


def get_timestamp(doc):
    timestamp = doc[0].timestamp()
    return timestamp


def import_docs(doc_dir):
    """
    docs = {
        'doc_id': [
            'date <Datetime>',
            'title <List>: list of words',
            'abstract <List>: list of words'
        ]
    }
    """
    logging.info('Import documents')
    doc_dir = args.doc_dir
    docs = defaultdict(list)
    cnt = 0
    for file in os.listdir(doc_dir):
        cnt = cnt + 1
        if cnt % 1000 == 0: logging.info('Reading {} files'.format(cnt))
        date, title, abstract = parse_xml(os.path.join(doc_dir, file))
        key = int(re.search(r'\d+', file).group())
        docs[key] = [date, title, abstract, key]
    return docs


def main(args):
    docs = import_docs(args.doc_dir)

    # Preprocessing documents
    # One hot encode date
    dates = []
    for key in docs:
        dates.append(date2vec(docs[key][0]))
    enc = OneHotEncoder(sparse=False)
    enc.fit(dates)
    del dates

    # Transform words to phrases
    phraser = gensim.utils.SaveLoad.load(args.phraser)
    title_corpus = []
    abstract_corpus = []
    for key in docs:
        docs[key][1] = phraser[docs[key][1]]
        docs[key][2] = phraser[docs[key][2]]
        title_corpus.append(docs[key][1])
        abstract_corpus.append(docs[key][2])

    # Calculate TF-IDF
    title_dict = gensim.corpora.Dictionary(title_corpus)
    title_corpus = [title_dict.doc2bow(title) for title in title_corpus]
    abstract_dict = gensim.corpora.Dictionary(abstract_corpus)
    abstract_corpus = [abstract_dict.doc2bow(abstract) for abstract in abstract_corpus]
    title_tfidf_model = gensim.models.TfidfModel(title_corpus)
    abstract_tfidf_model = gensim.models.TfidfModel(abstract_corpus)
    del title_corpus, abstract_corpus

    # Word2vec
    w2v_model = gensim.models.Word2Vec.load(args.w2v_model)
    word_vectors = w2v_model.wv
    del w2v_model

    for key in docs:
        title =  title_dict.doc2bow(docs[key][1])
        abstract = abstract_dict.doc2bow(docs[key][2])
        title_tfidf = title_tfidf_model[title]
        abstract_tfidf = abstract_tfidf_model[abstract]
        title_vector = [word_vectors[title_dict[index]] * tfidf for index, tfidf in title_tfidf if title_dict[index] in word_vectors.vocab]
        abstract_vector = [word_vectors[abstract_dict[index]] * tfidf for index, tfidf in abstract_tfidf if abstract_dict[index] in word_vectors.vocab]
        if len(title_vector) == 0:
            logging.error('document id {} missing title vector'.format(key))
            title_vector = np.zeros((1, word_vectors.vector_size))
        if len(abstract_vector) == 0:
            logging.error('document id {} missing abstract vector'.format(key))
            abstract_vector = np.zeros((1, word_vectors.vector_size))
        title_vector = np.average(title_vector, axis=0)
        abstract_vector = np.average(abstract_vector, axis=0)
        docs[key][1], docs[key][2] = title_vector, abstract_vector


    logging.info('Sampling training data')
    # Positive sampling
    train_graph = np.loadtxt(args.train_file).astype(np.int32)
    test_graph = np.loadtxt(args.test_file).astype(np.int32)
    max_index = max(np.max(train_graph), np.max(test_graph)) + 1
    adj_matrix = np.zeros((max_index, max_index), dtype=np.int8)
    X_p = []
    for u, v in train_graph:
        adj_matrix[u][v] = 1
        timedelta = get_timedelta(docs[u], docs[v])
        if timedelta > 0:
            x = get_features(docs[u], docs[v], enc, adj_matrix)
            X_p.append(x)
    X_p = np.array(X_p)
    y_p = np.ones(X_p.shape[0])

    # Get topological sort on training data
    g = Graph(train_graph)
    topo_sort = g.topological_sort()

    # Negative sampling
    X_n, y_n = negative_sampling(test_graph, adj_matrix, topo_sort, docs, X_p.shape[0], enc)

    # Concatenate positive and negative samples
    X, y = np.concatenate((X_p, X_n)), np.concatenate((y_p, y_n))


    logging.info('Get testing data')
    X_test = []
    false_indices = []
    for index, [u, v] in enumerate(test_graph):
        x = get_features(docs[u], docs[v], enc, adj_matrix)
        timedelta = get_timedelta(docs[u], docs[v])
        if timedelta < 0:
            false_indices.append(index)
        X_test.append(x)
    X_test = np.array(X_test)
 

    # Feature scaling
    scaler = MinMaxScaler()
    scaler.fit(np.concatenate((X, X_test)))
    X = scaler.transform(X)
    X_test = scaler.transform(X_test)


    # Split validation data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=args.test_size, shuffle=True)

    # # Train classifier
    logging.info('Train classifier')
    # clf = XGBClassifier(max_depth=80, n_jobs=8, verbose=True)
    # clf.fit(X_train, y_train,
    #         eval_set=[(X_train, y_train), (X_val, y_val)],
    #         eval_metric='error',
    #         early_stopping_rounds=10,
    #         verbose=True)
    # logging.info('Validation score: {}'.format(clf.score(X_val, y_val)))

    model = Sequential()

    model.add(Dense(256, input_dim=X.shape[1], activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])
    es = EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto')
    model.fit(X_train, y_train, batch_size=100, epochs=100, callbacks=[es], validation_data=(X_val, y_val))
    score = model.evaluate(X_val, y_val)
    print('Validation score:', score)


    y_test = model.predict(X_test)
    y_test[y_test >= 0.5] = 1
    y_test[y_test < 0.5] = 0

    for index in false_indices:
        y_test[index] = 0

    pred = y_test.astype(np.int8)
    print('# of 0:', pred[pred == 0].shape[0])
    print('# of 1:', pred[pred == 1].shape[0])

    with open(args.output_file, 'w') as f:
        f.write('query_id,prediction\n')
        for index, [value] in enumerate(pred):
            f.write('{},{}\n'.format(index+1, value))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    main(args)
