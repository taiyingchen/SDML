import argparse
import logging
import os
import re
from dateutil.parser import parse
import datetime
from collections import defaultdict
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC
from scipy.sparse import csr_matrix
from xgboost import XGBClassifier, XGBRegressor
import time
import math
import scipy
from graph import Graph

import numpy as np


INTERVAL_DAY = 32


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('doc_dir')
    parser.add_argument('trains_file')
    parser.add_argument('test_file')
    parser.add_argument('output_file')
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
                break
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
    
    while True:
        try:
            date = parse(date)
            break
        except:
            date = ' '.join(date.split()[:-1])
    date = date.replace(tzinfo=None)
    timestamp = time.mktime(date.timetuple())
    return int(timestamp), ' '.join(title), ' '.join(abstract)  


def negative_sampling(adj_matrix, topo_sort, docs, n_sample, encoder=None):
    logging.info('Negative sampling')
    hop_matrix = csr_matrix(adj_matrix)
    hop_matrix = hop_matrix * hop_matrix
    row, col = hop_matrix.nonzero()
    X = []

    topo_dict = defaultdict(int)
    for index, node_id in enumerate(topo_sort):
        topo_dict[node_id] = index

    while len(X) < int(n_sample):
        index = np.random.randint(0, len(row))
        u, v = row[index], col[index]
        # if v not in adj_matrix[u]:
        if topo_dict[u] < topo_dict[v] and adj_matrix[u][v] == 0:
            timedelta = get_timedelta(docs[u], docs[v])
            if timedelta > 0:
                # adj_matrix[u].add(v)
                adj_matrix[u][v] = 1
                x = get_features(docs[u], docs[v])
                X.append(x)

    # X = encoder.transform(X)
    X = np.array(X)
    y = np.zeros((X.shape[0]))
    return X, y


def get_features(doc_1, doc_2):
    X = []
    X.append(doc_1[0])
    X.append(doc_2[0])
    X.append(get_timedelta(doc_1, doc_2))
    X = np.array(X)
    X = X // (60*60*24*INTERVAL_DAY) # seconds to days
    return X


def get_timedelta(doc_1, doc_2):
    timedelta = doc_1[0] - doc_2[0]
    return timedelta


def main(args):
    doc_dir = args.doc_dir
    docs = defaultdict(list)
    min_date = math.inf
    max_date = 0
    for file in os.listdir(doc_dir):
        date, title, abstract = parse_xml(os.path.join(doc_dir, file))
        key = int(re.search(r'\d+', file).group())
        docs[key] = [date, title, abstract]
        if date < min_date:
            min_date = date
        if date > max_date:
            max_date = date

    for key in docs:
        docs[key][0] -= min_date

    # Preprocessing
    # logging.info('Min-Max scaling')
    # scaler = MinMaxScaler()
    # scaler.fit_transform(X)

    logging.info('Get training data')
    # Positive sampling
    train_graph = np.loadtxt(args.trains_file).astype(np.int32)
    max_index = np.max(train_graph) + 1
    # adj_matrix = defaultdict(set)
    adj_matrix = np.zeros((max_index, max_index), dtype=np.int8)
    X_p = []
    for u, v in train_graph:
        # adj_matrix[u].add(v)
        adj_matrix[u][v] = 1
        timedelta = get_timedelta(docs[u], docs[v])
        if timedelta > 0:
            x = get_features(docs[u], docs[v])
            X_p.append(x)
    X_p = np.array(X_p)
    y_p = np.ones(X_p.shape[0])
    # adj_matrix = csr_matrix(adj_matrix)

    # Get topological sort on training data
    g = Graph(train_graph)
    topo_sort = g.topological_sort()

    # Negative sampling
    X_n, y_n = negative_sampling(adj_matrix, topo_sort, docs, X_p.shape[0])

    # Concatenate positive and negative samples
    X, y = np.concatenate((X_p, X_n)), np.concatenate((y_p, y_n))


    logging.info('Get testing data')
    test_graph = np.loadtxt(args.test_file).astype(np.int32)
    X_test = []
    false_indices = []
    for index, [u, v] in enumerate(test_graph):
        x = get_features(docs[u], docs[v])
        timedelta = get_timedelta(docs[u], docs[v])
        if timedelta < 0:
            false_indices.append(index)
            x[2] = 0
        X_test.append(x)
    X_test = np.array(X_test)
    # X_test = scaler.transform(X_test)
    # X_test = enc.transform(X_test)


    # Preprocessing
    logging.info('One hot encode features')
    # enc = OneHotEncoder(n_values=np.max(X_p)+1, handle_unknown='ignore')
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(np.concatenate((X, X_test)))
    X = enc.transform(X)
    X_test = enc.transform(X_test)

    # Split validation data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=args.test_size, shuffle=True)


    # # Train classifier
    logging.info('Train classifier')
    clf = XGBClassifier(n_estimators=200, max_depth=40, verbose=True)
    clf.fit(X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        eval_metric='logloss',
        verbose=True)
    logging.info('Validation score: {}'.format(clf.score(X_val, y_val)))    


    y_test = clf.predict(X_test).astype(np.float64)
    # median = np.median(y_test)
    # y_test = (y_test > median)
    for index in false_indices:
        y_test[index] = 0

    pred = y_test.astype(np.int8)
    print('# of 0:', pred[pred == 0].shape[0])
    print('# of 1:', pred[pred == 1].shape[0])

    with open(args.output_file, 'w') as f:
        f.write('query_id,prediction\n')
        for index, value in enumerate(pred):
            f.write('{},{}\n'.format(index+1, value))

    
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    main(args)
