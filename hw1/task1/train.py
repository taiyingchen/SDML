import argparse
import sys
import logging
from collections import defaultdict

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC, LinearSVR

from graph import Graph


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('train_file')
    parser.add_argument('test_seen_file')
    parser.add_argument('test_file')
    parser.add_argument('embedding_file')
    parser.add_argument('embedding_file_format')
    parser.add_argument('output_file')
    parser.add_argument('--negative_size', type=float, default=1.0)
    parser.add_argument('--test_size', type=float, default=0.1)
    return parser.parse_args()


def import_embedding(file, format):
    logging.info('Import embedding model')
    if format == 'prune':
        embed = np.genfromtxt(file, delimiter=',')
    elif format == 'deepwalk':
        readfile = np.genfromtxt(file, delimiter=' ')
        embed = {}
        for i in readfile:
            embed[int(i[0])] = i[1:]
    return embed


def get_seen_nodes(files):
    nodes = set()
    for file in files:
        graph = np.loadtxt(file).astype(np.int32)
        for i, j in graph:
            nodes.update([i, j])
    return nodes


def positive_sampling(train_file, test_seen_file, embed):
    logging.info('Positive sampling')
    X = []
    g = Graph(train_file)
    train_graph = np.loadtxt(train_file).astype(np.int32)
    test_seen_graph = np.loadtxt(test_seen_file).astype(np.int32)
    max_index = max(train_graph.max(), test_seen_graph.max()) + 1
    adj_matrix = np.zeros((max_index, max_index), dtype=np.int8)
    adj_matrix = defaultdict(set)

    # Sampling from train.txt
    for i, j in train_graph:
        X.append(np.concatenate((embed[i], embed[j]), axis=0))
        # Construct adjacency matrix
        # adj_matrix[int(i)][int(j)] = 1
        g.add_edge(i, j)
        adj_matrix[i].add(j)

    # Sampling from test-seen.txt
    for i, j in test_seen_graph:
        X.append(np.concatenate((embed[i], embed[j]), axis=0))
        # Construct adjacency matrix
        # adj_matrix[int(i)][int(j)] = 1
        adj_matrix[i].add(j)

    X = np.array(X)
    y = np.ones(X.shape[0])
    # adj_matrix = csr_matrix(adj_matrix)
    topo_sort = g.topological_sort()
    return X, y, adj_matrix, topo_sort


def negative_sampling(embed, adj_matrix, topo_sort, n_sample):
    logging.info('Negative sampling')
    X = []
    hop_matrix = adj_matrix

    while len(X) < int(n_sample):
        if len(X) % 10000 == 0: print(len(X))
        index_i, index_j = np.random.randint(1, len(topo_sort), 2)
        if index_i > index_j:
            index_i, index_j = index_j, index_i
        i, j = topo_sort[index_i], topo_sort[index_j]
        # if hop_matrix[i][j] == 0:
        if j not in hop_matrix[i]:
            # Add i->j to adjacency matrix to prevent duplicate negative sample
            # hop_matrix[i][j] = 1
            hop_matrix[i].add(j)
            X.append(np.concatenate((embed[i], embed[j]), axis=0))

    X = np.array(X)
    y = np.zeros((X.shape[0]))
    return X, y


# def nearest_neighbor_negative_sampling():



def import_test(file, embed, format):
    logging.info('Import testing data')
    X = []
    graph = np.loadtxt(file).astype(np.int32)
    if format == 'prune':
        for i, j in graph:
            X.append(np.concatenate((embed[i], embed[j]), axis=0))
    elif format == 'deepwalk':
        for i, j in graph:
            if i in embed and j in embed:
                X.append(np.concatenate((embed[i], embed[j]), axis=0))
            else:
                X.append(np.concatenate((np.ones(embed[1].shape), np.ones(embed[1].shape)), axis=0))
    X = np.array(X)
    return X


def import_test_for_similarity(file, embed, seen_nodes):
    X = []
    Y = []
    graph = np.loadtxt(file).astype(np.int32)
    for i, j in graph:
        # if i in embed and j in embed:
        if i in seen_nodes and j in seen_nodes:
            X.append(embed[i])
            Y.append(embed[j])
        else:
            X.append(np.ones(embed[1].shape))
            Y.append(np.ones(embed[1].shape))
    X = np.array(X)
    Y = np.array(Y)
    return X, Y


def predict_similarity(X, Y):
    sim = []
    for x, y in zip(X, Y):
        sim.append(cosine_similarity([x], [y])[0])
    sim = np.array(sim)
    median = np.median(sim)
    pred = (sim >= median).astype(np.int8)
    print(pred[pred == 0].shape)
    print(pred[pred == 1].shape)
    return pred


def main(args):
    embed = import_embedding(args.embedding_file, args.embedding_file_format)

    """
    Classify by similarity
    """
    # avg = avg_similarity(args.train_file, args.test_seen_file, embed)
    # seen_nodes = get_seen_nodes([args.train_file, args.test_seen_file])
    # X_test, Y_test = import_test_for_similarity(args.test_file, embed, seen_nodes)
    # pred = predict_similarity(X_test, Y_test)
    # np.savetxt(args.output_file, pred, fmt='%1d')

    """
    Classify by positive/negative sampling
    """
    X_p, y_p, adj_matrix, topo_sort = positive_sampling(args.train_file, args.test_seen_file, embed)
    X_n, y_n = negative_sampling(embed, adj_matrix, topo_sort, len(X_p)*args.negative_size)
    X, y = np.concatenate((X_p, X_n)), np.concatenate((y_p, y_n))
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=args.test_size, shuffle=True)
    X_test = import_test(args.test_file, embed, args.embedding_file_format)

    # Train classifier
    clf = LinearSVC()
    clf.fit(X_train, y_train)
    print(clf.score(X_val, y_val))
    # Prediction
    y_test = clf.predict(X_test).astype(np.bool)
    np.savetxt(args.output_file, y_test, fmt='%1d')
    print(y_test[y_test == 0].shape)
    print(y_test[y_test == 1].shape)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO) 
    args = parse_args()
    main(args)
