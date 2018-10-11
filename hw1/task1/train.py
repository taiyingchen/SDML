import argparse
import sys
import logging
from collections import defaultdict

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier
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
        readfile = np.genfromtxt(file, delimiter=' ', skip_header=1)
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
    # adj_matrix = defaultdict(set)

    # Sampling from train.txt
    for u, v in train_graph:
        X.append(np.concatenate((embed[u], embed[v]), axis=0))
        # Construct adjacency matrix
        adj_matrix[u][v] = 1
        g.add_edge(u, v)
        # adj_matrix[u].add(v)

    # Sampling from test-seen.txt
    for u, v in test_seen_graph:
        X.append(np.concatenate((embed[u], embed[v]), axis=0))
        # Construct adjacency matrix
        adj_matrix[u][v] = 1
        # adj_matrix[u].add(v)

    X = np.array(X)
    y = np.ones(X.shape[0])
    # adj_matrix = csr_matrix(adj_matrix)
    topo_sort = g.topological_sort()
    return X, y, adj_matrix, topo_sort


def negative_sampling(embed, adj_matrix, topo_sort, n_sample):
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
            adj_matrix[u][v] = 1
            X.append(np.concatenate((embed[u], embed[v]), axis=0))

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


def import_test_for_similarity(test_graph, embed, seen_nodes):
    X = []
    Y = []
    for i, j in test_graph:
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


def get_cosine_similarity(X, Y):
    sim = []
    for x, y in zip(X, Y):
        sim.append(cosine_similarity([x], [y])[0])
    sim = np.array(sim)
    return sim


def get_euclidean_similarity(X, Y):
    dist = np.linalg.norm(X-Y, axis=1)
    sim = 1 / dist
    return sim


def predict_by_similarity(sim):
    median = np.median(sim)
    pred = (sim >= median).astype(np.int8)
    print(pred[pred == 0].shape)
    print(pred[pred == 1].shape)
    return pred


def get_ensemble_similarity(models, test_file, seen_nodes, test_graph):
    total_sim = []
    for model in models:
        logging.info('Ensemble {}'.format(model))
        embed = import_embedding(model, 'deepwalk')
        X_test, Y_test = import_test_for_similarity(test_graph, embed, seen_nodes)
        sim = get_cosine_similarity(X_test, Y_test)
        total_sim.append(sim)
    total_sim = np.array(total_sim)
    avg_sim = np.mean(total_sim, axis=0)
    return avg_sim


def main(args):
    """
    Ensemble by averaging similarity
    """
    # test_graph = np.loadtxt(args.test_file).astype(np.int32)
    # seen_nodes = get_seen_nodes([args.train_file, args.test_seen_file])
    # avg_sim = get_ensemble_similarity(['hw1/task1/models/deepwalk_128.embeddings', 'hw1/task1/models/deepwalk_256.embeddings', 'hw1/task1/models/deepwalk_512.embeddings', 'hw1/task1/models/deepwalk_1024.embeddings'], args.test_file, seen_nodes, test_graph)
    # pred = predict_by_similarity(avg_sim)
    # np.savetxt(args.output_file, pred, fmt='%1d')
    # return

    embed = import_embedding(args.embedding_file, args.embedding_file_format)
    """
    Classify by similarity
    """
    # # avg = avg_similarity(args.train_file, args.test_seen_file, embed)
    # test_graph = np.loadtxt(args.test_file).astype(np.int32)
    # seen_nodes = get_seen_nodes([args.train_file, args.test_seen_file])
    # X_test, Y_test = import_test_for_similarity(test_graph, embed, seen_nodes)
    # sim = get_cosine_similarity(X_test, Y_test)
    # # sim = get_euclidean_similarity(X_test, Y_test)
    # pred = predict_by_similarity(sim)
    # np.savetxt(args.output_file, pred, fmt='%1d')

    """
    Classify by positive/negative sampling
    """
    X_p, y_p, adj_matrix, topo_sort = positive_sampling(args.train_file, args.test_seen_file, embed)
    X_n, y_n = negative_sampling(embed, adj_matrix, topo_sort, len(X_p)*args.negative_size)
    X, y = np.concatenate((X_p, X_n)), np.concatenate((y_p, y_n))
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=args.test_size, shuffle=True)
    X_test = import_test(args.test_file, embed, args.embedding_file_format)
    del adj_matrix, X_p, y_p, X_n, y_n

    # Train classifier
    logging.info('Train classifier')
    clf = RandomForestClassifier(n_estimators=100, n_jobs=100, verbose=True)
    clf.fit(X_train, y_train)
        # eval_set=[(X_train, y_train), (X_val, y_val)],
        # eval_metric='logloss',
        # verbose=True)
    logging.info('Validation score: {}'.format(clf.score(X_val, y_val)))
    # Prediction
    y_test = clf.predict(X_test).astype(np.int8)
    # median = np.median(y_test)
    # y_test = (y_test > median)
    print('# of 0:', y_test[y_test == 0].shape[0])
    print('# of 1:', y_test[y_test == 1].shape[0])
    with open(args.output_file, 'w') as f:
        f.write('query_id,prediction\n')
        for index, value in enumerate(y_test):
            f.write('{},{}\n'.format(index+1, value))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO) 
    args = parse_args()
    main(args)
