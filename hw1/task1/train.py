import argparse
import sys

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import cosine_similarity


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
    X = []
    train_nodes = set()
    train_graph = np.loadtxt(train_file).astype(np.int32)
    test_seen_graph = np.loadtxt(test_seen_file).astype(np.int32)
    max_index = max(train_graph.max(), test_seen_graph.max()) + 1
    adj_matrix = np.zeros((max_index, max_index)).astype(np.int8)

    # Sampling from train.txt
    for i, j in train_graph:
        X.append(np.concatenate((embed[i], embed[j]), axis=0))
        # Construct adjacency matrix
        adj_matrix[int(i)][int(j)] = 1
        """
        if i not in adj_matrix:
            adj_matrix[i] = set()
        adj_matrix[i].add(j)
        """
        train_nodes.update([i, j])
    # Sampling from test-seen.txt
    for i, j in test_seen_graph:
        X.append(np.concatenate((embed[i], embed[j]), axis=0))
        # Construct adjacency matrix
        adj_matrix[int(i)][int(j)] = 1
        """
        if i not in adj_matrix:
            adj_matrix[i] = set()
        adj_matrix[i].add(j)
        """

    X = np.array(X)
    y = np.ones(X.shape[0])
    return X, y, adj_matrix, train_nodes


def negative_sampling(embed, adj_matrix, train_nodes, n_sample):
    X = []
    hop_matrix = adj_matrix

    while len(X) < int(n_sample):
        i, j = np.random.randint(1, len(embed), 2)
        if (i in train_nodes and j in train_nodes) and hop_matrix[i][j] == 0:
            # Add i->j to adjacency matrix to prevent duplicate negative sample
            hop_matrix[i][j] = 0
            X.append(np.concatenate((embed[i], embed[j]), axis=0))

    X = np.array(X)
    y = np.zeros((X.shape[0],))
    return X, y

    
def avg_similarity(train_file, test_seen_file, embed):
    X = []
    Y = []
    train_graph = np.loadtxt(train_file).astype(np.int32)
    test_seen_graph = np.loadtxt(test_seen_file).astype(np.int32)
    graph = np.concatenate((train_graph, test_seen_graph))
    for i, j in graph:
        X.append(embed[i])
        Y.append(embed[j])
    X = np.array(X)
    Y = np.array(Y)
    
    sim = []
    for x, y in zip(X, Y):
        sim.append(cosine_similarity([x], [y])[0])
    # sim = np.linalg.norm(X-Y, axis=1)
    sim = np.array(sim)
    sim = sim[sim >= 0]
    return np.average(sim)


def import_test(file, embed):
    X = []
    graph = np.loadtxt(file).astype(np.int32)
    for i, j in graph:
        X.append(np.concatenate((embed[i], embed[j]), axis=0))
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
    # avg = avg_similarity(args.train_file, args.test_seen_file, embed)
    seen_nodes = get_seen_nodes([args.train_file, args.test_seen_file])
    X_test, Y_test = import_test_for_similarity(args.test_file, embed, seen_nodes)
    pred = predict_similarity(X_test, Y_test)
    np.savetxt(args.output_file, pred, fmt='%1d')

    """
    X_p, y_p, adj_matrix, train_nodes = positive_sampling(args.train_file, args.test_seen_file, embed)
    X_n, y_n = negative_samplinsg(embed, adj_matrix, train_nodes, len(X_p)*args.negative_size)
    X, y = np.concatenate((X_p, X_n)), np.concatenate((y_p, y_n))
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=args.test_size, shuffle=True)
    X_test = import_test(args.test_file, embed)

    # Train classifier
    clf = LinearSVC()
    clf.fit(X_train, y_train)
    print(clf.score(X_val, y_val))
    # Prediction
    y_test = clf.predict(X_test).astype(np.bool)
    np.savetxt(args.output_file, y_test, fmt='%1d')
    """


if __name__ == '__main__':
    args = parse_args()
    main(args)
