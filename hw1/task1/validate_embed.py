import numpy as np
import argparse
import logging
from sklearn.metrics.pairwise import cosine_similarity
from scipy import stats


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('validation_file')
    parser.add_argument('embedding_file')
    parser.add_argument('embedding_file_format')
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


def get_similarity(X, Y):
    sim = []
    for x, y in zip(X, Y):
        sim.append(cosine_similarity([x], [y])[0])
    sim = np.array(sim)
    return sim


def import_validation_data(graph, embed):
    X = []
    Y = []
    for u, v in graph:
        X.append(embed[u])
        Y.append(embed[v])
    X = np.array(X)
    Y = np.array(Y)
    return X, Y


def main(args):
    graph = np.loadtxt(args.validation_file).astype(np.int32)
    embed = import_embedding(args.embedding_file, args.embedding_file_format)
    X, Y = import_validation_data(graph, embed)
    sim = get_similarity(X, Y)
    print(stats.describe(sim))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    main(args)
