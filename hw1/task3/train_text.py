import argparse
import logging
import os
import re
from dateutil.parser import parse
import datetime
import time
import math
from utils import *

import numpy as np
from collections import defaultdict

INTERVAL_DAY = 32


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a Word2Vec encoding on input, and store the resulting model in output")
    parser.add_argument('--doc_dir', type=str, required=True,
                        help='a dir with paper info')
    parser.add_argument('--input', type=str, required=True,
                        help='a *.npy file with parsed titles')
    parser.add_argument('--size', type=int, default=100,
                        help='size of encoding vectors')
    parser.add_argument('--window', type=int, default=10,
                        help='size of window scanning over text')
    parser.add_argument('--mincount', type=int, default=5,
                        help='minimum number of times a word has to appear to participate')
    parser.add_argument('--output', type=str, required=True,
                        help='output filename for saving the model')
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

    title = parse_title(' '.join(title))
    abstract = parse_title(' '.join(abstract))
    return date, title, abstract


def parse_date(date):
    date = ' '.join(date.split()[:4])
    date = parse(date)
    timestamp = time.mktime(date.timetuple())
    return timestamp


def main(args):
    inputfile = args.input
    size = args.size
    window = args.window
    mincount = args.mincount
    outputfile = args.output
    doc_dir = args.doc_dir

    print("Training model with\n")
    print("{0:30} = {1}".format("input", inputfile))
    print("{0:30} = {1}".format("size", size))
    print("{0:30} = {1}".format("window", window))
    print("{0:30} = {1}".format("mincount", mincount))

    all_titles = np.atleast_2d(np.load(inputfile))[0][0]
    all_years = sorted(list(all_titles.keys()))
    titles = get_titles_for_years(all_titles, all_years)
    
    for file in os.listdir(doc_dir):
        date, title, abstract = parse_xml(os.path.join(doc_dir, file))
        titles.append(title)
        titles.append(abstract)

    ngram_titles, bigrams, ngrams = get_ngrams(titles)

    model = gensim.models.Word2Vec(ngram_titles, window=window, min_count=mincount, size=size)
    print("Saving to {0}".format(outputfile))
    model.save(outputfile)
    print("Done!")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    main(args)
