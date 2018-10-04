import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("train_file")
parser.add_argument("test_seen_file")
parser.add_argument("output_file")
args = parser.parse_args()

def merge_file(train_file, test_seen_file, output_file):
    with open(train_file, 'r') as f, open(test_seen_file, 'r') as d, open(output_file, 'w') as g:
        for line in f:
            g.write(line)
        for line in d:
            g.write(line)

def main():
    merge_file(args.train_file, args.test_seen_file, args.output_file)

if __name__ == '__main__':
    main()