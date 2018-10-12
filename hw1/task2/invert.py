#!/usr/bin/env python3

import sys


def main(pred_file):
    with open(pred_file, 'r') as f, open(pred_file + '_inv.csv', 'w') as g:
        print('query_id,prediction', file=g)
        f.readline()
        for idx, line in enumerate(f):
            line = 1 - int(line.split(',')[1])
            print('%d,%d' % (1 + idx, int(line)), file=g)


if __name__ == '__main__':
    if len(sys.argv) != 1 + 1:
        print('Input error. (Usage: python3 %s <pred.txt>)' % (sys.argv[0]), file=sys.stderr)
        sys.exit(1)

    main(sys.argv[1])
