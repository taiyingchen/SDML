import csv
import logging

import jieba.posseg
from pypinyin import Style, lazy_pinyin


def main(args):
    with open(args.input) as raw_file, open(args.output, 'w') as csv_file:
        writer = csv.writer(csv_file)
        for idx, line in enumerate(raw_file):
            sentence = line.strip().split(' ')
            if len(sentence) == 0:
                pos_tags = []
                tone = 'NONE'
            else:
                try:
                    segments, pos_tags = zip(*jieba.posseg.cut(''.join(sentence)))
                    tone = lazy_pinyin(sentence[-1][-1], style=Style.FINALS, strict=False)[0]
                except:
                    segments, pos_tags = '', []
                    tone = 'err'
                # sentence[-1][-1]: the last word of the last token in the sentence
            seg_length = len(sentence)
            writer.writerow([' '.join(pos_tags), seg_length, tone])
            if idx % 1000 == 0:
                logging.info(f'{idx+1} sentences done')


def parse_args():
    import argparse
    parser = argparse.ArgumentParser('seg and pos')
    parser.add_argument('input', default='./data/new_poetry.csv')
    parser.add_argument('output', default='./data/new_poetry.txt')
    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
    main(parse_args())
