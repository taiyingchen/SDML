import os
import numpy as np
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('prediction_dir')
    parser.add_argument('output_file')
    return parser.parse_args()


def main(args):
    total_pred = []
    files = os.listdir(args.prediction_dir)
    for file in files:
        pred = np.loadtxt(os.path.join(args.prediction_dir, file))
        total_pred.append(pred)
    pred = np.mean(total_pred, axis=0)
    pred[pred < 0.5] = 0
    pred[pred >= 0.5] = 1
    print('# of 0:', pred[pred == 0].shape[0])
    print('# of 1:', pred[pred == 1].shape[0])
    np.savetxt(args.output_file, pred, fmt='%1d')


if __name__ == '__main__':
    args = parse_args()
    main(args)
