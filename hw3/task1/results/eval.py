import argparse


def parse_args():
    parser = argparse.ArgumentParser('Evaluate hw3-1 control results')
    parser.add_argument('--predict_file', required=True)
    parser.add_argument('--test_file', required=True)
    return parser.parse_args()


def main(args):
    total_pos = 0
    total_row = 0
    hit_pos = 0
    hit_rhyme = 0
    hit_len = 0

    with open(args.predict_file) as pf, open(args.test_file) as tf:
        for prow, trow in zip(pf, tf):
            # ground truth format: 'p m v d m NOP e NOE 5 NOR'

            # Parsing ground truth
            gt = trow.split(' ')
            gt = gt[gt.index('EOS')+1:] # Start element after EOS
            NOP_index = gt.index('NOP')
            gt_pos = gt[:NOP_index] # POS tagging
            gt_rhyme = gt[NOP_index+1] # rhyme
            gt_len = int(gt[NOP_index+3]) # length

            # Parsing predict features
            features = prow.split(',')
            pred_pos = features[0].split(' ')
            pred_rhyme = features[2].strip()
            pred_len = int(features[1])
            
            # Calculate accuracy
            for i in range(len(gt_pos)):
                if i < len(pred_pos) and gt_pos[i] == pred_pos[i]:
                    hit_pos += 1
                total_pos += 1
            if gt_rhyme == pred_rhyme:
                hit_rhyme += 1
            if gt_len == pred_len:
                hit_len += 1
            total_row += 1
    
    print('POS tagging accuracy: ', hit_pos/total_pos)
    print('Rhyme accuracy:', hit_rhyme/total_row)
    print('Length accuracy:', hit_len/total_row)


if __name__ == "__main__":
    args = parse_args()
    main(args)