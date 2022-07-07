from collections import defaultdict
import sys
# f_pred = "DEV.output"
# f_test = "data/DEV.annotations"


def read_relations_file(fname):
    relations = []
    with open(fname, 'r') as f:
        lines = f.readlines()
        for line in lines:
            sent_id, e1, rel, e2, sentence = line.strip().split('\t')
            if rel == 'Work_For':
                relations.append((sent_id, e1, rel, e2))
    return relations


def compute_score(f_pred, f_test):
    pred = read_relations_file(f_pred)
    test = read_relations_file(f_test)
    true_pos = sum([1 for r in test if r in pred])
    precision = true_pos/len(pred) if len(pred) > 0 else 1
    recall = true_pos/len(test)
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    print("F1 score: ", f1_score)
    print("Precision: ", precision)
    print("Recall: ", recall)

    return f1_score, precision, recall


if __name__ == '__main__':
    f_test = sys.argv[1]
    f_pred = sys.argv[2]
    compute_score(f_pred, f_test)