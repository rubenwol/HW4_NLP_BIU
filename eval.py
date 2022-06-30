from collections import defaultdict
f_pred = "DEV.output"
f_test = "data/DEV.annotations"


def read_relations_file(fname):
    relations = []
    with open(fname, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if fname == "data/DEV.annotations" or fname == "data/TRAIN.annotations":
                sent_id, e1, rel, e2, sentence = line.strip().split('\t')
            else:
                sent_id, e1, rel, e2 = line.strip().split('\t')
            if rel == 'Work_For':
                relations.append((sent_id, e1, rel, e2))
    return relations


def score(pred, test):
    true_pos = sum([1 for r in test if r in pred])
    precision = true_pos/len(pred) if len(pred) != 0 else 1
    recall = true_pos/len(test)
    f1_score = 2 * (precision * recall) / (precision + recall)
    print("F1 score: ", f1_score)
    print("Precision: ", precision)
    print("Recall: ", recall)

def compute_score(f_pred, f_test):
    pred = read_relations_file(f_pred)
    test = read_relations_file(f_test)
    score(pred, test)
