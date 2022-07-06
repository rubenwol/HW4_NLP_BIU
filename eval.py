# TODO: accuracy f1 precision recall
from collections import defaultdict
f_pred = "DEV.output"
f_test = "data/DEV.annotations"

# def read_relations(f_test_name, f_pred_name):
#     test_relations = []
#     pred_relations = []
#     data_lines = []
#     pred_lines = []
#
#     with open(f_test_name, 'r') as f:
#         data_lines = f.readlines()
#
#     with open(f_pred_name, 'r') as f:
#         pred_lines = f.readlines()
#         pred_lines = [line.strip().split('\t') for line in pred_lines]
#
#     for test_line in data_lines:
#         sent_id, e1, rel, e2, sentence = test_line.strip().split('\t')
#         if rel == 'Work_For':
#             test_relations.append((sent_id, e1, rel, e2, 1))
#         else:
#             test_relations.append((sent_id, e1, rel, e2, 0))
#         if (sent_id, e1, rel, e2) in pred_lines:
#             pred_relations.append((sent_id, e1, rel, e2, 1))
#         else:
#
#
#     with open(f_pred_name, 'r') as f:
#         lines = f.readlines()
#         for line in lines:
#             sent_id, e1, rel, e2 = line.strip().split('\t')
#             pred_relations.append((sent_id, e1, rel, e2, 1))
#
#     for test_relations
#
#     return test_relations


def read_relations_file(fname):
    relations = []
    with open(fname, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if 'annotations' in fname:
                sent_id, e1, rel, e2, sentence = line.strip().split('\t')
            else:
                sent_id, e1, rel, e2 = line.strip().split('\t')
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

    return f1_score


# pred = read_relations_file(f_pred)
# test = read_relations_file(f_test)
# compute_score(pred, test)
