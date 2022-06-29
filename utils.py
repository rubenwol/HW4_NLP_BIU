from flair.data import Sentence
from flair.models import SequenceTagger
import itertools
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, BertModel

E1 = '[$]'
E2 = '[#]'
CLS = '[CLS]'
# load tagger
tagger = SequenceTagger.load("flair/ner-english-large")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


def read_file(fname):
    sentences = {}
    with open(fname, 'r') as f:
        lines = f.readlines()
        for line in lines:
            sent_id, sentence = line.strip().split('\t')
            sentence = sentence.replace("-LRB-", "(")
            sentence = sentence.replace("-RRB-", ")")
            sentences[sent_id] = sentence
    return sentences


def read_annoations_file(fname):
    annotations = []
    with open(fname, 'r') as f:
        lines = f.readlines()
        for line in lines:
            sent_id, e1, rel, e2, sentence = line.strip().split('\t')
            sentence = sentence.replace("-LRB-", "(")
            sentence = sentence.replace("-RRB-", ")")
            sentence = sentence[1:-1]
            annotations.append((sent_id, e1, rel, e2, sentence))
    return annotations


def is_distinct(t1, t2):
    if t1[1] <= t2[0] or t2[1] <= t1[0]:
        return True
    return False

def entity_index_distinct(e1_start_end, e2_start_end):
    if len(e1_start_end) <= len(e2_start_end):
        for t1 in e1_start_end:
            for t2 in e2_start_end:
                if is_distinct(t1, t2):
                    e1_start, e1_end = t1
        for t2 in e2_start_end:
            if is_distinct((e1_start, e1_end), t2):
                e2_start, e2_end = t2
    else:
        for t2 in e2_start_end:
            for t1 in e1_start_end:
                if is_distinct(t1, t2):
                    e2_start, e2_end = t2
        for t1 in e1_start_end:
            if is_distinct((e2_start, e2_end), t1):
                e1_start, e1_end = t1
    return e1_start, e1_end, e2_start, e2_end


def from_annotations_to_samples(annotations):
    samples = []
    for annotation in annotations:
        sent_id, e1, rel, e2, sentence = annotation
        tokens = sentence.split(' ')
        e1 = e1.replace("-LRB-", "(")
        e2 = e2.replace("-RRB-", ")")
        e1_tokens = e1.split(' ')
        e2_tokens = e2.split(' ')
        e1_start_end = [(i, i+len(e1_tokens)) for i in range(len(tokens) - len(e1_tokens) + 1) if e1_tokens == tokens[i:i + len(e1_tokens)]]
        e2_start_end = [(i, i+len(e2_tokens)) for i in range(len(tokens) - len(e2_tokens) + 1) if e2_tokens == tokens[i:i + len(e2_tokens)]]

        e1_start, e1_end, e2_start, e2_end = entity_index_distinct(e1_start_end, e2_start_end)

        if e1_end <= e2_start:
            tokens.insert(e2_end, E2)
            tokens.insert(e2_start, E2)
            tokens.insert(e1_end, E1)
            tokens.insert(e1_start, E1)

        else:
            # add special tokens
            tokens.insert(e1_end, E1)
            tokens.insert(e1_start, E1)        # add special tokens
            tokens.insert(e2_end, E2)
            tokens.insert(e2_start, E2)
        new_sentence = ' '.join(tokens)
        sample = (sent_id, new_sentence, e1, e2, rel)
        samples.append(sample)
    return samples




def get_relevant_pairs_entities(sentence):
    '''
    :param sentence: sentence train
    :return: relevent pairs
    '''
    # LOC_entities = []
    PER_entities = []
    ORG_entities = []
    # make example sentence
    sentence = Sentence(sentence)
    # predict NER tags
    tagger.predict(sentence)
    # iterate over entities
    for entity in sentence.get_spans('ner'):
        # entity : Span[0:2]: "Terry Hands" , sentence : Terry Hands , the subsidized theater ' s artistic director ...
        # Span[0:2] = Span[idx_start, idx_end]
        # idx_start : entity.tokens[0].idx - 1
        # idx_end : entity.tokens[-1].idx
        # maybe replace start and end position by [idx_start, idx_end]
        if entity.tag == 'PER':
            PER_entities.append((entity.text, entity.start_position, entity.end_position))
        # elif entity.tag == 'LOC':
        #     LOC_entities.append((entity.text, entity.start_position, entity.end_position))
        elif entity.tag == 'ORG':
            ORG_entities.append((entity.text, entity.start_position, entity.end_position))
    # relevant_pairs = list(itertools.product(PER_entities, LOC_entities))
    relevant_pairs = list(itertools.product(PER_entities, ORG_entities))
    # TODO: relevent_pairs to bert format and keep sentence id #
    return relevant_pairs



