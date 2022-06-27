from flair.data import Sentence
from flair.models import SequenceTagger
import itertools
# load tagger
tagger = SequenceTagger.load("flair/ner-english-large")

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

def get_relevant_pairs_entities(sentence):
    LOC_entities = []
    PER_entities = []
    # make example sentence
    sentence = Sentence(sentence)
    # predict NER tags
    tagger.predict(sentence)
    # iterate over entities
    for entity in sentence.get_spans('ner'):
        if entity.tag == 'LOC':
            LOC_entities.append((entity.text, entity.start_position, entity.end_position))
        elif entity.tag == 'PER':
            PER_entities.append((entity.text, entity.start_position, entity.end_position))
    relevant_pairs = list(itertools.product(LOC_entities, PER_entities))
    return relevant_pairs






