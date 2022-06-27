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


class RelationDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


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
            annotations.append((sent_id,e1, rel, e2, sentence))
    return annotations

def from_annotations_to_samples(annotations):
    samples = []
    for annotation in annotations:
        sent_id, e1, rel, e2, sentence = annotation
        tokens = sentence.split(' ')
        e1_tokens = e1.split(' ')
        e2_tokens = e2.split(' ')
        e1_start, e1_end = [(i,i+len(e1_tokens)) for i in range(len(tokens) - 1) if e1_tokens == tokens[i:i + len(e1_tokens)]][0]
        # add special tokens
        tokens.insert(e1_end, E1)
        tokens.insert(e1_start, E1)
        e2_start, e2_end = [(i,i+len(e2_tokens)) for i in range(len(tokens) - 1) if e2_tokens == tokens[i:i + len(e2_tokens)]][0]
        # add special tokens
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
    LOC_entities = []
    PER_entities = []
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
        if entity.tag == 'LOC':
            LOC_entities.append((entity.text, entity.start_position, entity.end_position))
        elif entity.tag == 'PER':
            PER_entities.append((entity.text, entity.start_position, entity.end_position))
    relevant_pairs = list(itertools.product(LOC_entities, PER_entities))
    return relevant_pairs



