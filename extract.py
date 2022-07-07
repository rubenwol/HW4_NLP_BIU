import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_scheduler
from model import BertForRelationExtraction
from utils import *
from tqdm.auto import tqdm
from eval import compute_score
import sys

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
lr = 2e-5

INPUT_FILE = sys.argv[1]
OUTPUT_FILE = sys.argv[2]

class RelationDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        label = 1 if self.samples[idx][4] == 'Work_For' else 0
        return self.samples[idx][1], label, self.samples[idx][0], self.samples[idx][2], self.samples[idx][3], self.samples[idx][5]

class RelationDatasetTest(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        '''

        :return: self.samples[idx][1] , self.samples[idx][0], self.samples[idx][2], self.samples[idx][3]
        or with word : sentence_for_bert, sent_id, e1, e2
        '''
        # tokens = tokenizer(self.samples[idx][1], truncation=True)
        return self.samples[idx][1], self.samples[idx][0], self.samples[idx][2], self.samples[idx][3], self.samples[idx][4]

def collate_fn(batch):
    sentences = list(map(lambda data: data[0], batch))
    labels = list(map(lambda data: data[1], batch))
    sent_ids = list(map(lambda data: data[2], batch))
    e1 = list(map(lambda data: data[3], batch))
    e2 = list(map(lambda data: data[4], batch))
    or_sentences = list(map(lambda data: data[5], batch))
    tokens = tokenizer(sentences, padding=True, truncation=True)
    input_ids = torch.tensor(tokens['input_ids'], device=device)
    attention_mask = torch.tensor(tokens['attention_mask'], device=device)
    e1_spans = (input_ids == E1_id).nonzero(as_tuple=True)[1].view(-1, 2).to(device)
    e2_spans = (input_ids == E2_id).nonzero(as_tuple=True)[1].view(-1, 2).to(device)
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'e1': e1,
        'e2': e2,
        'e1_spans': e1_spans,
        'e2_spans': e2_spans,
        'labels': labels,
        'sent_ids': sent_ids,
        'sentences': or_sentences
    }

def collate_fn_test(batch):
    sentences = list(map(lambda data: data[0], batch))
    sent_ids = list(map(lambda data: data[1], batch))
    e1 = list(map(lambda data: data[2], batch))
    e2 = list(map(lambda data: data[3], batch))
    or_sentences = list(map(lambda data: data[4], batch))
    tokens = tokenizer(sentences, padding=True, truncation=True)
    input_ids = torch.tensor(tokens['input_ids'], device=device)
    attention_mask = torch.tensor(tokens['attention_mask'], device=device)
    e1_spans = (input_ids == E1_id).nonzero(as_tuple=True)[1].view(-1, 2).to(device)
    e2_spans = (input_ids == E2_id).nonzero(as_tuple=True)[1].view(-1, 2).to(device)
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'e1': e1,
        'e2': e2,
        'e1_spans': e1_spans,
        'e2_spans': e2_spans,
        'sent_ids': sent_ids,
        'sentences': or_sentences
    }

def train_loop(train_dataloader, model, optimizer,loss_fn, lr_scheduler, progress_bar):
    cum_loss = 0
    for batch in train_dataloader:
        batch_input = {
            'input_ids': batch['input_ids'],
            'attention_mask': batch['attention_mask']
        }
        e1_spans = batch['e1_spans']
        e2_spans = batch['e2_spans']
        # need to identitif
        outputs = model(batch_input, e1_spans, e2_spans)
        labels = torch.tensor(batch['labels'], device=device)
        one_hot_labels = F.one_hot(labels, 2)
        loss = loss_fn(outputs, one_hot_labels.float())
        loss.backward()
        cum_loss += loss
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
    print(f'loss: {cum_loss / len(train_dataloader)}')



def eval_test(eval_dataloader, model, output_file):
    f = open(output_file, 'w')
    with torch.no_grad():
        for batch in eval_dataloader:
            batch_input = {
                'input_ids': batch['input_ids'],
                'attention_mask': batch['attention_mask']
            }
            e1_spans = batch['e1_spans']
            e2_spans = batch['e2_spans']
            sent_ids = batch['sent_ids']# list of sentence ids
            e1 = batch['e1']
            e2 = batch['e2']
            sentences = batch['sentences']
            outputs = model(batch_input, e1_spans, e2_spans)
            pred = torch.argmax(outputs, dim=1)# list of prediction (1 or 0) [1,1,0,0....]

            for i in range(len(pred)):
                if pred[i] == 1:
                    f.write(f'{sent_ids[i]}\t{e1[i]}\tWork_For\t{e2[i]}\t( {sentences[i]} )\n')
    f.close()



def main():
    global E1_id
    global E2_id

    train_annotations = read_annoations_file('data/TRAIN.annotations')
    train_dataset = from_annotation_to_samples_ner_train(train_annotations)
    train_dataset = RelationDataset(train_dataset)

    dev_corpus = read_file('data/Corpus.DEV.txt')
    dev_dataset = create_test_samples(dev_corpus)
    dev_dataset = RelationDatasetTest(dev_dataset)

    test_corpus = read_file(INPUT_FILE)
    test_dataset = create_test_samples(test_corpus)
    test_dataset = RelationDatasetTest(test_dataset)

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", special_tokens=True)
    tokenizer.add_tokens(['[$]', '[#]'])
    E1_id = tokenizer.convert_tokens_to_ids(E1)
    E2_id = tokenizer.convert_tokens_to_ids(E2)

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=16, collate_fn=collate_fn)
    dev_dataloader = DataLoader(dev_dataset, batch_size=16, collate_fn=collate_fn_test)
    test_dataloader = DataLoader(test_dataset, batch_size=16, collate_fn=collate_fn_test)

    best_model = BertForRelationExtraction()
    best_model.bert_model.resize_token_embeddings(len(tokenizer))

    model = BertForRelationExtraction()
    model.bert_model.resize_token_embeddings(len(tokenizer))

    optimizer = AdamW(model.parameters(), lr=lr)
    num_epochs = 10
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    loss_fn = nn.MSELoss()

    model.to(device)
    best_model.to(device)

    progress_bar = tqdm(range(num_training_steps))
    best_f1 = 0
    model.train()
    for epoch in range(num_epochs):
        model.train()
        train_loop(train_dataloader, model, optimizer, loss_fn, lr_scheduler, progress_bar)
        model.eval()
        print('EVALUATION TRAINING SET')
        # eval(train_dataloader, model)
        eval_test(train_dataloader, model, 'TRAIN.output')
        f1_train = compute_score('TRAIN.output', 'data/TRAIN.annotations')
        print('*'*20)
        print('EVALUATION DEV SET')
        # eval(eval_dataloader, model)
        eval_test(dev_dataloader, model, 'DEV.output')
        f1_dev, _ , _ = compute_score('DEV.output', 'data/DEV.annotations')
        print(lr_scheduler.get_last_lr()[0])
        if f1_dev > best_f1:
            print('BEST')
            best_model.load_state_dict(model.state_dict())

    best_model.eval()
    print()
    print('TEST')
    print(f'BEST LR: {lr}')
    eval_test(test_dataloader, best_model, OUTPUT_FILE)

if __name__ == '__main__':
    main()
