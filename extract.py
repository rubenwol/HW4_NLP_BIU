import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_scheduler
from model import BertForRelationExtraction
from utils import *
from tqdm.auto import tqdm

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class RelationDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        label = 1 if self.samples[idx][4] == 'Live_In' else 0
        # tokens = tokenizer(self.samples[idx][1], truncation=True)
        return self.samples[idx][1], label


def collate_fn(batch):
    sentences = list(map(lambda data: data[0], batch))
    labels = list(map(lambda data: data[1], batch))
    tokens = tokenizer(sentences, padding=True, truncation=True)
    input_ids = torch.tensor(tokens['input_ids'], device=device)
    attention_mask = torch.tensor(tokens['attention_mask'], device=device)
    e1_spans = (input_ids == E1_id).nonzero(as_tuple=True)[1].view(-1, 2).to(device)
    e2_spans = (input_ids == E2_id).nonzero(as_tuple=True)[1].view(-1, 2).to(device)
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'e1_spans': e1_spans,
        'e2_spans': e2_spans,
        'labels': labels
    }




if __name__ == '__main__':

    train_annotations = read_annoations_file('data/TRAIN.annotations')
    dev_annotations = read_annoations_file('data/DEV.annotations')
    train_dataset = from_annotations_to_samples(train_annotations)
    dev_dataset = from_annotations_to_samples(dev_annotations)
    train_dataset = RelationDataset(train_dataset)
    dev_dataset = RelationDataset(dev_dataset)

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", special_tokens=True)
    tokenizer.add_tokens(['[$]', '[#]'])
    E1_id = tokenizer.convert_tokens_to_ids(E1)
    E2_id = tokenizer.convert_tokens_to_ids(E2)

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=8, collate_fn=collate_fn)
    eval_dataloader = DataLoader(dev_dataset, batch_size=8, collate_fn=collate_fn)

    model = BertForRelationExtraction()
    model.bert_model.resize_token_embeddings(len(tokenizer))

    optimizer = AdamW(model.parameters(), lr=5e-5)
    num_epochs = 3
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    loss_fn = nn.MSELoss()

    model.to(device)
    progress_bar = tqdm(range(num_training_steps))

    model.train()
    for epoch in range(num_epochs):
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
            # print(f'Current loss: {loss}', end='\r')
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
        print(f'loss: {cum_loss/len(train_dataloader)}')

    # metric = load_metric("accuracy")
    # model.eval()
    # for batch in eval_dataloader:
    #     batch = {k: v.to(device) for k, v in batch.items()}
    #     with torch.no_grad():
    #         outputs = model(**batch)
    #
    #     logits = outputs.logits
    #     predictions = torch.argmax(logits, dim=-1)
    #     metric.add_batch(predictions=predictions, references=batch["labels"])
    #
    # metric.compute()
