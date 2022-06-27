import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertForSequenceClassification, BertModel
E1 = '[$]'
E2 = '[#]'
CLS = '[CLS]'




class BertForRelationExtraction(nn.Module):
    def __init__(self, model_path='bert-base-uncased', tokenizer_path='bert-base-uncased', outputs_dim=2, p=0.1):
        super(BertForRelationExtraction, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
        self.bert_model = BertModel.from_pretrainded(model_path)
        self.tokenizer.add_tokens([E1, E2], special_tokens=True)
        self.bert_model.resize_token_embeddings(len(self.tokenizer))
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(p=p)
        self.cls_linear = nn.Linear(self.bert_model.config.hidden_size, self.bert_model.config.hidden_size)
        self.entity_linear = nn.Linear(self.bert_model.config.hidden_size, self.bert_model.config.hidden_size)
        self.final_linear = nn.Linear(self.bert_model.config.hidden_size, outputs_dim)

    def forward(self, inputs, e1_spans, e2_spans):
        out = self.bert_model(inputs)
        e1_avg = torch.mean(out[e1_spans[0]: e1_spans[1]], dim=1)
        e2_avg = torch.mean(out[e2_spans[0]: e2_spans[1]], dim=1)
        out1 = self.cls_linear(self.tanh(self.dropout(out[0])))
        out2 = self.entity_linear(self.tanh(self.dropout(e1_avg)))
        out3 = self.entity_linear(self.tanh(self.dropout(e2_avg)))
        out = torch.cat((out1, out2, out3), dim=0)
        out = self.final_linear(self.dropout(out))
        return F.sigmoid(out)



    def preprocess_function(self, examples):
        return self.tokenizer(examples["text"], truncation=True)


