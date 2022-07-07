import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertForSequenceClassification, BertModel
E1 = '[$]'
E2 = '[#]'
CLS = '[CLS]'

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")



class BertForRelationExtraction(nn.Module):
    def __init__(self, model_path='bert-base-uncased', outputs_dim=2, p=0.2):
        super(BertForRelationExtraction, self).__init__()
        self.bert_model = BertModel.from_pretrained(model_path)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(p=p)
        self.cls_linear = nn.Linear(self.bert_model.config.hidden_size, self.bert_model.config.hidden_size)
        self.entity_linear = nn.Linear(self.bert_model.config.hidden_size, self.bert_model.config.hidden_size)
        self.final_linear = nn.Linear(self.bert_model.config.hidden_size*3, outputs_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs, e1_spans, e2_spans):
        out = self.bert_model(**inputs)
        last_hidden_state = out.last_hidden_state
        cls_hidden_state = out.pooler_output
        e1_avg = self.avg_e_index(last_hidden_state, e1_spans)
        e2_avg = self.avg_e_index(last_hidden_state, e2_spans)
        out1 = self.cls_linear(self.tanh(self.dropout(cls_hidden_state)))
        out2 = self.entity_linear(self.tanh(self.dropout(e1_avg)))
        out3 = self.entity_linear(self.tanh(self.dropout(e2_avg)))
        out = torch.cat((out1, out2, out3), dim=1).to(device)
        out = self.final_linear(self.dropout(out))
        return self.sigmoid(out)

    def avg_e_index(self, t, e_spans):
        avg = torch.zeros((t.size(0), t.size(2)), device=device)
        for i, (batch, e_span) in enumerate(zip(t, e_spans)):
            range_ = torch.arange(e_span[0], e_span[1])
            avg[i] = batch[range_].mean(dim=0)
        return avg





