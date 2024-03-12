import torch.nn as nn
from transformers import BertModel
import torch


class BERT_Transformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(BERT_Transformer, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.fc = nn.Linear(input_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, sender_input_ids, sender_attention_mask, subject_input_ids, subject_attention_mask, text_input_ids, text_attention_mask):
        sender_output = self.bert(input_ids=sender_input_ids, attention_mask=sender_attention_mask)['pooler_output']
        subject_output = self.bert(input_ids=subject_input_ids, attention_mask=subject_attention_mask)['pooler_output']
        text_output = self.bert(input_ids=text_input_ids, attention_mask=text_attention_mask)['pooler_output']
        
        combined_output = torch.cat((sender_output, subject_output, text_output), dim=1)
        output = self.fc(combined_output)
        output = self.dropout(output)
        output = self.out(output)
        return output
