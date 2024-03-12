from torch.utils.data import Dataset
from transformers import BertTokenizer
import torch


class EMAIL_Dataset(Dataset):
    def __init__(self, data, max_len):
        self.data = data
        self.max_len = max_len
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sender = str(self.data.iloc[idx]['sender'])
        subject = str(self.data.iloc[idx]['subject'])
        text = str(self.data.iloc[idx]['text'])
        label = self.data.iloc[idx]['label']
        
        sender_encoding = self.tokenize(sender)
        subject_encoding = self.tokenize(subject)
        text_encoding = self.tokenize(text)
        
        return {
            'sender_input_ids': sender_encoding['input_ids'],
            'sender_attention_mask': sender_encoding['attention_mask'],
            'subject_input_ids': subject_encoding['input_ids'],
            'subject_attention_mask': subject_encoding['attention_mask'],
            'text_input_ids': text_encoding['input_ids'],
            'text_attention_mask': text_encoding['attention_mask'],
            'label': torch.tensor(label, dtype=torch.long)
        }
    
    def tokenize(self, text):
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }
        
