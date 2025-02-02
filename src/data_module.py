import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from transformers import AutoTokenizer
import json

class QADataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pairs = [(q["short_question"], q["short_answer"])
                    for q in data if q["label"] == 1.0]
        
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        question, answer = self.pairs[idx]
        encoding = self.tokenizer(
            text=question,
            text_target=answer,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze().T,
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': encoding['labels'].squeeze().T
        }
        
class QADataModule(pl.LightningDataModule):
    def __init__(self, data_path, tokenizer_name, batch_size, max_length):
        super().__init__()
        self.data_path = data_path
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.batch_size = batch_size
        self.max_length = max_length
        
    def setup(self, stage=None):
        with open(self.data_path) as f:
            data = json.load(f)
            dataset = QADataset(data, self.tokenizer, self.max_length)
            self.train_data, self.val_data = torch.utils.data.random_split(dataset, [0.9, 0.1])
    
    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=15)
    
    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, num_workers=15)
    