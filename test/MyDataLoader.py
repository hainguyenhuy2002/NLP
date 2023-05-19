from torch.utils.data import Dataset
from transformers import AutoModel, AutoTokenizer
import torch

class MyDataset(Dataset):
    def __init__(self, data, max_content_length=1024, max_summary_length=256):
        self.data = data
        self.tokenizer = AutoTokenizer.from_pretrained("vinai/bartpho-syllable")
        self.max_summary_length = max_summary_length
        self.max_content_length = max_content_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Get the input and output sequences for the current example
        row = self.data.iloc[idx]
        content = row["original"]
        summary = row["summary"]
        
        # Tokenize the input text and truncate to max_seq_length
        content_input_ids = self.tokenizer(content, max_length=self.max_content_length, return_tensors="pt",
                                    padding='max_length', pad_to_max_length=True,
                                    truncation=True, return_token_type_ids=False)
        
        summary_input_ids = self.tokenizer(summary, max_length=self.max_summary_length, return_tensors="pt",
                                    padding='max_length', pad_to_max_length=True,
                                    truncation=True, return_token_type_ids=False)
        
        # Return the input and output sequences as PyTorch tensors
        return content_input_ids['input_ids'].reshape(-1), content_input_ids['attention_mask'].reshape(-1), \
            summary_input_ids['input_ids'].reshape(-1), summary_input_ids['attention_mask'].reshape(-1), summary      