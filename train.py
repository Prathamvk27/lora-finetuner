import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BartTokenizer
from datasets import load_dataset
from model import MyBartSummarizer # Importing your class
from tqdm.auto import tqdm

# 1. SETUP
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_checkpoint = "facebook/bart-base"
tokenizer = BartTokenizer.from_pretrained(model_checkpoint)

# 2. DATA PREPARATION (The part you had in .ipynb)
def prepare_data():
    dataset = load_dataset("cnn_dailymail", "3.0.0")
    
    def tokenize_fn(batch):
        # We need to tokenize input and the "target" (labels)
        inputs = tokenizer(batch["article"], truncation=True, padding="max_length", max_length=512)
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(batch["highlights"], truncation=True, padding="max_length", max_length=128)
        inputs["labels"] = labels["input_ids"]
        return inputs

    # Slicing for speed during learning
    tokenized_dataset = dataset["train"].select(range(1000)).map(tokenize_fn, batched=True)
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return DataLoader(tokenized_dataset, batch_size=4, shuffle=True)

# 3. TRAINING LOOP
def train():
    dataloader = prepare_data()
    model = MyBartSummarizer(model_checkpoint).to(device)
    
    # Optimizer only sees the head's parameters
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    
    # ignore_index=-100 tells the loss to ignore padding tokens
    loss_fct = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    model.train()
    for epoch in range(3):
        total_loss = 0
        for batch in tqdm(dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # In BART, decoder_input_ids are usually the labels shifted right
            # But for a custom head training, passing labels as decoder_input is okay
            logits = model(input_ids, attention_mask, labels)
            
            loss = loss_fct(logits.view(-1, 50265), labels.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch} Loss: {total_loss/len(dataloader)}")

if __name__ == "__main__":
    train()