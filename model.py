import torch
import torch.nn as nn
from transformers import BartModel

# this is head only training

class Summarization_head(nn.Module):
    def __init__(self, input_dim, vocab_size):
        super().__init__()
        self.custom_layers = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512,vocab_size)
        )

    def forward(self,x):
        return self.custom_layers(x)

class MyBartSummarizer(nn.Module):
    def __init__(self, checkpoint="facebook/bart-base"):
        super().__init__()

        self.bart = BartModel.from_pretrained(checkpoint)

        for param in self.bart.parameters():
            param.requires_grad = False

        self.head = Summarization_head(input_dim=768,vocab_size=50265)

    # forwarding after freezing bart
    def forward(self, input_ids, attention_mask, decoder_input_ids):
    # Pass data through the frozen bart

        outputs = self.bart(
            input_ids = input_ids,
            attention_mask = attention_mask,
            decoder_input_ids = decoder_input_ids
        )

        logits = self.head(outputs.last_hidden_state)
        return logits

