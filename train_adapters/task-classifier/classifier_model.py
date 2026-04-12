import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class ClassifierModel(nn.Module):
    def __init__(self, encoder_model='bert-base-uncased', emb_dim=768, classification_labels=5):
        super().__init__()
        self.encoder_model = AutoModel.from_pretrained(encoder_model)
        self.linear1 = nn.Linear(emb_dim, 512)
        self.dropout1 = nn.Dropout(0.3)
        self.linear2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(0.3)
        self.final_layer = nn.Linear(256, classification_labels)

    def forward(self, input_ids, attention_mask):
        x = self.encoder_model(input_ids=input_ids, attention_mask=attention_mask)
        x = x.last_hidden_state[:, 0, :]
        # first linear
        x = self.linear1(x)
        x = self.dropout1(x)
        x = F.gelu(x)
        # second linear
        x = self.linear2(x)
        x = self.dropout2(x)
        x = F.gelu(x)
        # final layer
        x = self.final_layer(x)
        return x

