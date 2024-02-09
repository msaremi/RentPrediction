import torch
from torch import nn
from transformers import BertModel


class MyModel(nn.Module):
    __bert_model_name = "bert-base-german-cased"

    def __init__(self, use_text):
        super().__init__()
        self.use_text = use_text
        generator = torch.Generator().manual_seed(120)
        self.gammatrix = nn.Parameter(torch.randn((2, 20), generator=generator) * 2)
        self.gammatrix.requires_grad = False

        if use_text:
            self.bert = BertModel.from_pretrained(MyModel.__bert_model_name)
            self.linear_bert = nn.Linear(self.bert.config.hidden_size, 100)
            self.activation_bert = nn.LeakyReLU()
            text_len = 100
        else:
            self.text_tensor = nn.Parameter(torch.Tensor())
            self.text_tensor.requires_grad = False
            text_len = 0

        self.addr_model = nn.ModuleList([
            nn.Linear(40, 120),
            nn.LeakyReLU(),
            nn.Linear(120, 120),
            nn.LeakyReLU(),
            nn.Linear(120, 120),
            nn.LeakyReLU(),
            nn.Linear(120, 100),
            nn.LeakyReLU(),
        ])

        self.feat_model = nn.ModuleList([
            nn.Linear(86 + 100 + text_len, 256),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 100),
            nn.LeakyReLU(),
            nn.Linear(100, 1),
            nn.LeakyReLU()
        ])

    def __gamma(self, x):
        a = x @ self.gammatrix
        cos = torch.cos(2 * torch.pi * a) + 1
        sin = torch.sin(2 * torch.pi * a) + 1
        return torch.hstack((cos, sin)).float()

    def forward(self, x):
        if self.use_text:
            ids = torch.cat([x.description.ids[:, :128], x.facilities.ids[:, :128]], -1)
            mask = torch.cat([x.description.mask[:, :128], x.facilities.mask[:, :128]], -1)
            text = self.bert(input_ids=ids, attention_mask=mask).pooler_output
            text = self.linear_bert(text)
            text = self.activation_bert(text)
        else:
            text = self.text_tensor

        addr = self.__gamma(x.address)

        for layer in self.addr_model:
            addr = layer(addr)

        out = torch.cat([
            x.miscellaneous, x.firing_types, x.heating_type, x.condition,
            x.interior_qual, x.type_of_flat, addr, text
        ], -1)

        for layer in self.feat_model:
            out = layer(out)

        return out
