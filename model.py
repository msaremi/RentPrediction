import torch
from torch import nn
from transformers import BertModel, BertConfig


class MyModel(nn.Module):
    __bert_model_name = "bert-base-german-cased"

    def __init__(self):
        super().__init__()
        generator = torch.Generator().manual_seed(120)
        self.gammatrix = nn.Parameter(torch.randn((2, 20), generator=generator) * 2)
        self.gammatrix.requires_grad = False

        config = BertConfig(
            vocab_size=30522,
            hidden_size=100,
            num_hidden_layers=2,
            num_attention_heads=10,
            intermediate_size=300,
            hidden_act='gelu',
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            max_position_embeddings=128,
            type_vocab_size=2,
            initializer_range=0.02,
            layer_norm_eps=1e-12,
            pad_token_id=0,
            position_embedding_type='absolute',
            use_cache=True,
            classifier_dropout=None
        )
        self.bert = BertModel(config)

        self.linear_addr1 = nn.Linear(40, 80)
        self.activation_addr1 = nn.LeakyReLU()
        self.linear_addr2 = nn.Linear(80, 80)
        self.activation_addr2 = nn.LeakyReLU()
        self.linear_addr3 = nn.Linear(80, 80)
        self.activation_addr3 = nn.Sigmoid()

        self.linear1 = nn.Linear(86 + 80 + 100, 100)
        self.activation1 = nn.LeakyReLU()
        self.dropout1 = nn.Dropout(0.1)
        self.linear2 = nn.Linear(100, 100)
        self.activation2 = nn.ELU()
        self.linear3 = nn.Linear(100, 1)
        self.activation3 = nn.ReLU()

    def __gamma(self, x):
        a = x @ self.gammatrix
        cos = torch.cos(2 * torch.pi * a) + 1
        sin = torch.sin(2 * torch.pi * a) + 1
        return torch.hstack((cos, sin)).float()

    def forward(self, x):
        ids = torch.cat([x.description.ids[:, :64], x.facilities.ids[:, :64]], -1)
        mask = torch.cat([x.description.mask[:, :64], x.facilities.mask[:, :64]], -1)
        text = self.bert(input_ids=ids, attention_mask=mask).pooler_output

        addr = self.__gamma(x.address)
        addr = self.linear_addr1(addr)
        addr = self.activation_addr1(addr)
        addr = self.linear_addr2(addr)
        addr = self.activation_addr2(addr)
        addr = self.linear_addr3(addr)
        addr = self.activation_addr3(addr)

        out = torch.cat([
            x.miscellaneous, x.firing_types, x.heating_type, x.condition, x.interior_qual, x.type_of_flat, addr, text
        ], -1)
        out = self.linear1(out)
        out = self.activation1(out)
        out = self.dropout1(out)
        out = self.linear2(out)
        out = self.activation2(out)
        out = self.linear3(out)
        out = self.activation3(out)
        return out
