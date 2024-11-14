# GRO722 problématique
# Auteur: Jean-Samuel Lauzon et  Jonathan Vincent
# Hivers 2021

import torch
from torch import nn


class trajectory2seq(nn.Module):
    def __init__(
        self,
        hidden_dim,
        n_layers,
        int2symb,
        symb2int,
        dict_size,
        device,
        maxlen,
        batch_size,
    ):
        super(trajectory2seq, self).__init__()
        # Definition des parametres
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.device = device
        self.symb2int = symb2int
        self.int2symb = int2symb
        self.dict_size = dict_size
        self.maxlen = maxlen
        self.batch_size = batch_size
        self.dropout_rate = 0.35
        self.forced_th = 0.5

        # Définition des couches du rnn
        self.text_embedding = nn.Embedding(self.dict_size, hidden_dim)
        self.encoder_layer = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=self.dropout_rate,
        )
        self.decoder_layer = nn.GRU(
            hidden_dim,
            hidden_dim,
            n_layers,
            batch_first=True,
            dropout=self.dropout_rate,
        )

        #Fully connected layers
        self.encoder_fc = nn.Linear(2, hidden_dim)
        self.attention_fc = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
        )
        self.decoder_fc = nn.Sequential(
            nn.Linear(hidden_dim, self.dict_size),
        )

        self.bi_fc = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim))

        # Couche attention
        self.similarity = nn.CosineSimilarity(dim=2)
        self.softmax = nn.Softmax(dim=1)

        self.to(device)

        self.print_num_params()

    def print_num_params(self):
        total_params = sum(p.numel() for p in self.parameters())
        print(f"params: {total_params}")

    def encoder(self, x):
        x_permuted = x.permute(0, 2, 1)

        x_resized = self.encoder_fc(x_permuted)

        out, hidden = self.encoder_layer(x_resized)
        out = self.bi_fc(out)
        return out, hidden

    def decoder(self, encoder_outs, hidden, target):
        batch_size = hidden.shape[1]
        vec_in = (
            torch.zeros((batch_size, 1)).to(self.device).long()
        )
        vec_out = (
            torch.zeros((batch_size, self.maxlen["txt"], 27)).to(self.device).float()
        )
        attention_w = torch.zeros(
            (batch_size, self.maxlen["coords"], self.maxlen["txt"])
        ).to(self.device)
        hidden = hidden[: -self.n_layers, :, :]

        for i in range(self.maxlen["txt"]):
            forced_teaching = torch.rand(1)
            embedded = self.text_embedding(vec_in)

            output, hidden = self.decoder_layer(embedded, hidden)
            a, w = self.attention(encoder_outs, output)
            attention_w[:, :, i] = w

            combined = torch.cat((output, a), dim=-1)
            output = self.attention_fc(combined)
            output = self.decoder_fc(output)

            argmax_output = output.argmax(dim=-1)
            vec_out[:, i, :] = output.squeeze(1)
            vec_in = argmax_output

            #Teacher forcing
            if forced_teaching > self.forced_th and target is not None:
                target_argmax = target[:, i, :].argmax(dim=1).unsqueeze(1)
                vec_in = target_argmax

        return vec_out, hidden, attention_w

    def forward(self, x, target):
        out, h = self.encoder(x)
        out, hidden, attn = self.decoder(out, h, target)
        return out, hidden, attn

    def attention(self, v, q):
        attn_score = self.similarity(v, q)
        w = self.softmax(attn_score)
        a = torch.bmm(w.unsqueeze(1), v)
        return a, w
