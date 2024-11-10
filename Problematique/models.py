# GRO722 problématique
# Auteur: Jean-Samuel Lauzon et  Jonathan Vincent
# Hivers 2021

import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

class trajectory2seq(nn.Module):
    def __init__(self, hidden_dim, n_layers, int2symb, symb2int, dict_size, device, maxlen, batch_size):
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

        # Définition des couches du rnn
        self.text_embedding = nn.Embedding(self.dict_size, hidden_dim)
        #self.coords_embedding = nn.Linear(2)
        self.encoder_layer = nn.GRU(input_size=3, hidden_size=hidden_dim, num_layers=n_layers, batch_first=True, bidirectional=False)
        self.decoder_layer = nn.GRU(hidden_dim, hidden_dim, n_layers, batch_first=True)

        # Définition de la couche dense pour la sortie
        self.attention_fc = nn.Sequential(
            nn.Linear(2*hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, self.dict_size),
        )

        # Couche attention
        self.similarity = nn.CosineSimilarity(dim=2)
        self.softmax = nn.Softmax(dim=1)

        self.to(device)

    def encoder(self, x):
        # Encodeur
        permutted = x.permute(0, 2, 1)

        positions = torch.arange(0, 457).unsqueeze(0).unsqueeze(2).expand(50,457,1)

        permutted_with_position = torch.cat((permutted, positions), dim=-1)

        out, hidden = self.encoder_layer(permutted_with_position)
        return out, hidden


    def decoder(self, encoder_outs, hidden):
        batch_size = hidden.shape[1] # Taille de la batch
        vec_in = torch.zeros((batch_size, 1)).to(self.device).long()  # Vecteur d'entrée pour décodage
        vec_out = torch.zeros((batch_size, self.maxlen['txt'], 27)).to(self.device).float()  # Vecteur de sortie du décodage
        attention_w = torch.zeros((batch_size, self.maxlen['coords'], self.maxlen['txt'])).to(self.device)
        
        for i in range(self.maxlen['txt']):
            embedded = self.text_embedding(vec_in)
            output, hidden = self.decoder_layer(embedded, hidden)

            a, w = self.attention(encoder_outs, output)
            attention_w[:, :, i] = w.squeeze(2)

            combined = torch.cat((output, a.permute(0,2,1)), dim=-1)
            output = self.attention_fc(combined)
            output = self.fc(output)
            argmax_output = output.argmax(dim=-1)
            vec_out[:, i,:] = output.squeeze(1)
            vec_in = argmax_output

        return vec_out, hidden, attention_w


    def forward(self, x):
        # Passant avant
        out, h = self.encoder(x)
        out, hidden, attn = self.decoder(out,h)
        return out, hidden, attn


    def attention(self, v, q):
        q = q.permute(0,2,1)
        attn_score = torch.bmm(v, q)
        w = self.softmax(attn_score)
        v = v.permute(0,2,1)
        a = torch.bmm(v, w)
        return a, w