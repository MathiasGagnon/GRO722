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
        self.coord_embedding = nn.Embedding(self.dict_size, hidden_dim)
        self.text_embedding = nn.Embedding(self.dict_size, hidden_dim)
        self.encoder_layer = nn.GRU(input_size=461, hidden_size=hidden_dim, num_layers=n_layers, batch_first=True)
        self.decoder_layer = nn.GRU(hidden_dim, hidden_dim, n_layers, batch_first=True)

        # Couche pour l'attention
        # À compléter

        # Définition de la couche dense pour la sortie
        self.fc = nn.Sequential(
            nn.Linear(2*hidden_dim, self.dict_size),
        )

        self.similarity = nn.CosineSimilarity(dim=-1)
        self.softmax = nn.Softmax(dim=1)

        self.to(device)

    def encoder(self, x):
        # Encodeur
        out, hidden = self.encoder_layer(x)
        return out, hidden


    def decoder(self, encoder_outs, hidden):
        batch_size = hidden.shape[1] # Taille de la batch
        vec_in = torch.zeros((batch_size, 1)).to(self.device).long()  # Vecteur d'entrée pour décodage
        vec_out = torch.zeros((batch_size, self.maxlen['txt'], 29)).to(self.device).float()  # Vecteur de sortie du décodage

        sos_token = self.symb2int['<sos>']
        vec_in[:, 0] = sos_token
        vec_out[:, 0, 0] = 1
        
        for i in range(self.maxlen['txt']-1):
            embedded = self.text_embedding(vec_in)  # Utiliser le bon embedding pour le texte
            output, hidden = self.decoder_layer(embedded, hidden)

            a = self.attention(encoder_outs, output)

            combined = torch.cat((output, a), dim=-1)

            output = self.fc(combined)  # Appliquer la couche linéaire
            argmax_output = output.argmax(dim=-1)
            vec_out[:, i+1,:] = output.squeeze(1)
            vec_in = argmax_output

        return vec_out, hidden, None


    def forward(self, x):
        # Passant avant
        out, h = self.encoder(x)
        out, hidden, attn = self.decoder(out,h)
        return out, hidden, attn


    def attention(self, v, q):
        attn_score = self.similarity(v, q)
        w = self.softmax(attn_score)
        a = torch.bmm(w.unsqueeze(1), v)
        return a