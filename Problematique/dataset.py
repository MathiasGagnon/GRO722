import torch
import numpy as np
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import re
import pickle

class HandwrittenWords(Dataset):
    """Ensemble de donnees de mots ecrits a la main."""

    def __init__(self, filename):
        # Lecture du text
        self.pad_symbol     = pad_symbol = '<pad>'
        self.start_symbol   = start_symbol = '<sos>'
        self.stop_symbol    = stop_symbol = '<eos>'

        self.data = dict()
        with open(filename, 'rb') as fp:
            self.data = pickle.load(fp)

        for value in self.data:
            value[0] = list(value[0])

        # Dictionnaires de symboles vers entiers (Tokenization)
        self.symb2int = {start_symbol: 0, stop_symbol: 1, pad_symbol: 2}
        cpt_symb_fr = 3

        for i in range(len(self.data)):
            value = self.data[i][0]
            for symb in value:
                if symb not in self.symb2int:
                    self.symb2int[symb] = cpt_symb_fr
                    cpt_symb_fr += 1

        # Dictionnaires d'entiers vers symboles
        self.int2symb = {v: k for k, v in self.symb2int.items()}


        # Ajout du padding pour les targets TODO: idk comment faire encore pour les mouvements
        self.max_len = dict()

        self.max_len = max(len(value) for value in self.data[0]) + 2

        for value in self.data:
            len_diff = self.max_len - len(value[0])
            value[0].insert(0, self.start_symbol)
            value[0].append(self.stop_symbol)
            if len_diff != 0:
                for i in range(len_diff): value[0].append(self.pad_symbol)

        self.dict_size = {'fr': len(self.int2symb)}
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        target_seq = self.data[idx][0]
        target_seq = [self.symb2int[i] for i in target_seq]
        return torch.tensor(self.data[idx][1]), torch.tensor(target_seq)

    def visualisation(self, idx):
        coord_seq, target_seq = [i.numpy() for i in self[idx]]
        target_seq = [self.int2symb[i] for i in target_seq]
        print('Input: ', ' ' + str(coord_seq))
        print('Cible: ', ' '.join(target_seq))
        return
        

if __name__ == "__main__":
    # Code de test pour aider à compléter le dataset
    a = HandwrittenWords('data_trainval.p')
    for i in range(10):
        a.visualisation(np.random.randint(0, len(a)))