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

        self.pad_coord     = [0,0]
        self.start_coord   = [1e10,1e10]
        self.stop_coord    = [1e10,1e10]

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
        self.dict_size = len(self.int2symb)

        # Ajout du padding pour les targets TODO: idk comment faire encore pour les mouvements
        self.max_len = dict()

        self.max_len['txt'] = max(len(value[0]) for value in self.data) + 2

        for value in self.data:
            len_diff = self.max_len['txt'] - len(value[0]) -2
            value[0].insert(0, self.start_symbol)
            value[0].append(self.stop_symbol)
            if len_diff != 0:
                for i in range(len_diff): value[0].append(self.pad_symbol)

            one_hot_encoded = []
            for symbol in value[0]:
                vec = np.zeros(self.dict_size)
                vec[self.symb2int[symbol]] = 1
                one_hot_encoded.append(vec)
            
            value[0] = np.array(one_hot_encoded)


        # Ajout du padding pour les coordonnées
        self.max_len['coords'] = 0
        for value in self.data:
            if len(value[1][1]) > self.max_len['coords']: self.max_len['coords'] = len(value[1][1])

        self.max_len['coords'] += 2

        for value in self.data:
            len_diff = self.max_len['coords'] - len(value[1][1])
            value[1] = np.insert(value[1],0, self.start_coord, axis=1)
            value[1] = np.insert(value[1], len(value[1][1]), self.stop_coord, axis=1)
            if len_diff != 0:
                for i in range(len_diff): value[1] = np.insert(value[1], len(value[1][1]),self.pad_coord, axis=1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx][1]), torch.tensor(self.data[idx][0])

    def visualisation(self, idx):
        coord_seq, target_seq = [i.numpy() for i in self[idx]]
        target_seq = [self.int2symb[i] for i in target_seq]
        # print('Input: ', ' ' + str(coord_seq))
        print('Cible: ', ' '.join(target_seq))
        return
        

if __name__ == "__main__":
    # Code de test pour aider à compléter le dataset
    a = HandwrittenWords('data_trainval.p')
    for i in range(10):
        a.visualisation(np.random.randint(0, len(a)))