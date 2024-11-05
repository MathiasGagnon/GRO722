# GRO722 problématique
# Auteur: Jean-Samuel Lauzon et  Jonathan Vincent
# Hivers 2021

import torch
from torch import nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from models import *
from dataset import *
from metrics import confusion_matrix, edit_distance

if __name__ == '__main__':

    # ---------------- Paramètres et hyperparamètres ----------------#
    force_cpu = False           # Forcer a utiliser le cpu?
    trainning = True           # Entrainement?
    test = False                # Test?
    learning_curves = True     # Affichage des courbes d'entrainement?
    gen_test_images = True     # Génération images test?
    seed = 1                # Pour répétabilité
    n_workers = 0           # Nombre de threads pour chargement des données (mettre à 0 sur Windows)

    # À compléter
    batch_size = 10             # Taille des lots
    n_epochs = 50               # Nombre d'iteration sur l'ensemble de donnees
    lr = 0.01                   # Taux d'apprentissage pour l'optimizateur

    n_hidden = 25               # Nombre de neurones caches par couche
    n_layers = 1                # Nombre de de couches

    train_val_split = .7        # Ratio des echantillions pour l'entrainement

    # ---------------- Fin Paramètres et hyperparamètres ----------------#

    # Initialisation des variables
    if seed is not None:
        torch.manual_seed(seed) 
        np.random.seed(seed)

    # Choix du device
    device = torch.device("cuda" if torch.cuda.is_available() and not force_cpu else "cpu")

    # Instanciation de l'ensemble de données
    dataset = HandwrittenWords('Problematique/data_trainval.p')
    
    # Séparation de l'ensemble de données (entraînement et validation)
    n_train_samp = int(len(dataset) * train_val_split)
    n_val_samp = len(dataset) - n_train_samp
    dataset_train, dataset_val = torch.utils.data.random_split(dataset, [n_train_samp, n_val_samp])
   

    # Instanciation des dataloaders
    dataload_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=n_workers)
    dataload_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=n_workers)


    # Instanciation du model
    model = trajectory2seq(hidden_dim=n_hidden,
        n_layers=n_layers, device=device, symb2int=dataset.symb2int,
        int2symb=dataset.int2symb, dict_size=dataset.dict_size, maxlen=dataset.max_len,batch_size = batch_size)
    model = model.to(device)


    # Initialisation des variables
    best_val_loss = np.inf # pour sauvegarder le meilleur model

    if trainning:

        # Fonction de coût et optimizateur
        criterion = nn.CrossEntropyLoss(ignore_index=2)  # ignorer les symboles <pad>
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        if learning_curves:
            train_dist =[] # Historique des distances
            train_loss=[] # Historique des coûts
            fig, ax = plt.subplots(1) # Initialisation figure

        for epoch in range(1, n_epochs + 1):
            # Entraînement
            running_loss_train = 0
            dist = 0
            for batch_idx, data in enumerate(dataload_train):
                # Formatage des données
                coord, cible = data
                coord = coord.to(device).float()
                cible = cible.to(device)

                optimizer.zero_grad()  # Mise a zero du gradient
                output, hidden, attn = model(coord)  # Passage avant
                loss = criterion(output.float(), cible)

                loss.backward()  # calcul du gradient
                optimizer.step()  # Mise a jour des poids
                running_loss_train += loss.item()

                # calcul de la distance d'édition
                output_list = output.detach().cpu().tolist()
                cible_list = cible.cpu().tolist()
                dist = []
                for out, cib in zip(output_list, cible_list):
                    dist.append(edit_distance(out, cib))
                # Affichage pendant l'entraînement
                print(
                    'Train - Epoch: {}/{} [{}/{} ({:.0f}%)] Average Loss: {:.6f} Average Edit Distance: {:.6f}'.format(
                        epoch, n_epochs, batch_idx * batch_size, len(dataload_train.dataset),
                                         100. * batch_idx * batch_size / len(dataload_train.dataset),
                                         running_loss_train / (batch_idx + 1),
                                         dist / len(dataload_train)), end='\r')

            print('Train - Epoch: {}/{} [{}/{} ({:.0f}%)] Average Loss: {:.6f} Average Edit Distance: {:.6f}'.format(
                epoch, n_epochs, (batch_idx + 1) * batch_size, len(dataload_train.dataset),
                                 100. * (batch_idx + 1) * batch_size / len(dataload_train.dataset),
                                 running_loss_train / (batch_idx + 1),
                                 dist / len(dataload_train)), end='\r')
            print('\n')
            # Affichage graphique
            if learning_curves:
                train_loss.append(running_loss_train / len(dataload_train))
                train_dist.append(dist / len(dataload_train))
                ax.cla()
                ax.plot(train_loss, label='training loss')
                ax.plot(train_dist, label='training distance')
                ax.legend()
                plt.draw()
                plt.pause(0.01)

            # Enregistrer les poids
            torch.save(model, 'model.pt')

            # Terminer l'affichage d'entraînement
        if learning_curves:
            plt.show()
            plt.close('all')

    if test:
        # Évaluation
        # À compléter

        # Charger les données de tests
        # À compléter

        # Affichage de l'attention
        # À compléter (si nécessaire)

        # Affichage des résultats de test
        # À compléter
        
        # Affichage de la matrice de confusion
        # À compléter

        pass