# GRO722 problématique
# Auteur: Jean-Samuel Lauzon et  Jonathan Vincent
# Hivers 2021
import os
import torch
from torch import nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from models import *
from dataset import *
from metrics import edit_distance, confusion_matrix
import torch.nn.functional as F
import matplotlib.pyplot as plt

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

if __name__ == '__main__':

    # ---------------- Paramètres et hyperparamètres ----------------#
    force_cpu = False           # Forcer a utiliser le cpu?
    training = False           # Entrainement?
    _test = True                # Test?
    learning_curves = True     # Affichage des courbes d'entrainement?
    gen_test_images = True     # Génération images test?
    seed = 1                # Pour répétabilité
    n_workers = 0           # Nombre de threads pour chargement des données (mettre à 0 sur Windows)

    # À compléter
    batch_size = 50           # Taille des lots
    n_epochs =  1           # Nombre d'iteration sur l'ensemble de donnees
    lr = 0.01                 # Taux d'apprentissage pour l'optimizateur

    n_hidden = 19               # Nombre de neurones caches par couche
    n_layers = 1                # Nombre de de couches

    train_val_split = .8        # Ratio des echantillions pour l'entrainement

    # ---------------- Fin Paramètres et hyperparamètres ----------------#

    # Initialisation des variables
    if seed is not None:
        torch.manual_seed(seed) 
        np.random.seed(seed)

    # Choix du device
    device = torch.device("cuda" if torch.cuda.is_available() and not force_cpu else "cpu")

    # Instanciation de l'ensemble de données
    dataset = HandwrittenWords('data_trainval.p')
    
    # Séparation de l'ensemble de données (entraînement et validation)
    n_train_samp = int(len(dataset) * train_val_split)
    n_val_samp = len(dataset) - n_train_samp
    dataset_train, dataset_val = torch.utils.data.random_split(dataset, [n_train_samp, n_val_samp])
   

    # Instanciation des dataloaders
    dataload_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=n_workers)
    dataload_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=n_workers)

    fig, (ax_loss, ax_dist) = plt.subplots(1, 2, figsize=(12, 5))


    # Instanciation du model
    model = trajectory2seq(hidden_dim=n_hidden,
        n_layers=n_layers, device=device, symb2int=dataset.symb2int,
        int2symb=dataset.int2symb, dict_size=dataset.dict_size, maxlen=dataset.max_len,batch_size = batch_size)
    model = model.to(device)


    # Initialisation des variables
    best_val_loss = np.inf # pour sauvegarder le meilleur model

    if training:

        # Fonction de coût et optimizateur
        criterion = nn.CrossEntropyLoss()  # ignorer les symboles <pad>
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        if learning_curves:
            train_dist =[] # Historique des distances
            train_loss=[] # Historique des coûts
            val_dist =[] # Historique des distances
            val_loss=[] # Historique des coûts

        for epoch in range(1, n_epochs + 1):
            # Entraînement
            running_loss_train = 0
            dist = 0
            for batch_idx, data in enumerate(dataload_train):
                # Formatage des données
                coord, cible, original_coords = data
                coord = coord.to(device).float()
                cible = cible.to(device)
                cible_onehot = F.one_hot(cible, 27)
                optimizer.zero_grad()  # Mise a zero du gradient
                output, hidden, attn = model(coord, cible_onehot)  # Passage avant
                test = output.view((-1, model.dict_size))
                loss = criterion(output.contiguous().view((-1, model.dict_size)), cible.view(-1))

                loss.backward()  # calcul du gradient
                optimizer.step()  # Mise a jour des poids
                running_loss_train += loss.item()

                 # calcul de la distance d'édition
                output_list = output.detach().cpu().tolist()
                cible_list = cible.cpu().tolist()
                for out, cible in zip(output_list, cible_list):
                    out_word = [dataset.int2symb[np.argmax(char)] for char in out if np.argmax(char) not in [0]]
                    out_str = ''.join(out_word)

                    cible_word = []
                    cible_word = [dataset.int2symb[char] for char in cible if char not in [0]]
                    cible_str = ''.join(cible_word)

                    dist += edit_distance(out_str, cible_str)
            print('Train - Epoch: {}/{} [{}/{} ({:.0f}%)] Average Loss: {:.6f} Average Edit Distance: {:.6f}'.format(
                    epoch, n_epochs, (batch_idx + 1) * batch_size, len(dataload_train.dataset),
                                         100. * (batch_idx + 1) * batch_size / len(dataload_train.dataset),
                                         running_loss_train / (batch_idx + 1),
                                         dist / len(dataload_train.dataset)))
            train_loss.append(running_loss_train / (len(dataload_train.dataset)/(batch_size)))
            train_dist.append(dist / len(dataload_train.dataset))

            # Valid
            running_loss_val = 0
            dist_val = 0
            for batch_idx, data in enumerate(dataload_val):
                coord, cible = data
                coord = coord.to(device).float()
                cible = cible.to(device)
                torch.no_grad()
                output, hidden, attn = model(coord, None)  # Passage avant
                test = output.view((-1, model.dict_size))
                loss = criterion(output.contiguous().view((-1, model.dict_size)), cible.view(-1))

                running_loss_val += loss.item()

                # calcul de la distance d'édition
                output_list = output.detach().cpu().tolist()
                cible_list = cible.cpu().tolist()
                for out, cible in zip(output_list, cible_list):
                    out_word = [dataset.int2symb[np.argmax(char)] for char in out if np.argmax(char) not in [0]]
                    out_str = ''.join(out_word)

                    cible_word = []
                    cible_word = [dataset.int2symb[char] for char in cible if char not in [0]]
                    cible_str = ''.join(cible_word)

                    dist_val += edit_distance(out_str, cible_str)

            print('Val - Epoch: {}/{} [{}/{} ({:.0f}%)] Average Loss: {:.6f} Average Edit Distance: {:.6f}'.format(
                epoch, n_epochs, (batch_idx + 1) * batch_size, len(dataload_val.dataset),
                                 100. * (batch_idx + 1) * batch_size / len(dataload_val.dataset),
                                 running_loss_val / (batch_idx + 1),
                                 dist_val / len(dataload_val.dataset)), end='\n')
            print('\n')
            val_loss.append(running_loss_val / (len(dataload_val.dataset)/(batch_size)))
            val_dist.append(dist_val / len(dataload_val.dataset))

            if learning_curves:
                ax_loss.cla()
                ax_loss.plot(train_loss, label='Training Loss', color='blue')
                ax_loss.plot(val_loss, label='Validation Loss', color='orange')
                ax_loss.set_title("Loss")
                ax_loss.set_xlabel("Epochs")
                ax_loss.set_ylabel("Loss")
                ax_loss.legend()

                ax_dist.cla()
                ax_dist.plot(train_dist, label='Training Distance', color='green')
                ax_dist.plot(val_dist, label='Validation Distance', color='red')
                ax_dist.set_title("Distance")
                ax_dist.set_xlabel("Epochs")
                ax_dist.set_ylabel("Distance")
                ax_dist.legend()

                plt.draw()
                plt.pause(0.01)

            # Enregistrer les poids
            if val_loss[-1] < best_val_loss:
                best_val_loss = val_loss[-1]
                torch.save(model.state_dict(), 'best_model_1.pt')
                print(f"Best model saved with validation loss: {best_val_loss}")

            # Terminer l'affichage d'entraînement
        if learning_curves:
            all_true = []
            all_pred = []

            model.load_state_dict(torch.load('best_model.pt'))

            with torch.no_grad():
                for batch_idx, data in enumerate(dataload_val):
                    coord, cible = data
                    coord = coord.to(device).float()
                    cible = cible.to(device)
                    
                    # Forward pass
                    output, hidden, attn = model(coord, None)
                    
                    # Get predictions
                    preds = output.argmax(dim=-1)
                    
                    # Collect true and predicted labels
                    all_true.extend(cible.cpu().numpy())
                    all_pred.extend(preds.cpu().numpy())

            # Convert collected labels to numpy arrays
            all_true = np.array(all_true)
            all_pred = np.array(all_pred)

            # Generate and plot the confusion matrix
            confusion_matrix(all_true, all_pred, dataset.int2symb)

    if _test:
        # Évaluation

        # Chargement des poids
        model.load_state_dict(torch.load('best_model.pt', map_location=torch.device('cpu')))
        dataset.symb2int = model.symb2int
        dataset.int2symb = model.int2symb

        for i in range(10):
            coord_seq, target_seq, original_coords = dataset[np.random.randint(0, len(dataset))]
            coord_seq = coord_seq[None, :].to(device).float()  # Shape [1, 2, *]

            output, hidden, attn = model(coord_seq, None)
            out = torch.argmax(output, dim=2).detach().cpu()[0, :].tolist()

            # Convert sequences to human-readable format
            in_seq = original_coords.squeeze(0).detach().cpu().numpy()  # Shape [2, *]
            target = [model.int2symb[i] for i in target_seq.detach().cpu().tolist()]
            out_seq = [model.int2symb[i] for i in out]

            print('Input:  ', in_seq)
            print('Target: ', ' '.join(target))
            print('Output: ', ' '.join(out_seq))
            print('')

            attn = attn.detach().cpu()[0, :, :]

            x_coords, y_coords = in_seq[0], in_seq[1]
            num_letters = len(out_seq)

            attn_normalized = attn / attn.max()

            for idx, letter in enumerate(out_seq):
                colors = attn_normalized[:, idx].numpy()

                plt.figure(figsize=(6, 6))

                scatter = plt.scatter(x_coords, y_coords, c=colors, cmap="Blues", s=50, edgecolor='black')

                # Labels and title
                plt.title(f"Lettre: '{letter}', Cible:'{out_seq}'")

                plt.colorbar(scatter, label='Attention Intensity')

                plt.show()
