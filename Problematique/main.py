# GRO722 problématique
# Auteur: Jean-Samuel Lauzon et  Jonathan Vincent
# Hivers 2021
import os
import torch
from torch import nn
import numpy as np
from torch.utils.data import DataLoader, ConcatDataset
from models import *
from dataset import *
from metrics import edit_distance, confusion_matrix
import torch.nn.functional as F
import matplotlib.pyplot as plt

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def log_progress(epoch, n_epochs, batch_idx, dataload, 
                          running_loss, dist, list_loss, list_dist, type):
    
    avg_loss = running_loss / (batch_idx + 1)
    avg_dist = dist / len(dataload.dataset)

    print(f'{type} - Epoch: {epoch}/{n_epochs} Average Loss: {avg_loss} Average Edit Distance: {avg_dist}')
    
    # Append average values
    list_loss.append(avg_loss)
    list_dist.append(avg_dist)

def calculate_edit_distance(output, cible, dataset, dist):
    output_list = output.detach().cpu().tolist()
    cible_list = cible.cpu().tolist()

    for out, target in zip(output_list, cible_list):
        out_word = [dataset.int2symb[np.argmax(char)] for char in out if np.argmax(char) not in [0]]
        out_str = ''.join(out_word)

        target_word = [dataset.int2symb[char] for char in target if char not in [0]]
        target_str = ''.join(target_word)

        dist += edit_distance(out_str, target_str)

    return dist

if __name__ == '__main__':

    # ---------------- Paramètres et hyperparamètres ----------------#
    force_cpu = False           # Forcer a utiliser le cpu?
    training = True           # Entrainement?
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
    dataset = HandwrittenWords('Problematique/data_trainval.p')
    dataset_test = HandwrittenWords('Problematique/data_test.p', dataset.max_len)

    dataset_test.int2symb = dataset.int2symb
    dataset_test.symb2int = dataset.symb2int     
    
    # Séparation de l'ensemble de données (entraînement et validation)
    n_train_samp = int(len(dataset) * train_val_split)
    n_val_samp = len(dataset) - n_train_samp
    dataset_train, dataset_val = torch.utils.data.random_split(dataset, [n_train_samp, n_val_samp])
   

    # Instanciation des dataloaders
    dataload_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=n_workers)
    dataload_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True, num_workers=n_workers)
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
            dist_train = 0
            dataset_list = [dataload_test, dataload_train]
            for sub_dataset in dataset_list:
                for batch_idx, data in enumerate(sub_dataset):
                    # Formatage des données
                    coord, cible, _ = data
                    coord = coord.to(device).float()
                    cible = cible.to(device)
                    cible_onehot = F.one_hot(cible, 27)
                    optimizer.zero_grad()  # Mise a zero du gradient
                    output, hidden, attn = model(coord, cible_onehot)  # Passage avant
                    loss = criterion(output.contiguous().view((-1, model.dict_size)), cible.view(-1))

                    loss.backward()  # calcul du gradient
                    optimizer.step()  # Mise a jour des poids
                    running_loss_train += loss.item()

                    # calcul de la distance d'édition
                    dist_train = calculate_edit_distance(output, cible, dataset, dist_train)

            log_progress(epoch, n_epochs, batch_idx, sub_dataset, running_loss_train, dist_train, train_loss, train_dist, 'Train')

            # Valid
            running_loss_val = 0
            dist_val = 0
            with torch.no_grad():
                for batch_idx, data in enumerate(dataload_val):
                    coord, cible, _ = data
                    coord = coord.to(device).float()
                    cible = cible.to(device)

                    output, hidden, attn = model(coord, None)
                    loss = criterion(output.contiguous().view((-1, model.dict_size)), cible.view(-1))

                    running_loss_val += loss.item()

                    # calcul de la distance d'édition
                    dist_val = calculate_edit_distance(output, cible, dataset, dist_val)

                log_progress(epoch, n_epochs, batch_idx, dataload_val, running_loss_val, dist_val, val_loss, val_dist, 'Val')

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
                torch.save(model.state_dict(), 'oracle.pt')
                print(f"Best model saved with validation loss: {best_val_loss}")

            # Terminer l'affichage d'entraînement
        if learning_curves:
            all_true = []
            all_pred = []

            model.load_state_dict(torch.load('oracle.pt'))

            with torch.no_grad():
                for batch_idx, data in enumerate(dataload_val):
                    coord, cible, _ = data
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
        dataset_test = HandwrittenWords('Problematique/data_test.p', dataset.max_len)
        dataload_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True, num_workers=n_workers)
        # Chargement des poids
        model.load_state_dict(torch.load('best_model_1.pt'))

        criterion = nn.CrossEntropyLoss()
        running_loss_test = 0
        dist_test = 0

        with torch.no_grad():
                for batch_idx, data in enumerate(dataload_test):
                    coord , cible, _ = data
                    coord = coord.to(device).float()
                    cible = cible.to(device)
                    cible_onehot = F.one_hot(cible, 27)
                    # Forward pass
                    output, hidden, attn = model(coord, cible_onehot)

                    loss = criterion(output.contiguous().view((-1, model.dict_size)), cible.view(-1))
                    running_loss_test += loss.item()

                    # calcul de la distance d'édition
                    dist_test = calculate_edit_distance(output, cible, dataset, dist_test)

                print(f'Test - Average Loss: {running_loss_test / (batch_idx + 1)} Average Edit Distance: {dist_test / len(dataload_test.dataset)} \n')
     

        for i in range(10):
            coord_seq, target_seq, original_coords = dataset_test[np.random.randint(0, len(dataset_test))]
            coord_seq = coord_seq[None, :].to(device).float()  # Shape [1, 2, *]

            output, hidden, attn = model(coord_seq, None)
            out = torch.argmax(output, dim=2).detach().cpu()[0, :].tolist()

            # Convert sequences to human-readable format
            in_seq = original_coords.squeeze(0).detach().cpu().numpy()  # Shape [2, *]
            target = [model.int2symb[i] for i in target_seq.detach().cpu().tolist()]
            out_seq = [model.int2symb[i] for i in out]

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
