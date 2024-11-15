# GRO722 problématique
# Auteur: Jean-Samuel Lauzon et  Jonathan Vincent
# Hivers 2021
import os
import torch
from torch import nn
import numpy as np
from torch.utils.data import DataLoader
from models import *
from dataset import *
from metrics import edit_distance, confusion_matrix
import torch.nn.functional as F
import matplotlib.pyplot as plt

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def log_progress(
    epoch, n_epochs, batch_idx, dataload, running_loss, dist, list_loss, list_dist, type
):

    avg_loss = running_loss / (batch_idx + 1)
    avg_dist = dist / len(dataload.dataset)

    print(
        f"{type} - Epoch: {epoch}/{n_epochs} Average Loss: {avg_loss} Average Edit Distance: {avg_dist}"
    )

    # Append average values
    list_loss.append(avg_loss)
    list_dist.append(avg_dist)


def calculate_edit_distance(output, cible, dataset, dist):
    output_list = output.detach().cpu().tolist()
    cible_list = cible.cpu().tolist()

    for out, target in zip(output_list, cible_list):
        out_word = [
            dataset.int2symb[np.argmax(char)]
            for char in out
            if np.argmax(char) not in [0]
        ]
        out_str = "".join(out_word)

        target_word = [dataset.int2symb[char] for char in target if char not in [0]]
        target_str = "".join(target_word)

        dist += edit_distance(out_str, target_str)

    return dist


if __name__ == "__main__":

    # ---------------- Paramètres et hyperparamètres ----------------#
    force_cpu = False  # Forcer a utiliser le cpu?
    training = True  # Entrainement?
    _test = True  # Test?
    learning_curves = True  # Affichage des courbes d'entrainement?
    gen_test_images = True  # Génération images test?
    seed = 1  # Pour répétabilité
    n_workers = (
        0  # Nombre de threads pour chargement des données (mettre à 0 sur Windows)
    )

    # À compléter
    batch_size = 200  # Taille des lots
    n_epochs = 800  # Nombre d'iteration sur l'ensemble de donnees
    lr = 0.015  # Taux d'apprentissage pour l'optimizateur

    n_hidden = 11  # Nombre de neurones caches par couche
    n_layers = 3  # Nombre de de couches

    train_val_split = 0.8  # Ratio des echantillions pour l'entrainement
    save_path = f"best_model_dropout_0.35_hidden_{n_hidden}_layers_{n_layers}.pt"
    load_path = "best_model_dropout_0.35_hidden_11_layers_3.pt"

    # ---------------- Fin Paramètres et hyperparamètres ----------------#

    # Initialisation des variables
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    # Choix du device
    device = torch.device("cuda" if torch.cuda.is_available() and not force_cpu else "cpu")

    # Instanciation de l'ensemble de données
    dataset = HandwrittenWords("data_trainval.p")

    # Séparation de l'ensemble de données (entraînement et validation)
    n_train_samp = int(len(dataset) * train_val_split)
    n_val_samp = len(dataset) - n_train_samp
    dataset_train, dataset_val = torch.utils.data.random_split(
        dataset, [n_train_samp, n_val_samp]
    )

    dataset_test = HandwrittenWords(
        "data_test.p", dataset.max_len, dataset.int2symb, dataset.symb2int
    )
    dataload_test = DataLoader(
        dataset_test, batch_size=batch_size, shuffle=True, num_workers=n_workers
    )

    # Instanciation des dataloaders
    dataload_train = DataLoader(
        dataset_train, batch_size=batch_size, shuffle=True, num_workers=n_workers
    )
    dataload_val = DataLoader(
        dataset_val, batch_size=batch_size, shuffle=False, num_workers=n_workers
    )

    # Instanciation du model
    model = trajectory2seq(
        hidden_dim=n_hidden,
        n_layers=n_layers,
        device=device,
        symb2int=dataset.symb2int,
        int2symb=dataset.int2symb,
        dict_size=dataset.dict_size,
        maxlen=dataset.max_len,
        batch_size=batch_size,
    )
    model = model.to(device)

    # model.load_state_dict(torch.load(load_path))

    if training:
        fig, (ax_loss, ax_dist) = plt.subplots(1, 2, figsize=(12, 5))

        # Initialisation des variables
        best_val_loss = np.inf  # pour sauvegarder le meilleur model

        # Fonction de coût et optimizateur
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        if learning_curves:
            train_dist = []  # Historique des distances
            train_loss = []  # Historique des coûts
            val_dist = []  # Historique des distances
            val_loss = []  # Historique des coûts

        for epoch in range(1, n_epochs + 1):
            # Entraînement
            running_loss_train = 0
            dist_train = 0
            for batch_idx, data in enumerate(dataload_train):
                # Formatage des données
                coord, cible, _ = data
                coord = coord.to(device).float()
                cible = cible.to(device)
                cible_onehot = F.one_hot(cible, 27)
                optimizer.zero_grad()  # Mise a zero du gradient
                output, hidden, attn = model(coord, cible_onehot)  # Passage avant
                loss = criterion(
                    output.contiguous().view((-1, model.dict_size)), cible.view(-1)
                )

                loss.backward()  # calcul du gradient
                optimizer.step()  # Mise a jour des poids
                running_loss_train += loss.item()

                # calcul de la distance d'édition
                dist_train = calculate_edit_distance(output, cible, dataset, dist_train)

            log_progress(
                epoch,
                n_epochs,
                batch_idx,
                dataload_train,
                running_loss_train,
                dist_train,
                train_loss,
                train_dist,
                "Train",
            )

            # Valid
            running_loss_val = 0
            dist_val = 0
            with torch.no_grad():
                for batch_idx, data in enumerate(dataload_val):
                    coord, cible, _ = data
                    coord = coord.to(device).float()
                    cible = cible.to(device)

                    output, hidden, attn = model(coord, None)
                    loss = criterion(
                        output.contiguous().view((-1, model.dict_size)), cible.view(-1)
                    )

                    running_loss_val += loss.item()

                    # calcul de la distance d'édition
                    dist_val = calculate_edit_distance(output, cible, dataset, dist_val)

                log_progress(
                    epoch,
                    n_epochs,
                    batch_idx,
                    dataload_val,
                    running_loss_val,
                    dist_val,
                    val_loss,
                    val_dist,
                    "Val",
                )

            if learning_curves:
                ax_loss.cla()
                ax_loss.plot(train_loss, label="Training Loss", color="blue")
                ax_loss.plot(val_loss, label="Validation Loss", color="orange")
                ax_loss.set_title("Loss")
                ax_loss.set_xlabel("Epochs")
                ax_loss.set_ylabel("Loss")
                ax_loss.legend()

                ax_dist.cla()
                ax_dist.plot(train_dist, label="Training Distance", color="green")
                ax_dist.plot(val_dist, label="Validation Distance", color="red")
                ax_dist.set_title("Distance")
                ax_dist.set_xlabel("Epochs")
                ax_dist.set_ylabel("Distance")
                ax_dist.legend()

                plt.draw()
                plt.pause(0.01)

            # Enregistrer les poids
            if val_loss[-1] < best_val_loss:
                best_val_loss = val_loss[-1]
                torch.save(model.state_dict(), save_path)
                print(f"Best model saved with validation loss: {best_val_loss}")

            # Terminer l'affichage d'entraînement
    if learning_curves:
        all_true = []
        all_pred = []

        model.load_state_dict(torch.load(save_path, map_location=torch.device('cpu')))

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
        model.load_state_dict(torch.load(save_path, map_location=torch.device('cpu')))

        # Chargement des poids
        criterion = nn.CrossEntropyLoss()
        running_loss_test = 0
        dist_test = 0

        with torch.no_grad():
            for batch_idx, data in enumerate(dataload_test):
                coord, cible, _ = data
                coord = coord.to(device).float()
                cible = cible.to(device)
                cible_onehot = F.one_hot(cible, 27)
                # Forward pass
                output, hidden, attn = model(coord, cible_onehot)

                loss = criterion(
                    output.contiguous().view((-1, model.dict_size)), cible.view(-1)
                )
                running_loss_test += loss.item()

                # calcul de la distance d'édition
                dist_test = calculate_edit_distance(
                    output, cible, dataset_test, dist_test
                )

            print(
                f"Test - Average Loss: {running_loss_test / (batch_idx + 1)} Average Edit Distance: {dist_test / len(dataload_test.dataset)} \n"
            )

        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        
        for i in range(3):
            coord_seq, target_seq, original_coords = dataset_test[np.random.randint(0, len(dataset_test))]
            coord_seq = coord_seq[None, :].to(device).float()

            output, _, _ = model(coord_seq, None)
            out = torch.argmax(output, dim=2).detach().cpu()[0, :].tolist()

            in_seq = original_coords.squeeze(0).detach().cpu().numpy()
            target = [model.int2symb[i] for i in target_seq.detach().cpu().tolist()]
            out_seq = [model.int2symb[i] for i in out]

            x_coords, y_coords = in_seq[0], in_seq[1]

            axs[i].scatter(x_coords, y_coords, cmap="Blues", s=50, edgecolor="black")
            axs[i].set_title(f"(Cible: {''.join(target)}, Sortie: {''.join(out_seq)})")
            axs[i].set_xlabel("X")
            axs[i].set_ylabel("Y")
        plt.show()

        for i in range(10):
            coord_seq, target_seq, original_coords = dataset_test[
                np.random.randint(0, len(dataset_test))
            ]
            coord_seq = coord_seq[None, :].to(device).float()

            output, hidden, attn = model(coord_seq, None)
            out = torch.argmax(output, dim=2).detach().cpu()[0, :].tolist()

            in_seq = original_coords.squeeze(0).detach().cpu().numpy()
            target = [model.int2symb[i] for i in target_seq.detach().cpu().tolist()]
            out_seq = [model.int2symb[i] for i in out]

            print("Target: ", " ".join(target))
            print("Output: ", " ".join(out_seq))
            print("")

            attn = attn.detach().cpu()[0, :, :]

            x_coords, y_coords = in_seq[0], in_seq[1]
            num_letters = len(out_seq)

            attn_normalized = attn / attn.max()

            for idx, letter in enumerate(out_seq):
                colors = attn_normalized[:, idx].numpy()

                plt.figure(figsize=(6, 6))

                scatter = plt.scatter(
                    x_coords, y_coords, c=colors, cmap="Blues", s=50, edgecolor="black"
                )

                plt.title(f"Lettre: '{letter}', Cible:'{target}', Prediction:'{out_seq}'")
                plt.colorbar(scatter, label="Attention Intensity")
                plt.show()
