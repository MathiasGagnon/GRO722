import torch
import numpy as np
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import re
import pickle
import copy


class HandwrittenWords(Dataset):
    """Ensemble de donnees de mots ecrits a la main."""

    def __init__(
        self,
        filename,
        previous_max_len=None,
        previous_int2symb=None,
        previous_symb2int=None,
        show=False,
    ):
        # Lecture du text
        self.pad_symbol = pad_symbol = "#"

        self.data = dict()
        with open(filename, "rb") as fp:
            self.data = pickle.load(fp)

        for value in self.data:
            value[0] = list(value[0])

        # Dictionnaires de symboles vers entiers (Tokenization)
        if previous_symb2int is None:
            self.symb2int = {pad_symbol: 0}
            cpt_symb_fr = 1

            for i in range(len(self.data)):
                value = self.data[i][0]
                for symb in value:
                    if symb not in self.symb2int:
                        self.symb2int[symb] = cpt_symb_fr
                        cpt_symb_fr += 1
        else:
            self.symb2int = previous_symb2int

        if previous_int2symb is None:
            self.int2symb = {v: k for k, v in self.symb2int.items()}
        else:
            self.int2symb = previous_int2symb

        self.original_coords = copy.deepcopy(self.data)

        for value in self.data:
            x_coords, y_coords = value[1][0], value[1][1]

            diff_x = [x_coords[i] - x_coords[i - 1] for i in range(1, len(x_coords))]
            diff_y = [y_coords[i] - y_coords[i - 1] for i in range(1, len(y_coords))]

            diff_x.insert(0, x_coords[0])
            diff_y.insert(0, y_coords[0])

            value[1][0] = diff_x
            value[1][1] = diff_y

        max_x, min_x = float("-inf"), float("inf")
        max_y, min_y = float("-inf"), float("inf")

        all_x, all_y = [], []

        for value in self.data:
            all_x.extend(value[1][0])
            all_y.extend(value[1][1])

        max_x, min_x = max(all_x), min(all_x)
        max_y, min_y = max(all_y), min(all_y)

        mean_x, mean_y = np.mean(all_x), np.mean(all_y)

        # for value in self.data:
        #     x_coords, y_coords = value[1][0], value[1][1]
        #     normalized_mean_x = [(x - mean_x) / (max_x - min_x) for x in x_coords]
        #     normalized_mean_y = [(y - mean_y) / (max_y - min_y) for y in y_coords]
        #     value[1][0] = normalized_mean_x
        #     value[1][1] = normalized_mean_y

        self.max_len = {}
        if previous_max_len is None:
            self.max_len["txt"] = max(len(value[0]) for value in self.data)
            self.max_len["coords"] = 0
            for value in self.data:
                if len(value[1][1]) > self.max_len["coords"]:
                    self.max_len["coords"] = len(value[1][1])
        else:
            self.max_len = previous_max_len

        for value in self.data:
            len_diff = self.max_len["txt"] - len(value[0])
            if len_diff != 0:
                for i in range(len_diff):
                    value[0].append(self.pad_symbol)

        for value in self.data:
            len_lol = len(value[1][1])
            last_val = value[1][:, len_lol - 1]
            len_diff = self.max_len["coords"] - len(value[1][1])
            if len_diff != 0:
                for i in range(len_diff):
                    value[1] = np.insert(value[1], len(value[1][1]), 0, axis=1)

        for value in self.original_coords:
            len_lol = len(value[1][1])
            last_val = value[1][:, len_lol - 1]
            len_diff = self.max_len["coords"] - len(value[1][1])
            if len_diff != 0:
                for i in range(len_diff):
                    value[1] = np.insert(value[1], len(value[1][1]), 0, axis=1)

        # Extract the first set of coordinates
        first_coords = self.data[20][
            1
        ]  # Assumes self.data[0][1] contains [x_coords, y_coords]
        x_coords = first_coords[0]
        y_coords = first_coords[1]

        # Plot the coordinates
        if show:
            plt.figure(figsize=(8, 6))
            plt.scatter(x_coords, y_coords, color="blue", marker="o")
            plt.title(self.data[20][0])
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.grid(True)

            # Annotate each point with its index
            for i, (x, y) in enumerate(zip(x_coords, y_coords)):
                plt.text(
                    x, y, f"{i}", fontsize=9, ha="right", color="red"
                )  # Adjust fontsize and position as needed

            plt.show()

        self.dict_size = len(self.int2symb)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        target_seq = self.data[idx][0]
        target_seq = [self.symb2int[i] for i in target_seq]
        return (
            torch.tensor(self.data[idx][1]),
            torch.tensor(target_seq),
            torch.tensor(self.original_coords[idx][1]),
        )

    def visualisation(self, idx):
        coord_seq, target_seq, _ = [i.numpy() for i in self[idx]]
        target_seq = [self.int2symb[i] for i in target_seq]
        print("Input: ", " " + str(coord_seq))
        print("Cible: ", " ".join(target_seq))
        return


if __name__ == "__main__":
    # Code de test pour aider à compléter le dataset
    a = HandwrittenWords("data_trainval.p")
    for i in range(10):
        a.visualisation(np.random.randint(0, len(a)))
