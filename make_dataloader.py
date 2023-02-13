import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms

class sudoku_dataset(Dataset):
    def __init__(self, path, train=False, transform=None, type=4):
        self.transform = transform
        self.type = type
        samples_cells = []
        samples_pixels = []
        samples_labels = []
        for root, dirs, files in os.walk(os.path.join(path)):
            for f in files:
                if "train_puzzle_pixels" in f:
                    with open(os.path.join(path, root, f), "r") as liner:
                        for i, l in enumerate(liner.readlines()):
                            pixels = []
                            for c in range(16):
                                step = 28 * 28
                                number = l.split("\t")[c * step:c * step + step]
                                pixels.append([float(n) for n in number])
                            samples_pixels.append(pixels)

                if "train_cell_labels" in f:
                    with open(os.path.join(path, root, f), "r") as liner:
                        for i, l in enumerate(liner.readlines()):
                            cells = [((c % 4, c // 4), int(j.split("_")[1])) for c, j in enumerate(l.split("\t"))]
                            samples_cells.append(cells)

                if "train_puzzle_labels" in f:
                    with open(os.path.join(path, root, f), "r") as liner:
                        for i, l in enumerate(liner.readlines()):
                            label = 0 if l.split("\t")[0] == "1" else 0
                            samples_labels.append(label)

        samples = [(p, c, l) for c, p, l in zip(samples_cells, samples_pixels, samples_labels)]

        self.samples = samples
    def __getitem__(self, index):
        item = self.samples[index]
        # TODO: transforms
        return item[0:1], item[2]

def  get_loaders(path, batch_size, type="mnist"):

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.1307,), (0.3081,))])

    # TODO: Dataset
    match type:
        case 'mnist':
            # TODO: MNIST

            trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                                  download=True, transform=transform)

            train_set, val_set = torch.utils.data.random_split(trainset, [50000, 10000])

            testset = torchvision.datasets.MNIST(root='./data', train=False,
                                                 download=True, transform=transform)

        case 'sudoku4':
            trainset = sudoku_dataset(path='./data', train=True,
                                      transform=transform)

            train_set, val_set = torch.utils.data.random_split(trainset, [50000, 10000])

            testset = sudoku_dataset(path='./data', train=False,
                                     transform=transform)
        case _:
            raise ValueError(f"Dataset {type} not supported.")

    trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                              shuffle=True, num_workers=2)

    valloader = torch.utils.data.DataLoader(val_set, batch_size=batch_size,
                                            shuffle=False, num_workers=2)

    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)

    return trainloader, valloader, testloader