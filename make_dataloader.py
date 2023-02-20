import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms
from PIL import Image

class sudoku_dataset(Dataset):
    def __init__(self, path, tr_va_te="train", transform=None, type=4):
        self.transform = transform
        self.type = type
        samples_cells = []
        samples_pixels = []
        samples_labels = []
        for root, dirs, files in os.walk(os.path.join(path)):
            for f in files:
                if tr_va_te+"_puzzle_pixels" in f:
                    with open(os.path.join(root, f), "r") as liner:
                        for i, l in enumerate(liner.readlines()):
                            pixels = []
                            for c in range(type*type):
                                step = 28 * 28
                                number = l.split("\t")[c * step:c * step + step]
                                pixels.append([float(n) for n in number])
                            samples_pixels.append(pixels)
                            if (tr_va_te == "val" or tr_va_te == "test") and i >= 20:
                                break

                if tr_va_te+"_cell_labels" in f:
                    with open(os.path.join(root, f), "r") as liner:
                        for i, l in enumerate(liner.readlines()):
                            # cells = [((c % type, c // type), int(j.split("_")[1])) for c, j in enumerate(l.split("\t"))]
                            cells = [int(j.split("_")[1]) for c, j in enumerate(l.split("\t"))]
                            samples_cells.append(cells)
                            if (tr_va_te == "val" or tr_va_te == "test") and i >= 20:
                                break

                if tr_va_te+"_puzzle_labels" in f:
                    with open(os.path.join( root, f), "r") as liner:
                        for i, l in enumerate(liner.readlines()):
                            label = 0 if l.split("\t")[0] == "1" else 0
                            samples_labels.append(label)
                            if (tr_va_te == "val" or tr_va_te == "test") and i >= 20:
                                break

        samples = [(p, c, l) for p, c, l in zip(samples_pixels, samples_cells, samples_labels)]

        self.samples = samples

    def __len__(self):
        return len(self.samples)
    def __getitem__(self, index):
        item = self.samples[index]
        imgs = []
        for n in item[0]:
            img = Image.fromarray(np.reshape(np.array(n), (28,28)), mode="L")

            if self.transform is not None:
                img = self.transform(img)

            imgs.append(img)
        imgs = torch.cat(imgs)
        labels = torch.tensor(np.array(item[1]))
        sudoku_label = item[2]
        return imgs, labels, sudoku_label

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
            train_set = sudoku_dataset(path=path, tr_va_te="train",
                                       transform=transform)

            val_set = sudoku_dataset(path=path, tr_va_te="val",
                                     transform=transform)

            testset = sudoku_dataset(path=path, tr_va_te="test",
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