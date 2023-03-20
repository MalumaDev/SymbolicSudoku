import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import math
from pathlib import Path

import click
import ltn
import pandas as pd
import torch
import torchmetrics
from torch import nn
from torch.nn import functional as F
from tqdm import trange
import numpy as np

from make_dataloader import get_loaders


# torch.backends.cudnn.benchmark = True

def isValidSudoku(board, n_classes):
    board2 = F.one_hot(board.argmax(dim=-1), num_classes=n_classes).view(-1, n_classes, n_classes, n_classes)
    sqrt_classes = int(math.sqrt(n_classes))
    res = torch.tensor([True] * board.shape[0]).to(board.device)
    for i in range(sqrt_classes):
        res *= board2[:, i * sqrt_classes: (i + 1) * sqrt_classes, i * sqrt_classes: (i + 1) * sqrt_classes, :].reshape(
            -1, n_classes, n_classes).sum(-2).all(1)
    return board2.sum(-2).all(-1).all(1) * board2.sum(-3).all(-1).all(1) * res


def distance(a, b, dist=0.1):
    return torch.exp(-torch.relu(torch.pairwise_distance(a, b) - dist))


def isValidSudokuDist(board, dist=0.1):
    correct = torch.ones(board.shape[0], dtype=torch.float, device=board.device, requires_grad=True)
    cl = board.shape[-1]
    scl = int(math.sqrt(cl))
    for i in range(board.shape[-1]):
        for j in range(board.shape[-1]):
            for k in range(j + 1, board.shape[-1]):
                # rows
                correct = correct * distance(board[:, i + j * board.shape[-1]],
                                             board[:, i + k * board.shape[-1]], dist)
                # columns
                correct = correct * distance(board[:, i * board.shape[-1] + j],
                                             board[:, i * board.shape[-1] + k], dist)
                # # squares
                correct = correct + distance(board[:,
                                             i * scl + j + (scl - 1) * cl * (i // scl) + (cl - scl) * (
                                                     j // scl)],
                                             board[:, i * scl + k + (scl - 1) * cl * (
                                                     i // scl) + (cl - scl) * (
                                                              k // scl)], dist
                                             )

    return correct


def equal_image_number(image, all_points_comb):
    x1, y1, x2, y2 = [all_points_comb[:, i] for i in range(4)]
    n_classes = image.shape[-1]

    a = image[torch.arange(image.size(0)), (x1 + y1 * n_classes)]
    b = image[torch.arange(image.size(0)), (x2 + y2 * n_classes)]

    res = torch.pairwise_distance(a, b)
    return torch.exp(-torch.relu(res - 0.1))


def isRightDigit(yp, y):
    return (yp * y).sum(-1).mean(-1)


def possible_to_be_same_number_not_equal(all_points_comb, square_classes):
    x1, y1, x2, y2 = [all_points_comb[:, i] for i in range(4)]

    samex = x1 == x2
    samey = y1 == y2
    same_square = (x1 // square_classes == x2 // square_classes) * (y1 // square_classes == y2 // square_classes)

    res = torch.logical_not(same_square) * torch.logical_not(samex + samey)
    # print(res)
    # for i in range(x1.shape[0]):
    #     print()
    #     res = samex * samey + torch.logical_not(same_square) * torch.logical_not(samex + samey)
    #     print(f"({x1[i].item(),y1[i].item()}) - ({x2[i].item(),y2[i].item()}) = {samex[i].item(),samey[i].item(),same_square[i].item()} - result: {res[i].item()}")

    return res


def possible_to_be_same_number(all_points_comb, square_classes):
    # print(x1.shape, y1.shape, x2.shape, y2.shape, square_classes.shape)
    x1, y1, x2, y2 = [all_points_comb[:, i] for i in range(4)]

    samex = x1 == x2
    samey = y1 == y2
    same_square = ((x1 // square_classes) == (x2 // square_classes)) * (
                (y1 // square_classes) == (y2 // square_classes))

    res = samex * samey + torch.logical_not(same_square) * torch.logical_not(samex + samey)

    return res


def calculate_points_pred(pred, all_points_comb):
    x1 = all_points_comb[:, 0]
    y1 = all_points_comb[:, 1]
    x2 = all_points_comb[:, 2]
    y2 = all_points_comb[:, 3]
    return (torch.pairwise_distance(pred[:, x1 + y1 * pred.shape[-1]],
                                    pred[:, x2 + y2 * pred.shape[-1]]).reshape(-1))


def calculate_points_labels(labels, all_points_comb, n_classes=4):
    x1 = all_points_comb[:, 0]
    y1 = all_points_comb[:, 1]
    x2 = all_points_comb[:, 2]
    y2 = all_points_comb[:, 3]
    return ((labels[:, x1 + y1 * n_classes] == labels[:, x2 + y2 * n_classes]).reshape(-1)).to(int)


@click.command()
@click.option('--epochs', default=200, help='Number of epochs to train.')
@click.option('--split', default=0, help='Number of split to train on (Max 11).')
@click.option('--batch-size', default=8, help='Batch size.')
@click.option('--lr', default=0.001, help='Learning rate.')
@click.option('--log_interval', default=100, help='How often to log results.')
@click.option('--dataset', default='mnist4', help='Dataset to use.')
@click.option('--generate_dataset', default=False, help='Generate dataset.', is_flag=True)
@click.option('--algorithm', default=0, help='Algorithm to use.')
def main(split, batch_size, lr, log_interval, dataset, generate_dataset, epochs, algorithm):
    assert 0 <= split <= 11, \
        "Number of split should be between 1 and 11!"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    ltn.device = device

    res_path = Path(f"results/{dataset}_split_{split:02d}.csv")
    res_path.parent.mkdir(parents=True, exist_ok=True)

    trainloader, valloader, testloader, n_classes = get_loaders(batch_size, type=dataset, split=split)
    if generate_dataset:
        exit()

    all_points_comb = []
    for i in trange(n_classes):
        for j in range(n_classes):
            for k in range(n_classes):
                for l in range(n_classes):
                    all_points_comb.append([i, j, k, l])

    all_points_comb = ltn.Variable(f"all_points_comb", torch.tensor(all_points_comb, device=device))
    correct = ltn.Constant(torch.tensor(1, device=device))
    wrong = ltn.Constant(torch.tensor(0))
    step_func = None

    match algorithm:
        case 0:
            class SudokuNet(nn.Module):
                def __init__(self, n_classes):
                    super().__init__()
                    self.n_classes = n_classes

                    self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
                    self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
                    self.conv2_drop = nn.Dropout2d()
                    self.fc1 = nn.Linear(320, 50)
                    self.fc2 = nn.Linear(50, self.n_classes)

                def forward(self, x):
                    original_batch_size = x.shape[0]

                    # transforms.ToPILImage()(x[0][0]).show()
                    x = x.reshape(-1, 1, 28, 28)

                    x = F.relu(F.max_pool2d(self.conv1(x), 2))
                    x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
                    x = x.view(-1, 320)
                    x = F.relu(self.fc1(x))
                    x = F.dropout(x, training=self.training)
                    x = self.fc2(x)
                    return F.softmax(x, dim=-1).reshape(original_batch_size, -1, self.n_classes)
            
            max_dist = 0.1

            Not = ltn.Connective(ltn.fuzzy_ops.NotStandard())
            Implies = ltn.Connective(ltn.fuzzy_ops.ImpliesReichenbach())

            Forall = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=2), quantifier="f")
            Exists = ltn.Quantifier(ltn.fuzzy_ops.AggregPMean(p=2), quantifier="e")
            tmp = torch.tensor(int(math.sqrt(n_classes)), device=device)
            square_classes = ltn.Constant(tmp, trainable=False)

            EqualImageNumber = ltn.Predicate(func=equal_image_number)
            PossibleToBeSameNumber = ltn.Predicate(func=possible_to_be_same_number)
            PossibleToBeSameNumberNotEqual = ltn.Predicate(func=possible_to_be_same_number_not_equal)

            sat_agg = ltn.fuzzy_ops.SatAgg()

            cnn = SudokuNet(n_classes=n_classes)
            cnn.to(device)
            cnn.train()

            optimizer = torch.optim.Adam(cnn.parameters(), lr=lr)

            def func(x, labels, sudoku_label, optimizer):
                s = ltn.Variable("s", sudoku_label)

                optimizer.zero_grad()
                result = cnn(x)
                result = ltn.Variable("result", result)

                loss = 1 - sat_agg(
                    Forall(ltn.diag(s, result),
                           Forall([all_points_comb],
                                  Implies(Not(PossibleToBeSameNumber(all_points_comb, square_classes)),
                                          Not(EqualImageNumber(result, all_points_comb)))),
                           cond_vars=s,
                           cond_fn=lambda v: v.value == correct.value,
                           ),
                )

                return loss, result

            step_func = func
        case 1:
            def CheckAllDifferent(result):              
                size = int(np.cbrt(result.shape[-1]))
                result = result.reshape(result.shape[0], size, size, size)
                output = [True for i in range(result.shape[0])]
                output = torch.tensor(output)
                result_t = result.clone().transpose(1, 2)
                for x in range(size):   
                  row = torch.argmax(result[:, x], dim=-1) 
                  col = torch.argmax(result_t[:, x], dim=-1)               
                  for n in range(size):
                    output = torch.logical_and(output, torch.any(row==n, dim=-1))
                    output = torch.logical_and(output, torch.any(col==n, dim=-1))
                
                grid_size = int(np.sqrt(size))
                for x1 in range(grid_size):   
                  for y1 in range(grid_size): 
                    square = torch.argmax(result[:, x1 * grid_size :x1 * grid_size + grid_size,  y1 * grid_size :y1 * grid_size + grid_size], dim=-1) 
                    square = square.view(square.shape[0], -1)
                    for n in range(size):
                      output = torch.logical_and(output, torch.any(square==n, dim=-1))

                return output

            class SudokuNet(nn.Module):
              def __init__(self, n_classes, training=True):
                  super().__init__()
                  self.n_classes = n_classes
                  self.training = training

                  self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
                  self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
                  self.conv2_drop = nn.Dropout2d()
                  self.fc1 = nn.Linear(320, 50)
                  self.fc2 = nn.Linear(50, self.n_classes)
                  self.fc3 = nn.Linear(5120, 320)
                  self.fc4 = nn.Linear(320, 2)

                  self.Not = ltn.Connective(ltn.fuzzy_ops.NotStandard())
                  self.Implies = ltn.Connective(ltn.fuzzy_ops.ImpliesReichenbach())

                  self.Forall = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=2), quantifier="f")
                  self.Exists = ltn.Quantifier(ltn.fuzzy_ops.AggregPMean(p=2), quantifier="e")

                  self.x = ltn.Variable("x", torch.tensor(range(n_classes)))
                  self.y = ltn.Variable("y", torch.tensor(range(n_classes)))
                  self.correct = ltn.Constant(torch.tensor([0, 1]))
                  self.wrong = ltn.Constant(torch.tensor([1, 0]))

                  self.Digit = ltn.Predicate(model=Predicate())
                  self.Puzzle = ltn.Predicate(model=Predicate())

                  self.Rules = ltn.Predicate(func=CheckAllDifferent)

              def forward(self, x, labels, sudoku_label):
                  original_batch_size = x.shape[0]

                  # transforms.ToPILImage()(x[0][0]).show()
                  x = x.reshape(-1, 1, 28, 28)

                  x = F.relu(F.max_pool2d(self.conv1(x), 2))
                  x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
                  y = x.clone().view(original_batch_size, -1)
                  x = x.view(-1, 320)
                  x = F.relu(self.fc1(x))
                  x = F.dropout(x, training=self.training)
                  x = self.fc2(x)
                  y = F.relu(self.fc3(y))
                  y = F.dropout(y, training=self.training)

                  y = self.fc4(y)
                  digits_pred1 = x.reshape(-1, self.n_classes) 
                  digits_pred2 = x.clone().reshape(original_batch_size, -1)   
                  puzzle_pred = y      
                  
                  labels = ltn.Variable("labels", nn.functional.one_hot(labels, self.n_classes).reshape(-1, self.n_classes))
                  sudoku_label = ltn.Variable("sudoku_label", nn.functional.one_hot(sudoku_label, 2))

                  digits = ltn.Variable("digits", digits_pred1)
                  digits_rules = ltn.Variable("digits_rules", digits_pred2)
                  puzzle = ltn.Variable("puzzle", puzzle_pred)

                  if self.training:
                    P1 = self.Forall(ltn.diag(digits, labels),
                          self.Digit(digits, labels)).value
                  
                  P2 = self.Forall(ltn.diag(puzzle, sudoku_label),
                        self.Puzzle(puzzle, sudoku_label)).value


                  P3 = self.Forall(ltn.diag(sudoku_label, digits_rules),
                                  self.Rules(digits_rules),
                           cond_vars=[sudoku_label],
                           cond_fn=lambda s: (s.value == self.wrong.value)[:, 0],
                           ).value    

                  P4 = self.Forall(ltn.diag(sudoku_label, digits_rules),
                                  self.Not(self.Rules(digits_rules)),
                           cond_vars=[sudoku_label],
                           cond_fn=lambda s: (s.value == self.wrong.value)[:, 0],
                           ).value  

                  if self.training:
                    sat = sat_agg(
                      P1,
                      P2,
                      P3,
                      P4
                    )
                  else:
                    sat = sat_agg(
                      P2,
                      P3,
                      P4
                    ) 
                  
                  return sat, F.softmax(digits_pred1, dim=-1).reshape(original_batch_size, -1, self.n_classes)
                                                   
            class Predicate(torch.nn.Module):
              def __init__(self):
                  super(Predicate, self).__init__()
                  self.softmax = nn.Softmax(dim=-1)

              def forward(self, x, l):
                  x = self.softmax(x)
                  out = torch.sum(x * l, dim=-1)
                  return out


            max_dist = 0.1
            
            sat_agg = ltn.fuzzy_ops.SatAgg()

            cnn = SudokuNet(n_classes=n_classes)
            cnn.to(device)
            cnn.train()

            optimizer = torch.optim.Adam(cnn.parameters(), lr=lr)

            def func(x, labels, sudoku_label, optimizer):
                optimizer.zero_grad()
                sat, result = cnn(x, labels, sudoku_label)

                loss = 1 - sat

                return loss, result

            step_func = func
        case 2:
            pass
        case _:
            print("Algorithm not found!")
            exit(0)

    auc = torchmetrics.AUROC(task="binary", num_classes=1)
    points_auc = torchmetrics.AUROC(task="binary", num_classes=1)
    accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=2)
    accuracyDist = torchmetrics.Accuracy()
    aucDist = torchmetrics.AUROC(task="multiclass", num_classes=n_classes)
    # Test metrics
    test_auc = torchmetrics.AUROC(task="multiclass", num_classes=n_classes)
    test_points_auc = torchmetrics.AUROC(task="binary", num_classes=1)
    test_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=2)
    test_accuracyDist = torchmetrics.Accuracy()
    test_aucDist = torchmetrics.AUROC(task="multiclass", num_classes=n_classes)

    # Val metrics
    val_auc = torchmetrics.AUROC(task="binary", num_classes=1)
    val_points_auc = torchmetrics.AUROC(task="binary", num_classes=1)
    val_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=2)
    val_accuracyDist = torchmetrics.Accuracy()
    val_aucDist = torchmetrics.AUROC(task="multiclass", num_classes=n_classes)

    df = pd.DataFrame(columns=["epoch", "train_loss", "train_auc", "train_points_auc", "train_accuracy",
                               "val_auc", "val_points_auc", "val_accuracy", "test_auc", "test_points_auc",
                               "test_accuracy", "test_accuracyDist", "test_aucDist"])

    for epoch in trange(epochs):
        for (batch_idx, batch) in enumerate(trainloader):
            x, labels, sudoku_label = batch

            x = x.to(device)
            labels = labels.to(device)
            sudoku_label = sudoku_label.to(device)

            loss, result = step_func(x, labels, sudoku_label, optimizer)
            if algorithm == 1:
              result = ltn.Variable("result", result)
            
            loss.backward()
            optimizer.step()

            res = isValidSudoku(result.value.detach(), n_classes).squeeze().cpu()
            accuracy(res, sudoku_label.squeeze().cpu())
            auc(res.to(float), sudoku_label.cpu())
            res = isValidSudokuDist(result.value.detach(), max_dist).squeeze().cpu()
            accuracyDist(res, sudoku_label.squeeze().cpu())
            aucDist(res, sudoku_label.squeeze().cpu())

            # auc(result.value.reshape(-1, n_classes).softmax(-1).cpu(), labels.reshape(-1).cpu())

            points_auc(calculate_points_pred(result.value.cpu().detach(), all_points_comb.value.cpu().detach()),
                       calculate_points_labels(labels.cpu().detach(), all_points_comb.value.cpu().detach(),
                                               n_classes=n_classes))

            if batch_idx % log_interval == 0:
                print(f"Loss: {loss.item()}")

        loss_log = 1 - loss.item()
        accuracy_log = accuracy.compute().cpu().numpy().item()
        accuracyDist_log = accuracyDist.compute().cpu().numpy().item()
        aucDist_log = aucDist.compute().cpu().numpy().item()
        auc_log = auc.compute().cpu().numpy().item()
        points_auc_log = points_auc.compute().cpu().numpy().item()

        print(
            f"Epoch {epoch} Training: Sat Level {loss_log:.5f} Puzzle Accuracy: {accuracy_log:.3f}, Puzzle AUC: {auc_log:.5f}"
            f" Task AUC: {points_auc_log:.5f}, Accuracy Dist: {accuracyDist_log:.3f}, AUC Dist: {aucDist_log:.3f}")

        with torch.no_grad():
            cnn.eval()
            for (batch_idx, batch) in enumerate(valloader):
                x, labels, sudoku_label = batch

                x = x.to(device)
                labels = labels.to(device)
                sudoku_label = sudoku_label.to(device)

                s = ltn.Variable("s", sudoku_label)

                if not algorithm == 2:
                  result = cnn(x)
                else:
                  _, result = cnn(x, labels, sudoku_label)
                result = ltn.Variable("result", result)

                res = isValidSudoku(result.value, n_classes).squeeze().cpu()
                val_accuracy(res, s.value.squeeze().cpu())
                val_auc(res.to(float), s.value.squeeze().cpu())
                res = isValidSudokuDist(result.value, max_dist).squeeze().cpu()
                val_accuracyDist(res, s.value.squeeze().cpu())
                val_aucDist(res, s.value.squeeze().cpu())

                # val_auc(result.value.reshape(-1, n_classes).softmax(-1).cpu(), labels.reshape(-1).cpu())

                val_points_auc(
                    calculate_points_pred(result.value.cpu(), all_points_comb.value.cpu()),
                    calculate_points_labels(labels.cpu(), all_points_comb.value.cpu(), n_classes=n_classes))

            # val_auc_log = val_auc.compute()
            val_points_auc_log = val_points_auc.compute().cpu().numpy().item()
            val_accuracy_log = val_accuracy.compute().cpu().numpy().item()
            val_accuracyDist_log = val_accuracyDist.compute().cpu().numpy().item()
            val_aucDist_log = val_aucDist.compute().cpu().numpy().item()
            val_auc_log = val_auc.compute().cpu().numpy().item()

            print(
                f"Epoch {epoch} Validation: Puzzle Accuracy: {val_accuracy_log:.3f}, Puzzle AUC: {val_auc_log:.3f}"
                f" Task AUC: {val_points_auc_log:.5f}, Accuracy Dist: {val_accuracyDist_log:.3f}, AUC Dist: {val_aucDist_log:.3f} ")

            df = df.append({"epoch": epoch,
                            "train_loss": loss_log,
                            "train_points_auc": points_auc_log,
                            "train_accuracy": accuracy_log,
                            "train_accuracyDist": accuracyDist_log,
                            "train_aucDist": aucDist_log,
                            "train_auc_log": auc_log,
                            "val_points_auc": val_points_auc_log,
                            "val_accuracy": val_accuracy_log,
                            "val_accuracyDist": val_accuracyDist_log,
                            "val_aucDist": val_aucDist_log,
                            "val_auc_log": val_auc_log,
                            }, ignore_index=True)
            df.to_csv(res_path, index=False)

        val_accuracy.reset()
        val_points_auc.reset()
        # val_auc.reset()
        cnn.train()

        accuracy.reset()
        points_auc.reset()
        # auc.reset()

    with torch.no_grad():
        cnn.eval()
        for (batch_idx, batch) in enumerate(testloader):
            x, labels, sudoku_label = batch

            x = x.to(device)
            labels = labels.to(device)
            sudoku_label = sudoku_label.to(device)

            s = ltn.Variable("s", sudoku_label)

            if not algorithm == 2:
              result = cnn(x)
            else:
              _, result = cnn(x, labels, sudoku_label)
            result = ltn.Variable("result", result)

            res = isValidSudoku(result.value, n_classes).squeeze().cpu()
            test_accuracy(res, s.value.squeeze().cpu())
            test_auc(res.to(float), s.value.squeeze().cpu())
            res = isValidSudokuDist(result.value, max_dist).squeeze().cpu()
            test_accuracyDist(res, s.value.squeeze().cpu())
            test_aucDist(res, s.value.squeeze().cpu())

            test_points_auc(
                calculate_points_pred(result.value.cpu(), all_points_comb.value.cpu()),
                calculate_points_labels(labels.cpu(), all_points_comb.value.cpu(), n_classes=n_classes))

            # val_auc_log = val_auc.compute()
        test_points_auc_log = test_points_auc.compute().cpu().numpy().item()
        test_accuracy_log = test_accuracy.compute().cpu().numpy().item()
        test_accuracyDist_log = test_accuracyDist.compute().cpu().numpy().item()
        test_aucDist_log = test_aucDist.compute().cpu().numpy().item()
        test_auc_log = test_auc.compute().cpu().numpy().item()

    df = df.append({"epoch": -1,
                    "test_auc": test_auc_log,
                    "test_points_auc": test_points_auc_log,
                    "test_accuracy": test_accuracy_log,
                    "test_accuracyDist": test_accuracyDist_log,
                    "test_aucDist": test_aucDist_log,
                    }, ignore_index=True)
    df.to_csv(res_path, index=False)


if __name__ == '__main__':
    main()
