import itertools
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

from make_dataloader import get_loaders


# torch.backends.cudnn.benchmark = True


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


def isValidSudoku(board, n_classes):
    board2 = F.one_hot(board.argmax(dim=-1), num_classes=n_classes).view(-1, n_classes, n_classes, n_classes)
    sqrt_classes = int(math.sqrt(n_classes))
    res = torch.tensor([True] * board.shape[0]).to(board.device)
    for i in range(sqrt_classes):
        res *= board2[:, i * sqrt_classes: (i + 1) * sqrt_classes, i * sqrt_classes: (i + 1) * sqrt_classes, :].reshape(
            -1, n_classes, n_classes).sum(-2).all(1)
    return board2.sum(-2).all(-1).all(1) * board2.sum(-3).all(-1).all(1) * res


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
    same_square = (x1 // square_classes == x2 // square_classes) * (y1 // square_classes == y2 // square_classes)

    res = samex * samey + torch.logical_not(same_square) * torch.logical_not(samex + samey)

    return res


def calculate_points_pred(pred, all_points_comb, max_dist=0.1):
    x1 = all_points_comb[:, 0]
    y1 = all_points_comb[:, 1]
    x2 = all_points_comb[:, 2]
    y2 = all_points_comb[:, 3]
    return (torch.pairwise_distance(pred[:, x1 + y1 * pred.shape[-1]],
                                    pred[:, x2 + y2 * pred.shape[-1]]).reshape(-1) < max_dist).to(int)


def calculate_points_labels(labels, all_points_comb, n_classes=4):
    x1 = all_points_comb[:, 0]
    y1 = all_points_comb[:, 1]
    x2 = all_points_comb[:, 2]
    y2 = all_points_comb[:, 3]
    return ((labels[:, x1 + y1 * n_classes] == labels[:, x2 + y2 * n_classes]).reshape(-1)).to(int)


@click.command()
@click.option('--epochs', default=20, help='Number of epochs to train.')
@click.option('--batch-size', default=65, help='Batch size.')
@click.option('--lr', default=0.001, help='Learning rate.')
@click.option('--log_interval', default=100, help='How often to log results.')
@click.option('--dataset', default='mnist4', help='Dataset to use.')
def main(epochs, batch_size, lr, log_interval, dataset):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    res_path = Path(f"results/{dataset}.csv")
    res_path.parent.mkdir(parents=True, exist_ok=True)

    trainloader, valloader, testloader, n_classes = get_loaders(batch_size, type=dataset)
    max_dist = 0.1

    Not = ltn.Connective(ltn.fuzzy_ops.NotStandard())
    Implies = ltn.Connective(ltn.fuzzy_ops.ImpliesReichenbach())

    Forall = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=2), quantifier="f")
    Exists = ltn.Quantifier(ltn.fuzzy_ops.AggregPMean(p=2), quantifier="e")

    square_classes = ltn.Constant(torch.tensor(int(math.sqrt(n_classes))))

    EqualImageNumber = ltn.Predicate(func=equal_image_number)
    PossibleToBeSameNumber = ltn.Predicate(func=possible_to_be_same_number)
    PossibleToBeSameNumberNotEqual = ltn.Predicate(func=possible_to_be_same_number_not_equal)

    sat_agg = ltn.fuzzy_ops.SatAgg()

    cnn = SudokuNet(n_classes=n_classes)
    cnn.to(device)
    cnn.train()

    # all_points_comb = ltn.Variable(f"all_points_comb", torch.tensor(
    #     list(itertools.product(*[list(range(n_classes)) for _ in range(n_classes)]))))

    all_points_comb = []
    for i in trange(n_classes):
        for j in range(n_classes):
            for k in range(n_classes):
                for l in range(n_classes):
                    all_points_comb.append([i, j, k, l])

    all_points_comb = ltn.Variable(f"all_points_comb", torch.tensor(all_points_comb, device=device))
    correct = ltn.Constant(torch.tensor(1, device=device))
    wrong = ltn.Constant(torch.tensor(0))

    optimizer = torch.optim.Adam(cnn.parameters(), lr=lr)

    auc = torchmetrics.AUROC(task="multiclass", num_classes=n_classes)
    points_auc = torchmetrics.AUROC(task="binary", num_classes=1)
    accuracy = torchmetrics.Accuracy(num_classes=2)
    # Test metrics
    test_auc = torchmetrics.AUROC(task="multiclass", num_classes=n_classes)
    test_points_auc = torchmetrics.AUROC(task="binary", num_classes=1)
    test_accuracy = torchmetrics.Accuracy(num_classes=2)

    df = pd.DataFrame(columns=["epoch", "train_loss", "train_auc", "train_points_auc", "train_accuracy",
                               "test_auc", "test_points_auc", "test_accuracy"])


    for epoch in trange(epochs):
        for (batch_idx, batch) in enumerate(trainloader):

            x, labels, sudoku_label = batch

            x = x.to(device)
            labels = labels.to(device)
            sudoku_label = sudoku_label.to(device)

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
            loss.backward()
            optimizer.step()

            accuracy(isValidSudoku(result.value.detach(), n_classes).squeeze().cpu(), s.value.squeeze().cpu())

            # auc(result.value.reshape(-1, n_classes).softmax(-1).cpu(), labels.reshape(-1).cpu())

            points_auc(calculate_points_pred(result.value.cpu().detach(), all_points_comb.value.cpu().detach(), max_dist=max_dist),
                       calculate_points_labels(labels.cpu().detach(), all_points_comb.value.cpu().detach(), n_classes=n_classes))

            if batch_idx % log_interval == 0:
                print(f"Loss: {loss.item()}")


        loss_log = 1 - loss.item()
        accuracy_log = accuracy.compute()
        # auc_log = auc.compute()
        points_auc_log = points_auc.compute()
        print(
            f"Epoch {epoch} Training: Sat Level {loss_log:.5f} Puzzle Accuracy: {accuracy_log:.3f},"
            f" Task AUC: {points_auc_log:.5f}")

        with torch.no_grad():
            cnn.eval()
            for (batch_idx, batch) in enumerate(valloader):
                x, labels, sudoku_label = batch

                x = x.to(device)
                labels = labels.to(device)
                sudoku_label = sudoku_label.to(device)

                s = ltn.Variable("s", sudoku_label)

                result = cnn(x)
                result = ltn.Variable("result", result)

                test_accuracy(isValidSudoku(result.value, n_classes).squeeze().cpu(), s.value.squeeze().cpu())

                # test_auc(result.value.reshape(-1, n_classes).softmax(-1).cpu(), labels.reshape(-1).cpu())

                test_points_auc(calculate_points_pred(result.value.cpu(), all_points_comb.value.cpu(), max_dist=max_dist),
                                calculate_points_labels(labels.cpu(), all_points_comb.value.cpu(), n_classes=n_classes))


            # test_auc_log = test_auc.compute()
            test_points_auc_log = test_points_auc.compute()
            test_accuracy_log = test_accuracy.compute()

            print(
                f"Epoch {epoch} Validation: Puzzle Accuracy: {test_accuracy_log:.3f},"
                f" Task AUC: {test_points_auc_log:.5f}")

            df = df.append({"epoch": epoch,
                            "train_loss": loss_log,
                            "train_points_auc": points_auc_log,
                            "train_accuracy": accuracy_log,
                            "test_points_auc": test_points_auc_log,
                            "test_accuracy": test_accuracy_log}, ignore_index=True)
            df.to_csv(res_path, index=False)

        test_accuracy.reset()
        test_points_auc.reset()
        # test_auc.reset()
        cnn.train()

        accuracy.reset()
        points_auc.reset()
        # auc.reset()



if __name__ == '__main__':
    main()
