import math

import click
import ltn
import torch
import torchmetrics
from torch import nn
from torch.nn import functional as F
from tqdm import trange, tqdm

from make_dataloader import get_loaders


class SudokuNet(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes

        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, self.n_classes)

    def forward(self, x):
        original_batch_size = x.shape[0]
        x = x.reshape(-1, 1, 28, 28)

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x).softmax(dim=-1)
        # out = torch.sum(x * l, dim=1)
        return x.reshape(original_batch_size, -1, self.n_classes)


def get_sub_square(x, i, j, type):
    return x[:, i * type: (i + 1) * type, j * type: (j + 1) * type, :]


def isValidSudoku(board, n_classes):
    board2 = F.one_hot(board.argmax(dim=-1), num_classes=n_classes).view(-1, n_classes, n_classes, n_classes)
    sqrt_classes = int(math.sqrt(n_classes))
    res = torch.tensor([True] * board.shape[0]).to(board.device)
    for i in range(sqrt_classes):
        res *= board2[:, i * sqrt_classes: (i + 1) * sqrt_classes, i * sqrt_classes: (i + 1) * sqrt_classes, :].reshape(
            -1, n_classes, n_classes).sum(-2).all(1)
    # print(res.shape)
    return board2.sum(-2).all(-1).all(1) * board2.sum(-3).all(-1).all(1) * res


@click.command()
@click.option('--epochs', default=10, help='Number of epochs to train.')
@click.option('--batch-size', default=2, help='Batch size.')
@click.option('--n_classes', default=4, help='Number of classes.')
@click.option('--lr', default=0.001, help='Learning rate.')
@click.option('--log_interval', default=10, help='How often to log results.')
@click.option('--dataset', default='sudoku4', help='Dataset to use.')
@click.option('--path', default='data/MNISTx4Sudoku', help='Path for dataset')
def main(epochs, batch_size, n_classes, lr, log_interval, dataset, path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trainloader, valloader, testloader = get_loaders(path, batch_size, type=dataset)

    Not = ltn.Connective(ltn.fuzzy_ops.NotStandard())
    And = ltn.Connective(ltn.fuzzy_ops.AndProd())
    Or = ltn.Connective(ltn.fuzzy_ops.OrProbSum())
    Implies = ltn.Connective(ltn.fuzzy_ops.ImpliesReichenbach())
    Dist = ltn.Predicate(func=
                         lambda x, y, x2, y2: torch.exp(
                             -1. * torch.sqrt(torch.sum(torch.square(x - x2) + torch.square(y - y2), dim=1))))
    Equiv = ltn.Connective(
        ltn.fuzzy_ops.Equiv(ltn.fuzzy_ops.AndProd(), ltn.fuzzy_ops.ImpliesReichenbach()))
    Forall = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=2), quantifier="f")
    Exists = ltn.Quantifier(ltn.fuzzy_ops.AggregPMean(p=6), quantifier="e")

    # SamePoint = ltn.Predicate(func=lambda x1, y1, x2, y2: (x1 == x2 * y1 == y2))

    square_classes = int(math.sqrt(n_classes))
    SameSquare = ltn.Predicate(
        func=lambda x1, y1, x2, y2: torch.logical_not((x1 == x2) * (y1 == y2)) *
                                    (x1 // square_classes == x2 // square_classes) * (
                                            y1 // square_classes == y2 // square_classes))
    EqualPosition = ltn.Predicate(func=lambda x1, x2, y1, y2: (x1 == x2) * (y1 == y2))

    EqualLine = ltn.Predicate(func=lambda x1, y1, x2, y2: ((x1 == x2) * torch.logical_not(y1 == y2)))

    EqualImageNumber = ltn.Predicate(
        func=lambda image, x1, y1, x2, y2: torch.exp(
            -1. * ((image[torch.arange(image.size(0)), (x1 + y1 * n_classes).squeeze()] - image[
                torch.arange(image.size(0)), (x2 + y2 * n_classes).squeeze()]) ** 2).sum(-1)))

    # Digit = ltn.Predicate(
    #     func=lambda image, x, y, l: (image[:, x, y] * l[:, x * len(l) + y]).sum(-1))

    sat_agg = ltn.fuzzy_ops.SatAgg(agg_op=ltn.fuzzy_ops.AggregPMean(p=2))

    cnn = SudokuNet(n_classes=n_classes)
    cnn.to(device)
    cnn.train()

    x1, x2, y1, y2 = (ltn.Variable(f"p{i}", torch.arange(n_classes)) for i in range(4))
    # print(x1)
    correct = ltn.Constant(torch.tensor(1))
    wrong = ltn.Constant(torch.tensor(0))

    optimizer = torch.optim.Adam(cnn.parameters(), lr=lr)

    auc = torchmetrics.AUROC(task="multiclass", num_classes=n_classes)
    accuracy = torchmetrics.Accuracy(num_classes=2)

    for epoch in trange(epochs):
        train_acc = 0
        pred_list = None
        label_list = None
        for (batch_idx, batch) in enumerate(tqdm(trainloader, leave=False)):
            if batch_idx == 100:
                break
            x, labels, sudoku_label = batch
            x = x.to(device)
            labels = labels.to(device)
            sudoku_label = sudoku_label.to(device)
            # print(x.shape)
            # print(labels.shape)
            # print(sudoku_label)
            onehot_labels = torch.nn.functional.one_hot(labels, num_classes=n_classes)
            onehot_labels = onehot_labels.reshape(batch_size, n_classes, n_classes, n_classes)
            onehot_sudoku_label = torch.nn.functional.one_hot(sudoku_label, num_classes=2)

            # print("onehot_labels:", onehot_labels.shape)

            l = ltn.Variable("l", onehot_labels)
            s = ltn.Variable("s", sudoku_label)
            sl = ltn.Variable("sl", onehot_sudoku_label)

            optimizer.zero_grad()
            result = cnn(x)
            # print("result:", result.shape)
            # print(result.sum(-1))
            result = ltn.Variable("result", result)
            # prediction = ltn.Variable("prediction", prediction)

            loss = 1. - sat_agg(
                Forall(s,
                       Forall([x1, y1, x2, y2],
                              Implies(Or(Or(SameSquare(x1, y1, x2, y2),
                                            EqualLine(x1, y1, x2, y2)),
                                         EqualLine(y1, x1, y2, x2)),
                                      Not(
                                          EqualImageNumber(result, x1, y1, x2, y2)))),
                       cond_vars=[s],
                       cond_fn=lambda s: s.value == correct.value,
                       ),

                Forall(s,
                       Exists([x1, y1, x2, y2],
                              Implies(Or(Or(SameSquare(x1, y1, x2, y2),
                                            EqualLine(x1, y1, x2, y2)),
                                         EqualLine(y1, x1, y2, x2)),
                                      EqualImageNumber(result, x1, y1, x2, y2))),
                       cond_vars=[s],
                       cond_fn=lambda s: s.value == wrong.value,
                       ),

                # Forall(
                #     ltn.diag(result, l),
                #     Digit(result, l)).value
                # ,
                #
                # Forall(
                #     ltn.diag(prediction, sl),
                #     isCorrect(prediction, sl)).value
            )

            # print(f"{isValidSudoku(result.value, n_classes).shape} {s.value.shape}")
            # print(isValidSudoku(result.value, n_classes).shape)
            accuracy(isValidSudoku(result.value, n_classes).squeeze().cpu(), s.value.squeeze().cpu())
            auc(result.value.reshape(-1, 4).cpu(), labels.reshape(-1).cpu())

            if batch_idx % log_interval == 0:
                print(f"Loss: {loss.item()}")

            loss.backward()
            optimizer.step()
        print(
            f"Epoch {epoch}: Sat Level {1 - loss.item():.5f} Puzzle Accuracy: {accuracy.compute():.3f}, Task AUC: {auc.compute():.5f}")
        accuracy.reset()
        auc.reset()

        # accuracy.reset()
        # print("Training finished at Epoch %d with Sat Level %.5f and Puzzle Accuracy: %.1f Puzzle AUC: %.5f" % (
        #     epoch, 1 - loss.item(), 100 * (train_acc / len(trainloader.dataset)), auc(pred_list, label_list).item()))


if __name__ == '__main__':
    main()
