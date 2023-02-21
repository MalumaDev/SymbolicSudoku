import click
import ltn
import torch
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
        self.fc4 = nn.Linear(self.n_classes * self.n_classes * 84, 2)

    def forward(self, x):
        original_batch_size = x.shape[0]
        x = x.reshape(original_batch_size, self.n_classes * self.n_classes, 28, 28).reshape(-1, 1, 28, 28)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        cells = self.fc3(x).softmax(dim=-2)
        cells = cells.reshape(original_batch_size, self.n_classes, self.n_classes, self.n_classes)
        puzzle = x.reshape(original_batch_size, -1)
        puzzle = self.fc4(x).softmax(dim=-1)
        # x = x.argmax(dim=-1)
        return cells


@click.command()
@click.option('--epochs', default=10, help='Number of epochs to train.')
@click.option('--batch-size', default=2, help='Batch size.')
@click.option('--n_classes', default=4, help='Number of classes.')
@click.option('--lr', default=0.001, help='Learning rate.')
@click.option('--log_interval', default=100, help='How often to log results.')
@click.option('--dataset', default='mnist', help='Dataset to use.')
@click.option('--path', default='', help='Path for dataset')
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

    SamePoint = ltn.Predicate(func=lambda x1, y1, x2, y2: (
            (F.one_hot(x1, num_classes=n_classes) * F.one_hot(x2, num_classes=n_classes)) * (
            F.one_hot(y1, num_classes=n_classes) * F.one_hot(y2, num_classes=n_classes))).sum(-1))
    SameSquare = ltn.Predicate(func=lambda x1, y1, x2, y2: (x1 // 3 == x2 // 3) * (y1 // 3 == y2 // 3))
    EqualPosition = ltn.Predicate(func=lambda x1, x2: x1 == x2)

    EqualLine = ltn.Predicate(func=lambda x1, y1, x2, y2: ((x1 == x2) * (y1 != y2)))

    EqualImageNumber = ltn.Predicate(
        func=lambda image, x1, y1, x2, y2: torch.exp(
            -1. * ((image[:, x1, y1] - image[:, x2, y2]) ** 2).reshape(image.shape[0], -1).sum(-1)))

    Digit = ltn.Predicate(
        func=lambda image, label: torch.exp(
            -1. * ((image * label) ** 2).reshape(image.shape[0], -1).sum(-1)))

    isCorrect = ltn.Predicate(
        func=lambda image, l: (image * l).sum(-1))

    sat_agg = ltn.fuzzy_ops.SatAgg(agg_op=ltn.fuzzy_ops.AggregPMean(p=2))

    cnn = SudokuNet(n_classes=n_classes)
    cnn.to(device)

    x1, x2, y1, y2 = (ltn.Variable(f"p{i}", torch.arange(n_classes)) for i in range(4))
    correct = ltn.Constant(torch.tensor(1))
    wrong = ltn.Constant(torch.tensor(0))

    optimizer = torch.optim.Adam(cnn.parameters(), lr=lr)

    for epoch in trange(epochs):
        for (batch_idx, batch) in enumerate(tqdm(trainloader, leave=False)):
            x, labels, sudoku_label = batch
            x.to(device)
            labels.to(device)
            sudoku_label.to(device)

            onehot_labels = torch.nn.functional.one_hot(labels, num_classes=n_classes)
            onehot_labels = onehot_labels.reshape(batch_size, n_classes, n_classes, n_classes)

            l = ltn.Variable("l", onehot_labels)
            s = ltn.Variable("s", sudoku_label)
            result = cnn(x)
            result = ltn.Variable("result", result)
            optimizer.zero_grad()
            loss = 1. - sat_agg(
                Forall(s,
                       Forall([x1, y1, x2, y2],
                              Implies(And(Not(SameSquare(x1, y1, x2, y2)),
                                          # And(Not(SamePoint(x1, y1, x2, y2)),
                                          #     Or(EqualPosition(x1, x2),
                                          #         EqualPosition(y1, y2)))),
                                          # # Abbiamo modificato creando una Not And per ridurre il numero di predicati da 3 a 2
                                          Not(And(EqualLine(x1, y1, x2, y2),
                                                  EqualLine(y1, x1, y2, x2)))),
                                      Not(
                                          EqualImageNumber(result, x1, y1, x2, y2)))),
                       cond_vars=[s],
                       cond_fn=lambda s: s.value == correct.value,
                       ).value.mean(),

                Forall(s,
                       Exists([x1, y1, x2, y2],
                              Implies(And(Not(SameSquare(x1, y1, x2, y2)),
                                          # And(Not(SamePoint(x1, y1, x2, y2)),
                                          #     Or(EqualPosition(x1, x2),
                                          #         EqualPosition(y1, y2)))),
                                          Not(And(EqualLine(x1, y1, x2, y2),
                                                  EqualLine(y1, x1, y2, x2)))),
                                      EqualImageNumber(result, x1, y1, x2, y2))),
                       cond_vars=[s],
                       cond_fn=lambda s: s.value == wrong.value,
                       ).value.mean(),

                Forall(
                    ltn.diag(result, l),
                    Digit(result, l)).value
                    )

            if batch_idx % log_interval == 0:
                print(f"Loss: {loss.item()}")
            loss.backward()
            optimizer.step()
        print("Epoch %d: Sat Level %.5f " % (epoch, 1 - loss.item()))

        print("Training finished at Epoch %d with Sat Level %.5f" % (epoch, 1 - loss.item()))


if __name__ == '__main__':
    main()
