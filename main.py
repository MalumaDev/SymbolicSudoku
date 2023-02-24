import math

import click
import ltn
import torch
import torchmetrics
from torch import nn
from torch.nn import functional as F
from tqdm import trange, tqdm

from make_dataloader import get_loaders


# torch.backends.cudnn.benchmark = True


class STEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return (input + 1e-8 > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        return torch.tanh(grad_output)


class StraightThroughEstimator(nn.Module):
    def __init__(self):
        super(StraightThroughEstimator, self).__init__()

    def forward(self, x):
        x = STEFunction.apply(x)
        return x


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

        # x = self.pool(F.leaky_relu(self.conv1(x)))
        # x = self.pool(F.leaky_relu(self.conv2(x)))
        # x = torch.flatten(x, 1)
        # x = F.leaky_relu(self.fc1(x))
        # x = F.leaky_relu(self.fc2(x))
        # x = self.fc3(x).softmax(dim=-1)
        # # out = torch.sum(x * l, dim=1)
        # return x.reshape(original_batch_size, -1, self.n_classes)


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


def equal_image_number(image, x1, y1, x2, y2):
    # image = image[batch_index]
    # x1 = x1[batch_index]
    # y1 = y1[batch_index]
    # x2 = x2[batch_index]
    # y2 = y2[batch_index]

    x1 = x1.squeeze()
    y1 = y1.squeeze()
    x2 = x2.squeeze()
    y2 = y2.squeeze()
    n_classes = image.shape[-1]

    a = image[torch.arange(image.size(0)), (x1 + y1 * n_classes)]
    b = image[torch.arange(image.size(0)), (x2 + y2 * n_classes)]

    def func(x, alpha=1e10):
        return (1- torch.sigmoid(-alpha * (x - x.max(-1, keepdims=True).values))) * 2

    return (func(a) * func(b)).sum(-1)

    # return (torch.heaviside(a - a.max(-1, keepdims=True).values, torch.tensor(1.,requires_grad=True)) * torch.heaviside(
    #     b - b.max(-1, keepdims=True).values, torch.tensor(1.,requires_grad=True))).sum(-1)

    # return torch.exp(
    #     -1. * ((image[torch.arange(image.size(0)), (x1 + y1 * n_classes)] - image[
    #         torch.arange(image.size(0)), (x2 + y2 * n_classes)]) ** 2).sum(-1)).unsqueeze(-1)


def equal_line(x1, y1, x2, y2):
    # x1 = x1[batch_index]
    # y1 = y1[batch_index]
    # x2 = x2[batch_index]
    # y2 = y2[batch_index]

    return (x1 == x2) * torch.logical_not(y1 == y2)


def equal_position(x1, y1, x2, y2):
    # x1 = x1[batch_index]
    # y1 = y1[batch_index]
    # x2 = x2[batch_index]
    # y2 = y2[batch_index]
    return (x1 == x2) * (y1 == y2)


def same_square(x1, y1, x2, y2, square_classes):
    # x1 = x1[batch_index]
    # y1 = y1[batch_index]
    # x2 = x2[batch_index]
    # y2 = y2[batch_index]
    x1 = x1.squeeze()
    y1 = y1.squeeze()
    x2 = x2.squeeze()
    y2 = y2.squeeze()

    # print(f"x1: {x1.shape}, y1: {y1.shape}, x2: {x2.shape}, y2: {y2.shape}, square_classes: {square_classes.shape}")
    #
    # print(f"x1 // square_classes: {(x1 // square_classes).shape}")
    # print(f"(x1 == x2): {(x1 == x2).shape}")
    # print((torch.logical_not((x1 == x2) * (y1 == y2)) * (x1 // square_classes == x2 // square_classes) * (
    #         y1 // square_classes == y2 // square_classes)).shape)

    return (torch.logical_not((x1 == x2) * (y1 == y2)) * (x1 // square_classes == x2 // square_classes) * (
            y1 // square_classes == y2 // square_classes)).unsqueeze(-1)


def isRightDigit(yp, y):
    # print(f"yp: {yp.shape}, y: {y.shape}")
    batch_size = yp.shape[0]
    # yp = yp.reshape(-1, yp.shape[-1])
    # y = y.reshape(-1, y.shape[-1])
    # print(f"yp: {yp.shape}, y: {y.shape}")

    # # distance between two vectors
    # dist = torch.exp(
    #     -1. * torch.sqrt(torch.sum(torch.square(yp - y), dim=-1)))
    # return dist.reshape(batch_size, -1).mean(-1)
    return (yp * y).sum(-1).mean(-1)


def possible_to_be_same_number(x1, y1, x2, y2, square_classes):
    print(x1.shape, y1.shape, x2.shape, y2.shape, square_classes.shape)



@click.command()
@click.option('--epochs', default=100, help='Number of epochs to train.')
@click.option('--batch-size', default=65, help='Batch size.')
@click.option('--n_classes', default=4, help='Number of classes.')
@click.option('--lr', default=0.001, help='Learning rate.')
@click.option('--log_interval', default=20, help='How often to log results.')
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
    # Equiv = ltn.Connective(
    #     ltn.fuzzy_ops.Equiv(ltn.fuzzy_ops.AndProd(), ltn.fuzzy_ops.ImpliesReichenbach()))
    Forall = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=2, stable=True), quantifier="f")
    Exists = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=2, stable=True), quantifier="e")

    # SamePoint = ltn.Predicate(func=lambda x1, y1, x2, y2: (x1 == x2 * y1 == y2))

    square_classes = ltn.Constant(torch.tensor(int(math.sqrt(n_classes))))
    SameSquare = ltn.Predicate(func=same_square)
    EqualPosition = ltn.Predicate(func=equal_position)

    EqualLine = ltn.Predicate(func=equal_line)

    EqualImageNumber = ltn.Predicate(func=equal_image_number)
    PossibleToBeSameNumber = ltn.Predicate(func=possible_to_be_same_number)

    # Mean = ltn.Predicate(func=mean)

    Digit = ltn.Predicate(func=isRightDigit)

    # sat_agg = ltn.fuzzy_ops.SatAgg(agg_op=ltn.fuzzy_ops.AggregPMean(p=2,))
    sat_agg = ltn.fuzzy_ops.SatAgg(agg_op=ltn.fuzzy_ops.AggregPMeanError(p=2, stable=True))

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
        for (batch_idx, batch) in enumerate(tqdm(trainloader)):
            # if batch_idx == 100:
            #     break

            x, labels, sudoku_label = batch

            # import matplotlib.pyplot as plt
            # image = torch.zeros(1,28,28*16)
            # for i in range(16):
            #     image[0, 0:28, i*28:(i+1)*28] = x[0][i]
            # plt.imshow(torchvision.transforms.ToPILImage()(image))
            # newline = ""
            # plt.title(" ".join([f"{newline if i %4 == 0 and i != 0 else ''}{n}" for i,n in enumerate(labels[0].tolist())]))
            # plt.tight_layout()
            # plt.show()
            x = x.to(device)
            labels = labels.to(device)
            sudoku_label = sudoku_label.to(device)
            # print(x.shape)
            # print(labels.shape)
            # print(sudoku_label)
            onehot_labels = torch.nn.functional.one_hot(labels, num_classes=n_classes)
            # onehot_labels = onehot_labels.reshape(x.shape[0], n_classes, n_classes, n_classes)
            onehot_sudoku_label = torch.nn.functional.one_hot(sudoku_label, num_classes=2)

            # print("onehot_labels:", onehot_labels.shape)

            l = ltn.Variable("l", onehot_labels)
            s = ltn.Variable("s", sudoku_label)
            batch_index = ltn.Variable("batch_index", torch.arange(x.shape[0]))
            sl = ltn.Variable("sl", onehot_sudoku_label)

            optimizer.zero_grad()
            result = cnn(x)
            # print("result:", result.shape)
            # print(result.sum(-1))
            result = ltn.Variable("result", result)
            # prediction = ltn.Variable("prediction", prediction)

            loss = 1 - sat_agg(
                Forall(ltn.diag(s, result),
                       Forall([x1, y1, x2, y2],
                              Implies(Or(Or(SameSquare(x1, y1, x2, y2, square_classes),
                                            EqualLine(x1, y1, x2, y2)),
                                         EqualLine(y1, x1, y2, x2)),
                                      Not(EqualImageNumber(result, x1, y1, x2, y2)))),
                       cond_vars=[s],
                       cond_fn=lambda v: v.value == correct.value,
                       p=2
                       ),
                Forall(ltn.diag(s, result), Exists([x1, y1, x2, y2],
                                                   Implies(Or(Or(SameSquare(x1, y1, x2, y2, square_classes),
                                                                 EqualLine(x1, y1, x2, y2)),
                                                              EqualLine(y1, x1, y2, x2)),
                                                           EqualImageNumber(result, x1, y1, x2, y2))),
                       cond_vars=[s],
                       cond_fn=lambda v: v.value == wrong.value,
                       p=2
                       ),

                # Forall(
                #     ltn.diag(result, l),
                #     Digit(result, l),
                # )

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
