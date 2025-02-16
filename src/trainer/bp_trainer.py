from dataclasses import dataclass
import torch
from torch.utils.data import DataLoader
from torch import nn
from torch.optim import Optimizer
from torcheval.metrics.functional import multiclass_f1_score, multiclass_accuracy

from src.util.progress_bar import create_progress_bar


@dataclass
class TrainMetrics:
    loss: float = 0


@dataclass
class EvalMetrics:
    loss: float = 0
    accuracy: float = 0
    f1: float = 0


@dataclass
class Metrics:
    eval: EvalMetrics
    train: TrainMetrics


@dataclass
class TrainFeedback:
    history: list[Metrics]


class Trainer:
    num_epochs: int
    train_loader: DataLoader
    eval_loader: DataLoader
    model: nn.Module
    loss_fun: nn.Module
    optimizer: Optimizer
    device: torch.device

    def __init__(
        self,
        num_epochs: int,
        train_loader: DataLoader,
        eval_loader: DataLoader,
        model: nn.Module,
        loss_fun: nn.Module,
        optimizer: Optimizer,
        device: torch.device,
    ):
        self.num_epochs = num_epochs
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.model = model
        self.loss_fun = loss_fun
        self.optimizer = optimizer
        self.device = device

    def train(self):
        num_batches = len(self.train_loader)
        history = []

        for epoch in create_progress_bar()(
            range(self.num_epochs), desc="Overall progress"
        ):
            self.model.train()

            with create_progress_bar()(range(num_batches)) as pb:
                pb.set_description("Epoch [%s]" % (epoch + 1))

                train_total_loss = 0

                for x, y_true in self.train_loader:
                    x = x.to(self.device)
                    y_true = y_true.to(self.device)

                    outputs = self.model(x)

                    loss = self.loss_fun(outputs, y_true)

                    train_total_loss += loss.item()

                    self.optimizer.zero_grad()
                    loss.backward()

                    self.optimizer.step()

                    pb.update()
                    pb.set_postfix(loss=loss.item())

                train_metrics = TrainMetrics(train_total_loss / num_batches)
                eval_metrics = self.evaluate()

                history.append(Metrics(eval_metrics, train_metrics))

                progress_postfix = {
                    "loss": train_metrics.loss,
                    "eval_loss": eval_metrics.loss,
                    "eval_accuracy": eval_metrics.accuracy,
                    "eval_f1": eval_metrics.f1,
                }

                pb.set_postfix(**progress_postfix)

        return TrainFeedback(history)

    def evaluate(self) -> EvalMetrics:
        return self.evaluate_with(self.eval_loader)

    def evaluate_with(self, loader: DataLoader) -> EvalMetrics:
        self.model.eval()
        with torch.no_grad():
            total_loss = 0
            total_accuracy = 0
            total_f1 = 0
            batch_count = len(loader)

            for x, y_true in loader:
                x = x.to(self.device)
                y_true = y_true.to(self.device)
                y_predicted = self.model(x)
                total_loss += self.loss_fun(y_predicted, y_true).item()

                total_accuracy += multiclass_accuracy(y_predicted, y_true).item()
                total_f1 += multiclass_f1_score(y_predicted, y_true).item()

            return EvalMetrics(
                total_loss / batch_count,
                total_accuracy / batch_count,
                total_f1 / batch_count,
            )
