import seaborn as sns

from matplotlib import pyplot as plt

from src.trainer.bp_trainer import TrainFeedback


def plot_loss_and_val_loss(feedback: TrainFeedback, size=(12, 6)) -> None:
    plt.figure(figsize=size)
    plt.plot([it.train.loss for it in feedback.history], label="Training Loss")
    plt.plot([it.eval.loss for it in feedback.history], label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


def plot_f1_score(feedback: TrainFeedback, size=(12, 6)) -> None:
    plt.figure(figsize=size)
    plt.plot([it.eval.f1 for it in feedback.history], label="F1-score")
    plt.title("Eval F1-score")
    plt.xlabel("Epoch")
    plt.ylabel("F-score")
    plt.legend()
    plt.show()


def plot_estimators_count_to_f1(x, y, size=(12, 6)):
    plt.figure(figsize=size)
    plt.plot(x, y, label="F1-score")
    plt.title("Estimator count to F1-score")
    plt.xlabel("Estimator count")
    plt.ylabel("F-score")
    plt.legend()
    plt.show()
