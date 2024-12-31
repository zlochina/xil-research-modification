import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# TODO: remove this class or replace variables with such as `model`, `optimizer`, `loss_fn` etc.
class ModelConfig:
    LEARNING_RATE = 1e-3
    BATCH_SIZE = 64
    EPOCHS = 5

    def __init__(self, learning_rate=LEARNING_RATE, batch_size=BATCH_SIZE, epochs=EPOCHS):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs

class XILUtils:
    @staticmethod
    def define_device() -> str:
        """
        Determines the appropriate device for PyTorch computations.

        This function checks the availability of CUDA and MPS (Metal Performance Shaders)
        and returns the most suitable device for computation. If neither CUDA nor MPS
        is available, it defaults to using the CPU.

        Returns:
            str: A string representing the device to be used for computations.
                 Possible values are "cuda", "mps", or "cpu".

        """
        device: str  = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        return device

    @staticmethod
    def rrr_train_loop(model: nn.Module, dataloader: DataLoader, loss_fn: nn.Module, optimizer: nn.Module,
                       batch_size: int, device: str):
        size = len(dataloader.dataset)
        interval = size // 10

        model.train()
        for batch, (X, y, A) in enumerate(dataloader):
            # move X and y to device
            X, y, A = X.to(device), y.to(device), A.to(device)
            # Compute prediction and loss
            pred = model(X)
            loss = loss_fn(pred, y, A)

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if (batch * batch_size) % interval < batch_size:
                loss, current = loss.item(), batch * batch_size + len(X)
                print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")

    @staticmethod
    def rrr_test_loop(model: nn.Module, dataloader: DataLoader, loss_fn: nn.Module, num_classes: int, device: str):
        model.eval()
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        test_loss, correct = 0, 0

        # with torch.no_grad(): prevents calculating gradients, which does not suit our rrr loss function
        for X, y, A in dataloader:
            # move X, y to device
            X, y, A = X.to(device), y.to(device), A.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y, A).item()
            correct += (F.one_hot(pred.argmax(1), num_classes=num_classes) == y).type(
                torch.float).sum().item() / num_classes

        test_loss /= num_batches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    @staticmethod
    def train_loop(dataloader: torch.utils.data.DataLoader, model: torch.nn.Module, loss_fn, optimizer, model_config: ModelConfig, device: str):

        size = len(dataloader.dataset)

        model.train()
        for batch, (X, y) in enumerate(dataloader):
            # move X and y to device
            X, y = X.to(device), y.to(device)
            # Compute prediction and loss
            pred = model(X)
            loss = loss_fn(pred, y)

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if batch % 100 == 0:
                loss, current = loss.item(), batch * model_config.batch_size + len(X)
                print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")

    # TODO: review how it is used in `pytorch.ipynb` and why I was forced to rewrite it in `08_MNIST.ipynb`
    @staticmethod
    def test_loop(dataloader, model, loss_fn, device):
        model.eval()
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        test_loss, correct = 0, 0

        with torch.no_grad():
            for X, y in dataloader:
                # move X, y to device
                X, y = X.to(device), y.to(device)
                pred = model(X)
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()

            test_loss /= num_batches
            correct /= size
            print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
