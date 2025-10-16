#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/10/16 17:19
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   18_linear_regression.py
# @Desc     :   

from multiprocessing import Process, set_start_method
from torch import Tensor, randn, tensor, nn, optim, device, float32
from torch.utils.data import TensorDataset, DataLoader

from utils.helper import Timer
from utils.highlighter import green, red
from utils.PT import device_checker


def data_setter(scalar: float = 0.02) -> TensorDataset:
    """ Linear Regression Example
    - y = 3x + 2 + N (N is noise)
    :param scalar: the scalar to control the noise level
    :return: the dataset containing features and labels
    """
    X: Tensor = randn(100, 1, dtype=float32)
    N: Tensor = randn(100, 1, dtype=float32).mul(scalar)  # Noise
    W: Tensor = tensor([3.0], dtype=float32)
    b: Tensor = tensor([2.0], dtype=float32)

    y: Tensor = X.mul(W).add(b).add(N)

    return TensorDataset(X, y)


def data_loader(dataset: TensorDataset, batches: int, shuffle: bool = True) -> DataLoader:
    """ Data Loader using Pytorch DataLoader class
    :param dataset: the dataset containing features and labels
    :param batches: the number of batches
    :param shuffle: whether to shuffle the data
    :return: the loader
    """
    return DataLoader(dataset, batch_size=batches, shuffle=shuffle)


def trainer(device_name: str, target_device: device, epochs: int, batches: int, alpha: float, ):
    """ Trainer Function """
    # Set a model
    model = nn.Linear(1, 1, dtype=float32, device=target_device)
    # Set an optimizer
    optimiser: optim.SGD = optim.SGD(model.parameters(), lr=alpha)
    # Set loss function and optimizer
    criterion: nn.MSELoss = nn.MSELoss()
    # Set dataset and dataloader
    dataset: TensorDataset = data_setter()
    loader: DataLoader = data_loader(dataset, batches)

    with Timer(f"Training on {device_name.upper()}"):
        losses: list[float] = []

        for i in range(epochs):
            batch_loss: float = 0.0
            batch_count: float = 0.0
            for j, (features, labels) in enumerate(loader):
                features, labels = features.to(target_device), labels.to(target_device)

                # Zero gradients
                optimiser.zero_grad()
                # Forward pass
                labels_hat: Tensor = model(features)
                loss: Tensor = criterion(labels_hat, labels)
                # Backward pass and optimization
                loss.backward()
                # Update parameters / gradients
                optimiser.step()

                if i % 10 == 0:
                    losses.append(loss.item())
                    print(
                        f"{device_name}: Batch {j + 1:<05} "
                        f"- Loss {green(f'{loss.item():.5f}') if loss.item() < 0.05 else red(f'{loss.item():.5f}')}"
                    )

                batch_loss += loss.item()
                batch_count += 1.0

            epoch_loss = batch_loss / batch_count
            print(
                f"{device_name}: Epoch {i + 1:<05} "
                f"- Loss {green(f'{epoch_loss:.5f}') if epoch_loss < 0.05 else red(f'{epoch_loss:.5f}')}"
            )

        print(f"{device_name}: Training completed. W = {model.weight.item():.3f}, b = {model.bias.item():.3f}")


def main() -> None:
    """ Main Function """
    # for macOS & PyTorch safety
    set_start_method("spawn", force=True)

    # Set Device
    devices = device_checker()

    # Set epochs, batches and ALPHA
    EPOCHS: int = 200
    BATCHES: int = 16
    ALPHA: float = 0.001

    processes: list[Process] = []
    for name, dev in devices.items():
        process: Process = Process(
            target=trainer,
            args=(name, dev, EPOCHS, BATCHES, ALPHA,),
        )
        processes.append(process)
        process.start()

    for process in processes:
        process.join()


if __name__ == "__main__":
    main()
