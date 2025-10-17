#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/10/16 20:35
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   18_linear_regression_pyside6.py
# @Desc     :   

from torch import Tensor, randn, tensor, nn, optim, device, float32, cat
from torch.utils.data import TensorDataset, DataLoader
from typing import override

from utils.helper import Timer
from utils.PT import device_checker

from PySide6.QtCore import Qt, Signal, QThread
from PySide6.QtCharts import QScatterSeries, QChart, QChartView, QLineSeries
from PySide6.QtGui import QPainter
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget,
                               QVBoxLayout, QHBoxLayout,
                               QPushButton, QLabel, QDoubleSpinBox)
from sys import argv, exit


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Linear Regression using PyTorch and PySide6")
        self.resize(1000, 500)
        self._widget = QWidget(self)
        self.setCentralWidget(self._widget)

        self._chart_points = QChart()
        self._view_points = QChartView(self._chart_points)
        self._chart_train = QChart()
        self._view_train = QChartView(self._chart_train)

        self._btn_labels = ["Plot", "Train", "Exit"]
        self._buttons = []
        self._spin_labels: list[str] = ["Alpha", "Epochs", "Batches"]
        self._spins: dict[str, QDoubleSpinBox] = {}

        self._losses = QLineSeries()
        self._params = QLineSeries()
        self._trainer = None
        self._scatter_xs = []
        self._scatter_ys = []
        self._fit_line = QLineSeries()

        self._setup()

    def _setup(self):
        _layout = QVBoxLayout()
        _row_views = QHBoxLayout()
        _row = QHBoxLayout()

        _outer = QHBoxLayout()
        for lbl in self._spin_labels:
            inner = QHBoxLayout()
            label = QLabel(lbl, self)
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            inner.addWidget(label)

            spin = QDoubleSpinBox()
            if lbl == "Alpha":
                spin.setMinimum(0.01)
                spin.setMaximum(1.0)
                spin.setSingleStep(0.01)
                spin.setValue(0.01)
            elif lbl == "Epochs":
                spin.setMinimum(1)
                spin.setMaximum(10000)
                spin.setSingleStep(1)
                spin.setValue(100)
            elif lbl == "Batches":
                spin.setMinimum(1)
                spin.setMaximum(500)
                spin.setSingleStep(1)
                spin.setValue(16)
            spin.setAlignment(Qt.AlignmentFlag.AlignCenter)
            inner.addWidget(spin)

            _outer.addLayout(inner)
            self._spins[lbl] = spin
        _layout.addLayout(_outer)

        # Chart View
        self._view_points.setRenderHint(QPainter.RenderHint.Antialiasing)
        self._view_train.setRenderHint(QPainter.RenderHint.Antialiasing)
        _row_views.addWidget(self._view_points)
        _row_views.addWidget(self._view_train)
        _layout.addLayout(_row_views)

        funcs = [
            self._click2plot,
            self._click2train,
            self.close,
        ]
        for i, label in enumerate(self._btn_labels):
            button = QPushButton(label, self)
            button.clicked.connect(funcs[i])
            if button.text() == "Train":
                button.setEnabled(False)
            self._buttons.append(button)
            _row.addWidget(button)
        _layout.addLayout(_row)

        self._widget.setLayout(_layout)

    def _click2plot(self) -> None:
        """ Plot random data points """
        self._scatter_xs = cat([x for x, _ in data_setter()]).numpy()
        self._scatter_ys = cat([y for _, y in data_setter()]).numpy()

        # Delete previous series
        self._chart_points.removeAllSeries()

        scatter = QScatterSeries()
        scatter.setName("y = 3x + 2 + N (N is noise)")
        scatter.setColor(Qt.GlobalColor.blue)

        for x, y in zip(self._scatter_xs, self._scatter_ys):
            scatter.append(float(x), float(y))

        # Add scatter series
        self._chart_points.addSeries(scatter)
        # Add fitted line series
        self._fit_line.setName("Fitted Line")
        self._fit_line.setColor(Qt.GlobalColor.red)
        self._chart_points.addSeries(self._fit_line)
        # Create default axes
        self._chart_points.createDefaultAxes()

        for button in self._buttons:
            if button.text() == "Train":
                button.setEnabled(True)
        for button in self._buttons:
            if button.text() == "Plot":
                button.setEnabled(False)

    def _click2train(self) -> None:
        """ Clear the chart """
        # Set Device
        device_name: str = "cpu"
        devices = device_checker(device_name)
        # Get parameters from spin boxes
        alpha = self._spins["Alpha"].value()
        epochs = int(self._spins["Epochs"].value())
        batches = int(self._spins["Batches"].value())
        print(f"Alpha: {alpha}, Epochs: {epochs}, Batches: {batches}")

        self._chart_train.removeAllSeries()
        self._losses.setName("Training Loss")
        self._chart_train.addSeries(self._losses)
        self._chart_train.createDefaultAxes()

        self._trainer = Trainer(device_name, devices[device_name], epochs, batches, alpha)
        self._trainer.losses.connect(self._losses_updater)
        self._trainer.params.connect(self._params_updater)
        self._trainer.start()

        for button in self._buttons:
            if button.text() == "Train":
                button.setEnabled(False)
        for button in self._buttons:
            if button.text() == "Plot":
                button.setEnabled(True)

    def _losses_updater(self, index: int, loss: float) -> None:
        self._losses.append(index, loss)

        # Set axis x range dynamically
        axis_x = self._chart_train.axes(Qt.Orientation.Horizontal, self._losses)[0]
        axis_x.setRange(0, index * 1.1)

        # Set axis y range dynamically
        axis_y = self._chart_train.axes(Qt.Orientation.Vertical, self._losses)[0]
        points_y = [self._losses.at(i).y() for i in range(self._losses.count())]
        if points_y:
            min_y = min(points_y)
            max_y = max(points_y)
            axis_y.setRange(min_y * 0.9, max_y * 1.1)

        # Update the view
        self._view_train.update()

    def _params_updater(self, weight: float, bias: float) -> None:
        # Get min and max x for line plotting
        x_min = min(self._scatter_xs)
        x_max = max(self._scatter_ys)

        # Clear previous line and append new fitted line
        self._fit_line.clear()
        self._fit_line.append(x_min, weight * x_min + bias)
        self._fit_line.append(x_max, weight * x_max + bias)

        self._view_points.update()


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


class Trainer(QThread):
    # index: int, loss: float
    losses = Signal(int, float)
    # Weight and Bias
    params = Signal(float, float)

    def __init__(self, device_name: str, target_device: device, epochs: int, batches: int, alpha: float):
        super().__init__()
        self._device_name = device_name
        self._target_device = target_device
        self._epochs = epochs
        self._batches = batches
        self._alpha = alpha

    @override
    def run(self):
        """ Trainer Function """
        # Set a model
        model = nn.Linear(1, 1, dtype=float32, device=self._target_device)
        # Set an optimizer
        optimiser: optim.SGD = optim.SGD(model.parameters(), lr=self._alpha)
        # Set loss function and optimizer
        criterion: nn.MSELoss = nn.MSELoss()
        # Set dataset and dataloader
        dataset: TensorDataset = data_setter()
        loader: DataLoader = data_loader(dataset, self._batches)

        with Timer(f"Training on {self._device_name.upper()}"):
            for i in range(self._epochs):
                # batch_loss: float = 0.0
                # batch_count: float = 0.0
                for j, (features, labels) in enumerate(loader):
                    features, labels = features.to(self._target_device), labels.to(self._target_device)

                    # Zero gradients
                    optimiser.zero_grad()
                    # Forward pass
                    labels_hat: Tensor = model(features)
                    loss: Tensor = criterion(labels_hat, labels)
                    # Backward pass and optimization
                    loss.backward()
                    # Update parameters / gradients
                    optimiser.step()

                    if j % 10 == 0:
                        self.losses.emit(i, loss.item())
                        self.params.emit(model.weight.item(), model.bias.item())


def main() -> None:
    """ Main Function """
    app = QApplication(argv)
    window = MainWindow()
    window.show()
    exit(app.exec())


if __name__ == "__main__":
    main()
