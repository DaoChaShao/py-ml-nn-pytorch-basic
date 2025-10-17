#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/10/17 22:07
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   19_DP_05_optimiser_02_lr_step_size.py
# @Desc     :

from torch import Tensor, tensor, float32, optim

from utils.helper import RandomSeed

from PySide6.QtCharts import QLineSeries, QChart, QChartView
from PySide6.QtGui import QPainter
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget,
                               QVBoxLayout, QHBoxLayout,
                               QPushButton, )
from sys import argv, exit


def f(X):
    return 0.05 * X[0] ** 2 + X[1] ** 2


def grad_descent_without_scheduler(X: Tensor, optimiser, epochs: int, batches: int) -> tuple[list, list]:
    """ Gradient Descent Optimizer """
    # Forward pass
    learning_rates = []
    outputs = []
    output = None
    for epoch in range(epochs):
        for batch in range(batches):
            optimiser.zero_grad()
            output = f(X)
            output.backward()
            optimiser.step()

        if epoch % 10 == 0:
            learning_rates.append(optimiser.param_groups[0]["lr"])
            outputs.append(output.item())

    print("Learning Rates:", learning_rates)
    print("Function Outputs:", outputs)

    return learning_rates, outputs


def grad_descent_with_scheduler(X: Tensor, optimiser, epochs: int, batches: int, scheduler) -> tuple[list, list]:
    """ Gradient Descent Optimizer """
    # Forward pass
    learning_rates = []
    outputs = []
    output = None
    for epoch in range(epochs):
        for batch in range(batches):
            optimiser.zero_grad()
            output = f(X)
            output.backward()
            optimiser.step()

        if epoch % 10 == 0:
            learning_rates.append(optimiser.param_groups[0]["lr"])
            outputs.append(output.item())

        # Update learning rate
        scheduler.step()

    print("Learning Rates:", learning_rates)
    print("Function Outputs:", outputs)

    return learning_rates, outputs


def train():
    with RandomSeed("Optimiser with Step Learning Rate Scheduler"):
        device = "cpu"

        X: Tensor = tensor([-7.0, 2.0], device=device, dtype=float32, requires_grad=True)

        ALPHA: float = 0.9
        EPOCHS: int = 100
        BATCHES: int = 16

        X_SGD = X.clone().detach().requires_grad_(True)
        X_sched = X.clone().detach().requires_grad_(True)

        optimizer_SGD = optim.SGD([X_SGD], lr=ALPHA)
        optimizer_sched = optim.SGD([X_sched], lr=ALPHA)
        steps: int = 10
        scale: float = 0.7
        scheduler = optim.lr_scheduler.StepLR(optimizer_sched, step_size=steps, gamma=scale)

        # Forward pass
        alphas_SGD, outs_SGD = grad_descent_without_scheduler(X_SGD, optimizer_SGD, EPOCHS, BATCHES)
        alphas_sched, outs_sched = grad_descent_with_scheduler(X_sched, optimizer_sched, EPOCHS, BATCHES, scheduler)

    return alphas_SGD, outs_SGD, alphas_sched, outs_sched


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Optimiser with Step Learning Rate Scheduler")
        self.resize(800, 400)
        self._widget = QWidget(self)
        self.setCentralWidget(self._widget)

        self._chart = QChart()
        self._view = QChartView(self._chart)
        self._btn_labels = ["Plot", "Clear", "Exit"]
        self._buttons = []

        self._line_titles: list[str] = [
            "SGD Learning Rate",
            "SGD Function Output",
            "StepLR Learning Rate",
            "StepLR Function Output",
        ]
        self._line_data: list = [*train()]

        self._setup()

    def _setup(self):
        _layout = QVBoxLayout()
        _row = QHBoxLayout()

        # Chart View
        self._view.setRenderHint(QPainter.RenderHint.Antialiasing)
        _layout.addWidget(self._view)

        funcs = [
            self._click2plot,
            self._click2clear,
            self.close,
        ]
        for i, label in enumerate(self._btn_labels):
            button = QPushButton(label, self)
            button.clicked.connect(funcs[i])
            if button.text() == "Clear":
                button.setEnabled(False)
            self._buttons.append(button)
            _row.addWidget(button)
        _layout.addLayout(_row)

        self._widget.setLayout(_layout)

    def _click2plot(self) -> None:
        """ Plot random data points """
        # Delete previous series
        self._chart.removeAllSeries()

        for i, line in enumerate(self._line_titles):
            series = QLineSeries()
            series.setName(line)
            data = self._line_data[i]
            for j, point in enumerate(data):
                series.append(j, point)
            self._chart.addSeries(series)

        self._chart.setTitle(" & ".join(self._line_titles))
        self._chart.createDefaultAxes()

        for button in self._buttons:
            if button.text() == "Clear":
                button.setEnabled(True)
        for button in self._buttons:
            if button.text() == "Plot":
                button.setEnabled(False)

    def _click2clear(self) -> None:
        """ Clear the chart """
        self._chart.setTitle("")
        self._chart.removeAllSeries()
        for axis in self._chart.axes():
            self._chart.removeAxis(axis)

        for button in self._buttons:
            if button.text() == "Clear":
                button.setEnabled(False)
        for button in self._buttons:
            if button.text() == "Plot":
                button.setEnabled(True)


def main() -> None:
    """ Main Function """
    app = QApplication(argv)
    window = MainWindow()
    window.show()
    exit(app.exec())


if __name__ == "__main__":
    main()
