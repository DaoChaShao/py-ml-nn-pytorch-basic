#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/10/17 20:14
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   19_DP_05_optimiser_01_momentum.py
# @Desc     :   

from torch import Tensor, tensor, float32, optim

from utils.helper import RandomSeed
from utils.highlighter import lines


def f(X):
    return 0.05 * X[0] ** 2 + X[1] ** 2


def grad_descent(X: Tensor, optimiser, epochs: int, batches: int) -> list:
    """ Gradient Descent Optimizer """
    coordinates = list()
    for e in range(epochs):
        for b in range(batches):
            optimiser.zero_grad()
            out = f(X)
            out.backward()
            optimiser.step()

            coordinates.append(X.detach().numpy())

    return coordinates


def main() -> None:
    """ Main Function """
    with RandomSeed("Optimiser with Momentum"):
        # Set device
        device = "cpu"

        # Initial point
        lines("Define Initial Point")
        X: Tensor = tensor([-7.0, 2.0], device=device, dtype=float32, requires_grad=True)
        print(X)

        # Optimizer with Momentum
        lines("Define Optimizers")
        alpha = 0.01
        momentum = 0.9
        params_SGD = X.clone().detach().requires_grad_(True)
        params_MOM = X.clone().detach().requires_grad_(True)
        optimiser_SGD = optim.SGD([params_SGD], lr=alpha)
        optimiser_MOM = optim.SGD([params_MOM], lr=alpha, momentum=momentum)

        # Gradient Descent with SGD or Momentum
        epochs = 50
        batches = 16
        cc_SGD = grad_descent(params_SGD, optimiser_SGD, epochs, batches)
        cc_MOM = grad_descent(params_MOM, optimiser_MOM, epochs, batches)
        print(cc_SGD)
        print(cc_MOM)


if __name__ == "__main__":
    main()
