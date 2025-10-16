#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/10/16 16:38
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   17_auto_grad_02_zero.py
# @Desc     :   

from asyncio import gather, run
from torch import Tensor, tensor, rand, float64, nn

from utils.highlighter import green, red
from utils.helper import RandomSeed, Beautifier


async def gradients_with_zero(X: Tensor, y: Tensor, W: Tensor, b: Tensor, epoch: int = 5) -> None:
    """ Demonstrate the effect of using zeroing gradients """
    for epoch in range(epoch):
        # Forward pass: compute the predicted y
        Z: Tensor = X * W + b
        # with Beautifier("Predicted Tensor"):
        #     print(f"Predicted Tensor: {Z}")

        # Compute and print the loss
        loss_fn: nn.MSELoss = nn.MSELoss()
        loss: Tensor = loss_fn(Z, y)
        # with Beautifier("Loss Tensor"):
        #     print(f"Loss Tensor: {loss}")

        # Check whether the dots are leaves dots
        # with Beautifier("Leaf Tensors"):
        #     print(f"Is X a leaf tensor? {green("True") if X.is_leaf else red("False")}")
        #     print(f"Is y a leaf tensor? {green("True") if y.is_leaf else red("False")}")
        #     print(f"Is W a leaf tensor? {green("True") if W.is_leaf else red("False")}")
        #     print(f"Is b a leaf tensor? {green("True") if b.is_leaf else red("False")}")
        #     print(f"Is y_hat a leaf tensor? {green("True") if Z.is_leaf else red("False")}")
        #     print(f"Is loss a leaf tensor? {green("True") if loss.is_leaf else red("False")}")

        # Perform backpropagation to compute gradients
        loss.backward()

        # Zero the gradients before the next iteration
        W.grad.zero_()
        b.grad.zero_()
        X.grad.zero_()
        y.grad.zero_()

    with Beautifier("Gradients with Zeroing"):
        print(f"Gradient of W: {W.grad}")
        print(f"Gradient of b: {b.grad}")
        print(f"Gradient of X: {X.grad}")
        print(f"Gradient of y: {y.grad}")


async def gradents_without_zero(X: Tensor, y: Tensor, W: Tensor, b: Tensor, epoch: int = 5) -> None:
    """ Demonstrate the effect of NOT using zeroing gradients """
    for epoch in range(epoch):
        # Forward pass: compute the predicted y
        Z: Tensor = X * W + b
        # with Beautifier("Predicted Tensor"):
        #     print(f"Predicted Tensor: {Z}")

        # Compute and print the loss
        loss_fn: nn.MSELoss = nn.MSELoss()
        loss: Tensor = loss_fn(Z, y)
        # with Beautifier("Loss Tensor"):
        #     print(f"Loss Tensor: {loss}")

        # Check whether the dots are leaves dots
        # with Beautifier("Leaf Tensors"):
        #     print(f"Is X a leaf tensor? {green("True") if X.is_leaf else red("False")}")
        #     print(f"Is y a leaf tensor? {green("True") if y.is_leaf else red("False")}")
        #     print(f"Is W a leaf tensor? {green("True") if W.is_leaf else red("False")}")
        #     print(f"Is b a leaf tensor? {green("True") if b.is_leaf else red("False")}")
        #     print(f"Is y_hat a leaf tensor? {green("True") if Z.is_leaf else red("False")}")
        #     print(f"Is loss a leaf tensor? {green("True") if loss.is_leaf else red("False")}")

        # Perform backpropagation to compute gradients
        loss.backward()

    with Beautifier("Gradients without Zeroing"):
        print(f"Gradient of W: {W.grad}")
        print(f"Gradient of b: {b.grad}")
        print(f"Gradient of X: {X.grad}")
        print(f"Gradient of y: {y.grad}")


async def main() -> None:
    """ Main Function """
    with RandomSeed("Simple Network"):
        # Define a tensor as a feature
        X: Tensor = tensor(10, dtype=float64, requires_grad=True)
        y: Tensor = tensor([[3]], dtype=float64, requires_grad=True)
        with Beautifier("Feature and Target Tensors"):
            print(f"Feature Tensor: {X}")
            print(f"Target Tensor: {y}")

        # Define the weights and bias
        W: Tensor = rand(1, 1, dtype=float64, requires_grad=True)
        b: Tensor = rand(1, 1, dtype=float64, requires_grad=True)
        with Beautifier("Weights and Bias Tensors"):
            print(f"Weights Tensor: {W}")
            print(f"Bias Tensor: {b}")

    tasks = [
        gradients_with_zero(
            X.detach().clone().requires_grad_(True),
            y.detach().clone().requires_grad_(True),
            W.detach().clone().requires_grad_(True),
            b.detach().clone().requires_grad_(True)
        ),
        gradents_without_zero(
            X.detach().clone().requires_grad_(True),
            y.detach().clone().requires_grad_(True),
            W.detach().clone().requires_grad_(True),
            b.detach().clone().requires_grad_(True)
        )
    ]
    await gather(*tasks)


if __name__ == "__main__":
    run(main())
