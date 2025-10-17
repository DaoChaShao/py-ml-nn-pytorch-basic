#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/10/17 18:47
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   19_DP_04_loss_01_BCE.py
# @Desc     :   

from torch import nn, Tensor, randn, backends, tensor, float32
from torchsummary import summary

from utils.helper import Beautifier, RandomSeed
from utils.highlighter import lines


class LinearModule(nn.Module):
    def __init__(self, features: int, inner_units: int, outer_units: int, categories: int, device: str) -> None:
        super(LinearModule, self).__init__()
        self._model = nn.Sequential(
            nn.Linear(features, inner_units, device=device),
            nn.Tanh(),
            nn.Linear(inner_units, outer_units, device=device),
            nn.ReLU(),
            nn.Linear(outer_units, categories, device=device),
        )

        self._model.apply(self._parameters_initializer)

    @staticmethod
    def _parameters_initializer(layer) -> None:
        if isinstance(layer, nn.Linear):
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x: Tensor) -> Tensor:
        out = self._model(x)
        return nn.functional.sigmoid(out)


def main() -> None:
    """ Main Function """
    with RandomSeed("BCELoss Example"):
        with Beautifier("Binary Cross Entropy Loss Example"):
            # Set device
            device: str = "mps" if backends.mps.is_available() else "cpu"

            # Create a dummy input tensor and target tensor
            lines("Input and Target Tensor")
            x: Tensor = randn((3, 2), device=device, dtype=float32)
            y: Tensor = tensor([[0, 1], [1, 0], [0, 1]], device=device, dtype=float32)
            print(f"Input Tensor:\n{x}")
            print(f"Target Tensor:\n{y}")

            # Create the model
            lines("Linear Module Sequential")
            model = LinearModule(x.shape[1], inner_units=3, outer_units=4, categories=y.shape[1], device=device)
            print(model)

            # Forward pass
            lines("Forward Pass Directly")
            output = model(x)

            # Define BCELoss
            # - y must be the One-Hot Encoding format with float dtype if you plan to use BCELoss
            criterion = nn.BCELoss()
            loss = criterion(output, y)
            print(f"Output Tensor after Sigmoid:\n{output}")
            print(f"BCELoss: {loss.item():.4f}")

            # Get Predicted classes
            lines("Predicted Classes")
            predicted = (output > 0.5).float()
            print(f"Predicted Classes (Threshold=0.5):\n{predicted}")


if __name__ == "__main__":
    main()
