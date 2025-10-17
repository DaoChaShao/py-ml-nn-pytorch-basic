#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/10/17 17:51
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   19_DP_03_module_02_sequential.py
# @Desc     :

from torch import nn, Tensor, randn, backends
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
        # 0 for column, 1 for row, -1 for last dimension
        return nn.functional.softmax(out, dim=-1)


def main() -> None:
    """ Main Function """
    with RandomSeed("Linear Module Sequential Example"):
        with Beautifier("Custom Linear Module Sequential Example"):
            # Set device
            # device: str = "cuda" if cuda.is_available() else "cpu"
            device: str = "mps" if backends.mps.is_available() else "cpu"
            # device = "cpu"

            # Create a random input tensor
            lines("Input Tensor Creation")
            x: Tensor = randn((2, 5), device=device)
            print(f"Input Tensor:\n{x}")

            # Create the model
            lines("Linear Module Sequential")
            model = LinearModule(x.shape[1], inner_units=3, outer_units=4, categories=3, device=device)
            print(model)

            # Forward pass
            lines("Random Output Tensor")
            out: Tensor = model(x)
            print(f"Output Tensor:\n{out}")

            # Get Predictive Class
            lines("Predictive Class")
            _, predicted = out.max(1)
            print(f"Predicted Class:\n{predicted}")


if __name__ == "__main__":
    main()
