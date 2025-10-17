#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/10/17 19:17
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   19_DP_04_loss_02_CCE.py
# @Desc     :   

from torch import nn, Tensor, randn, backends, float32, randint, int64

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
        return self._model(x)


def main() -> None:
    """ Main Function """
    with RandomSeed("BCELoss Example"):
        with Beautifier("Binary Cross Entropy Loss Example"):
            # Set device
            device: str = "mps" if backends.mps.is_available() else "cpu"

            # Create a dummy input tensor and target tensor
            lines("Input and Target Tensor")
            x: Tensor = randn((6, 5), device=device, dtype=float32)
            y: Tensor = randint(0, 3, (6,), device=device, dtype=int64)
            print(f"Input Tensor:\n{x}")
            print(f"Target Tensor:\n{y}")

            # Create the model
            lines("Linear Module Sequential")
            model = LinearModule(x.shape[1], inner_units=4, outer_units=5, categories=y.max().item() + 1, device=device)
            print(model)

            # Forward pass
            lines("Forward Pass Directly")
            outputs: Tensor = model(x)
            print(f"Model Outputs:\n{outputs}")

            # Define CCELoss
            criterion = nn.CrossEntropyLoss()
            loss: Tensor = criterion(outputs, y)
            print(f"Cross Entropy Loss: {loss.item():.4f}")

            # Get predicted classes
            # - _ means "values" of max logits
            # - predicted means "indices" of max logits
            lines("Predicted Classes")
            _, predicted = outputs.max(1)
            print(f"Predicted Classes:\n{predicted}")


if __name__ == "__main__":
    main()
