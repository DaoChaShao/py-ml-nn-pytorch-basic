#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/10/17 16:31
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   19_DP_03_module_01_forward.py
# @Desc     :

from torch import nn, Tensor, randn, backends, cuda
from torchsummary import summary

from utils.helper import Beautifier, RandomSeed
from utils.highlighter import lines


class LinearModule(nn.Module):
    def __init__(self, features: int, inner_units: int, outer_units: int, categories: int, device: str) -> None:
        super(LinearModule, self).__init__()
        self._inner = nn.Linear(features, inner_units, device=device)
        self._outer = nn.Linear(inner_units, outer_units, device=device)
        self._output = nn.Linear(outer_units, categories, device=device)

        nn.init.xavier_normal_(self._inner.weight)
        nn.init.kaiming_uniform_(self._outer.weight)

    def forward(self, x: Tensor) -> Tensor:
        out = self._inner(x)
        out = nn.functional.tanh(out)
        out = self._outer(out)
        out = nn.functional.relu(out)
        out = self._output(out)
        out = nn.functional.softmax(out, dim=-1)
        return out


def main() -> None:
    """ Main Function """
    with RandomSeed("Linear Module Example"):
        with Beautifier("Custom Linear Module Example"):
            # Set device
            # device: str = "cuda" if cuda.is_available() else "cpu"
            device: str = "mps" if backends.mps.is_available() else "cpu"
            # device = "cpu"

            # Create a random input tensor
            x: Tensor = randn((2, 5), device=device)
            print(f"Input Tensor:\n{x}")

            # Create the model
            lines("Linear Module")
            model = LinearModule(x.shape[1], inner_units=3, outer_units=4, categories=3, device=device)
            print(model)

            # Forward pass
            lines("Random Output Tensor")
            out: Tensor = model(x)
            print(f"Output Tensor:\n{out}")

            # Check parameters using named_parameters
            lines("Parameters in the Module")
            for name, param in model.named_parameters():
                print(f"{name}:\nparams={param.data}")
                lines()

            # Check parameters using state_dict
            lines("State Dict Parameters in the Module")
            print(model.state_dict())

            # Check the model structure using torchsummary
            # - pip install torchsummary or uv add torchsummary to install
            # - pip show torchsummary or uv tree to check installation
            # - summary works only on CPU and GPU devices
            # lines("State Model Summary Using torchsummary")
            # summary(model, input_size=(x.shape[1],), batch_size=x.shape[0], device=device)


if __name__ == "__main__":
    main()
