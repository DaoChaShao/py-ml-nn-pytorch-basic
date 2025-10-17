#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/10/17 13:29
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   19_DP_01_params_init_01_zero.py
# @Desc     :   

from torch import nn

from utils.helper import Beautifier
from utils.highlighter import lines


def main() -> None:
    """ Main Function """
    with Beautifier("Parameters Initialization"):
        model = nn.Sequential(
            nn.Linear(10, 2),
        )

        print("Model before initialization:")
        for name, param in model.named_parameters():
            print(f"{name}:\nparams={param.data},\nmean={param.data.mean():.4f}, std={param.data.std():.4f}")

        # Initialize parameters
        lines("Initialize parameters")
        for layer in model:
            if isinstance(layer, nn.Linear):
                nn.init.zeros_(layer.weight)
                nn.init.zeros_(layer.bias)

        print("Model after initialization:")
        for name, param in model.named_parameters():
            print(f"{name}:\nparams={param.data},\nmean={param.data.mean():.4f}, std={param.data.std():.4f}")


if __name__ == "__main__":
    main()
