#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/10/17 16:22
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   19_DP_02_dropout.py
# @Desc     :   

from torch import Tensor, randint, float32, nn

from utils.helper import Beautifier
from utils.highlighter import lines


def main() -> None:
    """ Main Function """
    with Beautifier("Dropout Layer Example"):
        # Create a tensor
        x: Tensor = randint(1, 10, (10,), dtype=float32)
        print(f"Original Tensor:\n{x}")
        print(f"Requires Grad: {x.requires_grad}")

        # Apply Dropout
        lines("Dropout Layer")
        dropout = nn.Dropout(p=0.5)
        output: Tensor = dropout(x)
        print(f"Tensor after Dropout:\n{output}")
        print(f"Requires Grad: {output.requires_grad}")


if __name__ == "__main__":
    main()
