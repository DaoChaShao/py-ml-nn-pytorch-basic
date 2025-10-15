#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/10/15 18:22
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   02_tensor_10_linear_space.py
# @Desc     :   

from random import randint
from torch import linspace, strided

from utils.decorator import beautifier


@beautifier
def linear_space_tensor():
    """ Create a tensor using linear space """
    start: int = randint(11, 21)
    end: int = randint(21, 31)
    steps: int = randint(1, 11)
    t = linspace(
        start,
        end,
        steps,
        dtype=None,
        layout=strided,
        device=None,
        requires_grad=False,
    )
    print(f"The linear space tensor from {start} to {end} with {steps} steps is:\n{t}")
    print(f"Tensor dimension is {t.dim()}")
    print(f"Tensor shape is {t.size()}")
    print(f"Tensor type is {t.dtype}")

    return t


def main() -> None:
    """ Main Function """
    linear_space_tensor()


if __name__ == "__main__":
    main()
