#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/10/15 11:24
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   02_tensor_01_scalar.py
# @Desc     :   

from random import randint
from torch import Tensor, tensor

from utils.decorator import beautifier


@beautifier
def scalar_tensor() -> Tensor:
    """ Create a tensor from a number """
    num = randint(1, 11)
    t = tensor(num)
    print(f"The random number is {num}.")
    print(f"The tensor is {t}.")
    print(f"Tensor dimension is {t.dim()}")
    print(f"Tensor shape is {t.size()}")
    print(f"Tensor type is {t.dtype}")

    return t


def main() -> None:
    """ Main Function """
    scalar_tensor()


if __name__ == "__main__":
    main()
