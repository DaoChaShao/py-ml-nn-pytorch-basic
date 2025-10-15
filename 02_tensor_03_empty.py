#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/10/15 18:15
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   02_tensor_03_empty.py
# @Desc     :   

from random import randint
from torch import Tensor

from utils.decorator import beautifier


@beautifier
def empty_tensor():
    """ Create a tensor focusing torch size """
    dimensions: int = randint(1, 6)
    rows: int = randint(1, 6)
    cols: int = randint(1, 6)
    t = Tensor(dimensions, rows, cols)
    print(f"The empty tensor with size ({dimensions}, {rows}, {cols}) is:\n{t}")
    print(f"Tensor dimension is {t.dim()}")
    print(f"Tensor shape is {t.size()}")
    print(f"Tensor type is {t.dtype}")


def main() -> None:
    """ Main Function """
    empty_tensor()


if __name__ == "__main__":
    main()
