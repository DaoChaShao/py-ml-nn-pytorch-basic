#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/10/15 21:43
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   06_calculate_01_add.py
# @Desc     :   

from random import randint
from torch import Tensor, int64
from typing import Union

from utils.decorator import beautifier


@beautifier
def addition(t1: Union[Tensor, int64], t2: Union[Tensor, int64]) -> Union[Tensor, int64]:
    """ Add two tensors """
    print(f"Input 1:\n{t1}")
    print(f"Input 2:\n{t2}")
    # not in-place addition using add
    t1.add(t2)
    print(f"Sum of tensors:\n{t1}")
    # In-place addition using add_
    t1.add_(t2)
    print(f"Sum of tensors:\n{t1}")

    return t1


def main() -> None:
    """ Main Function """
    dims: int = randint(1, 4)
    rows: int = randint(1, 5)
    cols: int = randint(1, 5)
    t = Tensor(dims, rows, cols)
    n = randint(1, 10)
    # print(t)
    # print(n)

    addition(t, n)


if __name__ == "__main__":
    main()
