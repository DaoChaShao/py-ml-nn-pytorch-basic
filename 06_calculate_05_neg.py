#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/10/15 22:00
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   06_calculate_05_neg.py
# @Desc     :   

from random import randint
from torch import Tensor, int64
from typing import Union

from utils.decorator import beautifier


@beautifier
def negative_tensor(t: Tensor) -> Tensor:
    """ Negate a tensor """
    print(f"Input tensor:\n{1}")
    t.neg()
    print(f"Negated tensor:\n{t}")
    t.neg_()
    print(f"Negated tensor:\n{t}")

    return t


def main() -> None:
    """ Main Function """
    dims: int = randint(1, 4)
    rows: int = randint(1, 5)
    cols: int = randint(1, 5)
    t = Tensor(dims, rows, cols)
    n = randint(1, 10)
    # print(t)
    # print(n)

    negative_tensor(t)


if __name__ == "__main__":
    main()
