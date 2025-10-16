#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/10/16 11:28
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   09_index_01.py
# @Desc     :   

from random import randint
from torch import Tensor, randint as py_randint, float64
from typing import Any

from utils.decorator import beautifier


@beautifier
def example(index: Any, low: int = 1, high: int = 10, dims: int = 2, rows: int = 3, cols: int = 4) -> None:
    """ Example Function """
    print(f"Index: {index}")

    t: Tensor = py_randint(low, high, (dims, rows, cols), dtype=float64)
    print(f"Full tensor shape:\n{t}")
    print(f"The selected element(s):\n{t[index]}")


def main() -> None:
    """ Main Function """
    low: int = 1
    high: int = 10
    dims: int = 2
    rows: int = 3
    cols: int = 4

    index_dim: int = randint(0, dims - 1)
    index_row: int = randint(0, rows - 1)
    index_col: int = randint(0, cols - 1)
    index: tuple[int, int, int] = (index_dim, index_row, index_col)

    example((index_dim,))
    example((index_dim, index_row))
    example(index, low, high, dims, rows, cols)

    # get the 2nd row in the whole tensor
    example((slice(None), 1, slice(None)), low, high, dims, rows, cols)


if __name__ == "__main__":
    main()
