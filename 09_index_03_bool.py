#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/10/16 12:35
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   09_index_03_bool.py
# @Desc     :   

from random import randint
from torch import Tensor, randint as py_randint, float64
from typing import Any

from utils.decorator import beautifier


@beautifier
def bool_tensor(
        threshold: float,
        dim: int = slice(None), row: int = slice(None), col: int = slice(None),
        low: int = 1, high: int = 10,
        dims: int = 2, rows: int = 3, cols: int = 4
) -> None:
    """ Example Function """
    print(f"Threshold: {threshold}")
    print(f"Indexing at dim={dim}, row={row}, col={col}")

    t: Tensor = py_randint(low, high, (dims, rows, cols), dtype=float64)
    print(f"Full tensor shape:\n{t}")

    mask = t[dim, row, col] > threshold
    print(f"Mask:\n{mask}")

    result: Any = t[dim, row, col][mask]
    print(f"Result:\n{result}")
    print(f"Result shape: {result.shape}")


def main() -> None:
    """ Main Function """
    low: int = 1
    high: int = 10
    dims: int = 2
    rows: int = 3
    cols: int = 4
    threshold: float = float(randint(low, high))

    index_dim: int = randint(0, dims - 1)
    index_row: int = randint(0, rows - 1)
    index_col: int = randint(0, cols - 1)

    bool_tensor(
        threshold=threshold,
        dim=index_dim,
        # row=index_row,
        # col=index_col,
        low=low, high=high,
        dims=dims, rows=rows, cols=cols
    )


if __name__ == "__main__":
    main()
