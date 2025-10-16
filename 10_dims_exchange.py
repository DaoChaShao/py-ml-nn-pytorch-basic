#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/10/16 12:54
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   10_dims_exchange.py
# @Desc     :   

from torch import Tensor, randint as py_randint, float64

from utils.decorator import beautifier


@beautifier
def example() -> Tensor:
    """ Example Function """
    t: Tensor = py_randint(1, 10, (2, 3, 6), dtype=float64)
    print(f"Full tensor shape:\n{t}")
    print(f"Full tensor shape: {t.shape}")
    return t


def main() -> None:
    """ Main Function """
    e = example()

    # dim means index of shape (2, 3, 6) - 0, 1, 2
    print(e.transpose(1, 2))


if __name__ == "__main__":
    main()
