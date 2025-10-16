#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/10/16 14:20
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   13_cat.py
# @Desc     :   

from torch import Tensor, tensor, ones, cat, randint as py_randint, float64

from utils.decorator import beautifier
from utils.helper import Beautifier
from utils.highlighter import green, red, lines


@beautifier
def cat_tensors_col():
    """ Cat Tensors by Column """
    out = ones(size=(4, 2))
    print(f"out_tensor: \n{green(out)}")
    print(f"out_tensor.shape: {out.shape}")

    left = tensor([[1, 2], [3, 4]])
    right = tensor([[5, 6], [7, 8]])

    # cat tensors by column
    dim_col = 0

    cat(
        tensors=(left, right),
        dim=dim_col,
        out=out,
    )

    print(f"out_tensor: \n{red(out)}")
    print(f"out_tensor.shape: {out.shape}")


@beautifier
def cat_tensors_row():
    """ Cat Tensors by Row """
    out = ones(size=(2, 4))
    print(f"out_tensor: \n{green(out)}")
    print(f"out_tensor.shape: {out.shape}")

    left = tensor([[1, 2], [3, 4]])
    right = tensor([[5, 6], [7, 8]])

    # cat tensors by row
    dim_row = 1

    cat(
        tensors=(left, right),
        dim=dim_row,
        out=out,
    )

    print(f"out_tensor: \n{red(out)}")
    print(f"out_tensor.shape: {out.shape}")


def example_a() -> Tensor:
    """ Example Function """
    t: Tensor = py_randint(1, 10, (2, 3, 6), dtype=float64)
    return t


def example_b() -> Tensor:
    """ Example Function """
    t: Tensor = py_randint(1, 10, (2, 5, 6), dtype=float64)
    return t


def main() -> None:
    """ Main Function """
    cat_tensors_col()
    cat_tensors_row()

    a = example_a()
    b = example_b()
    with Beautifier("Cat two tensors"):
        print(f"Tensor a shape:\n{a} with shape {a.shape}")
        print(f"Tensor b shape:\n{b} with shape {b.shape}")

        lines("cat by dim=1")
        # Due to the second dimension being different, we can only concatenate along dim=1
        c1: Tensor = cat((a, b), dim=1)
        print(f"Cat tensor shape by dim=1:\n{c1} with shape {c1.shape}")


if __name__ == "__main__":
    main()
