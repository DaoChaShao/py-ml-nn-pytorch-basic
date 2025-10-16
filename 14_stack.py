#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/10/16 14:35
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   14_stack.py
# @Desc     :   

from torch import ones, tensor, stack

from utils.decorator import beautifier
from utils.highlighter import green, red


@beautifier
def stack_tensors_col():
    """ Stack Tensors, Increase Dimension """
    out = ones(size=(2, 2, 2))
    print(f"out_tensor: \n{green(out)}")
    print(f"out_tensor.shape: {out.shape}")

    left = tensor([[1, 2], [3, 4]])
    right = tensor([[5, 6], [7, 8]])

    dim_col = 0

    stack(
        tensors=(left, right),
        dim=dim_col,
        out=out,
    )

    print(f"out_tensor: \n{red(out)}")
    print(f"out_tensor.shape: {out.shape}")


@beautifier
def stack_tensors_row():
    """ Stack Tensors, Increase Dimension """
    out = ones(size=(2, 2, 2))
    print(f"out_tensor: \n{green(out)}")
    print(f"out_tensor.shape: {out.shape}")

    left = tensor([[1, 2], [3, 4]])
    right = tensor([[5, 6], [7, 8]])

    dim_row = 1

    # 拼接张量
    stack(
        tensors=(left, right),
        dim=dim_row,
        out=out,
    )

    print(f"out_tensor: \n{red(out)}")
    print(f"out_tensor.shape: {out.shape}")


def main() -> None:
    """ Main Function """
    stack_tensors_col()
    stack_tensors_row()


if __name__ == "__main__":
    main()
