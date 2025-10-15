#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/10/15 22:53
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   06_calculate_03_mul.py
# @Desc     :   

from torch import Tensor, tensor

from utils.decorator import beautifier


@beautifier
def multiply_tensors(a: Tensor, b: Tensor) -> Tensor:
    """ Multiply two tensors """
    print(f"Tensor a:\n{a}")
    print(f"Tensor b:\n{b}")
    a.mul(b)
    print(f"Result of a.mul(b):\n{a}")
    a.mul_(b)
    print(f"Tensor a after in-place multiplication (a.mul_(b)):\n{a}")

    return a


def main() -> None:
    """ Main Function """
    a = tensor([5, 10, 15, 20])
    b = tensor([1, 2, 3, 4])
    # print(f"a: {a}")
    # print(f"b: {b}")

    multiply_tensors(a, b)


if __name__ == "__main__":
    main()
