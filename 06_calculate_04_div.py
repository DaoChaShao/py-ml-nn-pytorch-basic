#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/10/15 22:54
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   06_calculate_04_div.py
# @Desc     :   

from torch import Tensor, tensor, float64

from utils.decorator import beautifier


@beautifier
def divide_tensors(a: Tensor, b: Tensor) -> Tensor:
    """ Divide two tensors """
    print(f"Tensor a:\n{a}")
    print(f"Tensor b:\n{b}")
    a.div(b)
    print(f"Result of a.div(b):\n{a}")
    a.div_(b)
    print(f"Tensor a after in-place division (a.div_(b)):\n{a}")

    return a


def main() -> None:
    """ Main Function """
    a = tensor([5, 10, 15, 20], dtype=float64)
    b = tensor([1, 2, 3, 4], dtype=float64)
    # print(f"a: {a}")
    # print(f"b: {b}")

    divide_tensors(a, b)


if __name__ == "__main__":
    main()
