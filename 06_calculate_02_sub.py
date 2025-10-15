#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/10/15 22:50
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   06_calculate_02_sub.py
# @Desc     :   

from torch import Tensor, tensor

from utils.decorator import beautifier


@beautifier
def subtract_tensors(a: Tensor, b: Tensor) -> Tensor:
    """ Subtract two tensors """
    print(f"Tensor a:\n{a}")
    print(f"Tensor b:\n{b}")
    a.sub(b)
    print(f"Result of a.sub(b):\n{a}")
    a.sub_(b)
    print(f"Tensor a after in-place subtraction (a.sub_(b)):\n{a}")

    return a


def main() -> None:
    """ Main Function """
    a = tensor([5, 10, 15, 20])
    b = tensor([1, 2, 3, 4])
    # print(f"a: {a}")
    # print(f"b: {b}")

    subtract_tensors(a, b)


if __name__ == "__main__":
    main()
