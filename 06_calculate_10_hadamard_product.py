#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/10/15 22:30
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   06_calculate_10_hadamard_product.py
# @Desc     :   

from torch import Tensor

from utils.decorator import beautifier


@beautifier
def hadamard_product(t1: Tensor, t2: Tensor) -> Tensor:
    """ Hadamard Product of two tensors """
    print(f"Tensor 1:\n{t1}")
    print(f"Tensor 2:\n{t2}")

    t3 = t1 * t2
    print(f"Hadamard Product (t1 * t2):\n{t3}")

    t4 = t1.mul(t2)
    print(f"Hadamard Product using mul (t1.mul(t2)):\n{t4}")

    t5 = t1.mul_(t2)
    print(f"Hadamard Product using mul_ (t1.mul_(t2)):\n{t5}")

    print(f"Tensor 1:\n{t1}")
    print(f"Tensor 2:\n{t2}")

    return t3


def main() -> None:
    """ Main Function """
    t1 = Tensor([[1, 2], [3, 4]])
    t2 = t1.clone()
    # print(t1, id(t1))
    # print(t2, id(t2))

    hadamard_product(t1, t2)


if __name__ == "__main__":
    main()
