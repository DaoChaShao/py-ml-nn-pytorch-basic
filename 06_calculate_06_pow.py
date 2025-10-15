#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/10/15 22:01
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   06_calculate_06_pow.py
# @Desc     :   

from random import randint
from torch import Tensor, arange

from utils.decorator import beautifier


@beautifier
def pow_tensor(t: Tensor, exponent: int) -> Tensor:
    """ Raise a tensor to the power of exponent """
    print(f"Input tensor:\n{t}")
    t.pow(exponent)
    print(f"Tensor raised to the power of {exponent}:\n{t}")
    t.pow_(exponent)
    print(f"Tensor raised to the power of {exponent}:\n{t}")

    return t


def main() -> None:
    """ Main Function """
    start: int = randint(1, 11)
    end: int = randint(11, 21)
    step: int = randint(1, 4)
    t = arange(start, end, step)
    n = randint(1, 10)
    # print(t)
    # print(n)

    pow_tensor(t, n)


if __name__ == "__main__":
    main()
