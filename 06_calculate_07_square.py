#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/10/15 22:01
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   06_calculate_07_square.py
# @Desc     :   

from random import randint
from torch import Tensor, arange

from utils.decorator import beautifier


@beautifier
def square_tensor(t: Tensor) -> Tensor:
    """ Square a tensor """
    print(f"Input tensor:\n{t}")
    t.square()
    print(f"Squared tensor:\n{t}")
    t.square_()
    print(f"Squared tensor:\n{t}")

    return t


def main() -> None:
    """ Main Function """
    start: int = randint(1, 11)
    end: int = randint(11, 21)
    step: int = randint(1, 4)
    t = arange(start, end, step)
    # print(t)

    square_tensor(t)


if __name__ == "__main__":
    main()
