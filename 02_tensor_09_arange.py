#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/10/15 18:20
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   02_tensor_09_arange.py
# @Desc     :   

from random import randint
from torch import arange

from utils.decorator import beautifier


@beautifier
def range_tensor():
    """ Create a tensor using a range of numbers """
    start: int = randint(1, 11)
    end: int = randint(11, 21)
    step: int = randint(1, 3)
    t = arange(start, end, step)
    print(f"The range tensor from {start} to {end} with step {step} is:\n{t}")
    print(f"Tensor dimension is {t.dim()}")
    print(f"Tensor shape is {t.size()}")
    print(f"Tensor type is {t.dtype}")

    return t


def main() -> None:
    """ Main Function """
    range_tensor()


if __name__ == "__main__":
    main()
