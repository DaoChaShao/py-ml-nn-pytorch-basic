#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/10/15 22:27
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   06_calculate_08_log.py
# @Desc     :   

from random import randint
from torch import Tensor, arange, float64

from utils.decorator import beautifier


@beautifier
def logarithm_tensor(t: Tensor) -> Tensor:
    """ Logarithm of a tensor """
    print(f"Input tensor:\n{t}")
    t.log()
    print(f"Logarithm tensor:\n{t}")
    t.log_()
    print(f"Logarithm tensor:\n{t}")

    return t


def main() -> None:
    """ Main Function """
    start: int = randint(1, 11)
    end: int = randint(11, 21)
    step: int = randint(1, 4)
    t = arange(start, end, step, dtype=float64)
    # print(t)

    logarithm_tensor(t)


if __name__ == "__main__":
    main()
