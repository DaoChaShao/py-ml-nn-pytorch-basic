#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/10/15 23:58
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   08_stats_08_unique.py
# @Desc     :   

from torch import Tensor, randint, float64

from utils.decorator import beautifier


@beautifier
def unique_tensor(t: Tensor) -> Tensor:
    """ Unique elements of a tensor """
    print(f"Input tensor:\n{t}")
    result = t.unique()
    print(f"Unique elements tensor:\n{result}")

    return result


def main() -> None:
    """ Main Function """
    t = randint(0, 10, (3, 4, 5), dtype=float64)
    # print(t)

    unique_tensor(t)


if __name__ == "__main__":
    main()
