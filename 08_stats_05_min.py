#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/10/15 23:45
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   08_stats_05_min.py
# @Desc     :   

from torch import Tensor, randint, float64

from utils.decorator import beautifier


@beautifier
def min_tensor(t: Tensor, dims: int = -1, keep_dim: bool = False) -> Tensor:
    """ Minimum value of a tensor """
    min_val = None
    match dims:
        case 0:
            print(f"Input tensor:\n{t}")
            min_val = t.min(dim=dims, keepdim=keep_dim)
            print(f"Minimum value tensor along dimension {dims} (keepdim={keep_dim}):\n{min_val}")
        case 1:
            print(f"Input tensor:\n{t}")
            min_val = t.min(dim=dims, keepdim=keep_dim)
            print(f"Minimum value tensor along dimension {dims} (keepdim={keep_dim}):\n{min_val}")
        case 2:
            print(f"Input tensor:\n{t}")
            min_val = t.min(dim=dims, keepdim=keep_dim)
            print(f"Minimum value tensor along dimension {dims} (keepdim={keep_dim}):\n{min_val}")
        case _:
            print(f"Input tensor:\n{t}")
            min_val = t.min()
            print(f"Minimum value of all elements in the tensor:\n{min_val}")

    return min_val


def main() -> None:
    """ Main Function """
    t = randint(0, 10, (3, 4, 5), dtype=float64)
    # print(t)

    min_tensor(t)
    # dimensions, such as 3, does not exist
    min_tensor(t, dims=0, keep_dim=False)
    # rows, such as 5, does not exist, row elements min calculated
    min_tensor(t, dims=1, keep_dim=False)
    # columns, such as 4, does not exist, column elements min calculated
    min_tensor(t, dims=2, keep_dim=False)


if __name__ == "__main__":
    main()
