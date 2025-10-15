#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/10/15 23:41
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   08_stats_04_max.py
# @Desc     :   

from torch import Tensor, randint, float64

from utils.decorator import beautifier


@beautifier
def max_tensor(t: Tensor, dims: int = -1, keep_dims: bool = False) -> Tensor:
    """ Maximum value of a tensor """
    max_val = None
    match dims:
        case 0:
            print(f"Input tensor:\n{t}")
            max_val = t.max(dim=dims, keepdim=keep_dims)
            print(f"Maximum value tensor along dimension {dims} (keepdim={keep_dims}):\n{max_val}")
        case 1:
            print(f"Input tensor:\n{t}")
            max_val = t.max(dim=dims, keepdim=keep_dims)
            print(f"Maximum value tensor along dimension {dims} (keepdim={keep_dims}):\n{max_val}")
        case 2:
            print(f"Input tensor:\n{t}")
            max_val = t.max(dim=dims, keepdim=keep_dims)
            print(f"Maximum value tensor along dimension {dims} (keepdim={keep_dims}):\n{max_val}")
        case _:
            print(f"Input tensor:\n{t}")
            max_val = t.max()
            print(f"Maximum value of all elements in the tensor:\n{max_val}")

    return max_val


def main() -> None:
    """ Main Function """
    t = randint(0, 10, (3, 4, 5), dtype=float64)
    # print(t)

    max_tensor(t)
    # dimensions, such as 3, does not exist
    max_tensor(t, dims=0, keep_dims=False)
    # rows, such as 5, does not exist, row elements max calculated
    max_tensor(t, dims=1, keep_dims=False)
    # columns, such as 4, does not exist, column elements max calculated
    max_tensor(t, dims=2, keep_dims=False)


if __name__ == "__main__":
    main()
