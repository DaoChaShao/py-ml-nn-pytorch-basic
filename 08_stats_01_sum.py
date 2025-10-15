#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/10/15 23:17
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   08_stats_01_sum.py
# @Desc     :   

from torch import Tensor, randint

from utils.decorator import beautifier


@beautifier
def sum_tensor(t: Tensor, dims: int = -1, keep_dim: bool = False) -> Tensor:
    """ Sum of tensor elements over a given dimension """
    result = None
    match dims:
        case 0:
            print(f"Input tensor:\n{t}")
            result = t.sum(dim=dims, keepdim=keep_dim)
            print(f"Summed tensor along dimension {dims} (keepdim={keep_dim}):\n{result}")
        case 1:
            print(f"Input tensor:\n{t}")
            result = t.sum(dim=dims, keepdim=keep_dim)
            print(f"Summed tensor along dimension {dims} (keepdim={keep_dim}):\n{result}")
        case 2:
            print(f"Input tensor:\n{t}")
            result = t.sum(dim=dims, keepdim=keep_dim)
            print(f"Summed tensor along dimension {dims} (keepdim={keep_dim}):\n{result}")
        case _:
            print(f"Input tensor:\n{t}")
            result = t.sum()
            print(f"Summed all elements in the tensor:\n{result}")

    return result


def main() -> None:
    """ Main Function """
    t = randint(0, 10, (3, 4, 5))
    # print(t)

    sum_tensor(t)
    # dimensions, such as 3, does not exist
    sum_tensor(t, dims=0, keep_dim=False)
    # rows, such as 5, does not exist, row elements added
    sum_tensor(t, dims=1, keep_dim=False)
    # columns, such as 4, does not exist, column elements added
    sum_tensor(t, dims=2, keep_dim=False)


if __name__ == "__main__":
    main()
