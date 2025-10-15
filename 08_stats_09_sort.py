#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/10/16 00:00
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   08_stats_09_sort.py
# @Desc     :   

from torch import Tensor, randint, float64

from utils.decorator import beautifier


@beautifier
def sort_tensor(t: Tensor, dims: int = -1, descending: bool = False) -> Tensor:
    """ Sort elements of a tensor """
    result = None
    match dims:
        case 0:
            print(f"Input tensor:\n{t}")
            result, indices = t.sort(dim=0, descending=descending)
            print(f"Sorted tensor along dim 0:\n{result}")
        case 1:
            print(f"Input tensor:\n{t}")
            result, indices = t.sort(dim=1, descending=descending)
            print(f"Sorted tensor along dim 1:\n{result}")
        case 2:
            print(f"Input tensor:\n{t}")
            result, indices = t.sort(dim=2, descending=descending)
            print(f"Sorted tensor along dim 2:\n{result}")
        case _:
            print(f"Input tensor:\n{t}")
            result, indices = t.sort()
            print(f"Sorted tensor:\n{result}")

    return result


def main() -> None:
    """ Main Function """
    t = randint(0, 10, (3, 4, 5), dtype=float64)
    # print(t)

    sort_tensor(t)
    # Sort along dim 0
    sort_tensor(t, dims=0, descending=True)
    # Sort along dim 1
    sort_tensor(t, dims=1, descending=True)
    # Sort along dim 2
    sort_tensor(t, dims=2, descending=True)


if __name__ == "__main__":
    main()
