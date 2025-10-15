#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/10/15 23:49
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   08_stats_06_argmin.py
# @Desc     :   

from torch import Tensor, randint, float64

from utils.decorator import beautifier


@beautifier
def argmin_tensor(t: Tensor, dims: int = -1, keep_dim: bool = False) -> Tensor:
    """ Argmin of a tensor """
    result = None
    match dims:
        case 0:
            print(f"Input tensor:\n{t}")
            result = t.argmin(dim=0, keepdim=keep_dim)
            print(f"Argmin tensor along dim 0:\n{result}")
        case 1:
            print(f"Input tensor:\n{t}")
            result = t.argmin(dim=1, keepdim=keep_dim)
            print(f"Argmin tensor along dim 1:\n{result}")
        case 2:
            print(f"Input tensor:\n{t}")
            result = t.argmin(dim=2, keepdim=keep_dim)
            print(f"Argmin tensor along dim 2:\n{result}")
        case _:
            print(f"Input tensor:\n{t}")
            result = t.argmin()
            print(f"Argmin tensor:\n{result}")

    return result


def main() -> None:
    """ Main Function """
    t = randint(0, 10, (3, 4, 5), dtype=float64)
    # print(t)

    argmin_tensor(t)
    # Find the min along dim 0
    argmin_tensor(t, dims=0, keep_dim=True)
    # Find the min along dim 1
    argmin_tensor(t, dims=1, keep_dim=False)
    # Find the min along dim 2
    argmin_tensor(t, dims=2, keep_dim=True)


if __name__ == "__main__":
    main()
