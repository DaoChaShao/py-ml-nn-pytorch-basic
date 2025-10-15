#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/10/15 23:52
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   08_stats_07_argmax.py
# @Desc     :   

from torch import Tensor, randint, float64

from utils.decorator import beautifier


@beautifier
def argmax_tensor(t: Tensor, dims: int = -1, keep_dim: bool = False) -> Tensor:
    """ Argmax of a tensor """
    result = None
    match dims:
        case 0:
            print(f"Input tensor:\n{t}")
            result = t.argmax(dim=0, keepdim=keep_dim)
            print(f"Argmax tensor along dim 0:\n{result}")
        case 1:
            print(f"Input tensor:\n{t}")
            result = t.argmax(dim=1, keepdim=keep_dim)
            print(f"Argmax tensor along dim 1:\n{result}")
        case 2:
            print(f"Input tensor:\n{t}")
            result = t.argmax(dim=2, keepdim=keep_dim)
            print(f"Argmax tensor along dim 2:\n{result}")
        case _:
            print(f"Input tensor:\n{t}")
            result = t.argmax()
            print(f"Argmax tensor:\n{result}")

    return result


def main() -> None:
    """ Main Function """
    t = randint(0, 10, (3, 4, 5), dtype=float64)
    # print(t)

    argmax_tensor(t)
    # Find the max along dim 0
    argmax_tensor(t, dims=0, keep_dim=True)
    # Find the max along dim 1
    argmax_tensor(t, dims=1, keep_dim=False)
    # Find the max along dim 2
    argmax_tensor(t, dims=2, keep_dim=True)


if __name__ == "__main__":
    main()
