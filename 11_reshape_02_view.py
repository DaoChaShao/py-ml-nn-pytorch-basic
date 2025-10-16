#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/10/16 13:22
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   11_reshape_02_view.py
# @Desc     :   

from torch import Tensor, randint as py_randint, float64

from utils.helper import Beautifier


def example() -> Tensor:
    """ Example Function """
    t: Tensor = py_randint(1, 10, (2, 3, 6), dtype=float64)

    print(f"Full tensor shape: {t.shape}")
    return t


def main() -> None:
    """ Main Function """
    e = example()

    with Beautifier("Reshape using view"):
        print(f"Full tensor shape:\n{e}")
        print(f"View tensor shape:\n{e.view(2, 18)}")

    with Beautifier("Reshape using view with -1"):
        print(f"Full tensor shape:\n{e}")
        print(f"View tensor shape:\n{e.view(-1, 9)}")


if __name__ == "__main__":
    main()
