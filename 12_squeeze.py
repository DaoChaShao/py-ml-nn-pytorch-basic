#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/10/16 14:03
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   12_squeeze.py
# @Desc     :   

from torch import Tensor, randint as py_randint, float64

from utils.helper import Beautifier
from utils.highlighter import lines


def example() -> Tensor:
    """ Example Function """
    t: Tensor = py_randint(1, 10, (2, 3, 6), dtype=float64)
    return t


def main() -> None:
    """ Main Function """
    e = example()

    with Beautifier("Squeeze tensor"):
        print(f"Full tensor shape:\n{e} with shape {e.shape}")

        lines("t1 unsqueeze")
        t1: Tensor = e.unsqueeze(0)
        print(f"Unsqueeze tensor shape:\n{t1} with shape {t1.shape}")
        lines("t2 squeeze")
        t2: Tensor = t1.squeeze(0)
        print(f"Squeeze tensor shape:\n{t2} with shape {t2.shape}")
        lines("t3 squeeze again")
        t3: Tensor = t2.squeeze(0)
        print(f"Squeeze tensor shape:\n{t3} with shape {t3.shape}")


if __name__ == "__main__":
    main()
