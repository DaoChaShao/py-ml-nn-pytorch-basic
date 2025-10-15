#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/10/15 18:42
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   02_tensor_13_rand_perm.py
# @Desc     :   

from random import randint, seed
from torch import ones, int64, randperm, strided, initial_seed, manual_seed

from utils.decorator import beautifier


@beautifier
def rand_perm_creator():
    """ Create a tensor with a random permutation of integers from 0 to n - 1 """
    out = ones([1], dtype=int64)
    print(f"out_tensor: \n{out}")
    print(f"out_tensor.shape: {out.shape}")

    t = randperm(
        n=randint(3, 16),
        out=out,
        dtype=int64,
        layout=strided,
        device=None,
        requires_grad=False,
    )
    print(f"The tensor: \n{t}")
    print(f"The tensor shape is {t.shape}")
    print(f"The initial seed is {initial_seed()}")


def main() -> None:
    """ Main Function """
    rand_perm_creator()

    manual_seed(27)
    seed(27)
    rand_perm_creator()


if __name__ == "__main__":
    main()
