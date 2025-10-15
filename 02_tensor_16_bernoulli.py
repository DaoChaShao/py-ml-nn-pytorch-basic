#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/10/15 18:47
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   02_tensor_16_bernoulli.py
# @Desc     :   

from random import choice
from torch import ones, float32, tensor, bernoulli

from utils.decorator import beautifier


@beautifier
def bernoulli_creator():
    """ Create a tensor using Bernoulli distribution """
    out = ones([1], dtype=float32)
    print(f"out_tensor: \n{out}")
    print(f"out_tensor.shape: {out.shape}")

    probability = choice([0.1, 0.9])
    print(f"probability: \033[1;31m{probability}\033[0m")

    input_tensor = tensor(data=[0.9], dtype=float32)
    input_tensor = tensor(data=[probability], dtype=float32)
    print(f"input_tensor: \n{input_tensor}")
    print(f"input_tensor.shape: {input_tensor.shape}")

    t = bernoulli(
        input=input_tensor,
        generator=None,
        out=out,
    )
    print(f"tensor: \n\033[1;31m{t}\033[0m")
    print(f"tensor.shape: {t.shape}")


def main() -> None:
    """ Main Function """
    bernoulli_creator()


if __name__ == "__main__":
    main()
