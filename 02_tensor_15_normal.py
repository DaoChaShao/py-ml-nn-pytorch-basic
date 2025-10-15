#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/10/15 18:26
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   02_tensor_15_normal.py
# @Desc     :   

from torch import tensor, arange, float32, normal

from utils.decorator import beautifier


def mean_and_std():
    """ Define mean and std as scalars """
    mean_tensor = arange(
        start=1,
        end=5,
        step=1,
        dtype=float32,
    )
    print(f"mean = \n{mean_tensor}")
    print(f"mean shape = {mean_tensor.shape}")

    std_tensor = arange(
        start=1,
        end=5,
        step=1,
        dtype=float32,
    )
    print(f"std = \n{std_tensor}")
    print(f"std shape = {std_tensor.shape}")

    return mean_tensor, std_tensor


@beautifier
def normal_distribution_tensor_all():
    """ Normal Distribution (Gaussian Distribution): both mean and std are tensors with the same dimensions """
    out = tensor([1., 2., 3., 4.])
    print(f"out_tensor: \n{out}")
    print(f"out_tensor.shape: {out.shape}")

    mean_tensor, std_tensor = mean_and_std()

    nor = normal(
        mean=mean_tensor,
        std=std_tensor,
        out=out,
    )
    print(f"normal: \n{nor}")
    print(f"normal.shape: {nor.shape}")

    print(f"out_tensor: \n{out}")
    print(f"out_tensor.shape: {out.shape}")


@beautifier
def normal_distribution_num_all():
    """ Normal Distribution (Gaussian Distribution): both mean and std are scalars """
    out = tensor([1.])
    print(f"out_tensor: \n{out}")
    print(f"out_tensor.shape: {out.shape}")

    nor = normal(
        mean=0,
        std=1,
        size=(4,),
        out=out,
        dtype=float32,
    )
    print(f"normal: \n{nor}")
    print(f"normal.shape: {nor.shape}")

    print(f"out_tensor: \n{out}")
    print(f"out_tensor.shape: {out.shape}")


@beautifier
def normal_distribution_tensor_and_num():
    """ Normal Distribution (Gaussian Distribution): mean is a tensor, std is a scalar """
    out = tensor([1., 2., 3., 4.])
    print(f"out_tensor: \n{out}")
    print(f"out_tensor.shape: {out.shape}")

    mean_tensor, std_tensor = mean_and_std()
    std = 1

    nor = normal(
        mean=mean_tensor,
        std=std,
        out=out,
    )
    print(f"normal: \n{nor}")
    print(f"normal.shape: {nor.shape}")

    print(f"out_tensor: \n{out}")
    print(f"out_tensor.shape: {out.shape}")


def main() -> None:
    """ Main Function """
    normal_distribution_tensor_all()
    normal_distribution_num_all()
    normal_distribution_tensor_and_num()


if __name__ == "__main__":
    main()
