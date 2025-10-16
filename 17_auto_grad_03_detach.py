#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/10/16 16:21
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   17_auto_grad_03_detach.py
# @Desc     :   

from torch import Tensor, tensor, rand, float64

from utils.helper import RandomSeed, Beautifier
from utils.highlighter import green, red


def main() -> None:
    """ Main Function """
    with RandomSeed("Detach Example"):
        # Define a tensor as a feature
        X: Tensor = tensor(10, dtype=float64, requires_grad=True)
        y: Tensor = X.detach()
        with Beautifier("Feature and Detached Tensors"):
            print(f"Feature Tensor: {X}, and requires_grad is {green('True') if X.requires_grad else red('False')}")
            print(f"Detached Tensor: {y}, and requires_grad is {green('True') if y.requires_grad else red('False')}")

        # Check whether the ids and pointers are the same
        with Beautifier("ID and Pointer Check"):
            print(f"ID of X: {id(X)}, ID of y: {id(y)}")
            print(f"Pointer of X: {X.untyped_storage().data_ptr()}, Pointer of y: {y.untyped_storage().data_ptr()}")


if __name__ == "__main__":
    main()
