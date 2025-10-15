#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/10/15 23:02
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   07_save_memory.py
# @Desc     :   

from asyncio import gather, run
from torch import Tensor, tensor

from utils.decorator import beautifier


async def add_with_save(a: Tensor, b: Tensor) -> None:
    """ Save memory by using in-place operations """
    print(f"a id is {id(a)}")

    a += b
    print(f"a with save id is {id(a)}")


async def add_without_save(a: Tensor, b: Tensor) -> None:
    """ Without saving memory """
    print(f"a id is {id(a)}")

    a = a + b
    print(f"a without save id is {id(a)}")


async def multiply_with_save(a: Tensor, b: Tensor) -> None:
    """ Save memory by using in-place operations """
    print(f"a id is {id(a)}")

    a *= b
    print(f"a with save id is {id(a)}")


async def multiply_without_save(a: Tensor, b: Tensor) -> None:
    """ Without saving memory """
    print(f"a id is {id(a)}")

    a = a * b
    print(f"a without save id is {id(a)}")


async def multiply_with_save_(a: Tensor, b: Tensor) -> None:
    """ Save memory by using in-place operations """
    print(f"a id is {id(a)}")

    a[:] = a @ b
    print(f"a with save id is {id(a)}")


async def main() -> None:
    """ Main Function """
    a = tensor([5, 10, 15, 20])
    b = tensor([1, 2, 3, 4])
    # print(f"a: {a}")
    # print(f"b: {b}")

    tasks = [
        add_with_save(a, b), add_without_save(a, b),
        multiply_with_save(a, b), multiply_without_save(a, b),
        multiply_with_save_(a, b)
    ]
    await gather(*tasks)


if __name__ == "__main__":
    run(main())
