#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/10/15 10:51
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   PT.py
# @Desc     :   

from torch import cuda, backends, device


def device_checker() -> device:
    """ Check Available Device (CPU, GPU, MPS) """
    if cuda.is_available():
        count: int = cuda.device_count()
        print(f"Number of available GPU(s): {count}")
        for i in range(count):
            print(f"GPU {i}: {cuda.get_device_name(i)}")
            print(f"- Memory Usage:")
            print(f"- Allocated: {round(cuda.memory_allocated(i) / 1024 ** 3, 1)} GB")
            print(f"- Cached:    {round(cuda.memory_reserved(i) / 1024 ** 3, 1)} GB")
        return device("cuda")
    elif backends.mps.is_available():
        return device("mps")
    else:
        return device("cpu")
