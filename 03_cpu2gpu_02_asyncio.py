#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/10/15 13:34
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   03_cpu2gpu_02_asyncio.py
# @Desc     :   

from asyncio import gather, run
from time import perf_counter
from torch import cuda, device, mps, nn, randn


async def performance_tester(device_object: device) -> None:
    """ Create a simple neural network and move it to the specified device """
    start = perf_counter()
    model = nn.Sequential(
        nn.Linear(5000, 2000),
        nn.ReLU(),
        nn.Linear(2000, 1000),
        nn.ReLU(),
        nn.Linear(1000, 500),
        nn.ReLU(),
        nn.Linear(500, 100)
    ).to(device_object)

    data = randn(512, 5000).to(device_object)

    output = model(data)

    if device_object.type == "mps":
        mps.synchronize()
    elif device_object.type == "cuda":
        cuda.synchronize()

    end = perf_counter()
    print(f"Data training on {output.device} took {end - start:.5f} seconds.")


async def main() -> None:
    """ Main Function """
    devices: list[device] = [device("cpu"), device("mps")]

    tasks: list = [performance_tester(d) for d in devices]
    await gather(*tasks)


if __name__ == "__main__":
    run(main())
