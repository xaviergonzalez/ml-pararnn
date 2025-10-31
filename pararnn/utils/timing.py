#
#  For licensing see accompanying LICENSE file.
#  Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

import typing as typ
from dataclasses import dataclass

import torch
import torch.utils.benchmark

# Heavily reliant on the guide:
# https://www.speechmatics.com/company/articles-and-news/timing-operations-in-pytorch#


@dataclass(frozen=True)
class ProfilerConfig:
    num_warmup_reps: int = 10
    num_reps: int = 100


def time_cuda_kernel(
        kernel_cfg: typ.Dict,
        profiler_cfg: ProfilerConfig,
        init_kernel_input_data: typ.Callable,
        kernel: typ.Callable
) -> torch.Tensor:
    
    for i in range(profiler_cfg.num_warmup_reps):
        input_data = init_kernel_input_data(*kernel_cfg)
        kernel(*input_data)
    torch.cuda.synchronize()
    
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(profiler_cfg.num_reps)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(profiler_cfg.num_reps)]
    times = []
    
    for j in range(profiler_cfg.num_reps):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        input_data = init_kernel_input_data(*kernel_cfg)

        # This in particular has a huge (positive) impact on the timing for the torch parallel_reduction at small seq_lenghts!
        torch.cuda._sleep(1_000_000)
        
        start_events[j].record()
        kernel(*input_data)
        end_events[j].record()
    
    torch.cuda.synchronize()
    times.append( [s.elapsed_time(e) for s, e in zip(start_events, end_events)] )  # time in milliseconds

    return torch.tensor(times)


