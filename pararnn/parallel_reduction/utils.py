#
#  For licensing see accompanying LICENSE file.
#  Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

import torch
import parallel_reduce_cuda


def get_diag_chunk_size(dtype: torch.dtype) -> int:
    return parallel_reduce_cuda.get_diag_chunk_size(dtype)


def get_block_diag_2x2_chunk_size(dtype: torch.dtype) -> int:
    return parallel_reduce_cuda.get_block_diag_2x2_chunk_size(dtype)


def get_block_diag_3x3_chunk_size(dtype: torch.dtype) -> int:
    return parallel_reduce_cuda.get_block_diag_3x3_chunk_size(dtype)


def get_fused_gru_chunk_size(dtype: torch.dtype) -> int:
    return parallel_reduce_cuda.get_fused_gru_chunk_size(dtype)


def get_fused_lstm_cifg_chunk_size(dtype: torch.dtype) -> int:
    return parallel_reduce_cuda.get_fused_lstm_cifg_chunk_size(dtype)


def get_threads_per_block() -> int:
    return parallel_reduce_cuda.get_threads_per_block()


def get_threads_per_warp() -> int:
    return parallel_reduce_cuda.get_threads_per_warp()

