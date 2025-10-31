#
#  For licensing see accompanying LICENSE file.
#  Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

from setuptools import setup
import os
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# NB: there must be consistency between the values used in cxx and nvcc to define the macros
#  specifying the chunk_size in helpers.h. This "ensures" this: just set the environment variable
#  export CHUNK_SIZE_FLAGS="-DFLOAT64_CHUNK_SIZE_DIAG=<N> -DFLOAT64_CHUNK_SIZE_BLOCK_DIAG_2x2=<M>"
chunk_size_flags = os.environ.get("CHUNK_SIZE_FLAGS", "").split()
if len(chunk_size_flags) == 0:
    chunk_size_flags += ["-DFLOAT64_CHUNK_SIZE_DIAG=4", "-DFLOAT64_CHUNK_SIZE_BLOCK_DIAG_2x2=1"]

setup(
    name='parallel_reduce_cuda',
    ext_modules=[
        CUDAExtension(
            name='parallel_reduce_cuda',
            sources=[
                'pararnn/csrc/parallel_reduction_bindings.cpp',
                'pararnn/csrc/parallel_reduce.cu',
                'pararnn/csrc/fused_gru_diag.cu',
                'pararnn/csrc/fused_lstm_cifg_diag.cu',
            ],
            include_dirs=['pararnn/csrc'],
            extra_compile_args={
                'cxx': [
                    '-O3',
                ] + chunk_size_flags,
                'nvcc': [
                    '-O3',
                    '-U__CUDA_NO_HALF_OPERATORS__',         # enable half precision
                    '-U__CUDA_NO_HALF_CONVERSIONS__',
                    '-U__CUDA_NO_BFLOAT16_CONVERSIONS__',
                    '-D__CUDA_INCLUDE_HALF_OPERATORS__',
                    '-D__CUDA_INCLUDE_BFLOAT16_OPERATORS__',
                    '-Xcudafe', '--diag_suppress=1886',     # to suppress overwritten __align__ warning
                    # '-Xptxas="-v" -lineinfo'                # to track pressure on SM resources
                ] + chunk_size_flags
            }
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
