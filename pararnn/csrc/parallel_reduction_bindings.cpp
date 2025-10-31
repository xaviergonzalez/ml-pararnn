//
//  For licensing see accompanying LICENSE file.
//  Copyright (C) 2025 Apple Inc. All Rights Reserved.
//

#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "helpers.h"

unsigned int get_diag_chunk_size(at::ScalarType dtype) {
    switch (dtype) {
        case at::ScalarType::Double:   return dtype2chunkSizeDiag<double>;
        case at::ScalarType::Float:    return dtype2chunkSizeDiag<float>;
        case at::ScalarType::Half:     return dtype2chunkSizeDiag<at::Half>;
        case at::ScalarType::BFloat16: return dtype2chunkSizeDiag<at::BFloat16>;
        default:
            throw std::invalid_argument("Unsupported dtype in get_diag_chunk_size()");
    }
}

unsigned int get_block_diag_2x2_chunk_size(at::ScalarType dtype) {
    switch (dtype) {
        case at::ScalarType::Double:   return dtype2chunkSizeBlockDiag<2,double>;
        case at::ScalarType::Float:    return dtype2chunkSizeBlockDiag<2,float>;
        case at::ScalarType::Half:     return dtype2chunkSizeBlockDiag<2,at::Half>;
        case at::ScalarType::BFloat16: return dtype2chunkSizeBlockDiag<2,at::BFloat16>;
        default:
            throw std::invalid_argument("Unsupported dtype in get_block_diag_2x2_chunk_size()");
    }
}

unsigned int get_block_diag_3x3_chunk_size(at::ScalarType dtype) {
    switch (dtype) {
        case at::ScalarType::Double:   return dtype2chunkSizeBlockDiag<3,double>;
        case at::ScalarType::Float:    return dtype2chunkSizeBlockDiag<3,float>;
        case at::ScalarType::Half:     return dtype2chunkSizeBlockDiag<3,at::Half>;
        case at::ScalarType::BFloat16: return dtype2chunkSizeBlockDiag<3,at::BFloat16>;
        default:
            throw std::invalid_argument("Unsupported dtype in get_block_diag_3x3_chunk_size()");
    }
}

unsigned int get_fused_gru_chunk_size(at::ScalarType dtype) {
    switch (dtype) {
        case at::ScalarType::Double:   return dtype2chunkSizeGRU<double>;
        case at::ScalarType::Float:    return dtype2chunkSizeGRU<float>;
        case at::ScalarType::Half:     return dtype2chunkSizeGRU<at::Half>;
        case at::ScalarType::BFloat16: return dtype2chunkSizeGRU<at::BFloat16>;
        default:
            throw std::invalid_argument("Unsupported dtype in get_fused_gru_chunk_size()");
    }
}

unsigned int get_fused_lstm_cifg_chunk_size(at::ScalarType dtype) {
    switch (dtype) {
        case at::ScalarType::Double:   return dtype2chunkSizeLSTMCIFG<double>;
        case at::ScalarType::Float:    return dtype2chunkSizeLSTMCIFG<float>;
        case at::ScalarType::Half:     return dtype2chunkSizeLSTMCIFG<at::Half>;
        case at::ScalarType::BFloat16: return dtype2chunkSizeLSTMCIFG<at::BFloat16>;
        default:
            throw std::invalid_argument("Unsupported dtype in get_fused_lstm_cifg_chunk_size()");
    }
}

int64_t get_threads_per_block(){
    return static_cast<uint32_t>(THREADS_PER_BLOCK);
}

int64_t get_threads_per_warp(){
    return static_cast<uint32_t>(THREADS_PER_WARP);
}

// main torch.ops defined in this module
TORCH_LIBRARY(parallel_reduce_cuda, m) {
    m.def("parallel_reduce_diag_cuda(Tensor a, Tensor b) -> Tensor");
    m.def("parallel_reduce_block_diag_2x2_cuda(Tensor a, Tensor b) -> Tensor");
    m.def("parallel_reduce_block_diag_3x3_cuda(Tensor a, Tensor b) -> Tensor");
    m.def("fused_fwd_gru_diag_mh(Tensor a, Tensor b) -> Tensor");
    m.def("fused_bwd_gru_diag_mh(Tensor a, Tensor b, Tensor c, Tensor d) -> Tensor");
    m.def("fused_fwd_lstm_cifg_diag_mh(Tensor a, Tensor b, Tensor c) -> Tensor");
    m.def("fused_bwd_lstm_cifg_diag_mh(Tensor a, Tensor b, Tensor c, Tensor d, Tensor e) -> Tensor");
}

TORCH_LIBRARY_IMPL(parallel_reduce_cuda, CUDA, m) {
    m.impl("parallel_reduce_diag_cuda",           &parallel_reduce_diag_cuda);          // parallel reduce with diagonal Jacobian (CUDA) - (multi precision));
    m.impl("parallel_reduce_block_diag_2x2_cuda", &parallel_reduce_block_diag_cuda<2>); // parallel reduce with 2x2 block diagonal Jacobian (CUDA) - (multi precision));
    m.impl("parallel_reduce_block_diag_3x3_cuda", &parallel_reduce_block_diag_cuda<3>); // parallel reduce with 3x3 block diagonal Jacobian (CUDA) - (multi precision));
    m.impl("fused_fwd_gru_diag_mh",               &fused_fwd_gru_diag_mh);
    m.impl("fused_bwd_gru_diag_mh",               &fused_bwd_gru_diag_mh);
    m.impl("fused_fwd_lstm_cifg_diag_mh",         &fused_fwd_lstm_cifg_diag_mh);
    m.impl("fused_bwd_lstm_cifg_diag_mh",         &fused_bwd_lstm_cifg_diag_mh);
}

//other utils functions
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("get_threads_per_block",               &get_threads_per_block,            "Number of threads per block");
    m.def("get_threads_per_warp",                &get_threads_per_warp,             "Number of threads per warp");
    m.def("get_diag_chunk_size",                 &get_diag_chunk_size,              "Size of chunks in Thomas reduction for diag, given a torch.dtype");
    m.def("get_block_diag_2x2_chunk_size",       &get_block_diag_2x2_chunk_size,    "Size of chunks in Thomas reduction for blockdiag 2x2, given a torch.dtype");
    m.def("get_block_diag_3x3_chunk_size",       &get_block_diag_3x3_chunk_size,    "Size of chunks in Thomas reduction for blockdiag 2x2, given a torch.dtype");
    m.def("get_fused_gru_chunk_size",            &get_fused_gru_chunk_size,         "Size of chunks in Thomas reduction for fused GRU, given a torch.dtype");
    m.def("get_fused_lstm_cifg_chunk_size",      &get_fused_lstm_cifg_chunk_size,   "Size of chunks in Thomas reduction for fused LSTM-CIFG, given a torch.dtype");
}


