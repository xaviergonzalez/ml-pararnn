//
//  For licensing see accompanying LICENSE file.
//  Copyright (C) 2025 Apple Inc. All Rights Reserved.
//

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <math.h>

#include <torch/extension.h>
#include <cuda_fp16.h>          // Required for __half
#include <cuda_bf16.h>          // Required for __nv_bfloat16
#include <c10/util/Half.h>      // Defines c10::Half
#include <c10/util/BFloat16.h>  // Defines c10::BFloat16


#include "helpers.h"
#include "rnn_cell_impl.h"
#include "nonlinearities.h"
#include "fused_fwd_bwd_kernel.h"


// *********************************************************************************************************************
// Specialisation of RNNCellImpl for LSTM-CIFG Diag system
// *********************************************************************************************************************
template < typename scalar_t >
class LSTMCIFGCellDiagImpl;


template< typename scalar_type >
class RNNCellTraits< LSTMCIFGCellDiagImpl< scalar_type > >: public RNNCellBaseTraits{
public:
    using scalar_t = scalar_type;
    using rhs_t = scalar_t[2];
    using jac_t = scalar_t[2][2];
    static constexpr int chunk_size      = dtype2chunkSizeLSTMCIFG<scalar_t>;
    static constexpr int num_hidden_vars = 2;
    using lclVars_t = std::tuple< std::array< scalar_t, 3 >,
                                  std::array< std::array< scalar_t, 3 >, chunk_size >,
                                  std::array< scalar_t, 2 > >;
};


template < typename scalar_t >
class LSTMCIFGCellDiagImpl : public RNNCellBlockDiagImpl< scalar_t, 2, LSTMCIFGCellDiagImpl< scalar_t > > {
public:
    using Traits = RNNCellTraits< LSTMCIFGCellDiagImpl< scalar_t > >;
    using idx_t = typename Traits::idx_t;
    static constexpr int chunk_size      = Traits::chunk_size;
    static constexpr int num_hidden_vars = Traits::num_hidden_vars;
    using rhs_t = typename Traits::rhs_t;
    using jac_t = typename Traits::jac_t;
    using lclVars_t = typename Traits::lclVars_t;

    // Reading / writing from / to global ******************************************************************************
    __forceinline__ __device__ static auto initLclVars(){
        // TODO I actually need value-copiable types (or memset a buffer)
        std::array< scalar_t, 3 > a{};
        std::array< std::array< scalar_t, 3 >, chunk_size > b{};
        std::array< scalar_t, 2 > c{};
        return std::make_tuple(a,b,c);
    }

    __forceinline__ __device__ static void readDataFromGlobalFwd(
        int numMyEqs, int myHDataIdx,
        const scalar_t* Agl,
        const scalar_t* Bgl,
        const scalar_t* Cgl,
        lclVars_t& lclVars
    ){
        auto& [a, b, c] = lclVars;

        const int myNIdx = threadIdx.y + blockIdx.y * blockDim.y;
        #pragma unroll
        for( int v = 0; v < 3; ++v ){
            a[v] = Agl[ myNIdx * 3 + v];
        }
        #pragma unroll
        for( int v = 0; v < 2; ++v ){
            c[v] = Cgl[ myNIdx * 2 + v];
        }
        #pragma unroll
        for( int t = 0; t < chunk_size; ++t ){
            if( t<numMyEqs ){
                #pragma unroll
                for( int v = 0; v < 3; ++v ){
                    b[t][v] = Bgl[ 3*(myHDataIdx + t) + v];  // myBDataIdx = 3*myHDataIdx
                }
            }
        }
    }

    __forceinline__ __device__ static void readDataFromGlobalBwd(
        int numMyEqs, int numMyEqsShift1, int myHDataIdx,
        rhs_t (&h)[chunk_size],
        rhs_t (&rhs)[chunk_size],
        const scalar_t* Hgl,
        const scalar_t* gradgl,
        const scalar_t* Agl,
        const scalar_t* Bgl,
        const scalar_t* Cgl,
        lclVars_t& lclVars
    ){
        auto& [a, b, c] = lclVars;

        const int myNIdx = threadIdx.y + blockIdx.y * blockDim.y;
        #pragma unroll
        for( int v = 0; v < 3; ++v ){
            a[v] = Agl[ myNIdx * 3 + v];
        }
        #pragma unroll
        for( int v = 0; v < 2; ++v ){
            c[v] = Cgl[ myNIdx * 2 + v];
        }
        #pragma unroll
        for( int t = 0; t < chunk_size; ++t ){
            // Read input shifted by 1 for bwd! -> this breaks coalesced reads, but it's likely better than having to shift Jacobians by 1
            if( t<numMyEqsShift1 ){
                #pragma unroll
                for( int v = 0; v < 3; ++v ){
                    b[t][v] = Bgl[ 3*(myHDataIdx + t + 1) + v];  // myBDataIdx = 3*myHDataIdx
                }
            }
        }

        const rhs_t* myHglData    = (numMyEqs>0)? (rhs_t*) (    Hgl + myHDataIdx * num_hidden_vars ) : 0;
        const rhs_t* myGradglData = (numMyEqs>0)? (rhs_t*) ( gradgl + myHDataIdx * num_hidden_vars ) : 0;
        #pragma unroll
        for( int t = 0; t < chunk_size; ++t ){
            if( t<numMyEqs ){
                copyRhs(   h[t],    myHglData[t] );
                copyRhs( rhs[t], myGradglData[t] );
            }else{
                setToNeutralRhs(   h[t] ); //TODO: useless?
                setToNeutralRhs( rhs[t] );
            }
        }
    }



    // LSTM-CIFG cell definition ***************************************************************************************
    __forceinline__ __device__ static void recurrenceStep(
        const rhs_t (&h)[chunk_size],
        const lclVars_t& lclVars,
        rhs_t (&hp1)[chunk_size]
    ){
        auto& [a, b, c] = lclVars;

        const scalar_t one = static_cast<scalar_t>(1.0f);
        #pragma unroll 1
        for( int t = 0; t < chunk_size; ++t ){
            const scalar_t fgt = nonlinFgt( a[0] * h[t][1] + b[t][0] + c[0] * h[t][0] );
            const scalar_t ctx = nonlinCtx( a[2] * h[t][1] + b[t][2] );
            hp1[t][0] = fgt * h[t][0] + (one - fgt) * ctx;
            const scalar_t out = nonlinOut( a[1] * h[t][1] + b[t][1] + c[1] * hp1[t][0] );
            hp1[t][1] =    out * nonlinState(   hp1[t][0] );
        }
        return;
    }


    __forceinline__ __device__ static void computeJacobians(
        const rhs_t (&h)[chunk_size],
        const rhs_t (&hm1)[chunk_size],
        const lclVars_t& lclVars,
        jac_t (&jac)[chunk_size]
    ){
        auto& [a, b, c] = lclVars;

        const scalar_t one = static_cast<scalar_t>(1.0f);

        #pragma unroll 1
        for( int t = 0; t < chunk_size; ++t ){
            scalar_t fgt = a[0] * hm1[t][1] + b[t][0] + c[0] * hm1[t][0];
            scalar_t ctx = a[2] * hm1[t][1] + b[t][2];

            scalar_t jcf = c[0] * derivativeNonlinFgt(fgt);
            scalar_t jhf = a[0] * derivativeNonlinFgt(fgt);
            scalar_t jhc = a[2] * derivativeNonlinCtx(ctx);

            fgt = nonlinFgt(fgt);
            ctx = nonlinCtx(ctx);
            scalar_t out = a[1] * hm1[t][1] + b[t][1] + c[1] * h[t][0];
            scalar_t jho = a[1] * derivativeNonlinOut(out);
            scalar_t jco = c[1] * derivativeNonlinOut(out);

            out = nonlinOut(out);
            scalar_t temp = out * derivativeNonlinState( h[t][0] );
            scalar_t scc = nonlinState(h[t][0]);

            jac[t][0][0] = - (jcf * (hm1[t][0] - ctx) + fgt);
            jac[t][0][1] = - (jhf * (hm1[t][0] - ctx) + (one - fgt) * jhc);
            jac[t][1][0] = jac[t][0][0] * (jco * scc + temp);
            jac[t][1][1] = - (scc * (jho - jco * jac[t][0][1]) - temp * jac[t][0][1]);
        }
        return;
    }


    __forceinline__ __device__ static void assembleSystem(
        const rhs_t (&h)[chunk_size],
        const rhs_t (&hm1)[chunk_size],
        const lclVars_t& lclVars,
        rhs_t (&res)[chunk_size],
        jac_t (&jac)[chunk_size]
    ){
        auto& [a, b, c] = lclVars;

        const scalar_t one = static_cast<scalar_t>(1.0f);

        #pragma unroll 1        // decrease unroll depth to decrease register pressure
        for( int t = 0; t < chunk_size; ++t ){
            scalar_t fgt = a[0] * hm1[t][1] + b[t][0] + c[0] * hm1[t][0];
            scalar_t ctx = a[2] * hm1[t][1] + b[t][2];

            scalar_t jcf = c[0] * derivativeNonlinFgt(fgt);
            scalar_t jhf = a[0] * derivativeNonlinFgt(fgt);
            scalar_t jhc = a[2] * derivativeNonlinCtx(ctx);

            fgt = nonlinFgt(fgt);
            ctx = nonlinCtx(ctx);
            scalar_t out = a[1] * hm1[t][1] + b[t][1] + c[1] * h[t][0];
            scalar_t jho = a[1] * derivativeNonlinOut(out);
            scalar_t jco = c[1] * derivativeNonlinOut(out);

            out = nonlinOut(out);
            scalar_t temp = out * derivativeNonlinState( h[t][0] );
            scalar_t scc = nonlinState(h[t][0]);

            jac[t][0][0] = - (jcf * (hm1[t][0] - ctx) + fgt);
            jac[t][0][1] = - (jhf * (hm1[t][0] - ctx) + (one - fgt) * jhc);
            jac[t][1][0] = jac[t][0][0] * (jco * scc + temp);
            jac[t][1][1] = - (scc * (jho - jco * jac[t][0][1]) - temp * jac[t][0][1]);

            temp = fgt * hm1[t][0] + (one - fgt) * ctx;
            out = nonlinOut( a[1] * hm1[t][1] + b[t][1] + c[1] * temp );
            res[t][0] = - ( h[t][0] - temp );
            res[t][1] = - ( h[t][1] - out * nonlinState( temp ) );
        }
        return;
    }



private:

    // Nonlinearities defining the system, and their derivatives *******************************************************

    __forceinline__ __device__ static scalar_t nonlinFgt( scalar_t x ){
        return mySigmoid(x);
    }
    __forceinline__ __device__ static scalar_t nonlinOut( scalar_t x ){
        return mySigmoid(x);
    }
    __forceinline__ __device__ static scalar_t nonlinCtx( scalar_t x ){
        return myTanh(x);
    }
    __forceinline__ __device__ static scalar_t nonlinState( scalar_t x ){
        return myTanh(x);
    }

    __forceinline__ __device__ static scalar_t derivativeNonlinFgt( scalar_t x ){
        return sigmoidPrime(x);
    }
    __forceinline__ __device__ static scalar_t derivativeNonlinOut( scalar_t x ){
        return sigmoidPrime(x);
    }
    __forceinline__ __device__ static scalar_t derivativeNonlinCtx( scalar_t x ){
        return tanhPrime(x);
    }
    __forceinline__ __device__ static scalar_t derivativeNonlinState( scalar_t x ){
        return tanhPrime(x);
    }

};




// Implementation for LSTM CIFG diag MultiHead *************************************************************************
torch::Tensor fused_fwd_lstm_cifg_diag_mh(
        torch::Tensor A,
        torch::Tensor B,
        torch::Tensor C
    ){
    TORCH_CHECK(
           A.scalar_type() == B.scalar_type()
        && A.scalar_type() == C.scalar_type(),
        "Input tensors must have the same type"
    );

    const int num_hidden_vars = 2;

    // Tweak to make this func work with both batched and non-batched inputs
    const bool batched = B.dim() > 3;
    const unsigned int batchSize = batched? B.size(0) : 1;
    const unsigned int hiddenDim = B.size(batched+0);
    const unsigned int seqLength = B.size(batched+1);
//    idx_t numBlocks = B.size(batched+2);      // TODO: this must always be 3 - add a check!

    auto shape = B.sizes().vec(); shape.back() = num_hidden_vars;
    torch::Tensor h = torch::empty(shape, B.options());

    AT_DISPATCH_FLOATING_TYPES_AND_HALF_AND_BFLOAT16(h.scalar_type(), "Launching fused fwd pass for LSTM_CIFG diag MH", ([&] {
        using cuda_scalar_t = std::conditional_t<
            std::is_same_v<scalar_t, at::Half>, __half,                                            // to cast at::Half to cuda native __half
            std::conditional_t<std::is_same_v<scalar_t, at::BFloat16>, __nv_bfloat16, scalar_t>>;  // to cast at::BFloat16 to cuda native __nv_bfloat16
        using pimpl_t = LSTMCIFGCellDiagImpl< cuda_scalar_t >;
        fusedFwdLauncher< pimpl_t, MAX_SEQUENTIAL_STEPS_BLOCK_DIAG, const cuda_scalar_t*, const cuda_scalar_t*, const cuda_scalar_t* >(
            reinterpret_cast<cuda_scalar_t*>(A.data_ptr<scalar_t>()),
            reinterpret_cast<cuda_scalar_t*>(B.data_ptr<scalar_t>()),
            reinterpret_cast<cuda_scalar_t*>(C.data_ptr<scalar_t>()),
            reinterpret_cast<cuda_scalar_t*>(h.data_ptr<scalar_t>()),
            seqLength, hiddenDim, batchSize
        );
    }));

    return h;
}


torch::Tensor fused_bwd_lstm_cifg_diag_mh(
        torch::Tensor grad,
        torch::Tensor h,
        torch::Tensor A,
        torch::Tensor B,
        torch::Tensor C
    ){
    TORCH_CHECK(
           A.scalar_type() == B.scalar_type()
        && A.scalar_type() == C.scalar_type()
        && A.scalar_type() == grad.scalar_type()
        && A.scalar_type() == h.scalar_type(),
        "Input tensors must have the same type"
    );

    const int num_hidden_vars = 2;

    // Tweak to make this func work with both batched and non-batched inputs
    const bool batched = B.dim() > 3;
    const unsigned int batchSize = batched? B.size(0) : 1;
    const unsigned int hiddenDim = B.size(batched+0);
    const unsigned int seqLength = B.size(batched+1);
//    const idx_t numBlocks = B.size(batched+2);      // TODO: this must always be 3 - add a check!

    auto shape = B.sizes().vec(); shape.back() = num_hidden_vars;
    torch::Tensor dh = torch::empty(shape, B.options());

    AT_DISPATCH_FLOATING_TYPES_AND_HALF_AND_BFLOAT16(h.scalar_type(), "Launching fused bwd pass for LSTM_CIFG diag MH", ([&] {
        using cuda_scalar_t = std::conditional_t<
            std::is_same_v<scalar_t, at::Half>, __half,                                            // to cast at::Half to cuda native __half
            std::conditional_t<std::is_same_v<scalar_t, at::BFloat16>, __nv_bfloat16, scalar_t>>;  // to cast at::BFloat16 to cuda native __nv_bfloat16
        using pimpl_t = LSTMCIFGCellDiagImpl< cuda_scalar_t >;
        fusedBwdLauncher< pimpl_t, MAX_SEQUENTIAL_STEPS_BLOCK_DIAG, const cuda_scalar_t*, const cuda_scalar_t*, const cuda_scalar_t* >(
            reinterpret_cast<cuda_scalar_t*>(grad.data_ptr<scalar_t>()),
            reinterpret_cast<cuda_scalar_t*>(h.data_ptr<scalar_t>()),
            reinterpret_cast<cuda_scalar_t*>(A.data_ptr<scalar_t>()),
            reinterpret_cast<cuda_scalar_t*>(B.data_ptr<scalar_t>()),
            reinterpret_cast<cuda_scalar_t*>(C.data_ptr<scalar_t>()),
            reinterpret_cast<cuda_scalar_t*>(dh.data_ptr<scalar_t>()),
            seqLength, hiddenDim, batchSize
        );
    }));

    return dh;
}


