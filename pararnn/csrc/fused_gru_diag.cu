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
// Specialisation of RNNCellImpl for GRU Diag system
// *********************************************************************************************************************
template < typename scalar_t >
class GRUCellDiagImpl;


template< typename scalar_type >
class RNNCellTraits< GRUCellDiagImpl< scalar_type > >: public RNNCellBaseTraits{
public:
    using scalar_t = scalar_type;
    using rhs_t = scalar_t;
    using jac_t = scalar_t;
    static constexpr int chunk_size      = dtype2chunkSizeGRU<scalar_t>;
    static constexpr int num_hidden_vars = 1;
    using lclVars_t = std::tuple< std::array< scalar_t, 3 >,
                                  std::array< std::array< scalar_t, 3 >, chunk_size > >;

};


template < typename scalar_t >
class GRUCellDiagImpl : public RNNCellDiagImpl< scalar_t, GRUCellDiagImpl< scalar_t > > {
public:
    using Traits = RNNCellTraits< GRUCellDiagImpl< scalar_t > >;
    using idx_t = typename Traits::idx_t;
    static constexpr int chunk_size      = Traits::chunk_size;
    static constexpr int num_hidden_vars = Traits::num_hidden_vars;
    using rhs_t = typename Traits::rhs_t;
    using jac_t = typename Traits::jac_t;
    using lclVars_t = typename Traits::lclVars_t;

    // Reading / writing from / to global ******************************************************************************
    __forceinline__ __device__ static auto initLclVars(){
        std::array< scalar_t, 3 > a{};
        std::array< std::array< scalar_t, 3 >, chunk_size > b{};
        return std::make_tuple(a, b);
    }

    __forceinline__ __device__ static void readDataFromGlobalFwd(
        int numMyEqs, int myHDataIdx,
        const scalar_t* Agl,
        const scalar_t* Bgl,
        lclVars_t& lclVars
    ){
        auto& [a, b] = lclVars;

        const int myNIdx = threadIdx.y + blockIdx.y * blockDim.y;
        #pragma unroll
        for( int v = 0; v < 3; ++v ){
            a[v] = Agl[ myNIdx * 3 + v];
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
        lclVars_t& lclVars
    ){
        auto& [a, b] = lclVars;

        const int myNIdx = threadIdx.y + blockIdx.y * blockDim.y;
        #pragma unroll
        for( int v = 0; v < 3; ++v ){
            a[v] = Agl[ myNIdx * 3 + v];
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



    // GRU cell definition *********************************************************************************************
    __forceinline__ __device__ static void recurrenceStep(
        const rhs_t (&h)[chunk_size],
        const lclVars_t& lclVars,
        rhs_t (&hp1)[chunk_size]
    ){
        auto& [a, b] = lclVars;

        const scalar_t one = static_cast<scalar_t>(1.0f);
        #pragma unroll
        for( int t = 0; t < chunk_size; ++t ){
            const scalar_t z = nonlinUpd(   a[0] * h[t]     + b[t][0] );
            const scalar_t r = nonlinRes(   a[1] * h[t]     + b[t][1] );
            hp1[t] =       z * nonlinState( a[2] * h[t] * r + b[t][2] ) + (one-z) * h[t];
        }
        return;
    }


    __forceinline__ __device__ static void computeJacobians(
        const rhs_t (&)[chunk_size],    // h is unused in GRU
        const rhs_t (&hm1)[chunk_size],
        const lclVars_t& lclVars,
        jac_t (&jac)[chunk_size]
    ){
        auto& [a, b] = lclVars;

        const scalar_t one = static_cast<scalar_t>(1.0f);

        #pragma unroll
        for( int t = 0; t < chunk_size; ++t ){
            scalar_t z = a[0] * hm1[t] + b[t][0];
            scalar_t r = a[1] * hm1[t] + b[t][1];
            scalar_t jz = a[0] * derivativeNonlinUpd(z);
            scalar_t jr = a[1] * derivativeNonlinRes(r);
            z = nonlinUpd(z);
            r = nonlinRes(r);
            scalar_t h =  a[2] * hm1[t] * r + b[t][2];
            scalar_t jh = a[2] * derivativeNonlinState(h) * ( r + hm1[t] * jr );
            h = nonlinState(h);

            jac[t] = - ( (one - z) + (h - hm1[t]) * jz + z * jh);
        }

        return;
    }


    __forceinline__ __device__ void static assembleSystem(
        const rhs_t (&h)[chunk_size],
        const rhs_t (&hm1)[chunk_size],
        const lclVars_t& lclVars,
        rhs_t (&res)[chunk_size],
        jac_t (&jac)[chunk_size]
    ){
        auto& [a, b] = lclVars;

        const scalar_t one = static_cast<scalar_t>(1.0f);

        #pragma unroll
        for( int t = 0; t < chunk_size; ++t ){
            scalar_t z = a[0] * hm1[t] + b[t][0];
            scalar_t r = a[1] * hm1[t] + b[t][1];
            scalar_t jz = a[0] * derivativeNonlinUpd(z);
            scalar_t jr = a[1] * derivativeNonlinRes(r);
            z = nonlinUpd(z);
            r = nonlinRes(r);
            scalar_t s =  a[2] * hm1[t] * r + b[t][2];
            scalar_t jh = a[2] * derivativeNonlinState(s) * ( r + hm1[t] * jr );
            s = nonlinState(s);
            res[t] = - (h[t] - ( z * s + (one-z) * hm1[t]));
            jac[t] = - ( (one - z) + (s - hm1[t]) * jz + z * jh);
        }
        return;
    }



private:

    // Nonlinearities defining the system, and their derivatives *******************************************************

    __forceinline__ __device__ static scalar_t nonlinUpd( scalar_t x ){
        return mySigmoid(x);
    }
    __forceinline__ __device__ static scalar_t nonlinRes( scalar_t x ){
        return mySigmoid(x);
    }
    __forceinline__ __device__ static scalar_t nonlinState( scalar_t x ){
        return myTanh(x);
    }

    __forceinline__ __device__ static scalar_t derivativeNonlinUpd( scalar_t x ){
        return sigmoidPrime(x);
    }
    __forceinline__ __device__ static scalar_t derivativeNonlinRes( scalar_t x ){
        return sigmoidPrime(x);
    }
    __forceinline__ __device__ static scalar_t derivativeNonlinState( scalar_t x ){
        return tanhPrime(x);
    }

};




// Implementation for GRU diag *****************************************************************************************
torch::Tensor fused_fwd_gru_diag_mh(
        torch::Tensor A,
        torch::Tensor B
    ){
    TORCH_CHECK( A.scalar_type() == B.scalar_type(), "Input tensors must have the same type" );

    // Tweak to make this func work with both batched and non-batched inputs
    const bool batched = B.dim() > 3;
    const unsigned int batchSize = batched? B.size(0) : 1;
    const unsigned int hiddenDim = B.size(batched+0);
    const unsigned int seqLength = B.size(batched+1);
//    idx_t numBlocks = B.size(batched+2);      // TODO: this must always be 3 - add a check!

    auto shape = B.sizes().vec(); shape.pop_back();
    torch::Tensor h = torch::empty(shape, B.options());

    AT_DISPATCH_FLOATING_TYPES_AND_HALF_AND_BFLOAT16(h.scalar_type(), "Launching fused fwd pass for GRU diag MH", ([&] {
        using cuda_scalar_t = std::conditional_t<
            std::is_same_v<scalar_t, at::Half>, __half,                                            // to cast at::Half to cuda native __half
            std::conditional_t<std::is_same_v<scalar_t, at::BFloat16>, __nv_bfloat16, scalar_t>>;  // to cast at::BFloat16 to cuda native __nv_bfloat16
        using pimpl_t = GRUCellDiagImpl< cuda_scalar_t >;
        fusedFwdLauncher< pimpl_t, MAX_SEQUENTIAL_STEPS_DIAG, const cuda_scalar_t*, const cuda_scalar_t* >(
            reinterpret_cast<cuda_scalar_t*>(A.data_ptr<scalar_t>()),
            reinterpret_cast<cuda_scalar_t*>(B.data_ptr<scalar_t>()),
            reinterpret_cast<cuda_scalar_t*>(h.data_ptr<scalar_t>()),
            seqLength, hiddenDim, batchSize
        );
    }));

    return h;
}


torch::Tensor fused_bwd_gru_diag_mh(
        torch::Tensor grad,
        torch::Tensor h,
        torch::Tensor A,
        torch::Tensor B
    ){
    TORCH_CHECK(
           A.scalar_type() == B.scalar_type()
        && A.scalar_type() == grad.scalar_type()
        && A.scalar_type() == h.scalar_type(),
        "Input tensors must have the same type"
    );

    // Tweak to make this func work with both batched and non-batched inputs
    const bool batched = B.dim() > 3;
    const unsigned int batchSize = batched? B.size(0) : 1;
    const unsigned int hiddenDim = B.size(batched+0);
    const unsigned int seqLength = B.size(batched+1);
//    const idx_t numBlocks = B.size(batched+2);      // TODO: this must always be 3 - add a check!

    auto shape = B.sizes().vec(); shape.pop_back();
    torch::Tensor dh = torch::empty(shape, B.options());

    AT_DISPATCH_FLOATING_TYPES_AND_HALF_AND_BFLOAT16(h.scalar_type(), "Launching fused bwd pass for GRU diag MH", ([&] {
        using cuda_scalar_t = std::conditional_t<
            std::is_same_v<scalar_t, at::Half>, __half,                                            // to cast at::Half to cuda native __half
            std::conditional_t<std::is_same_v<scalar_t, at::BFloat16>, __nv_bfloat16, scalar_t>>;  // to cast at::BFloat16 to cuda native __nv_bfloat16
        using pimpl_t = GRUCellDiagImpl< cuda_scalar_t >;
        fusedBwdLauncher< pimpl_t, MAX_SEQUENTIAL_STEPS_DIAG, const cuda_scalar_t*, const cuda_scalar_t* >(
            reinterpret_cast<cuda_scalar_t*>(grad.data_ptr<scalar_t>()),
            reinterpret_cast<cuda_scalar_t*>(h.data_ptr<scalar_t>()),
            reinterpret_cast<cuda_scalar_t*>(A.data_ptr<scalar_t>()),
            reinterpret_cast<cuda_scalar_t*>(B.data_ptr<scalar_t>()),
            reinterpret_cast<cuda_scalar_t*>(dh.data_ptr<scalar_t>()),
            seqLength, hiddenDim, batchSize
        );
    }));

    return dh;
}


