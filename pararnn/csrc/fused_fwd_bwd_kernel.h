//
//  For licensing see accompanying LICENSE file.
//  Copyright (C) 2025 Apple Inc. All Rights Reserved.
//

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <math.h>

#include <torch/extension.h>
#include <cuda_fp16.h>          // Required for __half
#include <cuda_bf16.h>          // Required for __nv_bfloat16
#include <c10/util/Half.h>      // Defines c10::Half
#include <c10/util/BFloat16.h>  // Defines c10::BFloat16

#include "parallel_reduction_kernel.h"


template< typename pimpl_t,
          typename... Args >
__global__ void __launch_bounds__(1024, 1) parallelSolveFusedFwd(   // prescribing some bounds to limit register usage
    typename pimpl_t::idx_t seqLength,
    typename pimpl_t::idx_t hiddenDim,
    typename pimpl_t::idx_t batchSize,
    typename pimpl_t::scalar_t *Hgl,
    Args... glbVars,
    int blocksPerSeq = 1
){
    typedef typename pimpl_t::scalar_t scalar_t;
    typedef typename pimpl_t::idx_t idx_t;
    static constexpr int chunk_size = pimpl_t::chunk_size;
    static constexpr int threads_per_warp  = pimpl_t::threads_per_warp;
    static constexpr int threads_per_block = pimpl_t::threads_per_block;

    typedef typename pimpl_t::jac_t jac_t;
    typedef typename pimpl_t::rhs_t rhs_t;

    static_assert((std::is_same_v<Args, const scalar_t*> && ...),
                  "All passed global variables must be of type const scalar_t*");

    // - organise shared memory for later use
    const bool lastOfWarp = (threadIdx.x % threads_per_warp) == (threads_per_warp - 1);
    const bool firstOfWarp = (threadIdx.x % threads_per_warp) == 0;
    const int warpsPerBlock = 1 + ((threads_per_block - 1) / (threads_per_warp)); // = ceil((float)blockDim.x/threads_per_warp)
    const int wIdx = threadIdx.x / threads_per_warp;
    extern __shared__ __align__(sizeof(scalar_t)) unsigned char my_smem[];  // this ensures unique naming for different templates instantiations
    scalar_t *temp = reinterpret_cast<scalar_t *>(my_smem);                 // might be __align__ is not needed: see https://stackoverflow.com/questions/27570552/templated-cuda-kernel-with-dynamic-shared-memory
    jac_t *sharedJac = (jac_t*) (temp);
    rhs_t *sharedRhs = (rhs_t*) (temp + warpsPerBlock * sizeof(jac_t) / sizeof(scalar_t) );

    // - organise local register variables
    rhs_t fullSol[chunk_size+1];
    rhs_t (&h  )[chunk_size] = *reinterpret_cast<rhs_t (*)[chunk_size]>(&fullSol[1]);
    rhs_t (&hm1)[chunk_size] = *reinterpret_cast<rhs_t (*)[chunk_size]>(&fullSol[0]);

    pimpl_t::setToZeroRhs(hm1[0]);

    auto lclVars = pimpl_t::initLclVars();

    // Loop over the sequence length -----------------------------------------------------------------------------------
    for( int block = 0; block < blocksPerSeq; ++block ){

        if( block>0 ){
            // If this block doesn't cover the very beginning of the seq, I need to store the last element of the sol
            __syncthreads();
            if( threadIdx.x == threads_per_block-1 && blockIdx.x == 0 ){
                pimpl_t::copyRhs( sharedRhs[0], h[chunk_size-1] );
            }
        }
        // Read from Global --------------------------------------------------------------------------------------------
        const idx_t myHDataIdx = getMyDataIdx< chunk_size >( seqLength, hiddenDim, block );
        const int numMyEqs = countMyEqs< chunk_size >( seqLength, hiddenDim, batchSize, block );

        pimpl_t::readDataFromGlobalFwd(
            numMyEqs, myHDataIdx, glbVars..., lclVars
        );

        // Main Newton Iterations --------------------------------------------------------------------------------------
        pimpl_t::assembleInitialGuess( h, lclVars );

        if( block > 0 ){
            // If this block doesn't cover the very beginning of the seq, I need to use the previous last sol to solve for the first current one
            __syncthreads();
            if( threadIdx.x == 0 && blockIdx.x == 0 ){
                pimpl_t::copyRhs( hm1[0], sharedRhs[0] );
                // solve using previous sol
                pimpl_t::recurrenceStep( hm1, lclVars, h );     // TODO: this actually solves for the whole chunk!
            }
        }

        for( int it = 0; it < MAX_NEWTON_ITS; ++it ){
            pimpl_t::sendSolNextInWarp( hm1[0], h[chunk_size-1], firstOfWarp );
            if (lastOfWarp ){
                pimpl_t::copyRhs( sharedRhs[wIdx], h[chunk_size-1] );
            }
            __syncthreads();
            if ( firstOfWarp && wIdx>0 ){
                pimpl_t::copyRhs( hm1[0], sharedRhs[wIdx-1] );
            }
            jac_t jac[chunk_size];
            rhs_t rhs[chunk_size];

            pimpl_t::assembleSystem( h, hm1, lclVars, rhs, jac );
            // NB: no need to set "spilling" eqs (t>numEqs) to neutral, as they won't affect the sol (because causality)

            if( threadIdx.x == 0 && blockIdx.x == 0 ){
                pimpl_t::setToZeroJac( jac[0] );
            }

            // solve
            __syncthreads();
            parallelReductionSharedInner< pimpl_t >(
                jac, rhs, sharedJac, sharedRhs, numMyEqs, seqLength
            );

            pimpl_t::updateSolChunk( h, rhs );
        } // end Newton its

        pimpl_t::writeToGlobal( numMyEqs, myHDataIdx, h, Hgl );

    } // end loop over seqLength

    return;
}



template< typename pimpl_t,
          typename... Args >
__global__ void __launch_bounds__(1024, 1) parallelSolveFusedBwd(   // prescribing some bounds to limit register usage
    typename pimpl_t::idx_t seqLength,
    typename pimpl_t::idx_t hiddenDim,
    typename pimpl_t::idx_t batchSize,
    typename pimpl_t::scalar_t *dHgl,
    const typename pimpl_t::scalar_t *gradgl,
    const typename pimpl_t::scalar_t *Hgl,
    Args... glbVars,
    int blocksPerSeq = 1
){
    typedef typename pimpl_t::scalar_t scalar_t;
    typedef typename pimpl_t::idx_t idx_t;
    static constexpr int chunk_size = pimpl_t::chunk_size;
    static constexpr int threads_per_warp  = pimpl_t::threads_per_warp;
    static constexpr int threads_per_block = pimpl_t::threads_per_block;

    typedef typename pimpl_t::jac_t jac_t;
    typedef typename pimpl_t::rhs_t rhs_t;

    static_assert((std::is_same_v<Args, const scalar_t*> && ...),
                  "All passed global variables must be of type const scalar_t*");

    // - organise shared memory for later use
    const bool lastOfWarp = (threadIdx.x % threads_per_warp) == (threads_per_warp - 1);
    const bool firstOfWarp = (threadIdx.x % threads_per_warp) == 0;
    const int warpsPerBlock = 1 + ((threads_per_block - 1) / (threads_per_warp)); // = ceil((float)blockDim.x/threads_per_warp)
    const int wIdx = threadIdx.x / threads_per_warp;
    extern __shared__ __align__(sizeof(scalar_t)) unsigned char my_smem[];  // this ensures unique naming for different templates instantiations
    scalar_t *temp = reinterpret_cast<scalar_t *>(my_smem);                 // might be __align__ is not needed: see https://stackoverflow.com/questions/27570552/templated-cuda-kernel-with-dynamic-shared-memory
    jac_t *sharedJac = (jac_t*) (temp);
    rhs_t *sharedRhs = (rhs_t*) (temp + warpsPerBlock * sizeof(jac_t) / sizeof(scalar_t) );

    // - organise local register variables
    rhs_t fullSol[chunk_size+1];
    rhs_t (&hp1)[chunk_size] = *reinterpret_cast<rhs_t (*)[chunk_size]>(&fullSol[1]);
    rhs_t (&h  )[chunk_size] = *reinterpret_cast<rhs_t (*)[chunk_size]>(&fullSol[0]);
    pimpl_t::setToZeroRhs(hp1[chunk_size-1]);

    jac_t jac[chunk_size];
    rhs_t rhs[chunk_size];

    auto lclVars = pimpl_t::initLclVars();


    // Loop over the sequence length (backwards, this time) ------------------------------------------------------------
    for( int block = blocksPerSeq-1; block >= 0; --block ){
        if( block < blocksPerSeq-1 ){
            // If this block doesn't cover the very end of the seq, I need to store the first element of the sol
            __syncthreads();        // I think I can get rid of this sync
            if( threadIdx.x == 0 && blockIdx.x == 0 ){
                pimpl_t::copyRhs( sharedRhs[warpsPerBlock-1], h[0] );   // store hp1
                pimpl_t::copyRhs( ((rhs_t*)temp) [0], rhs[0] );         // store sol //TODO: Im abusing sharedJac! -> this works if jac is "heavier" than rhs
            }
        }

        const idx_t myHDataIdx = getMyDataIdx< chunk_size >( seqLength, hiddenDim, block );
        const int numMyEqs = countMyEqs< chunk_size >( seqLength, hiddenDim, batchSize, block );
        const int numMyEqsShift1 = countMyEqs< chunk_size >( seqLength, hiddenDim, batchSize, block, 1 );

        pimpl_t::readDataFromGlobalBwd(
            numMyEqs, numMyEqsShift1, myHDataIdx,  // NB: in bwd, locals must be shifted by one to the right!!
            h, rhs, Hgl, gradgl, glbVars..., lclVars
        );

        // Must do some shifting here
        // - first I need to pass to each thread the following value of h[t+1]
        pimpl_t::sendSolPrevInWarp( hp1[chunk_size-1], h[0], lastOfWarp );
        __syncthreads();
        if ( firstOfWarp && wIdx > 0  ){ // beware not to dirty the sol in sharedRhs[warpsPerBlock-1]
            pimpl_t::copyRhs( sharedRhs[wIdx-1], h[0] );
        }
        __syncthreads();
        if ( lastOfWarp && ( block < blocksPerSeq-1 || wIdx < warpsPerBlock-1 ) ){
            pimpl_t::copyRhs( hp1[chunk_size-1], sharedRhs[wIdx] );
        }

        // - so that I can compute the Jacobians properly, with the values of h, a and b correctly aligned
        pimpl_t::computeJacobiansTranspose( hp1, h, lclVars, jac );

        if( block < blocksPerSeq-1 ){
            // If this block doesn't cover the very end of the seq, I need to use the previous first sol to solve for the last current one
            __syncthreads();
            if( threadIdx.x == threads_per_block-1 && blockIdx.x == 0 ){
                jac_t temp2;
                pimpl_t::setToZeroJac( temp2 );
                pimpl_t::reduceEqs( temp2, jac[chunk_size-1], ((rhs_t*)temp) [0], rhs[chunk_size-1] );
            }
        }

        #pragma unroll
        for( int t=0; t<chunk_size; ++t ){
            if( t>numMyEqs){
                pimpl_t::setToNeutralJac(jac[t]);
                pimpl_t::setToNeutralRhs(rhs[t]);   //TODO: useless? Should've already been handled within read()
            }
        }

        // - set very last one to 0
        if( ( block < blocksPerSeq-1 && threadIdx.x == threads_per_block-1 && blockIdx.x == 0 )
         || ( (threadIdx.x + blockIdx.x * blockDim.x) == ((seqLength - 1) / chunk_size) ) ){
            pimpl_t::setToZeroJac( jac[chunk_size-1] );
        }

        // Solve!
        __syncthreads();
        parallelBwdReductionSharedInner< pimpl_t >(
            jac, rhs, sharedJac, sharedRhs, numMyEqs, seqLength - block * threads_per_block * chunk_size
        );

        // Write to global
        pimpl_t::writeToGlobal( numMyEqs, myHDataIdx, rhs, dHgl );

    } // end loop over seqLength

}





// Kernel launchers ****************************************************************************************************
template< typename pimpl_t,
          int max_sequential_steps,
          typename... Args >
void fusedFwdLauncher(
        Args... glbVars,
        typename pimpl_t::scalar_t *h,
        typename pimpl_t::idx_t seqLength,
        typename pimpl_t::idx_t hiddenDim,
        typename pimpl_t::idx_t batchSize
    ) {
    typedef typename pimpl_t::scalar_t scalar_t;
    typedef typename pimpl_t::idx_t idx_t;
    static constexpr int chunk_size = pimpl_t::chunk_size;
    static constexpr int threads_per_warp  = pimpl_t::threads_per_warp;
    static constexpr int threads_per_block = pimpl_t::threads_per_block;

    static_assert((std::is_same_v<Args, const scalar_t*> && ...),
                  "All passed global variables must be of type const scalar_t*");

    // Define grid and block dimensions
    const idx_t blocksPerSeq = 1 + ((seqLength - 1) / (threads_per_block * chunk_size)); // = ceil((float)seqLength/(THREADS_PER_BLOCK * chunk_size))
    dim3 blockSize( threads_per_block, 1, 1 ); // One-dimensional block

    // Define amount of shared memory necessary per block
    const int warpsPerBlock = 1 + ((threads_per_block - 1) / (threads_per_warp)); // = ceil((float)THREADS_PER_BLOCK/THREADS_PER_WARP)
    unsigned int sharedMemSize =  (sizeof(typename pimpl_t::jac_t) + sizeof(typename pimpl_t::rhs_t)) * warpsPerBlock;

//    if ( blocksPerSeq <= max_sequential_steps ){
//    std::cout<<"Launching kernel on gridsize  ("<< gridSize.x  <<","<< gridSize.y  <<","<< gridSize.z  <<",)"<<std::endl
//             <<"with blocks of size blockSize ("<< blockSize.x <<","<< blockSize.y <<","<< blockSize.z <<",)"<<std::endl
//             <<"for tensors ot size (B,N,T):  ("<< batchSize   <<","<< hiddenDim   <<","<< seqLength   <<",)"<<std::endl
//             <<"Total memory allocated: "       << sharedMemSize <<" for "<<warpsPerBlock<< " warps."<<std::endl;
    dim3 gridSize( 1, hiddenDim, batchSize );
    parallelSolveFusedFwd< pimpl_t, Args... >
                      <<<gridSize, blockSize, sharedMemSize>>>(
                      seqLength, hiddenDim, batchSize,
                      h, glbVars...,
                      blocksPerSeq);
//    }else{
//        // TODO: trigger the next level of the hierarchy if you're dealing with a really long sequence
//    }

}


template< typename pimpl_t,
          int max_sequential_steps,
          typename... Args >
void fusedBwdLauncher(
        const typename pimpl_t::scalar_t *grad,
        const typename pimpl_t::scalar_t *h,
        Args... glbVars,
        typename pimpl_t::scalar_t *dh,
        typename pimpl_t::idx_t seqLength,
        typename pimpl_t::idx_t hiddenDim,
        typename pimpl_t::idx_t batchSize
    ) {

    typedef typename pimpl_t::scalar_t scalar_t;
    typedef typename pimpl_t::idx_t idx_t;
    static constexpr int chunk_size = pimpl_t::chunk_size;
    static constexpr int threads_per_warp  = pimpl_t::threads_per_warp;
    static constexpr int threads_per_block = pimpl_t::threads_per_block;

    static_assert((std::is_same_v<Args, const scalar_t*> && ...),
                  "All passed global variables must be of type const scalar_t*");

    // Define grid and block dimensions
    const idx_t blocksPerSeq = 1 + ((seqLength - 1) / (threads_per_block * chunk_size)); // = ceil((float)seqLength/(THREADS_PER_BLOCK * chunk_size))
    dim3 blockSize( threads_per_block, 1, 1 ); // One-dimensional block

    // Define amount of shared memory necessary per block
    const int warpsPerBlock = 1 + ((threads_per_block - 1) / (threads_per_warp)); // = ceil((float)THREADS_PER_BLOCK/THREADS_PER_WARP)
    unsigned int sharedMemSize =  (sizeof(typename pimpl_t::jac_t) + sizeof(typename pimpl_t::rhs_t)) * warpsPerBlock;

//    if ( blocksPerSeq <= max_sequential_steps ){
//      std::cout<<"Launching kernel on gridsize  ("<< gridSize.x  <<","<< gridSize.y  <<","<< gridSize.z  <<",)"<<std::endl
//               <<"with blocks of size blockSize ("<< blockSize.x <<","<< blockSize.y <<","<< blockSize.z <<",)"<<std::endl
//               <<"for tensors ot size (B,N,T):  ("<< batchSize   <<","<< hiddenDim   <<","<< seqLength   <<",)"<<std::endl
//               <<"Total memory allocated: "       << sharedMemSize <<" for "<<warpsPerBlock<< " warps."<<std::endl;
    dim3 gridSize( 1, hiddenDim, batchSize );
    parallelSolveFusedBwd< pimpl_t, Args... >
                      <<<gridSize, blockSize, sharedMemSize>>>(
                      seqLength, hiddenDim, batchSize,
                      dh, grad, h, glbVars...,
                      blocksPerSeq);
//    }else{
//        // TODO: trigger the next level of the hierarchy if you're dealing with a really long sequence
//    }
}


