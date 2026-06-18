#pragma once
#include "utils.cuh"

namespace bmengine{

namespace functions {

// Wave-size abstraction (C10_WARP_SIZE pattern). NVIDIA warps are 32 lanes;
// AMD wavefronts are 64 on CDNA (gfx90a/gfx94x) and 32 on RDNA (gfx10xx/11xx).
// __GFX9__ is defined only during the device compilation pass. The warp/block
// reductions below are driven entirely by BM_WARP_SIZE, so they are correct on
// wave32 AND wave64 with no per-arch hack. The block-reduce shared array is
// sized to the worst case (min warp 32 -> 1024/32 = 32 partials, +1 result).
#if defined(__HIP_PLATFORM_AMD__) || defined(USE_HIP)
#if defined(__GFX9__)
#define BM_WARP_SIZE 64
#else
#define BM_WARP_SIZE 32
#endif
#else
#define BM_WARP_SIZE 32
#endif
#define BM_BLOCK_REDUCE_MAX_WARPS 33

// Full-wavefront shuffle helpers: on HIP the no-mask __shfl_down/__shfl span the
// whole (32- or 64-lane) wavefront; on CUDA keep the 0xFFFFFFFF full-warp mask.
template<typename T>
__inline__ __device__ T warpShflDown(T x, int offset) {
#if defined(__HIP_PLATFORM_AMD__) || defined(USE_HIP)
    return __shfl_down(x, offset);
#else
    return __shfl_down_sync(0xFFFFFFFF, x, offset);
#endif
}
template<typename T>
__inline__ __device__ T warpShfl(T x, int src) {
#if defined(__HIP_PLATFORM_AMD__) || defined(USE_HIP)
    return __shfl(x, src);
#else
    return __shfl_sync(0xFFFFFFFF, x, src);
#endif
}

// Width-W logical-warp shuffle: reduces within fixed W-lane sub-groups (W must
// divide the wave size). Kernels that tile a 32-lane NVIDIA warp (e.g. a 128-dim
// head as lane*4) must reduce over exactly 32 lanes on BOTH wave32 and wave64,
// so they use the Width form rather than the full-wavefront reduce above.
template<typename T>
__inline__ __device__ T warpShflDownW(T x, int offset, int width) {
#if defined(__HIP_PLATFORM_AMD__) || defined(USE_HIP)
    return __shfl_down(x, offset, width);
#else
    return __shfl_down_sync(0xFFFFFFFF, x, offset, width);
#endif
}
template<typename T>
__inline__ __device__ T warpShflW(T x, int src, int width) {
#if defined(__HIP_PLATFORM_AMD__) || defined(USE_HIP)
    return __shfl(x, src, width);
#else
    return __shfl_sync(0xFFFFFFFF, x, src, width);
#endif
}

template<typename T, int WIDTH>
__inline__ __device__ T warpReduceSumWidth(T x) {
#pragma unroll
    for (int offset = WIDTH / 2; offset > 0; offset /= 2)
        x += warpShflDownW<T>(x, offset, WIDTH);
    return x;
}
template<typename T, int WIDTH>
__inline__ __device__ T warpReduceMaxWidth(T x) {
#pragma unroll
    for (int offset = WIDTH / 2; offset > 0; offset /= 2) {
        T y = warpShflDownW<T>(x, offset, WIDTH);
        x = (x > y) ? x : y;
    }
    return x;
}
// Width-W max with broadcast of the result to every lane in the W-lane subgroup.
// Drop-in replacement for warpReduceMaxB in kernels that tile exactly W=32 lanes,
// keeping the reduction within the intended 32-element group on wave64.
template<typename T, int WIDTH>
__inline__ __device__ T warpReduceMaxWidthB(T x) {
    x = warpReduceMaxWidth<T, WIDTH>(x);
    return warpShflW<T>(x, 0, WIDTH);  // broadcast within the W-lane subgroup
}
// Width-W sum with broadcast of the result to every lane in the W-lane subgroup.
// Broadcast counterpart of warpReduceMaxWidthB; drop-in for warpReduceSumB in
// kernels that tile exactly W=32 lanes, keeping the reduction within the
// intended 32-element group on wave64.
template<typename T, int WIDTH>
__inline__ __device__ T warpReduceSumWidthB(T x) {
    x = warpReduceSumWidth<T, WIDTH>(x);
    return warpShflW<T>(x, 0, WIDTH);  // broadcast within the W-lane subgroup
}

template<typename T>
__inline__ __device__ T threadMax(T a, T b) {
    return (a > b) ? a : b;
}

template<typename T>
__inline__ __device__ T warpReduceMax(T x) {
    #pragma unroll
    for (int offset = BM_WARP_SIZE / 2; offset > 0; offset /= 2)
        x = threadMax<T>(x, warpShflDown<T>(x, offset));
    return x;
}
template<typename T>
__inline__ __device__ T warpReduceMaxB(T x) {
#pragma unroll
    for (int offset = BM_WARP_SIZE / 2; offset > 0; offset /= 2)
        x = threadMax<T>(x, warpShflDown<T>(x, offset));
    return warpShfl<T>(x, 0);  // broadcast
}

template<typename T>
__inline__ __device__ T threadMin(T a, T b) {
    return (a > b) ? b : a;
}

template<typename T>
__inline__ __device__ T warpReduceMin(T x) {
#pragma unroll
    for (int offset = BM_WARP_SIZE / 2; offset > 0; offset /= 2)
        x = threadMin<T>(x, warpShflDown<T>(x, offset));
    return x;
}

template<typename T>
__inline__ __device__ T warpReduceSum(T x) {
    #pragma unroll
    for (int offset = BM_WARP_SIZE / 2; offset > 0; offset /= 2)
        x += warpShflDown<T>(x, offset);
    return x;
}
template<typename T>
__inline__ __device__ T warpReduceSumB(T x) {
    #pragma unroll
    for (int offset = BM_WARP_SIZE / 2; offset > 0; offset /= 2)
        x += warpShflDown<T>(x, offset);
    return warpShfl<T>(x, 0);  // broadcast
}

template<typename T>
__inline__ __device__ T blockReduceMax(T x) {
    static __shared__ T shared[BM_BLOCK_REDUCE_MAX_WARPS];
    int lane = threadIdx.x % BM_WARP_SIZE;
    int wid = threadIdx.x / BM_WARP_SIZE;
    x = warpReduceMax<T>(x);
    if (lane == 0) shared[wid] = x;
    __syncthreads();
#if defined(__HIP_PLATFORM_AMD__) || defined(USE_HIP)
    int num_warps = (blockDim.x + BM_WARP_SIZE - 1) / BM_WARP_SIZE;
    x = (lane < num_warps) ? shared[lane] : T(-INFINITY);
#else
    x = (threadIdx.x < blockDim.x / BM_WARP_SIZE) ? shared[lane] : T(-INFINITY);
#endif
    if (wid == 0) {
        x = warpReduceMax<T>(x);
        if (lane == 0) shared[BM_BLOCK_REDUCE_MAX_WARPS - 1] = x;
    }
    __syncthreads();
    return shared[BM_BLOCK_REDUCE_MAX_WARPS - 1];  // avoid RAW hazard
}

template<typename T>
__inline__ __device__ T blockReduceMin(T x) {
    static __shared__ T shared[BM_BLOCK_REDUCE_MAX_WARPS];
    int lane = threadIdx.x % BM_WARP_SIZE;
    int wid = threadIdx.x / BM_WARP_SIZE;
    x = warpReduceMin<T>(x);
    if (lane == 0)
        shared[wid] = x;
    __syncthreads();
#if defined(__HIP_PLATFORM_AMD__) || defined(USE_HIP)
    int num_warps = (blockDim.x + BM_WARP_SIZE - 1) / BM_WARP_SIZE;
    x = (lane < num_warps) ? shared[lane] : T(INFINITY);
#else
    x = (threadIdx.x < blockDim.x / BM_WARP_SIZE) ? shared[lane] : T(INFINITY);
#endif
    if (wid == 0) {
        x = warpReduceMin<T>(x);
        if (lane == 0)
            shared[BM_BLOCK_REDUCE_MAX_WARPS - 1] = x;
    }
    __syncthreads();
    return shared[BM_BLOCK_REDUCE_MAX_WARPS - 1]; // avoid RAW hazard
}

template<typename T>
__inline__ __device__ T blockReduceSum(T x) {
    static __shared__ T shared[BM_BLOCK_REDUCE_MAX_WARPS];
    int lane = threadIdx.x % BM_WARP_SIZE;
    int wid = threadIdx.x / BM_WARP_SIZE;
    x = warpReduceSum<T>(x);
    if (lane == 0) shared[wid] = x;
    __syncthreads();
#if defined(__HIP_PLATFORM_AMD__) || defined(USE_HIP)
    int num_warps = (blockDim.x + BM_WARP_SIZE - 1) / BM_WARP_SIZE;
    x = (lane < num_warps) ? shared[lane] : T(0.);
#else
    x = (threadIdx.x < blockDim.x / BM_WARP_SIZE) ? shared[lane] : T(0.);
#endif
    if (wid == 0) {
        x = warpReduceSum<T>(x);
        if (lane == 0) shared[BM_BLOCK_REDUCE_MAX_WARPS - 1] = x;
    }
    __syncthreads();
    return shared[BM_BLOCK_REDUCE_MAX_WARPS - 1];  // avoid RAW hazard
}

} // namespace functions

} // namespace bmengine
