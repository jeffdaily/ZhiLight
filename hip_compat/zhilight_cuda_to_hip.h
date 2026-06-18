// ZhiLight CUDA -> HIP compatibility shim (ROCm/HIP port).
//
// Force-included on every HIP translation unit (CMAKE_HIP_FLAGS -include) and
// pulled in by the toolkit-named forwarding shims (cuda_runtime.h, cublas_v2.h,
// nccl.h, ...) placed on the HIP-only include path. On a CUDA build this file
// and the hip_compat/ shim dir are never on the include path, so the NVIDIA
// build is unchanged.
//
// Authored with assistance from Claude (Anthropic).
#pragma once

#if defined(USE_HIP) || defined(__HIP_PLATFORM_AMD__)

// libc host decls must win over HIP's __device__ memcpy/memset overloads once
// <hip/hip_runtime.h> is in scope, so include them first.
#include <cstring>
#include <cstdlib>

// Host C++ TUs (compiled by g++/clang++, not hipcc) need the AMD platform macro
// for <hip/hip_runtime.h> to expose the runtime API (hipMalloc, hipStream_t...).
// hipcc defines it already; defining it twice is harmless.
#ifndef __HIP_PLATFORM_AMD__
#define __HIP_PLATFORM_AMD__ 1
#endif

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
// hip_bf16.h uses clang elementwise builtins (__builtin_elementwise_fma etc.)
// that g++ lacks, so it only parses under the clang/HIP compiler. Host C++ TUs
// (g++) never use the bf16 type directly; skip it there.
#if defined(__clang__)
#include <hip/hip_bf16.h>
#endif

// ---- per-arch warp size (C10_WARP_SIZE pattern) -------------------------
// __GFX9__ is defined only in the device compilation pass; host code that
// needs the value must query hipGetDeviceProperties(...).warpSize at runtime.
#if defined(__GFX9__)
#define ZL_WARP_SIZE 64   // CDNA: gfx90a, gfx94x
#else
#define ZL_WARP_SIZE 32   // RDNA: gfx10xx, gfx11xx (and CUDA)
#endif

// ---- bf16 type spellings ------------------------------------------------
// HIP's hip_bf16.h provides __hip_bfloat16 / __hip_bfloat162 with the full set
// of arithmetic operators and the __float2bfloat16 / __bfloat162float helpers.
// Only available under clang (the bf16 header is clang-only, see above).
#if defined(__clang__)
using nv_bfloat16 = __hip_bfloat16;
using nv_bfloat162 = __hip_bfloat162;
#define __nv_bfloat16 __hip_bfloat16
#define __nv_bfloat162 __hip_bfloat162
#endif

// ---- cuda data-type enum (cublas/library_types) -------------------------
// hipblaslt/hipblas pull in hip_common's hipDataType; mirror the CUDA names.
#define cudaDataType_t hipDataType
#define cudaDataType   hipDataType
#define CUDA_R_32F  HIP_R_32F
#define CUDA_R_16F  HIP_R_16F
#define CUDA_R_16BF HIP_R_16BF
#define CUDA_R_32I  HIP_R_32I
#define CUDA_R_8I   HIP_R_8I
#define CUDA_R_8F_E4M3 HIP_R_8F_E4M3
#define CUDA_R_8F_E5M2 HIP_R_8F_E5M2

// ---- runtime: types -----------------------------------------------------
#define cudaError_t            hipError_t
#define cudaSuccess            hipSuccess
#define cudaErrorPeerAccessAlreadyEnabled hipErrorPeerAccessAlreadyEnabled
#define cudaStream_t           hipStream_t
#define cudaEvent_t            hipEvent_t
#define cudaDeviceProp         hipDeviceProp_t
#define cudaFuncAttributes     hipFuncAttributes

// ---- runtime: memcpy kinds ----------------------------------------------
#define cudaMemcpyHostToDevice   hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost   hipMemcpyDeviceToHost
#define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define cudaMemcpyHostToHost     hipMemcpyHostToHost

// ---- runtime: functions -------------------------------------------------
#define cudaMalloc             hipMalloc
#define cudaFree               hipFree
#define cudaMallocHost         hipHostMalloc
#define cudaHostAlloc          hipHostMalloc
#define cudaFreeHost           hipHostFree
#define cudaMemcpy             hipMemcpy
#define cudaMemcpyAsync        hipMemcpyAsync
#define cudaMemcpyPeer         hipMemcpyPeer
#define cudaMemsetAsync        hipMemsetAsync
#define cudaMemGetInfo         hipMemGetInfo
#define cudaGetDevice          hipGetDevice
#define cudaSetDevice          hipSetDevice
#define cudaGetDeviceCount     hipGetDeviceCount
#define cudaGetDeviceProperties hipGetDeviceProperties
#define cudaDeviceSynchronize  hipDeviceSynchronize
#define cudaDeviceCanAccessPeer hipDeviceCanAccessPeer
#define cudaDeviceEnablePeerAccess hipDeviceEnablePeerAccess
#define cudaGetLastError       hipGetLastError
#define cudaGetErrorString     hipGetErrorString
#define cudaDeviceGetAttribute hipDeviceGetAttribute

// ---- runtime: device attributes -----------------------------------------
#define cudaDevAttrMaxSharedMemoryPerBlockOptin hipDeviceAttributeSharedMemPerBlockOptin

// ---- runtime: streams ---------------------------------------------------
#define cudaStreamCreate            hipStreamCreate
#define cudaStreamCreateWithFlags   hipStreamCreateWithFlags
#define cudaStreamCreateWithPriority hipStreamCreateWithPriority
#define cudaStreamDestroy           hipStreamDestroy
#define cudaStreamSynchronize       hipStreamSynchronize
#define cudaStreamWaitEvent         hipStreamWaitEvent
#define cudaStreamNonBlocking       hipStreamNonBlocking

// ---- runtime: events ----------------------------------------------------
#define cudaEventCreate          hipEventCreate
#define cudaEventCreateWithFlags hipEventCreateWithFlags
#define cudaEventDestroy         hipEventDestroy
#define cudaEventRecord          hipEventRecord
#define cudaEventQuery           hipEventQuery
#define cudaEventSynchronize     hipEventSynchronize
#define cudaEventElapsedTime     hipEventElapsedTime
#define cudaEventDefault         hipEventDefault
#define cudaEventDisableTiming   hipEventDisableTiming

// ---- runtime: kernel func attributes ------------------------------------
// HIP takes the kernel entry as `const void*`; CUDA takes the typed function
// pointer. Wrap so call sites pass a typed kernel pointer unchanged.
#define cudaFuncAttributeMaxDynamicSharedMemorySize hipFuncAttributeMaxDynamicSharedMemorySize
#define cudaFuncSetAttribute(func, attr, value) \
    hipFuncSetAttribute(reinterpret_cast<const void*>(func), (attr), (value))
#define cudaFuncGetAttributes(attrptr, func) \
    hipFuncGetAttributes((attrptr), reinterpret_cast<const void*>(func))

// ---- device cache-hint load ---------------------------------------------
// __ldcs (streaming/last-use cache hint) has no HIP equivalent; map to the
// read-only __ldg load (same result, just a different cache hint). Used in
// device code only, so this is inert in host TUs.
#define __ldcs __ldg

// ---- kernel-parameter qualifier -----------------------------------------
// __grid_constant__ (CUDA sm70+ by-value kernel-param hint) has no HIP keyword;
// it is purely an optimization, so drop it on AMD.
#define __grid_constant__

// ---- L2 access-policy window: unsupported on CDNA, stub out -------------
// gfx90a has no programmable L2 persisting-cache window. The only caller
// (setL2AccessPolicyWindow) is already commented out at its call sites, and
// the device-limit set is best-effort, so map these to no-ops / drop them.
#define ZL_HIP_NO_L2_ACCESS_POLICY 1

#endif // USE_HIP
