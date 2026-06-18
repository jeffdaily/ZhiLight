#pragma once
// cublas (v2) -> hipBLAS name mapping (ZhiLight ROCm/HIP port).
#include "zhilight_cuda_to_hip.h"
#include <hipblas/hipblas.h>

#define cublasHandle_t       hipblasHandle_t
#define cublasStatus_t       hipblasStatus_t
#define cublasOperation_t    hipblasOperation_t
#define cublasComputeType_t  hipblasComputeType_t

#define cublasCreate         hipblasCreate
#define cublasDestroy        hipblasDestroy
#define cublasSetStream      hipblasSetStream
#define cublasHgemm          hipblasHgemm

#define CUBLAS_OP_N          HIPBLAS_OP_N
#define CUBLAS_OP_T          HIPBLAS_OP_T
#define CUBLAS_COMPUTE_16F   HIPBLAS_COMPUTE_16F
#define CUBLAS_COMPUTE_32F   HIPBLAS_COMPUTE_32F
#define CUBLAS_COMPUTE_32I   HIPBLAS_COMPUTE_32I

#define CUBLAS_STATUS_SUCCESS         HIPBLAS_STATUS_SUCCESS
#define CUBLAS_STATUS_NOT_INITIALIZED HIPBLAS_STATUS_NOT_INITIALIZED
#define CUBLAS_STATUS_ALLOC_FAILED    HIPBLAS_STATUS_ALLOC_FAILED
#define CUBLAS_STATUS_INVALID_VALUE   HIPBLAS_STATUS_INVALID_VALUE
#define CUBLAS_STATUS_ARCH_MISMATCH   HIPBLAS_STATUS_ARCH_MISMATCH
#define CUBLAS_STATUS_EXECUTION_FAILED HIPBLAS_STATUS_EXECUTION_FAILED
#define CUBLAS_STATUS_NOT_SUPPORTED   HIPBLAS_STATUS_NOT_SUPPORTED
