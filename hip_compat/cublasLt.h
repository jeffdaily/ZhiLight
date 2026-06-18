#pragma once
// cublasLt -> hipBLASLt name mapping (ZhiLight ROCm/HIP port).
// The algo-introspection surface (cublasLtMatmulAlgoInit /
// cublasLtMatmulAlgoConfigGetAttribute / tile+stage enums /
// cublasLtMatmulSearch_t) has no hipBLASLt equivalent and is handled with
// USE_HIP guards at the call sites (gemm.cpp), not here.
#include "zhilight_cuda_to_hip.h"
#include <hipblas/hipblas.h>
#include <hipblaslt/hipblaslt.h>

// plain cublas handle is referenced alongside cublasLt in the core headers
#define cublasHandle_t                hipblasHandle_t
#define cublasCreate                  hipblasCreate
#define cublasDestroy                 hipblasDestroy
#define cublasSetStream               hipblasSetStream
#define cublasHgemm                   hipblasHgemm

// handles / opaque descriptors
#define cublasLtHandle_t              hipblasLtHandle_t
#define cublasLtMatmulDesc_t          hipblasLtMatmulDesc_t
#define cublasLtMatrixLayout_t        hipblasLtMatrixLayout_t
#define cublasLtMatmulPreference_t    hipblasLtMatmulPreference_t
#define cublasLtMatmulAlgo_t          hipblasLtMatmulAlgo_t
#define cublasLtMatmulHeuristicResult_t hipblasLtMatmulHeuristicResult_t
#define cublasLtEpilogue_t            hipblasLtEpilogue_t

// scalar enums shared with hipblas
#define cublasComputeType_t          hipblasComputeType_t
#define cublasOperation_t            hipblasOperation_t
#define cublasStatus_t               hipblasStatus_t

#define CUBLAS_COMPUTE_16F           HIPBLAS_COMPUTE_16F
#define CUBLAS_COMPUTE_32F           HIPBLAS_COMPUTE_32F
#define CUBLAS_COMPUTE_32I           HIPBLAS_COMPUTE_32I
#define CUBLAS_OP_N                  HIPBLAS_OP_N
#define CUBLAS_OP_T                  HIPBLAS_OP_T

#define CUBLAS_STATUS_SUCCESS        HIPBLAS_STATUS_SUCCESS
#define CUBLAS_STATUS_NOT_INITIALIZED HIPBLAS_STATUS_NOT_INITIALIZED
#define CUBLAS_STATUS_ALLOC_FAILED   HIPBLAS_STATUS_ALLOC_FAILED
#define CUBLAS_STATUS_INVALID_VALUE  HIPBLAS_STATUS_INVALID_VALUE
#define CUBLAS_STATUS_ARCH_MISMATCH  HIPBLAS_STATUS_ARCH_MISMATCH
#define CUBLAS_STATUS_EXECUTION_FAILED HIPBLAS_STATUS_EXECUTION_FAILED
#define CUBLAS_STATUS_NOT_SUPPORTED  HIPBLAS_STATUS_NOT_SUPPORTED

// matmul descriptor attributes
#define CUBLASLT_MATMUL_DESC_TRANSA          HIPBLASLT_MATMUL_DESC_TRANSA
#define CUBLASLT_MATMUL_DESC_TRANSB          HIPBLASLT_MATMUL_DESC_TRANSB
#define CUBLASLT_MATMUL_DESC_EPILOGUE        HIPBLASLT_MATMUL_DESC_EPILOGUE
#define CUBLASLT_MATMUL_DESC_BIAS_POINTER    HIPBLASLT_MATMUL_DESC_BIAS_POINTER
#define CUBLASLT_MATMUL_DESC_A_SCALE_POINTER HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER
#define CUBLASLT_MATMUL_DESC_B_SCALE_POINTER HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER
#define CUBLASLT_EPILOGUE_BIAS               HIPBLASLT_EPILOGUE_BIAS
#define CUBLASLT_EPILOGUE_DEFAULT            HIPBLASLT_EPILOGUE_DEFAULT

// matrix layout attributes
#define CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT           HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT
#define CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET  HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET

// preference attributes
#define CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES

// functions (1:1)
#define cublasLtMatmul                       hipblasLtMatmul
#define cublasLtMatmulDescCreate             hipblasLtMatmulDescCreate
#define cublasLtMatmulDescDestroy            hipblasLtMatmulDescDestroy
#define cublasLtMatmulDescSetAttribute       hipblasLtMatmulDescSetAttribute
#define cublasLtMatrixLayoutCreate           hipblasLtMatrixLayoutCreate
#define cublasLtMatrixLayoutDestroy          hipblasLtMatrixLayoutDestroy
#define cublasLtMatrixLayoutSetAttribute     hipblasLtMatrixLayoutSetAttribute
#define cublasLtMatmulPreferenceCreate       hipblasLtMatmulPreferenceCreate
#define cublasLtMatmulPreferenceDestroy      hipblasLtMatmulPreferenceDestroy
#define cublasLtMatmulPreferenceSetAttribute hipblasLtMatmulPreferenceSetAttribute
#define cublasLtMatmulAlgoGetHeuristic       hipblasLtMatmulAlgoGetHeuristic
